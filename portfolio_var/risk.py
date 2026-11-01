"""Rolling joint-scenario VaR, coverage tests, diversification, and Euler attribution.

Latest risk summaries refer to the last matched forecast, not an additional
unrealized forecast. Each portfolio retains its retrospective inception holdings.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import xlog1py, xlogy
from scipy.stats import chi2

from portfolio_var import config
from portfolio_var.pricing import FixedHoldings, value_holdings
from portfolio_var.simulation import fit_gaussian


@dataclass(frozen=True)
class EmpiricalQuantile:
    """Stable scenario indices are zero-based; weights are shared with Euler."""

    value: float
    lower_scenario: int
    upper_scenario: int
    lower_rank: int
    upper_rank: int
    upper_weight: float
    boundary_tie: bool


@dataclass(frozen=True)
class RollingRisk:
    holdings: FixedHoldings
    backtest: pd.DataFrame
    coverage_tests: pd.DataFrame
    diversification: pd.DataFrame
    euler_contributions: pd.DataFrame


def empirical_quantile(losses, confidence: float | None = None) -> EmpiricalQuantile:
    """Linearly interpolate the two ordered losses, preserving signed values."""
    losses = np.asarray(losses, dtype=float)
    probability = config.VAR_CONFIDENCE if confidence is None else confidence
    if losses.ndim != 1 or not len(losses) or not np.isfinite(losses).all():
        raise ValueError("An empirical quantile requires a nonempty finite loss vector.")
    if not 0 <= probability <= 1 or config.QUANTILE_METHOD != "linear":
        raise ValueError("The configured quantile must be linear with confidence in [0, 1].")
    order = np.argsort(losses, kind="stable")
    rank = (len(losses) - 1) * probability
    lower_rank, upper_rank = int(np.floor(rank)), int(np.ceil(rank))
    weight = float(rank - lower_rank)
    lower, upper = int(order[lower_rank]), int(order[upper_rank])
    value = losses[lower] + weight * (losses[upper] - losses[lower])
    tied = np.count_nonzero(losses == losses[lower]) > 1 or np.count_nonzero(losses == losses[upper]) > 1
    return EmpiricalQuantile(float(value), lower, upper, lower_rank, upper_rank, weight, bool(tied))


def euler_contributions(component_losses, quantile: EmpiricalQuantile) -> np.ndarray:
    """Interpolate signed losses from the exact portfolio-quantile scenarios.

    For fixed unit marks, component losses equal q_i times unit losses. Their
    quantile interpolation therefore gives q_i * dVaR/dq_i, with stable ordering
    selecting a deterministic allocation when portfolio losses tie.
    """
    losses = np.asarray(component_losses, dtype=float)
    lower, upper = losses[quantile.lower_scenario], losses[quantile.upper_scenario]
    return lower + quantile.upper_weight * (upper - lower)


def exception_flags(realized_losses, forecast_var) -> np.ndarray:
    """An equality with forecast VaR is not an exception."""
    return np.greater(realized_losses, forecast_var)


def scenario_losses(holdings: FixedHoldings, origin: pd.Series, joint_shocks, elapsed_steps: int) -> np.ndarray:
    """Apply observed or Gaussian joint shocks to current levels and age once."""
    shocks = np.asarray(joint_shocks, dtype=float)
    if shocks.ndim != 2 or shocks.shape[1] != len(config.FACTOR_ORDER) or not len(shocks) or not np.isfinite(shocks).all():
        raise ValueError("One-step scenarios require a nonempty finite matrix of eight joint shocks.")
    curve = origin.loc[list(config.TREASURY_SERIES.values())].to_numpy(dtype=float)
    adjusted = origin.loc[[f"{ticker}_Adj Close" for ticker in config.ETF_TICKERS]].to_numpy(dtype=float)
    current = value_holdings(holdings, curve, adjusted, elapsed_steps)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            next_values = value_holdings(
                holdings, curve + shocks[:, :len(curve)],
                adjusted * np.exp(shocks[:, len(curve):]),
                elapsed_steps + config.VAR_HORIZON_STEPS,
            )
            losses = current - next_values
    except FloatingPointError as exc:
        raise ValueError("Joint shocks could not produce finite one-step losses.") from exc
    if not np.isfinite(losses).all():
        raise ValueError("Scenario losses must be finite.")
    return losses


def _bernoulli_log_likelihood(successes: int, failures: int, probability: float) -> float:
    # xlogy and xlog1py implement 0*log(0)=0, including boundary MLEs.
    return float(xlogy(successes, probability) + xlog1py(failures, -probability))


def coverage_tests(exceptions) -> pd.DataFrame:
    """Kupiec UC and Christoffersen independence/conditional coverage LR tests.

    Kupiec uses all indicators. Christoffersen conditions on the first, so its
    joint LR equals independence LR plus UC LR on the transition TARGETS,
    rather than adding the full-sample Kupiec statistic. Unobserved conditioning
    states make the Markov parameters unidentifiable and are flagged explicitly.
    Reference: Christoffersen (1998), Evaluating Interval Forecasts, sections 3.1-3.3.
    https://users.ssc.wisc.edu/~behansen/718/Christoffersen1998.pdf
    """
    flags = np.asarray(exceptions)
    if flags.ndim != 1 or not np.isin(flags, (0, 1)).all():
        raise ValueError("Coverage tests require a one-dimensional binary exception sequence.")
    probability = 1.0 - config.VAR_CONFIDENCE
    if not 0 < probability < 1 or not 0 < config.TEST_SIGNIFICANCE < 1:
        raise ValueError("Coverage probability and significance must lie strictly between zero and one.")
    flags = flags.astype(int)
    observations, count = len(flags), int(flags.sum())
    transitions = np.zeros((2, 2), dtype=int)
    for previous, current in zip(flags[:-1], flags[1:]):
        transitions[previous, current] += 1
    n00, n01, n10, n11 = map(int, transitions.ravel())
    uc, independence, conditional, conditioned_uc = np.nan, np.nan, np.nan, np.nan
    uc_reason = "No forecast indicators." if not observations else ""
    if observations:
        fitted = _bernoulli_log_likelihood(count, observations - count, count / observations)
        null = _bernoulli_log_likelihood(count, observations - count, probability)
        uc = max(0.0, 2 * (fitted - null))
    if observations < 2:
        markov_reason = "Fewer than two consecutive forecast indicators."
    elif n00 + n01 == 0 or n10 + n11 == 0:
        markov_reason = "An exception conditioning state has no outgoing transitions; its probability is unidentifiable."
    else:
        markov_reason = ""
        markov = (
            _bernoulli_log_likelihood(n01, n00, n01 / (n00 + n01))
            + _bernoulli_log_likelihood(n11, n10, n11 / (n10 + n11))
        )
        target_count, total = n01 + n11, observations - 1
        independent = _bernoulli_log_likelihood(target_count, total - target_count, target_count / total)
        conditional_null = _bernoulli_log_likelihood(target_count, total - target_count, probability)
        # LR statistics are nonnegative; max removes roundoff at equality.
        independence = max(0.0, 2 * (markov - independent))
        conditioned_uc = max(0.0, 2 * (independent - conditional_null))
        conditional = max(0.0, 2 * (markov - conditional_null))
    rows = []
    for name, statistic, degrees, reason, conditioned in (
        ("kupiec_unconditional", uc, 1, uc_reason, False),
        ("christoffersen_independence", independence, 1, markov_reason, True),
        ("christoffersen_conditional", conditional, 2, markov_reason, True),
    ):
        undefined = bool(reason)
        p_value = np.nan if undefined else float(chi2.sf(statistic, degrees))
        reject = None if undefined else bool(p_value < config.TEST_SIGNIFICANCE)
        rows.append({
            "test": name, "forecasts": observations, "exception_count": count,
            "exception_rate": count / observations if observations else np.nan,
            "expected_exception_rate": probability, "significance": config.TEST_SIGNIFICANCE,
            "n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "likelihood_observations": max(0, observations - 1) if conditioned else observations,
            "condition_on_first_indicator": conditioned,
            "statistic": statistic, "degrees_of_freedom": degrees, "p_value": p_value,
            "reject": reject,
            "decision": "undefined" if undefined else ("reject" if reject else "do_not_reject"),
            "undefined": undefined, "undefined_reason": reason,
            "conditioned_uc_statistic": conditioned_uc if name == "christoffersen_conditional" else np.nan,
        })
    return pd.DataFrame(rows)


def _euler_table(component_losses, quantile, holdings, origin_date, forecast_date, current_values) -> pd.DataFrame:
    contributions = euler_contributions(component_losses, quantile)
    total = float(contributions.sum())
    error = total - quantile.value
    tolerance = 32 * np.finfo(float).eps * max(1.0, abs(quantile.value), float(np.abs(contributions).sum()))
    if abs(error) > tolerance:
        raise ArithmeticError("Signed Euler contributions did not reconcile to portfolio VaR.")
    return pd.DataFrame([{
        "origin_date": origin_date, "forecast_date": forecast_date, "method": "parametric",
        "asset": asset, "quantity": holdings.quantities[i],
        "quantity_units": "USD face" if asset in config.INITIAL_BOND_MATURITIES_YEARS else "ETF units",
        "current_value_usd": current_values[i], "contribution_usd": contributions[i],
        "marginal_var_per_unit": contributions[i] / holdings.quantities[i],
        "lower_scenario_id": quantile.lower_scenario + 1, "upper_scenario_id": quantile.upper_scenario + 1,
        "lower_rank": quantile.lower_rank + 1, "upper_rank": quantile.upper_rank + 1,
        "lower_weight": 1 - quantile.upper_weight, "upper_weight": quantile.upper_weight,
        "lower_component_loss_usd": component_losses[quantile.lower_scenario, i],
        "upper_component_loss_usd": component_losses[quantile.upper_scenario, i],
        "quantile_boundary_tie": quantile.boundary_tie,
        "portfolio_var_usd": quantile.value, "sum_contributions_usd": total,
        "reconciliation_error_usd": error, "reconciliation_tolerance_usd": tolerance,
        "reconciled": abs(error) <= tolerance,
    } for i, asset in enumerate(config.ASSETS)])


def rolling_risk(levels: pd.DataFrame, factors: pd.DataFrame, holdings: FixedHoldings) -> RollingRisk:
    """Forecast the last 250 matched targets using the preceding 500 changes.

    Factor row j ends at level row j+1. An origin at level row i therefore uses
    factors.iloc[i-500:i], including the last change known at that origin and
    excluding its target. All historical shocks retain their observed joint rows.
    """
    first_origin = len(levels) - config.BACKTEST_FORECASTS - 1
    if config.VAR_HORIZON_STEPS != 1:
        raise ValueError("The rolling backtest requires the approved one-step horizon.")
    if first_origin < config.ESTIMATION_WINDOW:
        raise ValueError("Insufficient complete levels for the rolling estimation windows and targets.")
    if not factors.index.equals(levels.index[1:]) or tuple(factors.columns) != config.FACTOR_ORDER:
        raise ValueError("Rolling risk requires ordered factors matched to consecutive complete levels.")
    if holdings.inception_date != levels.index[first_origin].date():
        raise ValueError("Retrospective holdings must start at the first backtest forecast origin.")
    assets = config.ASSETS + ("portfolio",)
    rows, latest = [], {}
    for target in range(first_origin + 1, len(levels)):
        origin = target - 1
        elapsed = origin - first_origin
        origin_row, target_row = levels.iloc[origin], levels.iloc[target]
        origin_date, forecast_date = levels.index[origin].date().isoformat(), levels.index[target].date().isoformat()
        window = factors.iloc[origin - config.ESTIMATION_WINDOW:origin]
        current = value_holdings(
            holdings, origin_row.loc[list(config.TREASURY_SERIES.values())].to_numpy(),
            origin_row.loc[[f"{ticker}_Adj Close" for ticker in config.ETF_TICKERS]].to_numpy(), elapsed,
        )
        realized_next = value_holdings(
            holdings, target_row.loc[list(config.TREASURY_SERIES.values())].to_numpy(),
            target_row.loc[[f"{ticker}_Adj Close" for ticker in config.ETF_TICKERS]].to_numpy(), elapsed + 1,
        )
        component_realized = current - realized_next
        realized_losses = np.append(component_realized, component_realized.sum())
        current_all = np.append(current, current.sum())
        next_all = np.append(realized_next, realized_next.sum())
        mean, covariance = fit_gaussian(window)
        gaussian = config.parametric_var_rng(levels.index[target].date()).multivariate_normal(
            mean, covariance, size=config.PARAMETRIC_SCENARIOS,
            check_valid="raise", method=config.GAUSSIAN_SAMPLING_METHOD,
        )
        for method in config.RISK_METHODS:
            shocks = gaussian if method == "parametric" else window.to_numpy()
            losses = scenario_losses(holdings, origin_row, shocks, elapsed)
            portfolio_quantile = empirical_quantile(losses.sum(axis=1))
            standalone = np.array([empirical_quantile(losses[:, i]).value for i in range(len(config.ASSETS))])
            forecasts = np.append(standalone, portfolio_quantile.value)
            exceptions = exception_flags(realized_losses, forecasts)
            for i, asset in enumerate(assets):
                rows.append({
                    "origin_date": origin_date, "forecast_date": forecast_date,
                    "calibration_start_date": window.index[0].date().isoformat(),
                    "calibration_end_date": window.index[-1].date().isoformat(),
                    "calibration_changes": len(window), "elapsed_trading_steps": elapsed,
                    "method": method, "asset": asset, "scenario_count": len(shocks),
                    "confidence": config.VAR_CONFIDENCE,
                    "current_value_usd": current_all[i], "next_value_usd": next_all[i],
                    "var_usd": forecasts[i], "realized_pnl_usd": -realized_losses[i],
                    "realized_loss_usd": realized_losses[i], "exception": bool(exceptions[i]),
                })
            if target == len(levels) - 1:
                latest[method] = (standalone, portfolio_quantile, losses)
    backtest = pd.DataFrame(rows)
    tests = []
    for method in config.RISK_METHODS:
        for asset in assets:
            subset = backtest.loc[(backtest["method"] == method) & (backtest["asset"] == asset)]
            tests.append(coverage_tests(subset["exception"].to_numpy()).assign(method=method, asset=asset))
    diversification = []
    for method in config.RISK_METHODS:
        standalone, quantile, losses = latest[method]
        summed = float(standalone.sum())
        benefit = summed - quantile.value
        diversification.append({
            "origin_date": origin_date, "forecast_date": forecast_date, "method": method,
            "portfolio_var_usd": quantile.value, "sum_standalone_var_usd": summed,
            "diversification_benefit_usd": benefit,
            "diversification_fraction": benefit / summed if summed != 0 else np.nan,
            "fraction_defined": summed != 0,
        })
    euler = _euler_table(latest["parametric"][2], latest["parametric"][1], holdings, origin_date, forecast_date, current)
    coverage = pd.concat(tests, ignore_index=True)
    coverage = coverage[["method", "asset", *[column for column in coverage if column not in ("method", "asset")]]]
    return RollingRisk(holdings, backtest, coverage, pd.DataFrame(diversification), euler)
