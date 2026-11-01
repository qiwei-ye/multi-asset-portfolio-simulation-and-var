"""Joint IID Gaussian shocks and full repricing of one-year terminal holdings."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_var import config
from portfolio_var.pricing import FixedHoldings, value_holdings


@dataclass(frozen=True)
class ForwardSimulation:
    """Every terminal array uses the same paths and config.ASSETS ordering.

    Treasury prices are USD per USD 1 face; ETF prices are total-return marks
    per unit. Position and portfolio values are USD; returns are simple decimals.
    """

    holdings: FixedHoldings
    calibration_dates: pd.DatetimeIndex
    daily_mean: np.ndarray
    daily_covariance: np.ndarray
    steps: int
    cumulative_shocks: np.ndarray
    initial_values: np.ndarray
    terminal_prices: np.ndarray
    position_values: np.ndarray
    position_returns: np.ndarray
    portfolio_values: np.ndarray
    portfolio_returns: np.ndarray


def fit_gaussian(window: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the daily mean and sample covariance in the fixed factor order."""
    if tuple(window.columns) != config.FACTOR_ORDER:
        raise ValueError("Gaussian calibration requires the configured eight-factor ordering.")
    values = window.to_numpy(dtype=float)
    if len(values) < max(2, config.COVARIANCE_DDOF + 1) or not np.isfinite(values).all():
        raise ValueError("Gaussian calibration requires enough finite observed joint shocks.")
    mean = values.mean(axis=0) if config.ESTIMATE_SAMPLE_MEAN else np.zeros(values.shape[1])
    covariance = np.cov(values, rowvar=False, ddof=config.COVARIANCE_DDOF)
    return mean, covariance


def simulate_forward(
    levels: pd.DataFrame, factors: pd.DataFrame, holdings: FixedHoldings,
) -> ForwardSimulation:
    """Draw 252 daily joint shocks per path, accumulate them, and reprice once.

    Standalone outcomes are columns of this joint simulation, not independent
    reruns. Summing daily shocks retains IID trading steps and their estimated
    mean; no annualized substitute draw or recentering is used.
    """
    if levels.empty or holdings.inception_date != levels.index[-1].date():
        raise ValueError("Forward holdings must start on the latest aligned observation.")
    if not factors.index.equals(levels.index[1:]):
        raise ValueError("Factor dates must match changes between consecutive aligned levels.")
    if len(factors) < config.ESTIMATION_WINDOW:
        raise ValueError(f"Forward calibration requires {config.ESTIMATION_WINDOW} aligned changes.")
    if config.SIMULATION_PATHS <= 0 or config.SIMULATION_STEPS <= 0:
        raise ValueError("Forward paths and trading steps must be positive.")
    window = factors.iloc[-config.ESTIMATION_WINDOW:]
    mean, covariance = fit_gaussian(window)
    rng = config.simulation_rng()
    cumulative_shocks = rng.multivariate_normal(
        mean, covariance, size=(config.SIMULATION_PATHS, config.SIMULATION_STEPS),
        check_valid="raise", method=config.GAUSSIAN_SAMPLING_METHOD,
    ).sum(axis=1)
    origin = levels.iloc[-1]
    curve = origin.loc[list(config.TREASURY_SERIES.values())].to_numpy(dtype=float)
    adjusted = origin.loc[[f"{ticker}_Adj Close" for ticker in config.ETF_TICKERS]].to_numpy(dtype=float)
    initial_values = value_holdings(holdings, curve, adjusted, 0)
    yield_count = len(config.TREASURY_SERIES)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            terminal_curve = curve + cumulative_shocks[:, :yield_count]
            terminal_adjusted = adjusted * np.exp(cumulative_shocks[:, yield_count:])
            position_values = value_holdings(
                holdings, terminal_curve, terminal_adjusted, config.SIMULATION_STEPS,
            )
            terminal_prices = position_values / np.asarray(holdings.quantities)
            position_returns = position_values / initial_values - 1.0
            portfolio_values = position_values.sum(axis=1)
            portfolio_returns = portfolio_values / initial_values.sum() - 1.0
    except FloatingPointError as exc:
        raise ValueError("Forward shocks could not produce finite terminal valuations.") from exc
    return ForwardSimulation(
        holdings=holdings, calibration_dates=window.index.copy(),
        daily_mean=mean, daily_covariance=covariance, steps=config.SIMULATION_STEPS,
        cumulative_shocks=cumulative_shocks, initial_values=initial_values,
        terminal_prices=terminal_prices, position_values=position_values,
        position_returns=position_returns, portfolio_values=portfolio_values,
        portfolio_returns=portfolio_returns,
    )
