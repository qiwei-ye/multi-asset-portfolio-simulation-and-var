"""Terminal tables, distribution figures, and the generated configuration record."""

from datetime import date
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

import pandas as pd

from portfolio_var import config
from portfolio_var.risk import RollingRisk
from portfolio_var.simulation import ForwardSimulation


def _json_default(value):
    if isinstance(value, Path):
        return value.relative_to(config.PROJECT_ROOT).as_posix() if value.is_relative_to(config.PROJECT_ROOT) else value.as_posix()
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Cannot serialize configuration value of type {type(value).__name__}.")


def write_frozen_config() -> None:
    """Export effective config constants and every named seed assignment.

    This generated record is never loaded as settings. All consumers continue
    to use config.py; paths inside the project are recorded relative to it.
    """
    payload = {
        "schema_version": 1,
        "authority": "portfolio_var/config.py",
        "settings": {
            name: value for name, value in vars(config).items()
            if name.isupper() and not name.startswith("_")
        },
        "randomness": config.seed_manifest(),
        "environment": {
            "python_version": platform.python_version(),
            "dependency_versions": {name: version(name) for name in ("pandas", "numpy", "scipy", "requests", "yfinance", "matplotlib")},
        },
        "terminal_output_units": {
            "treasury_prices": "USD per USD 1 face at remaining maturity",
            "etf_prices": "USD per unit, inception Close times cumulative Adj Close total-return ratio",
            "position_and_portfolio_values": "USD",
            "returns": "Simple decimal returns relative to each inception value",
            "path_id": "One-based joint scenario identifier shared by every asset and the portfolio",
        },
        "risk_output_units": {
            "var_and_losses": "Signed USD; loss is current value minus next-step value",
            "pnl": "Signed USD; the negative of loss",
            "exception_rule": "Realized loss strictly greater than forecast VaR",
            "latest_summaries": config.LATEST_RISK_CONVENTION,
            "coverage_likelihoods": config.COVERAGE_LIKELIHOOD_CONVENTION,
            "coverage_reference": "https://users.ssc.wisc.edu/~behansen/718/Christoffersen1998.pdf",
        },
    }
    config.FROZEN_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.FROZEN_CONFIG_PATH.write_text(
        json.dumps(payload, default=_json_default, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


def terminal_outcomes(simulation: ForwardSimulation) -> pd.DataFrame:
    """One row per joint path, retaining prices, position outcomes, and totals."""
    columns = {}
    for i, asset in enumerate(config.ASSETS):
        columns[f"{asset}_terminal_price"] = simulation.terminal_prices[:, i]
        columns[f"{asset}_terminal_value_usd"] = simulation.position_values[:, i]
        columns[f"{asset}_return"] = simulation.position_returns[:, i]
    columns["portfolio_terminal_value_usd"] = simulation.portfolio_values
    columns["portfolio_return"] = simulation.portfolio_returns
    return pd.DataFrame(columns, index=pd.RangeIndex(1, len(simulation.portfolio_values) + 1, name="path_id"))


def plot_terminal_distributions(simulation: ForwardSimulation) -> None:
    """Render all five asset return marginals and their shared portfolio total."""
    os.environ.setdefault("MPLCONFIGDIR", str(config.MATPLOTLIB_CACHE_DIR))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 8.5), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    maturities = list(config.TERMINAL_BOND_MATURITIES_YEARS.values())
    initial_maturities = list(config.INITIAL_BOND_MATURITIES_YEARS.values())
    titles = [
        f"{initial:g}-year Treasury\n{remaining:g} year{'s' if remaining != 1 else ''} remaining"
        for initial, remaining in zip(initial_maturities, maturities)
    ] + list(config.ETF_TICKERS) + ["Portfolio"]
    returns = [simulation.position_returns[:, i] for i in range(len(config.ASSETS))] + [simulation.portfolio_returns]
    for i, (axis, title, outcomes) in enumerate(zip(axes.flat, titles, returns)):
        color = "#33658a" if i < len(config.ASSETS) else "#34785b"
        axis.hist(outcomes * 100, bins=60, density=True, color=color, alpha=0.9, edgecolor="white", linewidth=0.3)
        axis.axvline(0, color="#555555", linewidth=1, linestyle="--")
        axis.set_title(title, fontsize=11, pad=9)
        axis.set_xlabel("One-year return (%)", fontsize=9)
        axis.set_ylabel("Probability density", fontsize=9)
        axis.tick_params(labelsize=9)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("One-year terminal return distributions", fontsize=17, y=0.97)
    start, end = simulation.calibration_dates[0].date(), simulation.calibration_dates[-1].date()
    figure.text(
        0.5, 0.927,
        f"{len(simulation.portfolio_values):,} joint Gaussian paths | {simulation.steps} trading steps | "
        f"Inception {simulation.holdings.inception_date}\n"
        f"Daily calibration: {start} to {end} ({len(simulation.calibration_dates)} aligned changes)",
        ha="center", va="center", fontsize=10, color="#444444",
    )
    figure.text(
        0.5, 0.025,
        "Dashed lines indicate zero return. ETF outcomes use total-return marks; Treasury holdings are zero-coupon bonds.\n"
        "All positions stay fixed. Calibration sources: Federal Reserve Board and Yahoo Finance.",
        ha="center", va="center", fontsize=9, color="#444444",
    )
    figure.subplots_adjust(left=0.075, right=0.975, bottom=0.11, top=0.825, wspace=0.31, hspace=0.53)
    config.TERMINAL_DISTRIBUTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(config.TERMINAL_DISTRIBUTIONS_PATH, dpi=160)


def write_terminal_reports(simulation: ForwardSimulation) -> None:
    """Write the pathwise table and its distribution figure."""
    config.TERMINAL_OUTCOMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    terminal_outcomes(simulation).to_csv(
        config.TERMINAL_OUTCOMES_PATH, float_format="%.17g", lineterminator="\n",
    )
    plot_terminal_distributions(simulation)


def plot_var_backtest(risk: RollingRisk) -> None:
    """Compare both dated VaR forecasts with losses and method-specific exceptions."""
    os.environ.setdefault("MPLCONFIGDIR", str(config.MATPLOTLIB_CACHE_DIR))
    from matplotlib import dates as mdates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(13, 9.5), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    titles = [f"{maturity:g}-year Treasury holding" for maturity in config.INITIAL_BOND_MATURITIES_YEARS.values()] + list(config.ETF_TICKERS) + ["Portfolio"]
    for axis, asset, title in zip(axes.flat, config.ASSETS + ("portfolio",), titles):
        asset_rows = risk.backtest.loc[risk.backtest["asset"] == asset]
        parametric = asset_rows.loc[asset_rows["method"] == "parametric"].sort_values("forecast_date")
        historical = asset_rows.loc[asset_rows["method"] == "historical"].sort_values("forecast_date")
        dates = pd.to_datetime(parametric["forecast_date"])
        axis.plot(dates, parametric["realized_loss_usd"] / 1000, color="#666666", linewidth=0.8, label="Realized loss")
        axis.plot(dates, parametric["var_usd"] / 1000, color="#33658a", linewidth=1.25, label="Parametric 99% VaR")
        axis.plot(pd.to_datetime(historical["forecast_date"]), historical["var_usd"] / 1000, color="#c47b20", linewidth=1.25, label="Historical 99% VaR")
        parametric_hits = parametric.loc[parametric["exception"]]
        historical_hits = historical.loc[historical["exception"]]
        axis.scatter(pd.to_datetime(parametric_hits["forecast_date"]), parametric_hits["realized_loss_usd"] / 1000, marker="o", s=45, facecolors="none", edgecolors="#33658a", linewidths=1.3, zorder=4, label="Parametric exception")
        axis.scatter(pd.to_datetime(historical_hits["forecast_date"]), historical_hits["realized_loss_usd"] / 1000, marker="x", s=30, color="#c47b20", linewidths=1.2, zorder=5, label="Historical exception")
        axis.axhline(0, color="#888888", linewidth=0.6)
        axis.set_title(title, fontsize=11, pad=9)
        axis.set_ylabel("Loss / VaR (USD thousands)", fontsize=9)
        axis.tick_params(labelsize=9)
        axis.text(0.02, 0.96, f"Exceptions: P {len(parametric_hits)} / H {len(historical_hits)}", transform=axis.transAxes, va="top", fontsize=8, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        axis.xaxis.set_major_locator(locator)
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    dates = pd.to_datetime(risk.backtest["forecast_date"])
    count = risk.backtest["forecast_date"].nunique()
    figure.suptitle("99% one-step VaR backtest", fontsize=17, y=0.98)
    figure.text(
        0.5, 0.942,
        f"{count} matched forecasts: {dates.min().date()} to {dates.max().date()} | "
        f"{config.ESTIMATION_WINDOW} preceding changes at each origin\n"
        f"Fixed portfolio inception {risk.holdings.inception_date}; Treasury maturities shown at inception",
        ha="center", va="center", fontsize=10, color="#444444",
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.908), ncol=3, frameon=False, fontsize=9)
    figure.text(
        0.5, 0.025,
        "Loss = current value minus next-step value. Markers show strict exceedances of the matching VaR.\n"
        "Bonds age at each retained observation; positions stay fixed. Each retained interval is one trading step.",
        ha="center", va="center", fontsize=9, color="#444444",
    )
    figure.subplots_adjust(left=0.075, right=0.975, bottom=0.105, top=0.805, wspace=0.30, hspace=0.40)
    config.VAR_BACKTEST_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(config.VAR_BACKTEST_FIGURE_PATH, dpi=160)


def write_risk_reports(risk: RollingRisk) -> None:
    """Persist matched forecasts, all test outputs, and latest risk summaries."""
    for path, table in (
        (config.BACKTEST_PATH, risk.backtest),
        (config.COVERAGE_TESTS_PATH, risk.coverage_tests),
        (config.DIVERSIFICATION_PATH, risk.diversification),
        (config.EULER_CONTRIBUTIONS_PATH, risk.euler_contributions),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
    plot_var_backtest(risk)
