"""Terminal tables, distribution figures, and the generated configuration record."""

from datetime import date
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform

import pandas as pd

from portfolio_var import config
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
