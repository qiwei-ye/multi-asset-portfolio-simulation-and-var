"""Authoritative paths, approved MVP settings, and reproducible random streams.

Paths are declarations; create a directory only when writing its contents.
Loss is current value minus next-step value; exceptions use strict exceedance.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TREASURY_SNAPSHOT_PATH = DATA_DIR / "treasury_yields.csv"
ETF_SNAPSHOT_PATH = DATA_DIR / "etf_prices.csv"
YFINANCE_CACHE_DIR = DATA_DIR / ".yfinance_cache"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RETRIEVAL_MANIFEST_PATH = RESULTS_DIR / "retrieval_manifest.json"
TERMINAL_OUTCOMES_PATH = RESULTS_DIR / "terminal_outcomes.csv"
TERMINAL_DISTRIBUTIONS_PATH = FIGURES_DIR / "terminal_distributions.png"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FROZEN_CONFIG_PATH = ARTIFACTS_DIR / "frozen_config.json"
MATPLOTLIB_CACHE_DIR = PROJECT_ROOT / ".venv" / ".matplotlib"

DATA_START_DATE = date(2022, 9, 1)
DATA_END_DATE = date(2026, 8, 31)  # Inclusive; never extend coverage silently.
TREASURY_DATA_SOURCE = "Federal Reserve fitted zero-coupon yields"
TREASURY_DATA_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
TREASURY_SOURCE_URL = "https://www.federalreserve.gov/data/nominal-yield-curve.htm"
TREASURY_RIGHTS_URL = "https://www.federalreserve.gov/disclaimer.htm"
TREASURY_SERIES = {
    1: "SVENY01",
    2: "SVENY02",
    4: "SVENY04",
    5: "SVENY05",
    9: "SVENY09",
    10: "SVENY10",
}
YIELD_TENORS_YEARS = tuple(TREASURY_SERIES)
YIELD_PERCENT_TO_DECIMAL = 0.01
YIELD_INTERPOLATION = "linear"
ETF_DATA_SOURCE = "Yahoo Finance"
ETF_TICKERS = ("LQD", "SPY")
ETF_PRICE_FIELDS = ("Close", "Adj Close")
ETF_LEVEL_COLUMNS = tuple(
    f"{ticker}_{field}" for ticker in ETF_TICKERS for field in ETF_PRICE_FIELDS
)
YFINANCE_AUTO_ADJUST = False  # Retain both quoted Close and Adj Close.
ETF_RIGHTS_URL = "https://help.yahoo.com/kb/SLN2310.html"
YFINANCE_DOCUMENTATION_URL = "https://ranaroussi.github.io/yfinance/"
SOURCE_PERMISSIONS_REVIEW_DATE = date(2026, 10, 31)
DOWNLOAD_TIMEOUT_SECONDS = 30

INITIAL_CAPITAL_USD = 1_000_000.0
INITIAL_BOND_MATURITIES_YEARS = {"UST_2Y": 2.0, "UST_5Y": 5.0, "UST_10Y": 10.0}
ASSETS = tuple(INITIAL_BOND_MATURITIES_YEARS) + ETF_TICKERS
WEIGHT_RATIOS = (1, 1, 1, 3, 4)
INITIAL_WEIGHTS = {
    asset: ratio / sum(WEIGHT_RATIOS)
    for asset, ratio in zip(ASSETS, WEIGHT_RATIOS, strict=True)
}
ALLOW_FRACTIONAL_HOLDINGS = True
# Each portfolio fixes its own holdings at inception, without rebalancing,
# transaction costs, taxes, or funding costs. ETF marks use inception Close
# times the subsequent Adj Close ratio. Bond yields are continuous rates.

# This ordering governs every shock vector, mean, and covariance matrix.
FACTOR_ORDER = tuple(f"yield_change_{tenor}y" for tenor in YIELD_TENORS_YEARS) + tuple(
    f"{ticker}_log_total_return" for ticker in ETF_TICKERS
)
ESTIMATION_WINDOW = 500
ESTIMATE_SAMPLE_MEAN = True
COVARIANCE_DDOF = 1
GAUSSIAN_SAMPLING_METHOD = "svd"  # Supports singular sample covariance without jitter.
BACKTEST_FORECASTS = 250
MIN_ALIGNED_CHANGES = ESTIMATION_WINDOW + BACKTEST_FORECASTS
MIN_ALIGNED_LEVELS = MIN_ALIGNED_CHANGES + 1

# Consecutive complete aligned observations each advance this trading clock.
TRADING_STEPS_PER_YEAR = 252
TIME_STEP_YEARS = 1.0 / TRADING_STEPS_PER_YEAR
SIMULATION_PATHS = 10_000
SIMULATION_STEPS = 252
SIMULATION_HORIZON_YEARS = SIMULATION_STEPS / TRADING_STEPS_PER_YEAR
TERMINAL_BOND_MATURITIES_YEARS = {
    asset: maturity - SIMULATION_HORIZON_YEARS
    for asset, maturity in INITIAL_BOND_MATURITIES_YEARS.items()
}
# Standalone outcomes are marginals of the same joint simulation.
VAR_HORIZON_STEPS = 1
VAR_CONFIDENCE = 0.99
PARAMETRIC_SCENARIOS = 10_000
HISTORICAL_SCENARIOS = ESTIMATION_WINDOW  # Joint observed shocks, not recentered.
QUANTILE_METHOD = "linear"
TEST_SIGNIFICANCE = 0.05

MASTER_SEED = 42
# Append new branch names; never reorder these assignments.
SEED_BRANCH_NAMES = ("forward_simulation", "parametric_var")
_SEED_BRANCHES = dict(
    zip(
        SEED_BRANCH_NAMES,
        np.random.SeedSequence(MASTER_SEED).spawn(len(SEED_BRANCH_NAMES)),
        strict=True,
    )
)
# Reserve one child for EVERY calendar date before running any stage. Unused
# weekend/missing-date slots keep actual forecast assignments independent of
# data filtering, forecast subsets, and execution order. Keys are target dates
# (the next aligned observation), not the preceding calibration origin dates.
FORECAST_SEED_DATES = tuple(
    DATA_START_DATE + timedelta(days=offset)
    for offset in range((DATA_END_DATE - DATA_START_DATE).days + 1)
)
_FORECAST_SEEDS = dict(
    zip(
        FORECAST_SEED_DATES,
        _SEED_BRANCHES["parametric_var"].spawn(len(FORECAST_SEED_DATES)),
        strict=True,
    )
)


def simulation_rng() -> np.random.Generator:
    """Return a fresh stream for the joint forward simulation and its marginals."""
    return np.random.Generator(np.random.PCG64(_SEED_BRANCHES["forward_simulation"]))


def parametric_var_rng(forecast_date: date) -> np.random.Generator:
    """Return a fresh stream for a forecast's target (next-observation) date."""
    try:
        seed = _FORECAST_SEEDS[forecast_date]
    except KeyError as exc:
        raise ValueError(
            f"Forecast target date must be a datetime.date between "
            f"{DATA_START_DATE} and {DATA_END_DATE}, inclusive: {forecast_date!r}"
        ) from exc
    return np.random.Generator(np.random.PCG64(seed))


def seed_manifest() -> dict[str, object]:
    """Return JSON-ready entropy and spawn keys for saving with run settings."""
    return {
        "master_seed": MASTER_SEED,
        "bit_generator": "PCG64",
        "pool_size": _SEED_BRANCHES["forward_simulation"].pool_size,
        "branch_order": list(SEED_BRANCH_NAMES),
        "branches": {
            name: list(seed.spawn_key) for name, seed in _SEED_BRANCHES.items()
        },
        "forecast_date_convention": "target (next aligned observation) date",
        "parametric_var_by_forecast_date": {
            day.isoformat(): list(seed.spawn_key)
            for day, seed in _FORECAST_SEEDS.items()
        },
    }
