"""Zero-coupon valuation, total-return ETF marks, and fixed portfolio holdings."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from portfolio_var import config


def interpolate_yield(curve, maturity) -> np.ndarray:
    """Interpolate decimal zero yields; batch curves have tenors on the last axis.

    Output shape is curve.shape[:-1] + maturity.shape. Extrapolation is rejected.
    """
    curve = np.asarray(curve, dtype=float)
    maturity = np.asarray(maturity, dtype=float)
    tenors = np.asarray(config.YIELD_TENORS_YEARS, dtype=float)
    if curve.ndim == 0 or curve.shape[-1] != len(tenors) or not np.isfinite(curve).all():
        raise ValueError("A yield curve must contain six finite decimal zero yields in tenor order.")
    if not np.isfinite(maturity).all() or np.any((maturity < tenors[0]) | (maturity > tenors[-1])):
        raise ValueError("Remaining maturity must be within the observed 1-to-10-year curve; no extrapolation.")
    right = np.clip(np.searchsorted(tenors, maturity, side="right"), 1, len(tenors) - 1)
    left = right - 1
    fraction = (maturity - tenors[left]) / (tenors[right] - tenors[left])
    return curve[..., left] + fraction * (curve[..., right] - curve[..., left])


def age_maturities(initial_maturities, elapsed_steps: int) -> np.ndarray:
    """Age by retained trading observations, never by elapsed calendar days."""
    if not isinstance(elapsed_steps, (int, np.integer)) or elapsed_steps < 0:
        raise ValueError("Elapsed trading steps must be a nonnegative integer.")
    initial = np.asarray(initial_maturities, dtype=float)
    remaining = initial - elapsed_steps / config.TRADING_STEPS_PER_YEAR
    if not np.isfinite(initial).all() or np.any(initial < 0) or np.any(remaining < 0):
        raise ValueError("Maturities must be finite and cannot be aged past redemption.")
    return remaining


def bond_price(face_amount, zero_yield, maturity) -> np.ndarray:
    """Price a hypothetical zero-coupon holding: F * exp(-y(tau) * tau)."""
    face = np.asarray(face_amount, dtype=float)
    rate = np.asarray(zero_yield, dtype=float)
    maturity = np.asarray(maturity, dtype=float)
    if not all(np.isfinite(value).all() for value in (face, rate, maturity)):
        raise ValueError("Face amounts, yields, and maturities must be finite.")
    if np.any(face < 0) or np.any(maturity < 0):
        raise ValueError("Face amounts and remaining maturities must be nonnegative.")
    return face * np.exp(-rate * maturity)


def etf_total_return_mark(inception_close, inception_adj_close, current_adj_close) -> np.ndarray:
    """Preserve the inception quoted Close and compound subsequent total returns."""
    values = tuple(np.asarray(value, dtype=float) for value in (inception_close, inception_adj_close, current_adj_close))
    if any(not np.isfinite(value).all() or np.any(value <= 0) for value in values):
        raise ValueError("ETF Close and Adj Close inputs must be finite and positive.")
    close, initial_adjusted, adjusted = values
    return close * (adjusted / initial_adjusted)


@dataclass(frozen=True)
class FixedHoldings:
    """Quantities are face dollars for Treasuries, then ETF units, in asset order."""

    inception_date: date
    quantities: tuple[float, ...]
    inception_etf_close: tuple[float, ...]
    inception_etf_adj_close: tuple[float, ...]


def initialize_holdings(levels: pd.DataFrame, inception_date: date) -> FixedHoldings:
    """Allocate the prescribed capital once, allowing fractional quantities."""
    row = levels.loc[pd.Timestamp(inception_date)]
    curve = row.loc[list(config.TREASURY_SERIES.values())].to_numpy(dtype=float)
    maturities = np.array(list(config.INITIAL_BOND_MATURITIES_YEARS.values()))
    closes = row.loc[[f"{ticker}_Close" for ticker in config.ETF_TICKERS]].to_numpy(dtype=float)
    adjusted = row.loc[[f"{ticker}_Adj Close" for ticker in config.ETF_TICKERS]].to_numpy(dtype=float)
    etf_marks = etf_total_return_mark(closes, adjusted, adjusted)
    unit_prices = np.concatenate([bond_price(1.0, interpolate_yield(curve, maturities), maturities), etf_marks])
    allocations = config.INITIAL_CAPITAL_USD * np.array([config.INITIAL_WEIGHTS[asset] for asset in config.ASSETS])
    quantities = allocations / unit_prices
    return FixedHoldings(
        pd.Timestamp(inception_date).date(), tuple(quantities), tuple(closes), tuple(adjusted),
    )


def value_holdings(holdings: FixedHoldings, curve, current_adj_close, elapsed_steps: int) -> np.ndarray:
    """Fully reprice the same five quantities, including Treasury aging.

    Curves and adjusted ETF prices may contain broadcast-compatible scenario
    dimensions; the final output axis follows config.ASSETS.
    """
    curve = np.asarray(curve, dtype=float)
    adjusted = np.asarray(current_adj_close, dtype=float)
    if adjusted.ndim == 0 or adjusted.shape[-1] != len(config.ETF_TICKERS):
        raise ValueError("Adjusted ETF prices must have LQD and SPY on the final axis.")
    maturities = age_maturities(list(config.INITIAL_BOND_MATURITIES_YEARS.values()), elapsed_steps)
    # A redeemed zero-coupon holding pays its face amount. This boundary is
    # outside the approved one-year horizon, but needs no yield extrapolation.
    positive = maturities > 0
    bond_marks = np.ones(curve.shape[:-1] + maturities.shape)
    if positive.any():
        bond_marks[..., positive] = bond_price(
            1.0, interpolate_yield(curve, maturities[positive]), maturities[positive],
        )
    etf_marks = etf_total_return_mark(
        holdings.inception_etf_close, holdings.inception_etf_adj_close, adjusted,
    )
    batch_shape = np.broadcast_shapes(bond_marks.shape[:-1], etf_marks.shape[:-1])
    marks = np.concatenate([
        np.broadcast_to(bond_marks, batch_shape + (len(maturities),)),
        np.broadcast_to(etf_marks, batch_shape + (len(config.ETF_TICKERS),)),
    ], axis=-1)
    return marks * np.asarray(holdings.quantities)


def initialize_portfolios(levels: pd.DataFrame) -> dict[str, FixedHoldings]:
    """Give the forward and retrospective portfolios separate inception holdings.

    With N levels, the last 250 forecast targets start at position N-250.
    The first origin is one position earlier and already has 500 past changes.
    """
    if len(levels) < config.MIN_ALIGNED_LEVELS:
        raise ValueError(f"Portfolio inception requires at least {config.MIN_ALIGNED_LEVELS} complete aligned levels.")
    return {
        "forward": initialize_holdings(levels, levels.index[-1].date()),
        "retrospective": initialize_holdings(levels, levels.index[-config.BACKTEST_FORECASTS - 1].date()),
    }
