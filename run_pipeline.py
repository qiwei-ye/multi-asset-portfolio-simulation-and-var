"""Run data preparation, forward simulation, rolling VaR, and all reports."""

import pandas as pd

from portfolio_var import config
from portfolio_var.data import DataError, load_and_prepare
from portfolio_var.pricing import initialize_portfolios, value_holdings
from portfolio_var.reporting import write_frozen_config, write_risk_reports, write_terminal_reports
from portfolio_var.risk import rolling_risk
from portfolio_var.simulation import simulate_forward


def main() -> None:
    write_frozen_config()
    prepared = load_and_prepare()
    portfolios = initialize_portfolios(prepared.levels)
    print(
        f"Prepared {len(prepared.levels)} complete levels and {len(prepared.factors)} "
        f"eight-factor changes: {prepared.levels.index[0].date()} through {prepared.levels.index[-1].date()}."
    )
    for name, holdings in portfolios.items():
        row = prepared.levels.loc[pd.Timestamp(holdings.inception_date)]
        values = value_holdings(
            holdings, row.loc[list(config.TREASURY_SERIES.values())].to_numpy(),
            row.loc[[f"{ticker}_Adj Close" for ticker in config.ETF_TICKERS]].to_numpy(), 0,
        )
        print(f"{name.capitalize()} inception {holdings.inception_date}: USD {values.sum():,.2f} across {len(values)} fixed holdings.")
    print(f"Retrieval provenance and alignment: {config.RETRIEVAL_MANIFEST_PATH.relative_to(config.PROJECT_ROOT)}")
    simulation = simulate_forward(prepared.levels, prepared.factors, portfolios["forward"])
    write_terminal_reports(simulation)
    print(f"Simulated {len(simulation.portfolio_values):,} joint paths over {simulation.steps} trading steps.")
    risk = rolling_risk(prepared.levels, prepared.factors, portfolios["retrospective"])
    write_risk_reports(risk)
    print(f"Backtested {risk.backtest['forecast_date'].nunique()} matched forecasts per asset and portfolio under both VaR methods.")
    print(f"Latest parametric Euler reconciliation error: USD {risk.euler_contributions['reconciliation_error_usd'].iloc[0]:.3g}.")
    for path in (
        config.TERMINAL_OUTCOMES_PATH, config.TERMINAL_DISTRIBUTIONS_PATH,
        config.BACKTEST_PATH, config.COVERAGE_TESTS_PATH, config.DIVERSIFICATION_PATH,
        config.EULER_CONTRIBUTIONS_PATH, config.VAR_BACKTEST_FIGURE_PATH, config.FROZEN_CONFIG_PATH,
    ):
        print(f"Saved {path.relative_to(config.PROJECT_ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except DataError as exc:
        raise SystemExit(f"Pipeline failed: {exc}") from exc
