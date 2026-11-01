"""Run the implemented data preparation and fixed-holding valuation stages."""

import pandas as pd

from portfolio_var import config
from portfolio_var.data import DataError, load_and_prepare
from portfolio_var.pricing import initialize_portfolios, value_holdings


def main() -> None:
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


if __name__ == "__main__":
    try:
        main()
    except DataError as exc:
        raise SystemExit(f"Pipeline failed: {exc}") from exc
