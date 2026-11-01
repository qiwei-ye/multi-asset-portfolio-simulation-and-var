"""Consolidated preparation and valuation tests; fixtures are synthetic, not inputs."""

from datetime import date
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import requests

from portfolio_var import config, data, pricing


def input_frames(dates=None):
    if dates is None:
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    position = np.arange(len(dates))
    treasury = pd.DataFrame(
        {series: 4.0 + 0.001 * position for series in config.TREASURY_SERIES.values()}, index=dates,
    ).rename_axis("Date")
    etfs = pd.DataFrame({
        "LQD_Close": 100.0 + position,
        "LQD_Adj Close": 50.0 + 5 * position,
        "SPY_Close": 400.0 + 2 * position,
        "SPY_Adj Close": 200.0 + 10 * position,
    }, index=dates).rename_axis("Date")
    return treasury, etfs


class PreparationTests(unittest.TestCase):
    def test_alignment_precedes_changes_and_never_fills(self):
        treasury, etfs = input_frames()
        for i, series in enumerate(("SVENY01", "SVENY02", "SVENY04", "SVENY05", "SVENY09", "SVENY10"), start=1):
            treasury[series] = i + 0.001 * i * np.arange(len(treasury))
        treasury.loc["2024-01-05", "SVENY04"] = np.nan
        etfs.loc["2024-01-03", "LQD_Close"] = np.nan
        prepared = data.prepare_data(treasury.iloc[::-1], etfs.iloc[[2, 0, 4, 1, 3]])
        self.assertEqual(prepared.levels.index.strftime("%Y-%m-%d").tolist(), ["2024-01-02", "2024-01-04", "2024-01-08"])
        self.assertEqual(tuple(prepared.factors.columns), (
            "yield_change_1y", "yield_change_2y", "yield_change_4y", "yield_change_5y",
            "yield_change_9y", "yield_change_10y", "LQD_log_total_return", "SPY_log_total_return",
        ))
        np.testing.assert_allclose(prepared.levels["SVENY01"], [0.01, 0.01002, 0.01004])
        np.testing.assert_allclose(
            prepared.factors.iloc[:, :6],
            [[0.00002, 0.00004, 0.00006, 0.00008, 0.00010, 0.00012]] * 2,
            atol=1e-16,
        )
        np.testing.assert_allclose(prepared.factors["LQD_log_total_return"], [math.log(60 / 50), math.log(70 / 60)])
        np.testing.assert_allclose(prepared.factors["SPY_log_total_return"], [math.log(220 / 200), math.log(240 / 220)])
        self.assertEqual(prepared.alignment["retained_change_count"], 2)
        self.assertEqual(prepared.alignment["dropped_union_dates"], ["2024-01-03", "2024-01-05"])
        self.assertFalse(prepared.alignment["coverage_sufficient"])
        self.assertEqual(treasury.loc["2024-01-02", "SVENY01"], 1.0)

    def test_changes_do_not_use_levels_outside_the_fixed_period(self):
        dates = pd.to_datetime(["2022-08-31", "2022-09-01", "2026-08-31", "2026-09-01"])
        treasury, etfs = input_frames(dates)
        prepared = data.prepare_data(treasury, etfs)
        self.assertEqual(prepared.levels.index.tolist(), dates[1:3].tolist())
        self.assertEqual(prepared.factors.index.tolist(), [pd.Timestamp("2026-08-31")])
        self.assertEqual(prepared.alignment["requested_start_inclusive"], "2022-09-01")
        self.assertEqual(prepared.alignment["requested_end_inclusive"], "2026-08-31")

    def test_absent_source_dates_are_dropped_before_ratios(self):
        treasury, etfs = input_frames()
        prepared = data.prepare_data(treasury, etfs.drop(etfs.index[1]))
        self.assertNotIn(pd.Timestamp("2024-01-03"), prepared.levels.index)
        self.assertAlmostEqual(prepared.factors.iloc[0]["LQD_log_total_return"], math.log(60 / 50))

    def test_duplicate_dates_are_rejected(self):
        treasury, etfs = input_frames()
        with self.assertRaisesRegex(data.DataError, "unique"):
            data.prepare_data(pd.concat([treasury, treasury.iloc[[0]]]), etfs)

    def test_invalid_external_observations_are_rejected(self):
        for column, value in (("LQD_Close", 0), ("SPY_Adj Close", -1), ("SPY_Close", np.inf)):
            with self.subTest(column=column, value=value):
                treasury, etfs = input_frames()
                etfs.loc[etfs.index[0], column] = value
                with self.assertRaises(data.DataError):
                    data.prepare_data(treasury, etfs)
        treasury, etfs = input_frames()
        treasury = treasury.astype(object)
        treasury.iloc[0, 0] = "not a yield"
        with self.assertRaisesRegex(data.DataError, "nonnumeric"):
            data.prepare_data(treasury, etfs)

    def test_invalid_dates_and_missing_required_fields_are_rejected(self):
        treasury, etfs = input_frames()
        invalid = treasury.copy()
        invalid.index = ["invalid", *treasury.index[1:].strftime("%Y-%m-%d")]
        with self.assertRaisesRegex(data.DataError, "dates"):
            data.prepare_data(invalid, etfs)
        with self.assertRaisesRegex(data.DataError, "required columns"):
            data.prepare_data(treasury, etfs.drop(columns="SPY_Adj Close"))

    def test_empty_complete_alignment_is_reported_without_fabrication(self):
        treasury, etfs = input_frames()
        etfs.loc[:, "LQD_Close"] = np.nan
        prepared = data.prepare_data(treasury, etfs)
        self.assertTrue(prepared.levels.empty)
        self.assertTrue(prepared.factors.empty)
        self.assertFalse(prepared.alignment["coverage_sufficient"])
        self.assertEqual(prepared.alignment["retained_level_dates"], [])


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="portfolio-data-test-", dir=config.PROJECT_ROOT / ".venv")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        paths = patch.multiple(
            config,
            TREASURY_SNAPSHOT_PATH=root / "data" / "treasury_yields.csv",
            ETF_SNAPSHOT_PATH=root / "data" / "etf_prices.csv",
            YFINANCE_CACHE_DIR=root / "data" / ".yfinance_cache",
            RETRIEVAL_MANIFEST_PATH=root / "results" / "retrieval_manifest.json",
        )
        paths.start()
        self.addCleanup(paths.stop)
        self.treasury, self.etfs = input_frames(pd.bdate_range("2022-09-01", periods=751))

    def fetch(self):
        raw = ("Source research-product preamble\n\n" + self.treasury.to_csv(float_format="%.17g", lineterminator="\n")).encode()
        returned = self.etfs.copy()
        returned.columns = pd.MultiIndex.from_tuples([
            (field, ticker) for ticker in config.ETF_TICKERS for field in config.ETF_PRICE_FIELDS
        ], names=["Price", "Ticker"])
        response = Mock(content=raw, url=config.TREASURY_DATA_URL, status_code=200, headers={})
        with patch.object(data.requests, "get", return_value=response) as request, patch.object(data.yf, "download", return_value=returned) as download, patch.object(data.yf, "set_tz_cache_location"):
            prepared = data.load_and_prepare()
        self.assertEqual(request.call_count, 1)
        self.assertEqual(download.call_args.kwargs["end"], "2026-09-01")
        self.assertFalse(download.call_args.kwargs["auto_adjust"])
        self.assertFalse(download.call_args.kwargs["repair"])
        return prepared, raw

    def manifest(self):
        return json.loads(config.RETRIEVAL_MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_snapshots_are_exact_and_ordinary_reruns_make_no_requests(self):
        self.etfs.iloc[0, 0] = np.nextafter(100.0, np.inf)
        first, raw = self.fetch()
        self.assertEqual(config.TREASURY_SNAPSHOT_PATH.read_bytes(), raw)
        pd.testing.assert_frame_equal(data.load_snapshot(config.ETF_SNAPSHOT_PATH, "etf"), self.etfs, check_freq=False)
        before = self.manifest()
        modified_times = [path.stat().st_mtime_ns for path in (config.TREASURY_SNAPSHOT_PATH, config.ETF_SNAPSHOT_PATH)]
        with patch.object(data.requests, "get", side_effect=AssertionError("Unexpected network request")), patch.object(data.yf, "download", side_effect=AssertionError("Unexpected Yahoo request")):
            second = data.load_and_prepare()
        pd.testing.assert_frame_equal(first.levels, second.levels)
        pd.testing.assert_frame_equal(first.factors, second.factors)
        after = self.manifest()
        for source, path in (("treasury", config.TREASURY_SNAPSHOT_PATH), ("etf", config.ETF_SNAPSHOT_PATH)):
            self.assertEqual(after["sources"][source]["retrieval"], before["sources"][source]["retrieval"])
            self.assertEqual(after["sources"][source]["snapshot"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(after["sources"][source]["last_run_reused_snapshot"])
        self.assertEqual(modified_times, [path.stat().st_mtime_ns for path in (config.TREASURY_SNAPSHOT_PATH, config.ETF_SNAPSHOT_PATH)])
        self.assertTrue(after["sources"]["treasury"]["redistribution"]["permitted"])
        self.assertFalse(after["sources"]["etf"]["redistribution"]["permitted"])

    def test_hash_mismatch_remains_an_error_on_subsequent_runs(self):
        self.fetch()
        original = self.manifest()
        with config.TREASURY_SNAPSHOT_PATH.open("ab") as stream:
            stream.write(b"\n")
        for _ in range(2):
            with self.assertRaisesRegex(data.DataError, "SHA-256"):
                data.load_and_prepare()
            current = self.manifest()
            self.assertEqual(current["sources"]["treasury"]["snapshot"], original["sources"]["treasury"]["snapshot"])
            self.assertEqual(current["sources"]["etf"]["retrieval"], original["sources"]["etf"]["retrieval"])

    def test_inadequate_coverage_is_saved_before_failure(self):
        self.treasury, self.etfs = input_frames()
        with self.assertRaisesRegex(data.InsufficientCoverageError, "4 aligned changes"):
            self.fetch()
        manifest = self.manifest()
        self.assertEqual(manifest["status"], "insufficient_coverage")
        self.assertEqual(manifest["alignment"]["retained_level_count"], 5)
        self.assertEqual(manifest["alignment"]["retained_change_count"], 4)
        self.assertFalse(manifest["alignment"]["coverage_sufficient"])
        self.assertEqual(manifest["requested_period"], {"start_inclusive": "2022-09-01", "end_inclusive": "2026-08-31"})

    def test_source_failure_does_not_create_fabricated_prices(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError("Source unavailable: 503")
        with patch.object(data.requests, "get", return_value=response), self.assertRaisesRegex(data.DataError, "503"):
            data.load_and_prepare()
        self.assertFalse(config.TREASURY_SNAPSHOT_PATH.exists())
        self.assertFalse(config.ETF_SNAPSHOT_PATH.exists())
        self.assertEqual(self.manifest()["status"], "error")

    def test_missing_etf_field_is_not_replaced_with_close(self):
        config.TREASURY_SNAPSHOT_PATH.parent.mkdir(parents=True)
        self.treasury.to_csv(config.TREASURY_SNAPSHOT_PATH)
        wrong = pd.DataFrame({("Close", "LQD"): [100.0], ("Close", "SPY"): [400.0]}, index=pd.to_datetime(["2024-01-02"]))
        with patch.object(data.requests, "get", side_effect=AssertionError("Treasury should be reused")), patch.object(data.yf, "set_tz_cache_location"), patch.object(data.yf, "download", return_value=wrong), self.assertRaisesRegex(data.DataError, "both Close and Adj Close"):
            data.load_and_prepare()
        self.assertFalse(config.ETF_SNAPSHOT_PATH.exists())

    def test_preexisting_snapshots_do_not_get_invented_retrieval_dates(self):
        config.TREASURY_SNAPSHOT_PATH.parent.mkdir(parents=True)
        self.treasury.to_csv(config.TREASURY_SNAPSHOT_PATH)
        self.etfs.to_csv(config.ETF_SNAPSHOT_PATH)
        with patch.object(data.requests, "get", side_effect=AssertionError("Unexpected network request")), patch.object(data.yf, "download", side_effect=AssertionError("Unexpected Yahoo request")):
            data.load_and_prepare()
        self.assertIsNone(self.manifest()["sources"]["treasury"]["retrieval"]["retrieved_at_utc"])
        self.assertIsNone(self.manifest()["sources"]["etf"]["retrieval"]["retrieved_at_utc"])


class PricingTests(unittest.TestCase):
    def setUp(self):
        treasury, etfs = input_frames()
        self.levels = data.prepare_data(treasury, etfs).levels
        self.holdings = pricing.initialize_holdings(self.levels, date(2024, 1, 2))

    def test_linear_interpolation_on_knots_and_between_them(self):
        curve = np.array([0.01, 0.03, 0.04, 0.05, 0.06, 0.10])
        np.testing.assert_allclose(pricing.interpolate_yield(curve, [1.5, 3.0, 9.5]), [0.02, 0.035, 0.08])
        np.testing.assert_array_equal(pricing.interpolate_yield(curve, config.YIELD_TENORS_YEARS), curve)
        batched = pricing.interpolate_yield(np.stack([curve, curve + 0.02]), [1.5, 9.5])
        np.testing.assert_allclose(batched, [[0.02, 0.08], [0.04, 0.10]])

    def test_no_yield_extrapolation_or_invalid_curves(self):
        for maturity in (0.5, 10.1, np.nan):
            with self.subTest(maturity=maturity), self.assertRaises(ValueError):
                pricing.interpolate_yield(np.full(6, 0.04), maturity)
        with self.assertRaises(ValueError):
            pricing.interpolate_yield([0.04] * 5, 2)

    def test_bond_discounting_redemption_and_negative_yields(self):
        self.assertAlmostEqual(float(pricing.bond_price(100, 0.04, 5)), 100 * math.exp(-0.2))
        self.assertAlmostEqual(float(pricing.bond_price(100, -0.02, 5)), 100 * math.exp(0.1))
        self.assertEqual(float(pricing.bond_price(100, 0.04, 0)), 100)
        np.testing.assert_allclose(pricing.bond_price([100, 200], 0, [2, 5]), [100, 200])

    def test_aging_uses_trading_steps_and_reaches_required_terminal_maturities(self):
        np.testing.assert_array_equal(pricing.age_maturities([2, 5, 10], 252), [1, 4, 9])
        np.testing.assert_allclose(pricing.age_maturities([2, 5, 10], 1), np.array([2, 5, 10]) - 1 / 252)
        for steps in (-1, 0.5, 3000):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                pricing.age_maturities([2, 5, 10], steps)

    def test_etf_marks_normalize_to_inception_close(self):
        np.testing.assert_allclose(pricing.etf_total_return_mark([100, 400], [50, 200], [50, 200]), [100, 400])
        np.testing.assert_allclose(pricing.etf_total_return_mark([100, 400], [50, 200], [55, 180]), [110, 360])
        with self.assertRaises(ValueError):
            pricing.etf_total_return_mark(100, 0, 50)

    def test_fixed_initial_quantities_match_allocations(self):
        values = pricing.value_holdings(self.holdings, np.full(6, 0.04), [50, 200], 0)
        np.testing.assert_allclose(values, [100_000, 100_000, 100_000, 300_000, 400_000], atol=1e-9)
        np.testing.assert_allclose(self.holdings.quantities[-2:], [3000, 1000])
        self.assertAlmostEqual(sum(values), 1_000_000)

    def test_full_repricing_ages_bonds_and_keeps_positions_fixed(self):
        quantities = self.holdings.quantities
        values = pricing.value_holdings(self.holdings, np.full(6, 0.05), [100, 100], 1)
        expected_bonds = [quantities[i] * math.exp(-0.05 * (maturity - 1 / 252)) for i, maturity in enumerate((2, 5, 10))]
        np.testing.assert_allclose(values, [*expected_bonds, 600_000, 200_000])
        self.assertEqual(self.holdings.quantities, quantities)

    def test_aging_reinterpolates_a_nonflat_curve_before_valuation(self):
        curve = [0.01, 0.03, 0.04, 0.05, 0.06, 0.10]
        # Hand calculations use the adjacent tenor slopes after one step and
        # the 1-, 4-, and 9-year knots after 252 steps. Faces follow from the
        # initial USD 100,000 allocations discounted at the flat 4% curve.
        cases = (
            (1, (2 - 1 / 252, 5 - 1 / 252, 10 - 1 / 252),
             (0.03 - 0.02 / 252, 0.05 - 0.01 / 252, 0.10 - 0.04 / 252)),
            (252, (1, 4, 9), (0.01, 0.04, 0.06)),
        )
        for steps, maturities, yields in cases:
            with self.subTest(steps=steps):
                expected_bonds = [
                    100_000 * math.exp(0.04 * initial - rate * remaining)
                    for initial, rate, remaining in zip((2, 5, 10), yields, maturities)
                ]
                expected = [*expected_bonds, 330_000, 360_000]
                values = pricing.value_holdings(self.holdings, curve, [55, 180], steps)
                np.testing.assert_allclose(values, expected, rtol=1e-12)
                self.assertAlmostEqual(float(values.sum()), math.fsum(expected), places=8)

    def test_batch_scenarios_aggregate_the_same_component_positions(self):
        curves = np.array([[0.04] * 6, [0.05] * 6])
        adjusted = np.array([[50, 200], [100, 100]])
        batch = pricing.value_holdings(self.holdings, curves, adjusted, 252)
        self.assertEqual(batch.shape, (2, 5))
        for i in range(2):
            expected = [self.holdings.quantities[j] * math.exp(-curves[i, 0] * maturity) for j, maturity in enumerate((1, 4, 9))]
            expected.extend([300_000 * adjusted[i, 0] / 50, 400_000 * adjusted[i, 1] / 200])
            np.testing.assert_allclose(batch[i], expected)
            self.assertAlmostEqual(float(batch[i].sum()), sum(expected), places=8)

    def test_separate_forward_and_retrospective_inceptions(self):
        treasury, etfs = input_frames(pd.bdate_range("2022-09-01", periods=751))
        levels = data.prepare_data(treasury, etfs).levels
        portfolios = pricing.initialize_portfolios(levels)
        self.assertEqual(portfolios["forward"].inception_date, levels.index[750].date())
        self.assertEqual(portfolios["retrospective"].inception_date, levels.index[500].date())
        self.assertNotEqual(portfolios["forward"].quantities, portfolios["retrospective"].quantities)
        for holdings in portfolios.values():
            row = levels.loc[pd.Timestamp(holdings.inception_date)]
            values = pricing.value_holdings(holdings, row[list(config.TREASURY_SERIES.values())], row[["LQD_Adj Close", "SPY_Adj Close"]], 0)
            np.testing.assert_allclose(values, [100_000, 100_000, 100_000, 300_000, 400_000])

    def test_short_history_cannot_select_a_retrospective_origin(self):
        with self.assertRaisesRegex(ValueError, "751"):
            pricing.initialize_portfolios(self.levels)


if __name__ == "__main__":
    unittest.main()
