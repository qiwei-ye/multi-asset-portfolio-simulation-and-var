"""Consolidated correctness tests; fixtures are synthetic, not retained inputs."""

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

from portfolio_var import config, data, pricing, reporting, simulation


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


class SimulationTests(unittest.TestCase):
    def setUp(self):
        paths = patch.object(config, "SIMULATION_PATHS", 3)
        paths.start()
        self.addCleanup(paths.stop)
        treasury, etfs = input_frames(pd.bdate_range("2022-09-01", periods=751))
        self.prepared = data.prepare_data(treasury, etfs)
        self.holdings = pricing.initialize_portfolios(self.prepared.levels)["forward"]

    def run_simulation(self, factors=None):
        return simulation.simulate_forward(
            self.prepared.levels,
            self.prepared.factors if factors is None else factors,
            self.holdings,
        )

    def test_gaussian_estimates_use_sample_mean_and_n_minus_one_covariance(self):
        values = np.arange(8)[None, :] + np.array([0, 1, 3])[:, None]
        window = pd.DataFrame(values, columns=config.FACTOR_ORDER)
        mean, covariance = simulation.fit_gaussian(window)
        np.testing.assert_allclose(mean, np.arange(8) + 4 / 3)
        np.testing.assert_allclose(covariance, np.full((8, 8), 7 / 3))

    def test_gaussian_fit_rejects_wrong_factor_order_and_nonfinite_shocks(self):
        window = self.prepared.factors.iloc[-500:].copy()
        with self.assertRaisesRegex(ValueError, "ordering"):
            simulation.fit_gaussian(window.iloc[:, ::-1])
        window.iloc[0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "finite"):
            simulation.fit_gaussian(window)

    def test_daily_joint_shocks_accumulate_before_full_terminal_repricing(self):
        daily = np.zeros((3, 252, 8))
        first = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, math.log(1.1), math.log(0.9)])
        last = np.array([-0.0001, -0.0002, -0.0003, -0.0004, -0.0005, -0.0006, math.log(1.2), math.log(1.05)])
        daily[0, 0], daily[0, -1] = first, last
        daily[1, 15], daily[2, 125] = -first, last
        rng = Mock()
        rng.multivariate_normal.return_value = daily
        with patch.object(config, "simulation_rng", return_value=rng):
            result = self.run_simulation()
        expected_shocks = np.stack([first + last, -first, last])
        np.testing.assert_allclose(result.cumulative_shocks, expected_shocks, atol=1e-16)
        self.assertEqual(rng.multivariate_normal.call_args.kwargs["size"], (3, 252))
        self.assertEqual(result.steps, 252)
        origin = self.prepared.levels.iloc[-1]
        curve = origin[list(config.TREASURY_SERIES.values())].to_numpy()
        expected_prices = []
        for shocks in expected_shocks:
            bonds = [math.exp(-(curve[i] + shocks[i]) * maturity) for i, maturity in zip((0, 2, 4), (1, 4, 9))]
            etfs = [origin[f"{ticker}_Close"] * math.exp(shocks[6 + i]) for i, ticker in enumerate(("LQD", "SPY"))]
            expected_prices.append([*bonds, *etfs])
        expected_values = np.array(expected_prices) * np.asarray(self.holdings.quantities)
        np.testing.assert_allclose(result.terminal_prices, expected_prices, rtol=1e-12)
        np.testing.assert_allclose(result.position_values, expected_values, rtol=1e-12)
        np.testing.assert_allclose(result.position_returns, expected_values / [100_000, 100_000, 100_000, 300_000, 400_000] - 1, atol=1e-14)
        np.testing.assert_array_equal(result.portfolio_values, result.position_values.sum(axis=1))
        np.testing.assert_allclose(result.portfolio_returns, expected_values.sum(axis=1) / 1_000_000 - 1, atol=1e-14)

    def test_calibration_uses_only_the_latest_500_aligned_changes(self):
        original = self.run_simulation()
        changed = self.prepared.factors.copy()
        changed.iloc[:-500] = 1000
        repeated = self.run_simulation(changed)
        self.assertTrue(original.calibration_dates.equals(self.prepared.factors.index[-500:]))
        np.testing.assert_array_equal(original.daily_mean, repeated.daily_mean)
        np.testing.assert_array_equal(original.daily_covariance, repeated.daily_covariance)
        np.testing.assert_array_equal(original.position_values, repeated.position_values)

    def test_simulation_is_repeatable_independently_of_forecast_execution(self):
        first = self.run_simulation()
        for day in (date(2026, 8, 31), date(2024, 2, 29), date(2022, 9, 1)):
            config.parametric_var_rng(day).standard_normal((100, 8))
        second = self.run_simulation()
        np.testing.assert_array_equal(first.cumulative_shocks, second.cumulative_shocks)
        np.testing.assert_array_equal(first.terminal_prices, second.terminal_prices)
        np.testing.assert_array_equal(first.portfolio_values, second.portfolio_values)

    def test_forward_simulation_rejects_a_retrospective_inception(self):
        retrospective = pricing.initialize_portfolios(self.prepared.levels)["retrospective"]
        with self.assertRaisesRegex(ValueError, "latest aligned"):
            simulation.simulate_forward(self.prepared.levels, self.prepared.factors, retrospective)

    def test_forward_calibration_requires_500_matched_changes(self):
        with self.assertRaisesRegex(ValueError, "500 aligned changes"):
            simulation.simulate_forward(self.prepared.levels.iloc[-500:], self.prepared.factors.iloc[-499:], self.holdings)
        with self.assertRaisesRegex(ValueError, "Factor dates"):
            self.run_simulation(self.prepared.factors.iloc[:-1])

    def test_saved_standalone_outcomes_and_totals_share_identical_scenarios(self):
        result = self.run_simulation()
        table = reporting.terminal_outcomes(result)
        self.assertEqual(table.index.tolist(), [1, 2, 3])
        for i, asset in enumerate(config.ASSETS):
            np.testing.assert_array_equal(table[f"{asset}_terminal_price"], result.terminal_prices[:, i])
            np.testing.assert_array_equal(table[f"{asset}_terminal_value_usd"], result.position_values[:, i])
            np.testing.assert_array_equal(table[f"{asset}_return"], result.position_returns[:, i])
        component_columns = [f"{asset}_terminal_value_usd" for asset in config.ASSETS]
        np.testing.assert_array_equal(table[component_columns].sum(axis=1), table["portfolio_terminal_value_usd"])
        np.testing.assert_allclose(table["portfolio_return"], table["portfolio_terminal_value_usd"] / 1_000_000 - 1, atol=1e-14)
        with tempfile.TemporaryDirectory(prefix="portfolio-report-test-", dir=config.PROJECT_ROOT / ".venv") as temporary:
            path = Path(temporary) / "terminal.csv"
            with patch.object(config, "TERMINAL_OUTCOMES_PATH", path), patch.object(reporting, "plot_terminal_distributions"):
                reporting.write_terminal_reports(result)
            reloaded = pd.read_csv(path, index_col="path_id")
            np.testing.assert_array_equal(reloaded.to_numpy(), table.to_numpy())


class RandomnessTests(unittest.TestCase):
    def test_forecast_streams_ignore_order_subsets_and_simulation_consumption(self):
        dates = (date(2022, 9, 1), date(2024, 2, 29), date(2026, 8, 31))
        expected = {day: config.parametric_var_rng(day).standard_normal(32) for day in dates}
        for day in reversed(dates):
            config.simulation_rng().standard_normal(200)
            np.testing.assert_array_equal(expected[day], config.parametric_var_rng(day).standard_normal(32))
        self.assertFalse(np.array_equal(expected[dates[0]], expected[dates[1]]))

    def test_frozen_config_uses_effective_settings_and_reconstructs_every_named_seed(self):
        with tempfile.TemporaryDirectory(prefix="portfolio-config-test-", dir=config.PROJECT_ROOT / ".venv") as temporary:
            path = Path(temporary) / "frozen_config.json"
            with patch.object(config, "FROZEN_CONFIG_PATH", path), patch.object(config, "SIMULATION_PATHS", 11):
                reporting.write_frozen_config()
                contents = path.read_bytes()
                path.write_text('{"settings": {"SIMULATION_PATHS": -1}}', encoding="utf-8")
                reporting.write_frozen_config()
                self.assertEqual(path.read_bytes(), contents)
            frozen = json.loads(contents)
        self.assertEqual(frozen["authority"], "portfolio_var/config.py")
        self.assertEqual(frozen["settings"]["SIMULATION_PATHS"], 11)
        self.assertEqual(frozen["settings"]["DATA_START_DATE"], "2022-09-01")
        self.assertEqual(frozen["settings"]["PROJECT_ROOT"], ".")
        self.assertEqual(frozen["settings"]["TREASURY_SNAPSHOT_PATH"], "data/treasury_yields.csv")
        plan = frozen["randomness"]
        self.assertEqual(plan["master_seed"], 42)
        self.assertEqual(plan["branches"], {"forward_simulation": [0], "parametric_var": [1]})
        seed = np.random.SeedSequence(plan["master_seed"], spawn_key=tuple(plan["branches"]["forward_simulation"]), pool_size=plan["pool_size"])
        restored = np.random.Generator(np.random.PCG64(seed))
        np.testing.assert_array_equal(restored.standard_normal(32), config.simulation_rng().standard_normal(32))
        assignments = plan["parametric_var_by_forecast_date"]
        self.assertEqual(len(assignments), 1461)
        self.assertEqual(sorted(assignments), frozen["settings"]["FORECAST_SEED_DATES"])
        for day, spawn_key in assignments.items():
            seed = np.random.SeedSequence(plan["master_seed"], spawn_key=tuple(spawn_key), pool_size=plan["pool_size"])
            restored = np.random.Generator(np.random.PCG64(seed))
            np.testing.assert_array_equal(restored.standard_normal(8), config.parametric_var_rng(date.fromisoformat(day)).standard_normal(8))


if __name__ == "__main__":
    unittest.main()
