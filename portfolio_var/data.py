"""Retain source snapshots, align complete levels, and construct joint shocks."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from portfolio_var import config


class DataError(ValueError):
    """A source or retained input cannot support the requested preparation."""


class InsufficientCoverageError(DataError):
    """The unchanged requested period has too few complete aligned changes."""


@dataclass(frozen=True)
class PreparedData:
    """Levels contain decimal zero yields and both original ETF price fields."""

    levels: pd.DataFrame
    factors: pd.DataFrame
    alignment: dict


def _source_details() -> dict:
    return {
        "treasury": {
            "provider": config.TREASURY_DATA_SOURCE,
            "source_page": config.TREASURY_SOURCE_URL,
            "query": {"url": config.TREASURY_DATA_URL, "method": "GET"},
            "series": list(config.TREASURY_SERIES.values()),
            "snapshot_format": "Exact HTTP response bytes, including the source preamble and full history.",
            "quoted_units": "Percent per year; SVENY zero yields are continuously compounded.",
            "access": "Public CSV; no account or API key. Historical estimates may be revised.",
            "redistribution": {
                "permitted": True,
                "policy_url": config.TREASURY_RIGHTS_URL,
                "reviewed_on": config.SOURCE_PERMISSIONS_REVIEW_DATE.isoformat(),
                "basis": "Board website information is public domain unless otherwise indicated; cite the Board as source. No contrary notice is present for this dataset.",
                "attribution": "Board of Governors of the Federal Reserve System; Gurkaynak, Sack and Wright, The U.S. Treasury Yield Curve: 1961 to the Present.",
            },
        },
        "etf": {
            "provider": config.ETF_DATA_SOURCE,
            "source_pages": [f"https://finance.yahoo.com/quote/{ticker}/history/" for ticker in config.ETF_TICKERS],
            "query": {
                "library": "yfinance.download",
                "tickers": list(config.ETF_TICKERS),
                "start": config.DATA_START_DATE.isoformat(),
                "end_exclusive": (config.DATA_END_DATE + timedelta(days=1)).isoformat(),
                "interval": "1d",
                "auto_adjust": config.YFINANCE_AUTO_ADJUST,
                "back_adjust": False,
                "repair": False,
                "keepna": True,
                "rounding": False,
                "actions": False,
                "prepost": False,
                "ignore_tz": True,
                "threads": False,
                "group_by": "column",
                "multi_level_index": True,
                "timeout": config.DOWNLOAD_TIMEOUT_SECONDS,
            },
            "fields": list(config.ETF_PRICE_FIELDS),
            "snapshot_format": "CSV of returned Close and Adj Close doubles, serialized with 17 significant digits and reloaded with round-trip float parsing; no adjustment, filling, or repair.",
            "quoted_units": "USD per unit; Yahoo's adjusted-close convention supplies total-return ratios.",
            "access": "Retrieved locally with yfinance using public Yahoo endpoints for personal research; no official service guarantee. Access can be rate-limited or withdrawn.",
            "documentation_url": config.YFINANCE_DOCUMENTATION_URL,
            "redistribution": {
                "permitted": False,
                "policy_url": config.ETF_RIGHTS_URL,
                "reviewed_on": config.SOURCE_PERMISSIONS_REVIEW_DATE.isoformat(),
                "basis": "Yahoo's exchanges and data providers page explicitly prohibits redistribution. The ETF snapshot stays ignored; obtain it independently under applicable Yahoo terms.",
                "attribution": "Yahoo Finance and its underlying data providers; yfinance is an independent client, not a data licence.",
            },
        },
    }


def _validate_levels(frame: pd.DataFrame, columns: tuple, source: str) -> pd.DataFrame:
    missing = set(columns) - set(frame.columns)
    if missing or not frame.columns.is_unique:
        raise DataError(f"{source}: missing required columns {sorted(missing)} or duplicate columns.")
    result = frame.loc[:, list(columns)].copy()
    try:
        result.index = pd.to_datetime(result.index, format="%Y-%m-%d", errors="raise")
        result = result.apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise DataError(f"{source}: invalid dates or nonnumeric observations: {exc}") from exc
    if result.index.hasnans or result.index.has_duplicates:
        raise DataError(f"{source}: dates must be nonmissing and unique.")
    if result.index.tz is not None or not result.index.equals(result.index.normalize()):
        raise DataError(f"{source}: dates must be timezone-free daily observations.")
    if np.isinf(result.to_numpy()).any():
        raise DataError(f"{source}: infinite observations are invalid.")
    if source == "ETF" and (result <= 0).any().any():
        raise DataError("ETF: observed Close and Adj Close values must be positive.")
    return result.sort_index().rename_axis("Date")


def load_snapshot(path: Path, source: str) -> pd.DataFrame:
    """Load retained input without changing it; the Treasury CSV has a preamble."""
    columns = tuple(config.TREASURY_SERIES.values()) if source == "treasury" else config.ETF_LEVEL_COLUMNS
    raw = path.read_bytes()
    header = next((i for i, line in enumerate(raw.splitlines()) if line.startswith(b"Date,")), None)
    if header is None:
        raise DataError(f"{path.name}: expected a CSV header starting with Date,.")
    try:
        frame = pd.read_csv(
            BytesIO(raw), skiprows=header, usecols=["Date", *columns],
            na_values=["NA", "."],
        ).set_index("Date")
    except (ValueError, pd.errors.ParserError) as exc:
        raise DataError(f"{path.name}: invalid snapshot schema: {exc}") from exc
    return _validate_levels(frame, columns, "Treasury" if source == "treasury" else "ETF")


def prepare_data(treasury: pd.DataFrame, etfs: pd.DataFrame) -> PreparedData:
    """Align all ten required level fields BEFORE differencing; never fill gaps."""
    yield_columns = tuple(config.TREASURY_SERIES.values())
    treasury = _validate_levels(treasury, yield_columns, "Treasury")
    etfs = _validate_levels(etfs, config.ETF_LEVEL_COLUMNS, "ETF")
    start, end = pd.Timestamp(config.DATA_START_DATE), pd.Timestamp(config.DATA_END_DATE)
    treasury = treasury.loc[start:end]
    etfs = etfs.loc[start:end]
    union = treasury.join(etfs, how="outer").sort_index()
    levels = union.dropna(how="any").copy()
    levels.loc[:, list(yield_columns)] *= config.YIELD_PERCENT_TO_DECIMAL
    yield_changes = levels.loc[:, list(yield_columns)].diff().iloc[1:]
    adjusted_columns = [f"{ticker}_Adj Close" for ticker in config.ETF_TICKERS]
    adjusted = levels.loc[:, adjusted_columns]
    with np.errstate(divide="raise", over="raise", invalid="raise"):
        try:
            etf_returns = np.log(adjusted / adjusted.shift(1)).iloc[1:]
        except FloatingPointError as exc:
            raise DataError("ETF ratios could not produce finite log total returns.") from exc
    factors = pd.concat([yield_changes, etf_returns], axis=1)
    factors.columns = config.FACTOR_ORDER
    if not np.isfinite(factors.to_numpy()).all():
        raise DataError("Constructed factors contain nonfinite observations.")
    alignment = {
        "requested_start_inclusive": config.DATA_START_DATE.isoformat(),
        "requested_end_inclusive": config.DATA_END_DATE.isoformat(),
        "treasury_rows_in_period": len(treasury),
        "etf_rows_in_period": len(etfs),
        "union_level_count": len(union),
        "retained_level_count": len(levels),
        "retained_change_count": len(factors),
        "minimum_level_count": config.MIN_ALIGNED_LEVELS,
        "minimum_change_count": config.MIN_ALIGNED_CHANGES,
        "coverage_sufficient": len(factors) >= config.MIN_ALIGNED_CHANGES,
        "retained_level_dates": levels.index.strftime("%Y-%m-%d").tolist(),
        "retained_change_dates": factors.index.strftime("%Y-%m-%d").tolist(),
        "dropped_union_dates": union.index.difference(levels.index).strftime("%Y-%m-%d").tolist(),
        "missing_by_field": {column: int(count) for column, count in union.isna().sum().items()},
        "factor_order": list(config.FACTOR_ORDER),
        "factor_units": ["absolute decimal zero-yield change"] * len(yield_columns) + ["log total return"] * len(config.ETF_TICKERS),
        "missing_value_policy": "Complete-date alignment across all required fields; no filling.",
        "trading_clock": "Each consecutive retained observation is one step of 1/252 year, including gaps spanning multiple market days.",
    }
    return PreparedData(levels, factors, alignment)


def _retrieve_treasury(path: Path) -> dict:
    response = requests.get(config.TREASURY_DATA_URL, timeout=config.DOWNLOAD_TIMEOUT_SECONDS)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "response_url": response.url,
        "http_status": response.status_code,
        "last_modified": response.headers.get("Last-Modified"),
        "etag": response.headers.get("ETag"),
    }


def _retrieve_etfs(path: Path, query: dict) -> dict:
    yf.set_tz_cache_location(str(config.YFINANCE_CACHE_DIR))
    arguments = {key: value for key, value in query.items() if key not in ("library", "end_exclusive")}
    arguments["end"] = query["end_exclusive"]
    returned = yf.download(**arguments, progress=False)
    if returned is None or returned.empty:
        raise DataError("Yahoo Finance returned no ETF observations. Check access/rate limits or supply the retained snapshot; the period and source have not been changed.")
    selected = [(field, ticker) for ticker in config.ETF_TICKERS for field in config.ETF_PRICE_FIELDS]
    if not isinstance(returned.columns, pd.MultiIndex) or any(column not in returned.columns for column in selected):
        raise DataError("Yahoo Finance did not return both Close and Adj Close for every requested ETF.")
    frame = returned.loc[:, selected].copy()
    frame.columns = config.ETF_LEVEL_COLUMNS
    frame = _validate_levels(frame, config.ETF_LEVEL_COLUMNS, "ETF")
    for ticker in config.ETF_TICKERS:
        if frame[[f"{ticker}_{field}" for field in config.ETF_PRICE_FIELDS]].dropna().empty:
            raise DataError(f"Yahoo Finance returned no complete observations for {ticker}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, date_format="%Y-%m-%d", float_format="%.17g", lineterminator="\n")
    return {"retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "client_version": version("yfinance")}


def _write_manifest(manifest: dict) -> None:
    config.RETRIEVAL_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.RETRIEVAL_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


def load_and_prepare() -> PreparedData:
    """Reuse retained bytes; acquire only absent snapshots and record provenance.

    A changed snapshot hash fails visibly. To request a new source vintage,
    delete that snapshot explicitly; its next retrieval updates the manifest.
    """
    previous = {}
    if config.RETRIEVAL_MANIFEST_PATH.exists():
        try:
            previous = json.loads(config.RETRIEVAL_MANIFEST_PATH.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise DataError(f"Cannot read the retained retrieval manifest: {exc}") from exc
        if (
            not isinstance(previous, dict)
            or not isinstance(previous.get("sources"), dict)
            or not all(isinstance(entry, dict) for entry in previous["sources"].values())
        ):
            raise DataError("The retained retrieval manifest has an invalid structure.")
    manifest = {
        "schema_version": 1,
        "last_run_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preparing",
        "requested_period": {"start_inclusive": config.DATA_START_DATE.isoformat(), "end_inclusive": config.DATA_END_DATE.isoformat()},
        "python_version": platform.python_version(),
        "dependency_versions": {name: version(name) for name in ("pandas", "numpy", "scipy", "requests", "yfinance", "matplotlib")},
        "reproduction": "Ordinary runs reuse snapshots without source requests. Delete a snapshot explicitly to retrieve a new vintage. Yahoo ETF inputs must be acquired independently and remain private.",
        "sources": {name: dict(entry) for name, entry in previous.get("sources", {}).items()},
    }
    frames = {}
    details = _source_details()
    try:
        for source, path in (("treasury", config.TREASURY_SNAPSHOT_PATH), ("etf", config.ETF_SNAPSHOT_PATH)):
            old = previous.get("sources", {}).get(source, {})
            reused = path.exists()
            if reused and old.get("query") and old["query"] != details[source]["query"]:
                raise DataError(f"{path.name}: retained query differs from the requested configuration; explicitly remove the snapshot to reacquire it.")
            entry = {**old, **details[source]}
            manifest["sources"][source] = entry
            entry["availability"] = {"status": "loading_snapshot" if reused else "retrieving"}
            if reused:
                entry["retrieval"] = old.get("retrieval", {"retrieved_at_utc": None, "note": "Preexisting snapshot; original retrieval provenance is unavailable."})
            elif source == "treasury":
                entry["retrieval"] = _retrieve_treasury(path)
            else:
                entry["retrieval"] = _retrieve_etfs(path, entry["query"])
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if reused and old.get("snapshot", {}).get("sha256") not in (None, digest):
                raise DataError(f"{path.name}: snapshot SHA-256 differs from its recorded provenance; restore the retained file or explicitly remove it to reacquire.")
            entry["snapshot"] = {"path": f"data/{path.name}", "sha256": digest, "bytes": path.stat().st_size}
            entry["last_run_reused_snapshot"] = reused
            frames[source] = load_snapshot(path, source)
            entry["availability"] = {
                "status": "retained_snapshot_available" if reused else "retrieved_successfully",
                "source_rows": len(frames[source]),
                "first_source_date": frames[source].index.min().strftime("%Y-%m-%d") if len(frames[source]) else None,
                "last_source_date": frames[source].index.max().strftime("%Y-%m-%d") if len(frames[source]) else None,
                "online_checked_at_utc": entry["retrieval"].get("retrieved_at_utc"),
            }
            _write_manifest(manifest)
        prepared = prepare_data(frames["treasury"], frames["etf"])
        manifest["alignment"] = prepared.alignment
        if not prepared.alignment["coverage_sufficient"]:
            raise InsufficientCoverageError(
                f"Requested {config.DATA_START_DATE} through {config.DATA_END_DATE}: retained "
                f"{len(prepared.levels)} complete levels and {len(prepared.factors)} aligned changes; "
                f"require at least {config.MIN_ALIGNED_LEVELS} levels and {config.MIN_ALIGNED_CHANGES} changes. "
                "No dates were filled and the requested period was not changed. See results/retrieval_manifest.json."
            )
    except (DataError, requests.RequestException, OSError) as exc:
        manifest["status"] = "insufficient_coverage" if isinstance(exc, InsufficientCoverageError) else "error"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_manifest(manifest)
        if isinstance(exc, DataError):
            raise
        raise DataError(f"Input retrieval/loading failed: {exc}") from exc
    manifest["status"] = "ready"
    _write_manifest(manifest)
    return prepared
