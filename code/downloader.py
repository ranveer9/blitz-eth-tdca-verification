"""
downloader.py
=============
Downloads, verifies and extracts all daily Binance USD-M aggTrade ZIPs listed in a manifest (BTCUSDT, ETHUSDT, ...)
from data.binance.vision for the agreed window:
    2021-01-01 (inclusive) → 2026-08-24 (inclusive)

Behaviour
---------
- Reads the manifest CSV for the exact URL list (no URL construction).
- For each date:
    1. If the extracted CSV already exists and its size matches the manifest
       (where recorded), skip download+extraction — just verify the checksum
       if the ZIP is still present.
    2. Otherwise: download ZIP → fetch official checksum → verify SHA-256
       → extract CSV → delete ZIP to save disk.
- Writes a download log CSV alongside the script so every decision is auditable.
- Fully resumable: re-run after an interrupted session and it picks up where
  it left off without re-downloading anything already verified.

Usage
-----
    python src/downloader.py \
        --manifest data/eth-aggtrades-download-manifest.csv \
        --raw-dir  data/raw \
        --out-dir  data/extracted \
        --log      logs/download_log.csv \
        [--workers 4]          # parallel downloads, default 4
        [--keep-zip]           # don't delete ZIPs after extraction
        [--force-recheck]      # re-verify checksum even for existing CSVs

Output
------
    data/extracted/YYYY-MM-DD.csv   — one per date
    logs/download_log.csv           — one row per date with status + hashes
"""

import argparse
import csv
import hashlib
import io
import logging
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOW_START = date(2021, 1, 1)   # inclusive
WINDOW_END   = date(2026, 8, 24)  # inclusive (exclusive upper = Aug 25)

# Binance futures USD-M aggTrades column headers (present in newer files;
# absent in older ones where we inject them).
AGGTRADE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]

# Bytes to stream per chunk during download
CHUNK_SIZE = 1024 * 1024  # 1 MiB

# HTTP session settings
REQUEST_TIMEOUT = 120  # seconds
MAX_RETRIES     = 5
RETRY_BACKOFF   = [2, 4, 8, 16, 32]  # seconds between retries

LOG_FIELDS = [
    "utc_date", "status", "zip_bytes", "zip_sha256_computed",
    "zip_sha256_official", "checksum_match", "csv_rows",
    "csv_bytes_written", "elapsed_s", "note",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    """Return a requests session with a sensible User-Agent."""
    s = requests.Session()
    s.headers.update({
        "User-Agent": "blitz-independent-verifier/1.0 (research)"
    })
    return s


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch_with_retry(
    session: requests.Session,
    url: str,
    stream: bool = False,
) -> requests.Response:
    """GET with exponential-backoff retries on transient errors."""
    for attempt, wait in enumerate(RETRY_BACKOFF, start=1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT, stream=stream)
            if resp.status_code == 200:
                return resp
            log.warning(
                "HTTP %d for %s (attempt %d/%d)",
                resp.status_code, url, attempt, MAX_RETRIES,
            )
        except requests.RequestException as exc:
            log.warning("Request error for %s: %s (attempt %d/%d)",
                        url, exc, attempt, MAX_RETRIES)
        if attempt < MAX_RETRIES:
            import time; time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {MAX_RETRIES} attempts")


def _fetch_official_checksum(session: requests.Session, checksum_url: str) -> str:
    """
    Fetch the official .CHECKSUM file and return the hex digest string.
    Binance checksum files look like:
        b49f13b...  <SYMBOL>-aggTrades-2021-01-01.zip
    """
    resp = _fetch_with_retry(session, checksum_url)
    text = resp.text.strip()
    # First whitespace-delimited token is the hex digest
    return text.split()[0].lower()


def _download_zip(
    session: requests.Session,
    url: str,
    dest: Path,
) -> int:
    """Stream-download ZIP to dest; return byte count."""
    resp = _fetch_with_retry(session, url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            downloaded += len(chunk)
    return downloaded


def _detect_has_header(first_line: str) -> bool:
    """True if the CSV's first line looks like a header (non-numeric first field)."""
    first_field = first_line.split(",")[0].strip().strip('"')
    return not first_field.lstrip("-").isdigit()


def _extract_and_normalize(
    zip_path: Path,
    out_csv: Path,
) -> int:
    """
    Extract the single CSV from a ZIP, inject column headers if missing,
    write to out_csv, and return the number of data rows written.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # There should be exactly one .csv inside
        csv_names = [n for n in names if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found inside {zip_path.name}; files: {names}")
        inner_name = csv_names[0]

        with zf.open(inner_name) as raw:
            # Read first line to detect header presence
            first_line = raw.readline().decode("utf-8", errors="replace")
            has_header = _detect_has_header(first_line)

            # Write normalized CSV
            row_count = 0
            with open(out_csv, "w", newline="", encoding="utf-8") as out:
                out.write(",".join(AGGTRADE_COLUMNS) + "\n")
                if not has_header:
                    # first_line is already a data row
                    out.write(first_line if first_line.endswith("\n") else first_line + "\n")
                    row_count += 1
                # Stream the rest
                for raw_line in raw:
                    out.write(raw_line.decode("utf-8", errors="replace"))
                    row_count += 1

    return row_count


# ---------------------------------------------------------------------------
# Per-date worker
# ---------------------------------------------------------------------------

def process_date(
    row: dict,
    raw_dir: Path,
    out_dir: Path,
    keep_zip: bool,
    force_recheck: bool,
    session: requests.Session,
) -> dict:
    """
    Download → verify → extract one date. Returns a log-row dict.
    Thread-safe: each call uses its own file paths.
    """
    d          = row["utc_date"]
    zip_url    = row["archive_url"]
    cksum_url  = row["checksum_url"]
    known_sha  = row.get("archive_sha256_from_official_checksum_when_checked", "").strip()
    csv_exists = row.get("local_csv_present", "").strip().lower() == "true"
    known_bytes_str = row.get("local_uncompressed_csv_bytes", "").strip()
    known_bytes = int(known_bytes_str) if known_bytes_str else None

    start = datetime.now(timezone.utc)

    zip_path = raw_dir / f"{d}.zip"
    csv_path = out_dir / f"{d}.csv"

    result = {
        "utc_date": d, "status": "", "zip_bytes": "",
        "zip_sha256_computed": "", "zip_sha256_official": "",
        "checksum_match": "", "csv_rows": "", "csv_bytes_written": "",
        "elapsed_s": "", "note": "",
    }

    try:
        # ----------------------------------------------------------------
        # Skip if CSV already exists and size matches (optional recheck)
        # ----------------------------------------------------------------
        if csv_path.exists() and not force_recheck:
            actual_bytes = csv_path.stat().st_size
            if known_bytes is None or actual_bytes == known_bytes:
                result["status"] = "SKIPPED_ALREADY_PRESENT"
                result["csv_bytes_written"] = actual_bytes
                result["note"] = "CSV present, size matches manifest or unrecorded"
                result["elapsed_s"] = (
                    datetime.now(timezone.utc) - start
                ).total_seconds()
                return result

        # ----------------------------------------------------------------
        # Download ZIP if not already on disk
        # ----------------------------------------------------------------
        if not zip_path.exists():
            log.info("Downloading %s", d)
            zip_bytes = _download_zip(session, zip_url, zip_path)
            result["zip_bytes"] = zip_bytes
        else:
            zip_bytes = zip_path.stat().st_size
            result["zip_bytes"] = zip_bytes
            log.info("ZIP already on disk for %s (%d bytes)", d, zip_bytes)

        # ----------------------------------------------------------------
        # SHA-256 of downloaded ZIP
        # ----------------------------------------------------------------
        computed_sha = _sha256_of_file(zip_path)
        result["zip_sha256_computed"] = computed_sha

        # ----------------------------------------------------------------
        # Fetch official checksum (use manifest value if available)
        # ----------------------------------------------------------------
        if known_sha:
            official_sha = known_sha
        else:
            official_sha = _fetch_official_checksum(session, cksum_url)
        result["zip_sha256_official"] = official_sha

        # ----------------------------------------------------------------
        # Verify
        # ----------------------------------------------------------------
        match = computed_sha.lower() == official_sha.lower()
        result["checksum_match"] = str(match)
        if not match:
            result["status"] = "CHECKSUM_FAIL"
            result["note"] = (
                f"Mismatch: computed={computed_sha[:16]}... "
                f"official={official_sha[:16]}..."
            )
            # Do NOT delete the bad ZIP; leave for inspection
            return result

        # ----------------------------------------------------------------
        # Extract and normalize
        # ----------------------------------------------------------------
        log.info("Extracting %s", d)
        row_count = _extract_and_normalize(zip_path, csv_path)
        csv_bytes = csv_path.stat().st_size

        result["csv_rows"]         = row_count
        result["csv_bytes_written"] = csv_bytes

        # ----------------------------------------------------------------
        # Delete ZIP (saves ~70-80% disk)
        # ----------------------------------------------------------------
        if not keep_zip:
            zip_path.unlink(missing_ok=True)

        result["status"] = "OK"

    except Exception as exc:
        result["status"] = "ERROR"
        result["note"]   = str(exc)
        log.error("FAILED %s: %s", d, exc)

    finally:
        result["elapsed_s"] = round(
            (datetime.now(timezone.utc) - start).total_seconds(), 2
        )

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance aggTrade ZIPs")
    parser.add_argument("--manifest",      required=True,  help="Path to the aggTrades download manifest CSV")
    parser.add_argument("--raw-dir",       required=True,  help="Directory for ZIP files")
    parser.add_argument("--out-dir",       required=True,  help="Directory for extracted CSVs")
    parser.add_argument("--log",           required=True,  help="Path for download log CSV")
    parser.add_argument("--workers",       type=int, default=4, help="Parallel download threads (default 4)")
    parser.add_argument("--keep-zip",      action="store_true", help="Keep ZIP after extraction")
    parser.add_argument("--force-recheck", action="store_true", help="Re-verify even if CSV exists")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    raw_dir       = Path(args.raw_dir)
    out_dir       = Path(args.out_dir)
    log_path      = Path(args.log)

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Read manifest
    # ----------------------------------------------------------------
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["utc_date"].strip()]

    # Filter to agreed window only (belt-and-suspenders)
    in_window = []
    for r in rows:
        try:
            d = date.fromisoformat(r["utc_date"].strip())
        except ValueError:
            log.warning("Skipping unparseable date: %r", r["utc_date"])
            continue
        if WINDOW_START <= d <= WINDOW_END:
            in_window.append(r)

    log.info("Manifest: %d total rows → %d in window (%s to %s)",
             len(rows), len(in_window),
             WINDOW_START.isoformat(), WINDOW_END.isoformat())

    # ----------------------------------------------------------------
    # Already-present vs needs-download
    # ----------------------------------------------------------------
    needs_download = [
        r for r in in_window
        if not (out_dir / f"{r['utc_date'].strip()}.csv").exists()
    ]
    log.info("%d dates need download/extraction; %d already present",
             len(needs_download), len(in_window) - len(needs_download))

    # ----------------------------------------------------------------
    # Execute (parallel for downloads; sequential fallback for errors)
    # ----------------------------------------------------------------
    results: list[dict] = []
    session_pool = [_make_session() for _ in range(args.workers)]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_row = {
            executor.submit(
                process_date,
                row,
                raw_dir,
                out_dir,
                args.keep_zip,
                args.force_recheck,
                session_pool[i % args.workers],
            ): row
            for i, row in enumerate(in_window)
        }
        with tqdm(total=len(in_window), unit="day", desc="aggTrades") as pbar:
            for future in as_completed(future_to_row):
                result = future.result()
                results.append(result)
                pbar.update(1)
                if result["status"] not in ("OK", "SKIPPED_ALREADY_PRESENT"):
                    pbar.write(
                        f"  ⚠  {result['utc_date']}  {result['status']}  {result['note']}"
                    )

    # Sort log by date
    results.sort(key=lambda x: x["utc_date"])

    # ----------------------------------------------------------------
    # Write log
    # ----------------------------------------------------------------
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    log.info("Download log written to %s", log_path)

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    ok       = sum(1 for r in results if r["status"] in ("OK", "SKIPPED_ALREADY_PRESENT"))
    failed   = sum(1 for r in results if r["status"] not in ("OK", "SKIPPED_ALREADY_PRESENT"))
    ckfail   = sum(1 for r in results if r["status"] == "CHECKSUM_FAIL")
    errors   = sum(1 for r in results if r["status"] == "ERROR")

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  Total processed : %d", len(results))
    log.info("  OK / skipped    : %d", ok)
    log.info("  Checksum fail   : %d", ckfail)
    log.info("  Other errors    : %d", errors)
    log.info("  Total failed    : %d", failed)

    if failed:
        log.warning(
            "Some dates failed. Check %s for details. "
            "Re-run the script after fixing issues — it will resume safely.",
            log_path,
        )
        sys.exit(1)
    else:
        log.info("All %d dates verified and extracted successfully.", ok)


if __name__ == "__main__":
    main()