# update_prices_tw_close.py
# Purpose:
#   - Read symbols from transactions.xlsx
#   - Normalize TW symbols (e.g., "2330" -> "2330.TW")
#   - Download latest available Close/Adj Close via yfinance
#   - Save to prices_tw_close.csv
#
# Optional:
#   - If Excel files do not exist, this script can restore them from base64 env vars:
#       TX_XLSX_B64, FF_XLSX_B64
#     (In GitHub Actions you usually restore in workflow already, but this is a safety net.)

import os
import sys
import base64
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
import yfinance as yf

TX_FILE = "transactions.xlsx"
OUT_CSV = "prices_tw_close.csv"

# ---- If you want to support reading a second file later, keep it here (not required now)
FUND_FILE = "Funds Flow.xlsx"

# Environment keys for base64 restore (match workflow secrets)
ENV_TX_B64 = "TX_XLSX_B64"
ENV_FF_B64 = "FF_XLSX_B64"


def log(msg: str):
    print(msg, flush=True)


def restore_excel_from_env_if_missing():
    """
    If transactions.xlsx (and/or Funds Flow.xlsx) is missing,
    try to restore them from base64 in environment variables.
    """
    def write_b64(env_key: str, out_path: str):
        b64 = os.environ.get(env_key, "").strip()
        if not b64:
            return False
        try:
            data = base64.b64decode(b64)
            with open(out_path, "wb") as f:
                f.write(data)
            log(f"[OK] Restored {out_path} from {env_key} ({len(data):,} bytes)")
            return True
        except Exception as e:
            log(f"[ERROR] Failed restoring {out_path} from {env_key}: {e}")
            return False

    tx_exists = Path(TX_FILE).exists()
    ff_exists = Path(FUND_FILE).exists()

    if not tx_exists:
        log(f"[WARN] {TX_FILE} not found. Trying to restore from env: {ENV_TX_B64}")
        write_b64(ENV_TX_B64, TX_FILE)

    # Funds Flow isn't needed by this script, but many people want it restored too.
    if not ff_exists:
        log(f"[INFO] {FUND_FILE} not found. (Optional) Trying to restore from env: {ENV_FF_B64}")
        write_b64(ENV_FF_B64, FUND_FILE)


def normalize_symbol(sym: str) -> str:
    """
    Normalize symbol formats for yfinance.
    - "2330" -> "2330.TW"
    - keep "0050.TW" as is
    - allow "^TWII" etc.
    """
    if sym is None:
        return ""
    s = str(sym).strip().upper().replace(":", ".")
    if s == "" or s == "NAN" or s == "TOTAL":
        return ""

    # If already contains a dot suffix (.TW/.TWO etc.) or starts with ^, keep it
    if s.startswith("^"):
        return s
    if "." in s:
        return s

    # Pure digits -> assume TW stock
    if s.isdigit():
        return f"{s}.TW"

    # fallback: leave as-is
    return s


def load_symbols_from_transactions() -> list[str]:
    if not Path(TX_FILE).exists():
        log("")
        log("[ERROR] transactions.xlsx not found.")
        log("✅ 你需要把 transactions.xlsx 放在 repo 根目錄，或在 GitHub Actions 用 Secrets 還原。")
        log("   - workflow 內應該先 Restore Excel files 再跑這支 script")
        log(f"   - 或你可以在 Secrets 設定 {ENV_TX_B64}（base64）")
        log("")
        raise FileNotFoundError(TX_FILE)

    df = pd.read_excel(TX_FILE)

    # Try common column names
    col_candidates = ["Stock Symbol", "Symbol", "symbol", "Ticker", "ticker"]
    sym_col = None
    for c in col_candidates:
        if c in df.columns:
            sym_col = c
            break

    if sym_col is None:
        raise ValueError(f"Cannot find symbol column in {TX_FILE}. Expected one of: {col_candidates}")

    syms = df[sym_col].dropna().astype(str).tolist()
    syms = [normalize_symbol(s) for s in syms]
    syms = [s for s in syms if s]

    # unique + stable order
    seen = set()
    out = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)

    if not out:
        raise ValueError("No valid symbols found in transactions.xlsx")

    return out


def download_latest_close(tickers: list[str]) -> pd.DataFrame:
    """
    Download last ~10 trading days and pick latest available Close/Adj Close.
    """
    tickers = [t for t in tickers if t]
    if not tickers:
        return pd.DataFrame(columns=["symbol", "price", "date"])

    # yfinance can take space-joined tickers
    data = yf.download(
        tickers=" ".join(tickers),
        period="15d",
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if data is None or len(data) == 0:
        raise RuntimeError("yfinance returned empty data. (network issue or invalid tickers?)")

    # Determine which price field we use
    # Prefer Adj Close if it exists; otherwise Close
    price_field = None
    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = data.columns.get_level_values(0)
        if "Adj Close" in lvl0:
            price_field = "Adj Close"
        elif "Close" in lvl0:
            price_field = "Close"
        else:
            raise RuntimeError("yfinance data missing Close/Adj Close columns")
        px = data[price_field].copy()
    else:
        # single ticker case: columns are normal
        if "Adj Close" in data.columns:
            price_field = "Adj Close"
        elif "Close" in data.columns:
            price_field = "Close"
        else:
            raise RuntimeError("yfinance data missing Close/Adj Close columns")
        px = data[[price_field]].copy()
        px.columns = [tickers[0]]

    # Normalize index to date only
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px.dropna(how="all")

    if px.empty:
        raise RuntimeError("All downloaded prices are NaN. Check tickers.")

    # Latest available row
    latest_date = px.index[-1]
    latest_row = px.iloc[-1]

    out = pd.DataFrame({"symbol": latest_row.index.astype(str), "price": latest_row.values})
    out["date"] = pd.Timestamp(latest_date).strftime("%Y-%m-%d")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")

    return out


def main():
    log("=== update_prices_tw_close.py ===")

    # Safety: restore Excel if missing (optional)
    restore_excel_from_env_if_missing()

    # Load symbols
    symbols = load_symbols_from_transactions()
    log(f"[OK] Loaded {len(symbols)} symbols from {TX_FILE}")

    # Download latest close
    prices = download_latest_close(symbols)

    # Report missing tickers
    missing = prices[prices["price"].isna()]["symbol"].tolist()
    ok = prices[prices["price"].notna()].copy()

    # Save
    ok = ok.sort_values("symbol")
    ok.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    log(f"[OK] Saved: {OUT_CSV} ({ok.shape[0]} tickers)")

    if missing:
        log("")
        log("[WARN] Some tickers have no price (yfinance returned NaN). They were skipped:")
        for m in missing:
            log(f"  - {m}")

    # Print a small preview
    log("")
    log("Preview:")
    log(ok.head(10).to_string(index=False))

    log("=== done ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("")
        log("[FATAL] update_prices_tw_close.py failed.")
        log(str(e))
        # show file list for debugging in GitHub Actions
        try:
            log("\n[DEBUG] Files in workspace:")
            for p in sorted(Path(".").glob("*")):
                log(f" - {p.name}")
        except Exception:
            pass
        sys.exit(1)
