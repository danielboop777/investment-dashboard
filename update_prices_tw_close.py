# update_prices_tw_close.py
# Generate: prices_close.csv
# Columns: symbol,last,prev,chg_abs,chg_pct,asof
# ✅ NO yfinance: use FinMind

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from FinMind.data import DataLoader

TX_FILE = "transactions.xlsx"
PRICE_FILE = "prices_close.csv"


def load_symbols() -> list[str]:
    if not os.path.exists(TX_FILE):
        raise FileNotFoundError(f"Missing {TX_FILE}. Did you restore Excel from Secrets in workflow?")

    df = pd.read_excel(TX_FILE)
    if "Stock Symbol" not in df.columns:
        raise ValueError("transactions.xlsx missing column: 'Stock Symbol'")

    syms = (
        df["Stock Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(":", ".", regex=False)
        .tolist()
    )

    # keep only non-empty and not TOTAL
    syms = [s for s in syms if s and s not in ["TOTAL", "NAN"]]

    # 你目前已改成純數字：2330 / 0050
    # 這裡就不自動加 .TW / .TWO，完全用 FinMind stock_id
    syms = sorted(set(syms))
    return syms


def finmind_last_two_closes(stock_id: str, start_date: str, end_date: str) -> tuple[float | None, float | None, str | None]:
    """
    Return (last_close, prev_close, last_date_str)
    """
    dl = DataLoader()
    try:
        d = dl.taiwan_stock_daily(stock_id=stock_id, start_date=start_date, end_date=end_date)
    except Exception:
        return (None, None, None)

    if d is None or d.empty or ("date" not in d.columns) or ("close" not in d.columns):
        return (None, None, None)

    d = d[["date", "close"]].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d = d.dropna(subset=["date", "close"]).sort_values("date")
    if d.empty:
        return (None, None, None)

    last = float(d["close"].iloc[-1])
    prev = float(d["close"].iloc[-2]) if len(d) >= 2 else float(d["close"].iloc[-1])
    last_date = d["date"].iloc[-1].strftime("%Y-%m-%d")
    return (last, prev, last_date)


def main():
    tz_utc8 = timezone(timedelta(hours=8))
    now_utc8 = datetime.now(tz_utc8)

    symbols = load_symbols()
    if not symbols:
        raise RuntimeError("No symbols found in transactions.xlsx")

    # 抓最近 20 天，確保能拿到兩個交易日 close
    end_date = now_utc8.strftime("%Y-%m-%d")
    start_date = (now_utc8 - timedelta(days=30)).strftime("%Y-%m-%d")

    rows = []
    asof_any = None

    for sym in symbols:
        last, prev, last_date = finmind_last_two_closes(sym, start_date, end_date)

        if last is None or prev is None:
            chg_abs = None
            chg_pct = None
        else:
            chg_abs = last - prev
            chg_pct = (last / prev - 1.0) if prev != 0 else None

        if last_date:
            asof_any = asof_any or last_date

        rows.append({
            "symbol": sym,
            "last": last,
            "prev": prev,
            "chg_abs": chg_abs,
            "chg_pct": chg_pct,
            "asof": last_date,
        })

    out = pd.DataFrame(rows)

    # 讓缺值更乾淨（dashboard 會自己處理 NaN）
    out["last"] = pd.to_numeric(out["last"], errors="coerce")
    out["prev"] = pd.to_numeric(out["prev"], errors="coerce")
    out["chg_abs"] = pd.to_numeric(out["chg_abs"], errors="coerce")
    out["chg_pct"] = pd.to_numeric(out["chg_pct"], errors="coerce")

    out.to_csv(PRICE_FILE, index=False, encoding="utf-8")
    print(f"[OK] Wrote {PRICE_FILE} ({len(out):,} rows). Example:")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
