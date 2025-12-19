# update_prices_tw_close.py
# Generate: prices_close.csv
# Columns: symbol,date,last,prev,chg_abs,chg_pct
# ✅ NO yfinance: use FinMind

import os
import sys
import pandas as pd
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

    syms = [s for s in syms if s and s not in ["TOTAL", "NAN"]]
    return sorted(set(syms))


def finmind_last_two_closes(stock_id: str, start_date: str, end_date: str):
    """
    Return: (last_close, prev_close, last_date_str)
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

    end_date = now_utc8.strftime("%Y-%m-%d")
    start_date = (now_utc8 - timedelta(days=30)).strftime("%Y-%m-%d")

    rows = []
    for sym in symbols:
        last, prev, last_date = finmind_last_two_closes(sym, start_date, end_date)

        chg_abs = (last - prev) if (last is not None and prev is not None) else None
        chg_pct = ((last / prev) - 1.0) if (last is not None and prev not in [None, 0]) else None

        rows.append({
            "symbol": sym,
            "date": last_date,     # ✅ 這欄就是你 dashboard 期待的
            "last": last,
            "prev": prev,
            "chg_abs": chg_abs,
            "chg_pct": chg_pct,
        })

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["last"] = pd.to_numeric(out["last"], errors="coerce")
    out["prev"] = pd.to_numeric(out["prev"], errors="coerce")
    out["chg_abs"] = pd.to_numeric(out["chg_abs"], errors="coerce")
    out["chg_pct"] = pd.to_numeric(out["chg_pct"], errors="coerce")

    out.to_csv(PRICE_FILE, index=False, encoding="utf-8")
    print(f"[OK] Wrote {PRICE_FILE} ({len(out):,} rows)")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
