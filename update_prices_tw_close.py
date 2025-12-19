# update_prices_tw_close.py
# Purpose:
#   - Read transactions.xlsx -> extract Taiwan stock symbols
#   - Fetch latest close price via FinMind
#   - Save to prices_tw_close.csv
#
# Notes:
#   - No yfinance dependency
#   - Supports symbols like "2330" or "2330.TW" (we normalize to "2330")

import os
import pandas as pd
from datetime import datetime, timedelta, timezone

from FinMind.data import DataLoader

TX_FILE = "transactions.xlsx"
OUT_CSV = "prices_tw_close.csv"


def normalize_tw_symbol(sym: str) -> str:
    if sym is None:
        return ""
    s = str(sym).strip().upper().replace(":", ".")
    if s in ("", "NAN", "TOTAL"):
        return ""
    # 2330.TW -> 2330
    if s.endswith(".TW"):
        s = s[:-3]
    # 2330.TWO -> 2330 (你如果有上櫃，FinMind一樣用純數字)
    if s.endswith(".TWO"):
        s = s[:-4]
    # 只保留純數字
    if s.isdigit():
        return s
    # 如果不是數字（例如ETF代碼也會是數字），其他就丟掉
    return ""


def load_symbols() -> list[str]:
    if not os.path.exists(TX_FILE):
        raise FileNotFoundError(f"Missing file: {TX_FILE}. (Did you restore Excel in workflow?)")

    df = pd.read_excel(TX_FILE)

    # stock symbol column name might vary; try common names
    possible_cols = ["Stock Symbol", "symbol", "Symbol", "Ticker"]
    col = None
    for c in possible_cols:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError(f"Cannot find symbol column in {TX_FILE}. Columns={list(df.columns)}")

    syms = df[col].dropna().map(normalize_tw_symbol)
    syms = [s for s in syms.tolist() if s]
    return sorted(set(syms))


def finmind_last_close(symbol: str, api: DataLoader, start_date: str) -> tuple[str, float]:
    """
    Return (date_str, close_price) for the latest available trading day.
    """
    data = api.taiwan_stock_daily(stock_id=symbol, start_date=start_date)
    if data is None or data.empty:
        raise ValueError(f"No FinMind data for {symbol}")

    # FinMind columns typically include: date, close
    data = data.dropna(subset=["date", "close"]).sort_values("date")
    if data.empty:
        raise ValueError(f"FinMind data has no close for {symbol}")

    last = data.iloc[-1]
    return str(last["date"]), float(last["close"])


def main():
    symbols = load_symbols()
    if not symbols:
        raise ValueError("No valid Taiwan stock symbols found in transactions.xlsx")

    # GitHub runner is UTC; but TW market uses UTC+8.
    tz_utc8 = timezone(timedelta(hours=8))
    now_tw = datetime.now(tz_utc8)

    # Get last ~30 days to ensure we have the latest close (avoid holidays)
    start_date = (now_tw.date() - timedelta(days=30)).strftime("%Y-%m-%d")

    api = DataLoader()
    token = os.getenv("FINMIND_TOKEN", "").strip()
    if token:
        api.login_by_token(token)

    rows = []
    errors = []

    for sym in symbols:
        try:
            d, close = finmind_last_close(sym, api, start_date)
            rows.append({"symbol": sym, "date": d, "close": close})
        except Exception as e:
            errors.append(f"{sym}: {e}")

    out = pd.DataFrame(rows).sort_values(["symbol"])
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"Saved {OUT_CSV} with {len(out)} symbols.")
    if errors:
        print("Warnings (symbols failed):")
        for msg in errors:
            print(" -", msg)


if __name__ == "__main__":
    main()
