# update_prices_tw_close.py
# Fetch TW stock CLOSE prices and save to prices_close.csv
# Source: FinMind API (free, stable)

import requests
import pandas as pd
from datetime import datetime

FINMIND_API = "https://api.finmindtrade.com/api/v4/data"
PRICE_FILE = "prices_close.csv"

# 🔹 你用的股票（自動從 transactions.xlsx 讀）
TX_FILE = "transactions.xlsx"

def load_symbols():
    df = pd.read_excel(TX_FILE)
    syms = (
        df["Stock Symbol"]
        .astype(str)
        .str.upper()
        .str.replace(r"\.TW$|\.TWO$", "", regex=True)
        .unique()
        .tolist()
    )
    return sorted([s for s in syms if s and s != "TOTAL"])


def fetch_close(symbol):
    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": symbol,
        "start_date": "2015-01-01",
    }
    r = requests.get(FINMIND_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)[["date", "stock_id", "close"]]
    df.columns = ["date", "symbol", "close"]
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    symbols = load_symbols()
    all_df = []

    for s in symbols:
        print(f"Fetching {s}")
        df = fetch_close(s)
        if not df.empty:
            all_df.append(df)

    if not all_df:
        raise RuntimeError("No price data fetched")

    out = pd.concat(all_df).sort_values(["symbol", "date"])
    out.to_csv(PRICE_FILE, index=False)
    print(f"Saved {PRICE_FILE} ({len(out)} rows)")


if __name__ == "__main__":
    main()
