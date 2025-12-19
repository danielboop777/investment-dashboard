# save_portfolio_chart_for_dashboard.py
# Portfolio Value Trend (NO yfinance)
# - Use FinMind for historical prices
# - Use prices_tw_close.csv for latest close fallback
# - Output:
#     portfolio_chart.json
#     portfolio_chart_div.html

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

from FinMind.data import DataLoader
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

TX_FILE = "transactions.xlsx"
PRICE_CSV = "prices_tw_close.csv"

OUT_JSON = "portfolio_chart.json"
OUT_DIV = "portfolio_chart_div.html"


# ---------- helpers ----------
def normalize_symbol(sym: str) -> str:
    s = str(sym).strip().upper()
    if s.endswith(".TW"):
        s = s[:-3]
    if s.endswith(".TWO"):
        s = s[:-4]
    return s if s.isdigit() else ""


def load_transactions():
    df = pd.read_excel(TX_FILE)

    df = df.rename(columns={
        "Buy Time": "buy",
        "Sell Time": "sell",
        "Stock Symbol": "symbol",
        "# of Shares": "shares",
        "Cost Basis": "cost",
        "Sell Price": "sell_price",
        "Status": "status",
    })

    df["buy"] = pd.to_datetime(df["buy"], errors="coerce")
    df["sell"] = pd.to_datetime(df["sell"], errors="coerce")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    df["sell_price"] = pd.to_numeric(df["sell_price"], errors="coerce")

    df["symbol"] = df["symbol"].map(normalize_symbol)
    df = df[df["symbol"] != ""]
    df = df[df["shares"] != 0]

    df["status"] = df["status"].fillna("").str.lower()
    df.loc[(df["status"] == "") & df["sell"].notna(), "status"] = "sold"
    df.loc[(df["status"] == "") & df["sell"].isna(), "status"] = "hold"

    return df


def load_latest_prices():
    if not os.path.exists(PRICE_CSV):
        return {}
    df = pd.read_csv(PRICE_CSV)
    return dict(zip(df["symbol"].astype(str), df["close"].astype(float)))


def finmind_history(symbols, start_date):
    api = DataLoader()
    token = os.getenv("FINMIND_TOKEN", "").strip()
    if token:
        api.login_by_token(token)

    frames = []
    for sym in symbols:
        data = api.taiwan_stock_daily(stock_id=sym, start_date=start_date)
        if data is None or data.empty:
            continue
        frames.append(
            data[["date", "close"]]
            .assign(symbol=sym)
        )

    if not frames:
        raise RuntimeError("No FinMind price data fetched.")

    px = pd.concat(frames)
    px["date"] = pd.to_datetime(px["date"])
    return px.pivot(index="date", columns="symbol", values="close").sort_index()


# ---------- main ----------
def main():
    tx = load_transactions()
    if tx.empty:
        raise RuntimeError("No valid transactions found.")

    symbols = sorted(tx["symbol"].unique().tolist())
    start = tx["buy"].min().normalize()
    today = datetime.now(timezone(timedelta(hours=8))).date()
    start_date = start.strftime("%Y-%m-%d")

    prices_hist = finmind_history(symbols, start_date)

    dates = pd.date_range(start, today, freq="B")
    prices_hist = prices_hist.reindex(dates).ffill()

    latest_price_map = load_latest_prices()
    for sym, px in latest_price_map.items():
        if sym in prices_hist.columns:
            prices_hist.loc[dates[-1], sym] = px

    holdings = pd.DataFrame(0.0, index=dates, columns=prices_hist.columns)

    for _, r in tx.iterrows():
        buy = r["buy"].normalize()
        sell = r["sell"].normalize() if pd.notna(r["sell"]) else None
        sh = float(r["shares"])
        sym = r["symbol"]

        if r["status"] == "sold" and sell is not None:
            end = sell - pd.Timedelta(days=1)
        else:
            end = dates[-1]

        holdings.loc[buy:end, sym] += sh

    portfolio_value = (holdings * prices_hist).sum(axis=1)

    realized = pd.Series(0.0, index=dates)
    unrealized = pd.Series(0.0, index=dates)

    for _, r in tx.iterrows():
        buy = r["buy"].normalize()
        sh = float(r["shares"])
        cost = float(r["cost"])
        sym = r["symbol"]

        if r["status"] == "sold" and pd.notna(r["sell"]) and pd.notna(r["sell_price"]):
            sell = r["sell"].normalize()
            pnl = sh * (float(r["sell_price"]) - cost)
            realized.loc[sell:] += pnl
        else:
            pnl_series = sh * (prices_hist[sym] - cost)
            unrealized.loc[buy:] += pnl_series.loc[buy:]

    net = realized + unrealized

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    fig.add_trace(go.Scatter(x=dates, y=portfolio_value, name="Portfolio Value"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=net, name="Net Return"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=realized, name="Realized", line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=unrealized, name="Unrealized", line=dict(dash="dot")), row=1, col=1)

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=720,
        title=dict(text="Portfolio Value Trend", x=0.5),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.22),
        margin=dict(t=90, b=130, l=70, r=55),
    )

    pio.write_json(fig, OUT_JSON)
    with open(OUT_DIV, "w", encoding="utf-8") as f:
        f.write(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    print("Saved:", OUT_JSON, OUT_DIV)


if __name__ == "__main__":
    main()
