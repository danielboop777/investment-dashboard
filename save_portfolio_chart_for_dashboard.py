# save_portfolio_chart_for_dashboard.py
# FinMind version (Close price only)
#   Top = portfolio value (lines)
#   Bottom = transaction count (bars)

import os
import time
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

EXCEL_PATH = "transactions.xlsx"
SHEET_NAME = 0
COST_BASIS_IS_TOTAL = False

OUT_JSON = "portfolio_chart.json"
OUT_DIV = "portfolio_chart_div.html"

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


# =========================
# Helpers
# =========================
def _norm_symbol(x: str) -> str:
    s = str(x).strip().upper().replace(":", ".")
    # normalize TW tickers to FinMind style (no suffix)
    s = pd.Series([s]).str.replace(r"\.TW$|\.TWO$", "", regex=True).iloc[0]
    return s


def finmind_get_price_series(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """
    Return close price series (daily) from FinMind TaiwanStockPrice.
    symbol: e.g. "2330" (NOT 2330.TW), but we also normalize if user passes .TW/.TWO
    """
    token = os.environ.get("FINMIND_API_TOKEN", "")
    sym = _norm_symbol(symbol)

    params = {
        "dataset": "TaiwanStockPrice",
        "data_id": sym,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "token": token,
    }

    r = requests.get(FINMIND_URL, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    df = pd.DataFrame(j.get("data", []))
    if df.empty:
        raise ValueError(f"No FinMind data for {sym}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").set_index("date")

    s = df["close"].copy()
    s.index = pd.to_datetime(s.index.date)
    return s.sort_index()


def finmind_price_df(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Download close prices for multiple symbols by calling FinMind per symbol.
    (Simple + robust; OK for personal portfolio sizes.)
    """
    out = {}
    uniq = sorted({_norm_symbol(s) for s in symbols if str(s).strip()})
    for sym in uniq:
        try:
            out[sym] = finmind_get_price_series(sym, start, end)
        except Exception:
            out[sym] = pd.Series(dtype=float)
        time.sleep(0.12)  # be gentle to API

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


# =========================
# Load transactions
# =========================
def load_transactions(path: str, sheet=SHEET_NAME) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)

    df = df.rename(columns={
        "Buy Time": "buy_time",
        "Sell Time": "sell_time",
        "Stock Symbol": "symbol",
        "# of Shares": "shares",
        "Cost Basis": "cost_basis",
        "Sell Price": "sell_price",
        "Status": "status",
    })

    df["buy_time"] = pd.to_datetime(df["buy_time"], errors="coerce")
    df["sell_time"] = pd.to_datetime(df["sell_time"], errors="coerce")
    df["sell_price"] = pd.to_numeric(df["sell_price"], errors="coerce")
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["cost_basis"] = pd.to_numeric(df["cost_basis"], errors="coerce")

    # ✅ normalize symbols: 2330.TW / 6278.TWO -> 2330 / 6278
    df["symbol"] = df["symbol"].astype(str).map(_norm_symbol)

    df = df.dropna(subset=["buy_time", "symbol", "shares", "cost_basis"])
    df = df[df["shares"] != 0]

    df["status"] = df["status"].fillna("").str.lower().str.strip()
    df.loc[df["status"].isin(["sold", "sell", "closed"]), "status"] = "sold"
    df.loc[df["status"].isin(["hold", "holding", "open"]), "status"] = "hold"
    df.loc[(df["status"] == "") & df["sell_time"].notna(), "status"] = "sold"
    df.loc[(df["status"] == "") & df["sell_time"].isna(), "status"] = "hold"

    df["cost_per_share"] = (
        df["cost_basis"] / df["shares"]
        if COST_BASIS_IS_TOTAL
        else df["cost_basis"]
    )

    return df


# =========================
# Portfolio value
# =========================
def compute_portfolio_value(tx, prices, dates):
    holdings = pd.DataFrame(0.0, index=dates, columns=prices.columns)

    for _, r in tx.iterrows():
        buy = r["buy_time"].normalize()
        sell = r["sell_time"].normalize() if pd.notna(r["sell_time"]) else None
        end = sell - pd.Timedelta(days=1) if r["status"] == "sold" and sell else dates[-1]
        sym = r["symbol"]
        if sym in holdings.columns:
            holdings.loc[buy:end, sym] += float(r["shares"])

    return (holdings * prices).sum(axis=1)


# =========================
# Gains
# =========================
def compute_realized_unrealized(tx, prices, dates):
    realized = pd.Series(0.0, index=dates)
    unrealized = pd.Series(0.0, index=dates)

    for _, r in tx.iterrows():
        buy = r["buy_time"].normalize()
        sh = float(r["shares"])
        cost = float(r["cost_per_share"])
        sym = r["symbol"]

        if r["status"] == "sold" and pd.notna(r["sell_time"]):
            sell = r["sell_time"].normalize()
            realized.loc[sell:] += sh * (float(r["sell_price"]) - cost)
        else:
            if sym not in prices.columns:
                continue
            pnl = sh * (prices[sym] - cost)
            unrealized.loc[buy:] += pnl.loc[buy:]

    return realized, unrealized


# =========================
# Transactions
# =========================
def compute_transaction_counts(tx, dates):
    events = pd.concat([
        tx["buy_time"].dropna().dt.normalize(),
        tx["sell_time"].dropna().dt.normalize(),
    ])
    return events.value_counts().reindex(dates, fill_value=0)


# =========================
# Figure
# =========================
def build_figure(dates, pv, rg, ug, net, txns):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.03,
    )

    # --- Top lines ---
    fig.add_trace(go.Scatter(x=dates, y=pv, name="Portfolio Value", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=net, name="Net Return", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=rg, name="Realized Gain", mode="lines", line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ug, name="Unrealized Gain", mode="lines", line=dict(dash="dot")), row=1, col=1)

    # --- Bottom bars ---
    fig.add_trace(go.Bar(
        x=dates, y=txns,
        name="Transactions",
        marker_color="rgb(60,60,60)",
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=720,
        title=dict(text="Portfolio Value Trend", x=0.5, y=0.995, xanchor="center"),
        margin=dict(t=135, b=130, l=70, r=55),
        legend=dict(orientation="h", x=0.5, y=-0.22, xanchor="center"),
    )

    # Top Y-axis
    fig.update_yaxes(
        title_text="Value",
        tickprefix="$",
        tickformat=",.0f",
        autorange=True,
        rangemode="normal",
        fixedrange=False,
        row=1, col=1,
    )

    # Bottom Y-axis
    fig.update_yaxes(
        title_text="# of Transactions",
        rangemode="tozero",
        row=2, col=1,
    )

    # Range selector buttons
    fig.update_xaxes(
        title_text=None,
        row=1, col=1,
        rangeselector=dict(
            y=1.18,
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=2, label="2Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
        ),
    )

    fig.update_xaxes(
        title_text=None,
        row=2, col=1,
        rangeslider=dict(visible=True, thickness=0.1),
    )

    return fig


# =========================
# MAIN
# =========================
def main():
    tx = load_transactions(EXCEL_PATH)
    if tx.empty:
        raise ValueError("No valid transactions found in transactions.xlsx")

    start = tx["buy_time"].min().normalize()
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(start, end, freq="B")

    symbols = tx["symbol"].dropna().astype(str).unique().tolist()
    prices_raw = finmind_price_df(symbols, start, end)

    if prices_raw.empty:
        raise ValueError("No price data returned from FinMind for portfolio symbols.")

    prices = prices_raw.reindex(dates).ffill()

    pv = compute_portfolio_value(tx, prices, dates)
    rg, ug = compute_realized_unrealized(tx, prices, dates)
    net = rg + ug
    txns = compute_transaction_counts(tx, dates)

    fig = build_figure(dates, pv, rg, ug, net, txns)

    pio.write_json(fig, OUT_JSON)
    with open(OUT_DIV, "w", encoding="utf-8") as f:
        f.write(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    print(f"Saved:\n- {OUT_JSON}\n- {OUT_DIV}")


if __name__ == "__main__":
    main()
