# save_portfolio_chart_for_dashboard.py
# Yahoo-style:
#   Top = portfolio value (lines)
#   Bottom = transaction count (bars)
#
# ✅ NO yfinance version: use FinMind for TW stock prices

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from FinMind.data import DataLoader

EXCEL_PATH = "transactions.xlsx"
SHEET_NAME = 0
COST_BASIS_IS_TOTAL = False

OUT_JSON = "portfolio_chart.json"
OUT_DIV = "portfolio_chart_div.html"


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

    # 你已經把 symbol 改成純數字了（2330、0050），這裡就保持乾淨
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper().str.replace(":", ".", regex=False)

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
# Prices via FinMind
# =========================
def finmind_price_df(tickers, start, end) -> pd.DataFrame:
    """
    Return close prices DataFrame indexed by date (datetime64[ns]) columns=tickers
    """
    dl = DataLoader()

    start_s = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_s = pd.Timestamp(end).strftime("%Y-%m-%d")

    out = {}
    for t in sorted(set([str(x).strip() for x in tickers if str(x).strip()])):
        # FinMind 台股 stock_id 一般就是 "2330", "0050"
        try:
            d = dl.taiwan_stock_daily(stock_id=t, start_date=start_s, end_date=end_s)
        except Exception:
            d = pd.DataFrame()

        if d is None or d.empty:
            # 留空，後面會處理
            continue

        # 欄位通常包含 date / close
        if "date" not in d.columns or "close" not in d.columns:
            continue

        s = d[["date", "close"]].copy()
        s["date"] = pd.to_datetime(s["date"], errors="coerce")
        s["close"] = pd.to_numeric(s["close"], errors="coerce")
        s = s.dropna(subset=["date", "close"]).sort_values("date")
        if s.empty:
            continue

        out[t] = s.set_index("date")["close"]

    if not out:
        return pd.DataFrame()

    px = pd.concat(out, axis=1)
    px.index = pd.to_datetime(px.index.date)
    px = px.sort_index()
    return px


# =========================
# Portfolio value
# =========================
def compute_portfolio_value(tx, prices, dates):
    holdings = pd.DataFrame(0.0, index=dates, columns=prices.columns)

    for _, r in tx.iterrows():
        buy = r["buy_time"].normalize()
        sell = r["sell_time"].normalize() if pd.notna(r["sell_time"]) else None
        end = sell - pd.Timedelta(days=1) if r["status"] == "sold" and sell is not None else dates[-1]
        sym = r["symbol"]

        if sym in holdings.columns:
            if buy <= dates[-1]:
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
            if pd.notna(r["sell_price"]):
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

    fig.add_trace(go.Scatter(x=dates, y=pv, name="Portfolio Value", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=net, name="Net Return", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=rg, name="Realized Gain", mode="lines", line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=ug, name="Unrealized Gain", mode="lines", line=dict(dash="dot")), row=1, col=1)

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

    fig.update_yaxes(
        title_text="Value",
        tickprefix="$",
        tickformat=",.0f",
        autorange=True,
        fixedrange=False,
        row=1, col=1,
    )

    fig.update_yaxes(
        title_text="# of Transactions",
        rangemode="tozero",
        row=2, col=1,
    )

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

    start = tx["buy_time"].min().normalize()
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(start, end, freq="B")

    prices = finmind_price_df(tx["symbol"].tolist(), start, end)
    if prices.empty:
        raise RuntimeError("FinMind returned no prices. Check symbols in transactions.xlsx (should be like 2330 / 0050).")

    prices = prices.reindex(dates).ffill()

    pv = compute_portfolio_value(tx, prices, dates)
    rg, ug = compute_realized_unrealized(tx, prices, dates)
    net = rg + ug
    txns = compute_transaction_counts(tx, dates)

    fig = build_figure(dates, pv, rg, ug, net, txns)

    pio.write_json(fig, OUT_JSON)
    with open(OUT_DIV, "w", encoding="utf-8") as f:
        f.write(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    fig.show()


if __name__ == "__main__":
    main()
