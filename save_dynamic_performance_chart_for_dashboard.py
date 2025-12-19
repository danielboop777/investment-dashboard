# save_dynamic_performance_chart_for_dashboard.py
# FinMind version (Close price only)
# Outputs:
#   - dynamic_performance_chart.json
#   - dynamic_performance_chart_div.html

import os
import time
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

TX_FILE = "transactions.xlsx"
FUND_FILE = "Funds Flow.xlsx"

# Benchmarks (FinMind style)
BENCH_TWII = "TAIEX"   # Taiwan Weighted Index
BENCH_0050 = "0050"    # Yuanta Taiwan 50 ETF

OUT_JSON = "dynamic_performance_chart.json"
OUT_DIV  = "dynamic_performance_chart_div.html"

FINMIND_URL = "https://api.finmindtrade.com/api/v4/data"


# =========================
# Helpers
# =========================
def _norm_symbol(x: str) -> str:
    s = str(x).strip().upper().replace(":", ".")
    s = pd.Series([s]).str.replace(r"\.TW$|\.TWO$", "", regex=True).iloc[0]
    return s


def finmind_price_series(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
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


def finmind_price_df(symbols, start, end):
    out = {}
    for sym in sorted(set(symbols)):
        try:
            out[sym] = finmind_price_series(sym, start, end)
        except Exception:
            out[sym] = pd.Series(dtype=float)
        time.sleep(0.12)

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _as_1d(x):
    if isinstance(x, pd.Series):
        return x.to_numpy()
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0].to_numpy()
    return np.asarray(x).reshape(-1)


# =========================
# Load transactions
# =========================
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
    df["symbol"] = df["symbol"].astype(str).map(_norm_symbol)

    df["status"] = df["status"].fillna("").str.lower().str.strip()
    df.loc[df["status"].isin(["sold", "sell", "closed"]), "status"] = "sold"
    df.loc[df["status"].isin(["hold", "holding", "open"]), "status"] = "hold"
    df.loc[(df["status"] == "") & df["sell"].notna(), "status"] = "sold"
    df.loc[(df["status"] == "") & df["sell"].isna(), "status"] = "hold"

    df = df.dropna(subset=["buy", "symbol", "shares", "cost"])
    df = df[df["shares"] != 0]
    return df


def load_funds_flow(dates):
    ff = pd.read_excel(FUND_FILE)
    ff.columns = ["date", "cash_inflow", "cash_balance"]
    ff["date"] = pd.to_datetime(ff["date"], errors="coerce")
    ff["cash_balance"] = pd.to_numeric(ff["cash_balance"], errors="coerce")
    ff = ff.dropna(subset=["date", "cash_balance"]).set_index("date").sort_index()

    cash = ff["cash_balance"].reindex(dates).ffill().bfill()
    cash = cash.replace(0, np.nan).ffill().bfill().fillna(1e-9)
    return cash


def compute_net_return(tx, prices, dates):
    net = pd.Series(0.0, index=dates)

    for _, r in tx.iterrows():
        buy = r["buy"].normalize()
        sh = float(r["shares"])
        cost = float(r["cost"])
        sym = r["symbol"]

        if r["status"] == "sold" and pd.notna(r["sell"]) and pd.notna(r["sell_price"]):
            sell = r["sell"].normalize()
            pnl = sh * (float(r["sell_price"]) - cost)
            net.loc[sell:] += pnl
        else:
            if sym not in prices.columns:
                continue
            pnl = sh * (prices[sym] - cost)
            net.loc[buy:] += pnl.loc[buy:]

    return net


def safe_left(v, fallback=1e-9):
    v = float(v) if pd.notna(v) else fallback
    if not np.isfinite(v) or v == 0:
        return fallback
    return v


# =========================
# MAIN
# =========================
def main():
    tx = load_transactions()

    start = tx["buy"].min().normalize()
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(start, end, freq="B")

    symbols = sorted(tx["symbol"].unique().tolist())
    prices = finmind_price_df(symbols, start, end).reindex(dates).ffill()

    twii_s = finmind_price_series(BENCH_TWII, start, end).reindex(dates).ffill()
    etf_s  = finmind_price_series(BENCH_0050, start, end).reindex(dates).ffill()

    net_s  = compute_net_return(tx, prices, dates)
    cash_s = load_funds_flow(dates)

    x_dates  = list(pd.to_datetime(dates).to_pydatetime())
    net_raw  = _as_1d(net_s)
    cash_raw = _as_1d(cash_s)
    twii_raw = _as_1d(twii_s)
    etf_raw  = _as_1d(etf_s)

    net_left  = safe_left(net_raw[0], 0.0)
    twii_left = safe_left(twii_raw[0])
    etf_left  = safe_left(etf_raw[0])

    net_pct0  = (net_raw - net_left) / cash_raw
    twii_pct0 = (twii_raw / twii_left) - 1.0
    etf_pct0  = (etf_raw  / etf_left)  - 1.0

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_dates, y=net_raw,  visible=False, customdata=cash_raw))
    fig.add_trace(go.Scatter(x=x_dates, y=twii_raw, visible=False))
    fig.add_trace(go.Scatter(x=x_dates, y=etf_raw,  visible=False))

    fig.add_trace(go.Scatter(x=x_dates, y=net_pct0,  name="Net Return %", mode="lines"))
    fig.add_trace(go.Scatter(x=x_dates, y=twii_pct0, name="TWII %",       mode="lines"))
    fig.add_trace(go.Scatter(x=x_dates, y=etf_pct0,  name="0050 %",       mode="lines"))

    fig.update_layout(
        title=dict(text="Performance Analysis vs. Benchmark", x=0.5),
        template="plotly_white",
        hovermode="x unified",
        height=660,
        yaxis=dict(title="Performance", tickformat=".0%"),
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.10),
            rangeselector=dict(buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=2, label="2Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ])
        ),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.28),
        margin=dict(t=85, b=180, l=60, r=40),
    )

    pio.write_json(fig, OUT_JSON)
    with open(OUT_DIV, "w", encoding="utf-8") as f:
        f.write(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    print(f"Saved:\n- {OUT_JSON}\n- {OUT_DIV}")


if __name__ == "__main__":
    main()
