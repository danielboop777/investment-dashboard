# save_portfolio_chart_for_dashboard.py
# Yahoo-style:
#   Top = portfolio value (lines)
#   Bottom = transaction count (bars)
#
# Outputs:
#   - portfolio_chart.json
#   - portfolio_chart_div.html

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

EXCEL_PATH = "transactions.xlsx"
SHEET_NAME = 0
COST_BASIS_IS_TOTAL = False

OUT_JSON = "portfolio_chart.json"
OUT_DIV = "portfolio_chart_div.html"


def normalize_symbol(sym: str) -> str:
    if sym is None:
        return ""
    s = str(sym).strip().upper().replace(":", ".")
    if s in ("", "NAN", "TOTAL"):
        return ""
    if s.startswith("^"):
        return s
    if "." in s:
        return s
    if s.isdigit():  # 2330 -> 2330.TW
        return f"{s}.TW"
    return s


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

    df["buy_time"] = pd.to_datetime(df.get("buy_time"), errors="coerce")
    df["sell_time"] = pd.to_datetime(df.get("sell_time"), errors="coerce")
    df["sell_price"] = pd.to_numeric(df.get("sell_price"), errors="coerce")
    df["shares"] = pd.to_numeric(df.get("shares"), errors="coerce")
    df["cost_basis"] = pd.to_numeric(df.get("cost_basis"), errors="coerce")

    df["symbol"] = df.get("symbol").map(normalize_symbol)

    df = df.dropna(subset=["buy_time", "symbol", "shares", "cost_basis"])
    df = df[df["shares"] != 0]

    df["status"] = df.get("status").fillna("").astype(str).str.lower().str.strip()
    df.loc[df["status"].isin(["sold", "sell", "closed"]), "status"] = "sold"
    df.loc[df["status"].isin(["hold", "holding", "open"]), "status"] = "hold"
    df.loc[(df["status"] == "") & df["sell_time"].notna(), "status"] = "sold"
    df.loc[(df["status"] == "") & df["sell_time"].isna(), "status"] = "hold"

    # cost_per_share: if your "Cost Basis" column is total cost, set COST_BASIS_IS_TOTAL=True
    df["cost_per_share"] = (
        df["cost_basis"] / df["shares"]
        if COST_BASIS_IS_TOTAL
        else df["cost_basis"]
    )

    return df


# =========================
# Prices (robust)
# =========================
def download_prices(tickers, start, end) -> pd.DataFrame:
    tickers = [t for t in sorted(set(tickers)) if isinstance(t, str) and t.strip()]
    if not tickers:
        raise ValueError("No tickers to download.")

    data = yf.download(
        tickers=" ".join(tickers),
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
        group_by="column",
        threads=True,
    )

    if data is None or len(data) == 0:
        raise ValueError("yfinance returned empty data.")

    # MultiIndex: columns like ('Adj Close', '2330.TW')
    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = data.columns.get_level_values(0)
        if "Adj Close" in lvl0:
            px = data["Adj Close"].copy()
        elif "Close" in lvl0:
            px = data["Close"].copy()
        else:
            raise ValueError("yfinance data missing Close/Adj Close.")
    else:
        # single ticker case
        col = "Adj Close" if "Adj Close" in data.columns else ("Close" if "Close" in data.columns else None)
        if col is None:
            raise ValueError("yfinance data missing Close/Adj Close.")
        px = data[[col]].copy()
        px.columns = [tickers[0]]

    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px.sort_index().dropna(how="all")

    if px.empty:
        raise ValueError("All prices are NaN after cleanup.")

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

        if r["status"] == "sold" and pd.notna(r["sell_time"]) and pd.notna(r["sell_price"]):
            sell = r["sell_time"].normalize()
            realized.loc[sell:] += sh * (float(r["sell_price"]) - cost)
        else:
            if sym not in prices.columns:
                continue
            pnl = sh * (prices[sym] - cost)
            unrealized.loc[buy:] += pnl.loc[buy:]

    return realized, unrealized


# =========================
# Transactions count
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

    tickers = tx["symbol"].dropna().unique().tolist()

    prices = download_prices(tickers, start, end).reindex(dates).ffill()

    # 若有些 ticker 全部 NaN，直接丟掉避免後面算錯
    good_cols = [c for c in prices.columns if prices[c].notna().any()]
    dropped = [c for c in prices.columns if c not in good_cols]
    if dropped:
        print("[WARN] Dropped tickers with no price data:", dropped)
        prices = prices[good_cols]

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
