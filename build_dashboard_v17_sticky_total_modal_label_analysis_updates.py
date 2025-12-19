# build_dashboard_v17_sticky_total_modal_label_analysis_updates.py
# STATIC dashboard builder
# - NO yfinance
# - Read close prices from prices_close.csv
# - Output: dashboard.html

import math
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TX_FILE = "transactions.xlsx"
PRICE_FILE = "prices_close.csv"

CHART1_DIV = "portfolio_chart_div.html"
CHART2_DIV = "dynamic_performance_chart_div.html"
OUT_HTML = "dashboard.html"

# -----------------------------
# Price loader (STATIC)
# -----------------------------

def load_price_table():
    """
    Read prices_close.csv produced by update_prices_tw_close.py
    Acceptable columns (either set is OK):
      A) symbol, date, last, prev, chg_abs, chg_pct   (✅ current)
      B) symbol, date, close, prev, chg_abs, chg_pct  (also ok)

    Returns a DataFrame with normalized columns:
      symbol, date, close, prev, chg_abs, chg_pct
    """
    import os
    import pandas as pd
    import numpy as np

    if not os.path.exists(PRICE_FILE):
        raise FileNotFoundError(f"Missing {PRICE_FILE}")

    df = pd.read_csv(PRICE_FILE)
    df.columns = [c.strip().lower() for c in df.columns]

    # --- required: symbol ---
    if "symbol" not in df.columns:
        raise ValueError(f"{PRICE_FILE} missing column: symbol. Got: {list(df.columns)}")
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper().str.replace(":", ".", regex=False)

    # --- date column might be "date" or "asof" ---
    if "date" not in df.columns:
        if "asof" in df.columns:
            df = df.rename(columns={"asof": "date"})
        else:
            # not fatal; still allow without date
            df["date"] = pd.NaT
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # --- price column might be "close" or "last" ---
    if "close" not in df.columns:
        if "last" in df.columns:
            df = df.rename(columns={"last": "close"})
        else:
            raise ValueError(f"{PRICE_FILE} missing close/last column. Got: {list(df.columns)}")

    # --- normalize numeric fields ---
    for col in ["close", "prev", "chg_abs", "chg_pct"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # keep only columns we care
    keep = ["symbol", "date", "close", "prev", "chg_abs", "chg_pct"]
    df = df[keep].copy()

    # if duplicates, keep latest by date if available
    if df["date"].notna().any():
        df = df.sort_values(["symbol", "date"]).drop_duplicates("symbol", keep="last")
    else:
        df = df.drop_duplicates("symbol", keep="last")

    return df


def build_price_map_last_prev(price_df: pd.DataFrame):
    out = {}
    for sym, d in price_df.sort_values("date").groupby("symbol"):
        if len(d) == 0:
            continue
        last = d.iloc[-1]["close"]
        prev = d.iloc[-2]["close"] if len(d) >= 2 else last
        out[sym] = {
            "last": float(last),
            "prev": float(prev),
            "chg_abs": float(last - prev),
            "chg_pct": float(last / prev - 1.0) if prev != 0 else np.nan,
            "last_date": d.iloc[-1]["date"]
        }
    return out


def get_last_market_date(price_df: pd.DataFrame):
    return price_df["date"].max().normalize()


# -----------------------------
# Formatting helpers
# -----------------------------
def _is_bad(x):
    return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))


def fmt_money(x, decimals=0):
    if _is_bad(x):
        return "—"
    return f"${x:,.{decimals}f}"


def fmt_money_signed(x, decimals=0):
    if _is_bad(x):
        return "—"
    sign = "+" if x >= 0 else "-"
    return f"{sign}${abs(x):,.{decimals}f}"


def fmt_pct_signed(x, decimals=2):
    if _is_bad(x):
        return "—"
    return f"{x*100:+.{decimals}f}%"


def is_pos(x):
    try:
        return float(x) >= 0
    except Exception:
        return False


# -----------------------------
# Load transactions
# -----------------------------
def load_transactions(path=TX_FILE):
    df = pd.read_excel(path)

    rename_map = {
        "Buy Time": "buy",
        "Sell Time": "sell",
        "Stock Symbol": "symbol",
        "Name": "name",
        "# of Shares": "shares",
        "Cost Basis": "cost",
        "Sell Price": "sell_price",
        "Status": "status",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "name" not in df.columns:
        df["name"] = ""

    df["buy"] = pd.to_datetime(df.get("buy"), errors="coerce")
    df["sell"] = pd.to_datetime(df.get("sell"), errors="coerce")
    df["shares"] = pd.to_numeric(df.get("shares"), errors="coerce")
    df["cost"] = pd.to_numeric(df.get("cost"), errors="coerce")
    df["sell_price"] = pd.to_numeric(df.get("sell_price"), errors="coerce")

    df["symbol"] = (
        df.get("symbol")
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\.TW$|\.TWO$", "", regex=True)
    )
    df["name"] = df.get("name").astype(str).str.strip()

    df["status"] = df.get("status").fillna("").astype(str).str.lower().str.strip()
    df.loc[df["status"].isin(["sold", "sell", "closed"]), "status"] = "sold"
    df.loc[df["status"].isin(["hold", "holding", "open"]), "status"] = "hold"
    df.loc[(df["status"] == "") & df["sell"].notna(), "status"] = "sold"
    df.loc[(df["status"] == "") & df["sell"].isna(), "status"] = "hold"

    df = df.dropna(subset=["buy", "symbol", "shares", "cost"])
    df = df[df["shares"] != 0]

    return df


# -----------------------------
# MAIN
# -----------------------------
def main():
    tx = load_transactions()
    prices = load_price_table()
    price_map = build_price_map_last_prev(prices)
    last_mkt_day = get_last_market_date(prices)

    tz_utc8 = timezone(timedelta(hours=8))
    now_utc8 = pd.Timestamp(datetime.now(tz_utc8))
    report_date_str = now_utc8.strftime("%m/%d/%Y")
    dashboard_title = f"Investment Dashboard as of {report_date_str}"

    # ----- holdings -----
    hold = tx[tx["status"] == "hold"].copy()
    rows = []

    for sym, d in hold.groupby("symbol"):
        info = price_map.get(sym)
        if not info:
            continue

        shares = d["shares"].sum()
        avg_cost = np.average(d["cost"], weights=d["shares"])
        last_px = info["last"]

        mv = shares * last_px
        cost_total = shares * avg_cost
        unreal = mv - cost_total

        rows.append({
            "symbol": sym,
            "name": d["name"].iloc[0],
            "shares": shares,
            "avg_cost": avg_cost,
            "last": last_px,
            "market_value": mv,
            "unreal": unreal
        })

    h = pd.DataFrame(rows)
    portfolio_mv = float(h["market_value"].sum()) if not h.empty else 0.0
    unreal_total = float(h["unreal"].sum()) if not h.empty else 0.0

    pill_label = f"Dashboard generated (UTC+8): {now_utc8.strftime('%Y-%m-%d %H:%M')}"

    chart1 = Path(CHART1_DIV).read_text(encoding="utf-8") if Path(CHART1_DIV).exists() else ""
    chart2 = Path(CHART2_DIV).read_text(encoding="utf-8") if Path(CHART2_DIV).exists() else ""

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{dashboard_title}</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
</head>
<body>
  <h1>{dashboard_title}</h1>
  <p>{pill_label}</p>

  <h2>Overview</h2>
  <p>Portfolio Market Value: {fmt_money(portfolio_mv)}</p>
  <p>Total Unrealized Return: {fmt_money_signed(unreal_total)}</p>

  <h2>Portfolio Value Trend</h2>
  {chart1}

  <h2>Performance Analysis vs. Benchmark</h2>
  {chart2}

</body>
</html>
"""
    Path(OUT_HTML).write_text(html, encoding="utf-8")
    print(f"Saved: {OUT_HTML}")


if __name__ == "__main__":
    main()
