# save_dynamic_performance_chart_for_dashboard.py
# Builds your Dynamic Performance % chart and SAVES it for reuse in a separate dashboard:
#   - dynamic_performance_chart.json
#   - dynamic_performance_chart_div.html
#
# Inputs:
#   - transactions.xlsx
#   - Funds Flow.xlsx
#
# Improvements:
#   - Auto normalize symbols: "2330" -> "2330.TW"
#   - TWII (^TWII) missing -> fallback to 0050.TW (and rename label)
#   - More robust yfinance parsing
#   - Never blank lines on first load (precomputed full-range %)

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.io as pio

TX_FILE = "transactions.xlsx"
FUND_FILE = "Funds Flow.xlsx"

BENCH_TWII = "^TWII"
BENCH_0050 = "0050.TW"

OUT_JSON = "dynamic_performance_chart.json"
OUT_DIV  = "dynamic_performance_chart_div.html"


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
    if s.isdigit():
        return f"{s}.TW"
    return s


# ----------------------
# Helpers
# ----------------------
def _as_1d(x, name="array"):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name} has {x.shape[1]} columns; expected 1.")
        x = x.iloc[:, 0]
    if isinstance(x, pd.Series):
        x = x.to_numpy()
    x = np.asarray(x)
    if x.ndim == 2:
        if x.shape[1] == 1:
            x = x[:, 0]
        else:
            raise ValueError(f"{name} is {x.shape}, expected 1D or Nx1.")
    return x.reshape(-1)


def yf_price_series(ticker, start, end):
    data = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        interval="1d",
    )
    if data is None or len(data) == 0:
        raise ValueError(f"No data for {ticker}")

    col = "Adj Close" if "Adj Close" in data.columns else ("Close" if "Close" in data.columns else None)
    if col is None:
        raise ValueError(f"No Close/Adj Close for {ticker}")

    s = data[col].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s.sort_index()
    if s.dropna().empty:
        raise ValueError(f"All NaN for {ticker}")
    return s


def yf_price_df(tickers, start, end):
    tickers = [t for t in sorted(set(tickers)) if isinstance(t, str) and t.strip()]
    if not tickers:
        raise ValueError("No tickers.")

    data = yf.download(
        " ".join(tickers),
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,
        group_by="column",
        interval="1d",
        threads=True,
    )
    if data is None or len(data) == 0:
        raise ValueError(f"No data for tickers: {tickers}")

    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = data.columns.get_level_values(0)
        if "Adj Close" in lvl0:
            px = data["Adj Close"].copy()
        elif "Close" in lvl0:
            px = data["Close"].copy()
        else:
            raise ValueError("Missing Close/Adj Close.")
    else:
        col = "Adj Close" if "Adj Close" in data.columns else ("Close" if "Close" in data.columns else None)
        if col is None:
            raise ValueError("Missing Close/Adj Close.")
        px = data[[col]].copy()
        px.columns = [tickers[0]]

    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    px = px.sort_index().dropna(how="all")
    if px.empty:
        raise ValueError("All NaN in price df.")
    return px


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
    df["buy"] = pd.to_datetime(df.get("buy"), errors="coerce")
    df["sell"] = pd.to_datetime(df.get("sell"), errors="coerce")
    df["shares"] = pd.to_numeric(df.get("shares"), errors="coerce")
    df["cost"] = pd.to_numeric(df.get("cost"), errors="coerce")
    df["sell_price"] = pd.to_numeric(df.get("sell_price"), errors="coerce")
    df["symbol"] = df.get("symbol").map(normalize_symbol)

    df["status"] = df.get("status").fillna("").astype(str).str.lower().str.strip()
    df.loc[df["status"].isin(["sold", "sell", "closed"]), "status"] = "sold"
    df.loc[df["status"].isin(["hold", "holding", "open"]), "status"] = "hold"
    df.loc[(df["status"] == "") & df["sell"].notna(), "status"] = "sold"
    df.loc[(df["status"] == "") & df["sell"].isna(), "status"] = "hold"

    df = df.dropna(subset=["buy", "symbol", "shares", "cost"])
    df = df[df["shares"] != 0]
    return df


def load_funds_flow(dates):
    ff = pd.read_excel(FUND_FILE)

    # Expect 3 columns: date / cash_inflow / cash_balance
    ff.columns = ["date", "cash_inflow", "cash_balance"]
    ff["date"] = pd.to_datetime(ff["date"], errors="coerce")
    ff["cash_balance"] = pd.to_numeric(ff["cash_balance"], errors="coerce")
    ff = ff.dropna(subset=["date", "cash_balance"]).set_index("date").sort_index()

    cash = ff["cash_balance"].reindex(dates).ffill()
    if cash.isna().all():
        raise ValueError("Funds Flow cash_balance all NaN after alignment. Check Funds Flow.xlsx dates.")
    if pd.isna(cash.iloc[0]):
        cash = cash.bfill()

    cash = cash.where((cash.notna()) & (cash != 0), np.nan).ffill().bfill()
    cash = cash.fillna(1e-9)
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
            pnl_series = sh * (prices[sym] - cost)
            net.loc[buy:] += pnl_series.loc[buy:]
    return net


def safe_left(v, fallback=1e-9):
    v = float(v) if pd.notna(v) else fallback
    if not np.isfinite(v) or v == 0:
        return fallback
    return v


# ----------------------
# MAIN
# ----------------------
def main():
    tx = load_transactions()
    start = tx["buy"].min().normalize()
    end = pd.Timestamp.today().normalize()
    dates = pd.date_range(start, end, freq="B")

    symbols = sorted(tx["symbol"].dropna().unique().tolist())
    prices = yf_price_df(symbols, start, end).reindex(dates).ffill()

    # remove columns that are all NaN
    good_cols = [c for c in prices.columns if prices[c].notna().any()]
    prices = prices[good_cols]

    # --- Benchmarks (TWII fallback to 0050)
    bench_name = "TWII %"
    try:
        twii_s = yf_price_series(BENCH_TWII, start, end).reindex(dates).ffill()
    except Exception as e:
        print(f"[WARN] {BENCH_TWII} failed ({e}). Fallback to {BENCH_0050}.")
        twii_s = yf_price_series(BENCH_0050, start, end).reindex(dates).ffill()
        bench_name = "TW Benchmark (0050) %"

    # Always also show 0050 line (if you want same line twice you can remove one)
    try:
        etf_s = yf_price_series(BENCH_0050, start, end).reindex(dates).ffill()
    except Exception:
        etf_s = twii_s.copy()

    net_s  = compute_net_return(tx, prices, dates)
    cash_s = load_funds_flow(dates)

    x_dates  = list(pd.to_datetime(dates).to_pydatetime())
    net_raw  = _as_1d(net_s,  "net_raw")
    cash_raw = _as_1d(cash_s, "cash_raw")
    twii_raw = _as_1d(twii_s, "twii_raw")
    etf_raw  = _as_1d(etf_s,  "etf_raw")

    # Precompute full-range % so it shows immediately
    i0 = 0
    net_left  = safe_left(net_raw[i0], 0.0)
    twii_left = safe_left(twii_raw[i0], 1e-9)
    etf_left  = safe_left(etf_raw[i0], 1e-9)

    cash_safe = np.array(cash_raw, dtype=float)
    cash_safe[~np.isfinite(cash_safe)] = np.nan
    cash_safe[cash_safe == 0] = np.nan
    s_cash = pd.Series(cash_safe).ffill().bfill().fillna(1e-9).to_numpy()

    net_pct0  = (net_raw - net_left) / s_cash
    twii_pct0 = (twii_raw / twii_left) - 1.0
    etf_pct0  = (etf_raw  / etf_left)  - 1.0

    fig = go.Figure()

    # raw (hidden) for JS dynamic rebase
    fig.add_trace(go.Scatter(x=x_dates, y=net_raw,  name="Net (raw)",  visible=False, customdata=cash_raw))
    fig.add_trace(go.Scatter(x=x_dates, y=twii_raw, name="Bench (raw)", visible=False))
    fig.add_trace(go.Scatter(x=x_dates, y=etf_raw,  name="0050 (raw)",  visible=False))

    # visible %
    fig.add_trace(go.Scatter(
        x=x_dates, y=net_pct0, name="Net Return %", mode="lines",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Net Return %: %{y:.2%}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=x_dates, y=twii_pct0, name=bench_name, mode="lines",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Benchmark %: %{y:.2%}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=x_dates, y=etf_pct0, name="0050 %", mode="lines",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>0050 %: %{y:.2%}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="Performance Analysis vs. Benchmark",
            x=0.5, xanchor="center",
            y=0.985, yanchor="top",
        ),
        template="plotly_white",
        hovermode="x unified",
        height=660,
        yaxis=dict(title="Performance", tickformat=".0%"),
        xaxis=dict(
            title_text=None,
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
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.28, yanchor="top"),
        margin=dict(t=85, b=180, l=60, r=40),
    )

    # JS: rebase + autoscale in dashboard
    post_script = r"""
(function() {
  var gd = document.getElementById('dynamic_perf_chart') || document.querySelector('div.js-plotly-plot');
  if(!gd) return;

  var inUpdate = false;

  function getXRangeMS() {
    var xa = gd.layout.xaxis || {};
    if (xa.range && xa.range.length === 2) {
      return [new Date(xa.range[0]).getTime(), new Date(xa.range[1]).getTime()];
    }
    var xs = gd.data[0].x;
    return [new Date(xs[0]).getTime(), new Date(xs[xs.length - 1]).getTime()];
  }

  function idxLeftEdge(ms0) {
    var xs = gd.data[0].x;
    for (var i = 0; i < xs.length; i++) {
      if (new Date(xs[i]).getTime() >= ms0) return i;
    }
    return 0;
  }

  function idxWindow(ms0, ms1) {
    var xs = gd.data[0].x;
    var i0 = 0, i1 = xs.length - 1;

    for (var i = 0; i < xs.length; i++) {
      var t = new Date(xs[i]).getTime();
      if (t >= ms0) { i0 = i; break; }
    }
    for (var j = xs.length - 1; j >= 0; j--) {
      var t2 = new Date(xs[j]).getTime();
      if (t2 <= ms1) { i1 = j; break; }
    }
    return [i0, i1];
  }

  function rebaseAndAutoscale() {
    if (inUpdate) return;
    inUpdate = true;

    try {
      var r = getXRangeMS();
      var ms0 = r[0], ms1 = r[1];

      var i0 = idxLeftEdge(ms0);

      var netRaw  = gd.data[0].y;
      var cash    = gd.data[0].customdata;
      var twiiRaw = gd.data[1].y;
      var etfRaw  = gd.data[2].y;

      var netLeft  = netRaw[i0];
      var twiiLeft = twiiRaw[i0];
      var etfLeft  = etfRaw[i0];

      if (!isFinite(twiiLeft) || twiiLeft === 0) twiiLeft = 1e-9;
      if (!isFinite(etfLeft)  || etfLeft  === 0) etfLeft  = 1e-9;

      var netPct = netRaw.map(function(v, i){
        var denom = cash[i];
        if (!isFinite(denom) || denom === 0) denom = 1e-9;
        return (v - netLeft) / denom;
      });

      var twiiPct = twiiRaw.map(function(v){ return (v / twiiLeft) - 1; });
      var etfPct  = etfRaw.map(function(v){ return (v / etfLeft)  - 1; });

      // visible traces are indices 3,4,5
      Plotly.restyle(gd, {'y': [netPct]},  [3]);
      Plotly.restyle(gd, {'y': [twiiPct]}, [4]);
      Plotly.restyle(gd, {'y': [etfPct]},  [5]);

      var win = idxWindow(ms0, ms1);
      var a = win[0], b = win[1];
      if (b <= a) return;

      var yMin = Infinity, yMax = -Infinity;
      for (var k = 3; k <= 5; k++) {
        var ys = (k === 3) ? netPct : (k === 4 ? twiiPct : etfPct);
        for (var ii = a; ii <= b; ii++) {
          var vv = ys[ii];
          if (vv === null || vv === undefined || !isFinite(vv)) continue;
          if (vv < yMin) yMin = vv;
          if (vv > yMax) yMax = vv;
        }
      }

      if (isFinite(yMin) && isFinite(yMax)) {
        var span = (yMax - yMin);
        if (span === 0) span = Math.abs(yMax) || 0.01;
        var pad = span * 0.12;
        Plotly.relayout(gd, {'yaxis.range': [yMin - pad, yMax + pad]});
      }
    } finally {
      setTimeout(function(){ inUpdate = false; }, 30);
    }
  }

  setTimeout(rebaseAndAutoscale, 120);
  gd.on('plotly_relayout', function() { setTimeout(rebaseAndAutoscale, 25); });
})();
"""

    pio.write_json(fig, OUT_JSON)

    div = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        div_id="dynamic_perf_chart",
        post_script=post_script
    )
    with open(OUT_DIV, "w", encoding="utf-8") as f:
        f.write(div)

    print(f"Saved:\n- {OUT_JSON}\n- {OUT_DIV}")


if __name__ == "__main__":
    main()
