import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from math import sqrt

# ==== CONFIG ====

CONFIG = {
    "tickers": ["BA", "INTC", "IBM", "PFE", "T", "VZ", "MMM", "KHC"],
    "benchmark": "SPY",
    "start_date": "1998-01-01",
    "end_date": None,
    "rsi_period": 14,
    "bb_window": 20,
    "bb_zscore_cap": 2.5,
    "ema_fast": 21,
    "ema_slow": 63,
    "sma_anchor": 200,
    "atr_period": 14,
    "mr_rsi_scale_center": 50,
    "mr_rsi_scale_range": 30,
    "trend_atr_norm_cap": 3.0,
    "signal_mr_weight": 0.5,
    "signal_trend_weight": 0.5,
    "signal_min_threshold": 0.05,
    "vol_window": 20,
    "vol_median_window": 252,
    "vol_scalar_floor": 0.2,
    "dd_full_deleverage": -0.10,
    "target_portfolio_vol": 0.10,
    "ticker_vol_span": 42,
    "max_per_ticker": 0.20,
    "max_gross_exposure": 1.0,
    "rebalance_threshold": 0.02,
    "cost_per_unit_turnover": 0.001,
    "wf_train_years": 5,
    "wf_test_months": 6,
}

# ==== DATA LOADING ====

def load_data(cfg):
    """Download OHLCV data for all tickers + benchmark."""
    all_tickers = cfg["tickers"] + [cfg["benchmark"]]
    end = cfg["end_date"] or datetime.today().strftime("%Y-%m-%d")

    raw = yf.download(all_tickers, start=cfg["start_date"], end=end, progress=False)

    ticker_data = {}
    for t in all_tickers:
        try:
            tdf = pd.DataFrame()
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                tdf[col] = raw[(col, t)]
            tdf.dropna(subset=["Close"], inplace=True)
            if len(tdf) > 252:
                ticker_data[t] = tdf
        except (KeyError, TypeError):
            print(f"  [WARN] Could not load data for {t}, skipping.")

    return ticker_data


def align_dates(ticker_data):
    """Find common DatetimeIndex intersection, reindex, forward-fill up to 5 days."""
    if not ticker_data:
        return ticker_data

    common_index = None
    for t, df in ticker_data.items():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)

    aligned = {}
    for t, df in ticker_data.items():
        reindexed = df.reindex(common_index)
        reindexed = reindexed.ffill(limit=5)
        reindexed = reindexed.dropna(subset=["Close"])
        aligned[t] = reindexed

    # Re-intersect after dropna
    final_index = None
    for t, df in aligned.items():
        if final_index is None:
            final_index = df.index
        else:
            final_index = final_index.intersection(df.index)

    if final_index is not None:
        for t in aligned:
            aligned[t] = aligned[t].reindex(final_index)

    return aligned

# ==== INDICATORS ====

def compute_indicators(df, cfg):
    """Add indicator columns to ticker DataFrame."""
    out = df.copy()

    # RSI_14: Wilder RSI using ewm
    period = cfg["rsi_period"]
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

    # BB_zscore: (Close - SMA) / rolling_std
    bw = cfg["bb_window"]
    sma = out["Close"].rolling(bw, min_periods=bw).mean()
    std = out["Close"].rolling(bw, min_periods=bw).std()
    out["BB_zscore"] = (out["Close"] - sma) / std.replace(0, np.nan)

    # EMA_fast, EMA_slow
    out["EMA_fast"] = out["Close"].ewm(span=cfg["ema_fast"], adjust=False).mean()
    out["EMA_slow"] = out["Close"].ewm(span=cfg["ema_slow"], adjust=False).mean()

    # SMA_anchor
    sa = cfg["sma_anchor"]
    out["SMA_anchor"] = out["Close"].rolling(sa, min_periods=sa).mean()

    # ATR_14: EWM ATR
    atr_span = cfg["atr_period"]
    high = out["High"]
    low = out["Low"]
    prev_close = out["Close"].shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    out["ATR_14"] = tr.ewm(span=atr_span, adjust=False).mean()

    # Daily_Return
    out["Daily_Return"] = out["Close"].pct_change()

    # Realized_Vol: rolling annualized std
    vw = cfg["vol_window"]
    out["Realized_Vol"] = out["Daily_Return"].rolling(vw, min_periods=vw).std() * sqrt(252)

    # Vol_Median: rolling median of Realized_Vol
    vmw = cfg["vol_median_window"]
    out["Vol_Median"] = out["Realized_Vol"].rolling(vmw, min_periods=60).median()

    # Ticker_Vol: EWM annualized std
    out["Ticker_Vol"] = out["Daily_Return"].ewm(span=cfg["ticker_vol_span"], adjust=False).std() * sqrt(252)

    return out

# ==== SIGNALS ====

def compute_signals(df, cfg):
    """Add signal columns to ticker DataFrame."""
    out = df.copy()

    # sig_mr: mean reversion, long-only, range [0,1]
    bb_z = out["BB_zscore"]
    sig_mr_raw = (-bb_z / cfg["bb_zscore_cap"]).clip(0, 1)
    rsi = out["RSI_14"]
    mr_rsi_boost = ((cfg["mr_rsi_scale_center"] - rsi) / cfg["mr_rsi_scale_range"]).clip(0, 1)
    sig_mr = sig_mr_raw * (0.5 + 0.5 * mr_rsi_boost)
    out["sig_mr"] = sig_mr.fillna(0)

    # sig_trend: trend-following, long-only above 200 SMA, range [0,1]
    trend_raw = (out["EMA_fast"] - out["EMA_slow"]) / out["ATR_14"].replace(0, np.nan)
    trend_normalized = (trend_raw / cfg["trend_atr_norm_cap"]).clip(-1, 1)
    above_anchor = (out["Close"] > out["SMA_anchor"]).astype(float)
    sig_trend = (trend_normalized * above_anchor).clip(0, 1)
    out["sig_trend"] = sig_trend.fillna(0)

    # sig_composite
    sig_composite = cfg["signal_mr_weight"] * out["sig_mr"] + cfg["signal_trend_weight"] * out["sig_trend"]
    out["sig_composite"] = sig_composite.fillna(0)

    return out

# ==== REGIME / SCALARS ====

def compute_vol_scalar(df, cfg):
    """Compute volatility scalar Series from ticker DataFrame."""
    vol_ratio = df["Realized_Vol"] / df["Vol_Median"].replace(0, np.nan)
    vol_scalar = (2.0 - vol_ratio).clip(cfg["vol_scalar_floor"], 1.0)
    return vol_scalar.fillna(cfg["vol_scalar_floor"])


def compute_dd_scalar(equity, cfg):
    """Compute drawdown-based deleveraging scalar."""
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    dd_scalar = (1.0 + dd / abs(cfg["dd_full_deleverage"])).clip(0.0, 1.0)
    return dd_scalar

# ==== SIZING ====

def compute_target_weights(ticker_signals, cfg):
    """
    Compute target weights DataFrame from ticker_signals dict.
    ticker_signals: dict of ticker -> DataFrame with sig_composite and Ticker_Vol columns.
    Returns DataFrame indexed by dates, columns=tickers.
    """
    tickers = list(ticker_signals.keys())

    # Build sig_df and vol_df
    sig_parts = {}
    vol_parts = {}
    for t in tickers:
        sig_parts[t] = ticker_signals[t]["sig_composite"]
        vol_parts[t] = ticker_signals[t]["Ticker_Vol"]

    sig_df = pd.DataFrame(sig_parts).fillna(0)
    vol_df = pd.DataFrame(vol_parts)

    # Vectorized weight computation
    n_active = (sig_df > cfg["signal_min_threshold"]).sum(axis=1)
    per_vol_budget = cfg["target_portfolio_vol"] / np.sqrt(n_active.clip(lower=1))

    # raw_weight = per_vol_budget / vol (broadcast)
    vol_df_safe = vol_df.replace(0, np.nan)
    raw_weight = per_vol_budget.values.reshape(-1, 1) / vol_df_safe
    raw_weight = raw_weight.fillna(0)

    # signal_weight
    signal_weight = raw_weight * sig_df

    # Clip each cell to [0, max_per_ticker]
    signal_weight = signal_weight.clip(0, cfg["max_per_ticker"])

    # Rescale rows where row sum > max_gross_exposure
    row_sum = signal_weight.sum(axis=1)
    scale_mask = row_sum > cfg["max_gross_exposure"]
    scale_factor = cfg["max_gross_exposure"] / row_sum.where(scale_mask, other=1.0)
    signal_weight = signal_weight.multiply(scale_factor, axis=0)

    return signal_weight.fillna(0)

# ==== PORTFOLIO ====

def construct_portfolio(target_weights, ticker_data, cfg):
    """
    Apply rebalance threshold drift logic.
    Returns (actual_weights_df, turnover_series).
    """
    tickers = list(target_weights.columns)
    dates = target_weights.index

    # Build ticker_returns DataFrame
    returns_parts = {}
    for t in tickers:
        if t in ticker_data:
            returns_parts[t] = ticker_data[t]["Close"].pct_change()
        else:
            returns_parts[t] = pd.Series(0.0, index=dates)

    ticker_returns = pd.DataFrame(returns_parts).reindex(dates).fillna(0)

    actual_weights = pd.DataFrame(0.0, index=dates, columns=tickers)
    turnover = np.zeros(len(dates))

    actual_weights.iloc[0] = target_weights.iloc[0].values

    for i in range(1, len(dates)):
        prev_w = actual_weights.iloc[i - 1].values
        ret_t = ticker_returns.iloc[i].values

        port_ret_t = float(np.dot(prev_w, ret_t))
        denominator = 1.0 + port_ret_t
        if abs(denominator) < 1e-10:
            denominator = 1.0

        drift = prev_w * (1.0 + ret_t) / denominator

        proposed = target_weights.iloc[i].values
        diff = np.abs(proposed - drift)
        new_w = np.where(diff > cfg["rebalance_threshold"], proposed, drift)
        actual_weights.iloc[i] = new_w
        turnover[i] = float(np.abs(new_w - drift).sum())

    return actual_weights, pd.Series(turnover, index=dates)

# ==== COSTS ====

def compute_costs(turnover, cfg):
    """Compute daily transaction costs from turnover."""
    return turnover * cfg["cost_per_unit_turnover"]

# ==== TRADE LOG ====

def build_trade_log(adjusted_weights, ticker_data):
    """Build a trade log DataFrame from adjusted_weights and ticker_data."""
    trades = []

    for ticker in adjusted_weights.columns:
        weights = adjusted_weights[ticker]
        price_series = ticker_data.get(ticker, pd.DataFrame())
        if isinstance(price_series, pd.DataFrame) and "Close" in price_series.columns:
            prices = price_series["Close"]
        else:
            prices = pd.Series(dtype=float)

        in_trade = False
        entry_date = None
        entry_price = None
        max_w = 0.0

        for date in adjusted_weights.index:
            w = float(weights[date]) if date in weights.index else 0.0
            p = float(prices[date]) if date in prices.index else np.nan

            if not in_trade and w > 0.01:
                in_trade = True
                entry_date = date
                entry_price = p
                max_w = w

            elif in_trade:
                if w > max_w:
                    max_w = w
                if w <= 0.01:
                    exit_date = date
                    exit_price = p
                    if entry_price is not None and not np.isnan(entry_price) and entry_price != 0:
                        pnl = (exit_price - entry_price) / entry_price
                    else:
                        pnl = np.nan
                    holding_days = (exit_date - entry_date).days
                    trades.append({
                        "ticker": ticker,
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "holding_days": holding_days,
                        "pnl": pnl,
                        "max_weight": max_w,
                    })
                    in_trade = False
                    entry_date = None
                    entry_price = None
                    max_w = 0.0

        # Close open trade at end
        if in_trade:
            last_date = adjusted_weights.index[-1]
            last_price = float(prices[last_date]) if last_date in prices.index else np.nan
            if entry_price is not None and not np.isnan(entry_price) and entry_price != 0:
                pnl = (last_price - entry_price) / entry_price
            else:
                pnl = np.nan
            holding_days = (last_date - entry_date).days
            trades.append({
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": last_date,
                "entry_price": entry_price,
                "exit_price": last_price,
                "holding_days": holding_days,
                "pnl": pnl,
                "max_weight": max_w,
            })

    if not trades:
        return pd.DataFrame(columns=["ticker", "entry_date", "exit_date",
                                     "entry_price", "exit_price",
                                     "holding_days", "pnl", "max_weight"])

    return pd.DataFrame(trades)

# ==== BACKTEST ====

def _rescale_weights(w_df, cfg):
    """Clip to max_per_ticker and rescale rows exceeding max_gross_exposure."""
    w_df = w_df.clip(0, cfg["max_per_ticker"])
    row_sum = w_df.sum(axis=1)
    scale_mask = row_sum > cfg["max_gross_exposure"]
    scale_factor = cfg["max_gross_exposure"] / row_sum.where(scale_mask, other=1.0)
    w_df = w_df.multiply(scale_factor, axis=0)
    return w_df.fillna(0)


def run_backtest(ticker_data, ticker_signals, cfg):
    """Run full backtest and return results dict."""

    # Step 1: compute target weights
    target_weights = compute_target_weights(ticker_signals, cfg)

    # Step 2: compute avg vol_scalar across tickers and apply to weights
    vol_scalar_parts = {}
    for t in ticker_signals:
        vol_scalar_parts[t] = compute_vol_scalar(ticker_signals[t], cfg)
    vol_scalar_df = pd.DataFrame(vol_scalar_parts).reindex(target_weights.index)
    avg_vol_scalar = vol_scalar_df.mean(axis=1).fillna(cfg["vol_scalar_floor"])

    target_weights = target_weights.multiply(avg_vol_scalar, axis=0)
    target_weights = _rescale_weights(target_weights, cfg)

    # Step 3: construct portfolio with drift/rebalance logic
    actual_weights, turnover = construct_portfolio(target_weights, ticker_data, cfg)

    # Step 4: costs
    costs = compute_costs(turnover, cfg)

    # Step 5: ticker returns
    tickers = list(actual_weights.columns)
    returns_parts = {}
    for t in tickers:
        if t in ticker_data:
            returns_parts[t] = ticker_data[t]["Close"].pct_change()
        else:
            returns_parts[t] = pd.Series(0.0, index=actual_weights.index)
    ticker_returns = pd.DataFrame(returns_parts).reindex(actual_weights.index).fillna(0)

    # Step 6: gross returns
    shifted_w = actual_weights.shift(1).fillna(0)
    gross_ret = (shifted_w * ticker_returns).sum(axis=1)

    # Step 7: net_ret and equity
    net_ret = gross_ret - costs
    equity = (1 + net_ret).cumprod()

    # Step 8: dd_scalar
    dd_scalar = compute_dd_scalar(equity, cfg)

    # Step 9: second pass - apply dd_scalar
    adjusted_weights = actual_weights.multiply(dd_scalar, axis=0)
    adjusted_weights = _rescale_weights(adjusted_weights, cfg)

    extra_turnover = (adjusted_weights - actual_weights).abs().sum(axis=1)
    total_turnover = turnover + extra_turnover
    costs2 = compute_costs(total_turnover, cfg)

    shifted_adj = adjusted_weights.shift(1).fillna(0)
    gross_ret2 = (shifted_adj * ticker_returns).sum(axis=1)
    net_ret2 = gross_ret2 - costs2
    equity2 = (1 + net_ret2).cumprod()

    # Step 10: SPY benchmark
    spy_close = ticker_data[cfg["benchmark"]]["Close"]
    spy_ret = spy_close.pct_change().fillna(0)
    spy_equity = (1 + spy_ret).cumprod()

    # Align to common index
    common_idx = equity2.index.intersection(spy_equity.index)
    net_ret2 = net_ret2.reindex(common_idx).fillna(0)
    equity2 = equity2.reindex(common_idx)
    spy_ret = spy_ret.reindex(common_idx).fillna(0)
    spy_equity = spy_equity.reindex(common_idx)

    # Step 11 & 12: risk_scalar = avg_vol_scalar * dd_scalar
    avg_vol_scalar_aligned = avg_vol_scalar.reindex(common_idx).fillna(cfg["vol_scalar_floor"])
    dd_scalar_aligned = dd_scalar.reindex(common_idx).fillna(1.0)
    risk_scalar = avg_vol_scalar_aligned * dd_scalar_aligned

    # Step 13: trade log
    trade_log = build_trade_log(adjusted_weights, ticker_data)

    return {
        "portfolio_ret": net_ret2,
        "portfolio_equity": equity2,
        "spy_ret": spy_ret,
        "spy_equity": spy_equity,
        "actual_weights": actual_weights,
        "adjusted_weights": adjusted_weights,
        "turnover": total_turnover.reindex(common_idx).fillna(0),
        "costs": costs2.reindex(common_idx).fillna(0),
        "vol_scalar": avg_vol_scalar_aligned,
        "dd_scalar": dd_scalar_aligned,
        "risk_scalar": risk_scalar,
        "trade_log": trade_log,
        "ticker_signals": ticker_signals,
    }

# ==== METRICS ====

def sharpe_ratio(returns, periods=252):
    """Annualized Sharpe ratio."""
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(returns.mean() * periods / (std * sqrt(periods)))


def sortino_ratio(returns, periods=252):
    """Annualized Sortino ratio."""
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 0.0
    ds = downside.std()
    if ds == 0 or np.isnan(ds):
        return 0.0
    return float(returns.mean() * periods / (ds * sqrt(periods)))


def max_drawdown(equity):
    """Maximum drawdown as a negative fraction."""
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return float(dd.min())


def drawdown_series(equity):
    """Drawdown series as fraction."""
    peak = equity.cummax()
    return (equity - peak) / peak.replace(0, np.nan)


def cagr(equity, periods_per_year=252):
    """Compound annual growth rate."""
    n = len(equity)
    if n < 2:
        return 0.0
    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])
    if start <= 0 or end <= 0:
        return 0.0
    years = n / periods_per_year
    if years <= 0:
        return 0.0
    return float((end / start) ** (1.0 / years) - 1.0)


def calmar_ratio(equity):
    """Calmar ratio: CAGR / abs(max_drawdown)."""
    mdd = max_drawdown(equity)
    if mdd == 0 or np.isnan(mdd):
        return 0.0
    return float(cagr(equity) / abs(mdd))


def profit_factor(returns):
    """Sum of positive returns / abs(sum of negative returns)."""
    pos = returns[returns > 0].sum()
    neg = returns[returns < 0].sum()
    if neg == 0 or np.isnan(neg):
        return float("inf") if pos > 0 else 0.0
    return float(pos / abs(neg))


def compute_all_metrics(portfolio_ret, portfolio_equity, spy_ret, spy_equity,
                        turnover, costs, trade_log, adjusted_weights):
    """Compute all performance metrics."""
    n_days = len(portfolio_ret)

    # Portfolio metrics
    sh = sharpe_ratio(portfolio_ret)
    so = sortino_ratio(portfolio_ret)
    cg = cagr(portfolio_equity)
    mdd = max_drawdown(portfolio_equity)
    cal = calmar_ratio(portfolio_equity)
    pf = profit_factor(portfolio_ret)

    # SPY metrics
    spy_sh = sharpe_ratio(spy_ret)
    spy_cg = cagr(spy_equity)
    spy_mdd = max_drawdown(spy_equity)

    # Cost and turnover
    annual_turnover = turnover.sum() / (n_days / 252.0) if n_days > 0 else 0.0
    total_costs_val = costs.sum()
    cost_drag_annual = total_costs_val / (n_days / 252.0) if n_days > 0 else 0.0

    # Average exposure
    avg_exposure = float(adjusted_weights.sum(axis=1).mean())

    # Trade statistics from trade_log
    if trade_log is not None and len(trade_log) > 0 and "pnl" in trade_log.columns:
        valid_trades = trade_log.dropna(subset=["pnl"])
        total_trades = len(valid_trades)
        wins = valid_trades[valid_trades["pnl"] > 0]
        losses = valid_trades[valid_trades["pnl"] <= 0]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        avg_win = float(wins["pnl"].mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses["pnl"].mean()) if len(losses) > 0 else 0.0
        avg_holding_days = float(valid_trades["holding_days"].mean()) if total_trades > 0 else 0.0
        best_trade = float(valid_trades["pnl"].max()) if total_trades > 0 else 0.0
        worst_trade = float(valid_trades["pnl"].min()) if total_trades > 0 else 0.0

        # Max consecutive losses
        sorted_trades = valid_trades.sort_values("exit_date")
        streak = 0
        max_streak = 0
        for p in sorted_trades["pnl"]:
            if p <= 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        max_consec_losses = max_streak
    else:
        total_trades = 0
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        avg_holding_days = 0.0
        best_trade = 0.0
        worst_trade = 0.0
        max_consec_losses = 0

    final_equity = float(portfolio_equity.iloc[-1]) if len(portfolio_equity) > 0 else 1.0
    spy_final_equity = float(spy_equity.iloc[-1]) if len(spy_equity) > 0 else 1.0

    return {
        "sharpe": sh,
        "sortino": so,
        "cagr": cg,
        "max_drawdown": mdd,
        "calmar": cal,
        "profit_factor": pf,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_holding_days": avg_holding_days,
        "max_consec_losses": max_consec_losses,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "annual_turnover": annual_turnover,
        "total_costs": total_costs_val,
        "cost_drag_annual": cost_drag_annual,
        "spy_sharpe": spy_sh,
        "spy_cagr": spy_cg,
        "spy_max_drawdown": spy_mdd,
        "avg_exposure": avg_exposure,
        "final_equity": final_equity,
        "spy_final_equity": spy_final_equity,
    }

# ==== REPORTING ====

def print_report(met, results, cfg):
    """Print the main portfolio report."""
    print("=" * 70)
    print("  TRADER v5.0 — PORTFOLIO REPORT")
    print("=" * 70)
    print()
    print(f"{'':20s}{'PORTFOLIO':>16s}{'SPY B&H':>16s}")
    print(f"  Sharpe..........  {met['sharpe']:>10.3f}     {met['spy_sharpe']:>10.3f}")
    print(f"  Sortino.........  {met['sortino']:>10.3f}")
    print(f"  Calmar..........  {met['calmar']:>10.3f}")
    print(f"  CAGR............  {met['cagr']:>9.2%}      {met['spy_cagr']:>9.2%}")
    print(f"  Max Drawdown....  {met['max_drawdown']:>9.2%}      {met['spy_max_drawdown']:>9.2%}")
    print(f"  Profit Factor...  {met['profit_factor']:>10.2f}")
    print(f"  Final Equity....  {met['final_equity']:>9.3f}x     {met['spy_final_equity']:>9.3f}x")
    print()
    print("-" * 70)
    print("  COST & TURNOVER")
    print("-" * 70)
    print(f"  Annual Turnover:        {met['annual_turnover']:.1%}")
    print(f"  Total Cost Drag:        {met['cost_drag_annual']:.2%}/yr")
    print(f"  Avg Exposure:           {met['avg_exposure']:.1%}")
    print()
    print("-" * 70)
    print("  RISK SCALAR DISTRIBUTION")
    print("-" * 70)
    rs = results["risk_scalar"].dropna()
    if len(rs) > 0:
        p_high = (rs > 0.8).mean()
        p_mod = ((rs >= 0.5) & (rs <= 0.8)).mean()
        p_red = ((rs >= 0.2) & (rs < 0.5)).mean()
        p_off = (rs < 0.2).mean()
    else:
        p_high = p_mod = p_red = p_off = 0.0
    print(f"  >0.8 (high risk-on):    {p_high:.1%}")
    print(f"  0.5-0.8 (moderate):     {p_mod:.1%}")
    print(f"  0.2-0.5 (reduced):      {p_red:.1%}")
    print(f"  <0.2 (risk-off):        {p_off:.1%}")
    print()
    print("-" * 70)
    print("  TRADE STATISTICS")
    print("-" * 70)
    print(f"  Total Trades:           {met['total_trades']}")
    print(f"  Win Rate:               {met['win_rate']:.2%}")
    print(f"  Avg Win:                {met['avg_win']:.4f}")
    print(f"  Avg Loss:               {met['avg_loss']:.4f}")
    print(f"  Avg Holding (days):     {met['avg_holding_days']:.1f}")
    print(f"  Max Consec Losses:      {met['max_consec_losses']}")
    print(f"  Best Trade:             {met['best_trade']:.4f}")
    print(f"  Worst Trade:            {met['worst_trade']:.4f}")
    print("=" * 70)


def print_per_ticker_metrics(ticker_signals, ticker_data, adjusted_weights):
    """Print per-ticker summary table."""
    print()
    print("-" * 70)
    print("  PER-TICKER SUMMARY")
    print("-" * 70)
    print(f"  {'Ticker':<8} {'Avg Sig':>8} {'Avg Wt':>8} {'Trades':>8} {'Contribution':>14}")

    for ticker in ticker_signals:
        avg_sig = float(ticker_signals[ticker]["sig_composite"].mean())

        if ticker in adjusted_weights.columns:
            avg_wt = float(adjusted_weights[ticker].mean())
        else:
            avg_wt = 0.0

        # Count trade periods
        if ticker in adjusted_weights.columns:
            w = adjusted_weights[ticker]
            in_t = (w > 0.01).astype(int)
            transitions = in_t.diff().fillna(0)
            n_trades = int((transitions == 1).sum())
        else:
            n_trades = 0

        # Contribution: weighted return CAGR
        contribution = 0.0
        if ticker in adjusted_weights.columns and ticker in ticker_data:
            price_ret = ticker_data[ticker]["Close"].pct_change().fillna(0)
            w_shift = adjusted_weights[ticker].shift(1).fillna(0)
            common = w_shift.index.intersection(price_ret.index)
            if len(common) > 1:
                weighted_ret = w_shift.reindex(common) * price_ret.reindex(common)
                contrib_equity = (1 + weighted_ret).cumprod()
                n = len(contrib_equity)
                if n > 1:
                    start = float(contrib_equity.iloc[0])
                    end = float(contrib_equity.iloc[-1])
                    if start > 0 and end > 0:
                        years = n / 252.0
                        if years > 0:
                            contribution = (end / start) ** (1.0 / years) - 1.0

        print(f"  {ticker:<8} {avg_sig:>8.3f} {avg_wt:>7.2%} {n_trades:>8} {contribution:>12.2%}/yr")

    print("-" * 70)

# ==== PLOTTING ====

def plot_portfolio_overview(results):
    """Plot portfolio equity, drawdown, and risk scalar."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Subplot 1: Portfolio equity vs SPY
    ax = axes[0]
    ax.plot(results["portfolio_equity"].index, results["portfolio_equity"].values,
            color="black", lw=1.5, label="Portfolio")
    ax.plot(results["spy_equity"].index, results["spy_equity"].values,
            color="gray", lw=1.2, linestyle="--", label="SPY B&H")
    ax.set_title("Portfolio Equity vs SPY")
    ax.set_ylabel("Equity (x)")
    ax.legend()

    # Subplot 2: Drawdown
    ax = axes[1]
    dd = drawdown_series(results["portfolio_equity"])
    ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.4)
    ax.axhline(-0.10, color="darkred", linestyle="--", lw=1.2, label="Deleverage Threshold (-10%)")
    ax.set_title("Portfolio Drawdown")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=8)

    # Subplot 3: Risk scalar
    ax = axes[2]
    ax.plot(results["risk_scalar"].index, results["risk_scalar"].values,
            color="darkblue", lw=1.0)
    ax.axhline(0.5, color="gray", linestyle=":", lw=1.0)
    ax.set_ylim(0, 1.1)
    ax.set_title("Risk Scalar Over Time")
    ax.set_ylabel("Risk Scalar")

    fig.suptitle("Trader v5.0 — Portfolio Overview")
    plt.tight_layout()
    return fig


def plot_per_ticker_grid(results):
    """Plot per-ticker price and allocation grid."""
    ticker_signals = results["ticker_signals"]
    adjusted_weights = results["adjusted_weights"]
    tickers = list(ticker_signals.keys())

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    axes_flat = axes.flatten()

    for idx, ticker in enumerate(tickers[:8]):
        ax = axes_flat[idx]

        avg_sig = float(ticker_signals[ticker]["sig_composite"].mean())
        if ticker in adjusted_weights.columns:
            avg_wt = float(adjusted_weights[ticker].mean())
        else:
            avg_wt = 0.0

        # Plot Close on primary axis
        close = ticker_signals[ticker]["Close"]
        ax.plot(close.index, close.values, color="gray", lw=0.8, alpha=0.7)
        ax.set_ylabel("Price", fontsize=8)

        # Plot adjusted weight on secondary axis
        ax2 = ax.twinx()
        if ticker in adjusted_weights.columns:
            w = adjusted_weights[ticker]
            ax2.plot(w.index, w.values, color="black", lw=1.2)
        ax2.set_ylabel("Weight", fontsize=8)
        ax2.set_ylim(0, 0.25)

        ax.set_title(f"{ticker} | Avg Wt: {avg_wt:.1%} | Avg Sig: {avg_sig:.2f}", fontsize=9)

    # Hide unused subplots
    for idx in range(len(tickers), 8):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Trader v5.0 — Per-Ticker Allocation")
    plt.tight_layout()
    return fig


def plot_strategy_analysis(results):
    """Plot signal decomposition, avg allocation, rolling Sharpe."""
    ticker_signals = results["ticker_signals"]
    adjusted_weights = results["adjusted_weights"]
    portfolio_ret = results["portfolio_ret"]
    spy_ret = results["spy_ret"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Subplot 1: Signal decomposition for first ticker
    first_ticker = list(ticker_signals.keys())[0]
    ax = axes[0]
    sig_df = ticker_signals[first_ticker]
    ax.plot(sig_df.index, sig_df["sig_mr"].values, color="blue", lw=1.0, label="sig_mr")
    ax.plot(sig_df.index, sig_df["sig_trend"].values, color="orange", lw=1.0, label="sig_trend")
    ax.set_ylabel("Signal Strength")
    ax.set_title(f"Signal Decomposition ({first_ticker})")
    ax.legend()

    # Subplot 2: Avg allocation by ticker
    ax = axes[1]
    ticker_list = list(ticker_signals.keys())
    avg_wts = []
    for t in ticker_list:
        if t in adjusted_weights.columns:
            avg_wts.append(float(adjusted_weights[t].mean()))
        else:
            avg_wts.append(0.0)
    ax.bar(ticker_list, avg_wts, color="green")
    ax.set_title("Avg Allocation by Ticker")
    ax.set_ylabel("Avg Weight")
    ax.tick_params(axis="x", rotation=45)

    # Subplot 3: Rolling 120-day Sharpe
    ax = axes[2]
    window = 120
    if len(portfolio_ret) > window:
        roll_sharpe_port = portfolio_ret.rolling(window).apply(
            lambda x: x.mean() * 252 / (x.std() * sqrt(252)) if x.std() > 0 else 0.0,
            raw=True
        )
        roll_sharpe_spy = spy_ret.rolling(window).apply(
            lambda x: x.mean() * 252 / (x.std() * sqrt(252)) if x.std() > 0 else 0.0,
            raw=True
        )
        ax.plot(roll_sharpe_port.index, roll_sharpe_port.values,
                color="black", lw=1.0, label="Portfolio")
        ax.plot(roll_sharpe_spy.index, roll_sharpe_spy.values,
                color="gray", lw=1.0, linestyle="--", label="SPY")
    ax.axhline(0, color="lightgray", lw=1.0)
    ax.set_title("Rolling 120-day Sharpe")
    ax.set_ylabel("Sharpe")
    ax.legend()

    fig.suptitle("Trader v5.0 — Strategy Analysis")
    plt.tight_layout()
    return fig

# ==== VALIDATION ====

def walk_forward_test(ticker_data, ticker_signals_full, cfg):
    """Walk-forward validation with expanding training window."""
    first_ticker = list(ticker_signals_full.keys())[0]
    dates = ticker_signals_full[first_ticker].index

    window_results = []
    oos_rets = []

    train_start = dates[0]

    while True:
        train_end_date = train_start + pd.DateOffset(years=cfg["wf_train_years"])
        test_start_date = train_end_date
        test_end_date = test_start_date + pd.DateOffset(months=cfg["wf_test_months"])

        if test_end_date > dates[-1]:
            break

        # Subset ticker_data for test period
        test_ticker_data = {}
        for t, df in ticker_data.items():
            if t == cfg["benchmark"]:
                continue
            subset = df.loc[test_start_date:test_end_date]
            test_ticker_data[t] = subset

        test_ticker_data[cfg["benchmark"]] = ticker_data[cfg["benchmark"]].loc[test_start_date:test_end_date]

        # Subset ticker_signals for test period
        test_ticker_signals = {}
        for t, df in ticker_signals_full.items():
            subset = df.loc[test_start_date:test_end_date]
            test_ticker_signals[t] = subset

        # Skip if too few rows
        skip = False
        for t, df in test_ticker_signals.items():
            if len(df) < 20:
                skip = True
                break
        for t, df in test_ticker_data.items():
            if len(df) < 20:
                skip = True
                break

        if skip:
            train_start += pd.DateOffset(months=cfg["wf_test_months"])
            continue

        try:
            res = run_backtest(test_ticker_data, test_ticker_signals, cfg)
            met = compute_all_metrics(
                res["portfolio_ret"], res["portfolio_equity"],
                res["spy_ret"], res["spy_equity"],
                res["turnover"], res["costs"],
                res["trade_log"], res["adjusted_weights"]
            )

            window_results.append({
                "train_end": train_end_date,
                "test_start": test_start_date,
                "test_end": test_end_date,
                "sharpe": met["sharpe"],
                "cagr": met["cagr"],
                "max_drawdown": met["max_drawdown"],
                "n_days": len(res["portfolio_ret"]),
            })

            oos_rets.append(res["portfolio_ret"])
        except Exception as e:
            print(f"  [WARN] Walk-forward window failed: {e}")

        train_start += pd.DateOffset(months=cfg["wf_test_months"])

    # Aggregate OOS results
    if oos_rets:
        oos_ret_combined = pd.concat(oos_rets).sort_index()
        oos_equity = (1 + oos_ret_combined).cumprod()
        oos_sharpe = sharpe_ratio(oos_ret_combined)
        oos_cagr = cagr(oos_equity)
        oos_mdd = max_drawdown(oos_equity)
    else:
        oos_ret_combined = pd.Series(dtype=float)
        oos_equity = pd.Series(dtype=float)
        oos_sharpe = 0.0
        oos_cagr = 0.0
        oos_mdd = 0.0

    # Sharpe stability
    if window_results:
        w_sharpes = [w["sharpe"] for w in window_results]
        mean_sh = float(np.mean(w_sharpes))
        std_sh = float(np.std(w_sharpes))
        sharpe_stability = std_sh / abs(mean_sh) if mean_sh != 0 else float("nan")
    else:
        sharpe_stability = float("nan")

    return {
        "window_results": window_results,
        "oos_ret": oos_ret_combined,
        "oos_equity": oos_equity,
        "oos_sharpe": oos_sharpe,
        "oos_cagr": oos_cagr,
        "oos_max_drawdown": oos_mdd,
        "sharpe_stability": sharpe_stability,
    }


def print_validation_report(val_results):
    """Print walk-forward validation results."""
    print()
    print("=" * 70)
    print("  WALK-FORWARD VALIDATION REPORT")
    print("=" * 70)
    print()
    windows = val_results.get("window_results", [])
    if windows:
        print(f"  {'Window':>6}  {'Test Start':>12}  {'Test End':>12}  {'Sharpe':>8}  {'CAGR':>8}  {'MDD':>8}")
        print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}")
        for i, w in enumerate(windows):
            ts = w["test_start"].strftime("%Y-%m-%d") if hasattr(w["test_start"], "strftime") else str(w["test_start"])[:10]
            te = w["test_end"].strftime("%Y-%m-%d") if hasattr(w["test_end"], "strftime") else str(w["test_end"])[:10]
            print(f"  {i+1:>6}  {ts:>12}  {te:>12}  {w['sharpe']:>8.3f}  {w['cagr']:>7.2%}  {w['max_drawdown']:>7.2%}")
    else:
        print("  No walk-forward windows completed.")

    print()
    print("-" * 70)
    print("  OOS AGGREGATE METRICS")
    print("-" * 70)
    print(f"  OOS Sharpe:             {val_results['oos_sharpe']:.3f}")
    print(f"  OOS CAGR:               {val_results['oos_cagr']:.2%}")
    print(f"  OOS Max Drawdown:       {val_results['oos_max_drawdown']:.2%}")
    ss = val_results.get("sharpe_stability", float("nan"))
    if np.isnan(ss):
        print(f"  Sharpe Stability:       N/A")
    else:
        print(f"  Sharpe Stability (CV):  {ss:.3f}")
    print("=" * 70)

# ==== MAIN ====

def main():
    cfg = CONFIG.copy()

    print("Trader v5.0 — Loading data...")
    ticker_data = load_data(cfg)
    ticker_data = align_dates(ticker_data)

    spy_data = ticker_data.pop(cfg["benchmark"])
    available_tickers = [t for t in cfg["tickers"] if t in ticker_data]

    if not available_tickers:
        print("ERROR: No ticker data available. Exiting.")
        return

    ticker_data[cfg["benchmark"]] = spy_data  # put back for backtest

    print(f"Computing indicators for {len(available_tickers)} tickers...")
    ticker_signals = {}
    for ticker in available_tickers:
        df = compute_indicators(ticker_data[ticker], cfg)
        df = compute_signals(df, cfg)
        ticker_signals[ticker] = df

    print("Running backtest...")
    results = run_backtest(ticker_data, ticker_signals, cfg)

    print("Computing metrics...")
    met = compute_all_metrics(
        results["portfolio_ret"],
        results["portfolio_equity"],
        results["spy_ret"],
        results["spy_equity"],
        results["turnover"],
        results["costs"],
        results["trade_log"],
        results["adjusted_weights"],
    )

    print_report(met, results, cfg)
    print_per_ticker_metrics(ticker_signals, ticker_data, results["adjusted_weights"])

    print("Running walk-forward validation...")
    val = walk_forward_test(ticker_data, ticker_signals, cfg)
    print_validation_report(val)

    print("Generating plots...")
    fig1 = plot_portfolio_overview(results)
    fig2 = plot_per_ticker_grid(results)
    fig3 = plot_strategy_analysis(results)

    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
