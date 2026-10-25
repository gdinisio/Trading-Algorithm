# TRADER v5.0 — INSTITUTIONAL REDESIGN PLAN

---

## 1. EXECUTIVE DIAGNOSIS

### 1.1 Structural Weaknesses

**A. Binary signal logic destroys information.** All three strategies (BB, MA, MACD) output discrete {-1, 0, +1} signals. This discards the *degree* of conviction. A price barely touching the lower Bollinger band generates the same full-conviction long as a price 3 standard deviations below the band. This is the single largest source of alpha loss in the system.

**B. Strategies are highly correlated.** MA crossover and MACD are both trend-following using overlapping price data. In trend regimes, the system allocates 50% MA + 50% MACD — two signals measuring nearly the same thing. This provides false diversification and doubles down on trend exposure without diversification benefit.

**C. Mean reversion strategy is structurally disadvantaged.** The BB strategy can go long, short, or flat — but mean reversion on large-cap equities that have structural downward drift (the chosen tickers) frequently shorts bounces and gets stopped out. Mean reversion is most effective as a *long-only* strategy on stocks with positive expected return, or as a *long/short* strategy on spreads. Using it symmetrically on single names with negative drift is a structural drag.

**D. No position overlap management.** Three strategies run independently per ticker, generating three separate position/size pairs. The regime blender combines their *returns* (not positions), meaning the system cannot reason about net exposure. On a given day, BB might be short while MA and MACD are long — the blended return partially cancels, wasting capital and creating hidden churn.

### 1.2 Mathematical Flaws

**A. Position sizing uses today's price and ATR.** `position_sizing()` computes `atr_pct = atr / price` using same-day values, then this size is shifted by 1 in `compute_strategy_returns()`. This is correct for avoiding look-ahead, but the ATR itself is computed using today's high/low/close, which includes intraday information not available at market open. Minor bias, but worth noting.

**B. Daily loss limit clips returns, not positions.** `apply_daily_loss_limit(returns, -0.03)` clips the *realized return* after the fact. In real trading, you cannot retroactively limit a day's return — the loss already happened. This should instead reduce position sizes or trigger stop-losses. As implemented, it artificially improves drawdown metrics without corresponding to any executable action.

**C. The rolling Sharpe weighting scheme is unstable.** The `compute_dynamic_weights()` function shifts rolling Sharpe values to non-negative then normalizes. When all tickers have similar (or all negative) Sharpe ratios, the min-shift operation amplifies noise: tiny differences in Sharpe become large weight differences. Additionally, rolling Sharpe with a 60-day window on daily returns is extremely noisy (standard error of Sharpe ≈ 1/sqrt(60) ≈ 0.13 per unit Sharpe).

### 1.3 Statistical Weaknesses

**A. No transaction costs.** The system assumes costless, instantaneous execution. For 8 tickers with 3 strategies each, generating hundreds of trades, even small costs (5-10 bps per trade) compound into meaningful drag. Without cost modeling, the backtest overstates achievable returns.

**B. Overfitting surface is large.** The CONFIG dict has ~25 free parameters. With 8 tickers over ~10 years of daily data (~2500 points), the ratio of degrees of freedom to data points is concerning. The specific parameter values (bb_length=16, ma_fast=10, ma_slow=30, etc.) were likely selected on in-sample data with no out-of-sample validation.

**C. No walk-forward or out-of-sample testing.** The entire history is used both for parameter selection and performance evaluation. There is no train/test split, no walk-forward framework, and no robustness checks.

### 1.4 Risk Flaws

**A. Circuit breaker operates on equity that includes clipped losses.** The daily loss limit clips returns before the equity curve is built, but the circuit breaker uses that same equity curve. This means the circuit breaker sees an artificially shallow drawdown and activates later than it would in reality.

**B. No correlation-aware risk.** The system treats each ticker independently. If all 8 tickers are selling off simultaneously (systemic risk), the portfolio has 8x concentrated exposure in the same direction. There is no cross-asset volatility budget or correlation monitoring.

**C. Re-entry after circuit breaker is too aggressive.** Recovering from -15% drawdown to -5% drawdown (re-entry threshold) means the portfolio must gain ~11.8% from its trough while sitting in cash. But the system re-enters fully, not gradually, creating the risk of immediately hitting the circuit breaker again.

### 1.5 Signal Flaws

**A. RSI confirmation is backwards for trend-following.** For MA/MACD, the system requires RSI > 50 to go long. But the best trend entries often occur when RSI is recovering *from* oversold — i.e., RSI is 35 and rising. Requiring RSI > 50 means the system enters *after* the initial move has already occurred, reducing edge.

**B. Volume gate blocks too many valid signals.** Requiring volume > 1.5x its 20-day average is aggressive. Many legitimate breakouts begin on moderate volume and accelerate later. The gate blocks these entries entirely rather than scaling into them.

**C. Weekly confirmation creates whipsaw in transitions.** The weekly MA filter hard-blocks daily signals that disagree. At weekly trend transitions, this creates periods where *all* signals are blocked, forcing the system flat during precisely the periods when new trends are forming.

### 1.6 Implementation Weaknesses

**A. Single-file monolith.** 1000+ lines with no separation of concerns. Indicators, signals, regime, risk, portfolio, reporting, and plotting are all interleaved, making it impossible to test components independently.

**B. Python for-loops for trailing stops and regime detection.** These row-by-row loops are O(n) with high constant factor in pure Python. For 8 tickers × ~6500 days, this is slow but tolerable. The real issue is that the loop logic is fragile and hard to verify.

---

## 2. DESIGN PHILOSOPHY

### 2.1 Guiding Principles

1. **Continuous signals, not binary decisions.** Every alpha source should output a continuous conviction score in [-1, +1]. Binary signals destroy information and create unnecessary churn.

2. **Fewer, more orthogonal signals.** Replace three correlated indicators with signals that measure genuinely different things: mean reversion strength, trend strength, and volatility regime. Each should capture a distinct market dimension.

3. **Risk-first architecture.** Position sizes are determined by the risk budget, not by signal strength alone. Signal strength determines *direction and conviction*; risk management determines *how much*.

4. **Everything shifted, everything auditable.** Every computation has explicit t vs t+1 documentation. No value computed at time t is used to generate returns at time t.

5. **Transaction costs are first-class.** No signal is worth taking if its expected return doesn't exceed its expected cost. Cost-awareness is embedded in signal generation, not bolted on after.

6. **Long-only bias.** For these large-cap equities with flat-to-negative drift, *shorting is structurally negative expectancy* after costs and borrow. The system should be long-only with the ability to go to cash. This single change removes the largest source of losses.

7. **Simplicity over complexity.** Fewer parameters, wider windows, more robust signals. A system with 10 well-chosen parameters will outperform one with 25 overfitted parameters out of sample.

### 2.2 Risk Philosophy

- Target portfolio-level volatility, not per-trade risk
- Scale exposure inversely with realized cross-asset correlation
- Gradual de-risking, not binary circuit breakers
- Drawdown-responsive, not drawdown-reactive

### 2.3 Alpha Philosophy

- Mean reversion works best at extremes (BB z-score < -2) and only long
- Trend works best when confirmed by multiple timeframes
- The combination of mean reversion + trend is valuable because they are genuinely orthogonal
- Signal decay: recent signals matter more than stale ones

### 2.4 Robustness Philosophy

- Every parameter should be justified by market microstructure, not curve-fitting
- Walk-forward validation is mandatory
- Performance should degrade gracefully if any single parameter is moved ±30%

### 2.5 Bias Prevention Philosophy

- All signals computed at close of day t, applied to return from close t to close t+1
- All weights, sizes, and positions shifted by 1 day before computing returns
- No rolling window includes the day whose return it affects
- Transaction costs applied on every position change

---

## 3. FULL SYSTEM ARCHITECTURE

### 3.1 File Structure

```
v5/
├── config.py          # All configuration constants
├── data.py            # Data loading and validation
├── indicators.py      # Technical indicator computations
├── signals.py         # Alpha signal generation (continuous scores)
├── regime.py          # Volatility regime detection
├── sizing.py          # Position sizing and risk budgeting
├── portfolio.py       # Portfolio construction and weighting
├── risk.py            # Risk management overlays
├── costs.py           # Transaction cost modeling
├── backtest.py        # Backtesting engine (execution timing)
├── metrics.py         # Performance and risk metrics
├── reporting.py       # Console report generation
├── plotting.py        # All visualization
├── validation.py      # Walk-forward and robustness testing
└── main.py            # Orchestration entry point
```

### 3.2 Data Flow

```
main.py
  → data.py:load_data(cfg) → Dict[str, DataFrame]
  → For each ticker:
      → indicators.py:compute_indicators(df, cfg) → df with indicator columns
      → signals.py:compute_signals(df, cfg) → DataFrame with signal columns ["sig_mr", "sig_trend", "sig_composite"]
      → regime.py:detect_regime(df, cfg) → Series (continuous risk_scalar in [0, 1])
  → portfolio.py:construct_portfolio(ticker_signals, cfg) → weights DataFrame
  → sizing.py:compute_position_sizes(weights, ticker_data, cfg) → sized_positions DataFrame
  → costs.py:apply_costs(sized_positions, ticker_data, cfg) → net_returns Series
  → risk.py:apply_risk_overlays(net_returns, sized_positions, cfg) → final_returns Series
  → backtest.py:run_backtest(final_returns) → equity, drawdown, trade_log
  → metrics.py:compute_all_metrics(...) → metrics dict
  → reporting.py:print_report(...)
  → plotting.py:generate_all_plots(...)
  → validation.py:walk_forward_test(ticker_data, cfg) → validation_results
```

### 3.3 DataFrame Index Convention

All DataFrames are indexed by `DatetimeIndex` (trading days only). All per-ticker DataFrames share the same date index after alignment in `data.py`. The column naming convention:

- Raw price: `Open`, `High`, `Low`, `Close`, `Volume`
- Indicators: `RSI_14`, `BB_zscore`, `ATR_14`, `ADX_14`, `MA_50`, `MA_200`, `MACD_hist`, `VOL_ratio`
- Signals: `sig_mr` (mean reversion), `sig_trend` (trend), `sig_composite` (combined)
- Regime: `risk_scalar` (continuous, 0=full risk-off, 1=full risk-on)
- Positions: `target_weight` (desired allocation fraction per ticker)
- Sized: `sized_position` (target_weight × risk_scalar × vol_scalar)

### 3.4 Execution Timing (Critical)

All computations use data available at the close of day t. All positions are assumed to be executed at the close of day t (or equivalently, open of day t+1). Returns are computed as close-to-close from day t to day t+1.

Formally:
- `signal[t]` = f(price[0..t], volume[0..t], indicators[0..t])
- `position[t]` = g(signal[t], regime[t], risk_state[t])
- `return[t+1]` = position[t] × (close[t+1] - close[t]) / close[t] - costs[t]

Implementation: compute signals/positions for all dates, then `.shift(1)` the position series before multiplying by returns. This is how v4 already does it for per-strategy returns but NOT for portfolio weights — v5 will enforce this universally.

---

## 4. ALPHA & SIGNAL FRAMEWORK

### 4.1 Signal 1: Mean Reversion Score (`sig_mr`)

**Rationale:** Mean reversion captures oversold conditions where price is likely to revert toward its moving average. This is most effective as a *long-only* signal at extremes.

**Formula:**

```
bb_zscore[t] = (close[t] - SMA(close, 20)[t]) / rolling_std(close, 20)[t]
sig_mr[t] = clip(-bb_zscore[t] / 2.5, 0, 1)
```

- `SMA(close, 20)` = 20-day simple moving average of close
- `rolling_std(close, 20)` = 20-day rolling standard deviation of close
- `bb_zscore` measures how many standard deviations price is from its mean
- Dividing by 2.5 normalizes so that a 2.5-sigma deviation maps to full conviction
- `clip(0, 1)` makes this long-only: only fires when price is *below* the mean
- At z-score = 0 (price at mean): `sig_mr = 0`
- At z-score = -2.5 (price 2.5 sigma below mean): `sig_mr = 1.0`
- At z-score = +anything (price above mean): `sig_mr = 0`

**Window:** 20 days. Justification: approximately 1 trading month, the natural cycle for mean-reversion in large-cap equities.

**Lagging:** Uses close[t] and rolling window [t-19, t]. Signal at time t uses only data available at time t. Applied to return at t+1 via shift.

**RSI Confirmation (soft):**

```
rsi_14[t] = standard Wilder RSI with period 14
mr_rsi_boost[t] = clip((50 - rsi_14[t]) / 30, 0, 1)
sig_mr[t] = sig_mr[t] × (0.5 + 0.5 × mr_rsi_boost[t])
```

This scales the MR signal: at RSI=20, boost=1.0, signal is unmodified. At RSI=50, boost=0.0, signal is halved. At RSI=80, signal is halved (already 0 from zscore clipping anyway). This replaces the hard RSI gate with a soft scaling that preserves information.

### 4.2 Signal 2: Trend Score (`sig_trend`)

**Rationale:** Trend-following captures momentum persistence. We use a *single* composite trend measure instead of two correlated ones (MA + MACD).

**Formula:**

```
ma_fast[t] = EMA(close, 21)[t]
ma_slow[t] = EMA(close, 63)[t]
ma_anchor[t] = SMA(close, 200)[t]

trend_raw[t] = (ma_fast[t] - ma_slow[t]) / ATR_14[t]
```

- EMA 21 ≈ 1 month fast signal
- EMA 63 ≈ 1 quarter slow signal
- Normalizing by ATR makes the signal comparable across tickers and time periods
- SMA 200 acts as a structural filter (below)

**Normalization and bounding:**

```
trend_normalized[t] = clip(trend_raw[t] / 3.0, -1, 1)
```

Dividing by 3.0 means a trend strength of 3 ATRs maps to full conviction. This is empirically a strong trend.

**Anchor filter (long-only bias):**

```
above_200[t] = 1.0 if close[t] > ma_anchor[t] else 0.0
sig_trend[t] = clip(trend_normalized[t] × above_200[t], 0, 1)
```

When price is below the 200-day SMA, the trend signal is zeroed (go to cash, don't short). When above, the trend signal operates normally but only in the long direction (clip at 0). This enforces the long-only philosophy.

**Window:** 21/63/200 days. Justification: standard institutional timeframes (1-month, 1-quarter, 1-year). The 200-day SMA is the most-watched level in equity markets.

**Lagging:** All EMAs and SMAs use data up to and including close[t]. Applied to return at t+1 via shift.

### 4.3 Signal 3: Composite Score (`sig_composite`)

**Formula:**

```
sig_composite[t] = 0.5 × sig_mr[t] + 0.5 × sig_trend[t]
```

Equal weighting. The signals are orthogonal by construction:
- `sig_mr` fires when price is below its short-term mean (oversold condition)
- `sig_trend` fires when the medium-term trend is above the long-term trend AND price is above the 200-day SMA

These conditions are nearly mutually exclusive in extremes but can overlap in mild conditions, which is the correct behavior — mild conviction from both sides should combine.

**Output range:** [0, 1] where 0 = fully in cash, 1 = fully invested in this ticker.

**No orthogonalization needed** because the signals already measure different dimensions (level vs. momentum) and are both bounded in [0, 1].

---

## 5. REGIME / STATE LOGIC

### 5.1 Design Decision: Continuous Risk Scalar, Not Discrete Regimes

The v4 system uses discrete regimes (trend/range/transition/crisis) with hard boundaries. This creates problems:
- Boundary effects at regime transitions
- Hysteresis logic adds complexity and parameters
- The system must classify then act, rather than smoothly adapting

**v5 replaces discrete regimes with a continuous `risk_scalar` in [0, 1].**

### 5.2 Risk Scalar Computation

The risk scalar combines two components: volatility level and drawdown depth.

**Component 1: Volatility scaling**

```
realized_vol[t] = rolling_std(daily_returns, 20)[t] × sqrt(252)
vol_median[t] = rolling_median(realized_vol, 252)[t]
vol_ratio[t] = realized_vol[t] / vol_median[t]
vol_scalar[t] = clip(2.0 - vol_ratio[t], 0.2, 1.0)
```

- When vol equals its 1-year median: `vol_ratio = 1.0`, `vol_scalar = 1.0` (full risk)
- When vol is 2x median: `vol_ratio = 2.0`, `vol_scalar = 0.0` → clipped to 0.2 (minimum 20% exposure)
- When vol is 1.5x median: `vol_ratio = 1.5`, `vol_scalar = 0.5` (half risk)
- When vol is below median: `vol_scalar = 1.0` (capped, don't lever up)
- Floor of 0.2 ensures we never go fully to cash on volatility alone

**Component 2: Drawdown scaling**

```
equity[t] = cumulative portfolio equity up to day t
peak[t] = running max of equity[0..t]
dd[t] = (equity[t] - peak[t]) / peak[t]    # negative number
dd_scalar[t] = clip(1.0 + dd[t] / 0.10, 0.0, 1.0)
```

- At 0% drawdown: `dd_scalar = 1.0` (full risk)
- At -5% drawdown: `dd_scalar = 0.5` (half risk)
- At -10% drawdown: `dd_scalar = 0.0` (fully in cash)
- This is a *gradual* circuit breaker, not binary

**Combined risk scalar:**

```
risk_scalar[t] = vol_scalar[t] × dd_scalar[t]
```

- Both components multiply, so high vol + drawdown compounds the de-risking
- Range: [0.0, 1.0]

### 5.3 Minimum Duration

No minimum duration / hysteresis is needed because the inputs are already smooth (rolling 20-day vol, cumulative drawdown). The risk scalar changes gradually by construction.

### 5.4 How Risk Scalar Affects Allocation

The risk scalar multiplies the final position size for all tickers uniformly:

```
final_position[t] = target_position[t] × risk_scalar[t]
```

### 5.5 Lagging

- `vol_scalar[t]` uses `realized_vol[t]` which uses returns through close of day t. Since we shift positions by 1 before computing returns, this is applied to return at t+1. Correct.
- `dd_scalar[t]` uses equity through day t. Applied to position for day t+1. Correct.
- **Critical:** `dd_scalar` computation requires knowing equity up to day t. Since equity at day t depends on positions up to day t-1 (due to the 1-day shift), there is no circularity. The equity at day t is fully determined by close of day t and positions from day t-1 and earlier.

---

## 6. POSITION SIZING FRAMEWORK

### 6.1 Philosophy

Target a constant *portfolio-level* annualized volatility of 10%. Distribute this budget across tickers based on their signals and individual volatility.

### 6.2 Per-Ticker Volatility Estimate

```
ticker_vol[t] = EWMA_std(daily_returns, span=42)[t] × sqrt(252)
```

- 42-day EWMA ≈ 2-month half-life, responsive but not twitchy
- Annualized for interpretability

### 6.3 Target Volatility Position Sizing

```
target_portfolio_vol = 0.10  (10% annualized)
n_tickers = number of active tickers (those with sig_composite > 0.05)
per_ticker_vol_budget = target_portfolio_vol / sqrt(n_tickers)
raw_weight[t] = per_ticker_vol_budget / ticker_vol[t]
```

The `sqrt(n_tickers)` divisor accounts for diversification under the assumption of moderate correlation. For 8 tickers with average correlation ~0.4, `sqrt(8) ≈ 2.83` is a reasonable approximation to the effective number of independent bets.

### 6.4 Signal-Weighted Sizing

```
signal_weight[t] = raw_weight[t] × sig_composite[t]
```

A ticker with 0 signal gets 0 allocation. A ticker with full conviction gets the full vol-budget allocation.

### 6.5 Constraints

- **Per-ticker maximum:** `signal_weight[t] = min(signal_weight[t], 0.20)` — no single ticker exceeds 20% of portfolio
- **Total exposure maximum:** if `sum(signal_weight[t]) > 1.0`, scale all weights proportionally so they sum to 1.0. The portfolio is never levered.
- **Total exposure minimum:** no floor (can be 100% cash if all signals are 0)
- **No shorting:** all weights are in [0, 0.20] by construction since signals are in [0, 1]

### 6.6 Risk Scalar Application

```
final_weight[t] = signal_weight[t] × risk_scalar[t]
```

Applied after constraints. This can push total exposure below the per-ticker cap, which is intended.

### 6.7 Lagging

- `ticker_vol[t]` uses returns through day t (42-day EWMA). Position applied to return at t+1. Correct.
- `sig_composite[t]` uses indicators through day t. Correct.
- `risk_scalar[t]` uses equity through day t. Correct.
- All `final_weight[t]` values are shifted by 1 before multiplying by day t+1 returns.

---

## 7. PORTFOLIO CONSTRUCTION LOGIC

### 7.1 Weighting Methodology

No dynamic Sharpe-based weighting. Instead, the portfolio weights are directly determined by the signal-weighted sizing from Section 6. The rolling Sharpe approach in v4 was the primary source of instability. Replacing it with a signal-driven approach is simpler and more robust.

The logic is:

1. For each ticker, compute `sig_composite[t]` (Section 4)
2. Convert to `signal_weight[t]` via vol-targeting (Section 6)
3. Apply constraints (Section 6.5)
4. Apply risk scalar (Section 6.6)
5. Remainder (1 - sum of weights) is held in cash (earns 0 return)

### 7.2 Rebalancing Rules

Rebalance daily, but with a **turnover penalty** that creates natural stickiness.

The turnover control mechanism:

```
proposed_weight[t] = final_weight[t] as computed in Section 6
current_weight[t] = previous day's weight, drifted by one day's return

drift_adjusted[t] = current_weight[t-1] × (1 + ticker_return[t]) / portfolio_return_scale[t]
```

Where `portfolio_return_scale[t]` normalizes so the drifted weights still sum to their pre-drift sum.

**Trade threshold:** only rebalance a ticker if:

```
abs(proposed_weight[t] - drift_adjusted[t]) > 0.02
```

If the difference is ≤ 2%, keep the drifted weight and avoid trading. This reduces turnover by ~60-70% in typical conditions.

When rebalancing does occur, set `actual_weight[t] = proposed_weight[t]`.

### 7.3 No Covariance Estimation

The `sqrt(n_tickers)` diversification approximation in Section 6.3 is more robust than a covariance matrix estimated from rolling windows, which would be noisy and add parameters. For 8 large-cap US equities, the correlation structure is stable enough that this approximation is adequate.

### 7.4 Turnover Measurement

```
daily_turnover[t] = sum(abs(actual_weight[t] - drift_adjusted[t])) for all tickers
```

This is reported in the metrics.

---

## 8. RISK MANAGEMENT SYSTEM

### 8.1 Trade-Level Controls

**A. No stop-losses per trade.** The continuous signal naturally reduces conviction as conditions change, which progressively reduces position size. This is a *soft stop* driven by the alpha signal itself, not an arbitrary price level.

**B. Maximum holding period:** None. Positions are sized daily based on signals; there's no discrete "entry" or "exit" to hold for.

### 8.2 Portfolio-Level Controls

**A. Gross exposure cap:** Total allocation across all tickers ≤ 1.0 (no leverage). Enforced in Section 6.5.

**B. Per-ticker cap:** 20% maximum per ticker. Enforced in Section 6.5.

**C. Sector concentration:** Not explicitly modeled (all 8 tickers are different sectors by design of the ticker selection). No additional constraint needed.

### 8.3 Drawdown Controls

Handled by the continuous `dd_scalar` in Section 5.2. This is the gradual circuit breaker:
- Linearly scales from 100% exposure at 0% drawdown to 0% exposure at -10% drawdown
- Re-entry is also gradual: as equity recovers, exposure increases linearly

### 8.4 Volatility Controls

Handled by the continuous `vol_scalar` in Section 5.2. Scales exposure inversely with realized volatility relative to its median.

### 8.5 Kill Switch

If `risk_scalar[t] < 0.01` (effectively zero), the system is 100% cash. This is equivalent to a kill switch but reached gradually.

### 8.6 Capital Preservation Logic

The combination of:
1. Long-only (cannot lose more than invested)
2. Drawdown scalar (de-risks as drawdown deepens)
3. Vol scalar (de-risks as volatility spikes)
4. No leverage (exposure ≤ 100%)

...means the maximum possible loss in any single day is approximately the portfolio's gross exposure × the worst single-day return across all held tickers. With typical exposure of 40-70% and worst daily drops of ~10%, the maximum single-day portfolio loss is ~7%.

---

## 9. TRANSACTION COST & SLIPPAGE MODEL

### 9.1 Cost Assumptions

Model: proportional cost per unit of turnover.

```
cost_per_unit_turnover = 0.001  (10 basis points round-trip)
```

This accounts for:
- Bid-ask spread: ~3-5 bps for large-cap
- Market impact: ~2-3 bps for moderate sizing
- Commission: ~1-2 bps
- Total: ~8-10 bps, rounded to 10 bps

### 9.2 Turnover Measurement

```
turnover[t] = sum(abs(actual_weight[t] - drift_adjusted[t])) across all tickers
```

This measures *one-way* turnover. The cost model applies to full turnover (which already represents the full round of sells + buys).

### 9.3 How Costs Are Applied

```
gross_return[t] = sum(actual_weight[t-1] × ticker_return[t]) for all tickers
cost[t] = turnover[t] × cost_per_unit_turnover
net_return[t] = gross_return[t] - cost[t]
```

Costs are subtracted from portfolio returns on the day the rebalance occurs. The `actual_weight[t-1]` is the weight determined at end of day t-1, applied to day t's return.

### 9.4 Execution Timing

All rebalancing is assumed to occur at market close of day t (after signals are computed). Returns are close-to-close. This is a simplification — in practice, execution would occur at/near the open of day t+1, but close-to-close is standard for daily backtests and the error is small for large-cap liquid equities.

---

## 10. BIAS & DATA INTEGRITY SAFEGUARDS

### 10.1 Complete Timeline for Day t

1. Market opens at 9:30 AM. Prices trade throughout the day.
2. Market closes at 4:00 PM. Close price, high, low, volume are finalized.
3. After close: compute all indicators using data through close[t].
4. After close: compute signals `sig_mr[t]`, `sig_trend[t]`, `sig_composite[t]`.
5. After close: compute `risk_scalar[t]` using equity through day t (which was determined by position[t-1] × return[t]).
6. After close: compute `final_weight[t]` = f(signal[t], risk_scalar[t], ticker_vol[t]).
7. After close: determine if rebalance needed (compare to drifted weights).
8. If rebalance: execute trades at close (idealized). Record `actual_weight[t]`.
9. Day t+1: portfolio return = sum(actual_weight[t] × ticker_return[t+1]) - costs.

### 10.2 All Rolling Window Usage

| Computation | Window | Data Used | Applied To |
|---|---|---|---|
| RSI | 14-day EWM | returns[t-13..t] | signal at t → return at t+1 |
| BB z-score | 20-day SMA/std | close[t-19..t] | signal at t → return at t+1 |
| EMA fast (trend) | 21-day EWM | close[0..t] | signal at t → return at t+1 |
| EMA slow (trend) | 63-day EWM | close[0..t] | signal at t → return at t+1 |
| SMA anchor | 200-day SMA | close[t-199..t] | signal at t → return at t+1 |
| ATR | 14-day EWM | H/L/C[t-13..t] | normalization at t → return at t+1 |
| Realized vol | 20-day rolling std | returns[t-19..t] | vol_scalar at t → position at t → return at t+1 |
| Vol median | 252-day rolling median | realized_vol[t-251..t] | vol_scalar at t → position at t → return at t+1 |
| Ticker vol (sizing) | 42-day EWMA std | returns[0..t] | sizing at t → position at t → return at t+1 |

### 10.3 All Shift Rules

| Value | Shift | Reason |
|---|---|---|
| `actual_weight[t]` | Used for return at t+1 | Decision at close of t, earns/loses from t+1 |
| `risk_scalar[t]` | Incorporated into weight[t], used at t+1 | Computed at close of t |
| `turnover[t]` | Cost applied to return at t+1 | Trades executed at close of t, cost realized at t+1 |

**Implementation:** In the backtest engine, compute all weights for all dates first, then shift the entire weight DataFrame by 1 before multiplying by returns:

```
returns[t] = (shifted_weights[t] × ticker_returns[t]).sum() - costs[t]
```

where `shifted_weights[t] = actual_weight[t-1]`.

---

## 11. IMPLEMENTATION BLUEPRINT FOR SONNET

### 11.1 Build Order (Strict Dependency Order)

1. `config.py` — no dependencies
2. `metrics.py` — depends on: numpy, pandas only
3. `data.py` — depends on: config, yfinance, pandas
4. `indicators.py` — depends on: pandas, numpy
5. `signals.py` — depends on: indicators
6. `regime.py` — depends on: pandas, numpy
7. `costs.py` — depends on: pandas, numpy
8. `sizing.py` — depends on: regime, signals
9. `portfolio.py` — depends on: sizing, costs
10. `risk.py` — depends on: regime, metrics
11. `backtest.py` — depends on: portfolio, costs, risk, metrics
12. `reporting.py` — depends on: metrics
13. `plotting.py` — depends on: metrics
14. `validation.py` — depends on: backtest, metrics
15. `main.py` — depends on: all above

### 11.2 File-by-File Specification

---

#### `config.py`

Single dictionary `CONFIG`:

```python
CONFIG = {
    # Universe
    "tickers": ["BA", "INTC", "IBM", "PFE", "T", "VZ", "MMM", "KHC"],
    "benchmark": "SPY",
    "start_date": "1998-01-01",
    "end_date": None,  # None = today

    # Indicator windows
    "rsi_period": 14,
    "bb_window": 20,
    "bb_zscore_cap": 2.5,
    "ema_fast": 21,
    "ema_slow": 63,
    "sma_anchor": 200,
    "atr_period": 14,

    # Signal parameters
    "mr_rsi_scale_center": 50,
    "mr_rsi_scale_range": 30,
    "trend_atr_norm_cap": 3.0,

    # Signal combination
    "signal_mr_weight": 0.5,
    "signal_trend_weight": 0.5,
    "signal_min_threshold": 0.05,

    # Risk regime
    "vol_window": 20,
    "vol_median_window": 252,
    "vol_scalar_floor": 0.2,
    "dd_full_deleverage": -0.10,

    # Position sizing
    "target_portfolio_vol": 0.10,
    "ticker_vol_span": 42,
    "max_per_ticker": 0.20,
    "max_gross_exposure": 1.0,

    # Rebalancing
    "rebalance_threshold": 0.02,

    # Transaction costs
    "cost_per_unit_turnover": 0.001,

    # Walk-forward
    "wf_train_years": 5,
    "wf_test_months": 6,
}
```

Total: 27 parameters but most are structural (window lengths tied to market microstructure) not free (fitted) parameters. The truly free parameters that affect alpha are: `bb_zscore_cap`, `trend_atr_norm_cap`, `signal_mr_weight`, `signal_trend_weight` — only 4.

---

#### `data.py`

**Function:** `load_data(cfg: dict) -> dict[str, pd.DataFrame]`

- Downloads OHLCV for all tickers + benchmark via `yf.download`
- For each ticker, creates a DataFrame with columns: `Open`, `High`, `Low`, `Close`, `Volume`
- Drops rows where `Close` is NaN
- Drops tickers with fewer than 252 trading days of data (need at least 1 year for indicator warmup)
- Returns `{"BA": df_ba, "INTC": df_intc, ..., "SPY": df_spy}`
- Prints warnings for any tickers that couldn't be loaded

**Function:** `align_dates(ticker_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]`

- Finds the common date range across all tickers (intersection of all DatetimeIndexes)
- Reindexes each ticker DataFrame to this common index
- Forward-fills any gaps (max 5 days), drops any remaining NaN rows
- Returns the aligned dict

---

#### `indicators.py`

**Function:** `compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame`

Takes a single ticker's OHLCV DataFrame and adds indicator columns.

Adds the following columns:
- `RSI_14`: Wilder RSI with period `cfg["rsi_period"]`. Formula: standard Wilder RSI using `ewm(alpha=1/period)` on gains and losses.
- `BB_zscore`: `(Close - SMA(Close, cfg["bb_window"])) / rolling_std(Close, cfg["bb_window"])`. Use `min_periods=cfg["bb_window"]` (not 1) to avoid warmup distortion.
- `EMA_fast`: `Close.ewm(span=cfg["ema_fast"], adjust=False).mean()`
- `EMA_slow`: `Close.ewm(span=cfg["ema_slow"], adjust=False).mean()`
- `SMA_anchor`: `Close.rolling(cfg["sma_anchor"], min_periods=cfg["sma_anchor"]).mean()`
- `ATR_14`: Average True Range using `ewm(span=cfg["atr_period"])`. Standard TR = max(H-L, |H-prevC|, |L-prevC|).
- `Daily_Return`: `Close.pct_change()`
- `Realized_Vol`: `Daily_Return.rolling(cfg["vol_window"]).std() * np.sqrt(252)`. Use `min_periods=cfg["vol_window"]`.
- `Vol_Median`: `Realized_Vol.rolling(cfg["vol_median_window"], min_periods=60).median()`
- `Ticker_Vol`: `Daily_Return.ewm(span=cfg["ticker_vol_span"]).std() * np.sqrt(252)`

Returns the DataFrame with all original + new columns. Rows where indicators are NaN (warmup period) are kept (not dropped) — the backtest engine will handle this by starting after warmup.

---

#### `signals.py`

**Function:** `compute_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame`

Takes a single ticker's DataFrame (already with indicators) and adds signal columns.

**Step 1: Mean Reversion Signal**

```python
bb_z = df["BB_zscore"]
sig_mr_raw = (-bb_z / cfg["bb_zscore_cap"]).clip(0, 1)

rsi = df["RSI_14"]
mr_rsi_boost = ((cfg["mr_rsi_scale_center"] - rsi) / cfg["mr_rsi_scale_range"]).clip(0, 1)
sig_mr = sig_mr_raw * (0.5 + 0.5 * mr_rsi_boost)
```

Add column `sig_mr`.

**Step 2: Trend Signal**

```python
trend_raw = (df["EMA_fast"] - df["EMA_slow"]) / df["ATR_14"]
trend_normalized = (trend_raw / cfg["trend_atr_norm_cap"]).clip(-1, 1)
above_anchor = (df["Close"] > df["SMA_anchor"]).astype(float)
sig_trend = (trend_normalized * above_anchor).clip(0, 1)
```

Add column `sig_trend`.

**Step 3: Composite Signal**

```python
sig_composite = cfg["signal_mr_weight"] * sig_mr + cfg["signal_trend_weight"] * sig_trend
```

Add column `sig_composite`.

Returns the DataFrame with `sig_mr`, `sig_trend`, `sig_composite` columns added.

---

#### `regime.py`

**Function:** `compute_vol_scalar(df: pd.DataFrame, cfg: dict) -> pd.Series`

```python
vol_ratio = df["Realized_Vol"] / df["Vol_Median"]
vol_scalar = (2.0 - vol_ratio).clip(cfg["vol_scalar_floor"], 1.0)
```

Returns Series `vol_scalar` indexed by date.

**Function:** `compute_dd_scalar(equity: pd.Series, cfg: dict) -> pd.Series`

```python
peak = equity.cummax()
dd = (equity - peak) / peak
dd_scalar = (1.0 + dd / abs(cfg["dd_full_deleverage"])).clip(0.0, 1.0)
```

Returns Series `dd_scalar` indexed by date.

**Function:** `compute_risk_scalar(vol_scalar: pd.Series, dd_scalar: pd.Series) -> pd.Series`

```python
risk_scalar = vol_scalar * dd_scalar
```

Returns Series `risk_scalar` indexed by date.

---

#### `sizing.py`

**Function:** `compute_target_weights(ticker_signals: dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame`

Arguments:
- `ticker_signals`: dict mapping ticker name to its DataFrame (with indicators + signals)
- `cfg`: config dict

Returns: DataFrame indexed by date, columns = ticker names, values = target weight per ticker in [0, max_per_ticker].

Logic:
1. Build a DataFrame `sig_df` of `sig_composite` for all tickers (columns = tickers, index = dates).
2. Build a DataFrame `vol_df` of `Ticker_Vol` for all tickers.
3. For each date t:
   - `n_active[t]` = number of tickers where `sig_df[t] > cfg["signal_min_threshold"]`
   - If `n_active[t] == 0`: all weights are 0.
   - `per_ticker_vol_budget = cfg["target_portfolio_vol"] / sqrt(max(n_active[t], 1))`
   - `raw_weight[t, ticker] = per_ticker_vol_budget / vol_df[t, ticker]` for each ticker
   - `signal_weight[t, ticker] = raw_weight[t, ticker] × sig_df[t, ticker]`
   - Clip each to `[0, cfg["max_per_ticker"]]`
   - If sum > `cfg["max_gross_exposure"]`: scale all proportionally so sum = `cfg["max_gross_exposure"]`
4. Fill NaN weights with 0 (warmup period).

This entire computation should be vectorized using pandas DataFrames, not row-by-row loops. The `n_active` count and budget can be computed per-row using `.apply()` or vectorized boolean sums.

---

#### `portfolio.py`

**Function:** `construct_portfolio(target_weights: pd.DataFrame, ticker_data: dict[str, pd.DataFrame], cfg: dict) -> tuple[pd.DataFrame, pd.Series]`

Arguments:
- `target_weights`: DataFrame of target weights (from sizing.py)
- `ticker_data`: dict of ticker DataFrames (need Close prices for drift)
- `cfg`: config dict

Returns:
- `actual_weights`: DataFrame of actual weights after rebalance threshold filter
- `turnover`: Series of daily turnover

Logic:
1. Compute `ticker_returns`: DataFrame of daily close-to-close returns for all tickers.
2. Initialize `actual_weights` as a copy of `target_weights` for the first valid date.
3. For each subsequent date t:
   - Compute `drift_adjusted[t]` for each ticker: `actual_weights[t-1] × (1 + ticker_returns[t])`, then normalize so the sum equals the previous day's sum × (1 + portfolio_return[t]).
   - **Simplification:** Since we're not trying to track exact drift precisely, use the simpler approximation: `drift_adjusted[t, ticker] = actual_weights[t-1, ticker] × (1 + ticker_returns[t]) / (1 + sum(actual_weights[t-1] × ticker_returns[t]))`
   - For each ticker: if `abs(target_weights[t, ticker] - drift_adjusted[t, ticker]) > cfg["rebalance_threshold"]`: set `actual_weights[t, ticker] = target_weights[t, ticker]`, else set `actual_weights[t, ticker] = drift_adjusted[t, ticker]`
   - `turnover[t] = sum(abs(actual_weights[t] - drift_adjusted[t]))` across all tickers
4. Return `actual_weights` and `turnover`.

**Important:** This function uses a for-loop over dates because the drift-adjusted weights depend on the previous day's actual weights (sequential dependency). This is acceptable because the loop body is simple vectorized operations across tickers.

---

#### `costs.py`

**Function:** `compute_costs(turnover: pd.Series, cfg: dict) -> pd.Series`

```python
costs = turnover * cfg["cost_per_unit_turnover"]
```

Returns Series of daily transaction costs.

---

#### `backtest.py`

**Function:** `run_backtest(ticker_data: dict[str, pd.DataFrame], ticker_signals: dict[str, pd.DataFrame], cfg: dict) -> dict`

This is the main orchestration of a single backtest run. Steps:

1. `target_weights = sizing.compute_target_weights(ticker_signals, cfg)`
2. Compute initial `vol_scalar` for each ticker using `regime.compute_vol_scalar`. Take the *average* vol_scalar across tickers as the portfolio vol_scalar: `avg_vol_scalar = mean(vol_scalar for all tickers)`. Apply to target_weights: `target_weights = target_weights × avg_vol_scalar` (multiply each row of target_weights by the scalar for that date).
3. `actual_weights, turnover = portfolio.construct_portfolio(target_weights, ticker_data, cfg)`
4. `costs = costs_module.compute_costs(turnover, cfg)`
5. Compute `ticker_returns`: DataFrame of daily returns for all tickers.
6. Compute gross portfolio returns: `gross_ret[t] = sum(actual_weights[t-1] × ticker_returns[t])` — shift actual_weights by 1 here.
7. `net_ret = gross_ret - costs`
8. `equity = (1 + net_ret).cumprod()`
9. Compute `dd_scalar = regime.compute_dd_scalar(equity, cfg)`
10. **Second pass with drawdown overlay:** The dd_scalar creates a feedback loop (equity → dd_scalar → weights → equity). Resolve this iteratively:
    - Use the equity from step 8 to compute dd_scalar
    - Apply dd_scalar to actual_weights: `adjusted_weights = actual_weights × dd_scalar`
    - Recompute `gross_ret[t] = sum(adjusted_weights[t-1] × ticker_returns[t])`
    - Recompute costs based on the new adjusted_weights (recompute turnover from adjusted_weights)
    - `net_ret = gross_ret - costs`
    - `equity = (1 + net_ret).cumprod()`
    - **Do NOT iterate further.** One pass is sufficient because the dd_scalar is a monotonic function of drawdown and converges quickly. A second iteration would be circular.
11. Compute benchmark: `spy_ret = ticker_data["SPY"]["Close"].pct_change().fillna(0)`, `spy_equity = (1 + spy_ret).cumprod()`. Align to same date range as portfolio.
12. Build trade log using `build_trade_log()` (see below).

Returns dict:
```python
{
    "portfolio_ret": pd.Series,       # net daily returns
    "portfolio_equity": pd.Series,    # cumulative equity
    "spy_ret": pd.Series,
    "spy_equity": pd.Series,
    "actual_weights": pd.DataFrame,   # per-ticker daily weights
    "adjusted_weights": pd.DataFrame, # after dd_scalar
    "turnover": pd.Series,
    "costs": pd.Series,
    "risk_scalar": pd.Series,         # vol_scalar × dd_scalar
    "vol_scalar": pd.Series,
    "dd_scalar": pd.Series,
    "trade_log": pd.DataFrame,
    "ticker_signals": dict,           # for reporting
}
```

**Function:** `build_trade_log(adjusted_weights: pd.DataFrame, ticker_data: dict[str, pd.DataFrame]) -> pd.DataFrame`

For each ticker, identify periods where weight > 0 as "trades". A trade starts when weight goes from 0 (or < 0.01) to > 0.01, and ends when it returns to < 0.01.

For each trade, record:
- `ticker`: str
- `entry_date`: date when weight first exceeded 0.01
- `exit_date`: date when weight dropped below 0.01 (or last date)
- `entry_price`: Close price on entry_date
- `exit_price`: Close price on exit_date
- `holding_days`: (exit_date - entry_date).days
- `pnl`: (exit_price - entry_price) / entry_price
- `max_weight`: maximum weight during the trade

Returns a DataFrame of all trades across all tickers.

---

#### `metrics.py`

**Function:** `compute_all_metrics(portfolio_ret: pd.Series, portfolio_equity: pd.Series, spy_ret: pd.Series, spy_equity: pd.Series, turnover: pd.Series, costs: pd.Series, trade_log: pd.DataFrame) -> dict`

Returns dict with keys:
- `sharpe`: annualized Sharpe ratio = `mean(ret) × 252 / (std(ret) × sqrt(252))`
- `sortino`: annualized Sortino = `mean(ret) × 252 / (downside_std × sqrt(252))` where downside_std = std of negative returns only
- `cagr`: `(equity[-1] / equity[0]) ^ (252 / n_days) - 1`
- `max_drawdown`: min of drawdown series
- `calmar`: `cagr / abs(max_drawdown)`
- `profit_factor`: sum of positive returns / abs(sum of negative returns)
- `total_trades`: len(trade_log)
- `win_rate`: fraction of trades with pnl > 0
- `avg_win`: mean pnl of winning trades
- `avg_loss`: mean pnl of losing trades
- `avg_holding_days`: mean holding_days
- `max_consec_losses`: longest streak of losing trades (sequential by exit_date)
- `annual_turnover`: `turnover.sum() / (n_days / 252)` (average annual turnover)
- `total_costs`: `costs.sum()`
- `cost_drag_annual`: `total_costs / (n_days / 252)` (annualized cost drag)
- `spy_sharpe`: Sharpe of SPY
- `spy_cagr`: CAGR of SPY
- `spy_max_drawdown`: MDD of SPY
- `best_trade`: max pnl in trade_log
- `worst_trade`: min pnl in trade_log
- `avg_exposure`: mean of daily sum of absolute weights

Each of the sub-computations (sharpe_ratio, sortino_ratio, max_drawdown, cagr, calmar_ratio, profit_factor, drawdown_series) should be implemented as standalone helper functions within metrics.py, identical formulas to the v4 versions.

---

#### `reporting.py`

**Function:** `print_report(metrics: dict, ticker_results: dict, cfg: dict) -> None`

Prints a formatted console report.

Section 1: "TRADER v5.0 — PORTFOLIO REPORT"
- Portfolio vs SPY comparison table: Sharpe, Sortino, Calmar, CAGR, Max Drawdown, Profit Factor, Final Equity

Section 2: "PER-TICKER SUMMARY"
- For each ticker: CAGR (of that ticker's contribution), avg weight, avg signal, number of trades

Section 3: "TRADE STATISTICS"
- Total trades, win rate, avg win, avg loss, avg holding days, max consecutive losses, best/worst trade

Section 4: "COST & TURNOVER"
- Annual turnover, total costs, annual cost drag, average exposure

Section 5: "RISK REGIME"
- Percentage of time at each risk scalar level: >0.8, 0.5-0.8, 0.2-0.5, <0.2

**Function:** `print_per_ticker_metrics(ticker_signals: dict, ticker_data: dict, adjusted_weights: pd.DataFrame) -> None`

For each ticker, compute and print:
- Ticker name
- Average sig_mr, sig_trend, sig_composite
- Average weight
- Number of trades (periods where weight > 0.01)
- Ticker's standalone return contribution

---

#### `plotting.py`

**Function:** `plot_portfolio_overview(results: dict) -> matplotlib.figure.Figure`

Figure with 1 row, 3 columns:
1. **Equity curve:** portfolio equity (black solid) vs SPY equity (gray dashed). Y-axis: "Equity (x)". Title: "Portfolio Equity vs SPY".
2. **Drawdown:** filled red area showing drawdown_series. Y-axis: "Drawdown". Horizontal dashed line at dd_full_deleverage threshold. Title: "Portfolio Drawdown".
3. **Risk scalar:** line plot of risk_scalar over time. Y-axis: "Risk Scalar". Title: "Risk Exposure Over Time". Color: dark blue.

Figure size: (22, 6). Suptitle: "Trader v5.0 — Portfolio Overview".

**Function:** `plot_per_ticker_grid(results: dict) -> matplotlib.figure.Figure`

Figure with 2 rows, 4 columns (one subplot per ticker).

For each ticker subplot:
- Primary y-axis: Close price (gray, thin)
- Secondary y-axis: adjusted_weight for that ticker (black, thicker)
- Background shading: green when sig_composite > 0.5, yellow when 0.1-0.5, white when < 0.1
- Title: "{TICKER} | Avg Weight: {x:.1%}"

Figure size: (24, 10). Suptitle: "Trader v5.0 — Per-Ticker Allocation".

**Function:** `plot_strategy_analysis(results: dict) -> matplotlib.figure.Figure`

Figure with 1 row, 3 columns:
1. **Signal decomposition:** For one representative ticker (first in list), plot sig_mr (blue) and sig_trend (orange) over time. Y-axis: "Signal Strength". Title: "Signal Decomposition ({ticker})".
2. **Final equity bar chart:** Per-ticker final equity contribution. Green bars for positive, red for negative. Horizontal line for SPY final equity. Title: "Contribution by Ticker".
3. **Rolling 120-day Sharpe:** Portfolio (black) vs SPY (gray dashed). Y-axis: "Sharpe". Horizontal line at 0. Title: "Rolling 120-day Sharpe".

Figure size: (22, 6). Suptitle: "Trader v5.0 — Strategy Analysis".

All plots use `plt.tight_layout()`.

---

#### `validation.py`

**Function:** `walk_forward_test(ticker_data: dict[str, pd.DataFrame], cfg: dict) -> dict`

Implements expanding-window walk-forward testing:

1. Determine the full date range from `ticker_data`.
2. Set `train_start` = first date. Set `train_end` = `train_start + cfg["wf_train_years"]` years.
3. Set `test_start` = `train_end`. Set `test_end` = `test_start + cfg["wf_test_months"]` months.
4. For each walk-forward window:
   a. Train period: `[train_start, train_end)`. Test period: `[test_start, test_end)`.
   b. Subset all ticker data to the train period. Run `indicators.compute_indicators` and `signals.compute_signals` on the train subset. This is used ONLY for reporting in-sample metrics — the parameters are fixed from CONFIG, not optimized.
   c. Subset all ticker data to the test period. Run the full backtest on the test period: `backtest.run_backtest(test_data, test_signals, cfg)`.
   d. Record: window number, test_start, test_end, test Sharpe, test CAGR, test MDD, test annual turnover.
   e. Advance: `train_end += cfg["wf_test_months"]` months. `test_start = train_end`. `test_end = test_start + cfg["wf_test_months"]` months.
   f. Continue until `test_end` exceeds the last date in the data.
5. Also compute full-sample backtest metrics for comparison.

Returns dict:
```python
{
    "windows": [
        {"window": 1, "test_start": date, "test_end": date, "sharpe": float, "cagr": float, "mdd": float, "turnover": float},
        ...
    ],
    "full_sample_metrics": dict,
    "oos_aggregate_sharpe": float,  # Sharpe of concatenated OOS returns
    "oos_aggregate_cagr": float,
    "oos_aggregate_mdd": float,
    "sharpe_stability": float,  # std of per-window Sharpes / mean of per-window Sharpes
}
```

**Function:** `print_validation_report(validation_results: dict) -> None`

Prints a formatted table of walk-forward results:
- Per-window: test period, Sharpe, CAGR, MDD
- Aggregate OOS metrics
- Full-sample vs OOS comparison (ratio of OOS Sharpe to full-sample Sharpe; values close to 1.0 indicate low overfitting)
- Sharpe stability metric

---

#### `main.py`

**Function:** `main() -> None`

Execution order:

1. `cfg = CONFIG.copy()`
2. Print "Trader v5.0 — Loading data..."
3. `ticker_data = data.load_data(cfg)`
4. `ticker_data = data.align_dates(ticker_data)`
5. Separate benchmark: `spy_data = ticker_data.pop(cfg["benchmark"])`
6. `available_tickers = [t for t in cfg["tickers"] if t in ticker_data]`
7. Validate: if `len(available_tickers) == 0` or `spy_data is None`, print error and return.
8. Put SPY back for backtest use: `ticker_data[cfg["benchmark"]] = spy_data`
9. Print f"Computing indicators for {len(available_tickers)} tickers..."
10. `ticker_signals = {}` — for each ticker in available_tickers:
    - `df = ticker_data[ticker].copy()`
    - `df = indicators.compute_indicators(df, cfg)`
    - `df = signals.compute_signals(df, cfg)`
    - `ticker_signals[ticker] = df`
11. Print "Running backtest..."
12. `results = backtest.run_backtest(ticker_data, ticker_signals, cfg)`
13. Print "Computing metrics..."
14. `met = metrics.compute_all_metrics(results["portfolio_ret"], results["portfolio_equity"], results["spy_ret"], results["spy_equity"], results["turnover"], results["costs"], results["trade_log"])`
15. `reporting.print_report(met, results, cfg)`
16. `reporting.print_per_ticker_metrics(ticker_signals, ticker_data, results["adjusted_weights"])`
17. Print "Running walk-forward validation..."
18. `val = validation.walk_forward_test(ticker_data, cfg)`
19. `validation.print_validation_report(val)`
20. Print "Generating plots..."
21. `fig1 = plotting.plot_portfolio_overview(results)`
22. `fig2 = plotting.plot_per_ticker_grid(results)`
23. `fig3 = plotting.plot_strategy_analysis(results)`
24. `plt.show()`
25. Print "Done."

---

## 12. VALIDATION & EVALUATION PLAN

### 12.1 Performance Metrics

| Metric | Formula | Target |
|---|---|---|
| Sharpe Ratio | mean(ret) × 252 / (std(ret) × sqrt(252)) | > 0.5 |
| Sortino Ratio | mean(ret) × 252 / (downside_std × sqrt(252)) | > 0.7 |
| CAGR | (final_equity / initial_equity) ^ (252/n) - 1 | > 5% |
| Profit Factor | sum(positive ret) / abs(sum(negative ret)) | > 1.2 |

### 12.2 Risk Metrics

| Metric | Formula | Target |
|---|---|---|
| Max Drawdown | min of drawdown series | > -15% |
| Calmar Ratio | CAGR / abs(MDD) | > 0.5 |
| Average Exposure | mean(daily sum of weights) | 30-70% |
| Maximum Single-Day Loss | min(daily returns) | > -5% |

### 12.3 Stability Metrics

| Metric | Computation | Target |
|---|---|---|
| Sharpe Stability | std(per-window Sharpe) / mean(per-window Sharpe) | < 1.0 |
| OOS Sharpe Ratio | Sharpe of concatenated OOS returns | > 0.3 |
| OOS/IS Sharpe Ratio | OOS Sharpe / Full-sample Sharpe | > 0.6 |
| Percentage of OOS windows with Sharpe > 0 | count(window Sharpe > 0) / total windows | > 60% |

### 12.4 Overfitting Diagnostics

1. **Parameter sensitivity:** For each of the 4 free parameters (bb_zscore_cap, trend_atr_norm_cap, signal_mr_weight, signal_trend_weight), vary by ±30% and re-run the full backtest. The Sharpe ratio should not drop by more than 30% relative. (Not auto-implemented; manual check by user.)

2. **Walk-forward consistency:** The ratio of OOS Sharpe to IS Sharpe should be > 0.6. If lower, overfitting is likely.

3. **Turnover reasonableness:** Annual turnover should be < 500% (i.e., the portfolio turns over less than 5x per year). Higher values suggest signal noise is driving unnecessary trading.

### 12.5 Walk-Forward Plan

- **Training window:** 5 years expanding (i.e., train on all data up to the test start)
- **Test window:** 6 months
- **Number of windows:** approximately (total_years - 5) × 2 windows (e.g., for 1998-2026: ~42 windows)
- **Metric per window:** Sharpe, CAGR, MDD, turnover
- **Aggregate:** concatenate all OOS return series and compute single Sharpe/CAGR/MDD

### 12.6 Stress Testing

The walk-forward test implicitly stress-tests across major market events:
- Dot-com crash (2000-2002)
- Financial crisis (2007-2009)
- COVID crash (2020)
- 2022 rate hike cycle

No separate stress testing module is needed. The per-window walk-forward results will show performance during each of these periods.

---

## PLANNING COMPLETE. AWAITING IMPLEMENTATION MODEL.
