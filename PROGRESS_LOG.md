# Zargul Trader — Progress Log

Running record of every training run, backtest, and milestone.
(Restart of the project, July 2026. Code baseline: commit 0304127, Sep 29 2025.)

---

## 2026-07-18 — Phase 1: First retrain of BTC-USD main (SUCCESS)

**Setup**
- Code: Sep-29 baseline + 5 local compatibility fixes (see git log 9afb837..09b26d7)
- Data: 26,279 hourly candles, 2023-07-19 → 2026-07-18 (full 3 years via TwelveData pagination)
- Labels (unified ATR-based): Buy 36.2% / Sell 49.9% / Hold 13.8%
- LunarCrush: unavailable (402 Payment Required) — social columns zero-filled by design
- Hardware: CPU training (DML_VISIBLE_DEVICES=-1; DirectML GPU lacks CudnnRNN op for LSTM)

**Result**
- Trained 96/100 epochs in 663.4s (~11 min)
- Final: train signal accuracy 84.2%, **validation signal accuracy 75.7%**
- val_loss 0.649 vs train loss 0.385 (mild overfit gap — noted)
- Saved: trained_models/BTC-USD_main_model.h5 (999 KB) + scaler. No calibrator (removed by design, Sep 27 fix)

**Next:** Phase 2 — raw signal backtest (`raw_backtest.py --asset BTC-USD --strategy main --confidence 0.50`)

## 2026-07-18 — Phase 2: Raw signal backtest (PASSED)

**Command:** `raw_backtest.py --asset BTC-USD --strategy main --confidence 0.50`
**Window:** 5,000 1h candles, 2025-12-22 → 2026-07-18 (risk filters disabled)

| Metric | Value |
|---|---|
| Total trades | 899 |
| Win rate | 67.52% |
| Sharpe ratio | **4.30** |
| Max drawdown | 3% |
| Profit factor | 1.71 |
| Avg win / loss | +0.86% / -1.05% |

**Verdict:** Fresh model produces a strongly positive raw signal. "0 trades" and calibrator-blocker issues confirmed resolved.
**Caveat (recorded honestly):** window overlaps training data (in-sample) — results are optimistic by construction. True out-of-sample validation deferred to Phase 4 hold-out test + paper trading.

**Next:** Phase 3 — regime-specific optimization (Trending first), tightened search space per commit 0304127.

## 2026-07-19 — Phase 3: Trending-regime optimization (COMPLETE)

**Command:** `optimize_strategy.py --asset BTC-USD --strategy main --regime Trending --trials 100`
**Journey:** 100 trials over 2 sessions (2 system crashes survived via the Optuna sqlite save-game; final 3 trials lost to a brief internet outage — harmless). Memory-exhaustion root cause of the crashes fixed mid-run (pandas_ta cores=0, commit ccc511d).

**Champion (trial 57), score 3.922 (0.7×Sharpe + 0.3×ProfitFactor):**
| Parameter | Old config | Optimized |
|---|---|---|
| min_confidence | 0.60 | **0.75** |
| atr_threshold_multiplier | 1.5 | **2.7** |
| tp_atr_multiplier | 2.0 | **2.8** |
| sl_atr_multiplier | 1.5 | **2.1** |

Saved to optimized_strategies.json + BTC-USD_main_Trending.db (full study).
**Note:** effective backtest window was ~5,000 candles (~7 months) due to the single-request cap in the backtest data path — pagination upgrade for backtests planned before Phase 4.
**Next:** lock champion into config.py (pending user approval) → git tag → Phase 4 hold-out validation.

## 2026-07-19 — Phase 4: HOLD-OUT EXAM — FAILED (attempt 1 of 3)

**Setup:** model retrained with ZARGUL_TRAIN_CUTOFF=2026-04-01 (textbook: Apr 2023 → Apr 1 2026, 26,277 candles, 96 epochs, val acc 80.7%). Exam: simple_backtest, locked champion params, 2,160 unseen candles (2026-04-19 → 2026-07-18).

**Result:** 170 trades, win rate 43.53%, **Sharpe -5.77**, profit factor 0.50, max DD 2%.
**Verdict per pre-agreed table (<0):** curve-fit confirmed for this configuration. In-sample scores (raw 4.30, optimized 3.922) did not survive unseen data.

**Notes for diagnosis:**
- Confound: champion params were tuned on the FULL-history model; exam used a different bake ("new lock, old key"). Model-vs-params contribution not yet separated.
- Val accuracy 80.7% during training vs 43.5% real win rate suggests the random (non-chronological) validation split leaks overlapping sequences — training metrics are not trustworthy generalization signals.
- Next diagnostic: raw-signal exam (filters off) on the same unseen window to test whether the MODEL itself has any out-of-sample edge, independent of strategy params.

This counts as honest attempt #1 of the 3 agreed before shelving.

## 2026-07-19 — Phase 4b: Raw-signal exam on unseen window — ALSO NEGATIVE

**Command:** `raw_backtest.py --asset BTC-USD --strategy main --confidence 0.50 --days 90` (exam model, filters off)
**Window:** 2,160 unseen candles (2026-04-19 → 2026-07-18)

| Metric | Value |
|---|---|
| Trades | 301 |
| Win rate | 43.52% |
| Sharpe | **-6.06** |
| Profit factor | 0.49 |

**Verdict (per pre-agreed table, Sharpe < 0):** the MODEL itself is the memorizer — its out-of-sample signal carries no edge. Note: 43.5% win rate matches the ~42.9% a pure coin-flip signal would produce against the asymmetric TP(2.8×ATR)/SL(2.1×ATR) barriers — out-of-sample predictions are statistically indistinguishable from noise.

**Root-cause hypothesis for attempt 2:** training-time validation is a random split over heavily overlapping sequences → leakage → val acc 80.7% was fiction. The model was never actually forced to generalize, and we had no honest signal during training to notice.

**Attempt 2 direction (deep door):** (1) chronological train/val split so training metrics tell the truth; (2) anti-memorization measures (stronger regularization / simpler capacity / feature pruning); (3) fast honest loop: 11-min bake + 3-min raw hold-out exam as the compass.

## 2026-07-19 — ATTEMPT 2, iteration 1: chronological validation (IMPROVED, NOT PASSED)

**Surgery (commit 21abcb1):** chronological 80/20 split + 20-seq purge gap + train-only scaler fit.
**Bake:** cutoff 2026-04-01; early-stopped at 19 epochs (vs 96 — honest validation detects the generalization ceiling fast). Honest val acc ~50.1% (naive baseline 42.5%; the old 80.7% was leak fiction).
**Raw exam (unseen Apr 20 → Jul 19):** 209 trades, win rate 48.80%, Sharpe **-3.41**, PF 0.65.

**Read:** real improvement over the memorizer (-6.06 → -3.41; 43.5% → 48.8% win) — the model now carries a small genuine signal, but not enough to overcome loss asymmetry. Still < 0.
**Next (per agreed sequence):** iteration 2 — simpler capacity / stronger dropout in MODEL_HYPERPARAMS (current values came from the leaky optimize_model study and are untrustworthy anyway).

## 2026-07-19 — ATTEMPT 2, iteration 2: small brain + high dropout (MARGINAL GAIN, KEY CLUE FOUND)

**Change (commit d56e308):** units 89→32, dropout 0.24→0.45.
**Raw exam (same hidden window):** 229 trades, win rate **50.22%**, Sharpe **-3.17**, PF 0.68, avg win +0.70% vs avg loss -1.04%.

**Key observation:** win rate is now well above the ~42.9% coin-flip baseline — the model carries genuine out-of-sample signal. The remaining money-leak is payoff asymmetry: wins average smaller than losses DESPITE TP (2.8×ATR) being farther than SL (2.1×ATR). Prime suspect: trailing stop (activates at 0.5% profit, trails 0.3%) capping winners at ~0.7% while losers run to full SL.

**Next:** inspect backtest engine exit logic (read-only), then a no-retrain experiment: same exam with trailing stop disabled to isolate the payoff leak.

## 2026-07-19 — ATTEMPT 2, diagnostics complete: the 1h picture is now fully understood

**Correction:** trailing-stop suspect cleared — backtest engine has no trailing logic; exits are pure SL/TP barrier checks.
**Strategy exam (locked 0.75-confidence params + honest model): 0 trades** — the honest model's confidence lives at ~0.47-0.62; the 0.75 threshold was calibrated to the old memorizer's inflated certainty ("new lock, old key").
**Confidence ladder (raw, hidden window):** 0.50→50.2%/-3.17 | 0.55→50.9%/-3.06 | 0.60→50.0%/-3.09 | 0.65→42%/-2.45 (n=19). **Flat: confidence does not rank trade quality OOS.**

**Complete 1h diagnosis:**
1. Model has a real but thin directional edge (~50% wins vs 42.9% coin-flip baseline)
2. Economics killed by costs: ~0.25% round trip vs ~1% ATR barriers (avg win +0.70 < avg loss -1.04)
3. No internal mechanism (confidence) to select better trades
→ The 1h timeframe with current features cannot clear the cost hurdle.

**Next (per agreed escalation): 4h timeframe** — ATR barriers ~2-3× larger, so fixed costs shrink from ~30% of the target to ~10%; user's own experience ("swing was giving good results") points the same way.

## 2026-07-19 — EXPERIMENT B (wider 1h exits): FAILED, reverted

**Test:** tp 2.8→4.0, sl 2.1→2.5, same honest model, hidden window. Result: 190 trades, win 46.32%, Sharpe **-3.54** (vs -3.17 baseline), PF 0.64. Avg win rose (+0.82) as theorized but not enough; wider stop raised avg loss (-1.11). Conclusion: the 1h edge cannot pay for itself at any exit geometry — timeframe escalation confirmed as the right move. Config reverted to 2.8/2.1 (git revert of f7c8775).
**Decision per pre-agreed gate:** GO Path A — 4h btc-swing with honest anti-memorization brain.

## 2026-07-19 — PATH A: 4h btc-swing HOLD-OUT EXAM — **PASSED** 🟢 (first honest positive)

**Setup:** btc-swing (4h, seq 90) retrained with honest brain (32u/0.45 drop, commit 8d20d06), cutoff 2026-04-01. Raw exam on hidden window (540 4h candles, ~Apr 20 → Jul 19), confidence 0.50.

| Metric | Value |
|---|---|
| Trades | 42 |
| Win rate | **61.90%** |
| Sharpe | **+3.53** |
| Profit factor | 1.47 |
| Avg win / loss | +2.11% / -2.33% |
| Max drawdown | 4% |

**Why it worked where 1h failed:** targets ~2%+ dwarf the ~0.25% round-trip cost; slower patterns generalize; honest training prevented memorization. Per pre-agreed verdict table (≥2.0): exceptional → proceed toward paper trading.

**Honest caveats:** n=42 trades — wide error bars; single 90-day window/season. Before locking: confirmation exam on a DIFFERENT hidden quarter (cutoff 2026-01-01, exam Jan–Mar) to rule out luck-of-the-quarter.

## 2026-07-19 — 4h CONFIRMATION EXAM (Jan cutoff, 6-month hidden window): NOT CONFIRMED

**Setup:** btc-swing rebaked with cutoff 2026-01-01; raw exam on 1,140 hidden 4h candles (Jan 10 → Jul 19).
**Result:** 106 trades, win rate 44.34%, Sharpe **-0.87**, PF 0.90, avg win +2.64 / avg loss -2.33, max DD 6%.
**Verdict (per bar stated before the run):** the Apr–Jul +3.53 is NOT confirmed. Economics held (payoff ratio > 1, costs negligible); win rate collapsed. Open question: hostile Jan–Mar season vs. lucky April window.
**Next:** dump per-trade history from raw_backtest (small script change), rerun, split results by month/regime to decide between the two stories before spending attempt 3.

## 2026-07-19 — Regime X-ray: the filter cries wolf on 4h

**Regime split of the 106 hidden-window trades:** Chaotic 98 (win 45.9%, -5.2%), Trending 6 (16.7%, -7.5%), Ranging 2. The filter tags ~92% of 4h candles Chaotic — including brilliant Feb (+20%) and catastrophic war-March (-32.6%) alike. As configured (entropy 1.5 / window 50 / ADX 25 — tuned for 1h), it cannot separate good months from deadly ones and would have blocked nearly all trading.
**User context:** Mar–Apr 2026 markets were war-distorted — the exact condition a working regime filter should catch.

**End-of-day battlefield map:**
1. 4h honest model: real edge in normal months (Jan +12.1, Feb +20.0, Jun +5.2 — all under 'Chaotic' tags)
2. War-March inverts the model (1/15 wins, -32.6%)
3. No working selector: confidence flat; regime filter mis-calibrated on 4h
4. NEXT SESSION MISSION: recalibrate regime detection for the 4h timeframe (offline threshold sweep against the tagged trade history / candle data), gate = does it separate Feb-like from Mar-like months without killing trade count. If yes → honest swing-strategy calibration → paper trading path.

## 2026-07-19 — Weather-vane mission: recalibrated 4h regime filter VALIDATED on trade history

**Root cause found:** entropy bins (±0.5% return buckets) are 1h-scaled; 4h moves exceed them constantly → entropy saturates ~2.0 → 92% "Chaotic". Bigger bins tested and rejected (they flag strong-trend Feb as chaos).
**Winning dial (pre-declared family, calibrated ONLY on 2023-2025):** keep ±0.5% bins, raise entropy threshold to the calibration p70 = 1.964 (vs old 1.5).
**2026 monthly test (never used for tuning):** Jan 2% / Feb 0% flagged; war-Mar 53% / Apr 47% flagged. On the real 106 trades: vane keeps 82 (+1.91%) and blocks 24 that lost -15.13% → window flips from -13.2% to +1.9%.
**Circuit breaker: tested and REJECTED** — losses cluster but recoveries follow immediately; pausing misses them (breaker alone -19.5%, worse than nothing). The one positive variant (2-loss/7d, +5.2%) is best-of-4-on-test-window = untrustworthy; not adopted.
**Honest status:** vane turns storm-loss into survival, not yet wealth (kept trades: 46% win, ~breakeven+). Next: implement per-timeframe regime config (4h entropy_threshold 1.964) + fresh-window validation (2025 hold-out with dial recalibrated on 2023-2024 only) before any paper-trading decision.

## 2026-07-19 — FULL-YEAR FRESH VALIDATION: vane FAILED, and the deeper truth surfaced

**Setup:** model cutoff 2025-07-01 (never saw the test year); exam Jul 2025 → Jul 2026, 357 trades; vane dial p70 calibrated only on pre-Jul-2025 data (1.967).
**Result:** vane OFF -97.7% (43.1% win) | vane ON -72.0% (44.0% win) — fails the pre-declared bar (clearly better AND positive).
**Killer detail:** vane blocked Nov-2025 (+22.6%, best cluster of the year) while keeping Feb-2026 (-34.8%). Same Feb where the Jan-cutoff bake made +20%: **month-level performance flips sign between bakes.**

**Honest conclusion:** the btc-swing model family (current features/labeling) shows no stable out-of-sample edge across a full year. Earlier positives (Apr-window +3.53; Jan/Feb months) were bake-specific variance, not repeatable skill. The 6-month vane success was partly luck of that window.

**Status vs. agreed framework:** attempt 2's ladder (honest training → small brain → timeframe escalation → regime vane) is exhausted. Attempt 3 (last before shelving) would require fundamentally new inputs: v3 features / different labeling horizon / historical sentiment — not parameter tweaks.

## 2026-07-19 — MASTER PLAN AGREED (user + Claude + Kimi adviser synthesis)

**User's true goal (stated explicitly):** a personal signals system — helps him work charts, gives signals with SL/TP/size, ~40% wrong acceptable, auto-execution NOT required. Formulas OR training both acceptable as signal sources.

**PHASE 1 — Complete the verdict (now):** honest-brain config for ETH/SOL swing → bake each with cutoff 2025-07-01 → full-year exam each. Gate: any asset ≥ 0 → paper-trade candidate; all < 0 → AI-prediction chapter formally closed with git tag.
**PHASE 2 — New ship:** new folder + NEW git repo (user's rule: strategy change = new program, port only important code). Port: data engine, indicators, recalibrated regime vane, risk engine, backtest engine + honest-lab tools, journal, dashboard. LSTM machinery stays in this museum repo.
**PHASE 3 — Guru Signal Library:** 3-5 documented legends' rule-sets (Turtle/Donchian, Bollinger squeeze, Wilder, Minervini template, Connors RSI-2) as formulas — no training, no bake variance. Each gets honest hold-out stat cards + regime splits. (Seed exists: guru_strategies.json / guru_wisdom.py from user's v1.)
**PHASE 4 — Living cockpit:** daily brief (trend/momentum/vol/regime/news headlines), signal alerts with stat cards, SL/TP/size calculator, journal tracking system + user decisions; 4-8 week live-signal review with same honesty. News = live context first; historical-news training only maybe later.

## 2026-07-19 — Phase 1 verdict, ETH: NEGATIVE

ETH-USD swing, honest brain, cutoff 2025-07-01 (trained 2022-07→2025-07, 50s bake). Full-year exam (2,280 hidden 4h candles): 350 trades, win 46.00%, Sharpe **-0.72**, PF 0.92, avg +3.40/-3.14. Closest to break-even of all full-year tests, but fails the pre-agreed ≥0 gate. One exam remains: SOL.

## 2026-07-19 — PHASE 1 COMPLETE: THE PREDICTION CHAPTER CLOSES

SOL-USD swing, honest brain, cutoff 2025-07-01. Full-year exam: 351 trades, win 44.16%, Sharpe **-0.66**, PF 0.93.

**FINAL VERDICT MATRIX (all full-year hidden exams, honest training):**
| Asset/TF | Sharpe |
|---|---|
| BTC 1h | -3.17 |
| BTC 4h | -2.25 |
| ETH 4h | -0.72 |
| SOL 4h | -0.66 |

Per the gate agreed in advance by user + both advisers: **the AI-prediction chapter is formally closed.** No further LSTM/prediction attempts in this repo. What this year of work produced and keeps: a complete honest laboratory (chronological validation, hold-out cutoffs, 15-minute bake+exam loop, per-trade X-rays, regime recalibration methodology), a fully documented negative result across 3 assets, and the discipline that prevented a single dollar from being risked on fiction.

**Tomorrow: Phase 2 — the new ship.** New repo, ported organs (data engine, indicators, vane, risk engine, backtest lab, journal, dashboard), and the Guru Signal Library mission: formula-based signals with honest stat cards, serving the user's true goal — a personal signals cockpit.
