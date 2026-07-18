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
