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
