"""
Regime recalibration study for the 4h timeframe (read-only analysis).

The live MarketRegimeFilter bins returns at fixed +/-0.5% edges — sized for
1h candles. Typical 4h moves exceed 0.5%, so returns spread across all four
bins and smoothed entropy saturates near log2(4)=2.0: ~92% of candles get
stamped "Chaotic" regardless of actual conditions.

This script replays the filter's exact math over ~3 years of 4h candles for
a menu of candidate calibrations (bin edge scale x entropy threshold
percentile). Anti-overfit rule: thresholds are derived ONLY from the
2023-2025 calibration period; 2026 is the untouched test period, reported
month by month. A good dial should flag war-distorted Mar-Apr 2026 as
Chaotic while leaving Jan/Feb/Jun tradeable.

Touches nothing in the trading system. Usage:
    python analyze_regime_4h.py
"""
import sys
import os

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

if __name__ == '__main__':
    from scripts.core.data_engine import DataMaster

    WINDOW = 50          # entropy lookback, same as REGIME_CONFIG
    ALPHA = 0.1          # EMA smoothing, same as REGIME_CONFIG
    ADX_THRESHOLD = 25   # unchanged; ADX is scale-free

    EDGE_SCALES = [0.005, 0.0075, 0.010, 0.0125, 0.015]  # bin edge candidates
    PERCENTILES = [70, 80, 85, 90]                        # threshold candidates

    dm = DataMaster()
    df = dm.get_training_data('BTC-USD', '4h', days=1095)
    if df.empty:
        sys.exit('No data — aborting.')

    returns = df['close'].pct_change()

    def entropy_stream(edge):
        """Replay smoothed entropy over the whole series for given bin edge."""
        vals = np.full(len(df), np.nan)
        ema = 1.0
        r = returns.values
        for i in range(WINDOW + 1, len(df)):
            win = r[i - WINDOW:i]
            bins = np.digitize(win, [-edge, 0.0, edge])  # 4 buckets
            counts = np.bincount(bins, minlength=4)
            p = counts / len(win)
            p = p[p > 0]
            raw = -np.sum(p * np.log2(p))
            ema = ALPHA * raw + (1 - ALPHA) * ema
            vals[i] = ema
        return pd.Series(vals, index=df.index)

    calib_mask = df.index < '2026-01-01'
    test_mask = ~calib_mask

    print(f"Data: {len(df)} candles {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"Calibration: {calib_mask.sum()} candles (<2026) | Test: {test_mask.sum()} candles (2026)")
    print(f"ADX>const {ADX_THRESHOLD}: calib {(df['adx'][calib_mask] > ADX_THRESHOLD).mean()*100:.0f}% | 2026 {(df['adx'][test_mask] > ADX_THRESHOLD).mean()*100:.0f}% of candles")

    for edge in EDGE_SCALES:
        ent = entropy_stream(edge)
        calib_ent = ent[calib_mask].dropna()
        line = f"edge ±{edge*100:.2f}% | calib entropy p50={calib_ent.quantile(0.5):.3f} p85={calib_ent.quantile(0.85):.3f}"
        print("\n" + "=" * 90)
        print(line)
        for pct in PERCENTILES:
            thr = calib_ent.quantile(pct / 100)
            chaotic = ent > thr
            monthly = chaotic[test_mask].groupby(df.index[test_mask].to_period('M')).mean() * 100
            cells = " ".join(f"{str(m)[-2:]}:{v:5.1f}%" for m, v in monthly.items())
            print(f"  thr=p{pct} ({thr:.3f}) -> 2026 %Chaotic by month: {cells}")
