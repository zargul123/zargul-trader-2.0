import json
import numpy as np
from textblob import TextBlob

class GuruDetector:
    def __init__(self):
        with open('guru_strategies.json') as f:
            self.strategies = json.load(f)

    def find_patterns(self, df, news):
        insights = []

        # Volume Patterns
        if self._high_volume_breakout(df):
            insights.append(next(s for s in self.strategies if "Volume Breakout" in s['topic']))
        if self._smart_money_accumulation(df):
            insights.append(next(s for s in self.strategies if "Smart Money" in s['topic']))

        # Technical Patterns
        if self._minervini_volatility(df):
            insights.append(next(s for s in self.strategies if "Volatility Contraction" in s['topic']))
        if self._wyckoff_accumulation(df):
            insights.append(next(s for s in self.strategies if "Accumulation/Distribution" in s['topic']))

        # News Analysis
        for headline in news:
            analysis = TextBlob(headline)
            for strategy in self.strategies:
                if any(kw.lower() in headline.lower() for kw in strategy['trigger_keywords']):
                    insights.append(strategy)

        return insights

    def _high_volume_breakout(self, df):
        return (df['vol_spike'].iloc[-1] > 3.0 and
                df['close'].iloc[-1] > df['high'].iloc[-5:-1].max())

    def _smart_money_accumulation(self, df):
        return (df['cmf'].iloc[-3:].mean() > 0.25 and
                df['vol_trend'].iloc[-1] > 0.05)

    def _minervini_volatility(self, df):
        return (df['close'].std() < 0.02 and 
                df['volume'].iloc[-1] > df['volume'].mean())

    def _wyckoff_accumulation(self, df):
        return (df['low'].iloc[-3] < df['low'].iloc[-2] and 
                df['low'].iloc[-1] < df['low'].iloc[-2] and 
                df['volume'].iloc[-1] > df['volume'].mean())