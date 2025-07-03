
import pandas as pd  # Add this at the top

class WhaleWatcher:
    def detect_spoofing(self, df):
        """Testable version with data validation"""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        last_candle = df.iloc[-1]
        return (
            (last_candle['volume'] > 3 * df['volume'].mean()) and 
            (abs(last_candle['close'] - last_candle['open']) / last_candle['open'] < 0.005)
        )

    def detect_hidden_liquidity(self, df):
        """Spot iceberg orders using tape reading"""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        return (df['volume'].mean() * 1.5 < df['volume'].iloc[-1]) and \
               (df['close'].pct_change().iloc[-1] < 0.001)

    def detect_quote_stuffing(self, df):
        """Identify rapid order cancellations"""
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        return df['trades'].rolling(5).std().iloc[-1] > 1000
