import numpy as np
import pandas as pd

class MarketRegimeFilter:
    """
    Analyzes the market's character to determine if it is Trending, Ranging, or Chaotic.
    This acts as a master filter to prevent trading in unfavorable conditions.
    """
    def __init__(self):
        """
        Initializes the filter with a neutral starting value for the smoothed entropy.
        The EMA state is held here to persist across multiple analysis cycles for a single asset.
        """
        self.entropy_ema = 1.0  # Start with a neutral value

    def _calculate_shannon_entropy(self, returns: pd.Series, window: int) -> float:
        """
        Calculates the Shannon Entropy for a given series of price returns.

        Args:
            returns: A pandas Series of percentage returns.
            window: The number of recent returns to use for the calculation.

        Returns:
            The calculated entropy value. A higher value indicates more randomness.
        """
        if returns.empty:
            return 0.0
        
        # Use the most recent `window` of returns
        recent_returns = returns.tail(window)
        
        # Discretize returns into bins
        # Bins: [-inf, -0.5%, 0%, 0.5%, +inf] -> strong down, down, up, strong up
        bins = pd.cut(recent_returns, bins=[-np.inf, -0.005, 0, 0.005, np.inf], labels=False, right=False)
        
        # Calculate probabilities of each bin
        counts = np.bincount(bins, minlength=4)
        probabilities = counts / len(recent_returns)
        
        # Filter out zero probabilities to avoid log(0) errors
        probabilities = probabilities[probabilities > 0]
        
        # Shannon Entropy formula
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        return entropy

    def get_regime(self, df: pd.DataFrame, adx_threshold: float, entropy_threshold: float, entropy_window: int, smoothing_alpha: float) -> str:
        """
        Determines the current market regime based on ADX and smoothed Shannon Entropy.

        Args:
            df: The market data DataFrame, must contain 'adx' and 'close' columns.
            adx_threshold: The ADX value above which the market is considered trending.
            entropy_threshold: The entropy value above which the market is considered chaotic.
            entropy_window: The lookback window for the entropy calculation.
            smoothing_alpha: The smoothing factor for the entropy EMA.

        Returns:
            A string representing the current market regime: "Trending", "Ranging", or "Chaotic".
        """
        # --- 1. Get Raw Indicator Values ---
        last_adx = df['adx'].iloc[-1]
        price_returns = df['close'].pct_change().dropna()
        
        # --- 2. Calculate Raw Entropy and Apply EMA Smoothing ---
        raw_entropy = self._calculate_shannon_entropy(price_returns, entropy_window)
        
        # Update the smoothed EMA value (stateful part of the class)
        self.entropy_ema = (smoothing_alpha * raw_entropy) + ((1 - smoothing_alpha) * self.entropy_ema)
        
        # --- 3. Classify the Regime (Chaos takes precedence) ---
        if self.entropy_ema > entropy_threshold:
            return "Chaotic"
        
        if last_adx > adx_threshold:
            return "Trending"
            
        return "Ranging"
