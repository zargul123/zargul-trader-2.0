
import pandas as pd
from scripts.core.market_manipulation_detector import WhaleWatcher

def main():
    # Create test data with known patterns
    df = pd.DataFrame({
        'open': [100, 101, 100.5, 102, 101.5],
        'close': [100, 101, 100.5, 102, 101.5],
        'volume': [100, 150, 300, 250, 400],
        'trades': [50, 60, 1200, 70, 1300]
    })

    ww = WhaleWatcher()
    
    # Run tests
    print("\n🔍 Market Manipulation Tests")
    print("----------------------------")
    print("🐋 Hidden Liquidity:", ww.detect_hidden_liquidity(df))
    print("📊 Quote Stuffing:", ww.detect_quote_stuffing(df))
    print("🎯 Spoofing:", ww.detect_spoofing(df))

if __name__ == "__main__":
    main()
