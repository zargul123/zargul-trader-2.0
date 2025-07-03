
import pandas as pd
from scripts.core.execution_engine import ExecutionEngine
from scripts.core.market_manipulation_detector import WhaleWatcher

def test_execution_engine():
    print("\n🔄 Testing Execution Engine")
    print("---------------------------")
    ee = ExecutionEngine()
    test_order = {
        'asset': 'BTC',
        'direction': 'long',
        'size': 0.1,
        'entry': 30000
    }
    ee.execute_order(test_order)

def test_market_manipulation():
    print("\n🔍 Testing Market Manipulation Detection")
    print("---------------------------------------")
    df = pd.DataFrame({
        'volume': [100, 200, 300, 400, 500],
        'close': [100, 101, 102, 103, 104],
        'open': [99, 100, 101, 102, 103],
        'trades': [50, 60, 70, 80, 90]
    })
    
    ww = WhaleWatcher()
    print('🎯 Spoofing Alert:', ww.detect_spoofing(df))
    print('🐋 Hidden Liquidity:', ww.detect_hidden_liquidity(df))
    print('📊 Quote Stuffing:', ww.detect_quote_stuffing(df))

if __name__ == "__main__":
    test_execution_engine()
    test_market_manipulation()
