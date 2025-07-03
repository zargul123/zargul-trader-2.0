import ccxt
from scripts.config import BINANCE_MAPPING  # Absolute import path

class ExecutionEngine:
    def __init__(self):
        self.paper_trading = True  # Start in safe mode
        self.stop_loss = 0

    def execute_order(self, signal):
        """Smart order routing with anti-detection"""
        if self.paper_trading:
            print(f"📜 Paper Trading: {signal}")
            return

        # Live trading logic
        order = self._iceberg_order(
            symbol=signal['asset'],
            side=signal['direction'],
            amount=signal['size'],
            price=signal['entry']
        )
        self._obfuscate_timing()

    def _obfuscate_timing(self):
        """Random delays to avoid pattern detection"""
        import random
        time.sleep(random.uniform(0.1, 1.5))

    def _iceberg_order(self, symbol, side, amount, price):
        """Execute iceberg order via Binance"""
        return self.exchange.create_order(
            symbol=BINANCE_MAPPING[symbol],
            type='ICEBERG',
            side=side,
            amount=amount,
            price=price,
            params={'icebergQty': amount * 0.7}
        )

    def _limit_order(self, symbol, side, amount, price):
        """Execute limit order via Binance"""
        return self.exchange.create_order(
            symbol=BINANCE_MAPPING[symbol],
            type='LIMIT',
            side=side,
            amount=amount,
            price=price
        )

class HedgeFundExecution(ExecutionEngine):
    def execute_order(self, signal):

    def update_stoploss(self, current_price, trail_pct=0.5):
        """Update trailing stop-loss that only moves up"""
        new_sl = current_price * (1 - trail_pct/100)
        self.stop_loss = max(self.stop_loss, new_sl)  # Only moves up

        if signal['confidence'] > 0.7:
            # Pyramid scaling - 50% now, 30% at +0.5%, 20% at +1%
            self._iceberg_order(symbol=signal['asset'],
                              side=signal['direction'],
                              amount=signal['size']*0.5,
                              price=signal['entry'])

            take_profit = signal['tp'] * 1.005 if signal['direction'] == 'long' else signal['tp'] * 0.995
            self._limit_order(symbol=signal['asset'],
                            side=signal['direction'],
                            amount=signal['size']*0.3,
                            price=take_profit)