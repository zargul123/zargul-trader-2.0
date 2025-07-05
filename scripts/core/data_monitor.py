import os
import sys
import threading
import time

# Add project root to Python path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.core.safety import armor_get

class DataHealthMonitor:
    def __init__(self):
        self.source_stats = {
            'binance': {'success': 0, 'fail': 0},
            'twelvedata': {'success': 0, 'fail': 0},
            'yahoo': {'success': 0, 'fail': 0}
        }
        self._start_monitor()

    def log_result(self, source, success):
        """Completely bulletproof logging"""
        try:
            source = str(source) if source is not None else 'unknown'
            if source not in self.source_stats:
                self.source_stats[source] = {'success': 0, 'failure': 0}
                
            key = 'success' if success else 'failure'
            self.source_stats[source][key] += 1
        except Exception as e:
            print(f"⚠️ Monitoring error: {str(e)}")

    def _start_monitor(self):
        def monitor_loop():
            while True:
                time.sleep(300)  # Check every 5 minutes
                self._adjust_strategy()

        threading.Thread(target=monitor_loop, daemon=True).start()

    def _adjust_strategy(self):
        """Dynamically adjust data source priority"""
        total_calls = sum(stats['success'] + stats['fail'] 
                       for stats in self.source_stats.values())
        if total_calls == 0:
            return

        # Calculate success rates
        rates = {
            source: armor_get(stats, 'success', 0) / (armor_get(stats, 'success', 0) + armor_get(stats, 'fail', 0) or 1)
            for source, stats in self.source_stats.items()
        }

        # Reorder priority dynamically
        global DATA_SOURCE_PRIORITY
        DATA_SOURCE_PRIORITY = {
            source: rank 
            for rank, source in enumerate(
                sorted(rates.keys(), key=lambda x: -rates[x]), 1)
        }