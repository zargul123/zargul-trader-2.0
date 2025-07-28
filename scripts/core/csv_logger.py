import pandas as pd
import os
from threading import Lock

# Define the exact column order for the CSV files
JOURNAL_COLUMNS = [
    'trade_id', 'timestamp', 'asset', 'direction', 'entry_price',
    'confidence', 'pct_change', 'strategy_name', 'stop_loss',
    'take_profit', 'close_price', 'outcome'
]
POSITIONS_COLUMNS = [
    'trade_id', 'asset', 'direction', 'entry_price', 'strategy_name',
    'timestamp', 'stop_loss', 'take_profit'
]

class CsvLogger:
    def __init__(self, journal_path='trading_journal.csv', positions_path='open_positions.csv'):
        self.journal_path = journal_path
        self.positions_path = positions_path
        self.lock = Lock()
        self._initialize_files()

    def _initialize_files(self):
        """Create CSV files with headers if they don't exist."""
        with self.lock:
            if not os.path.exists(self.journal_path) or os.path.getsize(self.journal_path) == 0:
                pd.DataFrame(columns=JOURNAL_COLUMNS).to_csv(self.journal_path, index=False)

            if not os.path.exists(self.positions_path) or os.path.getsize(self.positions_path) == 0:
                pd.DataFrame(columns=POSITIONS_COLUMNS).to_csv(self.positions_path, index=False)

    def add_trade(self, trade_data):
        """
        Appends a new trade to the trading_journal.csv and open_positions.csv,
        ensuring correct column order.
        """
        with self.lock:
            try:
                # Use a dictionary comprehension to handle potentially missing keys gracefully
                journal_data_point = {col: trade_data.get(col) for col in JOURNAL_COLUMNS}
                journal_entry = pd.DataFrame([journal_data_point], columns=JOURNAL_COLUMNS)
                journal_entry.to_csv(self.journal_path, mode='a', header=False, index=False)

                positions_data_point = {col: trade_data.get(col) for col in POSITIONS_COLUMNS}
                positions_entry = pd.DataFrame([positions_data_point], columns=POSITIONS_COLUMNS)
                positions_entry.to_csv(self.positions_path, mode='a', header=False, index=False)
                
                print("   └ 📝 Trade successfully logged to CSV files.")
                return True
            except Exception as e:
                print(f"   └ ❌ CRITICAL: Failed to write trade to CSV files: {e}")
                return False

    def close_trade(self, trade_id, close_price, outcome):
        """
        Updates the journal with the outcome and removes the trade from open_positions.csv,
        ensuring correct column order on write.
        """
        with self.lock:
            try:
                # Update the permanent journal
                journal_df = pd.read_csv(self.journal_path)
                trade_index = journal_df[journal_df['trade_id'] == trade_id].index
                if not trade_index.empty:
                    journal_df.loc[trade_index, 'close_price'] = close_price
                    journal_df.loc[trade_index, 'outcome'] = outcome
                    journal_df.to_csv(self.journal_path, index=False, columns=JOURNAL_COLUMNS)

                # Remove from open positions
                positions_df = pd.read_csv(self.positions_path)
                positions_df = positions_df[positions_df['trade_id'] != trade_id]
                positions_df.to_csv(self.positions_path, index=False, columns=POSITIONS_COLUMNS)

                print(f"   └ 📝 Trade {trade_id} successfully closed in CSV files.")
                return True
            except Exception as e:
                print(f"   └ ❌ CRITICAL: Failed to close trade in CSV files: {e}")
                return False