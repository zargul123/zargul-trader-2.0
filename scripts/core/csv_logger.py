import pandas as pd
import os
import csv
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
                with open(self.journal_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(JOURNAL_COLUMNS)

            if not os.path.exists(self.positions_path) or os.path.getsize(self.positions_path) == 0:
                with open(self.positions_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(POSITIONS_COLUMNS)

    def add_trade(self, trade_data):
        """
        Appends a new trade to the trading_journal.csv and open_positions.csv
        using the native csv library for speed and reliability.
        """
        with self.lock:
            try:
                # Append to journal
                journal_row = [trade_data.get(col) for col in JOURNAL_COLUMNS]
                with open(self.journal_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(journal_row)
                    f.flush()  # Force write to disk
                    os.fsync(f.fileno()) # Ensure it's written

                # Append to open positions
                positions_row = [trade_data.get(col) for col in POSITIONS_COLUMNS]
                with open(self.positions_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(positions_row)
                    f.flush() # Force write to disk
                    os.fsync(f.fileno()) # Ensure it's written

                print("   └ 📝 Trade successfully logged to CSV files.")
                return True
            except Exception as e:
                print(f"   └ ❌ CRITICAL: Failed to write trade to CSV files: {e}")
                return False

    def close_trade(self, trade_id, close_price, outcome):
        """
        Updates the journal and removes a trade from open_positions.csv using pandas,
        as this requires modifying the entire file.
        """
        with self.lock:
            try:
                # --- Update Journal (Pandas is better for this) ---
                journal_df = pd.read_csv(self.journal_path)
                trade_index = journal_df[journal_df['trade_id'] == trade_id].index
                if not trade_index.empty:
                    journal_df.loc[trade_index, 'close_price'] = close_price
                    journal_df.loc[trade_index, 'outcome'] = outcome
                    journal_df.to_csv(self.journal_path, index=False, columns=JOURNAL_COLUMNS)

                # --- Remove from Open Positions (Pandas is better for this) ---
                positions_df = pd.read_csv(self.positions_path)
                positions_df = positions_df[positions_df['trade_id'] != trade_id]
                positions_df.to_csv(self.positions_path, index=False, columns=POSITIONS_COLUMNS)

                print(f"   └ 📝 Trade {trade_id} successfully closed in CSV files.")
                return True
            except Exception as e:
                print(f"   └ ❌ CRITICAL: Failed to close trade in CSV files: {e}")
                return False