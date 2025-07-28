import sqlite3
import pandas as pd
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path='trading_journal.db'):
        """
        Initializes the DatabaseManager, connects to the SQLite database,
        and ensures the necessary tables exist.
        """
        self.db_path = db_path
        self.conn = None
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._create_tables()
        except sqlite3.Error as e:
            print(f"❌ Database connection error: {e}")
            raise

    def _create_tables(self):
        """
        Creates the 'journal' and 'positions' tables if they do not already exist.
        This ensures the database schema is always ready.
        """
        try:
            cursor = self.conn.cursor()
            # Journal Table: A permanent, append-only log of all trade attempts.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS journal (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    confidence REAL,
                    pct_change REAL,
                    strategy_name TEXT,
                    stop_loss REAL,
                    take_profit REAL,
                    close_price REAL,
                    outcome TEXT
                )
            """)
            # Positions Table: A temporary state of currently open trades.
            # A trade is deleted from here when it's closed.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    trade_id TEXT PRIMARY KEY,
                    asset TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    strategy_name TEXT,
                    timestamp TEXT NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    FOREIGN KEY (trade_id) REFERENCES journal (trade_id)
                )
            """)
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"❌ Database table creation error: {e}")
            self.conn.rollback()

    def add_trade(self, trade_data):
        """
        Adds a new trade to the database in a single, atomic transaction.
        - Inserts the full record into the 'journal' table.
        - Inserts the active trade into the 'positions' table.
        """
        try:
            cursor = self.conn.cursor()
            # Add to permanent journal
            cursor.execute("""
                INSERT INTO journal (trade_id, timestamp, asset, direction, entry_price, confidence, pct_change, strategy_name, stop_loss, take_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['trade_id'],
                trade_data['timestamp'].isoformat(),
                trade_data['asset'],
                trade_data['direction'],
                trade_data['entry_price'],
                trade_data.get('confidence'),
                trade_data.get('pct_change'),
                trade_data['strategy_name'],
                trade_data['stop_loss'],
                trade_data['take_profit']
            ))
            # Add to active positions
            cursor.execute("""
                INSERT INTO positions (trade_id, asset, direction, entry_price, strategy_name, timestamp, stop_loss, take_profit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['trade_id'],
                trade_data['asset'],
                trade_data['direction'],
                trade_data['entry_price'],
                trade_data['strategy_name'],
                trade_data['timestamp'].isoformat(),
                trade_data['stop_loss'],
                trade_data['take_profit']
            ))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"❌ Database error adding trade {trade_data.get('trade_id')}: {e}")
            self.conn.rollback()
            return False

    def close_trade(self, trade_id, close_price, outcome):
        """
        Closes an active trade in the database in a single, atomic transaction.
        - Updates the 'journal' with the closing price and outcome.
        - Deletes the trade from the 'positions' table.
        """
        try:
            cursor = self.conn.cursor()
            # Update the journal with the outcome
            cursor.execute("""
                UPDATE journal
                SET close_price = ?, outcome = ?
                WHERE trade_id = ?
            """, (close_price, outcome, trade_id))
            # Remove from active positions
            cursor.execute("DELETE FROM positions WHERE trade_id = ?", (trade_id,))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"❌ Database error closing trade {trade_id}: {e}")
            self.conn.rollback()
            return False

    def load_open_positions(self):
        """

        Loads all currently open positions from the database and returns them
        as a pandas DataFrame, matching the format of the old CSV file.
        """
        try:
            df = pd.read_sql_query("SELECT * FROM positions", self.conn)
            # Convert timestamp back to datetime object for consistency, though not strictly needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
            print(f"❌ Database error loading open positions: {e}")
            # Return an empty DataFrame with the correct columns in case of error
            return pd.DataFrame(columns=['trade_id', 'asset', 'direction', 'entry_price', 'strategy_name', 'timestamp', 'stop_loss', 'take_profit'])

    def __del__(self):
        """Ensures the database connection is closed when the object is destroyed."""
        if self.conn:
            self.conn.close()
