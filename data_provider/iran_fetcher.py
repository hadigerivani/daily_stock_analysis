# data_provider/iran_fetcher.py
import pandas as pd
from datetime import datetime, timedelta
from .base import BaseFetcher
import logging

# Note: This library must be installed with pip install pytse-client
import pytse_client as tse

logger = logging.getLogger(__name__)

class IranFetcher(BaseFetcher):
    def __init__(self):
        super().__init__()
        self.name = "IranFetcher"
        self.priority = 2  # Set priority

    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch raw data from pytse_client for Tehran Stock Exchange.
        This is the abstract method required by BaseFetcher.
        """
        try:
            logger.info(f"[IranFetcher] Fetching raw data for {stock_code} from {start_date} to {end_date}")

            # Download data from pytse_client
            data = tse.download(symbols=stock_code, write_to_csv=False)

            if stock_code not in data or data[stock_code].empty:
                logger.warning(f"[IranFetcher] No data found for {stock_code}")
                return pd.DataFrame()

            df = data[stock_code].copy()

            # Ensure required columns exist
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"[IranFetcher] Missing column {col} in data for {stock_code}")
                    return pd.DataFrame()

            # Convert date column
            df['date'] = pd.to_datetime(df['date'])

            # Filter by date range
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
            df = df.loc[mask]

            if df.empty:
                logger.warning(f"[IranFetcher] No data in date range for {stock_code}")
                return pd.DataFrame()

            return df

        except Exception as e:
            logger.error(f"[IranFetcher] Error fetching raw data for {stock_code}: {e}")
            return pd.DataFrame()

    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        Normalize the raw DataFrame to standard columns.
        This is the abstract method required by BaseFetcher.
        """
        if df.empty:
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

        df = df.copy()

        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Standard columns: date, open, high, low, close, volume
        # Rename columns to standard names if needed
        rename_map = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        df = df.rename(columns=rename_map)

        # Ensure all standard columns exist
        standard_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in standard_cols:
            if col not in df.columns:
                df[col] = None

        # Keep only standard columns
        df = df[standard_cols]

        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Sort by date
        df = df.sort_values('date', ascending=True).reset_index(drop=True)

        logger.info(f"[IranFetcher] Normalized data for {stock_code}: {len(df)} rows")
        return df
