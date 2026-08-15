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

    def get_daily_data(self, stock_code: str, start_date: str, end_date: str):
        """
        Fetch historical daily data for a Tehran Stock Exchange symbol.
        stock_code: symbol code, e.g. 'ولملت' or 'فولاد'
        """
        try:
            logger.info(f"[IranFetcher] Getting daily data for {stock_code}")
            
            # Fetch data from pytse_client
            data = tse.download(symbols=stock_code, write_to_csv=False)
            
            if stock_code not in data or data[stock_code].empty:
                logger.warning(f"[IranFetcher] No data found for {stock_code}")
                return None
                
            df = data[stock_code]
            
            # Rename columns to standard project format
            df = df.rename(columns={
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            df['date'] = pd.to_datetime(df['date'])
            
            # Filter by date range
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
            df = df.loc[mask]
            
            if df.empty:
                logger.warning(f"[IranFetcher] No data in date range for {stock_code}")
                return None
                
            return df

        except Exception as e:
            logger.error(f"[IranFetcher] Error fetching data for {stock_code}: {e}")
            return None

    def get_realtime_data(self, stock_code: str):
        """Fetch real-time data (if needed)"""
        # Currently not implemented; can be added later if needed
        logger.info(f"[IranFetcher] Realtime data not implemented for {stock_code}")
        return None
