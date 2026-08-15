# data_provider/iran_fetcher.py
import pandas as pd
from datetime import datetime, timedelta
from .base import BaseFetcher
import logging
import pytse_client as tse  # pip install pytse-client

logger = logging.getLogger(__name__)

class IranFetcher(BaseFetcher):
    def __init__(self, priority: int = 2):
        super().__init__(priority=priority)
        self.name = "IranFetcher"

    def get_daily_data(self, stock_code: str, start_date: str, end_date: str):
        """
        دریافت داده‌های تاریخی یک سهم از بورس تهران.
        stock_code: باید کد نماد باشد، مثلاً 'ولملت' یا 'فولاد'
        """
        try:
            logger.info(f"[IranFetcher] Getting daily data for {stock_code}")
            
            # دانلود داده‌های تاریخی برای یک نماد خاص
            # توجه: تابع download لیستی از دیکشنری‌ها برمی‌گرداند
            data = tse.download(symbols=stock_code, write_to_csv=False)
            
            if stock_code not in data or data[stock_code].empty:
                logger.warning(f"[IranFetcher] No data found for {stock_code}")
                return None
                
            df = data[stock_code]
            
            # تبدیل فرمت داده‌ها به فرمت استاندارد پروژه
            df = df.rename(columns={
                'date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # اطمینان از اینکه ستون تاریخ از نوع datetime است
            df['date'] = pd.to_datetime(df['date'])
            
            # فیلتر کردن بر اساس بازه تاریخ
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
        """دریافت داده‌های لحظه‌ای (در صورت نیاز)"""
        # پیاده‌سازی مشابه با استفاده از توابع لحظه‌ای pytse-client
        # (در حال حاضر این کتابخانه پشتیبانی کامل از داده لحظه‌ای دارد)
        pass
