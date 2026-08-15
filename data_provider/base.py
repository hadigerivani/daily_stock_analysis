# -*- coding: utf-8 -*-
"""
===================================
Data Source Base Class and Manager
===================================

Design Pattern: Strategy Pattern
- BaseFetcher: Abstract base class defining the unified interface
- DataFetcherManager: Strategy manager implementing automatic failover

Anti-blocking strategies:
1. Each fetcher has built-in flow control
2. Automatic fallback to next data source on failure
3. Exponential backoff retry mechanism
"""

import logging
import random
import time
from threading import BoundedSemaphore, RLock, Thread
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Callable, Optional, List, Tuple, Dict, Any

import pandas as pd
import numpy as np
from src.data.stock_index_loader import get_index_stock_name
from src.data.stock_mapping import STOCK_NAME_MAP, is_meaningful_stock_name
from src.services.run_diagnostics import record_provider_run, record_provider_run_started
from .fundamental_adapter import AkshareFundamentalAdapter
from .yfinance_fundamental_adapter import YfinanceFundamentalAdapter
from .realtime_types import CircuitBreaker

# Configure logging
logger = logging.getLogger(__name__)


# === Standard column definitions ===
STANDARD_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']


def unwrap_exception(exc: Exception) -> Exception:
    """
    Follow chained exceptions and return the deepest non-cyclic cause.
    """
    current = exc
    visited = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        next_exc = current.__cause__ or current.__context__
        if next_exc is None:
            break
        current = next_exc

    return current


def summarize_exception(exc: Exception) -> Tuple[str, str]:
    """
    Build a stable summary for logs while preserving the application-layer message.
    """
    root = unwrap_exception(exc)
    error_type = type(root).__name__
    message = str(exc).strip() or str(root).strip() or error_type
    return error_type, " ".join(message.split())


def normalize_stock_code(stock_code: str) -> str:
    """
    Normalize stock code by stripping exchange prefixes/suffixes.

    Accepted formats and their normalized results:
    - '600519'      -> '600519'   (already clean)
    - 'SH600519'    -> '600519'   (strip SH prefix)
    - 'SH.600519'   -> '600519'   (strip SH. prefix)
    - 'SZ000001'    -> '000001'   (strip SZ prefix)
    - 'SZ.000001'   -> '000001'   (strip SZ. prefix)
    - 'BJ920748'    -> '920748'   (strip BJ prefix, BSE)
    - 'BJ.920748'   -> '920748'   (strip BJ. prefix, BSE)
    - 'sh600519'    -> '600519'   (case-insensitive)
    - '600519.SH'   -> '600519'   (strip .SH suffix)
    - '000001.SZ'   -> '000001'   (strip .SZ suffix)
    - '920748.BJ'   -> '920748'   (strip .BJ suffix, BSE)
    - 'HK00700'     -> 'HK00700'  (keep HK prefix for HK stocks)
    - '1810.HK'     -> 'HK01810'  (normalize HK suffix to canonical prefix form)
    - '7203.T'      -> '7203.T'   (keep Japan Yahoo suffix form)
    - '005930.KS'   -> '005930.KS' (keep Korea Yahoo suffix form)
    - 'AAPL'        -> 'AAPL'     (keep US stock ticker as-is)

    This function is applied at the DataProviderManager layer so that
    all individual fetchers receive a clean 6-digit code (for A-shares/ETFs).
    """
    code = stock_code.strip()
    upper = code.upper()

    # Normalize HK prefix to a canonical 5-digit form (e.g. hk1810 -> HK01810)
    if upper.startswith('HK') and not upper.startswith('HK.'):
        candidate = upper[2:]
        if candidate.isdigit() and 1 <= len(candidate) <= 5:
            return f"HK{candidate.zfill(5)}"

    # Strip SH/SZ prefix (e.g. SH600519 -> 600519)
    if upper.startswith(('SH', 'SZ')) and not upper.startswith('SH.') and not upper.startswith('SZ.'):
        candidate = code[2:]
        # Only strip if the remainder looks like a valid numeric code
        if candidate.isdigit() and len(candidate) in (5, 6):
            return candidate

    # Strip dotted SH/SZ prefix (e.g. SH.600519 -> 600519)
    if upper.startswith(('SH.', 'SZ.')):
        candidate = code[3:]
        if candidate.isdigit() and len(candidate) in (5, 6):
            return candidate

    # Strip BJ prefix (e.g. BJ920748 -> 920748)
    if upper.startswith('BJ') and not upper.startswith('BJ.'):
        candidate = code[2:]
        if candidate.isdigit() and len(candidate) == 6:
            return candidate

    # Strip dotted BJ prefix (e.g. BJ.920748 -> 920748)
    if upper.startswith('BJ.'):
        candidate = code[3:]
        if candidate.isdigit() and len(candidate) == 6:
            return candidate

    # Strip .SH/.SZ/.BJ suffix (e.g. 600519.SH -> 600519, 920748.BJ -> 920748)
    # while preserving explicit Yahoo suffix forms for JP/KR.
    if '.' in code:
        base, suffix = code.rsplit('.', 1)
        if suffix.upper() == 'T' and base.isdigit() and len(base) in (4, 5):
            return f"{base}.{suffix.upper()}"
        if suffix.upper() in ('KS', 'KQ') and base.isdigit() and len(base) == 6:
            return f"{base}.{suffix.upper()}"
        if suffix.upper() == 'HK' and base.isdigit() and 1 <= len(base) <= 5:
            return f"HK{base.zfill(5)}"
        if base.upper() in ('SH', 'SS', 'SZ', 'BJ') and suffix.isdigit():
            return suffix
        if suffix.upper() in ('SH', 'SZ', 'SS', 'BJ') and base.isdigit():
            return base

    return code


ETF_PREFIXES = ("51", "52", "56", "58", "15", "16", "18")


def _is_us_market(code: str) -> bool:
    """Check if code is a US stock/index code (without Chinese prefixes)."""
    from .us_index_mapping import is_us_stock_code, is_us_index_code

    normalized = (code or "").strip().upper()
    return is_us_index_code(normalized) or is_us_stock_code(normalized)


def _is_hk_market(code: str) -> bool:
    """
    Determine if code is HK stock.

    Supports `HK00700` and plain 5-digit form (A-share ETF/stocks are usually 6 digits).
    """
    normalized = (code or "").strip().upper()
    if normalized.endswith(".HK"):
        base = normalized[:-3]
        return base.isdigit() and 1 <= len(base) <= 5
    if normalized.startswith("HK"):
        digits = normalized[2:]
        return digits.isdigit() and 1 <= len(digits) <= 5
    if normalized.isdigit() and len(normalized) == 5:
        return True
    return False


def _is_jp_market(code: str) -> bool:
    """Determine if code is a Japan Yahoo Finance suffix (e.g., 7203.T)."""
    normalized = (code or "").strip().upper()
    if not normalized.endswith(".T"):
        return False
    base = normalized[:-2]
    return base.isdigit() and len(base) in (4, 5)


def _is_kr_market(code: str) -> bool:
    """Determine if code is a Korea Yahoo Finance suffix (e.g., 005930.KS / 035720.KQ)."""
    normalized = (code or "").strip().upper()
    if not normalized.endswith((".KS", ".KQ")):
        return False
    base = normalized.rsplit(".", 1)[0]
    return base.isdigit() and len(base) == 6


def _is_etf_code(code: str) -> bool:
    """Determine if code is an A-share ETF fund (conservative rules)."""
    normalized = normalize_stock_code(code)
    return (
        normalized.isdigit()
        and len(normalized) == 6
        and normalized.startswith(ETF_PREFIXES)
    )


def _coerce_chip_metric(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        numeric = float(value)
        if np.isnan(numeric):
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def _is_meaningful_chip_distribution(chip: Any) -> bool:
    """Validate that a provider returned usable core chip metrics."""
    if chip is None:
        return False
    avg_cost = _coerce_chip_metric(getattr(chip, "avg_cost", None))
    concentration_90 = _coerce_chip_metric(getattr(chip, "concentration_90", None))
    concentration_70 = _coerce_chip_metric(getattr(chip, "concentration_70", None))
    return (
        avg_cost is not None
        and avg_cost > 0
        and (
            (concentration_90 is not None and concentration_90 >= 0)
            or (concentration_70 is not None and concentration_70 >= 0)
        )
    )


def _market_tag(code: str) -> str:
    """Return market tag: cn/us/hk/jp/kr."""
    if _is_us_market(code):
        return "us"
    if _is_hk_market(code):
        return "hk"
    if _is_jp_market(code):
        return "jp"
    if _is_kr_market(code):
        return "kr"
    return "cn"


def is_bse_code(code: str) -> bool:
    """
    Check if the code is a Beijing Stock Exchange (BSE) A-share code.

    BSE rules (2026):
    - New format (2024+): 92xxxx main trading codes
    - Historical ranges: 43xxxx, 83xxxx, 87xxxx, 88xxxx
    - Special instruments: 81xxxx convertible bonds, 82xxxx preferred shares
    - Subscription codes: 889xxx
    Note: 900xxx are Shanghai B-shares and must return False.
    """
    c = (code or "").strip().split(".")[0]
    if len(c) != 6 or not c.isdigit():
        return False

    if c.startswith("900"):
        return False

    return c.startswith(("92", "43", "81", "82", "83", "87", "88"))


def is_st_stock(name: str) -> bool:
    """
    Check if the stock is an ST or *ST stock based on its name.

    ST stocks have special trading rules and typically a ±5% limit.
    """
    n = (name or "").upper()
    return 'ST' in n


def is_kc_cy_stock(code: str) -> bool:
    """
    Check if the stock is a STAR Market (科创板) or ChiNext (创业板) stock based on its code.

    - STAR Market: Codes starting with 688
    - ChiNext: Codes starting with 300
    Both have a ±20% limit.
    """
    c = (code or "").strip().split(".")[0]
    return c.startswith("688") or c.startswith("30")


def canonical_stock_code(code: str) -> str:
    """
    Return the canonical (uppercase) form of a stock code.

    This is a display/storage layer concern, distinct from normalize_stock_code
    which strips exchange prefixes. Apply at system input boundaries to ensure
    consistent case across BOT, WEB UI, API, and CLI paths (Issue #355).

    Examples:
        'aapl'    -> 'AAPL'
        'AAPL'    -> 'AAPL'
        '600519'  -> '600519'  (digits are unchanged)
        'hk00700' -> 'HK00700'
    """
    return (code or "").strip().upper()


class DataFetchError(Exception):
    """Base exception for data fetching errors."""
    pass


class RateLimitError(DataFetchError):
    """API rate limit exception."""
    pass


class DataSourceUnavailableError(DataFetchError):
    """Data source unavailable exception."""
    pass


class BaseFetcher(ABC):
    """
    Abstract base class for data sources.

    Responsibilities:
    1. Define a unified data fetching interface
    2. Provide data normalization methods
    3. Implement common technical indicator calculations

    Subclasses must implement:
    - _fetch_raw_data(): Fetch raw data from the specific source
    - _normalize_data(): Convert raw data to standard format
    """

    name: str = "BaseFetcher"
    priority: int = 99  # Lower number = higher priority
    allow_empty_daily_data: bool = False

    @abstractmethod
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch raw data from the data source (subclass must implement).

        Args:
            stock_code: Stock code, e.g., '600519', '000001'
            start_date: Start date, format 'YYYY-MM-DD'
            end_date: End date, format 'YYYY-MM-DD'

        Returns:
            Raw data DataFrame (column names vary by source)
        """
        pass

    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        Normalize column names to standard format (subclass must implement).

        Standard columns: ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        """
        pass

    def get_main_indices(self, region: str = "cn") -> Optional[List[Dict[str, Any]]]:
        """
        Get major index real-time quotes.

        Args:
            region: Market region, cn=A-share, us=US stocks

        Returns:
            List[Dict]: Index list, each element contains:
                - code: index code
                - name: index name
                - current: current value
                - change: change points
                - change_pct: change percentage
                - volume: volume
                - amount: amount
        """
        return None

    def get_market_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get market advance-decline statistics.

        Returns:
            Dict: containing:
                - up_count: up count
                - down_count: down count
                - flat_count: flat count
                - limit_up_count: limit up count
                - limit_down_count: limit down count
                - total_amount: total turnover
        """
        return None

    def get_sector_rankings(self, n: int = 5) -> Optional[Tuple[List[Dict], List[Dict]]]:
        """
        Get sector gainers and losers.

        Args:
            n: Number of top sectors to return

        Returns:
            Tuple: (gainers list, losers list)
        """
        return None

    def get_concept_rankings(self, n: int = 5) -> Optional[Tuple[List[Dict], List[Dict]]]:
        """
        Get concept/theme gainers and losers.

        Returns:
            Tuple: (gainers list, losers list)
        """
        return None

    def get_hot_stocks(self, n: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        Get the most popular stocks.

        Returns:
            List[Dict]: Popular stocks list
        """
        return None

    def get_limit_up_pool(
        self,
        date: Optional[str] = None,
        n: int = 20,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get limit-up pool / streak board.

        Args:
            date: YYYYMMDD, default determined by specific data source
            n: Number of entries to return
        """
        return None

    def get_daily_data(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Get daily data (unified entry point).

        Process:
        1. Calculate date range
        2. Call subclass to fetch raw data
        3. Normalize column names
        4. Calculate technical indicators

        Args:
            stock_code: Stock code
            start_date: Start date (optional)
            end_date: End date (optional, default today)
            days: Number of days to fetch (used when start_date not specified)

        Returns:
            Standardized DataFrame with technical indicators
        """
        # Calculate date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date is None:
            # Fetch approximately days * 2 calendar days to ensure enough trading days
            from datetime import timedelta
            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days * 2)
            start_date = start_dt.strftime('%Y-%m-%d')

        request_start = time.time()
        logger.info(f"[{self.name}] Getting daily data for {stock_code}: range={start_date} ~ {end_date}")

        try:
            # Step 1: Fetch raw data
            raw_df = self._fetch_raw_data(stock_code, start_date, end_date)

            if raw_df is None:
                raise DataFetchError(f"[{self.name}] No data for {stock_code}")
            if raw_df.empty:
                elapsed = time.time() - request_start
                logger.info(
                    f"[{self.name}] {stock_code} returned empty daily data: range={start_date} ~ {end_date}, "
                    f"elapsed={elapsed:.2f}s"
                )
                if self.allow_empty_daily_data:
                    return pd.DataFrame(columns=STANDARD_COLUMNS)
                raise DataFetchError(f"[{self.name}] No data for {stock_code}")

            # Step 2: Normalize column names
            df = self._normalize_data(raw_df, stock_code)

            # Step 3: Data cleaning
            df = self._clean_data(df)

            # Step 4: Calculate technical indicators
            df = self._calculate_indicators(df)

            elapsed = time.time() - request_start
            logger.info(
                f"[{self.name}] {stock_code} successful: range={start_date} ~ {end_date}, "
                f"rows={len(df)}, elapsed={elapsed:.2f}s"
            )
            return df

        except Exception as e:
            elapsed = time.time() - request_start
            error_type, error_reason = summarize_exception(e)
            logger.error(
                f"[{self.name}] {stock_code} failed: range={start_date} ~ {end_date}, "
                f"error_type={error_type}, elapsed={elapsed:.2f}s, reason={error_reason}"
            )
            raise DataFetchError(f"[{self.name}] {stock_code}: {error_reason}") from e

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the data.

        Handles:
        1. Ensure date column is datetime
        2. Convert numeric columns to proper types
        3. Drop rows with null values in key columns
        4. Sort by date
        """
        df = df.copy()

        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with null close or volume
        df = df.dropna(subset=['close', 'volume'])

        # Sort by date ascending
        df = df.sort_values('date', ascending=True).reset_index(drop=True)

        return df

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.

        Indicators:
        - MA5, MA10, MA20: Moving averages
        - Volume_Ratio: Today's volume / 5-day average volume
        """
        df = df.copy()

        # Moving averages
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()

        # Volume ratio: today's volume / previous 5-day average volume
        avg_volume_5 = df['volume'].rolling(window=5, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / avg_volume_5.shift(1)
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)

        # Round to 2 decimals
        for col in ['ma5', 'ma10', 'ma20', 'volume_ratio']:
            if col in df.columns:
                df[col] = df[col].round(2)

        return df

    @staticmethod
    def random_sleep(min_seconds: float = 1.0, max_seconds: float = 3.0) -> None:
        """
        Smart random sleep (Jitter).

        Anti-blocking strategy: simulate human behavior with irregular delays.
        """
        sleep_time = random.uniform(min_seconds, max_seconds)
        logger.debug(f"Random sleep {sleep_time:.2f} seconds...")
        time.sleep(sleep_time)


class DataFetcherManager:
    """
    Data source strategy manager.

    Responsibilities:
    1. Manage multiple data sources (sorted by priority)
    2. Automatic failover
    3. Provide unified data fetching interface

    Failover strategy:
    - Prefer high-priority data sources
    - Automatically fallback to the next on failure
    - Raise exception when all sources fail
    """

    _DAILY_MARKET_FETCHER_SUPPORT = {
        "EfinanceFetcher": {"cn"},
        "TencentFetcher": {"cn"},
        "AkshareFetcher": {"cn", "hk"},
        "TushareFetcher": {"cn", "hk"},
        "PytdxFetcher": {"cn"},
        "BaostockFetcher": {"cn"},
        "YfinanceFetcher": {"cn", "hk", "us", "jp", "kr"},
        "LongbridgeFetcher": {"hk", "us"},
        "FinnhubFetcher": {"us"},
        "AlphaVantageFetcher": {"us"},
    }
    _daily_source_health = CircuitBreaker(failure_threshold=3, cooldown_seconds=300.0)

    def __init__(self, fetchers: Optional[List[BaseFetcher]] = None):
        """
        Initialize the manager.

        Args:
            fetchers: List of data sources (optional, auto-created by priority by default)
        """
        self._fetchers: List[BaseFetcher] = []
        self._fetchers_lock = RLock()
        self._fetchers_by_name: Dict[str, BaseFetcher] = {}
        self._fetcher_call_locks: Dict[int, RLock] = {}
        self._fetcher_call_locks_lock = RLock()
        self._stock_name_cache: Dict[str, str] = {}
        self._stock_name_cache_lock = RLock()

        if fetchers:
            # Sort by priority
            self._fetchers = sorted(fetchers, key=lambda f: f.priority)
            self._refresh_fetcher_indexes_locked()
        else:
            # Default fetchers will be lazy-loaded on first use
            self._init_default_fetchers()
        self._fundamental_adapter = AkshareFundamentalAdapter()
        self._yfinance_fundamental_adapter = YfinanceFundamentalAdapter()
        self._tickflow_fetcher = None
        self._tickflow_api_key: Optional[str] = None
        self._tickflow_lock = RLock()
        self._fundamental_cache: Dict[str, Dict[str, Any]] = {}
        self._fundamental_cache_lock = RLock()
        self._fundamental_timeout_worker_limit = 8
        self._fundamental_timeout_slots = BoundedSemaphore(self._fundamental_timeout_worker_limit)

    def _ensure_concurrency_guards(self) -> None:
        """Lazily initialize thread-safety primitives for test scaffolds using __new__."""
        if not hasattr(self, "_fetchers_lock") or self._fetchers_lock is None:
            self._fetchers_lock = RLock()
        if not hasattr(self, "_fetchers_by_name") or self._fetchers_by_name is None:
            self._fetchers_by_name = {}
        if not hasattr(self, "_fetcher_call_locks") or self._fetcher_call_locks is None:
            self._fetcher_call_locks = {}
        if not hasattr(self, "_fetcher_call_locks_lock") or self._fetcher_call_locks_lock is None:
            self._fetcher_call_locks_lock = RLock()
        if not hasattr(self, "_stock_name_cache") or self._stock_name_cache is None:
            self._stock_name_cache = {}
        if not hasattr(self, "_stock_name_cache_lock") or self._stock_name_cache_lock is None:
            self._stock_name_cache_lock = RLock()

    def _get_fetchers_snapshot(self) -> List[BaseFetcher]:
        self._ensure_concurrency_guards()
        with self._fetchers_lock:
            return list(getattr(self, "_fetchers", []))

    def _refresh_fetcher_indexes_locked(self) -> None:
        self._fetchers_by_name = {fetcher.name: fetcher for fetcher in self._fetchers}

    def _get_fetcher_by_name(self, fetcher_name: str, capability: str = "") -> Optional[BaseFetcher]:
        self._ensure_concurrency_guards()
        with self._fetchers_lock:
            fetcher = self._fetchers_by_name.get(fetcher_name)
            if fetcher is None and self._fetchers:
                self._refresh_fetcher_indexes_locked()
                fetcher = self._fetchers_by_name.get(fetcher_name)
        if fetcher is None:
            return None
        if not self._is_fetcher_available(fetcher, capability=capability):
            return None
        return fetcher

    @staticmethod
    def _call_availability_probe(fetcher: BaseFetcher, probe_name: str, capability: str) -> Optional[bool]:
        probe = getattr(fetcher, probe_name, None)
        if not callable(probe):
            return None
        try:
            if probe_name == "is_available_for_request":
                return bool(probe(capability))
            return bool(probe())
        except TypeError:
            return bool(probe())
        except Exception as exc:
            logger.debug(
                "[Data source availability] %s.%s check failed (capability=%s): %s",
                fetcher.name,
                probe_name,
                capability or "default",
                exc,
            )
            return False

    @classmethod
    def _is_fetcher_available(cls, fetcher: BaseFetcher, capability: str = "") -> bool:
        for probe_name in ("is_available_for_request", "is_available", "_is_available"):
            result = cls._call_availability_probe(fetcher, probe_name, capability)
            if result is not None:
                return result
        return True

    def _get_fetcher_call_lock(self, fetcher: BaseFetcher) -> RLock:
        self._ensure_concurrency_guards()
        fetcher_id = id(fetcher)
        with self._fetcher_call_locks_lock:
            lock = self._fetcher_call_locks.get(fetcher_id)
            if lock is None:
                lock = RLock()
                self._fetcher_call_locks[fetcher_id] = lock
            return lock

    def _call_fetcher_method(self, fetcher: BaseFetcher, method_name: str, *args, **kwargs):
        """Serialize shared fetcher state access through manager-owned per-instance locks."""
        method = getattr(fetcher, method_name)
        with self._get_fetcher_call_lock(fetcher):
            return method(*args, **kwargs)

    @classmethod
    def _filter_daily_fetchers_for_market(
        cls,
        fetchers: List[BaseFetcher],
        market: str,
    ) -> List[BaseFetcher]:
        """Skip built-in daily fetchers that are known not to support a market."""

        kept: List[BaseFetcher] = []
        skipped: List[str] = []
        for fetcher in fetchers:
            supported = cls._DAILY_MARKET_FETCHER_SUPPORT.get(fetcher.name)
            if supported is not None and market not in supported:
                skipped.append(fetcher.name)
            else:
                kept.append(fetcher)

        if skipped:
            logger.info(
                "[Data source routing] %s daily skip unsupported sources: %s",
                market,
                ", ".join(skipped),
            )
        return kept

    @classmethod
    def _filter_fetchers_by_capability(
        cls,
        fetchers: List[BaseFetcher],
        capability: str,
    ) -> List[BaseFetcher]:
        """Skip request-time unavailable fetchers before entering route-specific loops."""
        kept: List[BaseFetcher] = []
        skipped: List[str] = []

        for fetcher in fetchers:
            if cls._is_fetcher_available(fetcher, capability=capability):
                kept.append(fetcher)
            else:
                skipped.append(fetcher.name)

        if skipped:
            logger.info(
                "[Data source routing] %s skip temporarily unavailable sources: %s",
                capability or "request",
                ", ".join(skipped),
            )

        return kept

    @classmethod
    def _daily_health_key(cls, fetcher: BaseFetcher, market: str) -> str:
        return f"daily_data:{market}:{fetcher.name}"

    @classmethod
    def _is_daily_source_available(
        cls,
        fetcher: BaseFetcher,
        market: str,
    ) -> bool:
        key = cls._daily_health_key(fetcher, market)
        if cls._daily_source_health.is_available(key):
            return True
        logger.info(
            "[Data source health] %s daily skip temporarily melted source: %s",
            market,
            fetcher.name,
        )
        return False

    @staticmethod
    def _daily_source_unavailable_error(fetcher: BaseFetcher) -> str:
        return f"[{fetcher.name}] (CircuitOpen) Data source temporarily melted"

    @classmethod
    def _record_daily_source_success(cls, fetcher: BaseFetcher, market: str) -> None:
        cls._daily_source_health.record_success(cls._daily_health_key(fetcher, market))

    @classmethod
    def _record_daily_source_failure(cls, fetcher: BaseFetcher, market: str, error: str) -> None:
        cls._daily_source_health.record_failure(cls._daily_health_key(fetcher, market), error=error)

    @classmethod
    def reset_daily_source_health(cls) -> None:
        """Reset daily source health state for tests/admin diagnostics."""
        cls._daily_source_health.reset()

    def _get_cached_stock_name(self, stock_code: str) -> Optional[str]:
        self._ensure_concurrency_guards()
        with self._stock_name_cache_lock:
            return self._stock_name_cache.get(stock_code)

    def _cache_stock_name(self, stock_code: str, name: Optional[str]) -> Optional[str]:
        if name is None:
            return None
        self._ensure_concurrency_guards()
        with self._stock_name_cache_lock:
            self._stock_name_cache[stock_code] = name
        return name

    def _get_tickflow_fetcher(self):
        """Lazily create a TickFlow fetcher for market-review-only calls."""
        from src.config import get_config

        config = get_config()
        api_key = (getattr(config, "tickflow_api_key", None) or "").strip()

        if not hasattr(self, "_tickflow_lock") or self._tickflow_lock is None:
            self._tickflow_lock = RLock()

        with self._tickflow_lock:
            current_fetcher = getattr(self, "_tickflow_fetcher", None)
            current_key = getattr(self, "_tickflow_api_key", None)

            if not api_key:
                if current_fetcher is not None and hasattr(current_fetcher, "close"):
                    try:
                        current_fetcher.close()
                    except Exception as exc:
                        logger.debug("[TickFlowFetcher] Failed to close old instance: %s", exc)
                self._tickflow_fetcher = None
                self._tickflow_api_key = None
                return None

            if current_fetcher is not None and current_key == api_key:
                return current_fetcher

            if current_fetcher is not None and hasattr(current_fetcher, "close"):
                try:
                    current_fetcher.close()
                except Exception as exc:
                    logger.debug("[TickFlowFetcher] Failed to close during instance switch: %s", exc)

            try:
                from .tickflow_fetcher import TickFlowFetcher

                fetcher = TickFlowFetcher(api_key=api_key)
                self._tickflow_fetcher = fetcher
                self._tickflow_api_key = api_key
                return fetcher
            except Exception as exc:
                logger.warning("[TickFlowFetcher] Initialization failed: %s", exc)
                self._tickflow_fetcher = None
                self._tickflow_api_key = None
                return None

    def close(self) -> None:
        """Best-effort release of manager-owned resources."""
        if not hasattr(self, "_tickflow_lock") or self._tickflow_lock is None:
            self._tickflow_lock = RLock()

        with self._tickflow_lock:
            current_fetcher = getattr(self, "_tickflow_fetcher", None)
            self._tickflow_fetcher = None
            self._tickflow_api_key = None

        if current_fetcher is not None and hasattr(current_fetcher, "close"):
            try:
                current_fetcher.close()
            except Exception as exc:
                logger.debug("[TickFlowFetcher] Failed to close manager resource: %s", exc)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Best-effort cleanup during interpreter shutdown.
            pass

    def _get_fundamental_cache_key(self, stock_code: str, budget_seconds: Optional[float] = None) -> str:
        """Generate fundamental cache key (including budget bucket to avoid cross-pollution)."""
        normalized_code = normalize_stock_code(stock_code)
        if budget_seconds is None:
            return f"{normalized_code}|budget=default"
        try:
            budget = max(0.0, float(budget_seconds))
        except (TypeError, ValueError):
            budget = 0.0
        # 100ms bucket to balance cache reuse and scenario isolation.
        budget_bucket = int(round(budget * 10))
        return f"{normalized_code}|budget={budget_bucket}"

    def _prune_fundamental_cache(self, ttl_seconds: int, max_entries: int) -> None:
        """Prune expired and overflow fundamental cache items."""
        with self._fundamental_cache_lock:
            if not self._fundamental_cache:
                return

            now_ts = time.time()
            if ttl_seconds > 0:
                cache_items = list(self._fundamental_cache.items())
                expired_keys = [
                    key
                    for key, value in cache_items
                    if now_ts - float(value.get("ts", 0)) > ttl_seconds
                ]
                for key in expired_keys:
                    self._fundamental_cache.pop(key, None)

            if max_entries > 0 and len(self._fundamental_cache) > max_entries:
                overflow = len(self._fundamental_cache) - max_entries
                sorted_items = sorted(
                    list(self._fundamental_cache.items()),
                    key=lambda item: float(item[1].get("ts", 0)),
                )
                for key, _ in sorted_items[:overflow]:
                    self._fundamental_cache.pop(key, None)

    @staticmethod
    def _try_scalar_isna(value: Any, context: str) -> Optional[bool]:
        """Return scalar pd.isna result, or None when fallback is needed."""
        if isinstance(value, (dict, list, tuple, set, pd.DataFrame, pd.Series, pd.Index)):
            return None

        if isinstance(value, np.ndarray):
            if value.ndim != 0:
                return None
            value = value.item()

        try:
            isna_result = pd.isna(value)
        except (TypeError, ValueError) as exc:
            if hasattr(value, "__array__"):
                logger.debug(
                    "[%s] pd.isna failed for array-like object; re-raise: value_type=%s error_type=%s",
                    context,
                    type(value).__name__,
                    type(exc).__name__,
                )
                raise
            logger.debug(
                "[%s] pd.isna fallback: value_type=%s error_type=%s",
                context,
                type(value).__name__,
                type(exc).__name__,
            )
            return None

        if isinstance(isna_result, (bool, np.bool_)):
            return bool(isna_result)

        if isinstance(isna_result, np.ndarray):
            if isna_result.ndim == 0:
                return bool(isna_result.item())
            logger.debug(
                "[%s] pd.isna returned non-scalar result: value_type=%s result_type=%s",
                context,
                type(value).__name__,
                type(isna_result).__name__,
            )
            return None

        logger.debug(
            "[%s] pd.isna returned unexpected result type: value_type=%s result_type=%s",
            context,
            type(value).__name__,
            type(isna_result).__name__,
        )
        return None

    @staticmethod
    def _is_missing_board_value(value: Any) -> bool:
        """Return True when a board field value should be treated as missing."""
        if value is None:
            return True
        is_missing = DataFetcherManager._try_scalar_isna(value, "board_value")
        if is_missing is True:
            return True
        text = str(value).strip()
        return text == "" or text.lower() in {"nan", "none", "null", "na", "n/a"}

    @staticmethod
    def _normalize_belong_boards(raw_data: Any) -> List[Dict[str, Any]]:
        """Normalize belong-board results from heterogeneous providers."""
        if DataFetcherManager._is_missing_board_value(raw_data):
            return []

        normalized: List[Dict[str, Any]] = []
        dedupe = set()

        if isinstance(raw_data, pd.DataFrame):
            if raw_data.empty:
                return []
            name_col = next(
                (
                    col
                    for col in raw_data.columns
                    if str(col) in {"板块名称", "板块", "所属板块", "板块名", "name", "industry"}
                ),
                None,
            )
            code_col = next(
                (
                    col
                    for col in raw_data.columns
                    if str(col) in {"板块代码", "代码", "code"}
                ),
                None,
            )
            type_col = next(
                (
                    col
                    for col in raw_data.columns
                    if str(col) in {"板块类型", "类别", "type"}
                ),
                None,
            )
            if name_col is None:
                return []
            for _, row in raw_data.iterrows():
                board_name_raw = row.get(name_col, "")
                if DataFetcherManager._is_missing_board_value(board_name_raw):
                    continue
                board_name = str(board_name_raw).strip()
                if board_name in dedupe:
                    continue
                dedupe.add(board_name)
                item = {"name": board_name}
                if code_col is not None:
                    board_code_raw = row.get(code_col, "")
                    if not DataFetcherManager._is_missing_board_value(board_code_raw):
                        item["code"] = str(board_code_raw).strip()
                if type_col is not None:
                    board_type_raw = row.get(type_col, "")
                    if not DataFetcherManager._is_missing_board_value(board_type_raw):
                        item["type"] = str(board_type_raw).strip()
                normalized.append(item)
            return normalized

        if isinstance(raw_data, dict):
            raw_data = [raw_data]

        if isinstance(raw_data, (list, tuple, set)):
            for item in raw_data:
                if isinstance(item, dict):
                    board_name_raw = (
                        item.get("name")
                        or item.get("board_name")
                        or item.get("板块名称")
                        or item.get("板块")
                        or item.get("所属板块")
                        or item.get("板块名")
                        or item.get("industry")
                        or item.get("行业")
                    )
                    if DataFetcherManager._is_missing_board_value(board_name_raw):
                        continue
                    board_name = str(board_name_raw).strip()
                    if board_name in dedupe:
                        continue
                    dedupe.add(board_name)
                    normalized_item: Dict[str, Any] = {"name": board_name}
                    code_raw = (
                        item.get("code")
                        or item.get("板块代码")
                        or item.get("代码")
                    )
                    if not DataFetcherManager._is_missing_board_value(code_raw):
                        normalized_item["code"] = str(code_raw).strip()
                    type_raw = (
                        item.get("type")
                        or item.get("板块类型")
                        or item.get("类别")
                    )
                    if not DataFetcherManager._is_missing_board_value(type_raw):
                        normalized_item["type"] = str(type_raw).strip()
                    normalized.append(normalized_item)
                    continue
                if DataFetcherManager._is_missing_board_value(item):
                    continue
                board_name = str(item).strip()
                if board_name in dedupe:
                    continue
                dedupe.add(board_name)
                normalized.append({"name": board_name})
            return normalized

        if not DataFetcherManager._is_missing_board_value(raw_data):
            board_name = str(raw_data).strip()
            return [{"name": board_name}]
        return []

    def _init_default_fetchers(self):
        """
        Initialize default data sources without priority arguments.
        This method is called when no fetchers are provided in __init__.
        """
        with self._fetchers_lock:
            self._fetchers = [
                EfinanceFetcher(),
                TencentFetcher(),
                AkshareFetcher(),
                PytdxFetcher(),
                BaostockFetcher(),
                YfinanceFetcher(),
            ]

            # Add IranFetcher for Tehran Stock Exchange (if available)
            try:
                from .iran_fetcher import IranFetcher
                self._fetchers.append(IranFetcher())
            except ImportError:
                pass

            # Optional fetchers based on configuration
            from src.config import get_config
            from .tushare_fetcher import TushareFetcher
            from .longbridge_fetcher import LongbridgeFetcher
            from .finnhub_fetcher import FinnhubFetcher
            from .alphavantage_fetcher import AlphaVantageFetcher

            config = get_config()

            tushare_token = (getattr(config, "tushare_token", None) or "").strip()
            if tushare_token:
                self._fetchers.append(TushareFetcher())

            if LongbridgeFetcher.has_configured_credentials(config):
                self._fetchers.append(LongbridgeFetcher())

            finnhub_api_key = (getattr(config, "finnhub_api_key", None) or "").strip()
            if finnhub_api_key:
                self._fetchers.append(FinnhubFetcher())

            alphavantage_api_key = (getattr(config, "alphavantage_api_key", None) or "").strip()
            if alphavantage_api_key:
                self._fetchers.append(AlphaVantageFetcher())

            # Sort by priority (lower number = higher priority)
            self._fetchers.sort(key=lambda f: f.priority)
            self._refresh_fetcher_indexes_locked()

            # Log initialization
            priority_info = ", ".join([f"{f.name}(P{f.priority})" for f in self._fetchers])
            logger.info(f"Initialized {len(self._fetchers)} data sources (by priority): {priority_info}")

    def add_fetcher(self, fetcher: BaseFetcher) -> None:
        """Add a data source and re-sort by priority."""
        self._ensure_concurrency_guards()
        with self._fetchers_lock:
            self._fetchers.append(fetcher)
            self._fetchers.sort(key=lambda f: f.priority)
            self._refresh_fetcher_indexes_locked()

    def get_daily_data(
        self,
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> Tuple[pd.DataFrame, str]:
        """
        Get daily data (automatic failover).

        Failover strategy:
        1. US indices/US stocks are routed directly to YfinanceFetcher
        2. Other codes try from highest priority source
        3. Catch exceptions and fallback to next source
        4. Record failure reasons for each source
        5. Raise detailed exception if all sources fail

        Args:
            stock_code: Stock code
            start_date: Start date
            end_date: End date
            days: Number of days to fetch

        Returns:
            Tuple[DataFrame, str]: (data, successful data source name)

        Raises:
            DataFetchError: When all sources fail
        """
        from .us_index_mapping import is_us_index_code, is_us_stock_code

        # Normalize code (strip SH/SZ prefix etc.)
        stock_code = normalize_stock_code(stock_code)

        fetchers = self._get_fetchers_snapshot()
        errors = []
        request_start = time.time()

        # Fast path: US stocks use dedicated routing; HK filters unsupported sources
        is_us_index = is_us_index_code(stock_code)
        is_us = is_us_index or is_us_stock_code(stock_code)
        is_hk = (not is_us) and _is_hk_market(stock_code)
        is_jp = (not is_us) and (not is_hk) and _is_jp_market(stock_code)
        is_kr = (not is_us) and (not is_hk) and _is_kr_market(stock_code)
        market = "us" if is_us else "hk" if is_hk else "jp" if is_jp else "kr" if is_kr else "cn"
        if market != "cn":
            fetchers = self._filter_daily_fetchers_for_market(fetchers, market)
        fetchers = self._filter_fetchers_by_capability(fetchers, capability="daily_data")
        total_fetchers = len(fetchers)

        if total_fetchers == 0:
            market_label = "US index" if is_us_index else "US" if is_us else "HK" if is_hk else "A-share"
            error_summary = f"{market_label} {stock_code} failed:\nNo available data sources"
            logger.error(f"[Data source termination] {stock_code} failed: {error_summary}")
            raise DataFetchError(error_summary)

        # US (including indices) uses dedicated routing; HK uses general loop below
        if is_us:
            prefer_lb = self._longbridge_preferred(capability="daily_data") and not is_us_index
            if is_us_index:
                # Indices always prefer Yfinance (Longbridge doesn't provide index K-line)
                source_order = ["YfinanceFetcher", "FinnhubFetcher"]
            elif prefer_lb:
                source_order = ["LongbridgeFetcher", "FinnhubFetcher", "AlphaVantageFetcher", "YfinanceFetcher"]
            else:
                source_order = ["FinnhubFetcher", "AlphaVantageFetcher", "YfinanceFetcher", "LongbridgeFetcher"]
            market_label = "US index" if is_us_index else "US"

            for order_index, src_name in enumerate(source_order):
                fallback_to = (
                    source_order[order_index + 1]
                    if order_index + 1 < len(source_order)
                    else None
                )
                for attempt, fetcher in enumerate(fetchers, start=1):
                    if fetcher.name != src_name:
                        continue
                    if not self._is_daily_source_available(fetcher, market):
                        errors.append(self._daily_source_unavailable_error(fetcher))
                        break
                    attempt_start = time.time()
                    try:
                        role = "primary" if src_name == source_order[0] else "fallback"
                        logger.info(
                            f"[Data source attempt {attempt}/{total_fetchers}] [{fetcher.name}] "
                            f"{market_label} {stock_code} {role} routing..."
                        )
                        record_provider_run_started(
                            data_type="daily_data",
                            provider=fetcher.name,
                            operation="get_daily_data",
                        )
                        df = self._call_fetcher_method(
                            fetcher,
                            "get_daily_data",
                            stock_code=stock_code,
                            start_date=start_date,
                            end_date=end_date,
                            days=days,
                        )
                        if df is not None and not df.empty:
                            duration_ms = int((time.time() - attempt_start) * 1000)
                            record_provider_run(
                                data_type="daily_data",
                                provider=fetcher.name,
                                operation="get_daily_data",
                                success=True,
                                latency_ms=duration_ms,
                                record_count=len(df),
                            )
                            elapsed = time.time() - request_start
                            logger.info(
                                f"[Data source complete] {stock_code} using [{fetcher.name}]: "
                                f"rows={len(df)}, elapsed={elapsed:.2f}s"
                            )
                            self._record_daily_source_success(fetcher, market)
                            return df, fetcher.name
                        duration_ms = int((time.time() - attempt_start) * 1000)
                        record_provider_run(
                            data_type="daily_data",
                            provider=fetcher.name,
                            operation="get_daily_data",
                            success=False,
                            latency_ms=duration_ms,
                            error_type="empty",
                            error_message="empty result",
                            fallback_to=fallback_to,
                            record_count=0,
                        )
                        if df is not None and df.empty:
                            self._record_daily_source_success(fetcher, market)
                    except Exception as e:
                        error_type, error_reason = summarize_exception(e)
                        error_msg = f"[{fetcher.name}] ({error_type}) {error_reason}"
                        duration_ms = int((time.time() - attempt_start) * 1000)
                        record_provider_run(
                            data_type="daily_data",
                            provider=fetcher.name,
                            operation="get_daily_data",
                            success=False,
                            latency_ms=duration_ms,
                            error_type=error_type,
                            error_message=error_reason,
                            fallback_to=fallback_to,
                        )
                        logger.warning(
                            f"[Data source failed {attempt}/{total_fetchers}] [{fetcher.name}] {stock_code}: "
                            f"error_type={error_type}, reason={error_reason}"
                        )
                        self._record_daily_source_failure(fetcher, market, error_reason)
                        errors.append(error_msg)
                    break

            error_summary = f"{market_label} {stock_code} failed:\n" + "\n".join(errors)
            elapsed = time.time() - request_start
            logger.error(f"[Data source termination] {stock_code} failed: elapsed={elapsed:.2f}s\n{error_summary}")
            raise DataFetchError(error_summary)

        # General loop for other markets (HK, CN, etc.)
        for attempt, fetcher in enumerate(fetchers, start=1):
            if not self._is_daily_source_available(fetcher, market):
                errors.append(self._daily_source_unavailable_error(fetcher))
                continue
            attempt_start = time.time()
            fallback_to = fetchers[attempt].name if attempt < total_fetchers else None
            try:
                logger.info(f"[Data source attempt {attempt}/{total_fetchers}] [{fetcher.name}] fetching {stock_code}...")
                record_provider_run_started(
                    data_type="daily_data",
                    provider=fetcher.name,
                    operation="get_daily_data",
                )
                df = self._call_fetcher_method(
                    fetcher,
                    "get_daily_data",
                    stock_code=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    days=days
                )

                if df is not None and not df.empty:
                    duration_ms = int((time.time() - attempt_start) * 1000)
                    record_provider_run(
                        data_type="daily_data",
                        provider=fetcher.name,
                        operation="get_daily_data",
                        success=True,
                        latency_ms=duration_ms,
                        record_count=len(df),
                    )
                    elapsed = time.time() - request_start
                    logger.info(
                        f"[Data source complete] {stock_code} using [{fetcher.name}]: "
                        f"rows={len(df)}, elapsed={elapsed:.2f}s"
                    )
                    self._record_daily_source_success(fetcher, market)
                    return df, fetcher.name
                duration_ms = int((time.time() - attempt_start) * 1000)
                record_provider_run(
                    data_type="daily_data",
                    provider=fetcher.name,
                    operation="get_daily_data",
                    success=False,
                    latency_ms=duration_ms,
                    error_type="empty",
                    error_message="empty result",
                    fallback_to=fallback_to,
                    record_count=0,
                )
                if df is not None and df.empty:
                    self._record_daily_source_success(fetcher, market)

            except Exception as e:
                error_type, error_reason = summarize_exception(e)
                error_msg = f"[{fetcher.name}] ({error_type}) {error_reason}"
                duration_ms = int((time.time() - attempt_start) * 1000)
                record_provider_run(
                    data_type="daily_data",
                    provider=fetcher.name,
                    operation="get_daily_data",
                    success=False,
                    latency_ms=duration_ms,
                    error_type=error_type,
                    error_message=error_reason,
                    fallback_to=fallback_to,
                )
                logger.warning(
                    f"[Data source failed {attempt}/{total_fetchers}] [{fetcher.name}] {stock_code}: "
                    f"error_type={error_type}, reason={error_reason}"
                )
                self._record_daily_source_failure(fetcher, market, error_reason)
                errors.append(error_msg)
                if attempt < total_fetchers:
                    next_fetcher = fetchers[attempt]
                    logger.info(f"[Data source switch] {stock_code}: [{fetcher.name}] -> [{next_fetcher.name}]")
                continue

        # All sources failed
        error_summary = f"All data sources failed for {stock_code}:\n" + "\n".join(errors)
        elapsed = time.time() - request_start
        logger.error(f"[Data source termination] {stock_code} failed: elapsed={elapsed:.2f}s\n{error_summary}")
        raise DataFetchError(error_summary)

    @property
    def available_fetchers(self) -> List[str]:
        """Return list of available data source names."""
        return [f.name for f in self._get_fetchers_snapshot()]

    def prefetch_realtime_quotes(self, stock_codes: List[str]) -> int:
        """
        Bulk prefetch real-time quotes before analysis starts.

        Strategy:
        1. Check if priority includes a bulk fetch source (efinance/akshare_em)
        2. If not, skip prefetch (Sina/Tencent are per-stock queries)
        3. If stock count >= 5 and using bulk source, prefetch to fill cache

        Args:
            stock_codes: List of stock codes to analyze

        Returns:
            Number of stocks prefetched (0 means skipped)
        """
        # Normalize all codes
        stock_codes = [normalize_stock_code(c) for c in stock_codes]

        from src.config import get_config

        config = get_config()

        if not getattr(config, "prefetch_realtime_quotes", True):
            logger.debug("[Prefetch] component=realtime_prefetch action=skip reason=disabled")
            return 0

        if not config.enable_realtime_quote:
            logger.debug("[Prefetch] component=realtime_prefetch action=skip reason=realtime_quote_disabled")
            return 0

        priority = config.realtime_source_priority.lower()
        bulk_sources = ['efinance', 'akshare_em', 'tushare']

        priority_list = [s.strip() for s in priority.split(',')]
        first_bulk_source_index = None
        for i, source in enumerate(priority_list):
            if source in bulk_sources:
                first_bulk_source_index = i
                break

        if first_bulk_source_index is None or first_bulk_source_index >= 2:
            logger.info(
                "[Prefetch] component=realtime_prefetch action=skip reason=no_early_bulk_source priority=%s",
                priority,
            )
            return 0

        if len(stock_codes) < 5:
            logger.info(
                "[Prefetch] component=realtime_prefetch action=skip reason=small_batch "
                "stock_count=%d threshold=5 bulk_source=%s",
                len(stock_codes),
                priority_list[first_bulk_source_index],
            )
            return 0

        bulk_source = priority_list[first_bulk_source_index]
        logger.info(
            "[Prefetch] component=realtime_prefetch action=start stock_count=%d bulk_source=%s first_code=%s",
            len(stock_codes),
            bulk_source,
            stock_codes[0],
        )

        try:
            first_code = stock_codes[0]
            quote = self.get_realtime_quote(first_code)

            if quote:
                logger.info(
                    "[Prefetch] component=realtime_prefetch action=complete status=success "
                    "stock_count=%d bulk_source=%s",
                    len(stock_codes),
                    bulk_source,
                )
                return len(stock_codes)
            else:
                logger.warning(
                    "[Prefetch] component=realtime_prefetch action=complete status=failed "
                    "stock_count=%d bulk_source=%s fallback=per_stock",
                    len(stock_codes),
                    bulk_source,
                )
                return 0

        except Exception as e:
            logger.error(
                "[Prefetch] component=realtime_prefetch action=complete status=error "
                "stock_count=%d bulk_source=%s error=%s",
                len(stock_codes),
                bulk_source,
                e,
            )
            return 0

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _parse_realtime_timestamp(value: Any) -> Optional[datetime]:
        if value in (None, ""):
            return None
        if isinstance(value, datetime):
            parsed = value
        else:
            text = str(value).strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _realtime_fetcher_token(fetcher_name: str, **kw) -> str:
        if fetcher_name == "AkshareFetcher" and kw.get("source") == "hk":
            return "akshare_hk"
        mapping = {
            "LongbridgeFetcher": "longbridge",
            "YfinanceFetcher": "yfinance",
            "AkshareFetcher": "akshare",
            "FinnhubFetcher": "finnhub",
            "AlphaVantageFetcher": "alphavantage",
            "EfinanceFetcher": "efinance",
            "TushareFetcher": "tushare",
        }
        return mapping.get(fetcher_name, fetcher_name.replace("Fetcher", "").lower())

    def _enrich_realtime_quote(
        self,
        quote,
        *,
        fallback_from: Optional[str] = None,
        realtime_cache_ttl: Optional[int] = None,
    ):
        """Attach runtime metadata without inventing provider-side timestamps."""
        if quote is None:
            return None

        fetched_at = self._utc_now_iso()
        setattr(quote, "fetched_at", fetched_at)
        if fallback_from:
            setattr(quote, "fallback_from", str(fallback_from))

        provider_dt = self._parse_realtime_timestamp(
            getattr(quote, "provider_timestamp", None)
        )
        if provider_dt is None:
            setattr(quote, "provider_timestamp", None)
            setattr(quote, "stale_seconds", None)
            setattr(quote, "is_stale", None)
            return quote

        setattr(quote, "provider_timestamp", provider_dt.isoformat())
        fetched_dt = self._parse_realtime_timestamp(fetched_at) or datetime.now(timezone.utc)
        stale_seconds = max(0, int((fetched_dt - provider_dt).total_seconds()))
        ttl = realtime_cache_ttl if realtime_cache_ttl is not None else 600
        setattr(quote, "stale_seconds", stale_seconds)
        setattr(quote, "is_stale", stale_seconds > int(ttl))
        return quote

    def get_realtime_quote(self, stock_code: str, *, log_final_failure: bool = True):
        """
        Get real-time quote data (automatic failover).

        Failover strategy (according to configured priority):
        1. US stocks: use YfinanceFetcher.get_realtime_quote()
        2. EfinanceFetcher
        3. AkshareFetcher(source="em")  - Eastmoney
        4. AkshareFetcher(source="sina") - Sina
        5. AkshareFetcher(source="tencent") - Tencent
        6. Return None (graceful degradation)

        Args:
            stock_code: Stock code
            log_final_failure: Whether to log the final "all sources failed" summary

        Returns:
            UnifiedRealtimeQuote object, or None if all sources fail.
        """
        raw_stock_code = (stock_code or "").strip()
        stock_code = normalize_stock_code(stock_code)

        from .akshare_fetcher import _is_us_code
        from .us_index_mapping import is_us_index_code
        from src.config import get_config

        config = get_config()

        if not config.enable_realtime_quote:
            logger.debug(f"[Realtime quote] disabled, skipping {stock_code}")
            return None

        is_us_index = is_us_index_code(stock_code)
        is_us = is_us_index or _is_us_code(stock_code)
        is_hk = (not is_us) and _is_hk_market(stock_code)
        is_jp = (not is_us) and (not is_hk) and _is_jp_market(stock_code)
        is_kr = (not is_us) and (not is_hk) and _is_kr_market(stock_code)

        if is_jp or is_kr:
            market_label = "Japan" if is_jp else "Korea"
            quote = self._try_fetcher_quote(stock_code, "YfinanceFetcher")
            if quote is not None:
                logger.info(f"[Realtime quote] {market_label} {stock_code} successful (source: YfinanceFetcher)")
                return self._enrich_realtime_quote(
                    quote,
                    realtime_cache_ttl=getattr(config, "realtime_cache_ttl", None),
                )
            if log_final_failure:
                logger.info(f"[Realtime quote] {market_label} {stock_code} no available source")
            return None

        if is_us or is_hk:
            prefer_lb = self._longbridge_preferred() and not is_us_index
            if is_us:
                primary_src = "LongbridgeFetcher" if prefer_lb else "YfinanceFetcher"
                secondary_src = "YfinanceFetcher" if prefer_lb else "LongbridgeFetcher"
                market_label = "US index" if is_us_index else "US"
                primary_kw: dict = {}
                secondary_kw: dict = {}
            else:
                primary_src = "LongbridgeFetcher" if prefer_lb else "AkshareFetcher"
                secondary_src = "AkshareFetcher" if prefer_lb else "LongbridgeFetcher"
                market_label = "HK"
                primary_kw = {"source": "hk"} if primary_src == "AkshareFetcher" else {}
                secondary_kw = {"source": "hk"} if secondary_src == "AkshareFetcher" else {}

            primary_token = self._realtime_fetcher_token(primary_src, **primary_kw)
            primary_quote = self._try_fetcher_quote(stock_code, primary_src, **primary_kw)
            fallback_from = primary_token if primary_quote is None else None
            if primary_quote is not None:
                logger.info(f"[Realtime quote] {market_label} {stock_code} successful (source: {primary_src})")
            primary_quote = self._supplement_quote(
                stock_code, primary_quote, secondary_src, **secondary_kw,
            )
            # US stocks (non-index) try Finnhub/AlphaVantage to fill missing fields
            if is_us and not is_us_index and primary_quote is not None:
                for extra_src in ["FinnhubFetcher", "AlphaVantageFetcher"]:
                    primary_quote = self._supplement_quote(
                        stock_code, primary_quote, extra_src,
                    )
            if primary_quote is not None:
                return self._enrich_realtime_quote(
                    primary_quote,
                    fallback_from=fallback_from,
                    realtime_cache_ttl=getattr(config, "realtime_cache_ttl", None),
                )
            if log_final_failure:
                logger.info(f"[Realtime quote] {market_label} {stock_code} no available source")
            return None

        # A-share general loop
        source_priority = [
            source.strip().lower()
            for source in config.realtime_source_priority.split(',')
            if source.strip()
        ]

        errors = []
        failed_sources: List[str] = []
        primary_quote = None
        primary_fallback_from: Optional[str] = None
        supplement_attempts = 0

        for source_index, source in enumerate(source_priority):
            attempt_start = time.time()
            fallback_to = source_priority[source_index + 1] if source_index + 1 < len(source_priority) else None
            fetcher = None
            try:
                quote = None

                if source == "efinance":
                    fetcher = self._get_fetcher_by_name("EfinanceFetcher", capability="realtime_quote")
                    if fetcher is not None and hasattr(fetcher, 'get_realtime_quote'):
                        record_provider_run_started(
                            data_type="realtime_quote",
                            provider=fetcher.name,
                            operation="get_realtime_quote",
                        )
                        quote = self._call_fetcher_method(fetcher, 'get_realtime_quote', stock_code)

                elif source == "akshare_em":
                    fetcher = self._get_fetcher_by_name("AkshareFetcher", capability="realtime_quote")
                    if fetcher is not None and hasattr(fetcher, 'get_realtime_quote'):
                        record_provider_run_started(
                            data_type="realtime_quote",
                            provider=fetcher.name,
                            operation="get_realtime_quote",
                        )
                        quote = self._call_fetcher_method(fetcher, 'get_realtime_quote', stock_code, source="em")

                elif source == "akshare_sina":
                    fetcher = self._get_fetcher_by_name("AkshareFetcher", capability="realtime_quote")
                    if fetcher is not None and hasattr(fetcher, 'get_realtime_quote'):
                        record_provider_run_started(
                            data_type="realtime_quote",
                            provider=fetcher.name,
                            operation="get_realtime_quote",
                        )
                        quote = self._call_fetcher_method(fetcher, 'get_realtime_quote', stock_code, source="sina")

                elif source in ("tencent", "akshare_qq"):
                    fetcher = self._get_fetcher_by_name("AkshareFetcher", capability="realtime_quote")
                    if fetcher is not None and hasattr(fetcher, 'get_realtime_quote'):
                        record_provider_run_started(
                            data_type="realtime_quote",
                            provider=fetcher.name,
                            operation="get_realtime_quote",
                        )
                        quote = self._call_fetcher_method(fetcher, 'get_realtime_quote', stock_code, source="tencent")

                elif source == "tushare":
                    fetcher = self._get_fetcher_by_name("TushareFetcher", capability="realtime_quote")
                    if fetcher is not None and hasattr(fetcher, 'get_realtime_quote'):
                        record_provider_run_started(
                            data_type="realtime_quote",
                            provider=fetcher.name,
                            operation="get_realtime_quote",
                        )
                        quote = self._call_fetcher_method(fetcher, 'get_realtime_quote', raw_stock_code or stock_code)

                provider_name = fetcher.name if fetcher is not None else source

                if quote is not None and quote.has_basic_data():
                    record_provider_run(
                        data_type="realtime_quote",
                        provider=provider_name,
                        operation="get_realtime_quote",
                        success=True,
                        latency_ms=int((time.time() - attempt_start) * 1000),
                        fallback_to=fallback_to if primary_quote is None and self._quote_needs_supplement(quote) else None,
                        record_count=1,
                    )
                    if primary_quote is None:
                        primary_quote = quote
                        primary_fallback_from = failed_sources[0] if failed_sources else None
                        logger.info(f"[Realtime quote] {stock_code} successful (source: {source})")
                        if not self._quote_needs_supplement(primary_quote):
                            return self._enrich_realtime_quote(
                                primary_quote,
                                fallback_from=primary_fallback_from,
                                realtime_cache_ttl=getattr(config, "realtime_cache_ttl", None),
                            )
                        logger.debug(f"[Realtime quote] {stock_code} missing fields, trying to supplement")
                    else:
                        supplement_attempts += 1
                        if supplement_attempts > 1:
                            logger.debug(f"[Realtime quote] {stock_code} supplement attempts reached limit, stopping")
                            break
                        merged = self._merge_quote_fields(primary_quote, quote)
                        if merged:
                            logger.info(f"[Realtime quote] {stock_code} supplemented from {source}: {merged}")
                        if not self._quote_needs_supplement(primary_quote):
                            break
                else:
                    record_provider_run(
                        data_type="realtime_quote",
                        provider=provider_name,
                        operation="get_realtime_quote",
                        success=False,
                        latency_ms=int((time.time() - attempt_start) * 1000),
                        error_type="empty",
                        error_message="empty or incomplete quote",
                        fallback_to=fallback_to,
                        record_count=0,
                    )
                    if primary_quote is None:
                        failed_sources.append(source)

            except Exception as e:
                error_msg = f"[{source}] failed: {str(e)}"
                error_type, error_reason = summarize_exception(e)
                record_provider_run(
                    data_type="realtime_quote",
                    provider=getattr(fetcher, "name", source),
                    operation="get_realtime_quote",
                    success=False,
                    latency_ms=int((time.time() - attempt_start) * 1000),
                    error_type=error_type,
                    error_message=error_reason,
                    fallback_to=fallback_to,
                )
                logger.info(f"[Realtime quote] {stock_code} {error_msg}, trying next source")
                errors.append(error_msg)
                if primary_quote is None:
                    failed_sources.append(source)
                continue

        if primary_quote is not None:
            return self._enrich_realtime_quote(
                primary_quote,
                fallback_from=primary_fallback_from,
                realtime_cache_ttl=getattr(config, "realtime_cache_ttl", None),
            )

        if log_final_failure:
            if errors:
                logger.info(f"[Realtime quote] {stock_code} all sources failed: {'; '.join(errors)}")
            else:
                logger.info(f"[Realtime quote] {stock_code} no available source")

        return None

    # Fields worth supplementing from secondary sources when the primary
    # source returns None for them. Ordered by importance.
    _SUPPLEMENT_FIELDS = [
        'volume_ratio', 'turnover_rate',
        'pe_ratio', 'pb_ratio', 'total_mv', 'circ_mv',
        'amplitude',
    ]

    @classmethod
    def _quote_needs_supplement(cls, quote) -> bool:
        """Check if any key supplementary field is still None."""
        for f in cls._SUPPLEMENT_FIELDS:
            if getattr(quote, f, None) is None:
                return True
        return False

    @classmethod
    def _merge_quote_fields(cls, primary, secondary) -> list:
        """
        Copy non-None fields from secondary into primary where primary is None.
        Returns list of field names that were filled.
        """
        filled = []
        for f in cls._SUPPLEMENT_FIELDS:
            if getattr(primary, f, None) is None:
                val = getattr(secondary, f, None)
                if val is not None:
                    setattr(primary, f, val)
                    filled.append(f)
        return filled

    def _longbridge_preferred(self, capability: str = "realtime_quote") -> bool:
        """Return True when Longbridge keys are configured and available."""
        return self._get_fetcher_by_name(
            "LongbridgeFetcher",
            capability=capability,
        ) is not None

    def _try_fetcher_quote(self, stock_code: str, fetcher_name: str, **kw):
        """Try to get a realtime quote from a named fetcher; returns quote or None."""
        fetcher = self._get_fetcher_by_name(fetcher_name, capability="realtime_quote")
        if fetcher is None or not hasattr(fetcher, 'get_realtime_quote'):
            record_provider_run(
                data_type="realtime_quote",
                provider=fetcher_name,
                operation="get_realtime_quote",
                success=False,
                error_type="unavailable",
                error_message="fetcher unavailable",
            )
            return None
        attempt_start = time.time()
        try:
            record_provider_run_started(
                data_type="realtime_quote",
                provider=fetcher.name,
                operation="get_realtime_quote",
            )
            q = self._call_fetcher_method(fetcher, 'get_realtime_quote', stock_code, **kw)
            if q is not None and q.has_basic_data():
                record_provider_run(
                    data_type="realtime_quote",
                    provider=fetcher.name,
                    operation="get_realtime_quote",
                    success=True,
                    latency_ms=int((time.time() - attempt_start) * 1000),
                    record_count=1,
                )
                return q
            record_provider_run(
                data_type="realtime_quote",
                provider=fetcher.name,
                operation="get_realtime_quote",
                success=False,
                latency_ms=int((time.time() - attempt_start) * 1000),
                error_type="empty",
                error_message="empty or incomplete quote",
                record_count=0,
            )
        except Exception as e:
            error_type, error_reason = summarize_exception(e)
            record_provider_run(
                data_type="realtime_quote",
                provider=fetcher.name,
                operation="get_realtime_quote",
                success=False,
                latency_ms=int((time.time() - attempt_start) * 1000),
                error_type=error_type,
                error_message=error_reason,
            )
            logger.debug(f"[Realtime quote] {stock_code} {fetcher_name} failed: {e}")
        return None

    def _supplement_quote(self, stock_code: str, primary_quote, fetcher_name: str, **kw):
        """Supplement primary_quote with data from fetcher_name."""
        if primary_quote is not None:
            if not self._quote_needs_supplement(primary_quote):
                return primary_quote
            try:
                secondary = self._try_fetcher_quote(stock_code, fetcher_name, **kw)
                if secondary is not None:
                    filled = self._merge_quote_fields(primary_quote, secondary)
                    if filled:
                        logger.info(f"[Realtime quote] {stock_code} supplemented from {fetcher_name}: {filled}")
            except Exception as e:
                logger.debug(f"[Realtime quote] {stock_code} supplement from {fetcher_name} failed: {e}")
            return primary_quote

        q = self._try_fetcher_quote(stock_code, fetcher_name, **kw)
        if q is not None:
            logger.info(f"[Realtime quote] {stock_code} successful from {fetcher_name} (standalone)")
        return q

    def get_chip_distribution(self, stock_code: str):
        """
        Get chip distribution data (with circuit breaker and multi-source fallback).

        Strategy:
        1. Check configuration switch
        2. Check circuit breaker status
        3. Try multiple data sources in priority order
        4. Return None (graceful degradation) if all sources fail

        Args:
            stock_code: Stock code

        Returns:
            ChipDistribution object or None
        """
        stock_code = normalize_stock_code(stock_code)

        from .realtime_types import get_chip_circuit_breaker
        from src.config import get_config

        config = get_config()

        if not config.enable_chip_distribution:
            logger.debug(f"[Chip distribution] disabled, skipping {stock_code}")
            return None

        circuit_breaker = get_chip_circuit_breaker()

        candidate_fetchers = []
        for fetcher in self._get_fetchers_snapshot():
            if not hasattr(fetcher, 'get_chip_distribution'):
                continue

            fetcher_name = fetcher.name
            source_key = f"{fetcher_name.replace('Fetcher', '').lower()}_chip"

            if not circuit_breaker.is_available(source_key):
                logger.debug(f"[Circuit breaker] {fetcher_name} chip interface melted, trying next")
                continue

            candidate_fetchers.append((fetcher, fetcher_name, source_key))

        for index, (fetcher, fetcher_name, source_key) in enumerate(candidate_fetchers):
            fallback_to = (
                candidate_fetchers[index + 1][1]
                if index + 1 < len(candidate_fetchers)
                else None
            )
            attempt_start = time.time()
            try:
                record_provider_run_started(
                    data_type="chip",
                    provider=fetcher_name,
                    operation="get_chip_distribution",
                )
                chip = self._call_fetcher_method(fetcher, 'get_chip_distribution', stock_code)
                latency_ms = int((time.time() - attempt_start) * 1000)
                if _is_meaningful_chip_distribution(chip):
                    record_provider_run(
                        data_type="chip",
                        provider=fetcher_name,
                        operation="get_chip_distribution",
                        success=True,
                        latency_ms=latency_ms,
                        record_count=1,
                    )
                    circuit_breaker.record_success(source_key)
                    logger.info(f"[Chip distribution] {stock_code} successful (source: {fetcher_name})")
                    return chip
                else:
                    record_provider_run(
                        data_type="chip",
                        provider=fetcher_name,
                        operation="get_chip_distribution",
                        success=False,
                        latency_ms=latency_ms,
                        error_type="empty",
                        error_message="empty or incomplete chip distribution",
                        fallback_to=fallback_to,
                        record_count=0,
                    )
                    if chip is not None:
                        logger.warning(
                            "[Chip distribution] %s returned incomplete or placeholder, trying next",
                            fetcher_name,
                        )
                    circuit_breaker.record_inconclusive(source_key)
            except Exception as e:
                error_type, error_reason = summarize_exception(e)
                record_provider_run(
                    data_type="chip",
                    provider=fetcher_name,
                    operation="get_chip_distribution",
                    success=False,
                    latency_ms=int((time.time() - attempt_start) * 1000),
                    error_type=error_type,
                    error_message=error_reason,
                    fallback_to=fallback_to,
                )
                logger.warning(f"[Chip distribution] {fetcher_name} failed for {stock_code}: {e}")
                circuit_breaker.record_failure(source_key, str(e))
                continue

        logger.warning(f"[Chip distribution] {stock_code} all sources failed")
        return None

    def get_stock_name(self, stock_code: str, allow_realtime: bool = True) -> Optional[str]:
        """
        Get stock Chinese name (automatic failover).

        Tries multiple sources:
        1. Memory cache
        2. Local mapping and stocks.index.json
        3. Real-time quote (if allowed)
        4. Each fetcher's get_stock_name method

        Args:
            stock_code: Stock code
            allow_realtime: Whether to query realtime quote first. Set False to avoid
                expensive realtime source calls.

        Returns:
            Stock name or None if all fail.
        """
        raw_stock_code = (stock_code or "").strip()
        stock_code = normalize_stock_code(stock_code)
        static_name = STOCK_NAME_MAP.get(stock_code)

        # 1. Check cache
        cached_name = self._get_cached_stock_name(stock_code)
        if cached_name is not None:
            return cached_name

        if is_meaningful_stock_name(static_name, stock_code):
            return self._cache_stock_name(stock_code, static_name) or static_name

        index_name = get_index_stock_name(stock_code)
        if is_meaningful_stock_name(index_name, stock_code):
            return self._cache_stock_name(stock_code, index_name) or index_name

        # 2. Try real-time quote (fastest)
        if allow_realtime:
            quote = self.get_realtime_quote(raw_stock_code or stock_code, log_final_failure=False)
            if quote and hasattr(quote, 'name') and is_meaningful_stock_name(getattr(quote, 'name', ''), stock_code):
                name = quote.name
                self._cache_stock_name(stock_code, name)
                logger.info(f"[Stock name] from realtime quote: {stock_code} -> {name}")
                return name

        # 3. Try each fetcher
        from .akshare_fetcher import _is_us_code
        is_us = _is_us_code(stock_code)
        _US_CAPABLE_FETCHERS = {"YfinanceFetcher", "LongbridgeFetcher", "FinnhubFetcher", "AlphaVantageFetcher"}
        for fetcher in self._get_fetchers_snapshot():
            if not hasattr(fetcher, 'get_stock_name'):
                continue
            if is_us and fetcher.name not in _US_CAPABLE_FETCHERS:
                continue
            if not self._is_fetcher_available(fetcher, capability="stock_name"):
                continue
            try:
                name = self._call_fetcher_method(fetcher, 'get_stock_name', stock_code)
                if is_meaningful_stock_name(name, stock_code):
                    self._cache_stock_name(stock_code, name)
                    logger.info(f"[Stock name] from {fetcher.name}: {stock_code} -> {name}")
                    return name
            except Exception as e:
                logger.debug(f"[Stock name] {fetcher.name} failed: {e}")
                continue

        logger.warning(f"[Stock name] all sources failed for {stock_code}")
        return ""

    def get_belong_boards(self, stock_code: str) -> List[Dict[str, Any]]:
        """
        Get stock membership boards through capability probing.

        Keep this at manager layer to avoid changing BaseFetcher abstraction.
        """
        stock_code = normalize_stock_code(stock_code)
        if _market_tag(stock_code) != "cn":
            return []
        candidate_fetchers = [
            fetcher
            for fetcher in self._fetchers
            if hasattr(fetcher, "get_belong_board")
        ]
        for index, fetcher in enumerate(candidate_fetchers):
            fallback_to = (
                candidate_fetchers[index + 1].name
                if index + 1 < len(candidate_fetchers)
                else None
            )
            start = time.time()
            try:
                record_provider_run_started(
                    data_type="belong_boards",
                    provider=fetcher.name,
                    operation="get_belong_board",
                )
                raw_data = fetcher.get_belong_board(stock_code)
                boards = self._normalize_belong_boards(raw_data)
                if boards:
                    record_provider_run(
                        data_type="belong_boards",
                        provider=fetcher.name,
                        operation="get_belong_board",
                        success=True,
                        latency_ms=int((time.time() - start) * 1000),
                        record_count=len(boards),
                    )
                    logger.info(f"[{fetcher.name}] get belong boards success: {stock_code}, count={len(boards)}")
                    return boards
                record_provider_run(
                    data_type="belong_boards",
                    provider=fetcher.name,
                    operation="get_belong_board",
                    success=False,
                    latency_ms=int((time.time() - start) * 1000),
                    error_type="empty",
                    error_message="empty belong boards",
                    fallback_to=fallback_to,
                    record_count=0,
                )
            except Exception as e:
                error_type, error_reason = summarize_exception(e)
                record_provider_run(
                    data_type="belong_boards",
                    provider=fetcher.name,
                    operation="get_belong_board",
                    success=False,
                    latency_ms=int((time.time() - start) * 1000),
                    error_type=error_type,
                    error_message=error_reason,
                    fallback_to=fallback_to,
                )
                logger.debug(f"[{fetcher.name}] get belong boards failed: {e}")
                continue
        return []

    def prefetch_stock_names(self, stock_codes: List[str], use_bulk: bool = False) -> None:
        """
        Pre-fetch stock names into cache before parallel analysis (Issue #455).

        When use_bulk=False, only calls get_stock_name per code (no get_stock_list),
        avoiding full-market fetch. Sequential execution to avoid rate limits.

        Args:
            stock_codes: Stock codes to prefetch.
            use_bulk: If True, may use get_stock_list (full fetch). Default False.
        """
        if not stock_codes:
            return
        stock_codes = [normalize_stock_code(c) for c in stock_codes]
        if use_bulk:
            self.batch_get_stock_names(stock_codes)
            return
        for code in stock_codes:
            self.get_stock_name(code, allow_realtime=False)

    def batch_get_stock_names(self, stock_codes: List[str]) -> Dict[str, str]:
        """
        Batch get stock Chinese names.

        First tries to get stock list from bulk-capable sources,
        then fetches individually for remaining missing codes.

        Args:
            stock_codes: List of stock codes

        Returns:
            Dictionary of {code: name}
        """
        result = {}
        missing_codes = set(stock_codes)

        # 1. Check cache
        self._ensure_concurrency_guards()
        with self._stock_name_cache_lock:
            for code in stock_codes:
                cached_name = self._stock_name_cache.get(code)
                if cached_name is not None:
                    result[code] = cached_name
                    missing_codes.discard(code)

        if not missing_codes:
            return result

        # 2. Try bulk fetch
        for fetcher in self._get_fetchers_snapshot():
            if not hasattr(fetcher, 'get_stock_list') or not missing_codes:
                continue
            if not self._is_fetcher_available(fetcher, capability="stock_list"):
                continue
            try:
                stock_list = self._call_fetcher_method(fetcher, 'get_stock_list')
                if stock_list is not None and not stock_list.empty:
                    cache_updates: Dict[str, str] = {}
                    for _, row in stock_list.iterrows():
                        code = row.get('code')
                        name = row.get('name')
                        if code and name:
                            cache_updates[code] = name
                            if code in missing_codes:
                                result[code] = name
                                missing_codes.discard(code)

                    if cache_updates:
                        with self._stock_name_cache_lock:
                            self._stock_name_cache.update(cache_updates)

                    if not missing_codes:
                        break

                    logger.info(f"[Stock name] bulk from {fetcher.name} complete, remaining {len(missing_codes)}")
            except Exception as e:
                logger.debug(f"[Stock name] {fetcher.name} bulk failed: {e}")
                continue

        # 3. Individual fetch for remaining
        for code in list(missing_codes):
            name = self.get_stock_name(code)
            if name:
                result[code] = name
                missing_codes.discard(code)

        logger.info(f"[Stock name] batch complete, success {len(result)}/{len(stock_codes)}")
        return result

    def get_main_indices(self, region: str = "cn") -> List[Dict[str, Any]]:
        """Get major index real-time quotes (automatic failover)."""
        if region == "cn":
            tickflow_fetcher = self._get_tickflow_fetcher()
            if tickflow_fetcher is not None:
                try:
                    data = tickflow_fetcher.get_main_indices(region=region)
                    if data:
                        logger.info("[TickFlowFetcher] get indices success")
                        return data
                except Exception as e:
                    logger.warning(f"[TickFlowFetcher] get indices failed: {e}")

        for fetcher in self._fetchers:
            try:
                data = fetcher.get_main_indices(region=region)
                if data:
                    logger.info(f"[{fetcher.name}] get indices success")
                    return data
            except Exception as e:
                logger.warning(f"[{fetcher.name}] get indices failed: {e}")
                continue
        return []

    def get_market_stats(self, *, purpose: str = "unspecified") -> Dict[str, Any]:
        """Get market advance-decline statistics (automatic failover)."""
        logger.info("[MarketStats] component=market_stats action=start purpose=%s", purpose)
        tickflow_fetcher = self._get_tickflow_fetcher()
        if tickflow_fetcher is not None:
            started_at = time.monotonic()
            try:
                data = tickflow_fetcher.get_market_stats()
                elapsed = time.monotonic() - started_at
                if data:
                    logger.info(
                        "[MarketStats] component=market_stats action=provider_success "
                        "purpose=%s provider=TickFlowFetcher elapsed=%.2fs",
                        purpose,
                        elapsed,
                    )
                    return data
                logger.info(
                    "[MarketStats] component=market_stats action=provider_empty "
                    "purpose=%s provider=TickFlowFetcher elapsed=%.2fs",
                    purpose,
                    elapsed,
                )
            except Exception as e:
                elapsed = time.monotonic() - started_at
                logger.warning(
                    "[MarketStats] component=market_stats action=provider_failed "
                    "purpose=%s provider=TickFlowFetcher elapsed=%.2fs error=%s",
                    purpose,
                    elapsed,
                    e,
                )

        for fetcher in self._fetchers:
            started_at = time.monotonic()
            try:
                data = fetcher.get_market_stats()
                elapsed = time.monotonic() - started_at
                if data:
                    logger.info(
                        "[MarketStats] component=market_stats action=provider_success "
                        "purpose=%s provider=%s elapsed=%.2fs",
                        purpose,
                        fetcher.name,
                        elapsed,
                    )
                    return data
                logger.info(
                    "[MarketStats] component=market_stats action=provider_empty "
                    "purpose=%s provider=%s elapsed=%.2fs",
                    purpose,
                    fetcher.name,
                    elapsed,
                )
            except Exception as e:
                elapsed = time.monotonic() - started_at
                logger.warning(
                    "[MarketStats] component=market_stats action=provider_failed "
                    "purpose=%s provider=%s elapsed=%.2fs error=%s",
                    purpose,
                    fetcher.name,
                    elapsed,
                    e,
                )
                continue
        logger.warning("[MarketStats] component=market_stats action=complete status=empty purpose=%s", purpose)
        return {}

    def _run_with_timeout(
        self,
        task: Callable[[], Any],
        timeout_seconds: float,
        task_name: str,
    ) -> Tuple[Optional[Any], Optional[str], int]:
        """
        Execute a task in a short-lived thread and enforce a timeout.

        Returns:
            (result, error, duration_ms)
        """
        start = time.time()
        timeout_value = max(0.0, timeout_seconds)
        if timeout_value <= 0:
            return None, f"{task_name} timeout", 0
        result_holder: Dict[str, Any] = {}
        error_holder: Dict[str, Exception] = {}

        if not self._fundamental_timeout_slots.acquire(blocking=False):
            return None, f"{task_name} timeout worker pool exhausted", int(timeout_value * 1000)

        def runner() -> None:
            try:
                result_holder["value"] = task()
            except Exception as exc:
                error_holder["value"] = exc
            finally:
                try:
                    self._fundamental_timeout_slots.release()
                except ValueError:
                    pass

        worker = Thread(target=runner, daemon=True, name=f"fundamental-{task_name}")
        try:
            worker.start()
        except Exception as exc:
            try:
                self._fundamental_timeout_slots.release()
            except ValueError:
                pass
            return None, str(exc), int((time.time() - start) * 1000)
        worker.join(timeout=timeout_value)
        if worker.is_alive():
            return None, f"{task_name} timeout", int(timeout_value * 1000)
        if "value" in error_holder:
            return None, str(error_holder["value"]), int((time.time() - start) * 1000)
        return result_holder.get("value"), None, int((time.time() - start) * 1000)

    def _run_with_retry(
        self,
        task: Callable[[], Any],
        timeout_seconds: float,
        task_name: str,
    ) -> Tuple[Optional[Any], Optional[str], int]:
        """
        Execute a task with bounded budget and best-effort retries.

        Returns:
            (result, error, total_duration_ms)
        """
        config = self._get_fundamental_config()
        attempts = max(1, int(config.fundamental_retry_max))
        remaining_seconds = max(0.0, float(timeout_seconds))
        total_cost_ms = 0
        last_error: Optional[str] = None

        for _ in range(attempts):
            if remaining_seconds <= 0:
                break
            result, err, cost_ms = self._run_with_timeout(task, remaining_seconds, task_name)
            total_cost_ms += cost_ms
            remaining_seconds = max(0.0, remaining_seconds - cost_ms / 1000)
            if err is None:
                return result, None, total_cost_ms
            last_error = err
            if remaining_seconds <= 0:
                break

        return None, last_error, total_cost_ms

    def _get_fundamental_config(self):
        from src.config import get_config
        return get_config()

    @staticmethod
    def _normalize_source_chain(
        entries: Any,
        provider: str,
        result: str,
        duration_ms: int,
    ) -> List[Dict[str, Any]]:
        """Normalize free-form source chain entries to structured dict list."""
        if entries is None:
            return [{"provider": provider, "result": result, "duration_ms": duration_ms}]

        normalized: List[Dict[str, Any]] = []
        if not isinstance(entries, (list, tuple)):
            entries = [entries]

        for item in entries:
            if isinstance(item, dict):
                normalized.append({
                    "provider": str(item.get("provider") or provider),
                    "result": str(item.get("result") or result),
                    "duration_ms": int(item.get("duration_ms", duration_ms)),
                })
                continue

            if item is None:
                continue

            provider_name = str(item)
            normalized.append({
                "provider": provider_name,
                "result": result,
                "duration_ms": duration_ms,
            })

        if not normalized:
            return [{"provider": provider, "result": result, "duration_ms": duration_ms}]

        return normalized

    @staticmethod
    def _block_status(payload: Dict[str, Any], available: bool = True) -> str:
        if not available:
            return "not_supported"
        if not payload:
            return "partial"
        return "ok"

    @staticmethod
    def _build_fundamental_block(
        status: str,
        payload: Optional[Dict[str, Any]] = None,
        source_chain: Optional[List[Dict[str, Any]]] = None,
        errors: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "coverage": {"status": status},
            "source_chain": source_chain or [],
            "errors": errors or [],
            "data": payload or {},
        }

    @staticmethod
    def _has_meaningful_payload(payload: Any) -> bool:
        if payload is None:
            return False
        if isinstance(payload, str):
            normalized = payload.strip().lower()
            return normalized not in ("", "-", "nan", "none", "null", "n/a", "na")
        if isinstance(payload, dict):
            return any(DataFetcherManager._has_meaningful_payload(v) for v in payload.values())
        if isinstance(payload, pd.DataFrame):
            if payload.empty:
                return False
            return any(
                DataFetcherManager._has_meaningful_payload(v)
                for v in payload.to_numpy().flat
            )
        if isinstance(payload, (pd.Series, pd.Index)):
            return any(DataFetcherManager._has_meaningful_payload(v) for v in payload.tolist())
        if isinstance(payload, np.ndarray):
            if payload.ndim == 0:
                payload = payload.item()
            else:
                return any(
                    DataFetcherManager._has_meaningful_payload(v)
                    for v in payload.flat
                )
        if isinstance(payload, (list, tuple, set)):
            return any(DataFetcherManager._has_meaningful_payload(v) for v in payload)
        if DataFetcherManager._try_scalar_isna(payload, "fundamental_payload") is True:
            return False
        return True

    @staticmethod
    def _infer_block_status(payload: Any, fallback_status: str) -> str:
        if DataFetcherManager._has_meaningful_payload(payload):
            return "ok"
        if fallback_status in ("failed", "partial", "not_supported"):
            return fallback_status
        return "partial"

    @staticmethod
    def _should_cache_fundamental_context(context: Any) -> bool:
        if not isinstance(context, dict):
            return False
        status = str(context.get("status", "")).strip().lower()
        if status == "ok":
            return True
        if status == "failed":
            return False
        for block in (
            "valuation",
            "growth",
            "earnings",
            "institution",
            "capital_flow",
            "dragon_tiger",
            "boards",
        ):
            payload = context.get(block, {})
            if isinstance(payload, dict) and DataFetcherManager._has_meaningful_payload(payload.get("data")):
                return True
        return False

    def _build_market_not_supported(self, market: str, reason: str) -> Dict[str, Any]:
        blocks = {
            "valuation": self._build_fundamental_block(
                "partial" if market == "etf" else "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                [reason],
            ),
            "growth": self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                [reason],
            ),
            "earnings": self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                [reason],
            ),
            "institution": self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                [reason],
            ),
            "capital_flow": self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                [reason],
            ),
            "dragon_tiger": self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                [reason],
            ),
            "boards": self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                [reason],
            ),
        }
        return {
            "market": market,
            "status": "partial" if market == "etf" else "not_supported",
            "coverage": {
                block: blocks[block]["status"] for block in blocks
            },
            "source_chain": [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
            "errors": [reason],
            **blocks,
        }

    def _build_offshore_fundamental_context(
        self,
        stock_code: str,
        market: str,
        budget_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """HK/US fundamental aggregation via yfinance."""
        from src.config import get_config

        config = get_config()
        stage_timeout = float(
            budget_seconds if budget_seconds is not None else config.fundamental_stage_timeout_seconds
        )
        stage_timeout = max(0.0, stage_timeout)
        fetch_timeout = float(config.fundamental_fetch_timeout_seconds)
        fetch_timeout = max(0.0, fetch_timeout)

        cache_ttl = int(config.fundamental_cache_ttl_seconds)
        cache_max_entries = max(0, int(getattr(config, "fundamental_cache_max_entries", 256)))
        cache_key = self._get_fundamental_cache_key(stock_code, stage_timeout)
        if cache_ttl > 0:
            self._prune_fundamental_cache(cache_ttl, cache_max_entries)
            with self._fundamental_cache_lock:
                cache_item = self._fundamental_cache.get(cache_key)
                if cache_item:
                    age = time.time() - float(cache_item.get("ts", 0))
                    if age <= cache_ttl:
                        return cache_item.get("context", {})

        result_ctx: Dict[str, Any] = {
            "market": market,
            "valuation": {},
            "growth": {},
            "earnings": {},
            "institution": {},
            "capital_flow": {},
            "dragon_tiger": {},
            "boards": {},
            "belong_boards": [],
            "coverage": {},
            "source_chain": [],
            "errors": [],
        }
        start_ts = time.time()

        # Valuation: reuse realtime quote payload
        valuation_timeout = min(fetch_timeout, stage_timeout) if stage_timeout > 0 else 0
        if valuation_timeout > 0:
            quote_payload, valuation_err, valuation_ms = self._run_with_retry(
                lambda: self.get_realtime_quote(stock_code),
                valuation_timeout,
                "fundamental_valuation",
            )
        else:
            quote_payload, valuation_err, valuation_ms = None, "fundamental stage timeout", 0
        valuation_payload = {
            "pe_ratio": getattr(quote_payload, "pe_ratio", None) if quote_payload else None,
            "pb_ratio": getattr(quote_payload, "pb_ratio", None) if quote_payload else None,
            "total_mv": getattr(quote_payload, "total_mv", None) if quote_payload else None,
            "circ_mv": getattr(quote_payload, "circ_mv", None) if quote_payload else None,
        }
        valuation_status = self._infer_block_status(
            valuation_payload,
            "partial" if quote_payload is not None else "not_supported",
        )
        if valuation_status == "partial" and valuation_err and not self._has_meaningful_payload(valuation_payload):
            valuation_status = "failed"
        result_ctx["valuation"] = self._build_fundamental_block(
            valuation_status,
            valuation_payload,
            self._normalize_source_chain(
                [{"provider": "realtime_quote", "result": valuation_status, "duration_ms": valuation_ms}],
                "realtime_quote",
                valuation_status,
                valuation_ms,
            ),
            [valuation_err] if valuation_err else [],
        )

        # Fundamental bundle via yfinance.
        bundle_timeout = min(fetch_timeout, max(stage_timeout - (time.time() - start_ts), 0.0))
        if bundle_timeout <= 0:
            bundle_payload, bundle_err, bundle_ms = {}, "fundamental stage timeout", 0
        else:
            bundle_payload, bundle_err, bundle_ms = self._run_with_retry(
                lambda: self._yfinance_fundamental_adapter.get_fundamental_bundle(stock_code),
                bundle_timeout,
                "fundamental_bundle_yfinance",
            )
        if not isinstance(bundle_payload, dict):
            bundle_payload = {}

        bundle_chain = self._normalize_source_chain(
            bundle_payload.get("source_chain", []),
            "fundamental_bundle_yfinance",
            str(bundle_payload.get("status", "not_supported")),
            bundle_ms,
        )
        adapter_errors = list(bundle_payload.get("errors", []))
        if bundle_err:
            adapter_errors.append(bundle_err)

        growth_payload = bundle_payload.get("growth", {}) if isinstance(bundle_payload.get("growth"), dict) else {}
        earnings_payload = bundle_payload.get("earnings", {}) if isinstance(bundle_payload.get("earnings"), dict) else {}
        belong_boards = bundle_payload.get("belong_boards") if isinstance(bundle_payload.get("belong_boards"), list) else []

        growth_status = self._infer_block_status(growth_payload, str(bundle_payload.get("status", "not_supported")))
        earnings_status = self._infer_block_status(earnings_payload, str(bundle_payload.get("status", "not_supported")))

        result_ctx["growth"] = self._build_fundamental_block(
            growth_status,
            growth_payload,
            bundle_chain,
            list(adapter_errors),
        )
        result_ctx["earnings"] = self._build_fundamental_block(
            earnings_status,
            earnings_payload,
            bundle_chain,
            list(adapter_errors),
        )

        # institution / capital_flow / dragon_tiger / boards: not supported
        for block in ("institution", "capital_flow", "dragon_tiger", "boards"):
            result_ctx[block] = self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                ["not supported for offshore market"],
            )

        result_ctx["belong_boards"] = belong_boards

        block_statuses = {
            "valuation": result_ctx["valuation"].get("status", "not_supported"),
            "growth": growth_status,
            "earnings": earnings_status,
            "institution": "not_supported",
            "capital_flow": "not_supported",
            "dragon_tiger": "not_supported",
            "boards": "not_supported",
        }
        result_ctx["coverage"] = block_statuses
        for block in ("valuation", "growth", "earnings", "institution", "capital_flow", "dragon_tiger", "boards"):
            result_ctx["errors"].extend(result_ctx[block].get("errors", []))
            result_ctx["source_chain"].extend(result_ctx[block].get("source_chain", []))

        active_statuses = {"valuation": valuation_status, "growth": growth_status, "earnings": earnings_status}
        if all(value == "not_supported" for value in active_statuses.values()):
            result_ctx["status"] = "not_supported"
        elif "failed" in active_statuses.values() or "partial" in active_statuses.values():
            result_ctx["status"] = "partial"
        else:
            result_ctx["status"] = "ok"

        result_ctx["elapsed_ms"] = int((time.time() - start_ts) * 1000)
        if cache_ttl > 0 and self._should_cache_fundamental_context(result_ctx):
            with self._fundamental_cache_lock:
                self._fundamental_cache[cache_key] = {
                    "ts": time.time(),
                    "context": result_ctx,
                }
            self._prune_fundamental_cache(cache_ttl, cache_max_entries)
        return result_ctx

    def build_failed_fundamental_context(self, stock_code: str, reason: str) -> Dict[str, Any]:
        """Build a consistent failed-context payload for caller-side fallback."""
        market = _market_tag(stock_code)
        block_names = (
            "valuation",
            "growth",
            "earnings",
            "institution",
            "capital_flow",
            "dragon_tiger",
            "boards",
        )
        blocks = {
            block: self._build_fundamental_block(
                "failed",
                {},
                [{"provider": "fundamental_pipeline", "result": "failed", "duration_ms": 0}],
                [reason],
            )
            for block in block_names
        }
        return {
            "market": market,
            "status": "failed",
            "coverage": {block: "failed" for block in block_names},
            "source_chain": [{"provider": "fundamental_pipeline", "result": "failed", "duration_ms": 0}],
            "errors": [reason],
            **blocks,
        }

    def get_fundamental_context(
        self,
        stock_code: str,
        budget_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Aggregate fundamental blocks with fail-open semantics.
        """
        from src.config import get_config

        config = get_config()
        if not config.enable_fundamental_pipeline:
            return self._build_market_not_supported(
                market=_market_tag(stock_code),
                reason="fundamental pipeline disabled",
            )

        stock_code = normalize_stock_code(stock_code)
        market = _market_tag(stock_code)
        is_etf = _is_etf_code(stock_code)
        if market in {"us", "hk", "jp", "kr"}:
            return self._build_offshore_fundamental_context(
                stock_code,
                market=market,
                budget_seconds=budget_seconds,
            )

        stage_timeout = float(
            budget_seconds if budget_seconds is not None else config.fundamental_stage_timeout_seconds
        )
        stage_timeout = max(0.0, stage_timeout)
        fetch_timeout = float(config.fundamental_fetch_timeout_seconds)
        fetch_timeout = max(0.0, fetch_timeout)

        cache_ttl = int(config.fundamental_cache_ttl_seconds)
        cache_max_entries = max(0, int(getattr(config, "fundamental_cache_max_entries", 256)))
        cache_key = self._get_fundamental_cache_key(stock_code, stage_timeout)
        if cache_ttl > 0:
            self._prune_fundamental_cache(cache_ttl, cache_max_entries)
            with self._fundamental_cache_lock:
                cache_item = self._fundamental_cache.get(cache_key)
                if cache_item:
                    age = time.time() - float(cache_item.get("ts", 0))
                    if age <= cache_ttl:
                        return cache_item.get("context", {})

        remaining_seconds = stage_timeout
        result_ctx: Dict[str, Any] = {
            "market": market,
            "valuation": {},
            "growth": {},
            "earnings": {},
            "institution": {},
            "capital_flow": {},
            "dragon_tiger": {},
            "boards": {},
            "coverage": {},
            "source_chain": [],
            "errors": [],
        }

        start_ts = time.time()

        def _consume_budget(consumed_ms: int) -> None:
            nonlocal remaining_seconds
            remaining_seconds = max(0.0, remaining_seconds - consumed_ms / 1000.0)

        valuation_timeout = min(fetch_timeout, remaining_seconds)
        if valuation_timeout > 0:
            quote_payload, valuation_err, valuation_ms = self._run_with_retry(
                lambda: self.get_realtime_quote(stock_code),
                valuation_timeout,
                "fundamental_valuation",
            )
            _consume_budget(valuation_ms)
        else:
            quote_payload, valuation_err, valuation_ms = None, "fundamental stage timeout", 0

        valuation_payload = {
            "pe_ratio": getattr(quote_payload, "pe_ratio", None) if quote_payload else None,
            "pb_ratio": getattr(quote_payload, "pb_ratio", None) if quote_payload else None,
            "total_mv": getattr(quote_payload, "total_mv", None) if quote_payload else None,
            "circ_mv": getattr(quote_payload, "circ_mv", None) if quote_payload else None,
        }
        valuation_status = self._infer_block_status(
            valuation_payload,
            "partial" if quote_payload is not None else "not_supported",
        )
        if valuation_status == "partial" and valuation_err and not self._has_meaningful_payload(valuation_payload):
            valuation_status = "failed"
        result_ctx["valuation"] = self._build_fundamental_block(
            valuation_status,
            valuation_payload,
            self._normalize_source_chain(
                [{"provider": "realtime_quote", "result": valuation_status, "duration_ms": valuation_ms}],
                "realtime_quote",
                valuation_status,
                valuation_ms,
            ),
            [valuation_err] if valuation_err else [],
        )

        # growth / earnings / institution (one AkShare call)
        if remaining_seconds <= 0:
            bundle_status = "failed"
            bundle_payload: Dict[str, Any] = {}
            bundle_errors = ["fundamental stage timeout"]
            bundle_ms = 0
        else:
            bundle_timeout = min(fetch_timeout, remaining_seconds)
            bundle_payload, bundle_err_msg, bundle_ms = self._run_with_retry(
                lambda: self._fundamental_adapter.get_fundamental_bundle(stock_code),
                bundle_timeout,
                "fundamental_bundle",
            )
            _consume_budget(bundle_ms)
            if not isinstance(bundle_payload, dict):
                bundle_status = "failed"
                bundle_payload = {}
                bundle_errors = ["fundamental_bundle failed"]
                if bundle_err_msg:
                    bundle_errors.append(bundle_err_msg)
            else:
                bundle_status = str(bundle_payload.get("status", "not_supported"))
                bundle_errors = [bundle_err_msg] if bundle_err_msg else []

        bundle_chain = self._normalize_source_chain(
            bundle_payload.get("source_chain", []),
            "fundamental_bundle",
            bundle_status,
            bundle_ms,
        ) if isinstance(bundle_payload, dict) else self._normalize_source_chain(
            None,
            "fundamental_bundle",
            bundle_status,
            bundle_ms,
        )
        growth_payload = bundle_payload.get("growth", {}) if isinstance(bundle_payload, dict) else {}
        earnings_payload = bundle_payload.get("earnings", {}) if isinstance(bundle_payload, dict) else {}
        institution_payload = bundle_payload.get("institution", {}) if isinstance(bundle_payload, dict) else {}
        if not isinstance(growth_payload, dict):
            growth_payload = {}
        else:
            growth_payload = dict(growth_payload)
        if not isinstance(earnings_payload, dict):
            earnings_payload = {}
        else:
            earnings_payload = dict(earnings_payload)
        if not isinstance(institution_payload, dict):
            institution_payload = {}
        else:
            institution_payload = dict(institution_payload)

        # Derive TTM dividend yield from already-fetched quote price; avoid extra quote calls.
        earnings_extra_errors: List[str] = []
        dividend_payload = earnings_payload.get("dividend")
        if isinstance(dividend_payload, dict):
            dividend_payload = dict(dividend_payload)
            ttm_cash_raw = dividend_payload.get("ttm_cash_dividend_per_share")
            ttm_cash = None
            if ttm_cash_raw is not None:
                try:
                    ttm_cash = float(ttm_cash_raw)
                except (TypeError, ValueError):
                    earnings_extra_errors.append("invalid_ttm_cash_dividend_per_share")
            if isinstance(quote_payload, dict):
                latest_price_raw = quote_payload.get("price")
            else:
                latest_price_raw = getattr(quote_payload, "price", None) if quote_payload else None
            latest_price = None
            if latest_price_raw is not None:
                try:
                    latest_price = float(latest_price_raw)
                except (TypeError, ValueError):
                    latest_price = None
            ttm_yield = None
            if ttm_cash is not None:
                if latest_price is not None and latest_price > 0:
                    ttm_yield = round(ttm_cash / latest_price * 100.0, 4)
                else:
                    earnings_extra_errors.append("invalid_price_for_ttm_dividend_yield")

            dividend_payload["ttm_dividend_yield_pct"] = ttm_yield
            if ttm_yield is not None:
                dividend_payload["yield_formula"] = "ttm_cash_dividend_per_share / latest_price * 100"
            earnings_payload["dividend"] = dividend_payload

        adapter_errors = list(bundle_payload.get("errors", [])) if isinstance(bundle_payload, dict) else []
        adapter_errors.extend(bundle_errors)
        growth_errors = list(adapter_errors)
        earnings_errors = list(adapter_errors)
        earnings_errors.extend(earnings_extra_errors)
        institution_errors = list(adapter_errors)

        growth_status = self._infer_block_status(growth_payload, bundle_status)
        earnings_status = self._infer_block_status(earnings_payload, bundle_status)
        institution_status = self._infer_block_status(institution_payload, bundle_status)

        result_ctx["growth"] = self._build_fundamental_block(
            growth_status,
            growth_payload,
            bundle_chain,
            growth_errors,
        )
        result_ctx["earnings"] = self._build_fundamental_block(
            earnings_status,
            earnings_payload,
            bundle_chain,
            earnings_errors,
        )
        result_ctx["institution"] = self._build_fundamental_block(
            institution_status,
            institution_payload,
            bundle_chain,
            institution_errors,
        )

        # capital flow
        if is_etf:
            result_ctx["capital_flow"] = self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                ["etf not fully supported"],
            )
            result_ctx["dragon_tiger"] = self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                ["etf not fully supported"],
            )
            result_ctx["boards"] = self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                ["etf not fully supported"],
            )
            result_ctx["status"] = "partial"
        else:
            capital_flow_budget = min(fetch_timeout, remaining_seconds)
            capital_flow_start = time.time()
            result_ctx["capital_flow"] = self.get_capital_flow_context(
                stock_code,
                budget_seconds=capital_flow_budget,
            )
            _consume_budget(int((time.time() - capital_flow_start) * 1000))

            dragon_tiger_budget = min(fetch_timeout, remaining_seconds)
            dragon_tiger_start = time.time()
            result_ctx["dragon_tiger"] = self.get_dragon_tiger_context(
                stock_code,
                budget_seconds=dragon_tiger_budget,
            )
            _consume_budget(int((time.time() - dragon_tiger_start) * 1000))

            result_ctx["boards"] = self.get_board_context(
                stock_code,
                budget_seconds=min(fetch_timeout, remaining_seconds),
            )

        block_statuses = {
            "valuation": result_ctx["valuation"].get("status", "not_supported"),
            "growth": result_ctx["growth"].get("status", "not_supported"),
            "earnings": result_ctx["earnings"].get("status", "not_supported"),
            "institution": result_ctx["institution"].get("status", "not_supported"),
            "capital_flow": result_ctx["capital_flow"].get("status", "not_supported"),
            "dragon_tiger": result_ctx["dragon_tiger"].get("status", "not_supported"),
            "boards": result_ctx["boards"].get("status", "not_supported"),
        }
        result_ctx["coverage"] = block_statuses
        for block in (
            "valuation",
            "growth",
            "earnings",
            "institution",
            "capital_flow",
            "dragon_tiger",
            "boards",
        ):
            result_ctx["errors"].extend(result_ctx[block].get("errors", []))
            result_ctx["source_chain"].extend(result_ctx[block].get("source_chain", []))

        if is_etf:
            result_ctx["status"] = (
                "not_supported" if all(value == "not_supported" for value in block_statuses.values()) else "partial"
            )
        elif all(value == "not_supported" for value in block_statuses.values()):
            result_ctx["status"] = "not_supported"
        elif "failed" in block_statuses.values() or "partial" in block_statuses.values():
            result_ctx["status"] = "partial"
        else:
            result_ctx["status"] = "ok"

        result_ctx["elapsed_ms"] = int((time.time() - start_ts) * 1000)
        if cache_ttl > 0 and self._should_cache_fundamental_context(result_ctx):
            with self._fundamental_cache_lock:
                self._fundamental_cache[cache_key] = {
                    "ts": time.time(),
                    "context": result_ctx,
                }
            self._prune_fundamental_cache(cache_ttl, cache_max_entries)
        return result_ctx

    def get_capital_flow_context(self, stock_code: str, budget_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Capital flow block (fail-open)."""
        from src.config import get_config

        config = get_config()
        stock_code = normalize_stock_code(stock_code)
        timeout = float(budget_seconds if budget_seconds is not None else config.fundamental_fetch_timeout_seconds)
        if _market_tag(stock_code) != "cn" or _is_etf_code(stock_code):
            return self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                ["not supported"],
            )

        if timeout <= 0:
            return self._build_fundamental_block(
                "failed",
                {},
                [{"provider": "fundamental_pipeline", "result": "failed", "duration_ms": 0}],
                ["fundamental stage timeout"],
            )
        payload, err, cost_ms = self._run_with_retry(
            lambda: self._fundamental_adapter.get_capital_flow(stock_code),
            timeout,
            "capital_flow",
        )
        if not isinstance(payload, dict):
            return self._build_fundamental_block(
                "failed",
                {},
                [{"provider": "fundamental_pipeline", "result": "failed", "duration_ms": cost_ms}],
                [err or "capital_flow failed"],
            )

        stock_flow = payload.get("stock_flow") or {}
        sector_rankings = payload.get("sector_rankings") or {}
        has_stock_flow = False
        if isinstance(stock_flow, dict):
            has_stock_flow = any(v is not None for v in stock_flow.values())
        has_sector_rankings = bool(sector_rankings.get("top")) or bool(sector_rankings.get("bottom"))
        adapter_status = str(payload.get("status", "not_supported"))
        if has_stock_flow or has_sector_rankings:
            capital_flow_status = "ok"
        elif adapter_status == "not_supported":
            capital_flow_status = "not_supported"
        else:
            capital_flow_status = "partial"

        return self._build_fundamental_block(
            capital_flow_status,
            {
                "stock_flow": payload.get("stock_flow", {}),
                "sector_rankings": payload.get("sector_rankings", {}),
            },
            self._normalize_source_chain(
                payload.get("source_chain", []),
                "capital_flow",
                capital_flow_status,
                cost_ms,
            ),
            list(payload.get("errors", [])) + ([err] if err else []),
        )

    def get_dragon_tiger_context(self, stock_code: str, budget_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Dragon-tiger board block (fail-open)."""
        from src.config import get_config

        config = get_config()
        stock_code = normalize_stock_code(stock_code)
        timeout = float(budget_seconds if budget_seconds is not None else config.fundamental_fetch_timeout_seconds)
        if _market_tag(stock_code) != "cn" or _is_etf_code(stock_code):
            return self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                ["not supported"],
            )

        if timeout <= 0:
            return self._build_fundamental_block(
                "failed",
                {},
                [{"provider": "fundamental_pipeline", "result": "failed", "duration_ms": 0}],
                ["fundamental stage timeout"],
            )
        payload, err, cost_ms = self._run_with_retry(
            lambda: self._fundamental_adapter.get_dragon_tiger_flag(stock_code),
            timeout,
            "dragon_tiger",
        )
        if not isinstance(payload, dict):
            return self._build_fundamental_block(
                "failed",
                {},
                [{"provider": "fundamental_pipeline", "result": "failed", "duration_ms": cost_ms}],
                [err or "dragon_tiger failed"],
            )
        return self._build_fundamental_block(
            (payload.get("status") if isinstance(payload.get("status"), str) else "partial"),
            {
                "is_on_list": bool(payload.get("is_on_list", False)),
                "recent_count": int(payload.get("recent_count", 0)),
                "latest_date": payload.get("latest_date"),
            },
            self._normalize_source_chain(
                payload.get("source_chain", []),
                "dragon_tiger",
                str(payload.get("status", "ok")),
                cost_ms,
            ),
            list(payload.get("errors", [])) + ([err] if err else []),
        )

    def get_board_context(self, stock_code: str, budget_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Board rankings block (fail-open)."""
        from src.config import get_config

        config = get_config()
        stock_code = normalize_stock_code(stock_code)
        timeout = float(budget_seconds if budget_seconds is not None else config.fundamental_fetch_timeout_seconds)
        if _market_tag(stock_code) != "cn" or _is_etf_code(stock_code):
            return self._build_fundamental_block(
                "not_supported",
                {},
                [{"provider": "fundamental_pipeline", "result": "not_supported", "duration_ms": 0}],
                ["not supported"],
            )

        if timeout <= 0:
            return self._build_fundamental_block(
                "failed",
                {},
                [{"provider": "fundamental_pipeline", "result": "failed", "duration_ms": 0}],
                ["fundamental stage timeout"],
            )

        def task() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
            return self._get_sector_rankings_with_meta(5)

        rankings, err, cost_ms = self._run_with_retry(task, timeout, "boards")
        if isinstance(rankings, tuple) and len(rankings) == 4:
            top, bottom, chain, chain_error = rankings
            if chain_error and not err:
                err = chain_error
            if not top and not bottom:
                return self._build_fundamental_block(
                    "failed",
                    {},
                    chain if chain else [{"provider": "sector_rankings", "result": "failed", "duration_ms": cost_ms}],
                    [err or "boards empty from all sources"],
                )
            board_status = "ok" if top and bottom else "partial"
            return self._build_fundamental_block(
                board_status,
                {"top": top or [], "bottom": bottom or []},
                chain if chain else self._normalize_source_chain(
                    ["sector_rankings"],
                    "boards",
                    board_status,
                    cost_ms,
                ),
                [err] if err else [],
            )

        return self._build_fundamental_block(
            "failed",
            {},
            [{"provider": "sector_rankings", "result": "failed", "duration_ms": cost_ms}],
            [err or "boards failed"],
        )

    def _get_sector_rankings_with_meta(
            self,
            n: int = 5,
        ) -> Tuple[List[Dict], List[Dict], List[Dict[str, Any]], str]:
            """Get sector rankings with ordered fallback chain metadata."""
            source_chain: List[Dict[str, Any]] = []
            last_error = ""

            for fetcher in self._fetchers:
                if not hasattr(fetcher, 'get_sector_rankings'):
                    continue

                start = time.time()
                try:
                    data = fetcher.get_sector_rankings(n)
                    duration_ms = int((time.time() - start) * 1000)
                    if data and data[0] is not None and data[1] is not None:
                        source_chain.append(
                            {
                                "provider": fetcher.name,
                                "result": "ok",
                                "duration_ms": duration_ms,
                            }
                        )
                        logger.info(f"[{fetcher.name}] get sector rankings success")
                        return data[0], data[1], source_chain, ""

                    last_error = f"{fetcher.name} returned empty"
                    source_chain.append(
                        {
                            "provider": fetcher.name,
                            "result": "empty",
                            "duration_ms": duration_ms,
                            "error": last_error,
                        }
                    )
                except Exception as e:
                    error_type, error_reason = summarize_exception(e)
                    last_error = f"{fetcher.name} ({error_type}) {error_reason}"
                    duration_ms = int((time.time() - start) * 1000)
                    source_chain.append(
                        {
                            "provider": fetcher.name,
                            "result": "failed",
                            "duration_ms": duration_ms,
                            "error": error_reason,
                        }
                    )
                    logger.warning(f"[{fetcher.name}] get sector rankings failed: {error_reason}")

            return [], [], source_chain, last_error

    def get_sector_rankings(self, n: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """Get sector gainers and losers (automatic failover)."""
        top, bottom, _, last_error = self._get_sector_rankings_with_meta(n)
        if top or bottom:
            return top, bottom
        logger.warning(f"[Sector rankings] all sources failed, last error: {last_error}")
        return [], []

    def get_concept_rankings(self, n: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """Get concept/theme gainers and losers (automatic failover)."""
        last_error = ""
        for fetcher in self._fetchers:
            try:
                data = fetcher.get_concept_rankings(n)
                if data and (data[0] or data[1]):
                    logger.info(f"[{fetcher.name}] get concept rankings success")
                    return data[0] or [], data[1] or []
                last_error = f"{fetcher.name} returned empty"
            except Exception as e:
                error_type, error_reason = summarize_exception(e)
                last_error = f"{fetcher.name} ({error_type}) {error_reason}"
                logger.warning(f"[{fetcher.name}] get concept rankings failed: {error_reason}")
        if last_error:
            logger.warning(f"[Concept rankings] all sources failed, last error: {last_error}")
        return [], []

    def get_hot_stocks(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get popular stocks (automatic failover)."""
        last_error = ""
        for fetcher in self._fetchers:
            try:
                data = fetcher.get_hot_stocks(n)
                if data:
                    logger.info(f"[{fetcher.name}] get hot stocks success")
                    return data[:n]
                last_error = f"{fetcher.name} returned empty"
            except Exception as e:
                error_type, error_reason = summarize_exception(e)
                last_error = f"{fetcher.name} ({error_type}) {error_reason}"
                logger.warning(f"[{fetcher.name}] get hot stocks failed: {error_reason}")
        if last_error:
            logger.warning(f"[Hot stocks] all sources failed, last error: {last_error}")
        return []

    def get_limit_up_pool(
        self,
        date: Optional[str] = None,
        n: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get limit-up pool and streak board (automatic failover)."""
        last_error = ""
        for fetcher in self._fetchers:
            try:
                data = fetcher.get_limit_up_pool(date=date, n=n)
                if data:
                    logger.info(f"[{fetcher.name}] get limit-up pool success")
                    return data[:n]
                last_error = f"{fetcher.name} returned empty"
            except Exception as e:
                error_type, error_reason = summarize_exception(e)
                last_error = f"{fetcher.name} ({error_type}) {error_reason}"
                logger.warning(f"[{fetcher.name}] get limit-up pool failed: {error_reason}")
        if last_error:
            logger.warning(f"[Limit-up pool] all sources failed, last error: {last_error}")
        return []
