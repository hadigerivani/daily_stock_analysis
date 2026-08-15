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
    with self._fetchers_lock:
        # Import های محلی برای شکستن چرخه واردات
        from .efinance_fetcher import EfinanceFetcher
        from .tencent_fetcher import TencentFetcher
        from .akshare_fetcher import AkshareFetcher
        from .pytdx_fetcher import PytdxFetcher
        from .baostock_fetcher import BaostockFetcher
        from .yfinance_fetcher import YfinanceFetcher

        self._fetchers = [
            EfinanceFetcher(),
            TencentFetcher(),
            AkshareFetcher(),
            PytdxFetcher(),
            BaostockFetcher(),
            YfinanceFetcher(),
        ]

        # اضافه کردن IranFetcher برای بورس تهران
        try:
            from .iran_fetcher import IranFetcher
            self._fetchers.append(IranFetcher())
        except ImportError:
            pass

        # داده‌گیرهای اختیاری (همه با try/except)
        try:
            from .tushare_fetcher import TushareFetcher
            self._fetchers.append(TushareFetcher())
        except ImportError:
            pass

        try:
            from .longbridge_fetcher import LongbridgeFetcher
            self._fetchers.append(LongbridgeFetcher())
        except ImportError:
            pass

        try:
            from .finnhub_fetcher import FinnhubFetcher
            self._fetchers.append(FinnhubFetcher())
        except ImportError:
            pass

        try:
            from .alphavantage_fetcher import AlphaVantageFetcher
            self._fetchers.append(AlphaVantageFetcher())
        except ImportError:
            pass

        # مرتب‌سازی بر اساس اولویت
        self._fetchers.sort(key=lambda f: f.priority)
        self._refresh_fetcher_indexes_locked()

        # لاگ اطلاعات
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

# ... ادامه بقیه متدها (prefetch_realtime_quotes, get_realtime_quote, ...)
# تا انتهای فایل
