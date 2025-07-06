import ccxt
import pandas as pd
import numpy as np
from loguru import logger
import os
from abc import ABC, abstractmethod

class DataFetcherException(Exception):
    pass

class BaseDataFetcher(ABC):
    @abstractmethod
    def fetch_data(self, *args, **kwargs):
        pass

class DataFetcher(BaseDataFetcher):
    def __init__(self, exchange_name):
        """
        Initialize the DataFetcher with the given exchange name.
        :param exchange_name: Name of the exchange (e.g., 'binance', 'kraken').
        """
        try:
            self.exchange = getattr(ccxt, exchange_name)()
            logger.info(f"Exchange '{exchange_name}' initialized successfully.")
        except AttributeError:
            raise ValueError(f"Exchange '{exchange_name}' is not supported by CCXT.")

    def fetch_data(self, symbol, timeframe, limit=100, since=None, columns=None, as_type='df', save_path=None):
        """
        Fetch OHLCV data from the exchange.
        :param symbol: Trading pair (e.g., 'BTC/USDT').
        :param timeframe: Timeframe for the data (e.g., '1h', '5m').
        :param limit: Number of data points to fetch.
        :param since: Timestamp in milliseconds to start fetching data from (optional).
        :param columns: List of columns to return (optional).
        :param as_type: Output format: 'df' (default), 'np' (numpy array), 'dict' (list of dicts).
        :param save_path: Optional path to save the fetched data (csv or parquet).
        :return: Data in the requested format.
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            if columns:
                df = df[columns]
            logger.info(f"Fetched {len(df)} rows of data for {symbol} on {timeframe} timeframe.")
            if save_path:
                ext = os.path.splitext(save_path)[1].lower()
                if ext == '.csv':
                    df.to_csv(save_path, index=False)
                    logger.info(f"Data saved to {save_path} (CSV format).")
                elif ext == '.parquet':
                    df.to_parquet(save_path, index=False)
                    logger.info(f"Data saved to {save_path} (Parquet format).")
                else:
                    logger.warning(f"Unknown file extension for save_path: {save_path}")
            if as_type == 'np':
                return df.values
            elif as_type == 'dict':
                return df.to_dict('records')
            return df
        except Exception as e:
            logger.exception(f"Data fetch failed! Symbol: {symbol}, Timeframe: {timeframe}, Limit: {limit}, Since: {since}, Columns: {columns}, as_type: {as_type}")
            raise DataFetcherException(f"Failed to fetch data for {symbol} on {timeframe}: {e}") from e

    def fetch_latest(self, symbol, timeframe, last_timestamp, columns=None, as_type='df', save_path=None):
        """
        Fetch new data since the last timestamp (for incremental updates).
        :param last_timestamp: Last known timestamp in pandas.Timestamp or int (ms).
        """
        if isinstance(last_timestamp, pd.Timestamp):
            since = int(last_timestamp.value // 10**6)
        else:
            since = int(last_timestamp)
        return self.fetch_data(symbol, timeframe, since=since, columns=columns, as_type=as_type, save_path=save_path)

# Example usage (to be removed in production):
# fetcher = DataFetcher('binance')
# df = fetcher.fetch_data('BTC/USDT', '1h', limit=200, save_path='btc_1h.csv')
# df_new = fetcher.fetch_latest('BTC/USDT', '1h', last_timestamp=df['timestamp'].iloc[-1])

