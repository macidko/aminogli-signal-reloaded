import ccxt
import pandas as pd
import numpy as np
from loguru import logger

class DataFetcherException(Exception):
    pass

class DataFetcher:
    def __init__(self, exchange_name):
        """
        Initialize the DataFetcher with the given exchange name.
        :param exchange_name: Name of the exchange (e.g., 'binance', 'kraken').
        """
        try:
            self.exchange = getattr(ccxt, exchange_name)()
            print(f"Exchange '{exchange_name}' initialized successfully.")
        except AttributeError:
            raise ValueError(f"Exchange '{exchange_name}' is not supported by CCXT.")

    def fetch_data(self, symbol, timeframe, limit=100, since=None, columns=None, as_type='df'):
        """
        Fetch OHLCV data from the exchange.
        :param symbol: Trading pair (e.g., 'BTC/USDT').
        :param timeframe: Timeframe for the data (e.g., '1h', '5m').
        :param limit: Number of data points to fetch.
        :param since: Timestamp in milliseconds to start fetching data from (optional).
        :param columns: List of columns to return (optional).
        :param as_type: Output format: 'df' (default), 'np' (numpy array), 'dict' (list of dicts).
        :return: Data in the requested format.
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            if columns:
                df = df[columns]
            print(f"Fetched {len(df)} rows of data for {symbol} on {timeframe} timeframe.")
            if as_type == 'np':
                return df.values
            elif as_type == 'dict':
                return df.to_dict('records')
            return df
        except Exception as e:
            logger.exception(f"Data fetch failed! Symbol: {symbol}, Timeframe: {timeframe}, Limit: {limit}, Since: {since}, Columns: {columns}, as_type: {as_type}")
            raise DataFetcherException(f"Failed to fetch data for {symbol} on {timeframe}: {e}") from e

