import ccxt
import pandas as pd

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

    def fetch_data(self, symbol, timeframe, limit=100, since=None):
        """
        Fetch OHLCV data from the exchange.
        :param symbol: Trading pair (e.g., 'BTC/USDT').
        :param timeframe: Timeframe for the data (e.g., '1h', '5m').
        :param limit: Number of data points to fetch.
        :param since: Timestamp in milliseconds to start fetching data from (optional).
        :return: Pandas DataFrame containing OHLCV data.
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            print(f"Fetched {len(df)} rows of data for {symbol} on {timeframe} timeframe.")
            return df
        except Exception as e:
            print(f"Failed to fetch data: {e}")
            raise RuntimeError(f"Failed to fetch data: {e}")

# # Örnek kullanım
# if __name__ == "__main__":
#     # Parametreler
#     exchange_name = 'binance'
#     symbol = 'BTC/USDT'
#     timeframe = '1h'
#     limit = 500

#     # Veri Çekme
#     fetcher = DataFetcher(exchange_name)
#     data = fetcher.fetch_data(symbol, timeframe, limit)
#     print(data.head())