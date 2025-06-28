

import argparse
import os
import pandas as pd
from src.data.data_fetcher import DataFetcher

def main():
    parser = argparse.ArgumentParser(description="Aminogli Signal Reloaded - Main Entry Point")
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name (default: binance)')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair symbol (default: BTC/USDT)')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe (default: 1h)')
    parser.add_argument('--limit', type=int, default=100, help='Number of data points to fetch (default: 100)')
    parser.add_argument('--since', type=str, default=None, help='Start date/time (ISO format or timestamp in ms)')
    parser.add_argument('--save', action='store_true', help='Save fetched data to file')
    args = parser.parse_args()


    # since parametresini işle
    since = None
    if args.since:
        try:
            # ISO formatı veya timestamp desteği
            if args.since.isdigit():
                since = int(args.since)
            else:
                since = int(pd.Timestamp(args.since).timestamp() * 1000)
        except Exception as e:
            print(f"Invalid --since value: {e}")
            return

    # Veri çekici başlat
    fetcher = DataFetcher(args.exchange)
    df = fetcher.fetch_data(args.symbol, args.timeframe, args.limit, since=since)
    print(df.head())

    # Kaydetme opsiyonu
    if args.save:
        symbol_safe = args.symbol.replace('/', '_')
        out_dir = os.path.join('data', args.exchange, symbol_safe)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"{args.timeframe}_ohlcv.csv"
        out_path = os.path.join(out_dir, filename)
        df.to_csv(out_path, index=False)
        print(f"Data saved to {out_path}")

if __name__ == "__main__":
    main()
