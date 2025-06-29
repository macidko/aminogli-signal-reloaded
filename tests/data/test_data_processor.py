import pandas as pd
import numpy as np
import pytest
from src.data.data_processor import DataProcessor, DataProcessingException
from src.data.data_fetcher import DataFetcher

def get_sample_df():
    data = {
        'open': [1, 2, np.nan, 4, 5],
        'high': [2, 3, 4, 5, 6],
        'low': [0, 1, 2, 3, 4],
        'close': [1, 2, 3, 4, 5],
        'volume': [10, 20, 30, 40, 50]
    }
    return pd.DataFrame(data)

def test_fillna():
    df = get_sample_df()
    processor = DataProcessor(['fillna'])
    params = {'fillna': {'method': 'bfill'}}
    result = processor.process(df, params)
    assert not result.isnull().values.any()

def test_scale():
    df = get_sample_df().fillna(0)
    processor = DataProcessor(['scale'])
    params = {'scale': {'scaler_type': 'minmax'}}
    result = processor.process(df, params)
    assert np.allclose(result.min(), 0)
    assert np.allclose(result.max(), 1)

def test_add_indicators():
    df = get_sample_df().fillna(0)
    processor = DataProcessor(['add_indicators'])
    params = {'add_indicators': {'indicators': ['rsi', 'ema', 'sma']}}
    result = processor.process(df, params)
    assert 'rsi' in result.columns
    assert 'ema' in result.columns
    assert 'sma' in result.columns

def test_exception():
    df = get_sample_df()
    processor = DataProcessor(['not_a_method'])
    params = {}
    try:
        processor.process(df, params)
    except DataProcessingException as e:
        assert "not_a_method" in str(e)
    else:
        assert False, "Exception not raised!"

def test_integration_with_datafetcher():
    fetcher = DataFetcher('binance')
    df = fetcher.fetch_data('BTC/USDT', '1h', limit=20)
    processor = DataProcessor(['fillna', 'add_indicators', 'scale'])
    params = {
        'fillna': {'method': 'ffill'},
        'add_indicators': {'indicators': ['rsi', 'ema', 'sma']},
        'scale': {'scaler_type': 'minmax'}
    }
    result = processor.process(df, params)
    assert 'rsi' in result.columns
    assert 'ema' in result.columns
    assert 'sma' in result.columns
    # Sadece ana fiyat kolonlarında NaN olmamalı
    for col in ['open', 'high', 'low', 'close', 'volume']:
        assert not result[col].isnull().any(), f"NaN found in {col}"
