import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from loguru import logger
import numpy as np
from abc import ABC, abstractmethod

class DataProcessingException(Exception):
    pass

class BaseDataProcessor(ABC):
    @abstractmethod
    def process(self, df, params):
        pass

class DataProcessor(BaseDataProcessor):
    def __init__(self, steps):
        self.steps = steps  # Örn: ['fillna', 'scale', 'add_indicators', ...]

    def process(self, df, params):
        for step in self.steps:
            try:
                df = getattr(self, step)(df, **params.get(step, {}))
            except Exception as e:
                logger.exception(f"Hata! Adım: {step}, Parametreler: {params.get(step, {})}")
                raise DataProcessingException(f"Data processing failed at step '{step}' with params {params.get(step, {})}: {e}") from e
        return df

    def fillna(self, df, method='ffill'):
        return df.fillna(method=method)

    def remove_outliers(self, df, z_thresh=3):
        # Z-score ile outlier temizliği (sadece sayısal kolonlar)
        numeric_cols = df.select_dtypes(include=['number']).columns
        z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
        mask = (z_scores < z_thresh).all(axis=1)
        logger.info(f"Outlier temizliği: {len(df) - mask.sum()} satır çıkarıldı.")
        return df[mask]

    def scale(self, df, scaler_type='minmax'):
        scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        return df

    def add_indicators(self, df, indicators=None):
        if indicators is None:
            indicators = []
        for ind in indicators:
            if ind == 'rsi':
                df['rsi'] = self._rsi(df['close'])
            elif ind == 'ema':
                df['ema'] = df['close'].ewm(span=14, adjust=False).mean()
            elif ind == 'sma':
                df['sma'] = df['close'].rolling(window=14).mean()
            elif ind == 'macd':
                df['macd'] = self._macd(df['close'])
            elif ind == 'volatility':
                df['volatility'] = df['close'].rolling(window=14).std()
            elif ind == 'momentum':
                df['momentum'] = df['close'] - df['close'].shift(4)
            elif ind == 'rolling_mean':
                df['rolling_mean'] = df['close'].rolling(window=14).mean()
            # Başka göstergeler kolayca eklenebilir
        return df

    def add_lagged_features(self, df, columns=None, lags=1):
        # Gecikmeli fiyatlar ve göstergeler
        if columns is None:
            columns = ['close']
        for col in columns:
            for lag in range(1, lags+1):
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
        return df

    def _rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _macd(self, series, fast=12, slow=26):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    # Gerekirse encoding ve feature engineering fonksiyonları da eklenir
    def encode_categorical(self, df, columns=None):
        if columns is None:
            columns = []
        for col in columns:
            df[col] = df[col].astype('category').cat.codes
        return df

# Örnek kullanım (üretim ortamında kaldırılmalı):
# processor = DataProcessor(['fillna', 'remove_outliers', 'add_indicators', 'add_lagged_features', 'scale'])
# params = {
#     'fillna': {'method': 'ffill'},
#     'remove_outliers': {'z_thresh': 3},
#     'add_indicators': {'indicators': ['rsi', 'ema', 'macd', 'volatility', 'momentum', 'rolling_mean']},
#     'add_lagged_features': {'columns': ['close', 'volume'], 'lags': 3},
#     'scale': {'scaler_type': 'minmax'}
# }
# df = processor.process(df, params)
