import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from loguru import logger

class DataProcessingException(Exception):
    pass

class DataProcessor:
    def __init__(self, steps):
        self.steps = steps  # Örn: ['fillna', 'scale', 'add_indicators']

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
            # Başka göstergeler kolayca eklenebilir
        return df

    def _rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # Gerekirse encoding ve feature engineering fonksiyonları da eklenir
