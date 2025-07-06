from abc import ABC, abstractmethod
import pandas as pd

class BaseSplitter(ABC):
    @abstractmethod
    def split(self, X: pd.DataFrame, y: pd.Series, test_size=0.3):
        pass

class TimeSeriesSplitter(BaseSplitter):
    """
    Zaman serisi için sıralı eğitim/test ayrımı. Shuffle yok, veri sızıntısı yok.
    """
    def split(self, X: pd.DataFrame, y: pd.Series, test_size=0.3):
        n = len(X)
        split_idx = int(n * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        return X_train, X_test, y_train, y_test

# Genişletilebilirlik için örnek:
# class RollingWindowSplitter(BaseSplitter):
#     def split(self, X, y, window_size, step):
#         ... # Rolling window mantığı

# Kullanım örneği (üretim ortamında kaldırılmalı):
# splitter = TimeSeriesSplitter()
# X_train, X_test, y_train, y_test = splitter.split(X, y, test_size=0.3)
