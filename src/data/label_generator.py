from abc import ABC, abstractmethod
import pandas as pd

class BaseLabelGenerator(ABC):
    """
    Soyut label (sinyal) oluşturucu. Tüm modeller için ortak arayüz sağlar.
    """
    @abstractmethod
    def generate(self, df: pd.DataFrame, **kwargs) -> pd.Series:
        pass

class PriceDirectionLabelGenerator(BaseLabelGenerator):
    """
    Gelecek n bar sonrası fiyat değişimine göre yukarı/aşağı/yatay sinyal üretir.
    Parametreler:
        - n: Kaç bar sonrası (int)
        - threshold: Yüzdesel değişim eşiği (float, ör: 0.001)
        - target_col: Hangi fiyat kolonu (str, default: 'close')
        - direction_type: 'binary' (sadece yukarı/aşağı), 'multiclass' (yukarı/aşağı/yatay)
    """
    def generate(self, df: pd.DataFrame, n=1, threshold=0.001, target_col='close', direction_type='multiclass') -> pd.Series:
        future_return = (df[target_col].shift(-n) - df[target_col]) / df[target_col]
        if direction_type == 'binary':
            # Sadece yukarı/aşağı
            return (future_return > threshold).astype(int)
        else:
            # Yukarı/aşağı/yatay
            return future_return.apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))

# Genişletilebilirlik için örnek:
# class CustomLabelGenerator(BaseLabelGenerator):
#     def generate(self, df, **kwargs):
#         ... # Özel sinyal mantığı

# Kullanım örneği (üretim ortamında kaldırılmalı):
# label_gen = PriceDirectionLabelGenerator()
# df['signal'] = label_gen.generate(df, n=1, threshold=0.001, target_col='close', direction_type='multiclass')
