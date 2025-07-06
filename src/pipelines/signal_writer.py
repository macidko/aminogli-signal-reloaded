import os
import pandas as pd
from datetime import datetime

class BaseSignalWriter:
    """
    Soyut sinyal kaydedici. Tüm modeller ve pipeline'lar için ortak arayüz sağlar.
    """
    def save(self, signal_df: pd.DataFrame, model_name: str, run_id: str = None, output_dir: str = 'outputs'):
        raise NotImplementedError

class TimeSeriesSignalWriter(BaseSignalWriter):
    """
    Sinyalleri model adı, zaman damgası ve run_id ile ayrıştırılmış şekilde kaydeder.
    Her model ve her çalıştırma için ayrı dosya oluşturur.
    """
    def save(self, signal_df: pd.DataFrame, model_name: str, run_id: str = None, output_dir: str = 'outputs'):
        os.makedirs(output_dir, exist_ok=True)
        if run_id is None:
            run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"signals_{model_name}_{run_id}.csv"
        path = os.path.join(output_dir, filename)
        signal_df.to_csv(path, index=False)
        return path

# Kullanım örneği (üretim ortamında kaldırılmalı):
# writer = TimeSeriesSignalWriter()
# path = writer.save(signal_df, model_name='random_forest', run_id='20250706_123456')
# print(f"Sinyaller kaydedildi: {path}")
