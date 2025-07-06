import logging
import os
import pandas as pd
from src.data.data_fetcher import DataFetcher
from src.data.data_processor import DataProcessor
from src.data.label_generator import PriceDirectionLabelGenerator
from src.pipelines.splitter import TimeSeriesSplitter
from src.models.model_factory import get_model
from src.pipelines.signal_writer import TimeSeriesSignalWriter
from src.evaluation.metrics import classification_metrics
from src.evaluation.backtest import simple_backtest

# Log dosyası ayarları
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/pipeline.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


def run_full_pipeline(config):
    try:
        logging.info('Veri çekiliyor...')
        fetcher = DataFetcher(config['exchange'])
        df = fetcher.fetch_data(config['symbol'], config['timeframe'], config['limit'], since=config.get('since'))
        logging.info(f'Veri çekildi: {df.shape}')
    except Exception as e:
        logging.error(f'Veri çekme hatası: {e}')
        return

    try:
        logging.info('Veri işleniyor...')
        processor = DataProcessor(config['process_steps'])
        df_processed = processor.process(df, config['process_params'])
        logging.info(f'İşlenen veri: {df_processed.shape}')
    except Exception as e:
        logging.error(f'Veri işleme hatası: {e}')
        return

    try:
        logging.info('Label/sinyal üretiliyor...')
        labeler = PriceDirectionLabelGenerator()
        df_labeled = df_processed.copy()
        df_labeled['label'] = labeler.generate(
            df_processed,
            n=config['label_n'],
            threshold=config['label_threshold'],
            target_col=config.get('label_target_col', 'close'),
            direction_type=config.get('label_direction_type', 'multiclass')
        )
        logging.info('Label üretildi.')
    except Exception as e:
        logging.error(f'Label/sinyal üretim hatası: {e}')
        return

    try:
        logging.info('Eğitim/test ayrımı yapılıyor...')
        X = df_labeled.drop(columns=['label'])
        y = df_labeled['label']
        splitter = TimeSeriesSplitter()
        X_train, X_test, y_train, y_test = splitter.split(X, y, test_size=config['test_size'])

        # Model eğitimine uygun: datetime sütunlarını çıkar
        def drop_datetime_columns(df):
            return df.select_dtypes(exclude=['datetime', 'datetime64[ns]', 'datetime64'])
        X_train = drop_datetime_columns(X_train)
        X_test = drop_datetime_columns(X_test)

        logging.info(f'Train: {X_train.shape}, Test: {X_test.shape}')
    except Exception as e:
        logging.error(f'Split hatası: {e}')
        return

    try:
        logging.info('Model eğitiliyor...')
        model = get_model(config['model_name'], task='classification', **config['model_params'])
        model.fit(X_train, y_train)
        logging.info('Model eğitildi.')
    except Exception as e:
        logging.error(f'Model eğitimi hatası: {e}')
        return

    try:
        logging.info('Tahmin ve sinyal kaydı...')
        preds = model.predict(X_test)
        # Zaman damgası oluştur
        from datetime import datetime
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Sinyal kaydı için DataFrame oluştur
        signal_df = X_test.copy()
        signal_df['predicted_signal'] = preds
        signal_df['true_signal'] = y_test.values  # Gerçek label'ı da ekle
        signal_writer = TimeSeriesSignalWriter()
        signal_writer.save(signal_df, model_name=config['model_name'], run_id=run_id)
        logging.info('Sinyaller kaydedildi.')

        # Dosya isimleri için run_id kullan
        metrics_path = os.path.join('logs', f"metrics_{config['model_name']}_{run_id}.csv")
        backtest_path = os.path.join('logs', f"backtest_{config['model_name']}_{run_id}.csv")
    except Exception as e:
        logging.error(f'Sinyal kaydı hatası: {e}')
        return

    try:
        logging.info('Değerlendirme ve backtest...')
        metrics = classification_metrics(y_test, preds)
        metrics_simple = {k: v for k, v in metrics.items() if k in ['accuracy', 'f1', 'precision', 'recall']}
        pd.DataFrame([metrics_simple]).to_csv(metrics_path, index=False)
        backtest_df = signal_df.copy()
        backtest_df['close'] = X_test['close'] if 'close' in X_test.columns else df_labeled.loc[X_test.index, 'close']
        backtest_result = simple_backtest(backtest_df, signal_col='predicted_signal', price_col='close')
        backtest_result.to_csv(backtest_path, index=False)
        logging.info(f'Metrikler ve backtest kaydedildi: {metrics_path}, {backtest_path}')
    except Exception as e:
        logging.error(f'Değerlendirme/backtest hatası: {e}')
        return


if __name__ == '__main__':
    # Örnek config, ileride yaml/json'dan okunabilir
    config = {
        'exchange': 'binance',
        'symbol': 'BTC/USDT',
        'timeframe': '1h',
        'limit': 1000,
        'since': None,
        'process_steps': ['fillna', 'add_indicators', 'scale'],
        'process_params': {
            'fillna': {'method': 'ffill'},
            'add_indicators': {'indicators': ['rsi', 'ema', 'sma']},
            'scale': {'scaler_type': 'minmax'}
        },
        'label_threshold': 0.01,
        'label_n': 5,
        'label_target_col': 'close',
        'label_direction_type': 'multiclass',
        'test_size': 0.2,
        'model_name': 'random_forest',
        'model_params': {},
    }
    run_full_pipeline(config)
