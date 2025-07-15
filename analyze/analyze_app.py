import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))

import streamlit as st
import os
from analyze.signal_analysis import signal_distribution_analysis, financial_metrics_analysis, mae_analysis, signal_run_length_analysis, signal_lead_lag_analysis

st.title('Sinyal Analiz Paneli')

# 1. Run klasörü seçimi
outputs_dir = 'outputs'
run_dirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
selected_run = st.selectbox('Analiz etmek istediğiniz run klasörünü seçin:', run_dirs)
run_path = os.path.join(outputs_dir, selected_run)

# 2. Dosya seçimi (run klasörü içinden)
files = [f for f in os.listdir(run_path) if f.endswith('.csv')]
signal_files = [f for f in files if 'signals' in f or f == 'signals.csv']
backtest_files = [f for f in files if 'backtest' in f or f == 'backtest.csv']

# 3. Analiz seçimi
analizler = {
    'Sinyal Dağılımı ve Confusion Matrix': signal_distribution_analysis,
    'MAE Analizi': mae_analysis,
    'Sinyal Ardışıklık (Run-Length) Analizi': signal_run_length_analysis,
    'Sinyal Gecikme/İleri Kayma Analizi': signal_lead_lag_analysis,
}
selected_signal_file = st.selectbox('Sinyal dosyasını seçin:', signal_files)
selected_analysis = st.selectbox('Analiz türünü seçin:', list(analizler.keys()) + ['Finansal Metrikler (Backtest)'])

# 4. Analizi çalıştır
if selected_analysis == 'Finansal Metrikler (Backtest)':
    selected_backtest = st.selectbox('Backtest dosyasını seçin:', backtest_files)
    if st.button('Analizi Başlat'):
        st.write(f"Seçilen dosya: {selected_backtest}")
        financial_metrics_analysis(os.path.join(run_path, selected_backtest))
else:
    if st.button('Analizi Başlat'):
        st.write(f"Seçilen dosya: {selected_signal_file}")
        analizler[selected_analysis](os.path.join(run_path, selected_signal_file))
