import streamlit as st
import os
from analyze.signal_analysis import signal_distribution_analysis, financial_metrics_analysis, mae_analysis, signal_run_length_analysis, signal_lead_lag_analysis

st.title('Sinyal Analiz Paneli')

# 1. Dosya seçimi
outputs_dir = 'outputs'
files = [f for f in os.listdir(outputs_dir) if f.endswith('.csv') and 'signals_' in f]
backtest_files = [f for f in os.listdir('logs') if f.startswith('backtest_') and f.endswith('.csv')]

# 2. Analiz seçimi
analizler = {
    'Sinyal Dağılımı ve Confusion Matrix': signal_distribution_analysis,
    'MAE Analizi': mae_analysis,
    'Sinyal Ardışıklık (Run-Length) Analizi': signal_run_length_analysis,
    'Sinyal Gecikme/İleri Kayma Analizi': signal_lead_lag_analysis,
}
selected_file = st.selectbox('Analiz etmek istediğiniz sinyal dosyasını seçin:', files)
selected_analysis = st.selectbox('Analiz türünü seçin:', list(analizler.keys()) + ['Finansal Metrikler (Backtest)'])

# 3. Analizi çalıştır
if selected_analysis == 'Finansal Metrikler (Backtest)':
    selected_backtest = st.selectbox('Backtest dosyasını seçin:', backtest_files)
    if st.button('Analizi Başlat'):
        st.write(f"Seçilen dosya: {selected_backtest}")
        financial_metrics_analysis(os.path.join('logs', selected_backtest))
else:
    if st.button('Analizi Başlat'):
        st.write(f"Seçilen dosya: {selected_file}")
        analizler[selected_analysis](os.path.join(outputs_dir, selected_file))
