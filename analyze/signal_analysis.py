import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error
import seaborn as sns
import streamlit as st

# Sinyal dağılımı ve confusion matrix

def signal_distribution_analysis(signals_path):
    df = pd.read_csv(signals_path)
    if 'true_signal' not in df.columns or 'predicted_signal' not in df.columns:
        st.error('Seçilen dosyada true_signal veya predicted_signal kolonu yok!')
        return
    st.subheader('Gerçek sinyal dağılımı:')
    st.write(df['true_signal'].value_counts())
    st.subheader('Model tahmini dağılımı:')
    st.write(df['predicted_signal'].value_counts())
    st.subheader('Classification Report:')
    report = classification_report(df['true_signal'], df['predicted_signal'], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    cm = confusion_matrix(df['true_signal'], df['predicted_signal'], labels=[-1,0,1])
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[-1,0,1], yticklabels=[-1,0,1], ax=ax)
    ax.set_xlabel('Model Tahmini')
    ax.set_ylabel('Gerçek Sinyal')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)
    fig2, ax2 = plt.subplots(figsize=(15,4))
    ax2.plot(df['true_signal'].values, label='Gerçek Sinyal', alpha=0.7)
    ax2.plot(df['predicted_signal'].values, label='Model Sinyali', alpha=0.7)
    ax2.set_title('Gerçek vs Model Sinyali (Zaman Serisi)')
    ax2.set_xlabel('Zaman (index)')
    ax2.set_ylabel('Sinyal')
    ax2.legend()
    st.pyplot(fig2)

# Finansal metrikler ve trade simülasyonu (backtest.csv)
def financial_metrics_analysis(backtest_path):
    df = pd.read_csv(backtest_path)
    st.subheader('Backtest Sonuçları (Kümülatif Getiri ve Strateji Performansı)')
    if 'cum_strategy_return' in df.columns:
        st.line_chart(df['cum_strategy_return'])
        st.write(f"Son kümülatif strateji getirisi: {df['cum_strategy_return'].iloc[-1]:.4f}")
    if 'strategy_return' in df.columns:
        st.write(f"Ortalama trade getirisi: {df['strategy_return'].mean():.4f}")
    if 'return' in df.columns:
        st.write(f"Ortalama piyasa getirisi: {df['return'].mean():.4f}")
    # Sharpe oranı örneği
    if 'strategy_return' in df.columns:
        sharpe = df['strategy_return'].mean() / (df['strategy_return'].std() + 1e-8)
        st.write(f"Sharpe Oranı (basit): {sharpe:.4f}")
    st.line_chart(df['close'])
    st.caption('Fiyat serisi (close)')

# MAE ve regresyon metrikleri (sinyal dosyası)
def mae_analysis(signals_path):
    df = pd.read_csv(signals_path)
    if 'true_signal' not in df.columns or 'predicted_signal' not in df.columns:
        st.error('Seçilen dosyada true_signal veya predicted_signal kolonu yok!')
        return
    st.subheader('MAE (Mean Absolute Error)')
    mae = mean_absolute_error(df['true_signal'], df['predicted_signal'])
    st.write(f"MAE: {mae:.4f}")
    st.line_chart(abs(df['true_signal'] - df['predicted_signal']))
    st.caption('Tahmin hatasının zaman içindeki değişimi')

def signal_run_length_analysis(signals_path, signal_col='predicted_signal'):
    """
    Her sinyal tipi için ardışık tekrar sayılarını (run-length) analiz eder ve histogram verisi döndürür.
    """
    import numpy as np
    df = pd.read_csv(signals_path)
    if signal_col not in df.columns:
        st.error(f'Seçilen dosyada {signal_col} kolonu yok!')
        return
    signals = df[signal_col].values
    # Run-length encoding
    run_lengths = []
    run_values = []
    prev = signals[0]
    count = 1
    for s in signals[1:]:
        if s == prev:
            count += 1
        else:
            run_lengths.append(count)
            run_values.append(prev)
            prev = s
            count = 1
    run_lengths.append(count)
    run_values.append(prev)
    run_df = pd.DataFrame({'signal': run_values, 'length': run_lengths})
    st.subheader('Sinyal Ardışıklık (Run-Length) Analizi')
    for sig in sorted(run_df['signal'].unique()):
        lengths = run_df[run_df['signal'] == sig]['length']
        st.write(f'Sinyal: {sig} - Ardışık tekrar histogramı:')
        fig, ax = plt.subplots(figsize=(8,3))
        ax.hist(lengths, bins=range(1, lengths.max()+2), alpha=0.7, color='C0', rwidth=0.9)
        ax.set_title(f'Sinyal {sig} için ardışık tekrar (run-length) dağılımı')
        ax.set_xlabel('Ardışık tekrar sayısı (bar)')
        ax.set_ylabel('Frekans')
        st.pyplot(fig)
        # Metinsel özet
        st.write(f"Ortalama ardışık tekrar: {lengths.mean():.2f} bar, Medyan: {lengths.median():.2f} bar, Maksimum: {lengths.max()} bar, Minimum: {lengths.min()} bar")
        st.write(f"En sık görülen ardışık tekrar: {lengths.mode().values[0]} bar")
    return run_df


def signal_lead_lag_analysis(signals_path, pred_col='predicted_signal', true_col='true_signal', max_lag=10):
    """
    Model sinyalinin gerçek sinyale göre kaç bar önce/sonra geldiğini analiz eder.
    Pozitif değer: model sinyali geç kalmış, negatif: erken vermiş.
    """
    import numpy as np
    df = pd.read_csv(signals_path)
    if pred_col not in df.columns or true_col not in df.columns:
        st.error(f'Seçilen dosyada {pred_col} veya {true_col} kolonu yok!')
        return
    pred = df[pred_col].values
    true = df[true_col].values
    lead_lag_list = []
    for i in range(len(true)):
        if true[i] == pred[i]:
            continue  # Doğru tahmin, gecikme yok
        # Sinyal değişim noktası arıyoruz
        if i == 0 or true[i] == true[i-1]:
            continue
        # Bu noktada modelin aynı sinyali kaç bar önce/sonra verdiğini bul
        found = False
        for lag in range(-max_lag, max_lag+1):
            j = i + lag
            if j < 0 or j >= len(pred):
                continue
            if pred[j] == true[i] and (j == 0 or pred[j-1] != true[i]):
                lead_lag_list.append(lag)
                found = True
                break
        if not found:
            lead_lag_list.append(np.nan)
    st.subheader('Sinyal Gecikme/İleri Kayma (Lead/Lag) Analizi')
    lead_lag_arr = np.array([x for x in lead_lag_list if not np.isnan(x)])
    if len(lead_lag_arr) == 0:
        st.info('Gecikme/ileri kayma tespit edilemedi.')
        return
    fig, ax = plt.subplots(figsize=(8,3))
    ax.hist(lead_lag_arr, bins=range(-max_lag, max_lag+2), color='C1', alpha=0.7, rwidth=0.9)
    ax.set_title('Model Sinyalinin Gecikme/İleri Kayma Dağılımı')
    ax.set_xlabel('Gecikme (bar): Pozitif=geç, Negatif=erken')
    ax.set_ylabel('Frekans')
    st.pyplot(fig)
    st.write(f'Ortalama gecikme: {lead_lag_arr.mean():.2f} bar')
    st.write(f'Medyan gecikme: {np.median(lead_lag_arr):.2f} bar')
    # Metinsel özet
    st.write(f"Gecikme/ileri kayma aralığı: {lead_lag_arr.min()} ile {lead_lag_arr.max()} bar arası")
    st.write(f"En sık görülen gecikme/ileri kayma: {pd.Series(lead_lag_arr).mode().values[0]} bar")
    st.write(f"Toplam analiz edilen sinyal değişim noktası: {len(lead_lag_arr)}")
    return lead_lag_arr
