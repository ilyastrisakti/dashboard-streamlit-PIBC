# lib/forecast.py
import pandas as pd
import logging
from typing import Optional, Tuple, Any
import streamlit as st

# Import library forecasting
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Import konstanta internal
from .constants import FC_COL_DS, FC_COL_Y, COL_TANGGAL, COL_STOK

logger = logging.getLogger(__name__)

@st.cache_data(show_spinner=True)
def run_prophet_forecast(df: pd.DataFrame, periods: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Menjalankan algoritma Prophet untuk peramalan stok.
    Mengembalikan tuple: (DataFrame Historis, DataFrame Hasil Forecast)
    """
    # 1. Persiapan data (Rename kolom sesuai standar Prophet: ds, y)
    df_fc = df[[COL_TANGGAL, COL_STOK]].rename(columns={COL_TANGGAL: FC_COL_DS, COL_STOK: FC_COL_Y})
    
    # 2. Inisialisasi dan Fitting Model
    m = Prophet()
    m.fit(df_fc)
    
    # 3. Prediksi
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    
    return df_fc, forecast

@st.cache_data(show_spinner=True)
def run_holtwinters_forecast(df: pd.DataFrame, periods: int) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Menjalankan algoritma Holt-Winters (Exponential Smoothing).
    Mengembalikan tuple: (DataFrame Historis, DataFrame Hasil Forecast dengan kolom 'ds' dan 'yhat')
    """
    # 1. Persiapan data
    df_fc = df[[COL_TANGGAL, COL_STOK]].rename(columns={COL_TANGGAL: FC_COL_DS, COL_STOK: FC_COL_Y})
    
    try:
        # Tentukan seasonal periods (minimal 7 hari, atau setengah panjang data jika sangat sedikit)
        season = 7 if len(df_fc) >= 14 else (len(df_fc) // 2 if len(df_fc) > 2 else None)
        
        # 2. Fitting Model
        model = ExponentialSmoothing(
            df_fc[FC_COL_Y], 
            seasonal='add' if season else None, 
            seasonal_periods=season
        ).fit()
        
        # 3. Prediksi
        pred_values = model.forecast(periods)
        
        # 4. Buat DataFrame hasil prediksi agar formatnya seragam
        last_date = df_fc[FC_COL_DS].iloc[-1]
        date_range = pd.date_range(last_date, periods=periods+1)[1:]
        
        df_pred = pd.DataFrame({
            FC_COL_DS: date_range,
            'yhat': pred_values
        })
        
        return df_fc, df_pred
        
    except Exception as e:
        logger.error(f"Holt-Winters forecast failed: {e}")
        return df_fc, None