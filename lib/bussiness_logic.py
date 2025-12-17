# lib/bussiness_logic.py
# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st
from typing import Tuple
from .constants import *

@st.cache_data
def prepare_geo_data(df_flow: pd.DataFrame, geo_lookup: pd.DataFrame, flow_type: str) -> pd.DataFrame:
    """
    Menggabungkan data flow dengan geo lookup.
    PERBAIKAN: Menggunakan LEFT JOIN agar lokasi yang tidak punya koordinat
    TIDAK HILANG, melainkan masuk ke kategori 'Lainnya'/'Unknown'.
    """
    if df_flow is None or df_flow.empty or geo_lookup is None or geo_lookup.empty:
        return pd.DataFrame()

    # 1. Normalisasi Lookup
    gl = geo_lookup.copy()
    gl = gl.rename(columns={COL_LOKASI: "lokasi_lookup"}) 
    gl[COL_LOKASI_NORM] = gl["lokasi_lookup"].astype(str).str.strip().str.lower()

    # 2. Normalisasi Data Flow
    ff = df_flow.copy()
    if COL_LOKASI not in ff.columns:
        return pd.DataFrame()
    ff[COL_LOKASI_NORM] = ff[COL_LOKASI].astype(str).str.strip().str.lower()

    # 3. Aggregasi Awal (Peringan Beban)
    ff_agg = ff.groupby(COL_LOKASI_NORM, as_index=False)[flow_type].sum()

    # 4. MERGE DENGAN STRATEGI 'LEFT' (SOLUSI DATA HILANG)
    # Gunakan left join agar data transaksi (ff_agg) tetap bertahan walau tidak ada di gl
    df_map = ff_agg.merge(gl, on=COL_LOKASI_NORM, how="left")
    
    # 5. Handling Lokasi Tidak Dikenal (Missing Coordinates)
    # Jika lat/lon kosong (karena kota tidak ada di lookup), pakai koordinat 'Lainnya'
    # Ambil koordinat default dari baris 'Lainnya' atau 'Luar Jawa' di geo_lookup
    default_lat = -5.000  # Default Laut Jawa
    default_lon = 109.000
    
    # Coba cari koordinat 'Lainnya' dari lookup jika ada
    unknown_ref = gl[gl[COL_LOKASI_NORM].isin(['lainnya', 'unknown', 'luar jawa', 'luar pulau jawa'])]
    if not unknown_ref.empty:
        default_lat = unknown_ref.iloc[0]['lat']
        default_lon = unknown_ref.iloc[0]['lon']

    # Isi yang NaN dengan default
    df_map['lat'] = df_map['lat'].fillna(default_lat)
    df_map['lon'] = df_map['lon'].fillna(default_lon)
    df_map['lokasi_lookup'] = df_map['lokasi_lookup'].fillna(df_map[COL_LOKASI_NORM].str.title()) # Pakai nama asli jika lookup kosong

    # 6. Groupby Akhir
    df_agg_final = df_map.groupby(["lokasi_lookup", "lat", "lon"], as_index=False).agg({flow_type: 'sum'})

    return df_agg_final

@st.cache_data
def filter_and_aggregate_data(
    df: pd.DataFrame, 
    start_date: pd.Timestamp, 
    end_date: pd.Timestamp, 
    granularity: str
) -> pd.DataFrame:
    """
    Filter dan Aggregasi dengan Resampling untuk mengatasi grafik 'Erratic' (Zig-Zag).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # 1. Sorting & Filtering
    df = df.sort_values(COL_TANGGAL)
    mask = (df[COL_TANGGAL].dt.date >= start_date) & (df[COL_TANGGAL].dt.date <= end_date) # type: ignore
    df_filt = df[mask].copy()

    if df_filt.empty:
        return pd.DataFrame()

    # Set index untuk resampling
    df_filt = df_filt.set_index(COL_TANGGAL)

    # 2. Logika Resampling (Penambal Data Bolong)
    if granularity == "Harian":
        # Pakai 'D' (Daily) untuk mengisi tanggal yang loncat
        df_resampled = df_filt.resample('D').agg({
            COL_STOK: 'mean',  # Stok dirata-rata jika ada duplikat
            COL_MASUK: 'sum',  # Transaksi dijumlah
            COL_KELUAR: 'sum'
        })
        
        # FFILL: Stok hari ini = Stok kemarin (jika hari ini kosong/libur)
        # Ini membuat grafik stok "Mendatar" bukannya jatuh ke nol
        df_resampled[COL_STOK] = df_resampled[COL_STOK].ffill()
        
        # Transaksi diisi 0 jika kosong
        df_resampled[COL_MASUK] = df_resampled[COL_MASUK].fillna(0)
        df_resampled[COL_KELUAR] = df_resampled[COL_KELUAR].fillna(0)
        
        # Hitung Neraca
        df_resampled[COL_NERACA] = df_resampled[COL_MASUK] - df_resampled[COL_KELUAR]
        
        return df_resampled.reset_index()
        
    else:
        # Logika Bulanan/Tahunan
        resample_rule = 'M' if granularity == "Bulanan" else 'Y'
        agg_rules = {
            COL_STOK: 'mean',
            COL_MASUK: 'sum',
            COL_KELUAR: 'sum'
        }
        df_agg = df_filt.resample(resample_rule).agg(agg_rules)
        df_agg[COL_NERACA] = df_agg[COL_MASUK] - df_agg[COL_KELUAR]
        return df_agg.reset_index()