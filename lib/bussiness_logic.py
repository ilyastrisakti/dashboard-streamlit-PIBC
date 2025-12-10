# -*- coding: utf-8 -*-
"""
This module contains the core business logic and data processing functions
that are computationally expensive and should be cached.
"""
import pandas as pd
import streamlit as st
from typing import Tuple
from .constants import *

@st.cache_data
def prepare_geo_data(df_flow: pd.DataFrame, geo_lookup: pd.DataFrame, flow_type: str) -> pd.DataFrame:
    """
    Menggabungkan data flow dengan geo lookup dan melakukan agregasi.
    Fungsi ini di-cache agar tidak perlu hitung ulang setiap render.
    """
    if df_flow is None or df_flow.empty or geo_lookup is None or geo_lookup.empty:
        return pd.DataFrame()

    # Normalisasi nama lokasi
    gl = geo_lookup.copy()
    gl = gl.rename(columns={COL_LOKASI: "lokasi_lookup"}) # Pastikan nama kolom sesuai
    # Jika geo_lookup sudah punya kolom 'lokasi', rename jadi 'lokasi_lookup' agar tidak bentrok
    # Asumsi geo_lookup dari get_geo_lookup() punya kolom: lokasi, lat, lon
    
    gl[COL_LOKASI_NORM] = gl["lokasi_lookup"].astype(str).str.strip().str.lower()

    ff = df_flow.copy()
    if COL_LOKASI not in ff.columns:
        return pd.DataFrame()
    ff[COL_LOKASI_NORM] = ff[COL_LOKASI].astype(str).str.strip().str.lower()

    # Merge & Agregasi
    df_map = ff.merge(gl, on=COL_LOKASI_NORM, how="inner")
    
    if df_map.empty:
        return pd.DataFrame()

    # Groupby untuk mendapatkan total per lokasi unik, memastikan hasilnya DataFrame
    df_agg = df_map.groupby(["lokasi_lookup", "lat", "lon"], as_index=False).agg({flow_type: 'sum'})

    return df_agg

@st.cache_data
def filter_and_aggregate_data(
    df: pd.DataFrame, 
    start_date: pd.Timestamp, 
    end_date: pd.Timestamp, 
    granularity: str
) -> pd.DataFrame:
    """
    Filters the main DataFrame by date and then aggregates it based on the selected granularity.
    This is a cached function to improve performance.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # 1. Filter by date range
    mask = (df[COL_TANGGAL].dt.date >= start_date) & (df[COL_TANGGAL].dt.date <= end_date) # type: ignore
    df_filt = df[mask]

    if df_filt.empty:
        return pd.DataFrame()

    # 2. Aggregate based on granularity
    if granularity == "Harian":
        return df_filt.copy()
    else:
        df_to_resample = df_filt.set_index(COL_TANGGAL)
        resample_rule = 'M' if granularity == "Bulanan" else 'Y'
        agg_rules = {
            COL_STOK: 'mean',
            COL_MASUK: 'sum',
            COL_KELUAR: 'sum',
            COL_NERACA: 'sum'
        }
        df_agg = df_to_resample.resample(resample_rule).agg(agg_rules).reset_index()
        return df_agg