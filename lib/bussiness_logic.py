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
    # 1. Filter by date range
    mask = (df[COL_TANGGAL].dt.date >= start_date) & (df[COL_TANGGAL].dt.date <= end_date)
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