import pandas as pd
import numpy as np
from typing import Optional
from scipy.stats import linregress
import logging
from .constants import COL_TANGGAL, COL_STOK

def _clean_colname(c: Optional[str]) -> str:
    if c is None:
        return ""
    return str(c).replace("\n", " ").strip()

def price_df_with_tanggal(df_price: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df_price is None:
        return None
    p = df_price.copy()
    p.columns = [_clean_colname(c) for c in p.columns]
    if isinstance(p.index, pd.DatetimeIndex) or (p.index.name and str(p.index.name).lower() == COL_TANGGAL):
        p = p.reset_index()
        if 'index' in p.columns:
            p = p.rename(columns={'index':COL_TANGGAL})
    else:
        date_col = next((c for c in p.columns if c.lower() in ("date", COL_TANGGAL, "tgl", "hari")), None)
        if date_col:
            p = p.rename(columns={date_col: COL_TANGGAL})
        else:
            p[COL_TANGGAL] = pd.to_datetime(p.index, errors="coerce")
    p[COL_TANGGAL] = pd.to_datetime(p[COL_TANGGAL], errors="coerce")
    return p

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def calculate_regression(df_stock: pd.DataFrame, df_price: pd.DataFrame, rice_type: str):
    """
    Calculates linear regression between stock and price for a specific rice type.
    Returns a dictionary with regression stats and the merged dataframe.
    """
    logger = logging.getLogger(__name__)
    if df_stock is None or df_price is None: return None

    df_p = price_df_with_tanggal(df_price)
    if df_p is None or rice_type not in df_p.columns:
        logger.warning(f"calculate_regression: price data missing or rice_type '{rice_type}' not found.")
        return None

    if COL_TANGGAL not in df_stock.columns:
        df_stock = df_stock.reset_index()
        if df_stock.columns[0].lower() != COL_TANGGAL:
            df_stock = df_stock.rename(columns={df_stock.columns[0]: COL_TANGGAL})

    df_stock[COL_TANGGAL] = pd.to_datetime(df_stock[COL_TANGGAL], errors='coerce')
    df_p[COL_TANGGAL] = pd.to_datetime(df_p[COL_TANGGAL], errors='coerce')

    df_merge = pd.merge(df_stock, df_p[[COL_TANGGAL, rice_type]], on=COL_TANGGAL, how='inner')
    df_merge.dropna(inplace=True)

    if len(df_merge) < 2: return None

    lr = linregress(df_merge[COL_STOK], df_merge[rice_type])
    
    slope = getattr(lr, 'slope', None)
    intercept = getattr(lr, 'intercept', None)
    p_value = getattr(lr, 'pvalue', None)
    r_val = getattr(lr, 'rvalue', None)

    r2_val = None
    if r_val is not None:
        try:
            r_float = float(r_val)
            r2_val = r_float ** 2
        except (ValueError, TypeError):
            r2_val = None

    return {
        'slope': slope,
        'intercept': intercept,
        'r2': r2_val,
        'p_value': p_value,
        'df': df_merge
    }