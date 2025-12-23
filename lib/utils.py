# lib/utils.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from scipy.stats import linregress
import logging
import requests
from .constants import COL_TANGGAL, COL_STOK

# Setup Logger
logger = logging.getLogger(__name__)

def load_lottie_url(url: str):
    """
    Memuat data animasi lottie (JSON) dari URL.
    Added: Timeout untuk mencegah hanging request.
    """
    try:
        r = requests.get(url, timeout=5) # Timeout 5 detik
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        logger.warning(f"Gagal memuat Lottie: {e}")
        return None
        
def clean_colname(c: Optional[str]) -> str:
    """
    Membersihkan nama kolom: menghapus baris baru dan spasi berlebih.
    """
    if c is None:
        return ""
    return str(c).replace("\n", " ").strip()

def price_df_with_tanggal(df_price: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Memastikan DataFrame harga memiliki kolom 'tanggal' bertipe datetime.
    Cerdas mencari kolom tanggal, merename-nya, dan mengonversi tipenya.
    Juga mengurutkan data (penting untuk merge_asof).
    """
    if df_price is None or df_price.empty:
        return None
    
    p = df_price.copy()
    p.columns = [clean_colname(c) for c in p.columns]
    
    # Deteksi kolom tanggal atau index
    if isinstance(p.index, pd.DatetimeIndex) or (p.index.name and str(p.index.name).lower() == COL_TANGGAL):
        p = p.reset_index()
        if 'index' in p.columns:
            p = p.rename(columns={'index': COL_TANGGAL})
    else:
        # Cari kolom yang mirip 'date', 'tgl', dll
        date_col = next((c for c in p.columns if c.lower() in ("date", COL_TANGGAL, "tgl", "hari")), None)
        if date_col:
            p = p.rename(columns={date_col: COL_TANGGAL})
        else:
            # Fallback: coba convert index
            p[COL_TANGGAL] = pd.to_datetime(p.index, errors="coerce")
            
    # Konversi ke datetime dan Urutkan (Wajib untuk merge_asof)
    p[COL_TANGGAL] = pd.to_datetime(p[COL_TANGGAL], errors="coerce")
    p = p.sort_values(by=COL_TANGGAL)
    
    return p

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Konversi DataFrame ke CSV bytes (UTF-8) untuk download button."""
    return df.to_csv(index=False).encode('utf-8')

def calculate_regression(df_stock: pd.DataFrame, df_price: pd.DataFrame, rice_type: str) -> Optional[Dict[str, Any]]:
    """
    Menghitung regresi linear antara Stok vs Harga.
    
    IMPROVEMENT: 
    Menggunakan 'merge_asof' (backward) alih-alih 'inner join'.
    Ini mencocokkan stok tanggal X dengan harga tanggal X (jika ada) 
    atau harga terakhir yang diketahui sebelum X.
    """
    if df_stock is None or df_price is None: 
        return None

    # 1. Persiapan Data Harga
    df_p = price_df_with_tanggal(df_price)
    if df_p is None or rice_type not in df_p.columns:
        logger.warning(f"Regresi Gagal: Kolom '{rice_type}' tidak ditemukan di data harga.")
        return None

    # 2. Persiapan Data Stok
    if COL_TANGGAL not in df_stock.columns:
        df_stock = df_stock.reset_index()
        # Rename kolom pertama jika belum sesuai standar
        if not df_stock.empty and df_stock.columns[0].lower() != COL_TANGGAL:
             df_stock = df_stock.rename(columns={df_stock.columns[0]: COL_TANGGAL})

    # Pastikan format datetime
    df_stock[COL_TANGGAL] = pd.to_datetime(df_stock[COL_TANGGAL], errors='coerce')
    
    # 3. Sorting (Wajib untuk merge_asof)
    df_stock = df_stock.sort_values(COL_TANGGAL).dropna(subset=[COL_TANGGAL])
    df_p = df_p.sort_values(COL_TANGGAL).dropna(subset=[COL_TANGGAL])

    if df_stock.empty or df_p.empty:
        return None

    # 4. MERGE DATA (Logika Baru)
    # Gunakan merge_asof dengan direction='backward'
    # Stok tgl 10 cari harga tgl 10. Kalau tgl 10 libur, ambil harga tgl 9.
    try:
        df_merge = pd.merge_asof(
            df_stock, 
            df_p[[COL_TANGGAL, rice_type]], 
            on=COL_TANGGAL, 
            direction='backward'
        )
    except Exception as e:
        logger.error(f"Merge error: {e}")
        return None

    # Hapus baris yang harganya masih NaN (misal stok ada sebelum data harga pertama tercatat)
    df_merge.dropna(subset=[COL_STOK, rice_type], inplace=True)

    # 5. Cek Kecukupan Data
    if len(df_merge) < 5:
        logger.warning(f"Data point ({len(df_merge)}) terlalu sedikit untuk regresi yang valid.")
        return None

    # 6. Hitung Statistik Regresi
    slope, intercept, r_value, p_value, std_err = linregress(df_merge[COL_STOK], df_merge[rice_type])
    
    r2_val = None
    if r_value is not None:
        try:
            r2_val = r_value ** 2 # type: ignore
        except (ValueError, TypeError):
            r2_val = 0.0

    return {
        'slope': slope,
        'intercept': intercept,
        'r2': r2_val,
        'p_value': p_value,
        'std_err': std_err,
        'df': df_merge # DataFrame hasil gabungan dikembalikan untuk visualisasi scatter plot
    }