# lib/data.py
import pandas as pd
import numpy as np
import datetime
from typing import Optional, Tuple
import streamlit as st
from sqlalchemy import create_engine, text
from .utils import clean_colname
from .constants import *

@st.cache_data
def get_geo_lookup():
    geo_data = {
        COL_LOKASI: ['Bandung', 'Banten', 'Bekasi', 'Bogor', 'Bulog', 'Cianjur', 
                   'Cirebon', 'DKI', 'Jateng', 'Jatim', 'Karawang', 'Tangerang', 'Tj Priok'],
        'lat': [-6.9175, -6.1200, -6.2383, -6.5950, -6.2568, -6.8207, 
                -6.7061, -6.1751, -6.9667, -7.2575, -6.3290, -6.1781, -6.1044],
        'lon': [107.6191, 106.1518, 106.9756, 106.7997, 106.8431, 107.1432, 
                108.5570, 106.8272, 110.4167, 112.7521, 107.3007, 106.6300, 106.8835]
    }
    return pd.DataFrame(geo_data)

@st.cache_resource(ttl=3600)
def init_connection():
    try:
        # Coba ambil dari secrets
        if "connections" in st.secrets and "mysql_db" in st.secrets["connections"]:
            db_config = st.secrets["connections"]["mysql_db"]
            conn = (f"mysql+mysqlconnector://{db_config['username']}:{db_config['password']}@"
                    f"{db_config['host']}:{db_config['port']}/{db_config['database']}")
            return create_engine(conn)
        return None
    except Exception:
        return None

@st.cache_data(ttl=600)
def load_data_from_db(_engine, start_date: Optional[datetime.date] = None, end_date: Optional[datetime.date] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Memuat data dari database dengan strategi MULTI-QUERY yang disesuaikan dengan skema db_pibc_olap.
    """
    if _engine is None: return None, None, None, None, None

    params = {}
    date_filter = ""
    
    # OPTIMASI: Filter langsung menggunakan integer SK (YYYYMMDD) agar cepat
    if start_date and end_date:
        start_sk = int(start_date.strftime('%Y%m%d'))
        end_sk = int(end_date.strftime('%Y%m%d'))
        date_filter = "AND d.SK BETWEEN :start_sk AND :end_sk"
        params['start_sk'] = start_sk
        params['end_sk'] = end_sk

    try:
        with _engine.connect() as conn:
            # 1. QUERY STOK & NERACA UTAMA
            # Mengambil dari fact_rice_income_outcome yang sudah diagregasi
            q_stock = text(f"""
                SELECT 
                    d.YEAR, d.MONTH, d.DAY,
                    fio.TOTAL_WEIGHT_INCOME AS {COL_MASUK},
                    fio.TOTAL_WEIGHT_OUTCOME AS {COL_KELUAR},
                    fio.WEIGHT_STOCK AS {COL_STOK}
                FROM dim_date d
                JOIN fact_rice_income_outcome fio ON d.SK = fio.SK_DATE
                WHERE 1=1 {date_filter}
                ORDER BY d.SK
            """)
            df_main = pd.read_sql(q_stock, conn, params=params)
            
            if df_main.empty:
                return None, None, None, None, None

            # Konversi Tanggal
            df_main[COL_TANGGAL] = pd.to_datetime(df_main[["YEAR", "MONTH", "DAY"]].rename(columns={"YEAR": "year", "MONTH": "month", "DAY": "day"}), errors="coerce")
            df_main[COL_NERACA] = df_main[COL_MASUK] - df_main[COL_KELUAR]
            df_stock = df_main[[COL_TANGGAL, COL_STOK]].copy()

            # 2. QUERY HARGA (Pivot per Jenis Beras)
            q_price = text(f"""
                SELECT 
                    d.YEAR, d.MONTH, d.DAY,
                    fh.PRICE AS {COL_HARGA},
                    drt.RICE_TYPE_NAME AS {COL_NAMA_JENIS}
                FROM dim_date d
                JOIN fact_harga fh ON d.SK = fh.SK_DATE
                JOIN dim_rice_type drt ON fh.SK_RICE_TYPE = drt.SK
                JOIN dim_market dm ON fh.SK_MARKET = dm.SK
                WHERE dm.MARKET_NAME = 'Pasar Induk Beras Cipinang' 
                {date_filter}
            """)
            df_p_raw = pd.read_sql(q_price, conn, params=params)
            
            if not df_p_raw.empty:
                df_p_raw[COL_TANGGAL] = pd.to_datetime(df_p_raw[["YEAR", "MONTH", "DAY"]].rename(columns={"YEAR": "year", "MONTH": "month", "DAY": "day"}), errors="coerce")
                # Pivot agar nama beras jadi kolom
                df_price = df_p_raw.pivot_table(index=COL_TANGGAL, columns=COL_NAMA_JENIS, values=COL_HARGA, aggfunc='mean')
            else:
                df_price = pd.DataFrame()

            # 3. QUERY PETA MASUK (Supply)
            # PERBAIKAN: Menggunakan kolom 'WEIGHT' (bukan TOTAL_WEIGHT) sesuai tabel fact_rice_income
            q_masuk = text(f"""
                SELECT 
                    d.YEAR, d.MONTH, d.DAY,
                    dp.PLACE_NAME AS {COL_LOKASI},
                    fri.WEIGHT AS {COL_MASUK} 
                FROM dim_date d
                JOIN fact_rice_income fri ON d.SK = fri.SK_DATE
                JOIN dim_place dp ON fri.SK_PLACE = dp.SK
                WHERE 1=1 {date_filter}
            """)
            df_masuk_long = pd.read_sql(q_masuk, conn, params=params)
            
            if not df_masuk_long.empty:
                df_masuk_long[COL_TANGGAL] = pd.to_datetime(df_masuk_long[["YEAR", "MONTH", "DAY"]].rename(columns={"YEAR": "year", "MONTH": "month", "DAY": "day"}), errors="coerce")
                df_masuk_long[COL_LOKASI_NORM] = df_masuk_long[COL_LOKASI].astype(str).str.strip().str.lower()
            else:
                df_masuk_long = pd.DataFrame(columns=[COL_TANGGAL, COL_LOKASI, COL_LOKASI_NORM, COL_MASUK])

            # 4. QUERY PETA KELUAR (Distribution)
            # PERBAIKAN: Menggunakan kolom 'WEIGHT' (bukan TOTAL_WEIGHT) sesuai tabel fact_rice_outcome
            q_keluar = text(f"""
                SELECT 
                    d.YEAR, d.MONTH, d.DAY,
                    dp.PLACE_NAME AS {COL_LOKASI},
                    fro.WEIGHT AS {COL_KELUAR}
                FROM dim_date d
                JOIN fact_rice_outcome fro ON d.SK = fro.SK_DATE
                JOIN dim_place dp ON fro.SK_PLACE = dp.SK
                WHERE 1=1 {date_filter}
            """)
            df_keluar_long = pd.read_sql(q_keluar, conn, params=params)

            if not df_keluar_long.empty:
                df_keluar_long[COL_TANGGAL] = pd.to_datetime(df_keluar_long[["YEAR", "MONTH", "DAY"]].rename(columns={"YEAR": "year", "MONTH": "month", "DAY": "day"}), errors="coerce")
                df_keluar_long[COL_LOKASI_NORM] = df_keluar_long[COL_LOKASI].astype(str).str.strip().str.lower()
            else:
                df_keluar_long = pd.DataFrame(columns=[COL_TANGGAL, COL_LOKASI, COL_LOKASI_NORM, COL_KELUAR])

            return df_main, df_stock, df_masuk_long, df_keluar_long, df_price

    except ImportError:
        st.error("Library `mysql-connector-python` belum diinstall. Jalankan `pip install mysql-connector-python`")
        return None, None, None, None, None
    except Exception as e:
        # Tampilkan error detail untuk debugging
        st.error(f"⚠️ Terjadi kesalahan Database: {str(e)}")
        return None, None, None, None, None

@st.cache_data(ttl=3600)
def preprocess_data_from_excel(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Memproses file Excel menjadi DataFrame."""
    if uploaded_file is None: return None, None, None, None, None
    
    # Reset pointer file (Penting untuk menghindari file terbaca kosong)
    try: uploaded_file.seek(0)
    except: pass
    
    try:
        data = pd.read_excel(uploaded_file, sheet_name=None)
    except Exception as e:
        st.error(f"Gagal membaca file Excel. Pastikan format valid (.xlsx). Error: {e}")
        return None, None, None, None, None

    def get_clean_sheet(name):
        df = data.get(name)
        if df is not None:
            df = df.copy()
            df.columns = [clean_colname(c).strip() for c in df.columns]
        return df

    def _find_date_col(df):
        if df is None or df.columns.size == 0: return None
        for c in df.columns:
            if str(c).lower() in (COL_TANGGAL,'date','tgl','hari'): return c
        for c in df.columns:
            try:
                if np.issubdtype(df[c].dtype, np.datetime64): return c
            except: continue
        return df.columns[0]

    df_stock = get_clean_sheet(SHEET_RICE_STOCK)
    df_delivery = get_clean_sheet(SHEET_RICE_DELIVERY)
    df_source = get_clean_sheet(SHEET_RICE_SOURCE)
    df_price_raw = get_clean_sheet(SHEET_RICE_PRICE)

    if df_stock is None: return None, None, None, None, None

    # Price safe handling
    df_price = None
    if df_price_raw is not None:
        df_price_raw = df_price_raw.copy()
        df_price_raw.columns = [clean_colname(c).lower() for c in df_price_raw.columns]
        date_col = next((c for c in df_price_raw.columns if c in (COL_TANGGAL,'date','tgl')), None) or (df_price_raw.columns[0] if df_price_raw.shape[1] else None)
        if date_col is not None:
            df_price_raw[date_col] = pd.to_datetime(df_price_raw[date_col], errors='coerce')
            name_col = next((c for c in df_price_raw.columns if 'jenis' in c or 'type' in c or 'nama' in c), None)
            price_col = next((c for c in df_price_raw.columns if COL_HARGA in c or 'price' in c or 'value' in c), None)
            if name_col and price_col:
                df_price = df_price_raw.pivot_table(index=date_col, columns=name_col, values=price_col, aggfunc='mean')
                df_price.index.name = COL_TANGGAL
            else:
                df_price_raw = df_price_raw.set_index(date_col).sort_index()
                df_price_raw.index.name = COL_TANGGAL
                df_price = df_price_raw.copy()

    # Stock processing
    df_stock = df_stock.copy()
    date_col_stock = _find_date_col(df_stock)
    df_stock[date_col_stock] = pd.to_datetime(df_stock[date_col_stock], errors='coerce')
    df_stock = df_stock.rename(columns={date_col_stock: COL_TANGGAL})
    for want in [COL_STOK, COL_MASUK, COL_KELUAR]:
        found = next((c for c in df_stock.columns if want in c.lower()), None)
        if found: df_stock[want] = pd.to_numeric(df_stock[found], errors='coerce').fillna(0)
        elif want not in df_stock.columns: df_stock[want] = 0
    
    for col in [COL_STOK, COL_MASUK, COL_KELUAR]:
        if col in df_stock.columns: df_stock[col] = df_stock[col].clip(lower=0)

    df_stock[COL_TANGGAL] = pd.to_datetime(df_stock[COL_TANGGAL], errors='coerce')
    df_main = df_stock[[COL_TANGGAL,COL_STOK]].copy()
    
    if COL_MASUK in df_stock.columns: masuk_series = pd.to_numeric(df_stock[COL_MASUK], errors='coerce').fillna(0)
    else: masuk_series = pd.Series([0] * len(df_stock), index=df_stock.index, dtype=float)
    df_main['masuk_from_stock'] = masuk_series

    if COL_KELUAR in df_stock.columns: keluar_series = pd.to_numeric(df_stock[COL_KELUAR], errors='coerce').fillna(0)
    else: keluar_series = pd.Series([0] * len(df_stock), index=df_stock.index, dtype=float)
    df_main['keluar_from_stock'] = keluar_series

    # Source -> long
    if df_source is None: df_masuk_long = pd.DataFrame(columns=[COL_TANGGAL, COL_LOKASI, COL_LOKASI_NORM, COL_MASUK])
    else:
        df_source = df_source.copy()
        date_col_src = _find_date_col(df_source)
        if date_col_src:
            df_source[date_col_src] = pd.to_datetime(df_source[date_col_src], errors='coerce')
            df_source = df_source.rename(columns={date_col_src:COL_TANGGAL})
        else: df_source[COL_TANGGAL] = pd.NaT
        val_cols_src = [c for c in df_source.columns if c != COL_TANGGAL]
        if val_cols_src:
            df_masuk_long = df_source.melt(id_vars=COL_TANGGAL, value_vars=val_cols_src, var_name=COL_LOKASI, value_name=COL_MASUK)
            df_masuk_long[COL_LOKASI] = df_masuk_long[COL_LOKASI].astype(str).str.strip()
            df_masuk_long[COL_LOKASI_NORM] = df_masuk_long[COL_LOKASI].str.lower().str.strip()
            df_masuk_long[COL_MASUK] = pd.to_numeric(df_masuk_long[COL_MASUK], errors='coerce').fillna(0).clip(lower=0)
            df_masuk_long = df_masuk_long[df_masuk_long[COL_MASUK] > 0].reset_index(drop=True)
        else: df_masuk_long = pd.DataFrame(columns=[COL_TANGGAL, COL_LOKASI, COL_LOKASI_NORM, COL_MASUK])

    # Delivery -> long
    if df_delivery is None: df_keluar_long = pd.DataFrame(columns=[COL_TANGGAL, COL_LOKASI, COL_LOKASI_NORM, COL_KELUAR])
    else:
        df_delivery = df_delivery.copy()
        date_col_del = _find_date_col(df_delivery)
        if date_col_del:
            df_delivery[date_col_del] = pd.to_datetime(df_delivery[date_col_del], errors='coerce')
            df_delivery = df_delivery.rename(columns={date_col_del:COL_TANGGAL})
        else: df_delivery[COL_TANGGAL] = pd.NaT
        val_cols_del = [c for c in df_delivery.columns if c != COL_TANGGAL]
        if val_cols_del:
            df_keluar_long = df_delivery.melt(id_vars=COL_TANGGAL, value_vars=val_cols_del, var_name=COL_LOKASI, value_name=COL_KELUAR)
            df_keluar_long[COL_LOKASI] = df_keluar_long[COL_LOKASI].astype(str).str.strip()
            df_keluar_long[COL_LOKASI_NORM] = df_keluar_long[COL_LOKASI].str.lower().str.strip()
            df_keluar_long[COL_KELUAR] = pd.to_numeric(df_keluar_long[COL_KELUAR], errors='coerce').fillna(0).clip(lower=0)
            df_keluar_long = df_keluar_long[df_keluar_long[COL_KELUAR] > 0].reset_index(drop=True)
        else: df_keluar_long = pd.DataFrame(columns=[COL_TANGGAL, COL_LOKASI, COL_LOKASI_NORM, COL_KELUAR])

    # Aggregate
    df_masuk_date = df_masuk_long.groupby(COL_TANGGAL, as_index=False)[COL_MASUK].sum() if not df_masuk_long.empty else pd.DataFrame(columns=[COL_TANGGAL,COL_MASUK])
    df_keluar_date = df_keluar_long.groupby(COL_TANGGAL, as_index=False)[COL_KELUAR].sum() if not df_keluar_long.empty else pd.DataFrame(columns=[COL_TANGGAL,COL_KELUAR])
    
    # Merge dengan data utama
    df_main = df_main.merge(df_masuk_date, on=COL_TANGGAL, how='left')
    df_main = df_main.merge(df_keluar_date, on=COL_TANGGAL, how='left')
    
    # Prioritaskan data detail jika ada, jika tidak pakai data stok global
    df_main[COL_MASUK] = pd.to_numeric(df_main[COL_MASUK].fillna(df_main.get('masuk_from_stock', 0)), errors='coerce').fillna(0)
    df_main[COL_KELUAR] = pd.to_numeric(df_main[COL_KELUAR].fillna(df_main.get('keluar_from_stock', 0)), errors='coerce').fillna(0)
    df_main[COL_NERACA] = df_main[COL_MASUK] - df_main[COL_KELUAR]

    if not df_masuk_long.empty: df_masuk_long[COL_TANGGAL] = pd.to_datetime(df_masuk_long[COL_TANGGAL], errors='coerce')
    if not df_keluar_long.empty: df_keluar_long[COL_TANGGAL] = pd.to_datetime(df_keluar_long[COL_TANGGAL], errors='coerce')

    return df_main, df_stock, df_masuk_long, df_keluar_long, df_price