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
    """
    Referensi Koordinat Lokasi.
    Update: Koordinat disesuaikan agar titik berada di daratan (center of gravity)
    dan Luar Jawa digeser ke laut lepas agar visualisasi Arc lebih cantik.
    """
    geo_data = {
        COL_LOKASI: [
            # --- JAWA BARAT & BANTEN ---
            'Bandung', 'Cianjur', 'Cirebon', 'Indramayu', 
            'Karawang', 'Subang', 'Sukabumi', 'Sumedang',
            'Banten', 'Serang', 'Tangerang', 'Bekasi', 'Bogor', 
            
            # --- JAWA TENGAH (Digeser ke Tengah Daratan) ---
            'Jateng', 'Jawa Tengah', 'Solo', 'Sragen', 'Demak', 
            'Grobogan', 'Pati', 'Blora', 'Kendal', 'Pemalang', 'Tegal', 
            'Semarang', 'Yogya', 'Yogyakarta',
            
            # --- JAWA TIMUR (Digeser ke Tengah Daratan) ---
            'Jatim', 'Jawa Timur', 'Banyuwangi', 'Ngawi', 'Sidrap', # Sidrap (Sulsel) sering masuk data, kita mapping manual atau biarkan unknown
            
            # --- JAKARTA AREA ---
            'DKI', 'DKI Jakarta', 'Cipinang', 'Induk', 
            'Tj Priok', 'Tanjung Priok', 
            'Bulog', 'Ex Bulog', 'Gudang Bulog',
            
            # --- SUMBER LAIN / LUAR PULAU (Gate Laut) ---
            'Luar Jawa', 'Luar Pulau Jawa', 'Luar Negeri', 'Impor',
            'Lampung', 'Sumsel', 'Sulawesi', 'Makassar', 'NTB'
        ],
        'lat': [
            # JABAR
            -6.9175, -6.8207, -6.7320, -6.3276, 
            -6.3042, -6.5716, -6.9210, -6.8586,
            -6.4058, -6.1200, -6.1783, -6.2383, -6.5950,
            
            # JATENG (Jateng umum ditaruh di Magelang/Temanggung biar di tengah)
            -7.3000, -7.3000, -7.5755, -7.4278, -6.8909,
            -7.0266, -6.7465, -6.9698, -6.9197, -6.8950, -6.8694,
            -7.0051, -7.7955, -7.7955,
            
            # JATIM (Jatim umum ditaruh di Nganjuk/Kediri biar di tengah)
            -7.6000, -7.6000, -8.2192, -7.3963, -3.9000, # Sidrap Sulawesi ditaruh agak jauh
            
            # JAKARTA
            -6.1751, -6.1751, -6.2180, -6.2180, # DKI (Monas), Cipinang
            -6.1040, -6.1040,                   # Priok (Pelabuhan)
            -6.1500, -6.1500, -6.1500,          # Bulog (Kelapa Gading/Sunter)
            
            # LUAR JAWA (Laut Jawa Deep - Agar Arc Melengkung Cantik)
            -3.5000, -3.5000, -3.0000, -3.0000, 
            -5.2000, -3.5000, -4.0000, -5.1477, -8.6529 
        ],
        'lon': [
            # JABAR
            107.6191, 107.1432, 108.5523, 108.3198,
            107.3007, 107.7587, 106.9266, 107.9208,
            106.0640, 106.1518, 106.6319, 106.9756, 106.8166,
            
            # JATENG
            110.1500, 110.1500, 110.8243, 111.0189, 110.6396,
            110.9229, 111.0380, 111.4172, 110.2017, 109.3813, 109.1402,
            110.4381, 110.3695, 110.3695,
            
            # JATIM
            112.0000, 112.0000, 114.3691, 111.4552, 119.8500,
            
            # JAKARTA
            106.8650, 106.8650, 106.8960, 106.8960, # DKI digeser dikit ke timur biar gak numpuk Tangerang
            106.8830, 106.8830,                     # Priok
            106.9000, 106.9000, 106.9000,           # Bulog
            
            # LUAR JAWA
            107.0000, 107.0000, 107.0000, 107.0000, # Lurus utara Jakarta
            105.0000, 104.0000, 119.5000, 119.4327, 117.0000
        ]
    }
    
    # Hapus duplikat jika ada nama lokasi yang sama di list
    return pd.DataFrame(geo_data).drop_duplicates(subset=[COL_LOKASI])

@st.cache_resource(ttl=3600)
def init_connection():
    try:
        if "connections" in st.secrets and "mysql_db" in st.secrets["connections"]:
            db_config = st.secrets["connections"]["mysql_db"]
            conn = (f"mysql+mysqlconnector://{db_config['username']}:{db_config['password']}@"
                    f"{db_config['host']}:{db_config['port']}/{db_config['database']}")
            return create_engine(conn)
        return None
    except Exception as e:
        print(f"Database Connection Error: {e}")
        return None
    
def _parse_sk_date(df, sk_col='SK_DATE'):
    """
    Helper untuk parsing tanggal dari format SK (Integer YYYYMMDD) ke Datetime.
    """
    if df.empty: return df
    
    # 1. Coba ambil dari kolom terpisah YEAR, MONTH, DAY (jika ada)
    if 'YEAR' in df.columns and 'MONTH' in df.columns and 'DAY' in df.columns:
         df['temp_date'] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]].rename(columns={"YEAR": "year", "MONTH": "month", "DAY": "day"}), errors="coerce")
    else:
         df['temp_date'] = pd.NaT

    # 2. Parsing dari kolom SK (misal 20240101)
    df[sk_col] = df[sk_col].astype(str)
    mask_nat = df['temp_date'].isna()
    
    # Isi yang kosong dengan hasil parsing SK
    df.loc[mask_nat, COL_TANGGAL] = pd.to_datetime(df.loc[mask_nat, sk_col], format='%Y%m%d', errors='coerce')
    # Isi sisanya dengan temp_date
    df.loc[~mask_nat, COL_TANGGAL] = df.loc[~mask_nat, 'temp_date']
    
    if 'temp_date' in df.columns: df.drop(columns=['temp_date'], inplace=True)
    return df

@st.cache_data(ttl=600)
def load_data_from_db(_engine, start_date: Optional[datetime.date] = None, end_date: Optional[datetime.date] = None) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Memuat data dari Database dengan LOGIKA DETEKSI SATUAN (Kg -> Ton).
    """
    if _engine is None: return None, None, None, None, None

    params = {}
    date_filter_sql = ""
    if start_date and end_date:
        start_sk = int(start_date.strftime('%Y%m%d'))
        end_sk = int(end_date.strftime('%Y%m%d'))
        params = {"start_sk": start_sk, "end_sk": end_sk}
        date_filter_sql = "AND fio.SK_DATE BETWEEN :start_sk AND :end_sk"

    try:
        with _engine.connect() as conn:
            # ==========================================
            # 1. QUERY STOK (Global)
            # ==========================================
            # Catatan: Kita ambil RAW DATA (tanpa / 1000 di SQL) biar Python yang nentukan
            q_stock = text(f"""
                SELECT fio.SK_DATE, d.YEAR, d.MONTH, d.DAY,
                    fio.TOTAL_WEIGHT_INCOME AS {COL_MASUK},
                    fio.TOTAL_WEIGHT_OUTCOME AS {COL_KELUAR},
                    fio.WEIGHT_STOCK AS {COL_STOK}
                FROM fact_rice_income_outcome fio 
                LEFT JOIN dim_date d ON fio.SK_DATE = d.SK
                WHERE 1=1 {date_filter_sql}
                ORDER BY fio.SK_DATE ASC
            """)
            df_main = pd.read_sql(q_stock, conn, params=params)
            
            if not df_main.empty:
                df_main = _parse_sk_date(df_main, 'SK_DATE')
                
                # --- LOGIKA DETEKSI OTOMATIS (Sama seperti Excel) ---
                for c in [COL_MASUK, COL_KELUAR, COL_STOK]:
                    df_main[c] = pd.to_numeric(df_main[c], errors='coerce').fillna(0)
                    # Jika rata-rata > 50, asumsi Kg -> Bagi 1000
                    if df_main[c].mean() > 50:
                        df_main[c] = df_main[c] / 1000.0
                # ----------------------------------------------------
                
                df_main[COL_NERACA] = df_main[COL_MASUK] - df_main[COL_KELUAR]
                df_stock = df_main[[COL_TANGGAL, COL_STOK]].copy()
            else:
                df_main, df_stock = pd.DataFrame(), pd.DataFrame()

            # ==========================================
            # 2. QUERY HARGA (Tidak Perlu Diubah - Rupiah)
            # ==========================================
            q_price = text(f"""
                SELECT fh.SK_DATE, d.YEAR, d.MONTH, d.DAY, fh.PRICE AS {COL_HARGA},
                    CASE WHEN drt.RICE_TYPE_NAME IS NOT NULL THEN drt.RICE_TYPE_NAME
                         ELSE CONCAT('Beras ID ', fh.SK_RICE_TYPE) END AS {COL_NAMA_JENIS}
                FROM fact_harga fh 
                LEFT JOIN dim_date d ON fh.SK_DATE = d.SK
                LEFT JOIN dim_market dm ON fh.SK_MARKET = dm.SK
                LEFT JOIN dim_rice_type drt ON fh.SK_RICE_TYPE = drt.SK
                WHERE (dm.MARKET_NAME = 'Pasar Induk Beras Cipinang' OR dm.MARKET_NAME IS NULL)
                AND fh.SK_DATE BETWEEN :start_sk AND :end_sk
                ORDER BY fh.SK_DATE ASC
            """)
            df_p_raw = pd.read_sql(q_price, conn, params=params)
            df_price = pd.DataFrame()
            if not df_p_raw.empty:
                df_p_raw = _parse_sk_date(df_p_raw, 'SK_DATE').dropna(subset=[COL_TANGGAL])
                df_p_raw[COL_HARGA] = pd.to_numeric(df_p_raw[COL_HARGA], errors='coerce').replace(0, np.nan)
                df_price = df_p_raw.pivot_table(index=COL_TANGGAL, columns=COL_NAMA_JENIS, values=COL_HARGA, aggfunc='mean').fillna(method='ffill', limit=3)

            # ==========================================
            # 3. QUERY MASUK (Detail)
            # ==========================================
            q_masuk = text(f"""
                SELECT fri.SK_DATE, d.YEAR, d.MONTH, d.DAY,
                    CASE WHEN dp.PLACE_NAME IS NOT NULL THEN dp.PLACE_NAME
                         ELSE CONCAT('Lokasi ID ', fri.SK_PLACE) END AS {COL_LOKASI},
                    fri.WEIGHT AS {COL_MASUK} 
                FROM fact_rice_income fri
                LEFT JOIN dim_date d ON fri.SK_DATE = d.SK
                LEFT JOIN dim_place dp ON fri.SK_PLACE = dp.SK
                WHERE fri.SK_DATE BETWEEN :start_sk AND :end_sk
                ORDER BY fri.SK_DATE ASC
            """)
            df_masuk_long = pd.read_sql(q_masuk, conn, params=params)
            
            if not df_masuk_long.empty:
                df_masuk_long = _parse_sk_date(df_masuk_long, 'SK_DATE')
                df_masuk_long[COL_LOKASI_NORM] = df_masuk_long[COL_LOKASI].astype(str).str.strip().str.lower()
                
                # --- LOGIKA DETEKSI OTOMATIS ---
                df_masuk_long[COL_MASUK] = pd.to_numeric(df_masuk_long[COL_MASUK], errors='coerce').fillna(0)
                if df_masuk_long[COL_MASUK].mean() > 50:
                    df_masuk_long[COL_MASUK] = df_masuk_long[COL_MASUK] / 1000.0
                # -------------------------------
            else:
                df_masuk_long = pd.DataFrame(columns=[COL_TANGGAL, COL_LOKASI, COL_LOKASI_NORM, COL_MASUK])

            # ==========================================
            # 4. QUERY KELUAR (Detail)
            # ==========================================
            q_keluar = text(f"""
                SELECT fro.SK_DATE, d.YEAR, d.MONTH, d.DAY,
                    CASE WHEN dp.PLACE_NAME IS NOT NULL THEN dp.PLACE_NAME
                         ELSE CONCAT('Lokasi ID ', fro.SK_PLACE) END AS {COL_LOKASI},
                    fro.WEIGHT AS {COL_KELUAR}
                FROM fact_rice_outcome fro
                LEFT JOIN dim_date d ON fro.SK_DATE = d.SK
                LEFT JOIN dim_place dp ON fro.SK_PLACE = dp.SK
                WHERE fro.SK_DATE BETWEEN :start_sk AND :end_sk
                ORDER BY fro.SK_DATE ASC
            """)
            df_keluar_long = pd.read_sql(q_keluar, conn, params=params)
            
            if not df_keluar_long.empty:
                df_keluar_long = _parse_sk_date(df_keluar_long, 'SK_DATE')
                df_keluar_long[COL_LOKASI_NORM] = df_keluar_long[COL_LOKASI].astype(str).str.strip().str.lower()
                
                # --- LOGIKA DETEKSI OTOMATIS ---
                df_keluar_long[COL_KELUAR] = pd.to_numeric(df_keluar_long[COL_KELUAR], errors='coerce').fillna(0)
                if df_keluar_long[COL_KELUAR].mean() > 50:
                    df_keluar_long[COL_KELUAR] = df_keluar_long[COL_KELUAR] / 1000.0
                # -------------------------------
            else:
                df_keluar_long = pd.DataFrame(columns=[COL_TANGGAL, COL_LOKASI, COL_LOKASI_NORM, COL_KELUAR])

            return df_main, df_stock, df_masuk_long, df_keluar_long, df_price

    except Exception as e:
        st.error(f"⚠️ Error Database: {str(e)}")
        return None, None, None, None, None

@st.cache_data(ttl=3600)
def preprocess_data_from_excel(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Memproses file Excel menjadi DataFrame."""
    if uploaded_file is None: return None, None, None, None, None
    
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
            
            if found: 
                # 1. Definisikan 'val' saat kolom ditemukan
                val = pd.to_numeric(df_stock[found], errors='coerce').fillna(0)
                
                # 2. Cek apakah perlu dibagi 1000 (Konversi Kg ke Ton)
                if val.mean() > 5000: 
                    val = val / 1000.0
                
                # 3. Masukkan ke dataframe
                df_stock[want] = val
                
            elif want not in df_stock.columns: 
                # Jika kolom tidak ditemukan, jangan panggil 'val', langsung isi 0
                df_stock[want] = 0
        
    
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
            if df_masuk_long[COL_MASUK].mean() > 500: 
                df_masuk_long[COL_MASUK] = df_masuk_long[COL_MASUK] / 1000.0
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
            if df_keluar_long[COL_KELUAR].mean() > 500: 
                df_keluar_long[COL_KELUAR] = df_keluar_long[COL_KELUAR] / 1000.0
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