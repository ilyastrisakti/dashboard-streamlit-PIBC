import pandas as pd
import numpy as np
from typing import Optional, Tuple
import streamlit as st
from sqlalchemy import create_engine, text
from .utils import _clean_colname

@st.cache_data
def get_geo_lookup():
    geo_data = {
        'lokasi': ['Bandung', 'Banten', 'Bekasi', 'Bogor', 'Bulog', 'Cianjur', 
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
        db_config = st.secrets["connections"]["mysql_db"]
        conn = (f"mysql+mysqlconnector://{db_config['username']}:{db_config['password']}@"
                f"{db_config['host']}:{db_config['port']}/{db_config['database']}")
        return create_engine(conn)
    except Exception:
        return None

@st.cache_data(ttl=600)
def load_data_from_db(_engine) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    if _engine is None:
        return None, None, None, None, None
    query = text("""SELECT d.YEAR, d.MONTH, d.DAY, fio.TOTAL_WEIGHT_INCOME AS masuk, fio.TOTAL_WEIGHT_OUTCOME AS keluar, fio.WEIGHT_STOCK AS stok, fh.PRICE AS harga, drt.RICE_TYPE_NAME AS nama_jenis, dm.MARKET_NAME FROM dim_date d LEFT JOIN fact_rice_income_outcome fio ON d.SK = fio.SK_DATE LEFT JOIN fact_harga fh ON d.SK = fh.SK_DATE LEFT JOIN dim_rice_type drt ON fh.SK_RICE_TYPE = drt.SK LEFT JOIN dim_market dm ON fh.SK_MARKET = dm.SK WHERE dm.MARKET_NAME = 'Pasar Induk Beras Cipinang' ORDER BY d.SK;""")
    try:
        with _engine.connect() as conn:
            df = pd.read_sql(query, conn)
        df["tanggal"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]].rename(columns={"YEAR": "year", "MONTH": "month", "DAY": "day"}), errors="coerce")
        df_main = df.groupby("tanggal").agg(masuk=("masuk","sum"), keluar=("keluar","sum"), stok=("stok","mean")).reset_index()
        df_main.fillna(0, inplace=True)
        df_main["neraca"] = df_main["masuk"] - df_main["keluar"]
        df_stock = df_main[["tanggal", "stok"]].copy()
        df_masuk = df.groupby(["tanggal"])['masuk'].sum().reset_index()
        df_keluar = df.groupby(["tanggal"])['keluar'].sum().reset_index()
        df_price = df.pivot_table(index='tanggal', columns='nama_jenis', values='harga', aggfunc='mean')
        return df_main, df_stock, df_masuk, df_keluar, df_price
    except Exception as e:
        st.error(f"DB Error: {e}")
        return None, None, None, None, None

@st.cache_data(ttl=3600)
def preprocess_data_from_excel(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    # (kept same robust implementation used sebelumnya: baca sheet, wide->long, aggregate merge)
    data = pd.read_excel(uploaded_file, sheet_name=None)
    def get_clean_sheet(name):
        df = data.get(name)
        if df is not None:
            df = df.copy()
            df.columns = [_clean_colname(c).strip() for c in df.columns]
        return df

    def _find_date_col(df):
        if df is None or df.columns.size == 0:
            return None
        for c in df.columns:
            if str(c).lower() in ('tanggal','date','tgl','hari'):
                return c
        for c in df.columns:
            try:
                if np.issubdtype(df[c].dtype, np.datetime64):
                    return c
            except Exception:
                continue
        return df.columns[0]

    df_stock = get_clean_sheet('rice_stock')
    df_delivery = get_clean_sheet('rice_delivery')
    df_source = get_clean_sheet('rice_source')
    df_price_raw = get_clean_sheet('rice_price')

    if df_stock is None:
        return None, None, None, None, None

    # Price safe handling (same logic as before)...
    df_price = None
    if df_price_raw is not None:
        df_price_raw = df_price_raw.copy()
        df_price_raw.columns = [_clean_colname(c).lower() for c in df_price_raw.columns]
        date_col = next((c for c in df_price_raw.columns if c in ('tanggal','date','tgl')), None) or (df_price_raw.columns[0] if df_price_raw.shape[1] else None)
        if date_col is not None:
            df_price_raw[date_col] = pd.to_datetime(df_price_raw[date_col], errors='coerce')
            name_col = next((c for c in df_price_raw.columns if 'jenis' in c or 'type' in c or 'nama' in c), None)
            price_col = next((c for c in df_price_raw.columns if 'harga' in c or 'price' in c or 'value' in c), None)
            if name_col and price_col:
                df_price = df_price_raw.pivot_table(index=date_col, columns=name_col, values=price_col, aggfunc='mean')
                df_price.index.name = 'tanggal'
                df_price.index = pd.to_datetime(df_price.index, errors='coerce')
            else:
                df_price_raw = df_price_raw.set_index(date_col).sort_index()
                df_price_raw.index.name = 'tanggal'
                df_price_raw.index = pd.to_datetime(df_price_raw.index, errors='coerce')
                df_price = df_price_raw.copy()

    # Stock processing and wide->long migrate the same way as previous robust version
    df_stock = df_stock.copy()
    date_col_stock = _find_date_col(df_stock)
    df_stock[date_col_stock] = pd.to_datetime(df_stock[date_col_stock], errors='coerce')
    df_stock = df_stock.rename(columns={date_col_stock: 'tanggal'})
    for want in ['stok','masuk','keluar']:
        found = next((c for c in df_stock.columns if want in c.lower()), None)
        if found:
            df_stock[want] = pd.to_numeric(df_stock[found], errors='coerce').fillna(0)
        elif want not in df_stock.columns:
            df_stock[want] = 0
    df_stock['tanggal'] = pd.to_datetime(df_stock['tanggal'], errors='coerce')
    df_main = df_stock[['tanggal','stok']].copy()
    # ensure these are Series with the same length as df_stock so .fillna works
    if 'masuk' in df_stock.columns:
        masuk_series = pd.to_numeric(df_stock['masuk'], errors='coerce').fillna(0)
    else:
        masuk_series = pd.Series([0] * len(df_stock), index=df_stock.index, dtype=float)
    df_main['masuk_from_stock'] = masuk_series

    if 'keluar' in df_stock.columns:
        keluar_series = pd.to_numeric(df_stock['keluar'], errors='coerce').fillna(0)
    else:
        keluar_series = pd.Series([0] * len(df_stock), index=df_stock.index, dtype=float)
    df_main['keluar_from_stock'] = keluar_series

    # source -> long
    if df_source is None:
        df_masuk_long = pd.DataFrame(columns=['tanggal','lokasi','lokasi_norm','masuk'])
    else:
        df_source = df_source.copy()
        date_col_src = _find_date_col(df_source)
        if date_col_src:
            df_source[date_col_src] = pd.to_datetime(df_source[date_col_src], errors='coerce')
            df_source = df_source.rename(columns={date_col_src:'tanggal'})
        else:
            df_source['tanggal'] = pd.NaT
        val_cols_src = [c for c in df_source.columns if c != 'tanggal']
        if val_cols_src:
            df_masuk_long = df_source.melt(id_vars='tanggal', value_vars=val_cols_src, var_name='lokasi', value_name='masuk')
            df_masuk_long['lokasi'] = df_masuk_long['lokasi'].astype(str).str.strip()
            df_masuk_long['lokasi_norm'] = df_masuk_long['lokasi'].str.lower().str.strip()
            df_masuk_long['masuk'] = pd.to_numeric(df_masuk_long['masuk'], errors='coerce').fillna(0)
            df_masuk_long = df_masuk_long[df_masuk_long['masuk'] > 0].reset_index(drop=True)
        else:
            df_masuk_long = pd.DataFrame(columns=['tanggal','lokasi','lokasi_norm','masuk'])

    # delivery -> long
    if df_delivery is None:
        df_keluar_long = pd.DataFrame(columns=['tanggal','lokasi','lokasi_norm','keluar'])
    else:
        df_delivery = df_delivery.copy()
        date_col_del = _find_date_col(df_delivery)
        if date_col_del:
            df_delivery[date_col_del] = pd.to_datetime(df_delivery[date_col_del], errors='coerce')
            df_delivery = df_delivery.rename(columns={date_col_del:'tanggal'})
        else:
            df_delivery['tanggal'] = pd.NaT
        val_cols_del = [c for c in df_delivery.columns if c != 'tanggal']
        if val_cols_del:
            df_keluar_long = df_delivery.melt(id_vars='tanggal', value_vars=val_cols_del, var_name='lokasi', value_name='keluar')
            df_keluar_long['lokasi'] = df_keluar_long['lokasi'].astype(str).str.strip()
            df_keluar_long['lokasi_norm'] = df_keluar_long['lokasi'].str.lower().str.strip()
            df_keluar_long['keluar'] = pd.to_numeric(df_keluar_long['keluar'], errors='coerce').fillna(0)
            df_keluar_long = df_keluar_long[df_keluar_long['keluar'] > 0].reset_index(drop=True)
        else:
            df_keluar_long = pd.DataFrame(columns=['tanggal','lokasi','lokasi_norm','keluar'])

    # aggregate
    df_masuk_date = df_masuk_long.groupby('tanggal', as_index=False)['masuk'].sum() if not df_masuk_long.empty else pd.DataFrame(columns=['tanggal','masuk'])
    df_keluar_date = df_keluar_long.groupby('tanggal', as_index=False)['keluar'].sum() if not df_keluar_long.empty else pd.DataFrame(columns=['tanggal','keluar'])
    df_main = df_main.merge(df_masuk_date, on='tanggal', how='left')
    df_main = df_main.merge(df_keluar_date, on='tanggal', how='left')
    df_main['masuk'] = pd.to_numeric(df_main['masuk'].fillna(df_main.get('masuk_from_stock', 0)), errors='coerce').fillna(0)
    df_main['keluar'] = pd.to_numeric(df_main['keluar'].fillna(df_main.get('keluar_from_stock', 0)), errors='coerce').fillna(0)
    df_main['neraca'] = df_main['masuk'] - df_main['keluar']

    if not df_masuk_long.empty:
        df_masuk_long['tanggal'] = pd.to_datetime(df_masuk_long['tanggal'], errors='coerce')
    if not df_keluar_long.empty:
        df_keluar_long['tanggal'] = pd.to_datetime(df_keluar_long['tanggal'], errors='coerce')

    return df_main, df_stock, df_masuk_long, df_keluar_long, df_price