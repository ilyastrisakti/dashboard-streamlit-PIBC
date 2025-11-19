# -*- coding: utf-8 -*-
"""
Merged PIBC Explorer
- Base: preprod_PIBC_explorer.py
- New Features: DataXplorPIBC_1 (1).ipynb (Geo, Volatility, Heatmap, Regression, Inventory Cover)
"""
import logging
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff # Added for KDE Distribution
import streamlit as st
from prophet import Prophet
from scipy.stats import pearsonr, linregress # Added linregress
from sqlalchemy import create_engine, text
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_option_menu import option_menu

# --- Basic setup ---
st.set_page_config(
    page_title="Dashboard Analisis PIBC 🌾",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Custom CSS ---
st.markdown(
    """
<style>
.title { font-size: 2.4rem; font-weight: 700; color: #2E86AB; text-align: center; margin-bottom:0.5rem; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0 0; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
.stTabs [aria-selected="true"] { background-color: #2E86AB; color: white; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# 1. Utilities / Helpers
# -------------------------

def _clean_colname(c: Optional[str]) -> str:
    if c is None: return ""
    return str(c).replace("\n", " ").strip()

def price_df_with_tanggal(df_price: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df_price is None: return None
    p = df_price.copy()
    p.columns = [_clean_colname(c) for c in p.columns]
    
    # Normalize columns to find date
    if isinstance(p.index, pd.DatetimeIndex) or (p.index.name and str(p.index.name).lower() == "tanggal"):
        p = p.reset_index()
        p.rename(columns={p.columns[0]: "tanggal"}, inplace=True)
    else:
        date_col = next((c for c in p.columns if c.lower() in ("date", "tanggal", "tgl", "hari")), None)
        if date_col:
            p = p.rename(columns={date_col: "tanggal"})
        else:
            # Fallback: try index coercion
            p["tanggal"] = pd.to_datetime(p.index, errors="coerce")
    
    p["tanggal"] = pd.to_datetime(p["tanggal"], errors="coerce")
    return p

@st.cache_data
def convert_df_to_csv(df) :
    """Mengubah DataFrame menjadi format CSV utf-8 agar bisa diunduh."""
    return df.to_csv(index=False).encode('utf-8')

# --- New Helper: Geo Coordinates (Hardcoded from Notebook) ---
@st.cache_data
def get_geo_lookup():
    """Returns a DataFrame with lat/lon for known locations."""
    # Data from Notebook Feature 5
    geo_data = {
        'lokasi': ['Bandung', 'Banten', 'Bekasi', 'Bogor', 'Bulog', 'Cianjur', 
                   'Cirebon', 'DKI', 'Jateng', 'Jatim', 'Karawang', 'Tangerang', 'Tj Priok'],
        'lat': [-6.9175, -6.1200, -6.2383, -6.5950, -6.2568, -6.8207, 
                -6.7061, -6.1751, -6.9667, -7.2575, -6.3290, -6.1781, -6.1044],
        'lon': [107.6191, 106.1518, 106.9756, 106.7997, 106.8431, 107.1432, 
                108.5570, 106.8272, 110.4167, 112.7521, 107.3007, 106.6300, 106.8835]
    }
    return pd.DataFrame(geo_data)

# -------------------------
# 2. Database & Loading
# -------------------------
@st.cache_resource(ttl=3600)
def init_connection():
    try:
        db_config = st.secrets["connections"]["mysql_db"]
        connection_string = (
            f"mysql+mysqlconnector://{db_config['username']}:{db_config['password']}@"
            f"{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        return create_engine(connection_string)
    except Exception:
        return None

@st.cache_data(ttl=600)
def load_data_from_db(_engine):
    if _engine is None: return None, None, None, None, None
    
    query = text("""
        SELECT d.YEAR, d.MONTH, d.DAY, fio.TOTAL_WEIGHT_INCOME AS masuk, 
        fio.TOTAL_WEIGHT_OUTCOME AS keluar, fio.WEIGHT_STOCK AS stok, fh.PRICE AS harga,
        drt.RICE_TYPE_NAME AS nama_jenis, dm.MARKET_NAME
        FROM dim_date d
        LEFT JOIN fact_rice_income_outcome fio ON d.SK = fio.SK_DATE
        LEFT JOIN fact_harga fh ON d.SK = fh.SK_DATE
        LEFT JOIN dim_rice_type drt ON fh.SK_RICE_TYPE = drt.SK
        LEFT JOIN dim_market dm ON fh.SK_MARKET = dm.SK
        WHERE dm.MARKET_NAME = 'Pasar Induk Beras Cipinang' ORDER BY d.SK;
    """)
    
    try:
        with _engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        df["tanggal"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]].rename(columns={"YEAR": "year", "MONTH": "month", "DAY": "day"}), errors="coerce")
        
        # Aggregation Logic
        df_main = df.groupby("tanggal").agg(masuk=("masuk", "first"), keluar=("keluar", "first"), stok=("stok", "first")).reset_index()
        df_main.fillna(0, inplace=True)
        df_main["neraca"] = df_main["masuk"] - df_main["keluar"]
        
        df_stock = df_main[["tanggal", "stok"]].copy()
        df_masuk = df.groupby(["tanggal"])['masuk'].sum().reset_index() # Placeholder structure
        df_keluar = df.groupby(["tanggal"])['keluar'].sum().reset_index() # Placeholder structure
        df_price = df.pivot_table(index='tanggal', columns='nama_jenis', values='harga')
        
        return df_main, df_stock, df_masuk, df_keluar, df_price
    except Exception as e:
        st.error(f"DB Error: {e}")
        return None, None, None, None, None

@st.cache_data(ttl=3600)
def preprocess_data_from_excel(uploaded_file):
    data = pd.read_excel(uploaded_file, sheet_name=None)
    
    # Helper to safely get and clean dataframe
    def get_clean_sheet(name):
        df = data.get(name)
        if df is not None:
            df.columns = [_clean_colname(c).lower() for c in df.columns]
        return df

    df_stock = get_clean_sheet('rice_stock')
    df_delivery = get_clean_sheet('rice_delivery')
    df_source = get_clean_sheet('rice_source')
    df_price_raw = data.get('rice_price') # Keep original case for columns

    if df_stock is None:
        return None, None, None, None, None

    # Fix Price Data
    if df_price_raw is not None:
        df_price_raw.columns = [_clean_colname(c) for c in df_price_raw.columns]
        # Find date column
        date_col = next((c for c in df_price_raw.columns if c.lower() in ('date', 'tanggal')), None)
        if date_col:
            df_price_raw[date_col] = pd.to_datetime(df_price_raw[date_col], errors='coerce')
            df_price_raw = df_price_raw.set_index(date_col)
            df_price_raw.index.name = 'tanggal'

    # Fix Stock Data
    date_col_stock = next((c for c in df_stock.columns if c in ('tanggal', 'date', 'tgl')), df_stock.columns[0])
    df_stock[date_col_stock] = pd.to_datetime(df_stock[date_col_stock], errors='coerce')
    df_stock.rename(columns={date_col_stock: 'tanggal'}, inplace=True)
    
    # Fix numeric columns in stock
    for c in ['stok', 'masuk', 'keluar']:
        if c not in df_stock.columns:
            # Try to find mapping or create 0
            found = next((x for x in df_stock.columns if c in x), None)
            if found: df_stock.rename(columns={found: c}, inplace=True)
            else: df_stock[c] = 0

    df_main = df_stock[['tanggal', 'stok', 'masuk', 'keluar']].copy()
    df_main['neraca'] = df_main['masuk'] - df_main['keluar']

    # Process Source/Delivery for Maps
    # Standardization for Source
    if df_source is not None:
        date_col_src = next((c for c in df_source.columns if c in ('tanggal', 'date')), None)
        if date_col_src:
            df_source[date_col_src] = pd.to_datetime(df_source[date_col_src], errors='coerce')
            df_source.rename(columns={date_col_src: 'tanggal'}, inplace=True)
            val_cols = [c for c in df_source.columns if c != 'tanggal']
            df_masuk_long = df_source.melt(id_vars='tanggal', value_vars=val_cols, var_name='lokasi', value_name='masuk')
            df_masuk_long = df_masuk_long[df_masuk_long['masuk'] > 0]
        else:
            df_masuk_long = pd.DataFrame(columns=['tanggal', 'lokasi', 'masuk'])
    else:
        df_masuk_long = pd.DataFrame(columns=['tanggal', 'lokasi', 'masuk'])

    # Standardization for Delivery
    if df_delivery is not None:
        date_col_del = next((c for c in df_delivery.columns if c in ('tanggal', 'date')), None)
        if date_col_del:
            df_delivery[date_col_del] = pd.to_datetime(df_delivery[date_col_del], errors='coerce')
            df_delivery.rename(columns={date_col_del: 'tanggal'}, inplace=True)
            val_cols = [c for c in df_delivery.columns if c != 'tanggal']
            df_keluar_long = df_delivery.melt(id_vars='tanggal', value_vars=val_cols, var_name='lokasi', value_name='keluar')
            df_keluar_long = df_keluar_long[df_keluar_long['keluar'] > 0]
        else:
            df_keluar_long = pd.DataFrame(columns=['tanggal', 'lokasi', 'keluar'])
    else:
        df_keluar_long = pd.DataFrame(columns=['tanggal', 'lokasi', 'keluar'])

    return df_main, df_stock, df_masuk_long, df_keluar_long, df_price_raw

# -------------------------
# 3. Visualization Logic
# -------------------------

# --- Original Visualizations ---
def create_time_series(df, y_col, title, color=None):
    if df is None or df.empty or y_col not in df.columns: return go.Figure()
    return px.line(df, x='tanggal', y=y_col, title=title, template='plotly_white', line_shape='spline', color_discrete_sequence=[color] if color else None)

def create_balance_chart(df):
    if df is None or df.empty: return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['tanggal'], y=df['masuk'], name='Masuk', marker_color='#2E86AB'))
    fig.add_trace(go.Bar(x=df['tanggal'], y=-df['keluar'], name='Keluar', marker_color='#F24236'))
    fig.add_trace(go.Scatter(x=df['tanggal'], y=df['neraca'], name='Neraca', line=dict(color='black', width=2, dash='dot')))
    return fig.update_layout(title='Neraca Harian (Masuk vs Keluar)', barmode='relative', template='plotly_white')

# --- NEW VISUALIZATIONS (From Notebook) ---

# Feature: Heatmap Correlation
def create_price_heatmap(df_price):
    if df_price is None or df_price.empty:
        return None
    
    corr = df_price.corr()
    fig = px.imshow(corr, 
                    text_auto=".2f", 
                    aspect="auto", 
                    color_continuous_scale='RdYlGn',
                    title="Heatmap Korelasi Harga Antar Jenis Beras")
    return fig

# Feature: Volatility Analysis
def create_volatility_chart(df, target_col, window=7, title="Volatilitas"):
    if df is None or df.empty or target_col not in df.columns:
        return None
    
    df_vol = df.copy()
    if 'tanggal' not in df_vol.columns and isinstance(df_vol.index, pd.DatetimeIndex):
         df_vol = df_vol.reset_index()

    # Calculate Rolling Std Dev
    df_vol['volatility'] = df_vol[target_col].rolling(window=window).std()
    
    fig = px.line(df_vol, x='tanggal', y='volatility', 
                  title=title, 
                  labels={'volatility': f'Std Dev ({window} Hari)'},
                  template='plotly_white')
    fig.update_traces(line_color='#FF6F61')
    return fig

# Feature: Inventory Cover
def create_inventory_cover_chart(df_main):
    if df_main is None or df_main.empty: return None
    
    df_calc = df_main.copy().sort_values('tanggal')
    # Rolling avg of outcome (7 days)
    df_calc['avg_out_7d'] = df_calc['keluar'].rolling(window=7, min_periods=1).mean()
    df_calc['avg_out_7d'] = df_calc['avg_out_7d'].replace(0, 1) # Avoid div by zero
    
    df_calc['cover_days'] = df_calc['stok'] / df_calc['avg_out_7d']
    
    fig = px.line(df_calc, x='tanggal', y='cover_days',
                  title="Inventory Cover Days (Ketahanan Stok)",
                  labels={'cover_days': 'Hari'},
                  template='plotly_white')
    fig.add_hline(y=20, line_dash="dot", annotation_text="Aman (20 hari)", annotation_position="bottom right")
    return fig

# Feature: Geospatial Maps
def create_geo_map(df_flow, geo_lookup, flow_type='masuk'):
    if df_flow is None or df_flow.empty: return None
    
    # Aggregate flow by location
    df_agg = df_flow.groupby('lokasi')[flow_type].sum().reset_index()
    
    # Merge with Lat/Lon
    df_map = pd.merge(df_agg, geo_lookup, on='lokasi', how='inner')
    
    if df_map.empty: return None
    
    title = "Peta Asal Beras (Masuk)" if flow_type == 'masuk' else "Peta Distribusi Beras (Keluar)"
    color_scale = "Greens" if flow_type == 'masuk' else "Reds"
    
    fig = px.scatter_mapbox(
        df_map, lat="lat", lon="lon", size=flow_type, color=flow_type,
        hover_name="lokasi", hover_data=[flow_type],
        title=title,
        color_continuous_scale=color_scale,
        zoom=5, center={"lat": -6.8, "lon": 108},
        mapbox_style="open-street-map"
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig

# Feature: Stock Distribution (KDE)
def create_stock_distribution(df_stock):
    if df_stock is None or df_stock.empty: return None
    
    x = df_stock['stok'].dropna()
    hist_data = [x]
    group_labels = ['Distribusi Stok']
    
    try:
        fig = ff.create_distplot(hist_data, group_labels, show_hist=True, show_rug=False, curve_type='kde')
        fig.update_layout(title="Distribusi Level Stok (Histogram & KDE)", template='plotly_white')
        return fig
    except Exception as e:
        st.warning(f"Gagal membuat plot distribusi: {e}")
        return None

# Feature: Regression Stats
def calculate_regression(df_stock, df_price, rice_type):
    if df_stock is None or df_price is None: return None
    
    # Merge
    df_p = price_df_with_tanggal(df_price)
    df_merge = pd.merge(df_stock, df_p[['tanggal', rice_type]], on='tanggal', how='inner')
    df_merge.dropna(inplace=True)
    
    if len(df_merge) < 2: return None
    
    slope, intercept, r_value, p_value, std_err = linregress(df_merge['stok'], df_merge[rice_type])
    
    return {
        'slope': slope,
        'r2': r_value**2,
        'p_value': p_value,
        'df': df_merge
    }


# -------------------------
# 4. UI Rendering
# -------------------------

def render_sidebar():
    with st.sidebar:
        st.title("🌾 PIBC Explorer")
        
        if st.session_state.get('data_loaded'):
            if st.button("🔄 Reset Data"):
                st.session_state.data_loaded = False
                st.session_state.app_data = {}
                st.rerun()
            return

        source = option_menu("Sumber Data", ["Upload Excel", "Database"], icons=['file-earmark-spreadsheet', 'database'])
        
        if source == "Upload Excel":
            f = st.file_uploader("Upload File (.xlsx)", type=["xlsx"])
            if f:
                df_main, df_stock, df_masuk, df_keluar, df_price = preprocess_data_from_excel(f)
                if df_main is not None:
                    st.session_state.data_loaded = True
                    st.session_state.app_data = {
                        'df': df_main, 'df_stock': df_stock, 
                        'df_masuk': df_masuk, 'df_keluar': df_keluar, 
                        'df_price': df_price
                    }
                    st.rerun()
        else:
            if st.button("Connect DB"):
                eng = init_connection()
                if eng:
                    df_main, df_stock, df_masuk, df_keluar, df_price = load_data_from_db(eng)
                    if df_main is not None:
                        st.session_state.data_loaded = True
                        st.session_state.app_data = {
                            'df': df_main, 'df_stock': df_stock, 
                            'df_masuk': df_masuk, 'df_keluar': df_keluar, 
                            'df_price': df_price
                        }
                        st.rerun()

def render_metrics(df_filtered):
    if df_filtered is None or df_filtered.empty: return
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rata-rata Stok", f"{df_filtered['stok'].mean():,.0f} Ton")
    col2.metric("Total Masuk", f"{df_filtered['masuk'].sum():,.0f} Ton")
    col3.metric("Total Keluar", f"{df_filtered['keluar'].sum():,.0f} Ton")
    col4.metric("Net Neraca", f"{df_filtered['neraca'].sum():,.0f} Ton")

def render_main_ui():
    app_data = st.session_state.app_data
    df = app_data['df']
    df_price = app_data['df_price']
    df_masuk = app_data['df_masuk']
    df_keluar = app_data['df_keluar']

    # --- Sidebar Filters ---
    with st.sidebar:
        st.divider()
        st.subheader("Filter Dashboard")
        min_date, max_date = df['tanggal'].min(), df['tanggal'].max()
        start_date = st.date_input("Mulai", min_date)
        end_date = st.date_input("Sampai", max_date)
        
        rice_opts = list(df_price.columns) if df_price is not None else []
        selected_rice = st.selectbox("Jenis Beras (untuk Harga)", rice_opts) if rice_opts else None

    # --- Filtering ---
    mask = (df['tanggal'].dt.date >= start_date) & (df['tanggal'].dt.date <= end_date)
    df_filt = df[mask]
    
    df_masuk_filt = df_masuk[(df_masuk['tanggal'].dt.date >= start_date) & (df_masuk['tanggal'].dt.date <= end_date)] if df_masuk is not None and not df_masuk.empty else pd.DataFrame()
    df_keluar_filt = df_keluar[(df_keluar['tanggal'].dt.date >= start_date) & (df_keluar['tanggal'].dt.date <= end_date)] if df_keluar is not None and not df_keluar.empty else pd.DataFrame()
    
    df_price_filt = None
    if df_price is not None:
        df_price_filt = df_price[(df_price.index.date >= start_date) & (df_price.index.date <= end_date)]

    # --- UI Layout ---
    render_metrics(df_filt)

    # Tabs Definition (Added new tabs for Notebook features)
    tabs = st.tabs([
        "📈 Dashboard Utama", 
        "🗺️ Peta Geografis", 
        "🔍 Analisis Lanjutan", 
        "📊 Statistik & Regresi", 
        "🔮 Peramalan"
    ])

    # --- TAB 1: Dashboard Utama ---
    with tabs[0]:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(create_time_series(df_filt, 'stok', "Tren Stok Harian", "green"), use_container_width=True)
        with col_b:
            st.plotly_chart(create_balance_chart(df_filt), use_container_width=True)
        
        if selected_rice and df_price_filt is not None:
            # Preprocess price data for plotting
            p_plot = df_price_filt[[selected_rice]].reset_index()
            p_plot.columns = ['tanggal', selected_rice]
            st.plotly_chart(create_time_series(p_plot, selected_rice, f"Tren Harga: {selected_rice}", "blue"), use_container_width=True)

    # --- TAB 2: Peta Geografis (Feature from Notebook) ---
    with tabs[1]:
        st.subheader("Analisis Distribusi Geospasial")
        geo_lookup = get_geo_lookup()
        
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Peta Asal Barang (Masuk)**")
            fig_map_in = create_geo_map(df_masuk_filt, geo_lookup, 'masuk')
            if fig_map_in: st.plotly_chart(fig_map_in, use_container_width=True)
            else: st.info("Data lokasi masuk tidak tersedia.")
            
        with c2:
            st.write("**Peta Tujuan Barang (Keluar)**")
            fig_map_out = create_geo_map(df_keluar_filt, geo_lookup, 'keluar')
            if fig_map_out: st.plotly_chart(fig_map_out, use_container_width=True)
            else: st.info("Data lokasi keluar tidak tersedia.")
            
        with st.expander("Lihat Data Tabel Distribusi"):
            if not df_masuk_filt.empty:
                st.write("Top Supply (Masuk):")
                st.dataframe(df_masuk_filt.groupby('lokasi')['masuk'].sum().sort_values(ascending=False).head())

    # --- TAB 3: Analisis Lanjutan (Volatility & Inventory Cover) ---
    with tabs[2]:
        st.subheader("Analisis Kestabilan & Efisiensi")
        
        # 1. Inventory Cover
        st.markdown("##### 1. Inventory Cover Days")
        st.caption("Berapa hari stok saat ini mampu menutupi rata-rata permintaan keluar (Rolling 7 hari)?")
        fig_cover = create_inventory_cover_chart(df_filt)
        if fig_cover: st.plotly_chart(fig_cover, use_container_width=True)
        
        # 2. Volatility
        st.markdown("##### 2. Volatilitas Stok & Harga")
        c_vol1, c_vol2 = st.columns(2)
        with c_vol1:
            fig_v_stok = create_volatility_chart(df_filt, 'stok', window=30, title="Volatilitas Stok (30 Hari)")
            if fig_v_stok: st.plotly_chart(fig_v_stok, use_container_width=True)
            
        with c_vol2:
            if df_price_filt is not None and selected_rice:
                p_reset = df_price_filt[[selected_rice]].reset_index().rename(columns={df_price_filt.index.name: 'tanggal'})
                fig_v_price = create_volatility_chart(p_reset, selected_rice, window=7, title=f"Volatilitas Harga {selected_rice} (7 Hari)")
                if fig_v_price: st.plotly_chart(fig_v_price, use_container_width=True)

        # 3. Heatmap
        st.markdown("##### 3. Korelasi Antar Harga Beras")
        if df_price_filt is not None:
            fig_corr = create_price_heatmap(df_price_filt)
            if fig_corr: st.plotly_chart(fig_corr, use_container_width=True)

    # --- TAB 4: Statistik & Regresi (Regression Stats & Distribution) ---
    with tabs[3]:
        st.subheader("Analisis Statistik Mendalam")
        
        col_stat1, col_stat2 = st.columns([1, 1])
        
        with col_stat1:
            st.markdown("#### Distribusi Stok")
            fig_dist = create_stock_distribution(df_filt)
            if fig_dist: st.plotly_chart(fig_dist, use_container_width=True)
            
            st.write("Statistik Deskriptif Stok:")
            desc = df_filt['stok'].describe()
            st.dataframe(desc)

        with col_stat2:
            st.markdown(f"#### Regresi: Stok vs Harga ({selected_rice})")
            if selected_rice:
                reg_res = calculate_regression(df_filt, df_price, selected_rice)
                if reg_res:
                    st.info(f"Formula: Harga = {reg_res['slope']:.2f} * Stok + Intercept")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Slope", f"{reg_res['slope']:.2f}")
                    m2.metric("R-Squared", f"{reg_res['r2']:.4f}")
                    m3.metric("P-Value", f"{reg_res['p_value']:.4f}")
                    
                    if reg_res['p_value'] < 0.05:
                        st.success("Hubungan Signifikan (P < 0.05)")
                    else:
                        st.warning("Hubungan Tidak Signifikan")
                        
                    # Scatter Plot
                    fig_reg = px.scatter(reg_res['df'], x='stok', y=selected_rice, trendline='ols', title="Scatter Plot Regresi")
                    st.plotly_chart(fig_reg, use_container_width=True)

                    st.markdown("---")
                    csv_reg = convert_df_to_csv(reg_res['df'])

                    st.download_button(
                        label=" Download Data Hasil Regresi (CSV)",
                        data=csv_reg,
                        file_name=f'analisis_regresi_{selected_rice}.csv',
                        mime='text/csv',
                        key='download_reg'
                    )
                else:
                    st.warning("Data tidak cukup untuk regresi.")

    # --- TAB 5: Peramalan (Existing Feature) ---
    with tabs[4]:
        st.subheader("Peramalan Stok (Forecasting)")
        if len(df_filt) > 10:
            method = st.radio("Metode", ["Prophet", "Holt-Winters"], horizontal=True)
            days = st.slider("Horizon Hari", 7, 90, 30)
            
            df_fc = df_filt[['tanggal', 'stok']].rename(columns={'tanggal':'ds', 'stok':'y'})
            
            if method == "Prophet":
                m = Prophet()
                m.fit(df_fc)
                future = m.make_future_dataframe(periods=days)
                forecast = m.predict(future)
                
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=df_fc['ds'], y=df_fc['y'], name='Historis'))
                fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
                fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines', line_color='lightgrey', showlegend=False))
                fig_fc.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='lightgrey', name='Confidence Interval'))
                st.plotly_chart(fig_fc, use_container_width=True)

                output_prophet = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                    columns={'ds':'tanggal', 'yhat':'prediksi', 'yhat_lower': 'Batas Bawah', 'yhat_upper': 'Batas Atas'}    
                )
                csv_fc = convert_df_to_csv(output_prophet)

                st.download_button(
                    label=" Download Data Peramalan (CSV)",
                    data=csv_fc,
                    file_name='hasil_preamalan_prophet.csv',
                    mime='text/csv',
                    key='download_fc'
                )

            else: # Holt Winters
                try:
                    model = ExponentialSmoothing(df_fc['y'], seasonal='add', seasonal_periods=7).fit()
                    pred = model.forecast(days)
                    
                    # Create date range for forecast
                    last_date = df_fc['ds'].iloc[-1]
                    date_range = pd.date_range(last_date, periods=days+1)[1:]
                    
                    fig_hw = go.Figure()
                    fig_hw.add_trace(go.Scatter(x=df_fc['ds'], y=df_fc['y'], name='Historis'))
                    fig_hw.add_trace(go.Scatter(x=date_range, y=pred, name='Forecast (HW)'))
                    st.plotly_chart(fig_hw, use_container_width=True)
                except Exception as e:
                    st.error(f"Error Holt-Winters: {e}")
        else:
            st.warning("Data terlalu sedikit untuk peramalan.")

# --- Main Entry Point ---
def main():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.app_data = {}

    if not st.session_state.data_loaded:
        render_sidebar()
        st.info("👈 Silakan Upload Excel atau Koneksi Database di Sidebar.")
        
        # Optional: Landing Page Dummy Data visualization logic could go here
        st.markdown("<div class='title'>Selamat Datang di Dashboard Analisis Beras</div>", unsafe_allow_html=True)
    else:
        render_main_ui()

if __name__ == "__main__":
    main()