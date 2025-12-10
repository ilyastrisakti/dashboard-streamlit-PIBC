# -*- coding: utf-8 -*-
"""
PIBC Explorer final
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
from lib.forecast import run_prophet_forecast, run_holtwinters_forecast
from lib.plots import create_forecast_chart
from sqlalchemy import create_engine
from streamlit_option_menu import option_menu
from streamlit_plotly_events import plotly_events
from lib.constants import *
from lib.utils import price_df_with_tanggal, convert_df_to_csv, calculate_regression
from lib.data import (
    get_geo_lookup,
    init_connection,
    load_data_from_db,
    preprocess_data_from_excel
)
from lib.bussiness_logic import (
    filter_and_aggregate_data
)
from lib.plots import (
    DEFAULT_PLOTLY_TEMPLATE,
    create_time_series,
    create_balance_chart,
    create_price_heatmap,
    create_volatility_chart,
    create_regression_scatter,
    create_inventory_cover_chart,
    create_stock_distribution, # create_regression_scatter is not used directly, but calculate_regression is
    create_geo_map
)

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

/* Dark theme tab overrides */
[data-theme="dark"] .stTabs [data-baseweb="tab"] {
    background-color: #0f1720 !important;
    color: #cfeaf8 !important;
    border-radius: 6px 6px 0 0;
}
[data-theme="dark"] .stTabs [aria-selected="true"] { background-color: #164f6b !important; color: #fff !important; }

/* Guide / info boxes used across UI (works in both light & dark) */
.guide-box {
    background-color: #f0f2f6; 
    color: #0b1a24;
    padding: 12px;
    border-radius: 8px;
    font-size: 14px;
    border: 1px solid rgba(0,0,0,0.06);
}
[data-theme="dark"] .guide-box {
    background-color: #071025 !important;
    color: #E6F2FF !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

/* Slight text boost for small UI pieces in dark mode */
[data-theme="dark"] .stSmall, [data-theme="dark"] .stCaption {
    color: #dbeeff !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# Choose plotly template based on Streamlit theme
DEFAULT_PLOTLY_TEMPLATE = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"

# -------------------------
# 3. Visualization Logic
# -------------------------

# -------------------------
# 4. UI Rendering
# -------------------------

def render_sidebar():
    with st.sidebar:
        st.title("🌾 PIBC Explorer")
        
        if st.session_state.get('data_loaded'):
            if st.button("🔄 Reset Data / Koneksi Baru"):
                st.session_state.data_loaded = False
                st.session_state.app_data = {}
                st.rerun()
            return

        source = option_menu("Sumber Data", ["Upload Excel", "Database"], icons=['file-earmark-spreadsheet', 'database'])
        
        if source == "Upload Excel":
            f = st.file_uploader("Upload File (.xlsx)", type=["xlsx"])
            if f:
                result = preprocess_data_from_excel(f)
                if result is None:
                    st.error("Gagal memproses file. Periksa struktur file (.xlsx) dan sheet yang diperlukan.")
                else:
                    df_main, df_stock, df_masuk, df_keluar, df_price = result
                    if df_main is not None:
                        st.session_state.data_loaded = True
                        st.session_state.app_data = {
                            'df': df_main, COL_STOK: df_stock, 
                            COL_MASUK: df_masuk, COL_KELUAR: df_keluar, 
                            'df_price': df_price # df_price is special
                        }
                        st.rerun()
        else:
            st.markdown("## Koneksi Database")

            st.caption("Filter data di awal (Server-side) untuk performa lebih cepat:")
            col_d1, col_d2 = st.columns(2)
            # Default: None (Load All) atau set default (1 tahun terakhir)
            db_start_date = col_d1.date_input("Dari", value=None)
            db_end_date = col_d2.date_input("Sampai", value=None)

            st.caption("Gunakan st.secrets jika tersedia, atau isi form manual di bawah.")

            # Try using st.secrets first (button)
            try:
                secret_defaults = st.secrets.get("connections", {}).get("mysql_db", {})
            except Exception:
                secret_defaults = {}

            if st.button("Connect using st.secrets"):
                eng = init_connection()
                if eng is None:
                    st.error("st.secrets tidak ditemukan / konfigurasi salah. Coba koneksi manual di bawah.")
                else:
                    with st.spinner("Menghubungkan via st.secrets..."):
                        result = load_data_from_db(eng)
                        if result is None:
                            st.error("Gagal memuat data dari database. Periksa konfigurasi atau izin DB.")
                        else:
                            df_main, df_stock, df_masuk, df_keluar, df_price = result
                            if df_main is not None:
                                st.session_state.data_loaded = True
                                st.session_state.app_data = {
                                    'df': df_main, COL_STOK: df_stock,
                                    COL_MASUK: df_masuk, COL_KELUAR: df_keluar,
                                    'df_price': df_price
                                }
                                st.success("Connected & data loaded from DB.")
                                st.rerun()

            st.markdown("---")
            st.markdown("### Manual connection (jika st.secrets tidak dipakai)")
            host = st.text_input("Host", value=secret_defaults.get("host", "localhost"))
            port = st.text_input("Port", value=str(secret_defaults.get("port", "3306")))
            user = st.text_input("User", value=secret_defaults.get("username", ""), help="DB username")
            password = st.text_input("Password", value=secret_defaults.get("password", ""), type="password")
            database = st.text_input("Database", value=secret_defaults.get("database", ""))

            if st.button("Connect (Manual)"):
                if all([host, port, user, database]):
                    conn_str = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
                    try:
                        eng_manual = create_engine(conn_str)
                        with st.spinner("Mengambil data..."):
                            # PANGGIL DENGAN FILTER TANGGAL
                            result = load_data_from_db(eng_manual, start_date=db_start_date, end_date=db_end_date)
                            if result and result[0] is not None:
                                df_main, df_stock, df_masuk, df_keluar, df_price = result
                                st.session_state.data_loaded = True
                                st.session_state.app_data = {
                                    'df': df_main, COL_STOK: df_stock,
                                    COL_MASUK: df_masuk, COL_KELUAR: df_keluar,
                                    'df_price': df_price
                                }
                                st.rerun()
                    except Exception as e:
                        st.error(f"Koneksi Gagal: {e}")

def render_metrics(df_filtered):
    if df_filtered is None or df_filtered.empty: return
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Tambahkan parameter 'help' untuk tooltip
    col1.metric("Rata-rata Stok", f"{df_filtered[COL_STOK].mean():,.0f} Ton", 
                help="Rata-rata stok harian yang tersedia di gudang dalam periode waktu yang dipilih.")
                
    col2.metric("Total Masuk", f"{df_filtered[COL_MASUK].sum():,.0f} Ton", 
                help="Total akumulasi volume beras yang masuk ke pasar.")
                
    col3.metric("Total Keluar", f"{df_filtered[COL_KELUAR].sum():,.0f} Ton", 
                help="Total akumulasi volume beras yang didistribusikan keluar pasar.")
                
    col4.metric("Net Neraca", f"{df_filtered[COL_NERACA].sum():,.0f} Ton", 
                help="Selisih antara Total Masuk dikurangi Total Keluar. Positif = Surplus, Negatif = Defisit.")

def handle_drilldown(points):
    """Handles the logic when a user clicks a point on a chart for drill-down."""
    if not points:
        return

    # Get the date from the first clicked point
    clicked_date = pd.to_datetime(points[0]["x"])

    # Store the drill-down context in session state
    st.session_state.drilldown_context = {
        "date": clicked_date,
        "from_granularity": st.session_state.get('granularity_selector', 'Harian')
    }

def reset_date_filters():
    """Resets date filters in session state, forcing them to use data min/max."""
    if 'start_date_filter' in st.session_state:
        del st.session_state['start_date_filter']
    if 'end_date_filter' in st.session_state:
        del st.session_state['end_date_filter']
def render_main_ui():
    app_data = st.session_state.app_data
    df = app_data['df']
    df_price = app_data.get('df_price') # Use .get() for safety
    df_masuk = app_data[COL_MASUK]
    df_keluar = app_data[COL_KELUAR]

    # --- Handle Drill-down State Change ---
    if 'drilldown_context' in st.session_state and st.session_state.drilldown_context:
        context = st.session_state.drilldown_context
        from_granularity = context['from_granularity']
        clicked_date = context['date']

        if from_granularity == "Tahunan":
            st.session_state.granularity_selector = "Bulanan"
            # Set date range for the entire selected year
            st.session_state.start_date_filter = pd.Timestamp(year=clicked_date.year, month=1, day=1).date()
            st.session_state.end_date_filter = pd.Timestamp(year=clicked_date.year, month=12, day=31).date()
        elif from_granularity == "Bulanan":
            st.session_state.granularity_selector = "Harian"
            # Set date range for the entire selected month
            st.session_state.start_date_filter = clicked_date.replace(day=1).date()
            st.session_state.end_date_filter = (clicked_date + pd.offsets.MonthEnd(0)).date()
        
        # Clear the context to prevent re-triggering
        st.session_state.drilldown_context = None


    # --- Sidebar Filters ---
    with st.sidebar:
        st.divider()
        st.subheader("Filter Dashboard")
        min_ts = pd.to_datetime(df['tanggal'].min(), errors='coerce')
        max_ts = pd.to_datetime(df['tanggal'].max(), errors='coerce')
        min_date, max_date = (min_ts.date() if pd.notna(min_ts) else None, max_ts.date() if pd.notna(max_ts) else None)
        start_date = st.date_input("Mulai", value=st.session_state.get('start_date_filter', min_date), key='start_date_filter')
        end_date = st.date_input("Sampai", value=st.session_state.get('end_date_filter', max_date), key='end_date_filter')
        
        rice_opts = list(df_price.columns) if df_price is not None else []
        selected_rice = st.selectbox("Jenis Beras (untuk Harga)", rice_opts) if rice_opts else None

        st.divider()
        granularity = st.radio("Tingkat Agregasi Data", ["Harian", "Bulanan", "Tahunan"], key="granularity_selector", help="Klik pada bar/titik di grafik utama untuk melakukan 'drill-down' ke level yang lebih detail.", on_change=reset_date_filters)
            
        if granularity != "Harian":
            st.caption(f"💡 Filter tanggal akan diterapkan pada level {granularity.lower()}.")

    is_daily_view = (granularity == "Harian")

    # --- Performant Data Processing ---
    # Call the new cached function to get the aggregated data
    df_agg = filter_and_aggregate_data(df, start_date, end_date, granularity)

    # Filter geo and price data (these are usually smaller and less costly)
    df_masuk_filt = df_masuk[(df_masuk[COL_TANGGAL].dt.date >= start_date) & (df_masuk[COL_TANGGAL].dt.date <= end_date)] if df_masuk is not None and not df_masuk.empty else pd.DataFrame()
    df_keluar_filt = df_keluar[(df_keluar[COL_TANGGAL].dt.date >= start_date) & (df_keluar[COL_TANGGAL].dt.date <= end_date)] if df_keluar is not None and not df_keluar.empty else pd.DataFrame()
    
    df_price_filt = None
    if df_price is not None:
        # Ensure df_price index is timezone-naive before comparing with date objects
        if df_price.index.tz is not None:
            price_index_date = df_price.index.tz_localize(None).date
        else:
            price_index_date = df_price.index.date
        df_price_filt = df_price[(price_index_date >= start_date) & (price_index_date <= end_date)]

    # --- UI Layout ---
    render_metrics(df_agg)

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

        # Penjelasan Konsep Dasar Dashboard
        with st.expander("ℹ️ Panduan Membaca Dashboard Utama (Klik untuk buka)", expanded=True):
            st.markdown("""
            * **Tren Stok Harian (Kiri):** Garis ini menunjukkan riwayat ketersediaan beras di gudang. 
                * *Naik* = Penumpukan stok. *Turun* = Stok menipis (permintaan tinggi/pasokan kurang).
            * **Neraca Harian (Kanan):** Membandingkan arus barang.
                * **Batang Biru (Masuk):** Supply dari daerah.
                * **Batang Merah (Keluar):** Distribusi ke pasar.
                * **Titik Hitam (Net):** Surplus (di atas 0) atau Defisit (di bawah 0).
            """)
        
        col_a, col_b = st.columns(2)
        with col_a:
            fig_stok = create_time_series(df_agg, COL_STOK, f"Tren Stok {granularity}", "green", is_daily=is_daily_view)
            # Always use plotly_events to maintain state across tab switches.
            # The component can handle an empty figure. override_height should be int or None.
            # Setting it to a default int value if None is not explicitly allowed by the component.
            clicked_points_stok = plotly_events(fig_stok, click_event=bool(fig_stok.data), key="stok_events", override_height=400, override_width="100%")
            handle_drilldown(clicked_points_stok)

        with col_b:
            fig_balance = create_balance_chart(df_agg, granularity=granularity)
            clicked_points_balance = plotly_events(fig_balance, click_event=bool(fig_balance.data), key="balance_events", override_height=400, override_width="100%")
            handle_drilldown(clicked_points_balance)

        st.divider()

        if selected_rice:
            st.markdown(f"#### Tren Harga: {selected_rice}")
            st.markdown("""
            <div class="guide-box">
            Grafik ini menunjukkan pergerakan harga historis untuk jenis beras yang Anda pilih di sidebar. Gunakan ini untuk mengidentifikasi periode inflasi (kenaikan harga), deflasi (penurunan harga), dan membandingkannya secara visual dengan tren stok dan neraca di atas.
            </div>
            """, unsafe_allow_html=True)

        if selected_rice and df_price_filt is not None and not df_price_filt.empty:
            # Aggregate price data to match the main granularity
            if not is_daily_view:
                resample_rule = 'M' if granularity == "Bulanan" else 'Y'
                df_price_agg = df_price_filt[[selected_rice]].resample(resample_rule).mean()
            else:
                df_price_agg = df_price_filt[[selected_rice]]
            
            if not df_price_agg.empty:
                p_plot = df_price_agg.reset_index()
                p_plot = p_plot.rename(columns={'index': COL_TANGGAL})
                st.plotly_chart(create_time_series(p_plot, selected_rice, f"Tren Harga {granularity}: {selected_rice}", "blue", is_daily=is_daily_view), use_container_width=True, key="main_price_ts")

    # --- TAB 2: Peta Geografis (Feature from Notebook) ---
    with tabs[1]:
        if not is_daily_view:
            st.info(
                "ℹ️ **Catatan:** Tampilan Peta Geografis selalu menggunakan data **Harian** untuk menunjukkan "
                "detail lokasi asal/tujuan, terlepas dari pilihan agregasi di sidebar."
            )

        st.subheader("Analisis Distribusi Geospasial")
        
        # Panduan membaca peta
        st.markdown("""
            <div class="guide-box">
            ℹ️ <b>Panduan Peta:</b><br>
            • <b>Lingkaran Besar:</b> Menandakan volume tonase beras yang lebih besar.<br>
            • <b>Warna Pekat:</b> Konsentrasi tinggi pada satu titik lokasi.<br>
            • <b>Interaksi:</b> Arahkan mouse ke lingkaran untuk melihat detail angka Tonase.
            </div>
            <br>
            """, unsafe_allow_html=True)
        
        geo_lookup = get_geo_lookup()
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Peta Asal Barang (Masuk)**")
            fig_map_in = create_geo_map(df_masuk_filt, geo_lookup, COL_MASUK)
            if fig_map_in: st.plotly_chart(fig_map_in, use_container_width=True, key="geo_map_in")
            else: st.info("Data lokasi masuk tidak tersedia.")
            
        with c2:
            st.write("**Peta Tujuan Barang (Keluar)**")
            fig_map_out = create_geo_map(df_keluar_filt, geo_lookup, COL_KELUAR)
            if fig_map_out: st.plotly_chart(fig_map_out, use_container_width=True, key="geo_map_out")
            else: st.info("Data lokasi keluar tidak tersedia.")
            
        with st.expander("Lihat Data Tabel Distribusi"):
            if not df_masuk_filt.empty:
                st.write("Top Supply (Masuk):")
                st.dataframe(df_masuk_filt.groupby(COL_LOKASI)[COL_MASUK].sum().sort_values(ascending=False).head())

    # --- TAB 3: Analisis Lanjutan (Volatility & Inventory Cover) ---
    with tabs[2]:
        st.subheader("Analisis Kestabilan & Efisiensi")
        
        if is_daily_view:
            # 1. Inventory Cover
            st.markdown("##### 1. Inventory Cover Days")

            # Expander untuk penjelasan rumus
            with st.expander("📖 Cara Membaca Inventory Cover (Klik disini)"):
                st.markdown("""
                **Definisi:** Estimasi berapa hari stok saat ini akan bertahan jika tidak ada pasokan baru.
                
                $$
                \\text{Cover Days} = \\frac{\\text{Stok Hari Ini}}{\\text{Rata-rata Keluar (7 hari terakhir)}}
                $$
                
                **Panduan Indikator:**
                * 🟢 **> 20 Hari:** Stok Aman.
                * 🟡 **10 - 20 Hari:** Waspada.
                * 🔴 **< 10 Hari:** Kritis (Risiko kelangkaan tinggi).
                """)

            st.caption("Berapa hari stok saat ini mampu menutupi rata-rata permintaan keluar (Rolling 7 hari)?")
            fig_cover = create_inventory_cover_chart(df_agg, days_cover=20) # Match original logic
            if fig_cover: st.plotly_chart(fig_cover, use_container_width=True, key="inventory_cover_chart")
            
            # 2. Volatility
            st.markdown("##### 2. Volatilitas Stok & Harga")
            st.info("""
            **Apa itu Volatilitas?**
            Ini mengukur "kepanikan" pasar. Grafik yang **tinggi** menunjukkan harga/stok berubah-ubah secara drastis (tidak stabil) dalam waktu singkat.
            Grafik yang **rendah/datar** menunjukkan kondisi pasar yang tenang dan stabil.
            """)

            c_vol1, c_vol2 = st.columns(2)
            with c_vol1:
                fig_v_stok = create_volatility_chart(df_agg[[COL_TANGGAL, COL_STOK]], target_col=COL_STOK, window=30, title="Volatilitas Stok (30 Hari)")
                if fig_v_stok: st.plotly_chart(fig_v_stok, use_container_width=True, key="volatility_stok_chart")
                
            with c_vol2:
                if df_price_filt is not None and selected_rice:
                    p_reset = df_price_filt[[selected_rice]].reset_index().rename(columns={df_price_filt.index.name: COL_TANGGAL})
                    fig_v_price = create_volatility_chart(p_reset, target_col=selected_rice, title=f"Volatilitas Harga {selected_rice} (7 Hari)", window=7)
                    if fig_v_price: st.plotly_chart(fig_v_price, use_container_width=True, key="volatility_price_chart")
        else:
            st.warning(
                f"Analisis **Inventory Cover** dan **Volatility** tidak tersedia untuk tampilan **{granularity}** "
                "karena metrik ini dirancang untuk analisis harian."
            )

        st.divider()

        # 3. Heatmap
        st.markdown("### 3. Korelasi Antar Jenis Beras")
        with st.expander("📖 Cara Membaca Heatmap (Matriks Warna)"):
                st.markdown("""
                Matriks ini menunjukkan hubungan pergerakan harga antar jenis beras:
                * **Warna Hijau Tua (Mendekati 1.0):** Hubungan Kuat & Searah. Jika Beras A naik, Beras B **pasti ikut naik**. (Contoh: IR-64 I dan IR-64 II).
                * **Warna Merah (Mendekati -1.0):** Hubungan Terbalik. Jika Beras A naik, Beras B justru turun.
                * **Warna Pucat (Mendekati 0):** Tidak ada hubungan. Pergerakan harga mereka tidak saling mempengaruhi.
                """)
            
        if df_price_filt is not None:
                fig_corr = create_price_heatmap(df_price_filt, correlation=True) # Use correlation heatmap
                if fig_corr: st.plotly_chart(fig_corr, use_container_width=True, key="price_correlation_heatmap")

    # --- TAB 4: Statistik & Regresi (Regression Stats & Distribution) ---
    with tabs[3]:
        st.subheader("Analisis Statistik Mendalam")
        
        col_stat1, col_stat2 = st.columns([1, 1])
        
        with col_stat1:
            st.markdown("#### Distribusi Stok")
            # Penjelasan Ditribusi
            st.caption("""
            Grafik ini menjawab: **"Berapa level stok yang paling sering terjadi (Normal)?"**
            * **Puncak Gunung (Modus):** Menunjukkan level stok yang paling sering terjadi sehari-hari.
            * **Lebar Gunung:** Menunjukkan variasi stok. Semakin lebar, semakin tidak pasti ketersediaan stok di gudang.
            """)

            fig_dist = create_stock_distribution(df_agg)
            if fig_dist: st.plotly_chart(fig_dist, use_container_width=True, key="stock_distribution_chart")
            
            st.markdown("---")
            st.markdown("#### Statistik Deskriptif Stok")
            desc = df_agg[COL_STOK].describe()
            st.dataframe(desc)
            with st.expander("📖 Panduan Membaca Tabel Statistik Deskriptif"):
                st.markdown("""
                * **count:** Jumlah total periode (hari/bulan/tahun) yang dianalisis.
                * **mean (Rata-rata):** Level stok rata-rata selama periode tersebut. Ini adalah gambaran umum ketersediaan.
                * **std (Standar Deviasi):** Ukuran volatilitas stok. Angka yang **tinggi** berarti level stok sangat bervariasi (tidak stabil). Angka yang **rendah** berarti stok cenderung konsisten.
                * **min & max:** Level stok terendah dan tertinggi yang pernah tercatat dalam periode waktu yang dipilih.
                * **25%, 50% (Median), 75%:** Kuartil yang membagi data. Contohnya, '50%' (median) adalah nilai tengah; 50% dari waktu, stok berada di bawah angka ini.
                """)

        with col_stat2:
            st.markdown(f"#### Regresi: Stok vs Harga ({selected_rice})")
            if selected_rice:
                reg_res = calculate_regression(df_agg, df_price, selected_rice)
                if reg_res:
                    # Penjelasan interpretasi hasil
                    st.info(f"💡 **Interpretasi:** Setiap stok bertambah **1 Ton**, harga diprediksi berubah sebesar **Rp {reg_res['slope']:.2f}**.")
                    
                    with st.expander("🔍 Penjelasan Istilah Statistik"):
                        st.markdown("""
                        * **R-Squared (0-1):** Seberapa kuat stok mempengaruhi harga? (Makin dekat ke 1, makin kuat).
                        * **P-Value:** Tingkat kepercayaan. Jika angka ini **< 0.05**, berarti hubungan stok & harga adalah **Nyata (Signifikan)**, bukan kebetulan.
                        """)
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
                    fig_reg = px.scatter(
                        reg_res['df'], x=COL_STOK, y=selected_rice, 
                        trendline='ols', 
                        title="Scatter Plot Regresi",
                        template=DEFAULT_PLOTLY_TEMPLATE
                    )
                    st.plotly_chart(fig_reg, use_container_width=True, key="regression_scatter_plot")

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

   # --- TAB 5: Peramalan (Refactored) ---
    with tabs[4]:
        st.subheader("Peramalan Stok (Forecasting)")

        st.success("""
        🤖 **Tips Memilih Algoritma:**
        * Pilih **Prophet** jika data Anda memiliki tren jangka panjang yang kuat atau banyak data libur nasional.
        * Pilih **Holt-Winters** jika pola data Anda berulang secara mingguan/bulanan yang sangat teratur.
        """)

        if len(df_agg) > 10:
            col_met, col_hor = st.columns([1, 2])
            with col_met:
                method = st.radio("Metode", ["Prophet", "Holt-Winters"], horizontal=True)
            with col_hor:
                horizon_label = f"Horizon Peramalan ({granularity.replace('an', '')})"
                days = st.slider(horizon_label, 7, 90, 30)

            # --- LOGIKA BARU MENGGUNAKAN LIB/FORECAST.PY ---
            if method == "Prophet":
                with st.spinner("Menjalankan model Prophet..."):
                    df_hist, df_pred = run_prophet_forecast(df_agg, days)
                
                # Plotting menggunakan lib/plots.py
                fig_fc = create_forecast_chart(df_hist, df_pred, method="Prophet")
                st.plotly_chart(fig_fc, use_container_width=True, key="prophet_forecast_chart")

                # Persiapan Data Download
                output_csv = df_pred[[FC_COL_DS, 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                    columns={FC_COL_DS:'tanggal', 'yhat':'prediksi', 'yhat_lower': 'Batas Bawah', 'yhat_upper': 'Batas Atas'}    
                )
                csv_fc = convert_df_to_csv(output_csv)
                fname = 'hasil_peramalan_prophet.csv'
                

            else: # Holt-Winters
                with st.spinner("Menjalankan model Holt-Winters..."):
                    df_hist, df_pred = run_holtwinters_forecast(df_agg, days)
                
                if df_pred is not None:
                    # Plotting menggunakan lib/plots.py
                    fig_fc = create_forecast_chart(df_hist, df_pred, method="Holt-Winters")
                    st.plotly_chart(fig_fc, use_container_width=True, key="holtwinters_forecast_chart")
                    
                    # Persiapan Data Download
                    output_csv = df_pred.rename(columns={FC_COL_DS:'tanggal', 'yhat':'prediksi'})
                    csv_fc = convert_df_to_csv(output_csv)
                    fname = 'hasil_peramalan_holtwinters.csv'
                    
                else:
                    st.error("Gagal menjalankan peramalan Holt-Winters. Data mungkin tidak cukup atau pola tidak sesuai.")
                    csv_fc = None

            # Tombol Download (Konsisten untuk kedua metode)
            if csv_fc:
                st.download_button(
                    label="📥 Download Data Peramalan (CSV)",
                    data=csv_fc,
                    file_name=fname,
                    mime='text/csv',
                    key='download_fc'
                )
        else:
            st.warning("⚠️ Data terlalu sedikit untuk melakukan peramalan (minimal 10 titik data).")
            
def main():
    """Main entry point to run the Streamlit application."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.app_data = {}

    if not st.session_state.data_loaded:
        render_sidebar()
        st.info("👈 Silakan Upload Excel atau Koneksi Database di Sidebar.")
        st.markdown("<div class='title'>Selamat Datang di Dashboard Analisis Beras</div>", unsafe_allow_html=True)
    else:
        render_main_ui()

if __name__ == "__main__":
    main()