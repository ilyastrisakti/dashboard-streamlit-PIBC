# PIBC_explorer.py
# -*- coding: utf-8 -*-
import logging
import warnings
from typing import Optional, Tuple
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.badges import badge


import gc 
from streamlit_lottie import st_lottie

# --- Import Library Internal ---
from lib.forecast import run_prophet_forecast, run_holtwinters_forecast
from lib.plots import * 
from lib.constants import *
from lib.utils import convert_df_to_csv, calculate_regression, load_lottie_url
from lib.data import get_geo_lookup, init_connection, load_data_from_db, preprocess_data_from_excel
from lib.bussiness_logic import filter_and_aggregate_data, prepare_geo_data

# --- Import Konten Penjelasan (Modular) ---
try:
    from lib.content import (
        TXT_SIDEBAR_HELP, TXT_HOME_HEADER, TXT_HOME_BODY,
        TXT_MAP_HEADER, TXT_MAP_BODY, TXT_ANALYSIS_HEADER, TXT_ANALYSIS_BODY,
        TXT_STATS_HEADER, TXT_STATS_BODY, TXT_FORECAST_HEADER, TXT_FORECAST_BODY
    )
except ImportError:
    TXT_SIDEBAR_HELP = ""
    TXT_HOME_HEADER = TXT_MAP_HEADER = TXT_ANALYSIS_HEADER = TXT_STATS_HEADER = TXT_FORECAST_HEADER = "Info"
    TXT_HOME_BODY = TXT_MAP_BODY = TXT_ANALYSIS_BODY = TXT_STATS_BODY = TXT_FORECAST_BODY = "Konten..."

# --- Basic Setup ---
st.set_page_config(page_title="Dashboard PIBC 🌾", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# --- Assets ---
LOTTIE_URL = "https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json"

# --- CSS Modern ---
st.markdown("""
<style>
@keyframes fadeIn { from { opacity:0; transform: translateY(10px); } to { opacity:1; transform: translateY(0); } }
.main-title { font-size: 2.5rem; font-weight: 800; color: #2E86AB; margin: 0; animation: fadeIn 0.8s ease-out; }
.sub-title { font-size: 1.1rem; color: var(--text-color); opacity: 0.7; margin-bottom: 25px; animation: fadeIn 1.0s ease-out; }
div[data-testid="stMetric"] { background-color: var(--secondary-background-color); padding: 20px; border-radius: 12px; border: 1px solid rgba(128, 128, 128, 0.1); box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); transition: transform 0.2s ease, box-shadow 0.2s ease; animation: fadeIn 1.2s ease-out; }
div[data-testid="stMetric"]:hover { transform: translateY(-5px); box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1); border-color: #2E86AB; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: var(--secondary-background-color); padding: 10px; border-radius: 10px; }
.stTabs [data-baseweb="tab"] { height: 40px; border-radius: 6px; background-color: transparent; font-weight: 600; color: var(--text-color); }
.stTabs [aria-selected="true"] { background-color: #2E86AB !important; color: white !important; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
section[data-testid="stSidebar"] { background-color: var(--secondary-background-color); border-right: 1px solid rgba(128, 128, 128, 0.1); }
.streamlit-expanderHeader { font-weight: 600; color: #2E86AB; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# CALLBACKS (Anti-Looping)
# -------------------------
def on_upload_change():
    if st.session_state.uploaded_file_widget:
        st.session_state.data_source = 'excel'
        st.session_state.active_file = st.session_state.uploaded_file_widget
    else:
        st.session_state.data_source = None
        st.session_state.active_file = None

def on_db_connect_secrets():
    st.session_state.data_source = 'db_secrets'

def on_db_connect_manual():
    st.session_state.data_source = 'db_manual'

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def get_data_snapshot():
    source = st.session_state.get('data_source')
    
    if source == 'excel':
        f = st.session_state.get('active_file')
        if f: return preprocess_data_from_excel(f)
            
    elif source == 'db_secrets':
        d_start = st.session_state.get('db_start')
        d_end = st.session_state.get('db_end')
        eng = init_connection()
        return load_data_from_db(eng, d_start, d_end) if eng else (None,)*5
        
    elif source == 'db_manual':
        d_start = st.session_state.get('db_start')
        d_end = st.session_state.get('db_end')
        try:
            cfg = st.session_state
            conn_str = f"mysql+mysqlconnector://{cfg.db_user}:{cfg.db_pass}@{cfg.db_host}:3306/{cfg.db_name}"
            eng = create_engine(conn_str)
            return load_data_from_db(eng, d_start, d_end)
        except: return (None,)*5
        
    return None, None, None, None, None

def get_previous_period_data(df, start_date, end_date):
    """Mengambil data periode sebelumnya untuk perbandingan metrik."""
    duration = end_date - start_date
    prev_start = start_date - duration
    prev_end = start_date - pd.Timedelta(days=1)
    mask = (df[COL_TANGGAL].dt.date >= prev_start) & (df[COL_TANGGAL].dt.date <= prev_end)
    return df[mask]

def calculate_delta(curr, prev):
    if prev == 0 or pd.isna(prev) or pd.isna(curr): return None
    return ((curr - prev) / prev) * 100

# -------------------------
# UI Rendering
# -------------------------

def render_sidebar():
    with st.sidebar:
        lottie_json = load_lottie_url(LOTTIE_URL)
        if lottie_json: st_lottie(lottie_json, height=120, key="sidebar_anim")
        
        st.markdown("## 🌾 PIBC Explorer")
        st.caption("Sistem Analisis & Peramalan Stok Beras")
        st.markdown("---")

        if st.button("🧹 Reset Memori & Cache", use_container_width=True):
            st.cache_data.clear()
            st.session_state.clear()
            gc.collect()
            st.rerun()

        source = option_menu("Sumber Data", ["Upload Excel", "Database"], 
                             icons=['file-earmark-spreadsheet', 'database'], 
                             menu_icon="cast", default_index=0,
                             styles={"nav-link-selected": {"background-color": "#2E86AB"}})
        
        if source == "Upload Excel":
            st.file_uploader(
                "📂 Upload File (.xlsx)", 
                type=["xlsx"], 
                key="uploaded_file_widget", 
                on_change=on_upload_change
            )
            if st.session_state.get('data_source') == 'excel':
                st.success("✅ File Aktif")
        
        else: # Database
            st.info(TXT_SIDEBAR_HELP)
            c1, c2 = st.columns(2)
            now = pd.Timestamp.now().date()
            default_start = pd.to_datetime("2020-01-01").date()
            
            st.markdown("**Filter Data Awal (Server):**")
            st.date_input("Dari Tanggal", value=default_start, key="db_start")
            st.date_input("Sampai Tanggal", value=now, key="db_end")

            with st.expander("🔌 Konfigurasi Koneksi"):
                t1, t2 = st.tabs(["Otomatis", "Manual"])
                with t1:
                    st.button("Hubungkan (Secrets)", use_container_width=True, on_click=on_db_connect_secrets)
                with t2:
                    st.text_input("Host", "localhost", key="db_host")
                    st.text_input("User", "root", key="db_user")
                    st.text_input("Password", type="password", key="db_pass")
                    st.text_input("Database", "pibc_db", key="db_name")
                    st.button("Hubungkan (Manual)", use_container_width=True, on_click=on_db_connect_manual)

def render_metrics(df_curr, df_prev):
    if df_curr is None or df_curr.empty: return
    
    # Hitung nilai saat ini
    curr_stok = df_curr[COL_STOK].mean()
    curr_masuk = df_curr[COL_MASUK].sum()
    curr_keluar = df_curr[COL_KELUAR].sum()
    curr_neraca = df_curr[COL_NERACA].sum()

    # Hitung Delta (jika ada data sebelumnya)
    d_stok = d_masuk = d_keluar = None
    if not df_prev.empty:
        d_stok = calculate_delta(curr_stok, df_prev[COL_STOK].mean())
        d_masuk = calculate_delta(curr_masuk, df_prev[COL_MASUK].sum())
        d_keluar = calculate_delta(curr_keluar, df_prev[COL_KELUAR].sum())

    st.markdown("### 📊 Ringkasan Kinerja")
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("📦 Rata-rata Stok", f"{curr_stok:,.0f} Ton", f"{d_stok:,.1f}%" if d_stok else None, help="Rata-rata isi gudang.")
    c2.metric("🚛 Total Masuk", f"{curr_masuk:,.0f} Ton", f"{d_masuk:,.1f}%" if d_masuk else None, help="Total kiriman datang.")
    c3.metric("🚚 Total Keluar", f"{curr_keluar:,.0f} Ton", f"{d_keluar:,.1f}%" if d_keluar else None, delta_color="inverse", help="Total penjualan.")
    c4.metric("⚖️ Net Neraca", f"{curr_neraca:,.0f} Ton", "Surplus" if curr_neraca > 0 else "Defisit", 
              delta_color="normal" if curr_neraca > 0 else "inverse", help="Selisih Masuk - Keluar")
    st.markdown("---")
    
    style_metric_cards(
        background_color="#8D6F64",
        border_left_color="#2E86AB",
        border_color="#E0E0E0",
        box_shadow=True
    )

def render_main_ui():
    with st.spinner("🚀 Menghubungkan ke Data..."):
        df, df_stock, df_masuk, df_keluar, df_price = get_data_snapshot()

    if df is None:
        st.info("👈 Data belum dipilih atau koneksi database gagal. Cek pesan error di atas (jika ada).")
        return
    if df.empty:
        st.warning("⚠️ Data berhasil dimuat tapi KOSONG. Coba perluas rentang tanggal di sidebar.")
        return

    # --- Sidebar Filter ---
    with st.sidebar.form("filter_form"):
        st.subheader("🎛️ Filter Tampilan")
        min_ts = pd.to_datetime(df[COL_TANGGAL].min()); max_ts = pd.to_datetime(df[COL_TANGGAL].max())
        if pd.isna(min_ts): min_ts = pd.Timestamp.now()
        if pd.isna(max_ts): max_ts = pd.Timestamp.now()
            
        c1, c2 = st.columns(2)
        start_date = c1.date_input("Mulai", min_ts.date())
        end_date = c2.date_input("Sampai", max_ts.date())
        
        rice_opts = list(df_price.columns) if df_price is not None else []
        selected_rice = st.selectbox("Jenis Beras (Harga)", rice_opts) if rice_opts else None
        
        st.markdown("---")
        granularity = st.radio("Mode Tampilan:", ["Harian", "Bulanan", "Tahunan"], horizontal=True)
        submitted = st.form_submit_button("🔥 Terapkan Filter", use_container_width=True)

    # --- Data Processing ---
    is_daily = granularity == "Harian"
    df_agg = filter_and_aggregate_data(df, start_date, end_date, granularity)
    df_prev = get_previous_period_data(df, start_date, end_date) # Data periode sebelumnya untuk delta

    mask = lambda d: (d >= start_date) & (d <= end_date)
    df_m_filt = df_masuk[mask(df_masuk[COL_TANGGAL].dt.date)] if df_masuk is not None and not df_masuk.empty else pd.DataFrame()
    df_k_filt = df_keluar[mask(df_keluar[COL_TANGGAL].dt.date)] if df_keluar is not None and not df_keluar.empty else pd.DataFrame()
    df_p_filt = df_price[mask(df_price.index.date)] if df_price is not None and not df_price.empty else None

    # --- Header ---
    st.markdown("<div class='main-title'>Dashboard Analisis PIBvC</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-title'>Periode: {start_date.strftime('%d %b %Y')} s/d {end_date.strftime('%d %b %Y')} | Mode: {granularity}</div>", unsafe_allow_html=True)

    render_metrics(df_agg, df_prev)

    # --- Tabs ---
    tabs = st.tabs(["📈 Utama", "🗺️ Peta 3D", "🔍 Analisis", "📊 Statistik", "🔮 Peramalan"])

    # TAB 1: UTAMA
    with tabs[0]:
        with st.expander(TXT_HOME_HEADER, expanded=False): st.markdown(TXT_HOME_BODY)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(create_time_series(df_agg, COL_STOK, f"Tren Stok ({granularity})", "green", is_daily=is_daily), use_container_width=True)
        with c2: st.plotly_chart(create_balance_chart(df_agg, granularity=granularity), use_container_width=True)
        if selected_rice and df_p_filt is not None:
            st.divider()
            df_p_agg = df_p_filt[[selected_rice]].resample('M' if granularity=='Bulanan' else 'Y').mean() if not is_daily else df_p_filt[[selected_rice]]
            st.markdown(f"#### 🏷️ Tren Harga: {selected_rice}")
            st.plotly_chart(create_time_series(df_p_agg.reset_index(), selected_rice, f"Harga {selected_rice}", "#2E86AB", is_daily=is_daily, unit="Rp"), use_container_width=True)

    # TAB 2: PETA
    with tabs[1]:
        st.subheader("🗺️ Distribusi 3D")
        with st.expander(TXT_MAP_HEADER, expanded=False): st.markdown(TXT_MAP_BODY)
        c1, c2 = st.columns([1, 3])
        with c1: view = st.radio("Lihat Alur:", ["Barang Masuk (Supply)", "Barang Keluar (Distribusi)"])
        
        geo = get_geo_lookup()
        target_df, col_target = (df_m_filt, COL_MASUK) if view == "Barang Masuk (Supply)" else (df_k_filt, COL_KELUAR)
        
        df_map = prepare_geo_data(target_df, geo, col_target)
        if not df_map.empty:
            deck = create_pydeck_map(df_map, col_target)
            st.pydeck_chart(deck)
            # --- BAGIAN MODIFIKASI TABEL ---
            with st.expander(f"📋 Rincian Data {view}", expanded=True):
                
                # 1. Buat copy biar dataframe asli (untuk peta) tidak rusak
                df_display = df_map.copy()

                # --- [DEBUG] CEK TOTAL SEBELUM FILTER ---
                total_raw = df_display[col_target].sum()

                # Cari baris yang nilainya mencurigakan (Sangat besar > 10% total)
                suspects = df_display[df_display[col_target] > (total_raw * 0.1)]

                # Jika user mau melihat data mentah penyebab error
                if st.checkbox("🔍 Debug Mode (Cek Total Aneh)"):
                    st.write(f"Total Volume Mentah: {total_raw:,.0f}")
                    st.write("Tersangka Baris 'Raksasa' (Rekapitulasi?):")
                    st.dataframe(suspects)
                
                # --- [FIX] FILTER BARIS SAMPAH/REKAP ---
                # Buang baris yang namanya mengandung kata 'Total', 'Jumlah', 'Grand', atau kosong
                blacklist = ['total', 'jumlah', 'grand total', 'rekap', 'all', 'nan', 'unknown']
                
                # Filter baris blacklist
                mask_clean = ~df_display['lokasi_lookup'].astype(str).str.lower().isin(blacklist)
                df_display = df_display[mask_clean]

                # Opsional: Buang juga baris yang nilainya SANGAT BESAR (Misal > 50% dari total raw, biasanya itu Grand Total)
                # Logika: Jika satu baris sendirian menguasai > 80% total raw, kemungkinan itu baris Total.
                if not df_display.empty:
                    max_val = df_display[col_target].max()
                    if max_val > (total_raw * 0.9): 
                        df_display = df_display[df_display[col_target] != max_val]
                
                # 2. Hapus kolom teknis
                df_display = df_display.drop(columns=['lat', 'lon', 'normalized_elevation', 'normalized_width', 'target_lat', 'target_lon'], errors='ignore')
                
                # 3. Hitung Persentase Kontribusi (KALIKAN 100 DI SINI)
                total_clean = df_display[col_target].sum()
                # Ubah logic ini: dikali 100 agar jadi angka puluhan (misal: 26.3)
                df_display["Kontribusi"] = ((df_display[col_target] / total_clean) * 100) if total_clean > 0 else 0
                
                # 4. Rename & Display
                df_display = df_display.rename(columns={
                    "lokasi_lookup": "Daerah Asal/Tujuan",
                    col_target: "Volume (Ton)"
                })
                
                st.dataframe(
                    df_display.sort_values("Volume (Ton)", ascending=False),
                    column_config={
                        "Volume (Ton)": st.column_config.NumberColumn(
                            "Volume (Ton)",
                            format="%d Ton",
                        ),
                        "Kontribusi": st.column_config.ProgressColumn(
                            "Kontribusi (%)",
                            format="%.1f%%",   # Format akan menampilkan "26.3%"
                            min_value=0,
                            max_value=100,     # <--- PENTING: Ubah Max jadi 100
                        ),
                        "Daerah Asal/Tujuan": st.column_config.TextColumn(
                            "Daerah Asal/Tujuan",
                            width="medium"
                        )
                    },
                    use_container_width=True,
                    hide_index=True
                )
        else: st.warning("Data kosong untuk periode ini.")

    # TAB 3: ANALISIS
    with tabs[2]:
        colored_header(
        label="🔍 Kestabilan Stok",
        description="Analisis volatilitas dan ketahanan gudang",
        color_name="blue-70", # Pilihan warna: 'red-70', 'violet-70', dll
    )
        with st.expander(TXT_ANALYSIS_HEADER, expanded=True): st.markdown(TXT_ANALYSIS_BODY)
        if is_daily:
            c1, c2 = st.columns([3, 1])
            with c2:
                st.markdown("#### 🎛️ Simulasi")
                safe = st.slider("Target Aman (Hari)", 10, 60, 20)
                crit = st.slider("Batas Kritis (Hari)", 1, 15, 10)
            with c1: st.plotly_chart(create_inventory_cover_chart(df_agg, safe_days=safe, critical_days=crit), use_container_width=True)
            st.divider()
            c3, c4 = st.columns(2)
            with c3: st.plotly_chart(create_volatility_chart(df_agg, COL_STOK, 30, title="Volatilitas Stok"), use_container_width=True)
            with c4:
                if selected_rice and df_p_filt is not None:
                    p_res = df_p_filt[[selected_rice]].reset_index()
                    st.plotly_chart(create_volatility_chart(p_res, selected_rice, 7, title=f"Volatilitas Harga: {selected_rice}"), use_container_width=True)
        else: st.info("⚠️ Ubah filter ke 'Harian' untuk melihat analisis ini.")
        if df_p_filt is not None:
            st.divider()
            st.plotly_chart(create_price_heatmap(df_p_filt, correlation=True), use_container_width=True)

    # TAB 4: STATISTIK
    with tabs[3]:
        # Header Section
        c_head1, c_head2 = st.columns([3, 1])
        with c_head1:
            st.subheader("📊 Analisis Statistik Mendalam")
            st.caption("Membedah perilaku data stok dan hubungan ekonomi dengan harga.")
        with c_head2:
            with st.expander("Panduan Baca"):
                st.markdown(TXT_STATS_BODY)

        st.markdown("---")

        # --- BAGIAN 1: RAPOR STOK ---
        st.markdown("#### 1. Rapor Kinerja Gudang")
        if not df_agg.empty:
            s_mean = df_agg[COL_STOK].mean()
            s_max = df_agg[COL_STOK].max()
            s_min = df_agg[COL_STOK].min()
            s_std = df_agg[COL_STOK].std()
            s_cv = (s_std / s_mean) * 100 if s_mean else 0 

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Stok Normal (Rata-rata)", f"{s_mean:,.0f} Ton")
            k2.metric("Stok Tertinggi (Rekor)", f"{s_max:,.0f} Ton")
            k3.metric("Stok Terendah", f"{s_min:,.0f} Ton", delta_color="inverse")
            
            stab_label = "Stabil" if s_cv < 10 else ("Sedang" if s_cv < 20 else "Labil")
            k4.metric("Stabilitas Stok", stab_label, f"Var: {s_cv:.1f}%", delta_color="inverse" if s_cv > 20 else "normal")

            if stab_label == "Stabil":
                badge(type="github", name="Pass") # Hijau
            elif stab_label == "Labil":
                badge(type="pypi", name="Warning") # Merah/Kuning
            
            st.plotly_chart(create_stock_distribution(df_agg), use_container_width=True)
        else: st.warning("Data stok tidak tersedia.")

        st.markdown("---")

        # --- BAGIAN 2: REGRESI ---
        st.markdown(f"#### 2. Cek Hukum Pasar: Stok vs Harga ({selected_rice or 'Pilih Beras Dulu'})")
        if selected_rice:
            reg = calculate_regression(df_agg, df_price, selected_rice)
            if reg:
                col_reg_L, col_reg_R = st.columns([1, 2])
                with col_reg_L:
                    st.info("💡 **Hasil Analisis:**")
                    r2 = reg['r2']
                    strength = "Sangat Kuat" if r2 > 0.7 else "Kuat" if r2 > 0.5 else "Sedang" if r2 > 0.3 else "Lemah"
                    slope = reg['slope']
                    direction = "berlawanan" if slope < 0 else "searah"
                    logic_check = "✅ Sesuai Teori" if slope < 0 else "❌ Anomali"

                    st.markdown(f"""
                    Hubungan: **{strength}** (R²={r2:.2f}).
                    Arah: **{direction}**.
                    Setiap stok +1 Ton, harga berubah **Rp {slope:.2f}**.
                    Kesimpulan: **{logic_check}**
                    """)
                    
                    if reg['p_value'] < 0.05: st.success("Analisis Valid (Signifikan).")
                    else: st.warning("Analisis Tidak Signifikan.")

                with col_reg_R:
                    st.plotly_chart(create_regression_scatter(reg['df'], COL_STOK, selected_rice), use_container_width=True)
            else: st.warning("Data tidak cukup untuk regresi.")
        else: st.info("👈 Pilih Jenis Beras di Sidebar.")

        # --- BAGIAN 3: DOWNLOAD ---
        csv_data = convert_df_to_csv(df_agg)
        st.download_button(
            label="📥 Download Data Terolah (.csv)",
            data=csv_data,
            file_name=f"laporan_pibc_{start_date}_{end_date}.csv",
            mime='text/csv',
        )

    # TAB 5: PERAMALAN
    with tabs[4]:
        st.subheader("🔮 Mesin Peramal Stok")
        with st.expander(TXT_FORECAST_HEADER, expanded=True): st.markdown(TXT_FORECAST_BODY)
        if len(df_agg) > 20:
            c1, c2 = st.columns([1, 2])
            with c1:
                method = st.selectbox("Pilih Metode", ["Prophet", "Holt-Winters"])
                days = st.slider(f"Jangka Waktu Prediksi", 7, 90, 30)
            with c2:
                st.write("")
                st.write("")
                if st.button("✨ Mulai Prediksi AI", type="primary", use_container_width=True):
                    with st.spinner("Sedang menghitung..."):
                        func = run_prophet_forecast if method == "Prophet" else run_holtwinters_forecast
                        hist, pred = func(df_agg, days)
                        st.plotly_chart(create_forecast_chart(hist, pred, method), use_container_width=True)
                        if pred is not None:
                            with st.expander("Lihat Angka Hasil Ramalan"):
                                st.dataframe(pred.rename(columns={'ds':'Tanggal', 'yhat':'Prediksi Stok'}), use_container_width=True)
                        else:
                            st.warning("Gagal membuat ramalan. Coba metode lain atau periksa data.")
        else: st.warning("⚠️ Data sejarah terlalu sedikit.")

def main():
    if 'data_source' not in st.session_state: st.session_state.data_source = None
    if not st.session_state.data_source:
        render_sidebar()
        st.markdown("<div class='main-title' style='text-align:center; margin-top:50px;'>Selamat Datang di PIBC Explorer 🌾</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; color:grey;'>Sistem Analisis Data Pangan Terintegrasi</div>", unsafe_allow_html=True)
        st.info("👈 Mulai dengan menghubungkan data (Excel/Database) di menu sebelah kiri.")
    else:
        render_sidebar()
        render_main_ui()

if __name__ == "__main__":
    main()