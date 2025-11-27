import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from streamlit_option_menu import option_menu
from sqlalchemy import create_engine, text

from lib.data import (
    preprocess_data_from_excel,
    init_connection,
    load_data_from_db,
    get_geo_lookup
)
from lib.utils import price_df_with_tanggal, convert_df_to_csv
from lib.plots import (
    DEFAULT_PLOTLY_TEMPLATE,
    create_time_series,
    create_balance_chart,
    create_price_heatmap,
    create_volatility_chart,
    create_inventory_cover_chart,
    create_stock_distribution,
    create_regression_scatter,
    create_geo_map
)

# --- Session / helper ---
def _init_session_state():
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "app_data" not in st.session_state:
        st.session_state.app_data = {}
    if "filters" not in st.session_state:
        st.session_state.filters = {"date_range": None, "lokasi": None, "rice_type": None}

def _set_app_data(df_main, df_stock, df_masuk, df_keluar, df_price):
    st.session_state.app_data = {
        "df": df_main,
        "df_stock": df_stock,
        "df_masuk": df_masuk,
        "df_keluar": df_keluar,
        "df_price": df_price,
    }
    st.session_state.data_loaded = True

# --- Styling (dark & light) for readability ---
_styling = """
<style>
/* Tabs & guide box contrast for both themes */
.stTabs [data-baseweb="tab"] { background-color: #f0f2f6; color: #0b1a24; border-radius: 8px; }
.stTabs [aria-selected="true"] { background-color: #2E86AB; color: white; }
[data-theme="dark"] .stTabs [data-baseweb="tab"] { background-color: #0f1720 !important; color: #cfeaf8 !important; }
[data-theme="dark"] .stTabs [aria-selected="true"] { background-color: #164f6b !important; color: #fff !important; }

.guide-box { background-color:#f0f2f6; color:#0b1a24; padding:10px; border-radius:8px; border:1px solid rgba(0,0,0,0.06); }
[data-theme="dark"] .guide-box { background-color:#071025 !important; color:#E6F2FF !important; border:1px solid rgba(255,255,255,0.06) !important; }

.sidebar .stButton>button { width: 100%; }
</style>
"""
st.markdown(_styling, unsafe_allow_html=True)

# --- Sidebar controls (data source + connection) ---
def _sidebar_controls():
    st.sidebar.title("PIBC Explorer")
    st.sidebar.caption("Sumber Data")
    # use option menu for quick actions
    with st.sidebar:
        _ = option_menu("Menu", ["Load Data", "Connections", "Settings"], icons=["cloud-upload","database","gear"], menu_icon="cast", default_index=0)
    mode = st.sidebar.radio("Pilih sumber data", ["Excel (upload)", "Database"], index=0)
    st.sidebar.markdown("---")
    return mode

def _connect_db_manual_form():
    st.sidebar.markdown("### Koneksi DB Manual")
    host = st.sidebar.text_input("Host", value="localhost")
    port = st.sidebar.text_input("Port", value="3306")
    user = st.sidebar.text_input("User")
    password = st.sidebar.text_input("Password", type="password")
    database = st.sidebar.text_input("Database")
    if st.sidebar.button("Connect (Manual)"):
        if not all([host, port, user, password, database]):
            st.sidebar.error("Lengkapi semua field koneksi.")
            return None
        conn_str = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        try:
            eng_manual = create_engine(conn_str)
            with eng_manual.connect() as conn:
                conn.execute(text("SELECT 1"))
            return eng_manual
        except Exception as e:
            st.sidebar.error(f"Koneksi gagal: {e}")
            return None
    return None

# --- Load functions ---
def _load_from_excel():
    uploaded_file = st.sidebar.file_uploader("Upload file Excel (.xlsx)", type=["xlsx", "xls"])
    if uploaded_file:
        with st.spinner("Memproses file..."):
            try:
                df_main, df_stock, df_masuk, df_keluar, df_price = preprocess_data_from_excel(uploaded_file)
            except Exception as e:
                st.sidebar.error(f"Error memproses Excel: {e}")
                return
            if df_main is None:
                st.sidebar.error("Sheet rice_stock tidak ditemukan atau format tidak sesuai.")
                return
            _set_app_data(df_main, df_stock, df_masuk, df_keluar, df_price)
            st.sidebar.success("File berhasil diproses.")
    return None

def _load_from_db():
    if st.sidebar.button("Connect using st.secrets"):
        eng = init_connection()
        if eng is None:
            st.sidebar.error("st.secrets tidak ditemukan atau konfigurasi salah.")
            return
        with st.spinner("Mengambil data dari DB..."):
            df_main, df_stock, df_masuk, df_keluar, df_price = load_data_from_db(eng)
            if df_main is None:
                st.sidebar.error("Gagal memuat data dari DB.")
                return
            _set_app_data(df_main, df_stock, df_masuk, df_keluar, df_price)
            st.sidebar.success("Data DB berhasil dimuat.")
    # manual fallback
    eng_manual = _connect_db_manual_form()
    if eng_manual is not None:
        with st.spinner("Mengambil data DB (manual)..."):
            df_main, df_stock, df_masuk, df_keluar, df_price = load_data_from_db(eng_manual)
            if df_main is None:
                st.sidebar.error("Gagal memuat data dari DB (manual).")
                return
            _set_app_data(df_main, df_stock, df_masuk, df_keluar, df_price)
            st.sidebar.success("Data DB (manual) berhasil dimuat.")

# --- General filter helpers used across UI ---
def _apply_filters(df):
    if df is None or df.empty:
        return df
    import pandas as pd
    f = st.session_state.filters
    out = df.copy()

    # Ensure tanggal column is datetime before comparisons
    if "tanggal" in out.columns:
        out["tanggal"] = pd.to_datetime(out["tanggal"], errors="coerce")

    # Date-range (robust convert)
    dr = f.get("date_range")
    if dr:
        try:
            start = pd.to_datetime(dr[0], errors="coerce")
            end = pd.to_datetime(dr[1], errors="coerce")
            if pd.notna(start) and pd.notna(end):
                out = out[(out["tanggal"] >= start) & (out["tanggal"] <= end)]
        except Exception:
            # if conversion fails, ignore date filter
            pass

    # lokasi filter: normalize and compare to lokasi_norm
    loc = f.get("lokasi")
    if loc and "lokasi" in out.columns:
        out = out[out["lokasi_norm"].astype(str).str.lower() == str(loc).lower()]

    return out

# --- UI renderers (tabs, metrics, tables, filters) ---
def _render_filters(app_data):
    df = app_data.get("df")
    df_masuk = app_data.get("df_masuk")
    df_keluar = app_data.get("df_keluar")
    df_price = app_data.get("df_price")

    col_f1, col_f2 = st.sidebar.columns([2,1])
    with col_f1:
        st.sidebar.markdown("### Filters")
        if df is not None and "tanggal" in df.columns:
            # use date objects for widget and store canonical timestamps
            min_d = pd.to_datetime(df["tanggal"].min()).date()
            max_d = pd.to_datetime(df["tanggal"].max()).date()
            dr = st.sidebar.date_input("Range tanggal", value=(min_d, max_d))
            st.session_state.filters["date_range"] = (pd.to_datetime(str(dr[0])), pd.to_datetime(str(dr[1])))
        # location picker from combined locations of masuk/keluar + geo_lookup
        choices = set()
        if df_masuk is not None and "lokasi" in df_masuk.columns:
            choices |= set([str(x).strip() for x in df_masuk["lokasi"].unique()])
        if df_keluar is not None and "lokasi" in df_keluar.columns:
            choices |= set([str(x).strip() for x in df_keluar["lokasi"].unique()])
        choices = sorted([c for c in choices if c and c.strip()])
        pilihan = st.sidebar.selectbox("Filter lokasi (opsional)", options=["Semua"] + choices)
        st.session_state.filters["lokasi"] = None if pilihan == "Semua" else pilihan
    with col_f2:
        st.sidebar.markdown("### Quick actions")
        if st.sidebar.button("Reset filters"):
            st.session_state.filters = {"date_range": None, "lokasi": None, "rice_type": None}
        if st.sidebar.button("Reload data (clear cache)"):
            # Try the public API first; if it's not available in the
            # runtime (some environments/type-checkers), fall back to raising the
            # internal RerunException so Streamlit restarts the script. If neither
            # approach works, clear relevant session state and prompt the user to refresh.
            try:
                st.rerun()
            except Exception:
                try:
                    from streamlit.runtime.scriptrunner import RerunException
                    raise RerunException()
                except Exception:
                    st.sidebar.info("Silakan muat ulang halaman untuk menerapkan perubahan.")
                    st.session_state.data_loaded = False
                    st.session_state.app_data = {}

def _show_aggrid(title, df):
    if df is None:
        st.info(f"Tidak ada data: {title}")
        return
    st.subheader(title)
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(editable=False, resizable=True)
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    grid = AgGrid(df, gridOptions=gb.build(), fit_columns_on_grid_load=True, enable_enterprise_modules=False)
    return grid

def _render_overview():
    app_data = st.session_state.app_data
    df_main = app_data.get("df")
    df_masuk = app_data.get("df_masuk")
    df_keluar = app_data.get("df_keluar")

    # apply filters to main/outflow/inflow where applicable
    df_main_f = _apply_filters(df_main)
    df_masuk_f = _apply_filters(df_masuk)
    df_keluar_f = _apply_filters(df_keluar)

    tot_masuk = int(df_main_f["masuk"].sum()) if (df_main_f is not None and "masuk" in df_main_f.columns) else 0
    tot_keluar = int(df_main_f["keluar"].sum()) if (df_main_f is not None and "keluar" in df_main_f.columns) else 0
    neraca = tot_masuk - tot_keluar

    st.title("Dashboard Analisis PIBC 🌾")
    st.markdown("Ringkasan metrik dan grafik utama")

    # Penjelasan singkat & expandable detail
    st.info("Overview: metrik agregat harian, neraca (masuk - keluar) dan contoh time-series.")
    with st.expander("Penjelasan Fitur — Overview (klik untuk lihat detail)"):
        st.markdown("""
        **Apa yang ditampilkan**
        - Total Masuk / Total Keluar: jumlah ton yang terakumulasi pada rentang waktu yang sedang difilter.
        - Net Neraca: selisih Total Masuk dikurangi Total Keluar.
        - Neraca Harian: bar chart menampilkan masuk (+) dan keluar (-) per tanggal; garis neraca menunjukkan selisih harian.
        - Time Series: visualisasi seri waktu (stok/masuk/keluar/neraca).
        
        **Sumber data yang digunakan**
        - df_main: hasil merge/aggregasi utama (tanggal, stok, masuk, keluar, neraca) — berasal dari file Excel (rice_stock, rice_source, rice_delivery) atau DB.
        
        **Cara membaca**
        - Jika Neraca = 0 di seluruh rentang → cek sheet `rice_source` / `rice_delivery` apakah berisi data.
        - Jika angka terlihat terlalu kecil/besar → periksa satuan (ton) dan kemungkinan double-counting saat ada duplikasi tanggal.
        
        **Troubleshooting singkat**
        - Pastikan upload file Excel memiliki sheet `rice_stock` dan `rice_source` / `rice_delivery`.
        - Untuk DB, pastikan st.secrets atau form manual terisi dan SELECT 1 berhasil.
        """)
    # rest of overview UI...
    with st.expander("Pengaturan tampilan grafik"):
        sample_col = st.selectbox("Pilih seri sample untuk time-series", options=["stok","masuk","keluar","neraca"])
        st.session_state.filters["sample_col"] = sample_col

    st.subheader("Neraca Harian")
    st.plotly_chart(create_balance_chart(df_main_f), use_container_width=True)

    st.subheader("Time series sample")
    sample_col = st.session_state.filters.get("sample_col", "stok")
    st.plotly_chart(create_time_series(df_main_f, sample_col, f"{sample_col.title()} Harian", color="#2E86AB"), use_container_width=True)

def _render_data_tab():
    app_data = st.session_state.app_data
    df_stock = app_data.get("df_stock")
    df_masuk = app_data.get("df_masuk")
    df_keluar = app_data.get("df_keluar")
    df_price = app_data.get("df_price")

    st.header("Data — preview & export")
    st.markdown("Periksa data yang telah dimuat. Gunakan export untuk menyimpan CSV.")

    # Penjelasan per dataset
    with st.expander("Penjelasan Fitur — Data (klik untuk lihat detail)"):
        st.markdown("""
        **Tabel dan maksud kolom**
        - rice_stock: historis stok harian. Kolom penting: `tanggal`, `stok` (ton). Opsional: `masuk`, `keluar` sebagai fallback.
        - rice_source: data asal (wide format). Kolom: `tanggal` + banyak kolom lokasi (contoh: Karawang, Bandung...). Aplikasi akan melt → menjadi `lokasi`, `masuk`.
        - rice_delivery: data distribusi/keluar (wide format). Diproses mirip source → `lokasi`, `keluar`.
        - rice_price: harga per jenis per tanggal. Aplikasi mencoba pivot ke timeseries per jenis.
        
        **Format yang diharapkan**
        - `tanggal` kolom wajib ada atau terdeteksi; lokasi pada source/delivery diharapkan nama daerah (Harus distandarkan, e.g. 'Bandung' / 'bandung').
        
        **Tips**
        - Jika preview kosong → periksa header sheet (spelling/whitespace).
        - Gunakan tombol export untuk menyimpan CSV hasil preprocess.
        """)
    _show_aggrid("Stock (rice_stock)", df_stock)
    if df_stock is not None:
        st.download_button("Download stock CSV", convert_df_to_csv(df_stock), "stock.csv", "text/csv")

    _show_aggrid("Masuk (rice_source / aggregated)", df_masuk)
    if df_masuk is not None:
        st.download_button("Download masuk CSV", convert_df_to_csv(df_masuk), "masuk.csv", "text/csv")

    _show_aggrid("Keluar (rice_delivery / aggregated)", df_keluar)
    if df_keluar is not None:
        st.download_button("Download keluar CSV", convert_df_to_csv(df_keluar), "keluar.csv", "text/csv")

    if df_price is not None:
        st.subheader("Harga (rice_price)")
        ph = price_df_with_tanggal(df_price)
        _show_aggrid("Harga (pivoted)", ph)
        st.download_button("Download harga CSV", convert_df_to_csv(ph), "harga.csv", "text/csv")

def _render_map_tab():
    app_data = st.session_state.app_data
    df_masuk = app_data.get("df_masuk")
    df_keluar = app_data.get("df_keluar")
    geo_lookup = get_geo_lookup()

    st.header("Peta Asal & Distribusi")
    st.markdown('<div class="guide-box">ℹ️ Panduan Peta: Lingkaran = volume, warna = intensitas, hover untuk detail</div>', unsafe_allow_html=True)

    # Penjelasan peta
    with st.expander("Penjelasan Fitur — Peta (klik untuk lihat detail)"):
        st.markdown("""
        **Apa yang ditampilkan**
        - Dua peta: Asal (masuk) dan Distribusi (keluar). Tiap titik mewakili lokasi dari sheet source/delivery.
        - Ukuran lingkaran: berskala log untuk menghindari ukuran ekstrem; mewakili total ton di lokasi.
        - Warna: skala warna bergradasi (Hijau untuk Masuk, Merah untuk Keluar).

        **Bagaimana lokasi dicocokkan**
        - Aplikasi menormalisasi nama lokasi menjadi `lokasi_norm` (lowercase & strip) dan mencocokkan ke geo_lookup.
        - Jika lokasi tidak tampil → kemungkinan mismatch pada nama (spasi, singkatan). Periksa `df_masuk` / `df_keluar` kolom `lokasi` dan bandingkan dengan geo lookup.

        **Troubleshoot**
        - Untuk menemukan lokasi yang tidak cocok: buka tab Data → cek kolom `lokasi` di dataset masuk/keluar.
        - Sesuaikan geo_lookup (lib/data.get_geo_lookup) jika ada lokasi baru.
        """)
    # map controls and charts
    c1, c2 = st.columns([3,1])
    with c2:
        st.markdown("Pengaturan Peta")
        map_flow = st.selectbox("Tampilkan", ["Masuk", "Keluar"], index=0)
        st.session_state.filters["map_flow"] = "masuk" if map_flow.lower()=="masuk" else "keluar"

    col_map_l, col_map_r = st.columns(2)
    fig_in = create_geo_map(df_masuk, geo_lookup, flow_type="masuk")
    fig_out = create_geo_map(df_keluar, geo_lookup, flow_type="keluar")
    col_map_l.subheader("Masuk (Asal)")
    col_map_l.plotly_chart(fig_in, use_container_width=True)
    col_map_r.subheader("Keluar (Distribusi)")
    col_map_r.plotly_chart(fig_out, use_container_width=True)

def _render_price_tab():
    app_data = st.session_state.app_data
    df_price = app_data.get("df_price")

    st.header("Analisis Harga")
    if df_price is None:
        st.info("Tidak ada data harga")
        return

    with st.expander("Penjelasan Fitur — Harga (klik untuk lihat detail)"):
        st.markdown("""
        **Heatmap Harga**
        - Menampilkan harga tiap jenis beras (baris = jenis, kolom = tanggal).
        - Warna hijau → harga lebih rendah; merah → lebih tinggi (skala disesuaikan).

        **Volatility**
        - Menghitung rolling standard deviation (window default 7 hari) untuk tiap seri harga, lalu meng-aggregate dengan mean untuk menunjukkan volatility pasar.
        - Gunakan ini untuk mendeteksi periode fluktuasi harga yang tinggi.

        **Interpretasi**
        - Lonjakan volatility berarti harga jenis tertentu tidak stabil → pantau pasokan / permintaan.
        """)
    st.plotly_chart(create_price_heatmap(df_price), use_container_width=True)
    st.plotly_chart(create_volatility_chart(df_price), use_container_width=True)

def _render_statistics_tab():
    app_data = st.session_state.app_data
    df_main = app_data.get("df")
    if df_main is None:
        st.info("Tidak ada data utama")
        return

    with st.expander("Penjelasan Fitur — Statistik (klik untuk lihat detail)"):
        st.markdown("""
        **Distribusi & Inventory Cover**
        - Distribusi Stok: histogram + boxplot membantu melihat spread dan outlier stok.
        - Inventory Cover: memperkirakan berapa hari stok saat ini dapat menutupi rata-rata outflow (rolling window).
        
        **Regresi**
        - Contoh regresi (stok vs neraca) dengan nilai korelasi r. Gunakan untuk melihat hubungan linear sederhana.
        - Catatan: korelasi ≠ kausalitas. Periksa outlier dan periode waktu sebelum mengambil keputusan.

        **Penggunaan praktis**
        - Gunakan kombinasi chart untuk analisis: high volatility + menipisnya stok → potensi kenaikan harga.
        """)
    st.header("Statistik & Distribusi")
    st.plotly_chart(create_stock_distribution(df_main), use_container_width=True)
    st.plotly_chart(create_inventory_cover_chart(df_main, days_cover=30), use_container_width=True)

    st.subheader("Regresi Contoh: Neraca vs Stok")
    st.plotly_chart(create_regression_scatter(df_main, "stok", "neraca"), use_container_width=True)

def _render_tools_tab():
    st.header("Tools")
    app_data = st.session_state.app_data
    st.write("Export semua data, debug & quick actions")

    with st.expander("Penjelasan Fitur — Tools (klik untuk lihat detail)"):
        st.markdown("""
        **Download / Export**
        - Download per tabel: stock, masuk, keluar, harga.
        - Bila butuh semua file sekaligus: gunakan script eksternal untuk menggabungkan hasil download.

        **Debug / Inspect**
        - Session state: menampilkan tipe data yang tersimpan di session (bukan nilai sensitif).
        - Gunakan `Reset filters` dan `Reload data` di sidebar untuk memaksa clean-run.

        **Tips untuk admin**
        - Saat data DB berubah, klik `Reload data (clear cache)` atau restart app.
        - Periksa log (terminal) untuk stacktrace jika ada error data parsing.
        """)
    if st.button("Download semua CSV (separate)"):
        if app_data.get("df_stock") is not None:
            st.download_button("stock.csv", convert_df_to_csv(app_data["df_stock"]), "stock.csv")
        if app_data.get("df_masuk") is not None:
            st.download_button("masuk.csv", convert_df_to_csv(app_data["df_masuk"]), "masuk.csv")
        if app_data.get("df_keluar") is not None:
            st.download_button("keluar.csv", convert_df_to_csv(app_data["df_keluar"]), "keluar.csv")
        if app_data.get("df_price") is not None:
            st.download_button("harga.csv", convert_df_to_csv(price_df_with_tanggal(app_data["df_price"])), "harga.csv")

    st.markdown("---")
    st.write("Debug / quick inspect")
    if st.checkbox("Tampilkan session_state"):
        st.json({k: str(type(v)) for k, v in st.session_state.items()})

def run_app():
    st.set_page_config(page_title="Dashboard Analisis PIBC", layout="wide")
    _init_session_state()

    mode = _sidebar_controls()
    _render_filters(st.session_state.app_data if st.session_state.get("app_data") else {})

    # load controls
    if not st.session_state.data_loaded:
        if mode == "Excel (upload)":
            _load_from_excel()
        else:
            _load_from_db()

    if not st.session_state.data_loaded:
        st.info("Silakan upload Excel atau hubungkan ke DB di sidebar.")
        return

    # main area with tabs (full UI restored)
    tabs = st.tabs(["Overview", "Data", "Peta", "Harga", "Statistik", "Tools"])
    with tabs[0]:
        _render_overview()
    with tabs[1]:
        _render_data_tab()
    with tabs[2]:
        _render_map_tab()
    with tabs[3]:
        _render_price_tab()
    with tabs[4]:
        _render_statistics_tab()
    with tabs[5]:
        _render_tools_tab()