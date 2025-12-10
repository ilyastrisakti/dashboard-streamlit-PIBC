# lib/plots.py
import math
from typing import Optional
import pandas as pd
import numpy as np
from scipy.stats import linregress
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from .constants import *
from .constants import FC_COL_DS, FC_COL_Y

# Choose plotly template based on Streamlit theme (fallback)
try:
    _theme = st.get_option("theme.base")
except Exception:
    _theme = None
DEFAULT_PLOTLY_TEMPLATE = "plotly_dark" if _theme == "dark" else "plotly_white"


def _ensure_fig_template(fig: go.Figure, template: Optional[str]):
    """Sets the figure's template to the provided one or the app's default."""
    fig.layout.template = template or DEFAULT_PLOTLY_TEMPLATE
    return fig


def create_time_series(df: pd.DataFrame, y_col: str, title: str, color: Optional[str] = None, template: Optional[str] = None, is_daily: bool = True) -> go.Figure:
    """
    Creates a line/area chart for a single time-series column with range slider.
    """
    if df is None or df.empty or y_col not in df.columns:
        return go.Figure()
    
    line_shape = "spline" if is_daily else "hv"
    
    # Area chart
    fig = px.area(
        df, x=COL_TANGGAL, y=y_col, 
        title=title, 
        line_shape=line_shape, 
        color_discrete_sequence=[color] if color else None, 
        template=template or DEFAULT_PLOTLY_TEMPLATE
    )
    
    # Styling area gradient & markers
    if fig.data:
        line_color = fig.data[0].line.color
        # Create semi-transparent fill
        fill_color = f"rgba({int(line_color[1:3], 16)}, {int(line_color[3:5], 16)}, {int(line_color[5:7], 16)}, 0.1)" if line_color and '#' in line_color else "rgba(0,0,0,0.1)"
        
        fig.update_traces(
            fillcolor=fill_color,
            mode='lines' if is_daily else 'lines+markers',
            hovertemplate='%{x|%d %b %Y}<br><b>%{y:,.0f} Ton</b><extra></extra>'
        )

    # Add Range Slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ]),
            bgcolor="#2E86AB" if _theme != "dark" else "#1E1E1E",
            activecolor="#1E5F78" if _theme != "dark" else "#333333"
        )
    )
    
    fig.update_layout(
        margin={"r":10,"t":50,"l":10,"b":10},
        hovermode="x unified",
        yaxis_title=None,
        xaxis_title=None
    )
    return fig


def create_balance_chart(df: pd.DataFrame, template: Optional[str] = None, granularity: str = "Harian") -> go.Figure:
    """Creates a balance chart with bars for 'masuk'/'keluar' and a line for 'neraca'."""
    if df is None or df.empty:
        return go.Figure()
        
    width = None 
    if granularity == "Bulanan":
        width = 25 * 24 * 60 * 60 * 1000 
    elif granularity == "Tahunan":
        width = 300 * 24 * 60 * 60 * 1000

    fig = go.Figure()
    
    # Masuk
    fig.add_trace(go.Bar(
        x=df[COL_TANGGAL], y=df[COL_MASUK], 
        name="Masuk", marker_color="#2E86AB", width=width, opacity=0.8,
        hovertemplate='%{x|%d %b %Y}<br>Masuk: <b>%{y:,.0f}</b><extra></extra>'
    ))
    
    # Keluar (Visual negatif)
    fig.add_trace(go.Bar(
        x=df[COL_TANGGAL], y=-df[COL_KELUAR], 
        name="Keluar", marker_color="#F24236", width=width, opacity=0.8,
        customdata=df[COL_KELUAR],
        hovertemplate='%{x|%d %b %Y}<br>Keluar: <b>%{customdata:,.0f}</b><extra></extra>'
    ))
    
    # Neraca Line
    fig.add_trace(go.Scatter(
        x=df[COL_TANGGAL], y=df[COL_NERACA], 
        name="Neraca (Net)", mode='lines+markers', 
        line=dict(color="#FFD700", width=3), 
        marker=dict(size=6, symbol="circle", line=dict(width=1, color="black")),
        hovertemplate='%{x|%d %b %Y}<br>Net: <b>%{y:,.0f}</b><extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Neraca {granularity}", 
        barmode="relative", 
        template=template or DEFAULT_PLOTLY_TEMPLATE, 
        margin={"r":10,"t":50,"l":10,"b":10},
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_price_heatmap(df_price: pd.DataFrame, correlation: bool = False, template: Optional[str] = None) -> go.Figure:
    """Creates a heatmap for rice prices."""
    if df_price is None or df_price.empty:
        return go.Figure()
    
    fig = go.Figure()

    if correlation:
        corr = df_price.corr()
        # Mask upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr_masked = corr.mask(mask)
        
        fig.add_trace(go.Heatmap(
            z=corr_masked.values, x=corr_masked.columns, y=corr_masked.index,
            colorscale='RdYlGn', zmin=-1, zmax=1,
            text=np.round(corr_masked.values, 2), texttemplate="%{text}",
            xgap=1, ygap=1
        ))
        fig.update_layout(title="Matriks Korelasi Harga (Segitiga Bawah)")
    else:
        p = df_price.copy()
        if isinstance(p.index, pd.DatetimeIndex) and "tanggal" not in p.columns: p = p.reset_index()
        if COL_TANGGAL in p.columns: p = p.set_index(COL_TANGGAL)
        
        # Numeric only
        p_num = p.select_dtypes(include=[np.number])
        
        fig.add_trace(go.Heatmap(
            x=p_num.index, y=p_num.columns, z=p_num.T.values, 
            colorscale="Viridis", colorbar=dict(title="Harga")
        ))
        fig.update_layout(title="Heatmap Harga")
    
    fig = _ensure_fig_template(fig, template)
    return fig


def create_volatility_chart(df: pd.DataFrame, target_col: str, window: int = 7, template: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
    """Calculates and plots rolling volatility as Area Chart."""
    if df is None or df.empty or target_col not in df.columns: return go.Figure()
    
    df_vol = df.copy().sort_values(by=COL_TANGGAL)
    df_vol['volatility'] = df_vol[target_col].rolling(window=window).std().fillna(0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_vol[COL_TANGGAL], y=df_vol['volatility'],
        fill='tozeroy', mode='lines', line=dict(color='#FF6F61', width=2), name='Volatilitas'
    ))

    fig.update_layout(
        title=title or f"Volatilitas (Rolling {window} Hari)",
        margin={"r":10,"t":40,"l":10,"b":10},
        template=template or DEFAULT_PLOTLY_TEMPLATE
    )
    return fig


def create_inventory_cover_chart(df: pd.DataFrame, days_cover: int = 20, critical_days: int = 10, template: Optional[str] = None, safe_days: int = 20) -> go.Figure:
    """Calculates inventory cover with safety zones."""
    if df is None or df.empty: return go.Figure()
    
    df_calc = df.sort_values(COL_TANGGAL).copy()
    # Average outflow 7 days, avoid zero division
    df_calc['avg_out_7d'] = df_calc[COL_KELUAR].rolling(window=7, min_periods=1).mean().replace(0, 1)
    df_calc['cover_days'] = df_calc[COL_STOK] / df_calc['avg_out_7d']
    
    # Cap y-axis for readability
    y_max = min(df_calc['cover_days'].replace([np.inf, -np.inf], np.nan).max(), 100) 
    if pd.isna(y_max): y_max = 60

    fig = go.Figure()
    # Safety Zones
    fig.add_hrect(y0=0, y1=critical_days, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Kritis", annotation_position="top right")
    fig.add_hrect(y0=critical_days, y1=safe_days, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="Waspada", annotation_position="top right")
    fig.add_hrect(y0=safe_days, y1=y_max*1.5, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Aman", annotation_position="top right")

    fig.add_trace(go.Scatter(x=df_calc[COL_TANGGAL], y=df_calc['cover_days'], mode='lines', name='Cover Days', line=dict(color='#1f77b4', width=3)))

    fig.update_layout(
        title="Ketahanan Stok (Hari)", 
        yaxis=dict(range=[0, y_max], title="Hari"),
        margin={"r":10,"t":40,"l":10,"b":10},
        template=template or DEFAULT_PLOTLY_TEMPLATE
    )
    return fig


def create_stock_distribution(df: pd.DataFrame, template: Optional[str] = None) -> go.Figure:
    """
    Creates a histogram with a marginal violin plot and markers for Mean/Median.
    """
    if df is None or df.empty or "stok" not in df.columns: return go.Figure()
    
    mean_val = df[COL_STOK].mean()
    median_val = df[COL_STOK].median()

    fig = px.histogram(
        df, x=COL_STOK, 
        nbins=40, 
        marginal="violin", # Upgrade: Box -> Violin (Lebih estetik)
        title="Distribusi Stok (Sebaran Data)", 
        template=template or DEFAULT_PLOTLY_TEMPLATE, 
        color_discrete_sequence=['#2E86AB'],
        opacity=0.75
    )
    
    # Menambahkan Garis Penanda Rata-rata & Median
    fig.add_vline(x=mean_val, line_width=2, line_dash="dash", line_color="#FFD700", 
                  annotation_text=f"Rata-rata: {mean_val:,.0f}", annotation_position="top right")
    fig.add_vline(x=median_val, line_width=2, line_dash="dot", line_color="#FF6F61", 
                  annotation_text=f"Median: {median_val:,.0f}", annotation_position="top left")

    fig.update_layout(
        margin={"r":10,"t":50,"l":10,"b":10}, 
        bargap=0.1,
        xaxis_title="Volume Stok (Ton)",
        yaxis_title="Frekuensi (Hari)"
    )
    return fig


def create_regression_scatter(df: pd.DataFrame, x_col: str, y_col: str, template: Optional[str] = None) -> go.Figure:
    """
    Creates a scatter plot with regression line AND marginal distributions.
    """
    if df is None or df.empty: return go.Figure()
    clean = df[[x_col, y_col]].dropna()
    if clean.empty: return go.Figure()
    
    slope, intercept, r_value, p_value, std_err = linregress(clean[x_col], clean[y_col])
    
    # Upgrade: Menambahkan Marginal Histogram di sisi atas dan kanan
    fig = px.scatter(
        clean, x=x_col, y=y_col, 
        trendline="ols", 
        title=f"Analisis Korelasi (R²={r_value**2:.2f})" if r_value is not None else "Analisis Korelasi",  # type: ignore
        template=template or DEFAULT_PLOTLY_TEMPLATE,
        color_discrete_sequence=['#2E86AB'],
        marginal_x="histogram", # Histogram di atas
        marginal_y="histogram", # Histogram di kanan
        opacity=0.6,
        trendline_color_override="#FF6F61"
    )
    
    fig.update_layout(
        margin={"r":10,"t":50,"l":10,"b":10},
        xaxis_title="Volume Stok (Ton)",
        yaxis_title="Harga Beras (Rp)"
    )
    return fig


def create_geo_map(df_flow: pd.DataFrame, geo_lookup: pd.DataFrame, flow_type: str = "masuk", template: Optional[str] = None) -> go.Figure:
    # ... (Fallback function, jarang dipanggil) ...
    return go.Figure() 

def create_pydeck_map(df_agg: pd.DataFrame, flow_type: str = "masuk") -> pdk.Deck:
    # ... (Sama seperti kode sebelumnya, gunakan PyDeck) ...
    if df_agg is None or df_agg.empty: return None # type: ignore
    
    pibc_lat, pibc_lon = -6.218, 106.896
    df_agg['target_lat'] = pibc_lat
    df_agg['target_lon'] = pibc_lon
    
    max_val = df_agg[flow_type].max()
    df_agg['normalized_elevation'] = (df_agg[flow_type] / max_val) * 50000 if max_val > 0 else 0
    df_agg['normalized_width'] = (df_agg[flow_type] / max_val) * 10 if max_val > 0 else 1
    
    source_color = [0, 255, 150, 160] if flow_type == "masuk" else [255, 100, 100, 160]
    target_color = [0, 150, 255, 160] if flow_type == "masuk" else [255, 180, 0, 160]

    arc_layer = pdk.Layer("ArcLayer", data=df_agg, get_source_position=["lon", "lat"], get_target_position=["target_lon", "target_lat"], get_source_color=source_color, get_target_color=target_color, get_width="normalized_width + 2", get_tilt=15, pickable=True, auto_highlight=True)
    column_layer = pdk.Layer("ColumnLayer", data=df_agg, get_position=["lon", "lat"], get_elevation="normalized_elevation", elevation_scale=1, radius=2000, get_fill_color=source_color, pickable=True, extruded=True)
    pibc_layer = pdk.Layer("ScatterplotLayer", data=pd.DataFrame([{'name': 'PIBC', 'lat': pibc_lat, 'lon': pibc_lon}]), get_position=["lon", "lat"], get_color=[255, 255, 255, 200], get_radius=3000, pickable=True)

    tooltip = {"html": f"<b>Lokasi:</b> {{lokasi_lookup}}<br/><b>Volume:</b> {{{flow_type}}} Ton", "style": {"backgroundColor": "steelblue", "color": "white"}}
    
    # Map Style logic
    mapbox_token = None
    try: mapbox_token = st.secrets["mapbox"]["token"]
    except: pass
    
    map_style = "mapbox://styles/mapbox/dark-v10" if mapbox_token else "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
    api_keys = {"mapbox": mapbox_token} if mapbox_token else None

    return pdk.Deck(layers=[column_layer, arc_layer, pibc_layer], initial_view_state=pdk.ViewState(latitude=-6.5, longitude=107.5, zoom=7, pitch=45), tooltip=tooltip, map_style=map_style, api_keys=api_keys) # type: ignore


def create_forecast_chart(df_hist: pd.DataFrame, df_pred: pd.DataFrame, method: str = "Prophet", template: Optional[str] = None) -> go.Figure:
    # ... (Sama seperti sebelumnya) ...
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_hist[FC_COL_DS], y=df_hist[FC_COL_Y], name='Historis', line=dict(color='#2E86AB')))
    if 'yhat' in df_pred.columns:
        fig.add_trace(go.Scatter(x=df_pred[FC_COL_DS], y=df_pred['yhat'], name=f'Forecast ({method})', line=dict(color='#F24236', dash='dash')))
    if method == "Prophet" and 'yhat_upper' in df_pred.columns:
        fig.add_trace(go.Scatter(x=df_pred[FC_COL_DS], y=df_pred['yhat_upper'], fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
        fig.add_trace(go.Scatter(x=df_pred[FC_COL_DS], y=df_pred['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', fillcolor='rgba(200,200,200,0.3)', name='Confidence Interval'))
    fig.update_layout(title=f"Peramalan Stok ({method})", margin={"r":10,"t":40,"l":10,"b":10}, template=template or DEFAULT_PLOTLY_TEMPLATE)
    return fig