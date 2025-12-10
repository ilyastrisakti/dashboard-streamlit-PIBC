# new file
import math
from typing import Optional
import pandas as pd
import numpy as np
from scipy.stats import linregress
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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
    Creates a simple line chart for a single time-series column.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        y_col (str): The column to plot on the y-axis.
        title (str): The chart title.
        color (Optional[str], optional): Line color.
        template (Optional[str], optional): Plotly template.
        is_daily (bool): If False, uses a step-like line shape ('hv').
    """
    if df is None or df.empty or y_col not in df.columns:
        return go.Figure()
    
    line_shape = "spline" if is_daily else "hv"
    
    # Using px.area instead of px.line to create a filled area chart for better visual appeal
    fig = px.area(df, x=COL_TANGGAL, y=y_col, title=title, line_shape=line_shape, color_discrete_sequence=[color] if color else None, template=template or DEFAULT_PLOTLY_TEMPLATE)
    
    # Customize the fill to be a semi-transparent gradient
    if fig.data:
        line_color = fig.data[0].line.color
        fig.update_traces(
            fillcolor=f"rgba({int(line_color[1:3], 16)}, {int(line_color[3:5], 16)}, {int(line_color[5:7], 16)}, 0.2)" if line_color and '#' in line_color else "rgba(0,0,0,0.1)",
            mode='lines+markers' if not is_daily else 'lines' # Add markers for non-daily data
        )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    
    return fig


def create_balance_chart(df: pd.DataFrame, template: Optional[str] = None, granularity: str = "Harian") -> go.Figure:
    """
    Creates a balance chart with bars for 'masuk'/'keluar' and a line for 'neraca'.
    """
    if df is None or df.empty:
        return go.Figure()
        
    # Determine bar width based on granularity to make them look correct
    if granularity == "Bulanan":
        # Approx. 30 days in milliseconds
        width = 30 * 24 * 60 * 60 * 1000
    elif granularity == "Tahunan":
        # Approx. 365 days in milliseconds
        width = 365 * 24 * 60 * 60 * 1000
    else: # Harian
        width = None # Let Plotly decide

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[COL_TANGGAL], y=df[COL_MASUK], name="Masuk", marker_color="#2E86AB", width=width, opacity=0.7))
    fig.add_trace(go.Bar(x=df[COL_TANGGAL], y=-df[COL_KELUAR], name="Keluar", marker_color="#F24236", width=width, opacity=0.7))
    fig.add_trace(go.Scatter(x=df[COL_TANGGAL], y=df[COL_NERACA], name="Neraca", mode='lines+markers', line=dict(color="#FFA500", width=3), marker=dict(size=5)))
    fig.update_layout(title=f"Neraca {granularity} (Masuk vs Keluar)", barmode="relative", template=template or DEFAULT_PLOTLY_TEMPLATE, margin={"r":0,"t":40,"l":0,"b":0})
    return fig


def create_price_heatmap(df_price: pd.DataFrame, correlation: bool = False, template: Optional[str] = None) -> go.Figure:
    """
    Creates a heatmap for rice prices. Can show raw prices over time or a correlation matrix.
    """
    if df_price is None or df_price.empty:
        return go.Figure()
    
    if correlation:
        corr = df_price.corr()
        fig = px.imshow(corr,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdYlGn',
                        title="Heatmap Korelasi Harga Antar Jenis Beras")
        fig.data[0].texttemplate = "%{z:.2f}"
    else:
        p = df_price.copy()
        if isinstance(p.index, pd.DatetimeIndex) and "tanggal" not in p.columns:
            p = p.reset_index()
        if COL_TANGGAL in p.columns:
            p = p.set_index(COL_TANGGAL)
        zvals = p.select_dtypes(include=[np.number]).T.values
        xvals = p.index
        yvals = p.select_dtypes(include=[np.number]).columns
        fig = go.Figure(data=go.Heatmap(x=xvals, y=yvals, z=zvals, colorscale="RdYlGn_r", colorbar=dict(title="Harga")))
        fig.update_layout(title="Heatmap Harga per Jenis", xaxis_title="Tanggal", yaxis_title="Jenis")
    
    fig = _ensure_fig_template(fig, template)
    return fig


def create_volatility_chart(df: pd.DataFrame, target_col: str, window: int = 7, template: Optional[str] = None, title: Optional[str] = None) -> go.Figure:
    """
    Calculates and plots the rolling volatility (standard deviation) for a target column.
    """
    if df is None or df.empty or target_col not in df.columns:
        return go.Figure()
    
    df_vol = df.copy()
    if 'tanggal' not in df_vol.columns and isinstance(df_vol.index, pd.DatetimeIndex):
         df_vol = df_vol.reset_index()

    # Sort by date BEFORE calculating rolling std to ensure correctness
    df_vol = df_vol.sort_values(by=COL_TANGGAL)

    df_vol['volatility'] = df_vol[target_col].rolling(window=window).std()
    
    chart_title = title or f"Volatility (rolling std, window={window})"
    fig = px.line(df_vol, x=COL_TANGGAL, y='volatility', title=chart_title, labels={'volatility': f'Std Dev ({window} Hari)'}, template=template or DEFAULT_PLOTLY_TEMPLATE)
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    fig.update_traces(line_color='#FF6F61')
    return fig


def create_inventory_cover_chart(df: pd.DataFrame, days_cover: int = 20, template: Optional[str] = None) -> go.Figure:
    """
    Calculates and plots inventory cover days based on stock and rolling average outflow.
    """
    if df is None or df.empty:
        return go.Figure()
    df_calc = df.sort_values(COL_TANGGAL).copy()
    # Rolling avg of outcome (7 days)
    df_calc['avg_out_7d'] = df_calc[COL_KELUAR].rolling(window=7, min_periods=1).mean()
    # Avoid division by zero
    df_calc['avg_out_7d'] = df_calc['avg_out_7d'].replace(0, 1) 
    
    df_calc['cover_days'] = df_calc[COL_STOK] / df_calc['avg_out_7d']
    
    fig = px.line(df_calc, x=COL_TANGGAL, y='cover_days', title="Inventory Cover Days (Ketahanan Stok)", template=template or DEFAULT_PLOTLY_TEMPLATE)
    fig.add_hline(y=days_cover, line_dash="dot", annotation_text=f"Aman ({days_cover} hari)", annotation_position="bottom right")
    fig.update_layout(yaxis_title="Hari", margin={"r":0,"t":40,"l":0,"b":0})
    return fig


def create_stock_distribution(df: pd.DataFrame, template: Optional[str] = None) -> go.Figure:
    """
    Creates a histogram with a marginal box plot to show the distribution of stock values.
    """
    if df is None or df.empty or "stok" not in df.columns:
        return go.Figure()
    fig = px.histogram(df, x=COL_STOK, nbins=40, marginal="box", title="Distribusi Stok", template=template or DEFAULT_PLOTLY_TEMPLATE)
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig


def create_regression_scatter(df: pd.DataFrame, x_col: str, y_col: str, template: Optional[str] = None) -> go.Figure:
    """
    Creates a scatter plot with a linear regression trendline.
    """
    if df is None or df.empty or x_col not in df.columns or y_col not in df.columns:
        return go.Figure()
    clean = df[[x_col, y_col]].dropna()
    if clean.empty:
        return go.Figure()
    slope, intercept, r_value, p_value, std_err = linregress(clean[x_col], clean[y_col])
    fig = px.scatter(clean, x=x_col, y=y_col, trendline="ols", title=f"Scatter {y_col} vs {x_col} — r={r_value:.3f}", template=template or DEFAULT_PLOTLY_TEMPLATE)
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig


def create_geo_map(df_flow: pd.DataFrame, geo_lookup: pd.DataFrame, flow_type: str = "masuk", template: Optional[str] = None) -> go.Figure:
    """
    Create Mapbox scatter showing flows per location.
    df_flow must contain columns: tanggal, lokasi, lokasi_norm, <flow_type>
    geo_lookup must contain columns: lokasi (display name), lat, lon
    """
    if df_flow is None or df_flow.empty or geo_lookup is None or geo_lookup.empty:
        return go.Figure()

    # --- 1. Data Preparation ---
    # Define PIBC as the central point
    pibc_lat, pibc_lon = -6.218, 106.896 # Approx. coordinates for Pasar Induk Beras Cipinang
    pibc_name = "Pasar Induk Beras Cipinang"

    # Normalize lookup table and flow data
    gl = geo_lookup.copy()
    gl = gl.rename(columns={"lokasi": "lokasi_lookup"})
    gl[COL_LOKASI_NORM] = gl["lokasi_lookup"].astype(str).str.strip().str.lower()

    ff = df_flow.copy()
    if COL_LOKASI not in ff.columns:
        return go.Figure() # Cannot proceed without location column
    ff[COL_LOKASI_NORM] = ff[COL_LOKASI].astype(str).str.strip().str.lower()

    # Merge flow data with geo coordinates
    df_map = ff.merge(gl, on="lokasi_norm", how="inner")
    if df_map.empty:
        return go.Figure()

    # Aggregate data by location to get total flow
    df_agg = df_map.groupby(["lokasi_norm", "lokasi_lookup", "lat", "lon"], as_index=False)[flow_type].sum()

    # --- 2. Create Figure and Layers ---
    fig = go.Figure()

    # Layer 1: Flow Lines (from/to PIBC)
    # We need to create a dataframe for lines where each row is a point (start or end)
    line_data = []
    for i, row in df_agg.iterrows():
        line_data.append({'lat': row['lat'], 'lon': row['lon'], 'lokasi': row['lokasi_lookup'], 'volume': row[flow_type]})
        line_data.append({'lat': pibc_lat, 'lon': pibc_lon, 'lokasi': row['lokasi_lookup'], 'volume': row[flow_type]})
        line_data.append({'lat': None, 'lon': None, 'lokasi': None, 'volume': None}) # Break between lines

    df_lines = pd.DataFrame(line_data)

    fig.add_trace(go.Scattermapbox(
        mode="lines",
        lat=df_lines['lat'],
        lon=df_lines['lon'],
        line=dict(width=1, color="#888"),
        name="Alur Distribusi",
        hoverinfo='none'
    ))

    # Layer 2: Location Points (Bubbles)
    fig.add_trace(go.Scattermapbox(
        lat=df_agg['lat'],
        lon=df_agg['lon'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=df_agg[flow_type].apply(lambda v: max(5, math.log1p(v) * 3)),
            color=df_agg[flow_type],
            colorscale="Greens" if flow_type == "masuk" else "Reds",
            colorbar_title=f"Volume ({flow_type.capitalize()})",
            showscale=True,
            opacity=0.7
        ),
        hovertext=df_agg.apply(lambda row: f"{row['lokasi_lookup']}<br>Volume: {row[flow_type]:,.0f} Ton", axis=1),
        hoverinfo="text",
        name="Lokasi"
    ))

    # Layer 3: Central Point (PIBC)
    fig.add_trace(go.Scattermapbox(
        lat=[pibc_lat],
        lon=[pibc_lon],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=15,
            color='cyan',
            symbol='star'
        ),
        hovertext=pibc_name,
        hoverinfo="text",
        name="PIBC"
    ))

    # --- 3. Layout and Styling ---
    map_style = "carto-darkmatter" if (template or DEFAULT_PLOTLY_TEMPLATE) == "plotly_dark" else "carto-positron"
    
    fig.update_layout(
        title=f"Peta Aliran Beras {'Masuk' if flow_type=='masuk' else 'Keluar'}",
        mapbox_style=map_style,
        mapbox_center_lat=-6.5,
        mapbox_center_lon=108,
        mapbox_zoom=6,
        margin={"r":0,"t":40,"l":0,"b":0},
        showlegend=False
    )

    return fig

# Tambahkan import constants di bagian atas jika belum lengkap
from .constants import FC_COL_DS, FC_COL_Y

def create_forecast_chart(df_hist: pd.DataFrame, df_pred: pd.DataFrame, method: str = "Prophet", template: Optional[str] = None) -> go.Figure:
    """
    Membuat grafik peramalan gabungan antara data historis dan prediksi.
    Mendukung visualisasi area confidence interval jika metode adalah Prophet.
    """
    fig = go.Figure()
    
    # Plot Data Historis
    fig.add_trace(go.Scatter(
        x=df_hist[FC_COL_DS], 
        y=df_hist[FC_COL_Y], 
        name='Historis',
        line=dict(color='#2E86AB')
    ))
    
    # Plot Data Prediksi
    # Pastikan nama kolom prediksi konsisten ('yhat')
    if 'yhat' in df_pred.columns:
        fig.add_trace(go.Scatter(
            x=df_pred[FC_COL_DS], 
            y=df_pred['yhat'], 
            name=f'Forecast ({method})',
            line=dict(color='#F24236', dash='dash')
        ))
    
    # Khusus Prophet: Plot Confidence Interval (yhat_upper & yhat_lower)
    if method == "Prophet" and 'yhat_upper' in df_pred.columns and 'yhat_lower' in df_pred.columns:
        fig.add_trace(go.Scatter(
            x=df_pred[FC_COL_DS], 
            y=df_pred['yhat_upper'], 
            fill=None, 
            mode='lines', 
            line_color='lightgrey', 
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df_pred[FC_COL_DS], 
            y=df_pred['yhat_lower'], 
            fill='tonexty', 
            mode='lines', 
            line_color='lightgrey', 
            name='Confidence Interval'
        ))
        
    fig.update_layout(
        title=f"Peramalan Stok Menggunakan Metode {method}",
        xaxis_title="Tanggal",
        yaxis_title="Stok (Ton)",
        margin={"r":0,"t":40,"l":0,"b":0},
        template=template or DEFAULT_PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig