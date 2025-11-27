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

# Choose plotly template based on Streamlit theme (fallback)
try:
    _theme = st.get_option("theme.base")
except Exception:
    _theme = None
DEFAULT_PLOTLY_TEMPLATE = "plotly_dark" if _theme == "dark" else "plotly_white"


def _ensure_fig_template(fig: go.Figure, template: Optional[str]):
    fig.layout.template = template or DEFAULT_PLOTLY_TEMPLATE
    return fig


def create_time_series(df: pd.DataFrame, y_col: str, title: str, color: Optional[str] = None, template: Optional[str] = None) -> go.Figure:
    """Line chart for a single y column (tanggal vs y_col)"""
    if df is None or df.empty or y_col not in df.columns:
        return go.Figure()
    fig = px.line(df, x=COL_TANGGAL, y=y_col, title=title, line_shape="spline", color_discrete_sequence=[color] if color else None, template=template or DEFAULT_PLOTLY_TEMPLATE)
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig


def create_balance_chart(df: pd.DataFrame, template: Optional[str] = None) -> go.Figure:
    """Bar (masuk vs keluar) + neraca line"""
    if df is None or df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[COL_TANGGAL], y=df[COL_MASUK], name="Masuk", marker_color="#2E86AB"))
    fig.add_trace(go.Bar(x=df[COL_TANGGAL], y=-df[COL_KELUAR], name="Keluar", marker_color="#F24236"))
    fig.add_trace(go.Scatter(x=df[COL_TANGGAL], y=df[COL_NERACA], name="Neraca", line=dict(color="#222222", width=2, dash="dot")))
    fig.update_layout(title="Neraca Harian (Masuk vs Keluar)", barmode="relative", template=template or DEFAULT_PLOTLY_TEMPLATE, margin={"r":0,"t":40,"l":0,"b":0})
    return fig


def create_price_heatmap(df_price: pd.DataFrame, correlation: bool = False, template: Optional[str] = None) -> go.Figure:
    """Heatmap for price timeseries (tanggal x jenis)"""
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
    """Calculate rolling volatility (std) for each series and plot aggregated/small multiples"""
    if df is None or df.empty or target_col not in df.columns:
        return go.Figure()
    
    df_vol = df.copy()
    if 'tanggal' not in df_vol.columns and isinstance(df_vol.index, pd.DatetimeIndex):
         df_vol = df_vol.reset_index()

    df_vol['volatility'] = df_vol[target_col].rolling(window=window).std()
    
    chart_title = title or f"Volatility (rolling std, window={window})"
    fig = px.line(df_vol, x=COL_TANGGAL, y='volatility', title=chart_title, labels={'volatility': f'Std Dev ({window} Hari)'}, template=template or DEFAULT_PLOTLY_TEMPLATE)
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    fig.update_traces(line_color='#FF6F61')
    return fig


def create_inventory_cover_chart(df: pd.DataFrame, days_cover: int = 20, template: Optional[str] = None) -> go.Figure:
    """Inventory cover: stok / average daily outflow (rolling 7 days)"""
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
    """KDE / histogram of stok values"""
    if df is None or df.empty or "stok" not in df.columns:
        return go.Figure()
    fig = px.histogram(df, x=COL_STOK, nbins=40, marginal="box", title="Distribusi Stok", template=template or DEFAULT_PLOTLY_TEMPLATE)
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig


def create_regression_scatter(df: pd.DataFrame, x_col: str, y_col: str, template: Optional[str] = None) -> go.Figure:
    """Scatter with linear regression overlay and metrics (r, slope)"""
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
    if df_flow is None or df_flow.empty:
        return go.Figure()
    if geo_lookup is None or geo_lookup.empty:
        return go.Figure()
    # normalize lookup and flows
    gl = geo_lookup.copy()
    gl = gl.rename(columns={"lokasi": "lokasi_lookup"})
    gl[COL_LOKASI_NORM] = gl["lokasi_lookup"].astype(str).str.strip().str.lower()

    ff = df_flow.copy()
    if COL_LOKASI not in ff.columns:
        ff[COL_LOKASI] = ff.get("lokasi_lookup", "unknown")
    ff[COL_LOKASI_NORM] = ff[COL_LOKASI].astype(str).str.strip().str.lower()

    df_map = ff.merge(gl, on="lokasi_norm", how="inner")
    if df_map.empty:
        # return empty figure but not error
        return go.Figure()
    # aggregate
    df_agg = df_map.groupby(["lokasi_norm", "lokasi_lookup", "lat", "lon"], as_index=False)[flow_type].sum()
    df_agg["size"] = df_agg[flow_type].apply(lambda v: max(4, math.log1p(v + 1) * 4))
    fig = px.scatter_mapbox(df_agg, lat="lat", lon="lon", size="size", color=flow_type,
                            hover_name="lokasi_lookup", hover_data=[flow_type], zoom=5, center={"lat": -6.8, "lon": 108},
                            color_continuous_scale="Greens" if flow_type == "masuk" else "Reds",
                            template=template or DEFAULT_PLOTLY_TEMPLATE)
    fig.update_layout(mapbox_style="open-street-map", title=f"Peta {'Asal' if flow_type=='masuk' else 'Distribusi'} Beras", margin={"r":0,"t":40,"l":0,"b":0})
    return fig