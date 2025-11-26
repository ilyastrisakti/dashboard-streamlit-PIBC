# new file
import math
from typing import Optional
import pandas as pd
import numpy as np
from scipy.stats import linregress
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
    fig = px.line(df, x="tanggal", y=y_col, title=title, line_shape="spline", color_discrete_sequence=[color] if color else None, template=template or DEFAULT_PLOTLY_TEMPLATE)
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig


def create_balance_chart(df: pd.DataFrame, template: Optional[str] = None) -> go.Figure:
    """Bar (masuk vs keluar) + neraca line"""
    if df is None or df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["tanggal"], y=df["masuk"], name="Masuk", marker_color="#2E86AB"))
    fig.add_trace(go.Bar(x=df["tanggal"], y=-df["keluar"], name="Keluar", marker_color="#F24236"))
    fig.add_trace(go.Scatter(x=df["tanggal"], y=df["neraca"], name="Neraca", line=dict(color="#222222", width=2, dash="dot")))
    fig.update_layout(title="Neraca Harian (Masuk vs Keluar)", barmode="relative", template=template or DEFAULT_PLOTLY_TEMPLATE, margin={"r":0,"t":40,"l":0,"b":0})
    return fig


def create_price_heatmap(df_price: pd.DataFrame, template: Optional[str] = None) -> go.Figure:
    """Heatmap for price timeseries (tanggal x jenis)"""
    if df_price is None or df_price.empty:
        return go.Figure()
    # ensure date index -> columns if needed
    p = df_price.copy()
    if isinstance(p.index, pd.DatetimeIndex) and "tanggal" not in p.columns:
        p = p.reset_index()
    # pivot if necessary
    if "tanggal" in p.columns:
        p = p.set_index("tanggal")
    zvals = p.select_dtypes(include=[np.number]).T.values
    xvals = p.index
    yvals = p.select_dtypes(include=[np.number]).columns
    fig = go.Figure(data=go.Heatmap(
        x=xvals, y=yvals, z=zvals, colorscale="RdYlGn_r", colorbar=dict(title="Harga")
    ))
    fig.update_layout(title="Heatmap Harga per Jenis", template=template or DEFAULT_PLOTLY_TEMPLATE, xaxis_title="Tanggal", yaxis_title="Jenis")
    return fig


def create_volatility_chart(df_price: pd.DataFrame, window: int = 7, template: Optional[str] = None) -> go.Figure:
    """Calculate rolling volatility (std) for each series and plot aggregated/small multiples"""
    if df_price is None or df_price.empty:
        return go.Figure()
    p = df_price.select_dtypes(include=[np.number]).copy()
    rol = p.rolling(window=window).std()
    agg = rol.mean(axis=1).reset_index(name="volatility")
    if "tanggal" in agg.columns:
        xcol = "tanggal"
    else:
        agg = agg.rename(columns={agg.columns[0]: "tanggal"})
        xcol = "tanggal"
    fig = px.line(agg, x=xcol, y="volatility", title=f"Volatility (rolling std, window={window})", template=template or DEFAULT_PLOTLY_TEMPLATE)
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig


def create_inventory_cover_chart(df: pd.DataFrame, days_cover: int = 30, template: Optional[str] = None) -> go.Figure:
    """Inventory cover: stok / average daily outflow * days_cover"""
    if df is None or df.empty:
        return go.Figure()
    df = df.sort_values("tanggal").copy()
    # compute rolling mean keluar
    df["daily_outflow_mean"] = df["keluar"].rolling(window=days_cover, min_periods=1).mean()
    df["cover_days"] = np.where(df["daily_outflow_mean"] > 0, df["stok"] / df["daily_outflow_mean"], np.nan)
    fig = px.line(df, x="tanggal", y="cover_days", title=f"Inventory Cover (rolling {days_cover}d avg outflow)", template=template or DEFAULT_PLOTLY_TEMPLATE)
    fig.update_layout(yaxis_title="Days of Cover", margin={"r":0,"t":40,"l":0,"b":0})
    return fig


def create_stock_distribution(df: pd.DataFrame, template: Optional[str] = None) -> go.Figure:
    """KDE / histogram of stok values"""
    if df is None or df.empty or "stok" not in df.columns:
        return go.Figure()
    fig = px.histogram(df, x="stok", nbins=40, marginal="box", title="Distribusi Stok", template=template or DEFAULT_PLOTLY_TEMPLATE)
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
    gl["lokasi_norm"] = gl["lokasi_lookup"].astype(str).str.strip().str.lower()

    ff = df_flow.copy()
    if "lokasi" not in ff.columns:
        ff["lokasi"] = ff.get("lokasi_lookup", "unknown")
    ff["lokasi_norm"] = ff["lokasi"].astype(str).str.strip().str.lower()

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