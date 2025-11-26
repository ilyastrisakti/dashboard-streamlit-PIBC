import pandas as pd
from plotly.graph_objs import Figure
from lib.plots import create_geo_map

def test_geo_map_merge_and_agg():
    df_flow = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01","2025-01-01"]),
        "lokasi": ["Bandung","bandung"],
        "lokasi_norm": ["bandung","bandung"],
        "masuk": [10, 15]
    })
    geo_lookup = pd.DataFrame({
        "lokasi": ["Bandung"],
        "lat": [-6.9175],
        "lon": [107.6191]
    fig = create_geo_map(df_flow, geo_lookup, flow_type="masuk")
    # figure should contain aggregated marker size / color data
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) >= 1
    assert len(fig.data) >= 1
    marker_sizes = fig.data[0].marker.size
    marker_colors = fig.data[0].marker.color
    assert len(marker_sizes) == 1
    assert len(marker_colors) == 1
    # total masuk should be 25
    assert sum(marker_sizes) == 25
def test_geo_map_no_data():
    fig = create_geo_map(None, None)
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) == 0
    fig = create_geo_map(pd.DataFrame(), pd.DataFrame())
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) == 0
def test_geo_map_no_matching_locations():
    df_flow = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01"]),
        "lokasi": ["Unknown"],
        "lokasi_norm": ["unknown"],
        "masuk": [10]
    })
    geo_lookup = pd.DataFrame({
        "lokasi": ["Bandung"],
        "lat": [-6.9175],
        "lon": [107.6191]
    })
    fig = create_geo_map(df_flow, geo_lookup, flow_type="masuk")
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) == 0
def test_geo_map_empty_after_merge():
    df_flow = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01"]),
        "lokasi": ["Nonexistent"],
        "lokasi_norm": ["nonexistent"],
        "masuk": [10]
    })
    geo_lookup = pd.DataFrame({
        "lokasi": ["AlsoNonexistent"],
        "lat": [0],
        "lon": [0]
    })
    fig = create_geo_map(df_flow, geo_lookup, flow_type="masuk")
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) == 0
def test_geo_map_missing_columns():
    df_flow = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01"]),
        "lokasi": ["Bandung"],
        # missing lokasi_norm
        "masuk": [10]
    })
    geo_lookup = pd.DataFrame({
        "lokasi": ["Bandung"],
        "lat": [-6.9175],
        "lon": [107.6191]
    })
    fig = create_geo_map(df_flow, geo_lookup, flow_type="masuk")
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) == 0
def test_geo_map_flow_type_not_in_df():
    df_flow = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01"]),
        "lokasi": ["Bandung"],
        "lokasi_norm": ["bandung"],
        # missing 'masuk' column
        "keluar": [5]
    })
    geo_lookup = pd.DataFrame({
        "lokasi": ["Bandung"],
        "lat": [-6.9175],
        "lon": [107.6191]
    })
    fig = create_geo_map(df_flow, geo_lookup, flow_type="masuk")
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) == 0
def test_geo_map_flow_type_not_in_geo_lookup():
    df_flow = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01"]),
        "lokasi": ["Bandung"],
        "lokasi_norm": ["bandung"],
        "masuk": [10]
    })
    geo_lookup = pd.DataFrame({
        "lokasi": ["Jakarta"],
        "lat": [-6.2088],
        "lon": [106.8456]
    })
    fig = create_geo_map(df_flow, geo_lookup, flow_type="masuk")
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) == 0
def test_geo_map_multiple_locations():
    df_flow = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01","2025-01-01","2025-01-01"]),
        "lokasi": ["Bandung","Jakarta","Surabaya"],
        "lokasi_norm": ["bandung","jakarta","surabaya"],
        "masuk": [10, 20, 15]
    })
    geo_lookup = pd.DataFrame({
        "lokasi": ["Bandung","Jakarta","Surabaya"],
        "lat": [-6.9175, -6.2088, -7.2575],
        "lon": [107.6191, 106.8456, 112.7521]
    })
    fig = create_geo_map(df_flow, geo_lookup, flow_type="masuk")
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) == 1
    marker_sizes = fig.data[0].marker.size
    marker_colors = fig.data[0].marker.color
    assert len(marker_sizes) == 3
    assert len(marker_colors) == 3
    assert sum(marker_sizes) == 45  # 10+20+15
    assert sum(marker_colors) == 45  # 10+20+15
def test_geo_map_zero_flows():
    df_flow = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01","2025-01-01"]),
        "lokasi": ["Bandung","Jakarta"],
        "lokasi_norm": ["bandung","jakarta"],
        "masuk": [0, 0]
    })
    geo_lookup = pd.DataFrame({
        "lokasi": ["Bandung","Jakarta"],
        "lat": [-6.9175, -6.2088],
        "lon": [107.6191, 106.8456]
    })
    fig = create_geo_map(df_flow, geo_lookup, flow_type="masuk")
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) == 1
    marker_sizes = fig.data[0].marker.size
    marker_colors = fig.data[0].marker.color
    assert len(marker_sizes) == 2
    assert len(marker_colors) == 2
    assert sum(marker_sizes) == 0
    assert sum(marker_colors) == 0
def test_geo_map_large_flows():
    df_flow = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01","2025-01-01"]),
        "lokasi": ["Bandung","Jakarta"],
        "lokasi_norm": ["bandung","jakarta"],
        "masuk": [1000000, 5000000]
    })
    geo_lookup = pd.DataFrame({
        "lokasi": ["Bandung","Jakarta"],
        "lat": [-6.9175, -6.2088],
        "lon": [107.6191, 106.8456]
    })
    fig = create_geo_map(df_flow, geo_lookup, flow_type="masuk")
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) == 1
    marker_sizes = fig.data[0].marker.size
    marker_colors = fig.data[0].marker.color
    assert len(marker_sizes) == 2
    assert len(marker_colors) == 2
    assert sum(marker_sizes) > 0
    assert sum(marker_colors) == 6000000
def test_geo_map_negative_flows():
    df_flow = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01","2025-01-01"]),
        "lokasi": ["Bandung","Jakarta"],
        "lokasi_norm": ["bandung","jakarta"],
        "masuk": [-10, -20]
    })
    geo_lookup = pd.DataFrame({
        "lokasi": ["Bandung","Jakarta"],
        "lat": [-6.9175, -6.2088],
        "lon": [107.6191, 106.8456]
    })
    fig = create_geo_map(df_flow, geo_lookup, flow_type="masuk")
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) == 1
    marker_sizes = fig.data[0].marker.size
    marker_colors = fig.data[0].marker.color
    assert len(marker_sizes) == 2
    assert len(marker_colors) == 2
    assert sum(marker_sizes) == 0
    assert sum(marker_colors) == -30
def test_geo_map_single_entry():
    df_flow = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01"]),
        "lokasi": ["Bandung"],
        "lokasi_norm": ["bandung"],
        "masuk": [10]
    })
    geo_lookup = pd.DataFrame({
        "lokasi": ["Bandung"],
        "lat": [-6.9175],
        "lon": [107.6191]
    })
    fig = create_geo_map(df_flow, geo_lookup, flow_type="masuk")
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) == 1
    marker_sizes = fig.data[0].marker.size
    marker_colors = fig.data[0].marker.color
    assert len(marker_sizes) == 1
    assert len(marker_colors) == 1
    assert sum(marker_sizes) == 10
    assert sum(marker_colors) == 10
def test_geo_map_no_lat_lon():
    df_flow = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01"]),
        "lokasi": ["Bandung"],
        "lokasi_norm": ["bandung"],
        "masuk": [10]
    })
    geo_lookup = pd.DataFrame({
        "lokasi": ["Bandung"]
        # missing lat, lon
    })
    fig = create_geo_map(df_flow, geo_lookup, flow_type="masuk")
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) == 0
def test_geo_map_no_lokasi_column_in_flow():
    df_flow = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01"]),
        # missing lokasi column
        "lokasi_norm": ["bandung"],
        "masuk": [10]
    })
    geo_lookup = pd.DataFrame({
        "lokasi": ["Bandung"],
        "lat": [-6.9175],
        "lon": [107.6191]
    })
    fig = create_geo_map(df_flow, geo_lookup, flow_type="masuk")
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) == 0
def test_geo_map_no_lokasi_column_in_lookup():
    df_flow = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01"]),
        "lokasi": ["Bandung"],
        "lokasi_norm": ["bandung"],
        "masuk": [10]
    })
    geo_lookup = pd.DataFrame({
        # missing lokasi column
        "lat": [-6.9175],
        "lon": [107.6191]
    })
    fig = create_geo_map(df_flow, geo_lookup, flow_type="masuk")
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) == 0
def test_geo_map_non_string_location_names():
    df_flow = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01","2025-01-01"]),
        "lokasi": [123, 456],
        "lokasi_norm": ["123", "456"],
        "masuk": [10, 20]
    })
    geo_lookup = pd.DataFrame({
        "lokasi": ["123","456"],
        "lat": [-6.9175, -6.2088],
        "lon": [107.6191, 106.8456]
    })
    fig = create_geo_map(df_flow, geo_lookup, flow_type="masuk")
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.data) == 1
    marker_sizes = fig.data[0].marker.size
    marker_colors = fig.data[0].marker.color
    assert len(marker_sizes) == 2
    assert len(marker_colors) == 2
    assert sum(marker_sizes) == 30  # 10+20