import pandas as pd
import io
from lib.data import preprocess_data_from_excel

def _make_sample_excel_bytes():
    out = io.BytesIO()
    # stock
    df_stock = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01","2025-01-02"]),
        "stok": [100, 150]
    })
    # source (wide)
    df_src = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01","2025-01-02"]),
        "Karawang": [10, 5],
        "Bandung": [20, 10]
    })
    # delivery (wide)
    df_del = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01","2025-01-02"]),
        "Bogor": [8, 4],
        "Tangerang": [7, 3]
    })
    with pd.ExcelWriter(out) as w:
        df_stock.to_excel(w, sheet_name="rice_stock", index=False)
        df_src.to_excel(w, sheet_name="rice_source", index=False)
        df_del.to_excel(w, sheet_name="rice_delivery", index=False)
    out.seek(0)
    return out

def _make_invalid_excel_bytes():
    """Creates an in-memory Excel file that is missing the required 'rice_stock' sheet."""
    out = io.BytesIO()
    # This file only contains a non-essential sheet
    df_src = pd.DataFrame({
        "tanggal": pd.to_datetime(["2025-01-01"]),
        "Karawang": [10],
    })
    with pd.ExcelWriter(out) as w:
        df_src.to_excel(w, sheet_name="rice_source", index=False)
    out.seek(0)
    return out

def test_preprocess_excel_basic():
    """Tests preprocessing with a valid Excel file containing all required sheets."""
    buf = _make_sample_excel_bytes()
    df_main, df_stock, df_masuk, df_keluar, df_price = preprocess_data_from_excel(buf)
    assert df_main is not None
    # total masuk (sum of source values)
    assert int(df_main["masuk"].sum()) == 45  # 10+20 + 5+10
    assert int(df_main["keluar"].sum()) == 22  # 8+7 + 4+3

def test_preprocess_excel_missing_required_sheet():
    """Tests that preprocessing fails gracefully (returns None) when the required sheet is missing."""
    buf = _make_invalid_excel_bytes()
    # The function should return a tuple of Nones if the 'rice_stock' sheet is missing
    result = preprocess_data_from_excel(buf)
    assert result[0] is None, "df_main should be None when rice_stock sheet is missing"