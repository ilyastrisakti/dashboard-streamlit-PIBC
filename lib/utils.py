import pandas as pd
from typing import Optional

def _clean_colname(c: Optional[str]) -> str:
    if c is None:
        return ""
    return str(c).replace("\n", " ").strip()

def price_df_with_tanggal(df_price: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df_price is None:
        return None
    p = df_price.copy()
    p.columns = [_clean_colname(c) for c in p.columns]
    if isinstance(p.index, pd.DatetimeIndex) or (p.index.name and str(p.index.name).lower() == "tanggal"):
        p = p.reset_index()
        if 'index' in p.columns:
            p = p.rename(columns={'index':'tanggal'})
    else:
        date_col = next((c for c in p.columns if c.lower() in ("date", "tanggal", "tgl", "hari")), None)
        if date_col:
            p = p.rename(columns={date_col: "tanggal"})
        else:
            p["tanggal"] = pd.to_datetime(p.index, errors="coerce")
    p["tanggal"] = pd.to_datetime(p["tanggal"], errors="coerce")
    return p

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')