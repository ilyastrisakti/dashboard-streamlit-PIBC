# -*- coding: utf-8 -*-
"""
Centralized constants for the application.
This helps avoid "magic strings" and makes maintenance easier.
"""

# --- Column Names ---
COL_TANGGAL = "tanggal"
COL_STOK = "stok"
COL_MASUK = "masuk"
COL_KELUAR = "keluar"
COL_NERACA = "neraca"
COL_LOKASI = "lokasi"
COL_LOKASI_NORM = "lokasi_norm"
COL_HARGA = "harga"
COL_NAMA_JENIS = "nama_jenis"

# --- Forecasting Column Names (Prophet) ---
FC_COL_DS = "ds"
FC_COL_Y = "y"

# --- Excel Sheet Names ---
SHEET_RICE_STOCK = "rice_stock"
SHEET_RICE_DELIVERY = "rice_delivery"
SHEET_RICE_SOURCE = "rice_source"
SHEET_RICE_PRICE = "rice_price"

# --- Default Values ---
DEFAULT_UNKNOWN_LOCATION = "Unknown"