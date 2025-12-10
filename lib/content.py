# lib/content.py
# -*- coding: utf-8 -*-

"""
Kumpulan teks penjelasan untuk Dashboard PIBC.
Bahasa disesuaikan untuk masyarakat umum/pedagang (Non-Teknis).
"""

# --- SIDEBAR ---
TXT_SIDEBAR_HELP = """
**Tips:**
Gunakan filter tanggal di sini. Jangan tarik data dari tahun lama biar aplikasi tidak lemot.
"""

# --- TAB 1: DASHBOARD UTAMA ---
TXT_HOME_HEADER = "ℹ️ Panduan Membaca Grafik (Klik disini)"
TXT_HOME_BODY = """
**1. Grafik Stok (Hijau)**
Ini ibarat **"Isi Gudang Kita"**.
* 📈 **Naik:** Barang sedang banyak (Panen Raya atau barang tidak keluar).
* 📉 **Turun:** Stok menipis. Hati-hati kelangkaan!

**2. Grafik Neraca (Biru & Merah)**
Ini ibarat **"Buku Kas Keluar-Masuk"**.
* 🚛 **Biru (Masuk):** Kiriman beras datang dari daerah.
* 🚚 **Merah (Keluar):** Beras laku/dikirim ke pasar turunan.
* 🟡 **Titik Kuning:** Sisa hari itu (Masuk dikurangi Keluar).

**3. Grafik Harga (Biru Tua)**
* Melihat apakah harga sedang **Mahal (Naik)** atau **Murah (Turun)**.
* *Tips:* Bandingkan dengan Stok. Biasanya kalau Stok (Hijau) turun, Harga (Biru) akan naik.
"""

# --- TAB 2: PETA ---
TXT_MAP_HEADER = "ℹ️ Cara Pakai Peta 3D"
TXT_MAP_BODY = """
Peta ini menunjukkan dari mana beras datang atau ke mana beras pergi.

**Cara Main:**
* **Putar Peta:** Tahan tombol `Ctrl` di keyboard + Klik Kiri mouse + Geser.
* **Zoom:** Putar roda mouse.
* **Tiang Tinggi:** Artinya volumenya besar (Ribuan Ton).

**Tentang Tabel di Bawah Peta:**
Tabel itu adalah rincian angkanya. Anda bisa melihat **Nama Daerah** dan **Jumlah Ton** persisnya. Klik judul kolom `Volume` untuk mengurutkan dari yang terbanyak.
"""

# --- TAB 3: ANALISIS LANJUTAN ---
TXT_ANALYSIS_HEADER = "📖 Kamus Istilah Pasar"
TXT_ANALYSIS_BODY = """
**1. Inventory Cover (Ketahanan Gudang)**
Bahasa pasarnya: *"Kalau besok kiriman stop total, stok di gudang cukup buat jualan berapa hari?"*
* 🟢 **Aman:** Cukup buat > 20 hari. Tidur nyenyak.
* 🟡 **Waspada:** Cuma cukup 10-20 hari. Mulai cari barang.
* 🔴 **Bahaya:** Kurang dari 10 hari. Siap-siap harga loncat.

**2. Volatilitas (Kestabilan)**
Ini mengukur **"Kepanikan Pasar"**.
* **Grafik Datar/Rendah:** Pasar tenang, harga anteng.
* **Grafik Tinggi/Melonjak:** Pasar liar. Harga atau stok naik-turun drastis. Pedagang pusing.

**3. Matriks Korelasi (Hubungan Antar Beras)**
Ini melihat **"Siapa Berteman dengan Siapa"**.
* 🟩 **Hijau (Teman Akrab):** Kalau harga Beras A naik, Beras B **pasti ikut naik**.
* 🟥 **Merah (Musuhan):** Kalau harga Beras A naik, Beras B **malah turun** (atau sebaliknya).
* 🟨 **Kuning (Orang Asing):** Tidak ada hubungan. Harga jalan sendiri-sendiri.
"""

# --- TAB 4: STATISTIK ---
TXT_STATS_HEADER = "📊 Cara Baca Rapor Statistik (Mudah)"
TXT_STATS_BODY = """
Tabel di sebelah kiri adalah **"Rapor Kinerja Gudang"**. Jangan pusing dengan istilah Inggrisnya, ini artinya:

* **count (Jumlah Hari):** Berapa hari pasar buka dalam periode ini.
* **mean (Rata-rata):** Isi gudang **biasanya** segini. Ini angka normalnya.
* **std (Tingkat Goyang):** Seberapa **labil** stok kita.
    * *Angka Kecil:* Stok stabil (segitu-gitu aja).
    * *Angka Besar:* Stok sering kaget (kadang banjir, kadang langka).
* **min (Rekor Terendah):** Stok **paling tipis** yang pernah kejadian.
* **max (Rekor Tertinggi):** Stok **paling penuh** yang pernah kejadian.
* **50% (Nilai Tengah):** Angka penengah. Separuh waktu stok kita di bawah ini, separuh lagi di atas ini.

**Tentang Grafik Regresi (Kanan):**
Ini mengecek mitos ekonomi: *"Benar nggak sih kalau stok banyak harga turun?"*
* Lihat garis putus-putusnya. Kalau **Miring ke Bawah**, berarti benar (Stok Banyak = Harga Murah).
"""

# --- TAB 5: PERAMALAN ---
TXT_FORECAST_HEADER = "🔮 Tentang Mesin Peramal (AI)"
TXT_FORECAST_BODY = """
Fitur ini menggunakan Matematika untuk menebak stok masa depan. Ingat: **Ramalan tidak 100% pasti**, tapi bisa buat ancar-ancar.

**Pilih Metode yang Mana?**
1.  **Prophet (Kecerdasan Buatan):**
    * Paling canggih.
    * Pintar membaca musim (Tahu kapan Lebaran, Nataru, Panen Raya).
    * *Disarankan untuk prediksi jangka panjang.*
2.  **Holt-Winters (Hitungan Klasik):**
    * Lebih sederhana.
    * Cocok kalau pola pasarnya sangat teratur (berulang terus).
"""