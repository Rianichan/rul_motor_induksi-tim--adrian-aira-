# Prediksi Sisa Umur Motor Induksi · Tim [Adria-Aira]

**Kasus:** 03 · Sisa Umur Pakai Motor Induksi  
**Anggota:** Adrian Satria Rahardjo (2404300)
             Aira Althafeliq Nurfirly (2401439)
**Aplikasi:** https://drive.google.com/drive/folders/17noDJaEt9xB_jYcWGreCOq76bG33Yhjm?usp=sharing 
**Video demo:** https://drive.google.com/drive/folders/17noDJaEt9xB_jYcWGreCOq76bG33Yhjm?usp=sharing

## Masalah dan pengguna

Proyek ini membantu perencana pemeliharaan memperkirakan sisa umur motor pompa dan konveyor dari tipe motor serta pembacaan sensor. Prediksi digunakan sebagai bahan pertimbangan untuk merencanakan pemesanan motor pengganti dengan waktu pengiriman 2–4 minggu. Hasil model bukan jaminan kerusakan dan tetap perlu ditinjau oleh petugas.

## Data dan pembersihan

Data sintetis terdiri dari catatan harian 300 motor sejak dipasang sampai rusak. Satu motor memiliki banyak baris, sehingga split dilakukan berdasarkan `motor_id` yang sudah dinormalisasi agar motor tidak muncul di train sekaligus test. Baris identik dihapus. Label tipe motor diseragamkan. Nilai sensor yang kosong dan getaran RMS negatif ditangani dengan imputasi median yang dipelajari hanya dari data pengembangan. Suhu di atas 350°C dan beban di atas 100% tidak otomatis dihapus; aplikasi memberi peringatan untuk pembacaan ekstrem.

`data/raw/raw.csv` dipertahankan sebagai salinan data sumber. EDA dan model menggunakan data bersih di memori; notebook hanya memakai kelompok train dan validation untuk grafik dan pemilihan model.

## Hasil model

| Model | MAE validation semua | MAE validation RUL ≤60 | MAE test semua | MAE test RUL ≤60 |
|---|---:|---:|---:|---:|
| Baseline regresi linear | 45,74 hari | 31,02 hari | 54,90 hari | 32,15 hari |
| Regresi polinomial derajat 2 (model final) | 43,15 hari | 25,97 hari | 53,46 hari | 25,58 hari |

Metrik utama adalah MAE untuk baris dengan RUL ≤60 hari. Model final memperbaiki MAE kelompok kritis sebesar 20,4% dibanding baseline pada test yang ditahan. Evaluasi test dilakukan satu kali setelah pemilihan model menggunakan validation.

## Menjalankan aplikasi

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Masukkan tipe motor dan enam pembacaan sensor. Aplikasi menampilkan estimasi RUL, status tindakan, dan peringatan bila nilai sensor berada di luar rentang data pengembangan.

## Melatih ulang model

Jalankan perintah berikut dari folder utama proyek di Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File src\train_baseline.ps1 -CsvPath data\raw\raw.csv -OutputDir artifacts
python src\train_candidate.py data\raw\raw.csv artifacts\split_motor_kasus03.csv artifacts
python src\train_final.py data\raw\raw.csv artifacts\split_motor_kasus03.csv artifacts
```

Skrip baseline membuat manifest split 70/15/15 berdasarkan motor dengan seed 42. Skrip kandidat membandingkan model pada validation. Skrip final melatih ulang baseline dan model terpilih pada train + validation lalu menghitung hasil test satu kali. Jangan memakai hasil test untuk mengubah model.

## Struktur proyek

```text
app.py                         aplikasi Streamlit
src/                           fungsi prediksi dan skrip pelatihan
data/raw/                      brief dan CSV asli
data/processed/                manifest pembagian motor
artifacts/                     model JSON dan metrik
notebooks/                     notebook EDA
reports/                       laporan dan gambar EDA
```

## Keterbatasan

- Data yang tersedia sintetis dan belum mewakili variasi semua motor industri.
- MAE pada motor yang mendekati rusak masih sekitar 25,6 hari, hampir sama dengan rentang waktu pemesanan maksimum. Status aplikasi karena itu harus dipakai sebagai panduan, bukan pengganti pemeriksaan teknis.
- Aplikasi menerima satu pembacaan sensor dan tidak memodelkan tren berurutan atau riwayat pemeliharaan.
- Suhu ekstrem ditandai untuk verifikasi; validitas sensor dan satuan harus diperiksa sebelum penggunaan nyata.

## Berkas laporan dan demo

- Laporan: `reports/LAPORAN.md` .
- Alur video: `reports/ALUR_VIDEO.md`.
- Notebook EDA: `notebooks/01_eda_pembersihan.ipynb`.

## Kontribusi anggota

| Nama | NIM | Kontribusi |
|---|---|---|
| Adrian | [2404300] | EDA dan pembersihan |
| [Adrian & Aira] | [NIM] | Pemodelan dan evaluasi |
| Aira  | [2401439] | Aplikasi dan dokumentasi |
