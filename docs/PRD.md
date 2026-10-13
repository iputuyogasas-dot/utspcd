# PRD: Pipeline Peningkatan Kualitas Citra Minim Cahaya dan Berderau

| Butir | Isi |
|---|---|
| Mata kuliah | Pengolahan Citra Digital, Teknik Informatika |
| Bentuk | Project kelompok UTS (take-home) + presentasi/demo |
| Topik | No. 6: Peningkatan kualitas citra minim cahaya atau berderau |
| Bahasa dan pustaka | Python 3, NumPy, OpenCV, scikit-image, Matplotlib |
| Kelompok | `[isi nama kelompok]` (`[3/4/5]` anggota) |
| Tenggat | `[isi tanggal perpanjangan]` (tenggat awal: 13 Oktober 2026) |
| Status dokumen | v1.1 (koreksi dataset dan kebijakan AI, 13-10-2026) |

---

## 1. Ringkasan

Project ini membangun dan mengevaluasi sebuah pipeline pengolahan citra **klasik** (tanpa deep learning dan tanpa model pra-latih) yang menerima citra mentah hasil pemotretan minim cahaya atau berderau dan menghasilkan citra yang diperbaiki. Kualitas hasil diukur secara kuantitatif terhadap citra acuan (adegan sama, pencahayaan baik) memakai **PSNR** dan **SSIM**, ditambah ΔE (CIEDE2000) untuk memeriksa pergeseran warna.

Nilai utama project terletak pada pemahaman operasi dasar, bukan akurasi tertinggi: dua operasi ditulis sendiri dari nol dengan NumPy dan diverifikasi terhadap pustaka, beberapa metode dibandingkan secara terukur, dan kegagalan dianalisis jujur.

## 2. Latar belakang dan masalah

Citra yang diambil dalam cahaya minim memiliki luminans rendah, kontras sempit, dan (terutama pada ISO tinggi) derau yang kuat. Menaikkan kecerahan secara langsung ikut menguatkan derau, sehingga urutan dan pemilihan operasi (reduksi derau, perbaikan kontras, pemertajaman) sangat memengaruhi hasil. Tidak ada satu metode yang terbaik untuk semua kondisi, dan itu yang ingin dibuktikan lewat angka.

## 3. Tujuan dan indikator keberhasilan

### 3.1 Tujuan

1. Merancang pipeline klasik yang sesuai dengan karakteristik tiap kondisi pencahayaan (CPL 1).
2. Mengimplementasikan dua operasi dari nol dan membuktikan kebenarannya terhadap pustaka (CPL 2).
3. Membandingkan beberapa metode secara kuantitatif dengan metrik yang tepat (CPL 3).
4. Menganalisis penyebab kegagalan dan mengusulkan perbaikan (CPL 4).
5. Mengomunikasikan hasil secara tertulis dan lisan serta bekerja sama dalam tim (CPL 5).

### 3.2 Indikator keberhasilan

| Indikator | Target |
|---|---|
| Pipeline lebih baik dari baseline (citra mentah) pada PSNR dan SSIM di citra uji | Ya, pada rata-rata keseluruhan; per kondisi dilaporkan apa adanya |
| Implementasi dari nol vs pustaka | Selisih piksel dilaporkan pada ≥ 10 citra; target selisih maksimum sekecil mungkin dan penyebab selisih dijelaskan |
| Perbandingan metode | ≥ 2 tahap diuji dengan ≥ 2 metode alternatif; keputusan berdasar angka; uji signifikansi berpasangan |
| Analisis kegagalan | ≥ 3 citra uji terburuk (dipilih otomatis dari skor), penyebab per tahap, dan usulan perbaikan; idealnya perbaikan diuji dan angkanya dilaporkan |
| Reproduksibilitas | Satu perintah menjalankan seluruh pipeline dan mencetak tabel metrik |

Target numerik PSNR/SSIM **tidak ditetapkan di awal**, karena nilainya bergantung pada kualitas pasangan citra. Targetnya adalah peningkatan yang terukur terhadap baseline, bukan angka tertentu.

## 4. Ruang lingkup

**Dalam lingkup**
- Pipeline klasik end-to-end untuk peningkatan citra minim cahaya/berderau.
- Dua operasi dari nol, verifikasi, eksperimen perbandingan, tuning parameter, evaluasi akhir, analisis kegagalan.
- Laporan 8-12 halaman, repository, dan video/demo langsung.

**Di luar lingkup**
- Model deep learning atau model pra-latih (dilarang).
- Pengejaran skor tertinggi dengan cara yang melanggar aturan (misalnya menyetel parameter di citra uji).
- Pemrosesan video atau real-time.

## 5. Dataset

### 5.1 Struktur

- 10 skenario (`s01`-`s10`), masing-masing 1 citra acuan dan 5 citra gelap, total **60 citra** (50 pasangan evaluasi).
- Penamaan: `sXX_acuan.jpg` dan `sXX_k1.jpg` sampai `sXX_k5.jpg`.
- Seluruh citra dipotret sendiri oleh kelompok. Dataset publik (jika ditambahkan) wajib mencantumkan sumber dan lisensi.

### 5.2 Kondisi pengambilan

| Kode | Kondisi | Keterangan | ISO | Rana |
|---|---|---|---|---|
| K1 | Redup | Sebagian lampu mati | 100 | 1/30 |
| K2 | Sangat gelap | Satu lampu kecil saja | 100 | 1/30 |
| K3 | Berderau | Cahaya redup | 3200 | 1/90 |
| K4 | Tidak merata | Satu lampu dari samping | 100 | 1/30 |
| K5 | Remang | Tirai tertutup, tanpa lampu | 100 | 1/30 |
| Acuan | Terang | Semua lampu menyala | 800 | 1/20 |

Nilai ISO/rana sesuai EXIF (`data/metadata.csv`); koreksi dari Draf v1: K3 1/90 s (bukan 1/100) dan pengaturan acuan ditambahkan.

Memenuhi syarat minimal tiga kondisi berbeda. Pelaporan metrik wajib per kondisi.

### 5.3 Syarat citra acuan

Acuan harus berupa adegan **yang sama persis** dengan citra gelap, dipotret dengan tripod/penyangga tetap pada pencahayaan baik. Objek tidak boleh bergeser antara acuan dan citra gelap.

> **Risiko ditutup (13-10-2026):** `audit_dataset.py` pada `DATASET NEW` mengukur pergeseran global median 0,22 px (maks. 1,30 px pada skala 1/4) dan rotasi maks. 0,2°; posisi objek s08-s10 sama dengan acuan. Ke-50 pasangan valid (lihat `docs/dataset_notes.md`).

### 5.4 Ground truth

Citra acuan berfungsi sebagai ground truth untuk seluruh pasangan. Pembuatannya didokumentasikan (alat, kondisi, pengaturan kamera) di bagian Dataset laporan.

### 5.5 Pembagian data

- Pembagian **per skenario**, bukan per citra, agar lima citra dari adegan yang sama tidak bocor antara dev dan uji.
- Dev: 6 skenario (30 citra gelap, ±60%). Uji: 4 skenario (20 citra gelap, ±40%).
- Seed tetap (42), disimpan di `split.json`, dibuat sekali dan tidak diubah.
- Parameter disetel **hanya** di citra dev. Angka akhir dilaporkan **hanya** dari citra uji.

### 5.6 Privasi

Citra yang memuat wajah atau plat nomor orang lain hanya dipakai dengan izin dan disamarkan di laporan.

## 6. Rancangan pipeline

Minimal empat tahap berurutan. Rancangan awal:

```
Citra gelap (RGB)
   │
   ▼
[1] Pra-pemrosesan: baca, validasi ukuran, konversi ke ruang luminans-kromatik (YCrCb / Lab / HSV)
   │
   ▼
[2] Reduksi derau: median / Gaussian / bilateral / non-local means
   │
   ▼
[3] Perbaikan kontras (pada kanal luminans): gamma / ekualisasi histogram / CLAHE / Retinex
   │
   ▼
[4] Pasca-proses: pemertajaman ringan (unsharp mask), gabung kanal, kembali ke RGB
   │
   ▼
[5] Pengukuran: PSNR, SSIM, ΔE terhadap acuan
```

Urutan tahap 2 dan 3 sendiri menjadi variabel eksperimen. Alasan tiap tahap dan setiap nilai parameter harus dituliskan di laporan (hasil percobaan, bukan angka tanpa penjelasan).

## 7. Persyaratan fungsional

| ID | Persyaratan | Prioritas |
|---|---|---|
| F1 | Membaca dataset dari folder, memetakan kode kondisi, dan memvalidasi kelengkapan pasangan | Wajib |
| F2 | Membuat dan menyimpan `split.json` per skenario dengan seed tetap | Wajib |
| F3 | Menjalankan pipeline pada satu citra dan pada seluruh folder | Wajib |
| F4 | Menghitung PSNR, SSIM, ΔE; mendukung mask opsional | Wajib |
| F5 | Mencetak tabel metrik: rata-rata keseluruhan dan per kondisi (rata-rata ± simpangan baku) | Wajib |
| F6 | Menjalankan eksperimen perbandingan metode dan tuning di citra dev, menyimpan hasil ke CSV | Wajib |
| F7 | Skrip verifikasi dari-nol vs pustaka: selisih piksel (maks, rata-rata) dan waktu eksekusi pada ≥ 10 citra | Wajib |
| F8 | Memilih otomatis 3-5 citra uji terburuk dan menyimpan panel perbandingan (gelap, hasil, acuan, peta galat) | Wajib |
| F9 | Satu perintah (`python run_all.py`) atau satu notebook menjalankan semuanya | Wajib |
| F10 | Uji Wilcoxon signed-rank untuk membandingkan dua metode pada data berpasangan | Disarankan |
| F11 | Heatmap hasil grid search parameter | Disarankan |
| F12 | Metrik tambahan tanpa acuan (rata-rata luminans, entropi, simpangan baku) sebagai pelengkap | Opsional |

## 8. Dua operasi dari nol

Ditulis **hanya dengan NumPy**, tanpa fungsi siap pakai OpenCV atau scikit-image. Penggunaan alat bantu AI diizinkan dosen dan dicatat di lampiran; setiap anggota tetap harus mampu menjelaskan kodenya.

| Operasi | Pembanding | Catatan verifikasi |
|---|---|---|
| Ekualisasi histogram | `cv2.equalizeHist` | Selisih kecil mungkin muncul dari pembulatan pada LUT; jelaskan penyebabnya |
| Filter median | `cv2.medianBlur` | Samakan penanganan tepi (padding) dengan pembanding; jelaskan jika berbeda |

Protokol verifikasi:
- Minimal 10 citra; laporkan selisih piksel maksimum dan rata-rata.
- Bandingkan waktu eksekusi (versi sendiri akan lebih lambat; laporkan dan jelaskan).
- Uji kasus tepi: citra konstan, citra satu kanal, ukuran ganjil/genap, kernel berbeda.

## 9. Eksperimen dan evaluasi

### 9.1 Perbandingan metode (keputusan berdasar angka)

| Tahap | Metode yang dibandingkan |
|---|---|
| Perbaikan kontras | Gamma, ekualisasi histogram, CLAHE, Retinex (SSR/MSR) atau gamma adaptif |
| Reduksi derau | Median, Gaussian, bilateral, non-local means |
| Urutan | Denoise sebelum vs sesudah perbaikan kontras |
| Ruang warna | Perbaikan di kanal luminans (YCrCb/Lab) vs di tiap kanal RGB |

### 9.2 Tuning parameter (hanya citra dev)

Grid search kecil untuk gamma, clip limit dan ukuran tile CLAHE, ukuran kernel median/Gaussian, dan parameter unsharp. Hasil ditampilkan sebagai tabel atau heatmap beserta alasan pemilihan.

### 9.3 Evaluasi akhir (hanya citra uji)

- Konfigurasi dikunci setelah tuning di dev.
- Laporkan baseline (citra mentah) dan tiap metode: rata-rata keseluruhan dan per kondisi K1-K5, rata-rata ± simpangan baku.
- Sertakan tabel waktu eksekusi.
- Klaim "A lebih baik dari B" didukung uji Wilcoxon signed-rank.

### 9.4 Analisis kegagalan

- Pilih 3-5 citra uji dengan skor terburuk **secara otomatis**.
- Untuk tiap kasus: tunjukkan citra, tentukan tahap penyebab, jelaskan, dan usulkan perbaikan. Jika waktu memungkinkan, uji perbaikan dan laporkan angkanya.
- Hipotesis awal untuk diuji (bukan kesimpulan): K5 hampir tanpa informasi (kuantisasi 8-bit), derau berwarna menguat pada K3, halo/over-enhancement akibat CLAHE, dan area saturasi pada K4. Hasil pengujian: lihat laporan Bagian 7.2 (K5 tidak hampir hitam; kegagalan utama K3 berupa tekstur hilang saat pengambilan).

## 10. Luaran

| Luaran | Format | Isi |
|---|---|---|
| Repository kode | Tautan GitHub/Drive | Kode sumber, README cara menjalankan, `requirements.txt`, contoh citra, skrip evaluasi |
| Laporan akhir | PDF, 8-12 halaman (di luar lampiran) | Sistematika di bawah |
| Presentasi dan demo | Video / demo langsung | Pipeline berjalan dan hasil utama |

Semua luaran dikumpulkan melalui komti.

**Sistematika laporan:** (1) Pendahuluan, (2) Dasar teori singkat, (3) Dataset, (4) Rancangan pipeline, (5) Implementasi dari nol dan verifikasi, (6) Eksperimen, (7) Hasil dan pembahasan termasuk analisis kegagalan, (8) Kesimpulan dan saran, (9) Lampiran: pembagian kontribusi dan pernyataan penggunaan alat bantu AI.

## 11. Risiko dan mitigasi

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Objek bergeser antara acuan dan citra gelap | PSNR/SSIM tidak bermakna; selisih antarmetode kabur | Jalankan `audit_dataset.py`; foto ulang acuan dengan tripod, atau evaluasi dengan mask latar dan ungkapkan sebagai keterbatasan |
| Citra uji ikut dipakai saat tuning | Hasil bias, bisa dianggap pelanggaran | Split dikunci di `split.json`; skrip tuning hanya membaca dev |
| Kebocoran data antar citra satu skenario | Hasil terlalu optimistis | Pembagian per skenario |
| Filter median buatan sendiri terlalu lambat | Eksperimen memakan waktu | Pakai vektorisasi (misalnya `sliding_window_view`); eksperimen besar memakai versi pustaka, versi sendiri dipakai untuk verifikasi dan bagian pipeline yang dipilih |
| K5 hampir hitam | Skor sangat rendah di semua metode | Jadikan bahan analisis kegagalan, jangan disembunyikan |
| Waktu laporan dan demo habis | Nilai komunikasi turun | Sisakan 20-25% waktu total untuk laporan dan demo |
| Kode atau tulisan mirip kelompok lain | Sanksi berat | Kerjakan sendiri, catat semua bantuan |

## 12. Struktur repository

```
project/
├── data/                  # citra (atau tautan unduh) + split.json
├── src/
│   ├── dataset.py         # baca dataset, kondisi, split
│   ├── metrics.py         # PSNR, SSIM, ΔE
│   ├── from_scratch.py    # ekualisasi histogram + filter median (NumPy)
│   ├── preprocess.py
│   └── enhance.py         # gamma, CLAHE, Retinex, denoise, unsharp
├── experiments/           # verifikasi, perbandingan metode, tuning
├── results/               # CSV, gambar, panel kegagalan
├── audit_dataset.py
├── run_all.py             # satu perintah, cetak tabel metrik
├── requirements.txt
└── README.md
```

## 13. Peran tim

Setiap anggota memegang satu peran utama, dan **semua** anggota harus mampu menjelaskan seluruh pipeline saat tanya jawab. Setiap anggota harus punya commit atas namanya sendiri. Kelompok tiga orang menggabungkan peran ketua dengan satu peran lain.

| Peran | Nama | Tanggung jawab utama |
|---|---|---|
| Ketua dan integrator | `[isi]` | Rancangan pipeline, penggabungan modul, repository dan jadwal |
| Penanggung jawab data | `[isi]` | Pengambilan citra, ground truth, dokumentasi kondisi, audit pasangan |
| Pengembang algoritma | `[isi]` | Dua operasi dari nol dan verifikasinya |
| Penguji dan penulis | `[isi]` | Skrip evaluasi, perbandingan metode, analisis kegagalan, laporan |

## 14. Jadwal kerja

Sesuaikan tanggal dengan tenggat baru. Sisakan 20-25% waktu total untuk tahap 5.

| Tahap | Isi | Target selesai |
|---|---|---|
| 1 | Audit dataset (perbaiki/foto ulang pasangan), split, baseline, skrip metrik | `[tanggal]` |
| 2 | Dua operasi dari nol + verifikasi | `[tanggal]` |
| 3 | Eksperimen perbandingan dan tuning di citra dev | `[tanggal]` |
| 4 | Evaluasi final di citra uji, analisis kegagalan | `[tanggal]` |
| 5 | Laporan, README, video demo, lampiran AI dan kontribusi | `[tanggal]` |

## 15. Integritas akademik dan penggunaan AI

- Seluruh kode, data, dan tulisan adalah hasil kerja kelompok; setiap bantuan luar dinyatakan.
- Kode dari tutorial/dokumentasi hanya untuk bagian yang mengizinkan pustaka, dengan sumber di komentar kode.
- Alat bantu AI boleh dipakai sepenuhnya (izin dosen), termasuk untuk menulis kode dua operasi dari nol. Pencatatan di lampiran: alat, tujuan, dan bagian yang terbantu.
- Dilarang mengarang angka hasil atau ground truth, atau melaporkan citra orang lain sebagai citra sendiri (pelanggaran berat).
- Anggota yang tidak mampu menjelaskan kode atas namanya dapat diberi nilai individu nol.

## 16. Kriteria penerimaan (daftar periksa)

**Spesifikasi teknis wajib**
- [ ] ≥ 4 tahap berurutan
- [ ] Dua operasi dari nol, hanya NumPy
- [ ] Verifikasi vs pustaka pada ≥ 10 citra (selisih piksel + waktu)
- [ ] Perbandingan ≥ 2 metode pada minimal satu tahap, keputusan berdasar angka
- [ ] ≥ 3 citra gagal dengan penyebab per tahap dan usulan perbaikan
- [ ] Satu perintah/notebook mencetak tabel metrik
- [ ] Setiap parameter punya alasan atau hasil percobaan

**Dataset dan evaluasi**
- [ ] ≥ 50 citra, ≥ 30 dipotret sendiri
- [ ] ≥ 3 kondisi pengambilan berbeda
- [ ] Pasangan acuan sejajar dan tervalidasi
- [ ] Tuning hanya di dev (~60%), angka akhir hanya dari uji (~40%)
- [ ] Metrik: rata-rata keseluruhan dan per kondisi
- [ ] Privasi wajah/plat nomor dipenuhi

**Luaran**
- [ ] Repository: kode, README, `requirements.txt`, contoh citra, skrip evaluasi
- [ ] Laporan PDF 8-12 halaman sesuai sistematika
- [ ] Video/demo langsung
- [ ] Lampiran kontribusi dan pernyataan penggunaan AI
- [ ] Setiap anggota punya commit atas nama sendiri
