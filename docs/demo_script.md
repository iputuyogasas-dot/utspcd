# Naskah Demo dan Persiapan Tanya Jawab

Target durasi video/demo: 8-10 menit. Setiap anggota membawakan bagian perannya.

## Persiapan (sebelum merekam)

```
.venv\Scripts\activate
python run_all.py --quick          # pastikan berjalan (±15-30 detik)
```

Buka di tab terpisah: `laporan.pdf`, `results/demo/`, `results/figures/`,
`results/failures/s02_k3.png`, `src/from_scratch.py`.

## Alur

| Menit | Pembicara | Isi | Yang ditampilkan |
|---|---|---|---|
| 0:00-1:00 | Ketua | Masalah: foto minim cahaya gelap, kontras sempit, berderau. Tujuan: pipeline klasik, diukur dengan PSNR/SSIM/ΔE terhadap acuan | Slide judul, `contact_sheet.png` |
| 1:00-2:30 | Data | 10 skenario × (acuan + K1-K5), EXIF, audit keselarasan (48/50 lolos, sisanya efek derau), split per skenario seed 42 | `contact_sheet_boost.png`, `scene_differences.png`, `data/split.json` |
| 2:30-4:00 | Algoritma | Dua operasi dari nol: rumus LUT HE dan median dengan `sliding_window_view`; detail yang membuat hasil identik dengan OpenCV; tabel waktu | `src/from_scratch.py`, jalankan `python -m pytest tests/test_from_scratch.py -q` |
| 4:00-6:00 | Penguji | Eksperimen dev: kontras (gamma menang), reduksi derau bergantung resolusi, YCrCb + saturasi, tuning | `fig_contrast.png`, `denoise_vs_scale.png`, `heatmap_clahe.png` |
| 6:00-7:30 | Ketua | Demo langsung: `python run_all.py --quick`, tunjukkan tabel dan `results/demo/` | Terminal, folder demo |
| 7:30-9:00 | Penguji | Hasil uji per kondisi, analisis kegagalan K3, uji perbaikan A/C | `final_per_condition.png`, `s02_k3.png` |
| 9:00-9:30 | Ketua | Kesimpulan, keterbatasan, penggunaan AI | Bagian 8 laporan |

## Pertanyaan yang mungkin muncul

**Kenapa tidak memakai reduksi derau?**
Pada skala kerja 1/4, `INTER_AREA` sudah merata-ratakan 16 piksel sehingga derau
turun ±4×; filter tambahan malah mengaburkan tekstur (SSIM 0,723 → 0,690 untuk
median 3). Pada resolusi penuh urutannya terbalik (Gambar 4), jadi bila
memproses resolusi penuh kami akan memakai median 3×3 atau bilateral untuk ISO tinggi.

**Kenapa HE/CLAHE kalah dari gamma, padahal itu metode kontras standar?**
HE memaksa histogram seragam (rata-rata 131, simpangan baku 73), sedangkan acuan
rata-rata 96, simpangan baku 36,5. Metrik berbasis acuan menghukum citra yang
"lebih kontras dari aslinya". Gamma adaptif mengenai kecerahan acuan (97).

**Bagaimana gamma adaptif bekerja?**
γ = ln(target) / ln(mean), mean dan target pada skala 0-1. Dengan target 0,38,
citra yang rata-ratanya 0,1 diberi γ ≈ 0,42; citra yang lebih terang diberi γ
lebih dekat ke 1.

**Kenapa hasil ekualisasi histogram buatan sendiri bisa persis sama dengan OpenCV?**
Tiga hal disamakan: citra konstan dikembalikan apa adanya, skala dihitung dalam
float32, dan pembulatan round-half-to-even (`np.rint`). Untuk median, padding
replikasi tepi sama dengan `BORDER_REPLICATE`, dan k×k ganjil sehingga median
selalu nilai tengah persis.

**Kenapa versi sendiri jauh lebih lambat?**
NumPy menyalin k² nilai per piksel lalu mempartisinya; OpenCV memakai algoritma
median berbasis histogram dan SIMD di C++.

**Bagaimana memastikan tidak ada kebocoran data uji?**
Split per skenario (bukan per citra), runner eksperimen menolak skenario uji
(ada unit test-nya), dan konfigurasi dikunci dengan git tag `config-locked`
sebelum evaluasi uji pertama.

**Kenapa CLAHE dipakai padahal tidak signifikan?**
Kriteria pemilihan (peringkat gabungan tiga metrik) ditetapkan sebelum tuning dan
memilihnya. Kami melaporkan bahwa selisihnya tidak signifikan (p = 0,16) alih-alih
mengganti kriteria setelah melihat hasil.

**Kenapa K3 paling buruk?**
Energi frekuensi tinggi keluaran K3 hanya ±50% acuan: tekstur hilang saat
pengambilan (ISO 3200, 1/90 s, pengurangan derau internal ponsel). Pemertajaman
(perbaikan A) malah menurunkan SSIM karena yang diperkuat sisa derau.

**Kenapa PSNR "hanya" 17 dB?**
Acuan dan citra gelap berbeda isi cahaya: bayangan K4/K5, posisi pantulan meja,
dan gradien cahaya. Tanpa melihat acuan, kecerahan target per adegan juga tidak
bisa diketahui (s09/s10 lebih gelap 18-29 level dari acuan). Target PRD adalah
peningkatan terukur dari baseline, bukan angka tertentu.

**Kenapa SSIM dihitung pada skala 1/4?**
Waktu: metrik dan NLM pada resolusi penuh ±10× lebih lambat. Acuan diperkecil
dengan cara yang sama sehingga perbandingan adil; kami juga menunjukkan bahwa
kesimpulan reduksi derau berubah pada resolusi lain (keterbatasan yang dilaporkan).
