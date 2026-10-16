# Peningkatan Kualitas Citra Minim Cahaya dan Berderau

Project UTS Pengolahan Citra Digital (topik no. 6). Pipeline **klasik** (tanpa deep
learning) yang memperbaiki foto minim cahaya/berderau, lalu mengukurnya terhadap
foto acuan dari adegan yang sama memakai PSNR, SSIM, dan ΔE2000.

Hasil pada citra uji (4 skenario, 20 citra, tidak dipakai saat tuning):

| | PSNR (dB) ↑ | SSIM ↑ | ΔE2000 ↓ |
|---|---|---|---|
| Citra mentah | 9,68 ± 1,20 | 0,322 ± 0,067 | 27,73 ± 4,54 |
| **Pipeline final** | **17,21 ± 1,90** | **0,749 ± 0,049** | **12,84 ± 2,45** |

Lebih baik dari citra mentah pada 20/20 citra di ketiga metrik (Wilcoxon p < 0,001).
Rincian per kondisi: [`results/final/test.md`](results/final/test.md).

## Pipeline final

```
RGB ─► [1] perkecil 1/4 (INTER_AREA), RGB→YCrCb
    ─► [2] reduksi derau: tidak dipakai pada skala 1/4 (diuji: median, Gaussian, bilateral, NLM)
    ─► [3] kanal Y: gamma adaptif (target 0,38) → CLAHE (clip 1,0, tile 16)
    ─► [4] gabung Y+CrCb → RGB, saturasi ×1,5
```

Konfigurasi: [`configs/final.json`](configs/final.json).

## Instalasi

Python 3.11 (diuji di Windows 11).

```
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset

Dataset lengkap (60 JPEG, ±150 MB) tidak ikut di repository. Letakkan foldernya di
root project dengan nama `DATASET NEW/` (atau atur variabel lingkungan `PCD_DATA_DIR`, atau ubah `DATA_DIR` di
[`src/config.py`](src/config.py)):

```
DATASET NEW/
  s01_acuan.jpg  s01_k1.jpg ... s01_k5.jpg
  ...
  s10_acuan.jpg  s10_k1.jpg ... s10_k5.jpg
```

Tautan unduh: `https://drive.google.com/drive/folders/1Tm-5Zf7-Sblo6xMRi5gqsnMVrV1QAh7C?usp=drive_link`. 

## Menjalankan

| Perintah | Isi | Waktu* |
|---|---|---|
| `python run_all.py` | validasi dataset, audit pasangan, verifikasi operasi dari nol, evaluasi akhir di citra uji, analisis kegagalan; cetak tabel metrik | ±2 menit |
| `python run_all.py --quick` | demo: baseline vs pipeline final di citra uji + contoh gambar di `results/demo/` | ±30 detik |
| `python run_all.py --experiments` | seluruh eksperimen dan tuning di dev, lalu evaluasi akhir | ±15 menit |
| `python -m pytest` | 65 unit test (operasi dari nol vs OpenCV, metrik, penjaga data uji) | ±5 detik |
| `python -m src.pipeline "DATASET NEW" -o results/enhanced` | proses satu folder (atau satu file) dengan konfigurasi final | |

\* pada laptop 16 core; pertama kali sedikit lebih lama karena membuat cache citra di `data/cache/`.

Eksperimen satu per satu:

```
python audit_dataset.py                          # keselarasan acuan vs citra gelap
python -m experiments.dataset_stats              # EXIF, statistik, gambar dataset
python -m experiments.verify_from_scratch [--full]
python -m experiments.exp_contrast               # T29 metode kontras
python -m experiments.exp_denoise [--scale-study]# T30 reduksi derau (+ resolusi 1/2 dan penuh)
python -m experiments.exp_order_colorspace       # T31 urutan, ruang warna, warna
python -m experiments.tune                       # T32 grid search (heatmap)
python -m experiments.dev_stats                  # T33 Wilcoxon untuk keputusan dev
python -m experiments.final_eval                 # T35-T36 evaluasi uji (butuh final.json terkunci)
python -m experiments.failure_cases              # T37-T38 kasus terburuk
python -m experiments.fix_eval                   # T39 uji usulan perbaikan
```

## Struktur

```
src/
  config.py         path, seed, kondisi, skala kerja
  dataset.py        baca dataset, validasi pasangan, cache
  split.py          split dev/uji per skenario (terkunci)
  metrics.py        PSNR, SSIM, ΔE2000 (+ mask), metrik tanpa acuan
  from_scratch.py   ekualisasi histogram + filter median (NumPy saja)
  preprocess.py     konversi ruang warna
  enhance.py        gamma, gamma adaptif, HE, CLAHE, SSR/MSR, denoise, unsharp, saturasi
  pipeline.py       pipeline terkonfigurasi (JSON) + CLI
  evaluate.py       evaluasi dan tabel per kondisi
  stats.py          uji Wilcoxon signed-rank
experiments/        skrip eksperimen (lihat di atas)
tests/              unit test
configs/final.json  konfigurasi final (terkunci, git tag config-locked)
results/
  final/            hasil akhir di citra uji: test.md (baca ini dulu), Wilcoxon, waktu
  experiments/      eksperimen dev T29-T31 (.csv per citra, _summary, .md, _wilcoxon)
  tuning/           grid search T32 per langkah + best_config.json
  failures/         panel 4 citra uji terburuk + diagnosis per tahap
  fixes/            uji usulan perbaikan A/B/C (T39)
  dataset/          audit keselarasan + statistik dataset
  verification/     operasi dari nol vs OpenCV
  figures/          gambar untuk laporan

audit_dataset.py
run_all.py
```

## Aturan data

- Parameter disetel **hanya** di dev (s01, s03, s04, s06, s07, s08).
  `experiments/common.guarded_pairs` menolak memuat skenario uji; hanya
  `final_eval.py`/`run_all.py` yang boleh, dan hanya bila `configs/final.json`
  berstatus terkunci.
- Konfigurasi dikunci (commit + tag `config-locked`) sebelum citra uji dievaluasi.

## Operasi dari nol

[`src/from_scratch.py`](src/from_scratch.py) berisi ekualisasi histogram dan filter
median yang hanya memakai NumPy. Keduanya identik dengan `cv2.equalizeHist` dan
`cv2.medianBlur` (selisih piksel 0 pada 10 citra dan 57 kasus uji), tetapi 6-207×
lebih lambat. Lihat [`results/verification/from_scratch.csv`](results/verification/from_scratch.csv).

## Tim

- I Putu Yoga Sastrawan ([@iputuyogasas-dot](https://github.com/iputuyogasas-dot))
- Regio Gatang Akar ([@Regipo](https://github.com/Regipo))
- Damara Reynaldi ([@damaara](https://github.com/damaara))
- Okta Pebrianto ([@atkoxf](https://github.com/atkoxf))
