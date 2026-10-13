# Alasan Pemilihan Parameter (T34)

Semua angka berikut berasal dari **dev** (6 skenario, 30 citra gelap, skala kerja 1/4).
Konfigurasi dikunci di `configs/final.json` dan ditandai dengan git tag `config-locked`
sebelum citra uji pertama kali dievaluasi.

Kriteria pemilihan (ditetapkan sebelum tuning): **peringkat rata-rata** dari tiga
metrik (PSNR ↑, SSIM ↑, ΔE2000 ↓); jika seri, pilih SSIM tertinggi. Uji
Wilcoxon signed-rank (`results/wilcoxon_dev.csv`) dipakai untuk menilai apakah
selisihnya bermakna. Dengan n = 30 dari 6 skenario, p-value dibaca sebagai
indikasi.

| Tahap | Parameter | Nilai | Bukti | Signifikansi (dev) |
|---|---|---|---|---|
| 1. Pra-pemrosesan | skala kerja | 1/4 (576×1024), `INTER_AREA` | Waktu per citra, skala 1/4 vs penuh: NLM 1,5 s vs 11,5 s; median dari nol 0,16 s vs 1,8 s; studi 4 konfigurasi × 30 citra butuh 140 s pada skala 1/2 dan 794 s pada resolusi penuh | - |
| 1. Pra-pemrosesan | ruang warna | YCrCb (kanal Y) | `results/exp_order_colorspace.md`: SSIM 0,723 vs Lab 0,718, RGB 0,718, rasio 0,703, HSV 0,700 | YCrCb > RGB per kanal: SSIM 30/30 citra, p < 0,001; > Lab pada ΔE p < 0,001 |
| 1. Pra-pemrosesan | white balance | tidak dipakai | gray-world memperburuk 30/30 citra di ketiga metrik | p < 0,001 |
| 2. Reduksi derau | metode | tidak dipakai | `results/exp_denoise.md`: semua filter menurunkan SSIM (median 3: 0,690; NLM h=3: 0,680; tanpa: 0,723) | tanpa filter vs median 3: SSIM p < 0,001 (27/30); vs Gaussian 3: p = 0,002 |
| 2. Reduksi derau | alasan | memperkecil 4×4 dengan `INTER_AREA` sudah merata-ratakan 16 piksel | `results/figures/denoise_vs_scale.png`: pada resolusi penuh urutannya terbalik (median 3 SSIM 0,722 vs tanpa 0,709; K3 bilateral 0,701 vs 0,645) | - |
| 3. Kontras | metode | gamma adaptif | `results/exp_contrast.md`: SSIM 0,723 vs HE 0,552, CLAHE 0,523, MSR 0,566, gamma 0,5 0,711 | vs gamma 0,5: ΔE p < 0,001; vs gamma tetap terbaik (g 0,4) tidak signifikan (p > 0,7) |
| 3. Kontras | target gamma adaptif | 0,38 | `results/figures/tuning_gamma_target.png`: SSIM puncak di 0,36-0,38, PSNR puncak di 0,38 | 0,38 vs 0,40: SSIM p = 0,007; PSNR, ΔE tidak signifikan |
| 3. Kontras | CLAHE setelah gamma | clip 1,0, tile 16×16 | `results/figures/heatmap_clahe.png`: SSIM turun bila clip naik; PSNR naik bila tile membesar | vs tanpa CLAHE: SSIM +0,004 (p = 0,16), ΔE −0,14 (p = 0,08), tidak signifikan |
| 3. Kontras | urutan | denoise sebelum kontras | Relevan bila filter dipakai: NLM sebelum kontras lebih baik (SSIM p < 0,001) | - |
| 4. Pasca-proses | unsharp amount | 0 (tidak dipakai) | `results/figures/heatmap_unsharp.png`: setiap amount > 0 menurunkan SSIM | amount 0 vs 0,25: 29/30 citra lebih baik, p < 0,001 |
| 4. Pasca-proses | penguatan saturasi | 1,5 | `results/tuning/5_saturation_summary.csv`: ΔE 12,66 → 12,12; optimum di dalam grid 1,0-2,5 | vs 1,0: ΔE p < 0,001, PSNR p = 0,005, SSIM tidak signifikan |

## Catatan keputusan

- **CLAHE dipertahankan walau tidak signifikan.** Kriteria peringkat gabungan
  yang ditetapkan di awal memilihnya, dan selisihnya kecil di kedua arah. Di uji,
  pipeline final mengungguli gamma adaptif tanpa CLAHE pada SSIM (p = 0,002) dan
  ΔE (p = 0,012), tetapi perbandingan itu juga memuat perbedaan target (0,38 vs
  0,40) dan saturasi, jadi kontribusi CLAHE sendiri tetap kecil.
- **Clip 1,0 bukan sekadar batas grid.** OpenCV memakai
  `max(1, clip × luas_ubin / 256)` sebagai batas bin, sehingga clip < 1 setara dengan 1.
  Tile 16 berada di tepi grid; tile yang lebih besar belum diuji (keterbatasan).
- **Saturasi diperlukan karena YCrCb.** Menaikkan Y tanpa menyentuh Cr/Cb
  menurunkan saturasi relatif (`results/figures/desaturation.png`); penguatan
  1,5 mengembalikan sebagian warna kayu.
- **Reduksi derau bergantung resolusi.** Pada skala kerja filter tidak membantu.
  Untuk dipakai pada resolusi penuh, rekomendasinya median 3×3 (semua kondisi)
  atau bilateral untuk K3; ini dari studi skala di dev, belum diuji pada citra uji.
