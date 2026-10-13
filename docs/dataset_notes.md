# Catatan Dataset

Sumber angka: `data/metadata.csv`, `results/dataset/dataset_stats.csv`, `results/dataset/audit.csv`
(dibuat oleh `experiments/dataset_stats.py` dan `audit_dataset.py`).

## Pengambilan

- Kamera: ponsel TECNO CN7c, f/1.8, panjang fokus 5,56 mm, dipasang tetap (tripod/penyangga).
- Tanggal pengambilan: 11 Oktober 2026.
- 10 skenario benda di atas meja kayu: s01 hard disk, s02 HT hijau, s03 HT merah,
  s04 remote TV, s05 dompet, s06 keyboard, s07 tas selempang, s08 ponsel,
  s09 kotak kemasan, s10 wadah bulat.
- Resolusi seluruh citra 2304×4096 (portrait), JPEG, tag orientasi EXIF kosong.

| Jenis | ISO | Rana | Rata-rata luminans (0-255) | Rentang antarskenario |
|---|---|---|---|---|
| Acuan | 800 | 1/20 s | 98,4 | 85,6-119,0 |
| K1 Redup | 100 | 1/30 s | 32,2 | 22,4-45,3 |
| K2 Sangat gelap | 100 | 1/30 s | 22,6 | 16,0-32,3 |
| K3 Berderau | 3200 | 1/90 s | 25,3 | 18,1-34,7 |
| K4 Tidak merata | 100 | 1/30 s | 19,9 | 12,2-30,6 |
| K5 Remang | 100 | 1/30 s | 24,5 | 13,7-36,5 |

Koreksi atas PRD Draf v1:
- Pengaturan kamera untuk acuan (ISO 800, 1/20 s) sebelumnya tidak tercantum.
- Rana K3 sebenarnya 1/90 s, bukan 1/100 s.
- K4 adalah kondisi rata-rata paling gelap. K5 tidak lebih gelap dari K2, sehingga
  hipotesis "K5 hampir tanpa informasi" di PRD 9.4 tidak didukung data.

## Ground truth

Citra acuan diambil dari posisi kamera yang sama dengan pencahayaan ruangan normal
(semua lampu menyala), ISO 800 dan 1/20 s. Acuan bukan citra "sempurna": ISO 800
masih menyisakan derau halus, dan ada pantulan lampu di permukaan meja. Acuan
diperlakukan sebagai target sebagaimana dipotret, tanpa penyuntingan.

## Audit keselarasan (T07)

`audit_dataset.py` mengukur pergeseran antara acuan dan tiap citra gelap pada
magnitudo gradien (tidak sensitif terhadap perbedaan kecerahan):

- Pergeseran global (phase correlation): median 0,2 px, maksimum 1,30 px pada skala
  kerja 1/4 (≈5 px pada resolusi penuh).
- Rotasi (ECC Euclidean): maksimum 0,2°.
- Pergeseran lokal per blok 4×4: maksimum 2,05 px.
- 48 dari 50 pasangan lolos ambang (global ≤ 1 px, rotasi ≤ 0,2°, blok ≤ 2 px).
  Dua yang ditandai, s01_k3 (1,03 px) dan s09_k3 (1,30 px), adalah citra ISO 3200.
  Mengompensasi pergeseran terukur tidak menaikkan SSIM maupun PSNR (perubahan
  ≤ 0,006 SSIM ke dua arah), jadi pergeseran tersebut adalah efek derau pada
  estimasi, bukan objek atau kamera yang bergeser.

Keputusan: ke-50 pasangan dipakai. Risiko "objek bergeser" di PRD 5.3 (dugaan
s08-s10) tidak terbukti untuk `DATASET NEW`; posisi objek sama di semua pasangan
(lihat `results/dataset/audit/sXX_overlay.png`, tepi yang sejajar tampak putih).

## Perbedaan adegan selain keselarasan (T08)

Kamera tidak bergeser, tetapi kondisi cahaya mengubah isi citra sehingga sebagian
galat tidak bisa dihapus oleh metode apa pun:

1. **Bayangan K4.** Lampu dari samping menimbulkan bayangan keras di sisi objek
   yang tidak ada di acuan.
2. **Bayangan tirai K5.** Bagian bawah bingkai K5 tertutup bayangan diagonal besar.
3. **Pantulan meja.** Posisi dan bentuk sorotan di permukaan meja berbeda di tiap
   kondisi karena sumber cahayanya berbeda.
4. **Warna.** Pengukuran per kanal menunjukkan citra gelap sama hangatnya dengan
   acuan (rasio R/G 1,18-1,26 vs 1,10-1,13). Kesan "netral/keabu-abuan" pada
   pemeriksaan awal ternyata berasal dari cara visualisasi: menaikkan kanal Y di
   YCrCb tanpa menyesuaikan Cr/Cb membuat warna memudar
   (`results/figures/desaturation.png`). Ini memengaruhi rancangan pipeline,
   bukan dataset; diuji di T31.

Keputusan tentang mask: metrik utama dihitung pada **seluruh citra tanpa mask**
agar tidak ada piksel yang disembunyikan. Pengaruh bayangan K4/K5 dibahas di
analisis kegagalan dan diuji dengan mask terpisah sebagai hasil tambahan (T39).

## Privasi (T09)

Ke-60 citra diperiksa secara visual. Isinya hanya benda mati di atas meja; tidak
ada wajah, plat nomor, layar menyala, atau label alamat. Tidak ada area yang
perlu disamarkan.

## Pembagian data (T10)

`data/split.json`, seed 42, per skenario (dibuat 13 Oktober 2026, terkunci):

- Dev (6 skenario, 30 citra gelap): s01, s03, s04, s06, s07, s08
- Uji (4 skenario, 20 citra gelap): s02, s05, s09, s10

Seluruh eksperimen dan tuning hanya membaca dev (dijaga oleh
`experiments/common.guarded_pairs`, diuji di `tests/test_metrics.py`).
