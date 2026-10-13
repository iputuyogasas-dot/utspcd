## Lampiran A. Pembagian kontribusi

| Peran | Nama | Kontribusi utama | Commit |
|---|---|---|---|
| Ketua dan integrator | [isi] | Rancangan pipeline, integrasi modul, repository, jadwal | [isi tautan] |
| Penanggung jawab data | [isi] | Pemotretan 60 citra, dokumentasi kondisi, audit pasangan | [isi tautan] |
| Pengembang algoritma | [isi] | Ekualisasi histogram dan filter median dari nol, verifikasi | [isi tautan] |
| Penguji dan penulis | [isi] | Skrip evaluasi, eksperimen, analisis kegagalan, laporan | [isi tautan] |

## Lampiran B. Pernyataan penggunaan alat bantu AI

Dosen pengampu mengizinkan penggunaan alat bantu AI tanpa batasan untuk project ini.
Penggunaannya kami catat sebagai berikut.

| Alat | Tujuan | Bagian yang terbantu |
|---|---|---|
| Claude (Anthropic), lewat Claude Code | Menulis kode dan menjalankan eksperimen | `src/` (termasuk `from_scratch.py`), `experiments/`, `audit_dataset.py`, `run_all.py`, `tests/` |
| Claude (Anthropic), lewat Claude Code | Analisis dan penulisan | Pemecahan PRD menjadi tugas, draf laporan ini, README, `docs/parameter_rationale.md`, `docs/dataset_notes.md` |

Yang dikerjakan kelompok sendiri: pemotretan seluruh 60 citra dan dokumentasi
kondisinya, [isi: peninjauan kode, pengecekan angka, penyuntingan laporan, presentasi].

Seluruh angka di laporan dihasilkan oleh skrip di repository dan dapat diulang
dengan `python run_all.py` (evaluasi akhir) dan `python run_all.py --experiments`
(seluruh eksperimen dev). Tidak ada angka atau ground truth yang dikarang atau
disunting manual. [Konfirmasi sebelum dikumpulkan: setiap anggota telah
mempelajari kode dan mampu menjelaskan seluruh tahap pipeline, termasuk kedua
operasi dari nol.]

## Lampiran C. Cara mereproduksi

```
pip install -r requirements.txt
python -m pytest                 # 65 unit test
python run_all.py                # audit, verifikasi, evaluasi uji, analisis kegagalan
python run_all.py --experiments  # + seluruh eksperimen dan tuning dev
python -m experiments.fix_eval   # perbaikan tambahan (Bagian 7.2)
python -m experiments.figures    # gambar laporan
```
