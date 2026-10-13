"""Konfigurasi terpusat: path, seed, peta kondisi, dan skala kerja."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Dataset lengkap (60 JPEG, ±150 MB) tidak ikut di-commit; lihat README.
# Lokasi lain bisa diatur lewat variabel lingkungan PCD_DATA_DIR.
DATA_DIR = Path(os.environ.get("PCD_DATA_DIR", ROOT / "DATASET NEW"))
SPLIT_PATH = ROOT / "data" / "split.json"
METADATA_PATH = ROOT / "data" / "metadata.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CONFIGS_DIR = ROOT / "configs"
CACHE_DIR = ROOT / "data" / "cache"

SEED = 42
N_DEV_SCENARIOS = 6  # 6 dev (30 citra gelap), 4 uji (20 citra gelap)

CONDITIONS = {
    "K1": "Redup",
    "K2": "Sangat gelap",
    "K3": "Berderau",
    "K4": "Tidak merata",
    "K5": "Remang",
}

# Citra asli 2304x4096 (9.4 MP). Semua eksperimen dan evaluasi berjalan pada
# skala 1/4 (576x1024): NLM dan median dari nol pada resolusi penuh butuh
# beberapa detik per citra, dan detail halus di atas skala ini tidak diperlukan
# untuk membandingkan metode. Acuan dan hasil selalu diukur pada skala yang sama.
WORK_SCALE = 0.25
