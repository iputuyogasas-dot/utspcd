"""Verifikasi operasi dari nol vs pustaka (F7, T20).

Pada 10 citra gelap lintas kondisi (2 per kondisi, dari skenario berbeda):
  - ekualisasi histogram: kanal Y vs cv2.equalizeHist
  - filter median k=3 dan k=5: RGB vs cv2.medianBlur
Dicatat selisih piksel maksimum, rata-rata, jumlah piksel berbeda, dan waktu
eksekusi (median dari 3 kali jalan). Dijalankan pada skala kerja; tambahkan
--full untuk resolusi asli 2304x4096.

Verifikasi ini tidak melibatkan citra acuan atau metrik kualitas, sehingga
memakai citra dari seluruh skenario tidak membocorkan informasi uji.
"""

import argparse
import time

import cv2
import numpy as np
import pandas as pd

from src import config
from src.dataset import list_pairs, load_image
from src.from_scratch import equalize_hist, median_filter

SCENARIOS_PER_CONDITION = {"K1": ["s01", "s06"], "K2": ["s02", "s07"], "K3": ["s03", "s08"],
                           "K4": ["s04", "s09"], "K5": ["s05", "s10"]}


def timed(fn, *args, repeat=3):
    times, out = [], None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args)
        times.append(time.perf_counter() - t0)
    return out, float(np.median(times))


def compare(ours, ref):
    diff = np.abs(ours.astype(np.int16) - ref.astype(np.int16))
    return {"max_diff": int(diff.max()), "mean_diff": float(diff.mean()),
            "n_diff_pixels": int((diff > 0).sum()), "n_pixels": int(diff.size)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="pakai resolusi asli (lambat)")
    args = ap.parse_args(argv)
    scale = 1.0 if args.full else config.WORK_SCALE

    pairs = [p for p in list_pairs() if p.scenario in SCENARIOS_PER_CONDITION[p.condition]]
    rows = []
    for p in pairs:
        rgb = np.ascontiguousarray(load_image(p.dark_path, scale))
        y = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[..., 0].copy()
        base = {"image": p.name, "condition": p.condition, "shape": "x".join(map(str, rgb.shape))}

        ours, t_ours = timed(equalize_hist, y)
        ref, t_lib = timed(cv2.equalizeHist, y)
        rows.append({**base, "operation": "equalize_hist", "ksize": None, **compare(ours, ref),
                     "t_ours_s": t_ours, "t_lib_s": t_lib})

        for k in (3, 5):
            ours, t_ours = timed(median_filter, rgb, k)
            ref, t_lib = timed(cv2.medianBlur, rgb, k)
            rows.append({**base, "operation": "median_filter", "ksize": k, **compare(ours, ref),
                         "t_ours_s": t_ours, "t_lib_s": t_lib})

    df = pd.DataFrame(rows)
    df["speed_ratio"] = df.t_ours_s / df.t_lib_s
    name = "from_scratch_full.csv" if args.full else "from_scratch.csv"
    config.VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.VERIFICATION_DIR / name, index=False)

    pd.set_option("display.width", 200)
    print(df[["image", "operation", "ksize", "max_diff", "mean_diff", "t_ours_s", "t_lib_s", "speed_ratio"]]
          .round(5).to_string(index=False))
    summary = df.groupby(["operation", "ksize"], dropna=False).agg(
        n=("image", "size"), max_diff=("max_diff", "max"), mean_diff=("mean_diff", "mean"),
        t_ours_ms=("t_ours_s", lambda s: 1000 * s.mean()), t_lib_ms=("t_lib_s", lambda s: 1000 * s.mean()),
        speed_ratio=("speed_ratio", "mean"))
    print("\nRingkasan:\n" + summary.round(3).to_string())
    return df


if __name__ == "__main__":
    main()
