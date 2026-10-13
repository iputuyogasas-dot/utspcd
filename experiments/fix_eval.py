"""T39: menguji usulan perbaikan dari analisis kegagalan.

Hasil di sini adalah hasil TAMBAHAN; configs/final.json tidak diubah.
Setiap perbaikan disetel hanya di dev dan diuji sekali di uji, dan hanya bila
mengalahkan pipeline final di dev.

Perbaikan A - pemertajaman khusus ISO tinggi (kasus gagal K3).
  Diagnosis: keluaran K3 hanya memiliki ±50% energi frekuensi tinggi acuan.
  Aturan: bila EXIF ISO >= 1600, tambahkan unsharp mask setelah pipeline final.
  sigma × amount disetel pada K3 dev (6 citra).

Perbaikan B - mask bayangan (K4/K5) untuk evaluasi.
  Mask disusun dari rasio iluminasi gelap/acuan (blur Gaussian besar). Pemeriksaan
  di dev menunjukkan K1, yang tidak punya bayangan, memiliki porsi piksel
  "jauh lebih gelap dari median rasio" sebesar K5. Perbedaan iluminasi bersifat
  menyeluruh (gradien cahaya), bukan hanya bayangan; mask ditolak sebagai alat
  evaluasi dan diganti perbaikan C.

Perbaikan C - koreksi iluminasi lokal parsial sebelum gamma.
  out = I · (mean(L)/L)^alpha dengan L = blur Gaussian besar dari luminans.
  sigma × alpha disetel di dev (semua kondisi).
"""

import copy

import cv2
import numpy as np
import pandas as pd

from experiments.common import run_configs
from experiments.final_eval import load_final
from experiments.tune import select
from src import config
from src.dataset import list_pairs, load_pair
from src.metrics import luminance
from src.split import load_split
from src.stats import compare_many

R = config.RESULTS_DIR
WIL_COLS = ["metric", "median_diff", "a_wins", "n", "p_value"]


def fix_a(final):
    grid = {"final": final}
    for s in (1.0, 2.0):
        for a in (0.25, 0.5, 1.0, 1.5, 2.0):
            c = copy.deepcopy(final)
            c["sharpen"] = {"sigma": s, "amount": a}
            grid[f"sharpen_s{s}_a{a}"] = c
    dev = run_configs(grid, subset="dev", conditions=["K3"], out_csv="fix_a_dev.csv")
    table = select(dev)
    table.round(4).to_csv(R / "fix_eval_a.csv")
    print("\nA. Tuning unsharp untuk ISO tinggi (dev K3, n=6):\n" + table.round(4).to_string())
    if table.index[0] == "final":
        print("Tidak ada varian yang mengalahkan pipeline final di dev; perbaikan A tidak diuji di uji.")
        return None
    return table.index[0]


def illumination_ratio_dev(sigma=15, threshold=0.6):
    """Porsi piksel dengan iluminasi relatif (gelap/acuan) < threshold × median, hanya dev."""
    dev = set(load_split()["dev"])
    rows = []
    for p in list_pairs():
        if p.scenario not in dev:
            continue
        dark, ref = (np.array(x) for x in load_pair(p))
        yd = cv2.GaussianBlur(luminance(dark).astype(np.float32) + 1, (0, 0), sigma)
        yr = cv2.GaussianBlur(luminance(ref).astype(np.float32) + 1, (0, 0), sigma)
        q = yd / yr
        q /= np.median(q)
        rows.append({"image": p.name, "condition": p.condition, "frac_below": float((q < threshold).mean()),
                     "p5": float(np.percentile(q, 5))})
    df = pd.DataFrame(rows)
    df.to_csv(R / "fix_eval_b_dev.csv", index=False)
    summary = df.groupby("condition")[["frac_below", "p5"]].agg(["mean", "min", "max"])
    print(f"\nB. Porsi piksel dengan iluminasi relatif < {threshold} × median (dev, n=6 per kondisi):\n"
          + summary.round(3).to_string())
    return df


def fix_c(final):
    grid = {"final": final}
    for sigma in (30, 60, 120):
        for alpha in (0.15, 0.3, 0.5):
            c = copy.deepcopy(final)
            c["contrast"] = [{"method": "illum_flatten", "sigma": sigma, "alpha": alpha}] + c["contrast"]
            grid[f"flat_s{sigma}_a{alpha}"] = c
    dev = run_configs(grid, subset="dev", out_csv="fix_c_dev.csv")
    table = select(dev)
    table.round(4).to_csv(R / "fix_eval_c_dev.csv")
    print("\nC. Tuning koreksi iluminasi parsial (dev, n=30):\n" + table.round(4).to_string())
    best = table.index[0]
    if best == "final":
        print("Tidak ada varian yang mengalahkan pipeline final di dev; perbaikan C tidak diuji di uji.")
        return None

    wil_dev = compare_many(dev, [(best, "final")])
    print(f"\nWilcoxon dev: {best} vs final\n" + wil_dev[WIL_COLS].round(4).to_string(index=False))
    test = run_configs({"final": final, best: grid[best]}, subset="test", allow_test=True,
                       out_csv="fix_c_test.csv")
    wil_test = compare_many(test, [(best, "final")])
    overall = test.groupby("method")[["psnr", "ssim", "delta_e"]].mean()
    per_cond = test.pivot_table(index="condition", columns="method", values=["psnr", "ssim", "delta_e"])
    print(f"\nC. Uji (n=20): {best} vs final\n" + overall.round(4).to_string())
    print(wil_test[WIL_COLS].round(4).to_string(index=False))
    print("\nPer kondisi:\n" + per_cond.round(3).to_string())
    pd.concat([wil_dev.assign(part="dev"), wil_test.assign(part="test")], ignore_index=True) \
        .to_csv(R / "fix_eval_c_wilcoxon.csv", index=False)
    per_cond.round(4).to_csv(R / "fix_eval_c_test_per_condition.csv")
    return best


def main():
    final = load_final()
    fix_a(final)
    illumination_ratio_dev()
    fix_c(final)


if __name__ == "__main__":
    main()
