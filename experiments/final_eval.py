"""T35/T36: evaluasi akhir pada citra UJI dengan konfigurasi yang sudah dikunci.

Hanya skrip ini (dan run_all.py yang memanggilnya) yang memuat skenario uji.
Konfigurasi dibaca dari configs/final.json; skrip menolak berjalan bila file itu
tidak berstatus "locked".

Metode yang dilaporkan:
  - baseline (citra mentah)
  - alternatif kontras dari T29 dengan parameter bawaan
  - pipeline final

Keluaran:
  results/experiments/baseline_dev.csv, results/final/baseline_test.csv   (T15)
  results/final/test.csv, test.md, test_summary.csv
  results/final/wilcoxon.csv
  results/final/timing.csv, timing_methods.csv
"""

import json
import time

import numpy as np
import pandas as pd

from experiments.common import report, run_configs
from src import config
from src.dataset import list_pairs, load_image
from src.evaluate import to_markdown
from src.pipeline import Pipeline
from src.stats import compare_many

FINAL_PATH = config.CONFIGS_DIR / "final.json"


def load_final():
    with open(FINAL_PATH, encoding="utf-8") as f:
        final = json.load(f)
    if not final.get("_meta", {}).get("locked"):
        raise RuntimeError("configs/final.json belum dikunci (_meta.locked); selesaikan tuning di dev dulu")
    return {k: v for k, v in final.items() if not k.startswith("_")}


def C(*steps):
    return {"color_space": "ycrcb", "contrast": list(steps)}


ALTERNATIVES = {
    "gamma_0.5": C({"method": "gamma", "g": 0.5}),
    "hist_eq": C({"method": "hist_eq", "impl": "scratch"}),
    "clahe": C({"method": "clahe", "clip": 2.0, "tile": 8}),
    "msr": C({"method": "msr"}),
    "adaptive_gamma_0.4": C({"method": "adaptive_gamma", "target": 0.4}),
}


def timing(final_cfg, n=4):
    """Waktu eksekusi pipeline final pada skala kerja dan resolusi penuh."""
    pipe = Pipeline(final_cfg)
    rows = []
    paths = [p.dark_path for p in list_pairs()][:n]
    for scale in (config.WORK_SCALE, 1.0):
        ts = []
        for path in paths:
            img = np.array(load_image(path, scale))
            t0 = time.perf_counter()
            pipe(img)
            ts.append(time.perf_counter() - t0)
        rows.append({"scale": scale, "shape": "x".join(map(str, img.shape)), "n": len(ts),
                     "mean_s": float(np.mean(ts)), "std_s": float(np.std(ts, ddof=1))})
    return pd.DataFrame(rows)


def main():
    final_cfg = load_final()

    # Baseline dev (T15) - dev boleh dibaca kapan saja.
    run_configs({"baseline": None}, subset="dev", out_csv="experiments/baseline_dev.csv", verbose=False)

    configs = {"baseline": None, **ALTERNATIVES, "final": final_cfg}
    df = run_configs(configs, subset="test", allow_test=True, out_csv="final/test.csv")
    df[df.method == "baseline"].to_csv(config.FINAL_DIR / "baseline_test.csv", index=False)
    summary = report(df, "final/test", "T35 Evaluasi akhir pada citra uji (n=20)")

    alts = df[~df.method.isin(["baseline", "final"])].groupby("method").ssim.mean()
    best_alt = alts.idxmax()
    tests = compare_many(df, [("final", "baseline"), ("final", best_alt)])
    tests.to_csv(config.FINAL_DIR / "wilcoxon.csv", index=False)
    print("\nWilcoxon (uji):\n" + tests[["a", "b", "metric", "median_diff", "a_wins", "n", "p_value", "better"]]
          .round(4).to_string(index=False))

    per_cond = summary[summary.method.isin(["baseline", "final"])].pivot_table(
        index="kondisi", columns="method", values=["psnr_mean", "ssim_mean", "delta_e_mean"], sort=False)
    print("\nBaseline vs final per kondisi:\n" + per_cond.round(3).to_string())

    t = timing(final_cfg)
    t.to_csv(config.FINAL_DIR / "timing.csv", index=False)
    method_time = df.groupby("method", sort=False).time_s.agg(["mean", "std"]).reset_index()
    method_time.to_csv(config.FINAL_DIR / "timing_methods.csv", index=False)
    print("\nWaktu pipeline final:\n" + t.round(4).to_string(index=False))

    with open(config.FINAL_DIR / "test.md", "a", encoding="utf-8") as f:
        f.write("\n## Wilcoxon signed-rank (uji)\n\n")
        f.write(to_markdown(tests[["a", "b", "metric", "median_diff", "a_wins", "n", "p_value", "better"]]
                            .round(4)) + "\n")
        f.write("\n## Waktu eksekusi pipeline final\n\n" + to_markdown(t.round(4)) + "\n")
        f.write("\n## Waktu per metode (skala kerja)\n\n" + to_markdown(method_time.round(4)) + "\n")
    return df


if __name__ == "__main__":
    main()
