"""T33: uji Wilcoxon signed-rank untuk setiap keputusan yang diambil di dev.

Menggabungkan uji dari T29-T31 dengan uji untuk keputusan tuning (T32), lalu
menyimpannya ke results/experiments/wilcoxon_dev.csv. Dengan n=30 citra (6 skenario),
citra dari skenario yang sama tidak sepenuhnya independen; p-value dibaca
sebagai indikasi, bukan bukti kuat.
"""

import pandas as pd

from src import config
from src.stats import compare_many

E = config.EXPERIMENTS_DIR
T = config.TUNING_DIR

TUNING_TESTS = [
    ("1_gamma.csv", "agamma_t0.38", "agamma_t0.4", "target gamma 0.38 vs 0.40"),
    ("1_gamma.csv", "agamma_t0.38", "gamma_g0.4", "gamma adaptif vs gamma tetap terbaik"),
    ("2_clahe.csv", "clahe_c1.0_t16", "no_clahe", "CLAHE ringan setelah gamma vs tanpa CLAHE"),
    ("3_kernel.csv", "none", "gaussian_k3", "tanpa filter vs filter terbaik"),
    ("4_unsharp.csv", "unsharp_s0.5_a0.0", "unsharp_s0.5_a0.25", "tanpa unsharp vs unsharp teringan"),
    ("5_saturation.csv", "sat_1.5", "sat_1.0", "saturasi 1.5 vs tanpa penguatan"),
    ("5_saturation.csv", "sat_1.5", "sat_1.75", "saturasi 1.5 vs 1.75"),
]

EARLIER = [("T29", "contrast_wilcoxon.csv"), ("T30", "denoise_wilcoxon.csv"),
           ("T31", "order_colorspace_wilcoxon.csv")]


def main():
    frames = []
    for exp, f in EARLIER:
        if (E / f).exists():
            frames.append(pd.read_csv(E / f).assign(experiment=exp, decision=""))
    for csv, a, b, decision in TUNING_TESTS:
        df = pd.read_csv(T / csv)
        frames.append(compare_many(df, [(a, b)]).assign(experiment="T32", decision=decision))
    out = pd.concat(frames, ignore_index=True)
    cols = ["experiment", "decision", "a", "b", "metric", "n", "median_diff", "mean_diff", "a_wins",
            "p_value", "better", "significant_0.05"]
    out = out[cols]
    out.to_csv(E / "wilcoxon_dev.csv", index=False)
    pd.set_option("display.width", 220)
    print(out[out.experiment == "T32"].round(4).to_string(index=False))
    return out


if __name__ == "__main__":
    main()
