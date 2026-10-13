"""Eksperimen T30: perbandingan metode reduksi derau (hanya dev).

Kontras memakai pemenang T29 (gamma adaptif, target 0.4, YCrCb). Reduksi derau
dijalankan sebelum perbaikan kontras (urutan diuji terpisah di T31). Karena
derau masih beramplitudo kecil di citra gelap, kekuatan filter (h NLM,
sigma_color bilateral) sengaja dibuat rendah.

  python -m experiments.exp_denoise                 # skala kerja 1/4
  python -m experiments.exp_denoise --scale-study   # ulang di skala 1/2 dan penuh
"""

import argparse

import pandas as pd

from experiments.common import ranking, report, run_configs
from src import config
from src.stats import compare_many

CONTRAST = [{"method": "adaptive_gamma", "target": 0.4}]


def D(**denoise):
    return {"color_space": "ycrcb", "denoise": denoise, "contrast": CONTRAST}


CONFIGS = {
    "none": D(method="none"),
    "median_3": D(method="median", ksize=3),
    "median_3_scratch": D(method="median", ksize=3, impl="scratch"),
    "median_5": D(method="median", ksize=5),
    "gaussian_5": D(method="gaussian", ksize=5),
    "bilateral": D(method="bilateral", d=7, sigma_color=15, sigma_space=5),
    "nlm_h3": D(method="nlm", h=3, h_color=3),
    "nlm_h6": D(method="nlm", h=6, h_color=6),
}

SCALE_CONFIGS = {k: CONFIGS[k] for k in ("none", "median_3", "bilateral", "nlm_h3")}


def scale_study():
    """Apakah kesimpulan berubah bila citra tidak diperkecil?

    Memperkecil 1/4 dengan INTER_AREA merata-ratakan 16 piksel sehingga derau
    sudah turun ±4x sebelum pipeline berjalan.
    """
    frames = []
    for scale in (0.5, 1.0):
        d = run_configs(SCALE_CONFIGS, scale=scale, workers=3 if scale == 1.0 else 6)
        d["scale"] = scale
        frames.append(d)
    df = pd.concat(frames)
    df.to_csv(config.EXPERIMENTS_DIR / "denoise_scale.csv", index=False)
    piv = df.pivot_table(index=["scale", "method"], columns="condition", values="ssim", aggfunc="mean")
    piv["Semua"] = df.groupby(["scale", "method"]).ssim.mean()
    piv["PSNR"] = df.groupby(["scale", "method"]).psnr.mean()
    piv["time_s"] = df.groupby(["scale", "method"]).time_s.mean()
    print("\nSSIM per skala dan kondisi:\n" + piv.round(4).to_string())
    piv.round(4).to_csv(config.EXPERIMENTS_DIR / "denoise_scale_summary.csv")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale-study", action="store_true")
    if ap.parse_args(argv).scale_study:
        return scale_study()

    df = run_configs(CONFIGS, out_csv="experiments/denoise.csv")
    report(df, "experiments/denoise", "T30 Perbandingan metode reduksi derau (dev)")
    rank = ranking(df)
    print("\nPeringkat (SSIM, lalu PSNR):\n" + rank.round(4).to_string())
    print("\nSSIM per kondisi:\n" + df.pivot_table(index="method", columns="condition", values="ssim")
          .reindex(rank.index).round(4).to_string())
    denoisers = [m for m in rank.index if m not in ("none", "median_3_scratch")]
    tests = compare_many(df, [("none", denoisers[0]), ("none", "nlm_h3"), ("median_3", "median_3_scratch")])
    tests.to_csv(config.EXPERIMENTS_DIR / "denoise_wilcoxon.csv", index=False)
    print("\nWilcoxon:\n" + tests[["a", "b", "metric", "median_diff", "a_wins", "n", "p_value", "better"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
