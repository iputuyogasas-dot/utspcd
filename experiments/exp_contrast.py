"""Eksperimen T29: perbandingan metode perbaikan kontras (hanya dev).

Tahap lain dibuat netral (tanpa reduksi derau dan pemertajaman, ruang YCrCb)
agar perbedaan hanya berasal dari metode kontras. Setiap metode memakai
parameter bawaan yang umum; parameter disetel kemudian di tune.py.
"""

from experiments.common import report, run_configs, ranking
from src import config
from src.stats import compare_many


def C(*steps, space="ycrcb"):
    return {"color_space": space, "contrast": list(steps)}


CONFIGS = {
    "baseline": None,
    "gamma_0.5": C({"method": "gamma", "g": 0.5}),
    "adaptive_gamma": C({"method": "adaptive_gamma", "target": 0.4}),
    "hist_eq": C({"method": "hist_eq", "impl": "scratch"}),
    "clahe": C({"method": "clahe", "clip": 2.0, "tile": 8}),
    "ssr": C({"method": "ssr", "sigma": 80}),
    "msr": C({"method": "msr"}),
    "adaptive_gamma+clahe": C({"method": "adaptive_gamma", "target": 0.4},
                              {"method": "clahe", "clip": 1.5, "tile": 8}),
}


def main():
    df = run_configs(CONFIGS, out_csv="experiments/contrast.csv")
    report(df, "experiments/contrast", "T29 Perbandingan metode perbaikan kontras (dev)")
    rank = ranking(df)
    print("\nPeringkat (SSIM, lalu PSNR):\n" + rank.round(3).to_string())
    top = [m for m in rank.index if m != "baseline"][:3]
    tests = compare_many(df, [(top[0], top[1]), (top[0], top[2]), (top[0], "baseline")])
    tests.to_csv(config.EXPERIMENTS_DIR / "contrast_wilcoxon.csv", index=False)
    print("\nWilcoxon:\n" + tests[["a", "b", "metric", "median_diff", "a_wins", "n", "p_value", "better"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
