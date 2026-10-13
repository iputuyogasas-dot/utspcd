"""Eksperimen T31: urutan tahap, ruang warna, white balance, dan saturasi (hanya dev).

Basis: gamma adaptif target 0.4 (pemenang T29).
  A. Urutan  : denoise sebelum vs sesudah kontras (median 3 dan NLM h=3)
  B. Ruang   : YCrCb, Lab, HSV, tiap kanal RGB, rasio luminans
  C. Warna   : gray-world white balance; penguatan saturasi 1.2 / 1.4 (YCrCb)
"""

from experiments.common import ranking, report, run_configs
from src import config
from src.stats import compare_many

AG = [{"method": "adaptive_gamma", "target": 0.4}]


def cfg(space="ycrcb", denoise=None, order="denoise_first", wb=None, sat=1.0):
    return {"color_space": space, "contrast": AG, "denoise": denoise or {"method": "none"},
            "order": order, "white_balance": wb, "saturation": sat}


MED = {"method": "median", "ksize": 3}
NLM = {"method": "nlm", "h": 3, "h_color": 3}

GROUPS = {
    "A_urutan": {
        "median3_denoise_first": cfg(denoise=MED),
        "median3_contrast_first": cfg(denoise=MED, order="contrast_first"),
        "nlm3_denoise_first": cfg(denoise=NLM),
        "nlm3_contrast_first": cfg(denoise=NLM, order="contrast_first"),
    },
    "B_ruang_warna": {
        "ycrcb": cfg("ycrcb"),
        "lab": cfg("lab"),
        "hsv": cfg("hsv"),
        "rgb_per_kanal": cfg("rgb"),
        "rasio_luminans": cfg("ratio"),
    },
    "C_warna": {
        "ycrcb_gray_world": cfg(wb="gray_world"),
        "ycrcb_sat1.2": cfg(sat=1.2),
        "ycrcb_sat1.4": cfg(sat=1.4),
    },
}


def main():
    configs = {name: c for g in GROUPS.values() for name, c in g.items()}
    df = run_configs(configs)
    group_of = {name: g for g, items in GROUPS.items() for name in items}
    df["group"] = df.method.map(group_of)
    df.to_csv(config.EXPERIMENTS_DIR / "order_colorspace.csv", index=False)
    report(df, "experiments/order_colorspace", "T31 Urutan, ruang warna, white balance, saturasi (dev)")
    for g in GROUPS:
        print(f"\n--- {g} ---\n" + ranking(df[df.group == g]).round(4).to_string())

    tests = compare_many(df, [
        ("median3_denoise_first", "median3_contrast_first"),
        ("nlm3_denoise_first", "nlm3_contrast_first"),
        ("ycrcb", "rasio_luminans"), ("ycrcb", "lab"), ("ycrcb", "rgb_per_kanal"),
        ("ycrcb", "ycrcb_gray_world"), ("ycrcb_sat1.2", "ycrcb"), ("ycrcb_sat1.4", "ycrcb"),
    ])
    tests.to_csv(config.EXPERIMENTS_DIR / "order_colorspace_wilcoxon.csv", index=False)
    print("\nWilcoxon:\n" + tests[["a", "b", "metric", "median_diff", "a_wins", "n", "p_value", "better"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
