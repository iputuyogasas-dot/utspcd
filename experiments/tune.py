"""T32: tuning parameter dengan grid search kecil, HANYA pada citra dev (F11).

Tuning berjalan bertahap (coordinate search): setiap langkah memvariasikan satu
tahap dengan tahap lain dikunci pada pemenang langkah sebelumnya.

  1. Kecerahan : gamma tetap g dan gamma adaptif (target)
  2. CLAHE     : clip limit × ukuran tile setelah gamma (heatmap), dibanding tanpa CLAHE
  3. Derau     : ukuran kernel median / Gaussian, dibanding tanpa filter
  4. Unsharp   : sigma × amount (heatmap)
  5. Saturasi  : faktor penguatan saturasi (ruang YCrCb memudarkan warna, lihat T31)

Kriteria pemilihan: peringkat rata-rata dari tiga metrik (PSNR ↑, SSIM ↑, ΔE ↓)
pada 30 citra dev; seri dipecah dengan SSIM. Dengan begitu tidak ada satu
metrik yang mendominasi, dan parameter yang hanya menaikkan PSNR dengan
mengorbankan struktur (atau sebaliknya) tidak terpilih.

Keluaran: results/tuning/*.csv, results/figures/heatmap_*.png,
results/tuning/best_config.json.
"""

import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import run_configs
from src import config, plotstyle

OUT = config.TUNING_DIR

# Titik awal dari T29-T31: gamma adaptif di YCrCb, tanpa reduksi derau.
BASE = {
    "color_space": "ycrcb",
    "order": "denoise_first",
    "white_balance": None,
    "denoise": {"method": "none"},
    "contrast": [{"method": "adaptive_gamma", "target": 0.4}],
    "sharpen": None,
    "saturation": 1.0,
}


def select(df):
    """Rata-rata per metode + peringkat gabungan; baris pertama = pemenang."""
    agg = df.groupby("method", sort=False)[["psnr", "ssim", "delta_e", "time_s"]].mean()
    ranks = pd.concat([agg.psnr.rank(ascending=False), agg.ssim.rank(ascending=False),
                       agg.delta_e.rank(ascending=True)], axis=1)
    agg["mean_rank"] = ranks.mean(axis=1)
    return agg.sort_values(["mean_rank", "ssim"], ascending=[True, False])


def step(name, configs):
    df = run_configs(configs, out_csv=f"tuning/{name}.csv")
    table = select(df)
    table.round(4).to_csv(OUT / f"{name}_summary.csv")
    print(f"\n--- {name} ---\n" + table.round(4).to_string())
    return table, df


def variant(base, **changes):
    c = copy.deepcopy(base)
    c.update(copy.deepcopy(changes))
    return c


def save_heatmaps(table, rows, cols, key, title, fname, row_label, col_label, extra=None):
    """Heatmap SSIM dan PSNR untuk grid 2 parameter; sel terpilih diberi bingkai."""
    plotstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    best = table.index[0]
    for ax, metric, fmt in zip(axes, ["ssim", "psnr"], ["{:.3f}", "{:.2f}"]):
        vals = np.array([[table.loc[key(r, c), metric] for c in cols] for r in rows])
        pos = next(((i, j) for i, r in enumerate(rows) for j, c in enumerate(cols) if key(r, c) == best), None)
        plotstyle.heatmap(ax, vals, rows, cols, fmt=fmt, best=pos)
        ax.set_xlabel(col_label)
        ax.set_ylabel(row_label)
        ax.set_title(f"{metric.upper()} rata-rata (dev, n=30)", fontsize=10)
    note = f"Bingkai oranye = pilihan (peringkat gabungan). {extra or ''}"
    fig.suptitle(f"{title}\n{note}", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / fname, dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    best = copy.deepcopy(BASE)
    log = []

    # 1. Kecerahan
    cfgs = {f"gamma_g{g}": variant(best, contrast=[{"method": "gamma", "g": g}])
            for g in (0.30, 0.35, 0.40, 0.45, 0.50, 0.60)}
    targets = (0.30, 0.33, 0.36, 0.38, 0.40, 0.42, 0.45, 0.50)
    cfgs.update({f"agamma_t{t}": variant(best, contrast=[{"method": "adaptive_gamma", "target": t}])
                 for t in targets})
    table, _ = step("1_gamma", cfgs)
    best = cfgs[table.index[0]]
    log.append(("1_gamma", table.index[0]))
    plot_gamma(table, targets)

    # 2. CLAHE setelah gamma
    gamma_step = best["contrast"][0]
    clips, tiles = (1.0, 1.5, 2.0, 3.0), (2, 4, 8, 16)
    cfgs = {"no_clahe": copy.deepcopy(best)}
    cfgs.update({f"clahe_c{c}_t{t}": variant(best, contrast=[gamma_step, {"method": "clahe", "clip": c, "tile": t}])
                 for c in clips for t in tiles})
    table, _ = step("2_clahe", cfgs)
    best = cfgs[table.index[0]]
    log.append(("2_clahe", table.index[0]))
    no = table.loc["no_clahe"]
    save_heatmaps(table.drop(index="no_clahe"), clips, tiles, lambda c, t: f"clahe_c{c}_t{t}",
                  "CLAHE setelah gamma adaptif: clip limit × ukuran tile", "heatmap_clahe.png",
                  "clip limit", "ukuran tile (tile × tile)",
                  extra=f"Tanpa CLAHE: SSIM {no.ssim:.3f}, PSNR {no.psnr:.2f}.")

    # 3. Ukuran kernel reduksi derau
    cfgs = {"none": variant(best, denoise={"method": "none"})}
    for k in (3, 5, 7):
        cfgs[f"median_k{k}"] = variant(best, denoise={"method": "median", "ksize": k})
        cfgs[f"gaussian_k{k}"] = variant(best, denoise={"method": "gaussian", "ksize": k})
    table, _ = step("3_kernel", cfgs)
    best = cfgs[table.index[0]]
    log.append(("3_kernel", table.index[0]))

    # 4. Unsharp mask
    sigmas, amounts = (0.5, 1.0, 2.0, 3.0), (0.0, 0.25, 0.5, 0.75, 1.0)
    cfgs = {f"unsharp_s{s}_a{a}": variant(best, sharpen={"sigma": s, "amount": a})
            for s in sigmas for a in amounts}
    table, _ = step("4_unsharp", cfgs)
    best = cfgs[table.index[0]]
    log.append(("4_unsharp", table.index[0]))
    save_heatmaps(table, sigmas, amounts, lambda s, a: f"unsharp_s{s}_a{a}",
                  "Unsharp mask pada luminans: sigma × amount", "heatmap_unsharp.png",
                  "sigma (px)", "amount", extra="amount 0 = tanpa pemertajaman.")

    # 5. Saturasi
    cfgs = {f"sat_{f}": variant(best, saturation=f) for f in (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)}
    table, _ = step("5_saturation", cfgs)
    best = cfgs[table.index[0]]
    log.append(("5_saturation", table.index[0]))

    with open(OUT / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    pd.DataFrame(log, columns=["langkah", "pemenang"]).to_csv(OUT / "selection_log.csv", index=False)
    print("\nPemenang per langkah:")
    for s, w in log:
        print(f"  {s}: {w}")
    print("\nKonfigurasi terbaik:\n" + json.dumps(best, indent=2))


def plot_gamma(table, targets):
    plotstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    xs = list(targets)
    for ax, metric, label in zip(axes, ["psnr", "ssim", "delta_e"], ["PSNR (dB) ↑", "SSIM ↑", "ΔE2000 ↓"]):
        ys = [table.loc[f"agamma_t{t}", metric] for t in xs]
        ax.plot(xs, ys, color=plotstyle.HIGHLIGHT, marker="o", ms=5)
        i = int(np.argmax(ys) if metric != "delta_e" else np.argmin(ys))
        ax.annotate(f"{ys[i]:.3f}" if metric == "ssim" else f"{ys[i]:.2f}", (xs[i], ys[i]),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=plotstyle.INK_2)
        ax.set_xlabel("target rata-rata luminans")
        ax.set_title(label, fontsize=10)
    fig.suptitle("Gamma adaptif: pengaruh target kecerahan (dev, n=30)", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "tuning_gamma_target.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
