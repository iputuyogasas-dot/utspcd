"""Gambar ringkasan untuk laporan, dibuat dari CSV di results/.

  python -m experiments.figures          # semua gambar yang datanya tersedia
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, plotstyle
from src.dataset import list_pairs, load_pair
from src.split import load_split

R = config.RESULTS_DIR
F = config.FIGURES_DIR


def _bars(ax, labels, values, highlight, fmt, xlabel):
    colors = [plotstyle.HIGHLIGHT if l == highlight else plotstyle.BASE for l in labels]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    for yi, v in zip(y, values):
        ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=8, color=plotstyle.INK_2)
    ax.set_xlim(0, max(values) * 1.18)


def method_comparison(csv, out, title, label_map=None, chosen=None):
    """Batang per metode; `chosen` = metode yang dipilih (berdasarkan uji Wilcoxon), disorot."""
    df = pd.read_csv(R / csv)
    agg = df.groupby("method", sort=False)[["psnr", "ssim", "delta_e"]].mean().sort_values("ssim", ascending=False)
    labels = [label_map.get(m, m) if label_map else m for m in agg.index]
    best = (label_map or {}).get(chosen, chosen) if chosen else labels[0]
    plotstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(11, 0.42 * len(agg) + 1.4), sharey=True)
    _bars(axes[0], labels, agg.ssim.values, best, "{:.3f}", "SSIM ↑")
    _bars(axes[1], labels, agg.psnr.values, best, "{:.2f}", "PSNR (dB) ↑")
    _bars(axes[2], labels, agg.delta_e.values, best, "{:.2f}", "ΔE2000 ↓")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(F / out, dpi=150)
    plt.close(fig)


def denoise_scale():
    quarter = pd.read_csv(R / "experiments/denoise.csv")
    quarter = quarter[quarter.method.isin(["none", "median_3", "bilateral", "nlm_h3"])].assign(scale=0.25)
    other = pd.read_csv(R / "experiments/denoise_scale.csv")
    df = pd.concat([quarter, other])
    names = {"none": "tanpa filter", "median_3": "median 3×3", "bilateral": "bilateral", "nlm_h3": "NLM h=3"}
    colors = dict(zip(names, plotstyle.CATEGORICAL))
    plotstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (title, part) in zip(axes, [("K3 (ISO 3200)", df[df.condition == "K3"]), ("Semua kondisi", df)]):
        m = part.groupby(["method", "scale"]).ssim.mean().unstack(0)
        for method, label in names.items():
            ax.plot(m.index, m[method], marker="o", ms=5, color=colors[method], label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.25, 0.5, 1.0], ["1/4\n576×1024", "1/2\n1152×2048", "penuh\n2304×4096"])
        ax.set_xlim(0.2, 1.25)
        ax.set_ylabel("SSIM rata-rata (dev)")
        ax.set_title(title, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Reduksi derau baru membantu bila citra tidak diperkecil\n"
                 "(bandingkan urutan metode dalam satu skala; nilai SSIM antarskala tidak setara)", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(F / "denoise_vs_scale.png", dpi=150)
    plt.close(fig)


def final_per_condition():
    df = pd.read_csv(R / "final/test.csv")
    df = df[df.method.isin(["baseline", "final"])]
    conds = list(config.CONDITIONS)
    plotstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, metric, label in zip(axes, ["psnr", "ssim", "delta_e"], ["PSNR (dB) ↑", "SSIM ↑", "ΔE2000 ↓"]):
        g = df.groupby(["condition", "method"])[metric].agg(["mean", "std"]).unstack()
        x = np.arange(len(conds))
        for k, (method, color, name) in enumerate([("baseline", plotstyle.BASE, "citra mentah"),
                                                   ("final", plotstyle.HIGHLIGHT, "pipeline final")]):
            ax.bar(x + (k - 0.5) * 0.38, g["mean"][method].reindex(conds), 0.36,
                   yerr=g["std"][method].reindex(conds), color=color, label=name,
                   error_kw={"ecolor": plotstyle.MUTED, "lw": 1, "capsize": 2})
        ax.set_xticks(x, conds)
        ax.grid(axis="x", visible=False)
        ax.set_title(label, fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Citra uji (n=20, 4 per kondisi): rata-rata ± simpangan baku", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(F / "final_per_condition.png", dpi=150)
    plt.close(fig)


def examples():
    """Satu citra uji per kondisi: gelap | hasil | acuan."""
    from experiments.final_eval import load_final
    from src.pipeline import Pipeline

    pipe = Pipeline(load_final())
    test = load_split()["test"]
    scen = test[0]
    pairs = [p for p in list_pairs() if p.scenario == scen]
    plotstyle.apply()
    fig, axes = plt.subplots(3, 5, figsize=(10, 6.4))
    for j, p in enumerate(pairs):
        dark, ref = (np.array(x) for x in load_pair(p))
        for i, (img, label) in enumerate([(dark, "gelap"), (pipe(dark), "hasil"), (ref, "acuan")]):
            ax = axes[i, j]
            ax.imshow(img[::2, ::2])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if i == 0:
                ax.set_title(f"{p.condition} {config.CONDITIONS[p.condition]}", fontsize=9)
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle(f"Contoh hasil pada skenario uji {scen}", fontsize=10)
    fig.tight_layout()
    fig.savefig(F / "examples_test.png", dpi=150)
    plt.close(fig)


CONTRAST_LABELS = {"baseline": "citra mentah", "gamma_0.5": "gamma 0,5", "adaptive_gamma": "gamma adaptif",
                   "hist_eq": "ekualisasi histogram", "clahe": "CLAHE", "ssr": "SSR", "msr": "MSR",
                   "adaptive_gamma+clahe": "gamma adaptif + CLAHE"}
DENOISE_LABELS = {"none": "tanpa filter", "median_3": "median 3", "median_3_scratch": "median 3 (dari nol)",
                  "median_5": "median 5", "gaussian_5": "Gaussian 5", "bilateral": "bilateral",
                  "nlm_h3": "NLM h=3", "nlm_h6": "NLM h=6"}


def main():
    F.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("experiments/contrast.csv", lambda: method_comparison("experiments/contrast.csv", "fig_contrast.png",
                                                       "T29 Metode perbaikan kontras (dev, n=30); biru = dipilih",
                                                       CONTRAST_LABELS, chosen="adaptive_gamma")),
        ("experiments/denoise.csv", lambda: method_comparison("experiments/denoise.csv", "fig_denoise.png",
                                                      "T30 Metode reduksi derau, skala 1/4 (dev, n=30); biru = dipilih",
                                                      DENOISE_LABELS, chosen="none")),
        ("experiments/denoise_scale.csv", denoise_scale),
        ("final/test.csv", final_per_condition),
        ("final/test.csv", examples),
    ]
    for need, fn in jobs:
        if (R / need).exists():
            fn()
            print("dibuat:", fn.__name__ if hasattr(fn, "__name__") else need)


if __name__ == "__main__":
    main()
