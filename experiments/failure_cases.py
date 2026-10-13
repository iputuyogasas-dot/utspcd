"""T37/T38: pemilihan otomatis citra uji terburuk dan diagnosis per tahap (F8).

Aturan pemilihan (ditetapkan sebelum melihat hasil uji): N_WORST citra uji
dengan SSIM terendah pada pipeline final (results/final_test.csv).

Untuk setiap kasus:
  - panel: gelap | keluaran tiap tahap | hasil | acuan | peta galat
  - metrik setelah tiap tahap (di mana galat turun/naik)
  - diagnosis numerik:
      * selisih kecerahan  : rata-rata luminans hasil - acuan
      * galat struktur     : 1 - SSIM pada luminans setelah kecerahan disamakan
                              (sisa galat yang tidak bisa diperbaiki dengan menaikkan/menurunkan kecerahan)
      * galat per wilayah  : SSIM per pita horizontal (atas/tengah/bawah), untuk melihat bayangan lokal
      * galat warna        : ΔE setelah kecerahan disamakan
      * derau tersisa      : simpangan baku residual frekuensi tinggi hasil vs acuan

Keluaran: results/failures/*.png, results/failures/stage_metrics.csv,
results/failures/diagnosis.csv
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.final_eval import load_final
from src import config, plotstyle
from src.dataset import list_pairs, load_pair
from src.metrics import delta_e, full_reference, luminance, ssim
from src.pipeline import Pipeline

N_WORST = 4
OUT = config.RESULTS_DIR / "failures"


def select_worst(n=N_WORST):
    df = pd.read_csv(config.RESULTS_DIR / "final_test.csv")
    final = df[df.method == "final"].sort_values("ssim")
    return final.head(n), final


def gray_ssim(a, b):
    from skimage.metrics import structural_similarity
    return structural_similarity(a, b, data_range=255, gaussian_weights=True, sigma=1.5,
                                 use_sample_covariance=False)


def match_brightness(out, ref):
    """Skalakan hasil agar rata-rata luminansnya sama dengan acuan (hanya untuk diagnosis)."""
    gain = luminance(ref).mean() / max(luminance(out).mean(), 1e-6)
    return np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def high_freq_std(rgb):
    y = luminance(rgb).astype(np.float32)
    return float((y - cv2.GaussianBlur(y, (0, 0), 2.0)).std())


def diagnose(out, ref):
    y_out, y_ref = luminance(out), luminance(ref)
    matched = match_brightness(out, ref)
    yo = np.clip(luminance(matched), 0, 255).astype(np.uint8)
    yr = np.clip(y_ref, 0, 255).astype(np.uint8)
    h = yo.shape[0]
    bands = {name: gray_ssim(yo[s], yr[s]) for name, s in
             [("ssim_atas", slice(0, h // 3)), ("ssim_tengah", slice(h // 3, 2 * h // 3)),
              ("ssim_bawah", slice(2 * h // 3, h))]}
    return {"lum_out": float(y_out.mean()), "lum_ref": float(y_ref.mean()),
            "lum_diff": float(y_out.mean() - y_ref.mean()),
            "ssim_y_matched": gray_ssim(yo, yr), **bands,
            "delta_e_matched": delta_e(matched, ref),
            "hf_std_out": high_freq_std(out), "hf_std_ref": high_freq_std(ref)}


def panel(name, dark, stages, out, ref, metrics):
    plotstyle.apply()
    names = {"denoise": "reduksi derau", "contrast": "gamma + CLAHE"}
    keys = [k for k in stages if k != "4_postprocess"]
    tiles = [("Gelap (input)", dark)]
    for prev, k in zip(keys, keys[1:]):
        label = names.get(k.split("_", 1)[1], k)
        if np.array_equal(stages[k], stages[prev]):
            label += " (nonaktif)"
        tiles.append((label, stages[k]))
    tiles += [("Hasil (+saturasi)", out), ("Acuan", ref)]
    err = np.abs(luminance(out) - luminance(ref))
    n = len(tiles) + 1
    fig, axes = plt.subplots(1, n, figsize=(2.1 * n, 4.4))
    for ax, (title, img) in zip(axes, tiles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    im = axes[-1].imshow(err, cmap=plotstyle.SEQUENTIAL, vmin=0, vmax=80)
    axes[-1].set_title("|galat luminans|", fontsize=9)
    axes[-1].axis("off")
    fig.colorbar(im, ax=axes[-1], fraction=0.05, pad=0.02)
    fig.suptitle(f"{name}: PSNR {metrics['psnr']:.2f} dB, SSIM {metrics['ssim']:.3f}, "
                 f"ΔE {metrics['delta_e']:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=130)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    worst, all_final = select_worst()
    pipe = Pipeline(load_final())
    pairs = {p.name: p for p in list_pairs()}
    stage_rows, diag_rows = [], []

    print("Citra uji terburuk (SSIM terendah, pipeline final):")
    print(worst[["image", "condition", "psnr", "ssim", "delta_e"]].round(3).to_string(index=False))

    # Diagnosis untuk semua citra uji, agar kasus terburuk bisa dibandingkan dengan sisanya.
    for name in all_final.image:
        dark, ref = (np.array(x) for x in load_pair(pairs[name]))
        out, stages = pipe.run(dark, keep_intermediate=True)
        d = {"image": name, "condition": pairs[name].condition,
             "worst": name in set(worst.image), **full_reference(out, ref), **diagnose(out, ref)}
        diag_rows.append(d)
        if name in set(worst.image):
            for stage, img in [("0_input", dark), *stages.items()]:
                stage_rows.append({"image": name, "stage": stage, **full_reference(img, ref),
                                   "lum_mean": float(luminance(img).mean())})
            panel(name, dark, stages, out, ref, d)

    diag = pd.DataFrame(diag_rows)
    diag.to_csv(OUT / "diagnosis.csv", index=False)
    stages = pd.DataFrame(stage_rows)
    stages.to_csv(OUT / "stage_metrics.csv", index=False)
    pd.set_option("display.width", 220)
    print("\nMetrik per tahap (kasus terburuk):\n" + stages.round(3).to_string(index=False))
    cols = ["image", "condition", "ssim", "lum_diff", "ssim_y_matched", "ssim_atas", "ssim_tengah",
            "ssim_bawah", "delta_e", "delta_e_matched", "hf_std_out", "hf_std_ref"]
    print("\nDiagnosis (semua citra uji, urut SSIM):\n" +
          diag.sort_values("ssim")[cols].round(3).to_string(index=False))
    print("\nRata-rata per kondisi:\n" + diag.groupby("condition")[cols[2:]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
