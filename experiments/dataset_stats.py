"""Metadata EXIF, statistik luminans, dan gambar dataset untuk laporan (T06, T08, T11).

Keluaran:
  data/metadata.csv                         EXIF + statistik per file
  results/dataset/dataset_stats.csv                 ringkasan per kondisi
  results/figures/contact_sheet.png         10 skenario × (acuan, K1-K5), apa adanya
  results/figures/contact_sheet_boost.png   sama, citra gelap dikali gain agar rata-ratanya = acuan (hanya visualisasi)
  results/figures/hist_per_condition.png    histogram luminans rata-rata per kondisi
  results/figures/scene_differences.png     contoh perbedaan adegan (bayangan K4/K5, pantulan)
  results/figures/desaturation.png          efek mencerahkan Y saja (YCrCb) vs rasio luminans
"""

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src import config, plotstyle
from src.dataset import scan, load_image, save_rgb
from src.enhance import gamma
from src.metrics import luminance, no_reference
from src.preprocess import apply_on_luminance

EXIF_IFD = 0x8769
TAGS = {"iso": 0x8827, "exposure_s": 0x829A, "f_number": 0x829D, "focal_mm": 0x920A,
        "datetime": 0x9003}


def _text(v):
    """String EXIF dari ponsel ini diisi byte null; buang padding-nya."""
    return str(v or "").replace(chr(0), "").strip()


def exif_row(path):
    with Image.open(path) as im:
        ex = im.getexif()
        sub = ex.get_ifd(EXIF_IFD)
        row = {"file": path.name, "width": im.width, "height": im.height,
               "make": _text(ex.get(0x010F)), "model": _text(ex.get(0x0110)), "orientation": ex.get(0x0112)}
        for k, t in TAGS.items():
            v = sub.get(t)
            row[k] = float(v) if k in ("exposure_s", "f_number", "focal_mm") and v is not None else v
    if row["exposure_s"]:
        row["shutter"] = f"1/{round(1 / row['exposure_s'])}"
    return row


def metadata():
    rows = []
    for sc, files in sorted(scan().items()):
        for key in ["acuan", "k1", "k2", "k3", "k4", "k5"]:
            p = files[key]
            row = {"scenario": sc, "kind": "acuan" if key == "acuan" else key.upper(), **exif_row(p)}
            row.update(no_reference(np.asarray(load_image(p))))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(config.METADATA_PATH, index=False)
    return df


def summary(df):
    g = df.groupby("kind")
    out = pd.DataFrame({
        "n": g.size(),
        "iso": g.iso.agg(lambda s: "/".join(map(str, sorted(set(s))))),
        "shutter": g.shutter.agg(lambda s: "/".join(sorted(set(s)))),
        "mean_lum_mean": g.mean_lum.mean(), "mean_lum_min": g.mean_lum.min(),
        "mean_lum_max": g.mean_lum.max(), "std_lum_mean": g.std_lum.mean(),
        "entropy_mean": g.entropy.mean(),
    }).reindex(["acuan", "K1", "K2", "K3", "K4", "K5"])
    config.DATASET_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.round(2).to_csv(config.DATASET_RESULTS_DIR / "dataset_stats.csv")
    return out


def _thumb(rgb, w=216):
    h = int(rgb.shape[0] * w / rgb.shape[1])
    return cv2.resize(np.asarray(rgb), (w, h), interpolation=cv2.INTER_AREA)


def contact_sheets():
    groups = scan()
    raw_rows, boost_rows = [], []
    for sc in sorted(groups):
        ref = load_image(groups[sc]["acuan"])
        raw, boost = [_thumb(ref)], [_thumb(ref)]
        for k in range(1, 6):
            d = np.asarray(load_image(groups[sc][f"k{k}"]))
            raw.append(_thumb(d))
            gain = luminance(ref).mean() / max(luminance(d).mean(), 1e-6)
            boost.append(_thumb(np.clip(d.astype(np.float32) * gain, 0, 255).astype(np.uint8)))
        raw_rows.append(np.hstack(raw))
        boost_rows.append(np.hstack(boost))
    save_rgb(config.FIGURES_DIR / "contact_sheet.png", np.vstack(raw_rows))
    save_rgb(config.FIGURES_DIR / "contact_sheet_boost.png", np.vstack(boost_rows))


def histograms():
    groups = scan()
    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for kind in ["acuan", "k1", "k2", "k3", "k4", "k5"]:
        h = np.zeros(256)
        for sc in groups:
            y = np.clip(np.rint(luminance(np.asarray(load_image(groups[sc][kind])))), 0, 255).astype(np.uint8)
            h += np.bincount(y.ravel(), minlength=256)
        label = "Acuan" if kind == "acuan" else f"{kind.upper()} ({config.CONDITIONS[kind.upper()]})"
        ax.plot(h / h.sum(), label=label, lw=2.2 if kind == "acuan" else 1.4,
                color=plotstyle.INK if kind == "acuan" else plotstyle.CONDITION_COLORS[kind.upper()])
    ax.set_xlim(0, 255)
    ax.set_yscale("log")
    ax.set_xlabel("Luminans (0-255)")
    ax.set_ylabel("Proporsi piksel (log)")
    ax.set_title("Histogram luminans per kondisi (10 skenario)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "hist_per_condition.png", dpi=150)
    plt.close(fig)


def scene_differences(sc="s01"):
    files = scan()[sc]
    ref = np.asarray(load_image(files["acuan"]))
    tiles, titles = [_thumb(ref, 300)], ["Acuan"]
    for k in ["k1", "k4", "k5"]:
        d = np.asarray(load_image(files[k]))
        gain = luminance(ref).mean() / luminance(d).mean()
        tiles.append(_thumb(np.clip(d.astype(np.float32) * gain, 0, 255).astype(np.uint8), 300))
        titles.append(f"{k.upper()} × gain {gain:.1f}")
    fig, axes = plt.subplots(1, 4, figsize=(10, 4.6))
    for ax, t, title in zip(axes, tiles, titles):
        ax.imshow(t)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"{sc}: bayangan lampu samping (K4) dan bayangan tirai (K5) tidak ada di acuan", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "scene_differences.png", dpi=150)
    plt.close(fig)


def desaturation(name="s07_k2"):
    sc, k = name.split("_")
    files = scan()[sc]
    ref, d = np.asarray(load_image(files["acuan"])), np.asarray(load_image(files[k]))
    fn = lambda ch: gamma(ch, 0.45)
    tiles = [d, apply_on_luminance(d, fn, "ycrcb"), apply_on_luminance(d, fn, "ratio"), ref]
    titles = ["Gelap (K2)", "Gamma pada Y (YCrCb)", "Gamma via rasio luminans", "Acuan"]
    fig, axes = plt.subplots(1, 4, figsize=(10, 4.6))
    for ax, t, title in zip(axes, tiles, titles):
        ax.imshow(_thumb(t, 300))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle("Mencerahkan Y saja mempertahankan Cr/Cb absolut sehingga warna memudar", fontsize=10)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "desaturation.png", dpi=150)
    plt.close(fig)


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = metadata()
    s = summary(df)
    print(s.round(2).to_string())
    contact_sheets()
    histograms()
    scene_differences()
    desaturation()
    print("Gambar disimpan di", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
