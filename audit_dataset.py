"""Audit keselarasan pasangan acuan vs citra gelap (T07, PRD 5.3 dan 11).

Untuk setiap pasangan pada skala kerja:
  1. Kedua citra diubah ke magnitudo gradien (Sobel) yang sudah dinormalisasi,
     agar perbedaan pencahayaan tidak memengaruhi pencocokan.
  2. Pergeseran global diukur dengan phase correlation.
  3. Rotasi/skala kecil diperiksa dengan ECC (model Euclidean).
  4. Pergeseran lokal diukur per blok 4x4; blok bertekstur rendah diabaikan.
     Blok yang bergeser menandakan objek berpindah walau kamera diam.

Pasangan dinyatakan LOLOS bila |pergeseran global| <= 1 px, |rotasi| <= 0.2°,
dan pergeseran blok maksimum <= 2 px (semua pada skala 1/4, setara 4/8 px
di resolusi penuh).

Keluaran: results/dataset/audit.csv dan results/dataset/audit/sXX_overlay.png.
"""

import cv2
import numpy as np
import pandas as pd

from src import config
from src.dataset import list_pairs, load_image, save_rgb

GLOBAL_MAX_PX = 1.0
ROT_MAX_DEG = 0.2
BLOCK_MAX_PX = 2.0
BLOCKS = 4


def gradient_image(rgb):
    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = np.log1p(gray)  # tekan perbedaan kecerahan global
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 1.5)
    return (mag - mag.mean()) / (mag.std() + 1e-6)


def phase_shift(a, b):
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return dx, dy, resp


def ecc_rotation(a, b):
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_EUCLIDEAN,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5),
                                       None, 5)
    except cv2.error:
        return float("nan"), float("nan"), float("nan")
    angle = np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))
    return float(angle), float(warp[0, 2]), float(warp[1, 2])


def block_shifts(a, b, n=BLOCKS, min_texture=0.5):
    h, w = a.shape
    shifts = []
    for i in range(n):
        for j in range(n):
            ys, xs = slice(i * h // n, (i + 1) * h // n), slice(j * w // n, (j + 1) * w // n)
            pa, pb = a[ys, xs], b[ys, xs]
            if pa.std() < min_texture or pb.std() < min_texture:
                continue
            dx, dy, resp = phase_shift(np.ascontiguousarray(pa), np.ascontiguousarray(pb))
            if resp > 0.05:
                shifts.append(np.hypot(dx, dy))
    return (max(shifts) if shifts else float("nan")), len(shifts)


def overlay(ref, dark):
    """Tepi acuan di kanal hijau, tepi citra gelap di merah/biru: tepi sejajar tampak putih."""
    ea, eb = (np.clip(gradient_image(x) * 60, 0, 255).astype(np.uint8) for x in (ref, dark))
    return np.dstack([eb, ea, eb])


def main():
    rows = []
    out_dir = config.DATASET_RESULTS_DIR / "audit"
    by_scenario = {}
    for pair in list_pairs():
        ref, dark = load_image(pair.ref_path), load_image(pair.dark_path)
        ga, gb = gradient_image(ref), gradient_image(dark)
        dx, dy, resp = phase_shift(ga, gb)
        angle, ex, ey = ecc_rotation(ga, gb)
        bmax, nblocks = block_shifts(ga, gb)
        shift = float(np.hypot(dx, dy))
        ok = shift <= GLOBAL_MAX_PX and (np.isnan(angle) or abs(angle) <= ROT_MAX_DEG) \
            and (np.isnan(bmax) or bmax <= BLOCK_MAX_PX)
        rows.append({"image": pair.name, "scenario": pair.scenario, "condition": pair.condition,
                     "shift_dx": dx, "shift_dy": dy, "shift_px": shift, "phase_response": resp,
                     "ecc_rotation_deg": angle, "ecc_tx": ex, "ecc_ty": ey,
                     "block_shift_max_px": bmax, "blocks_used": nblocks,
                     "status": "LOLOS" if ok else "PERIKSA"})
        by_scenario.setdefault(pair.scenario, [ref]).append(overlay(ref, dark))

    for sc, imgs in by_scenario.items():
        ref = imgs[0]
        small = [cv2.resize(x, (ref.shape[1] // 2, ref.shape[0] // 2)) for x in imgs]
        save_rgb(out_dir / f"{sc}_overlay.png", np.hstack(small))

    df = pd.DataFrame(rows)
    df.to_csv(config.DATASET_RESULTS_DIR / "audit.csv", index=False)
    pd.set_option("display.width", 200)
    print(df[["image", "shift_px", "ecc_rotation_deg", "block_shift_max_px", "blocks_used", "status"]]
          .round(3).to_string(index=False))
    print(f"\nLOLOS: {(df.status == 'LOLOS').sum()} / {len(df)} pasangan")
    print(f"Pergeseran global maks: {df.shift_px.max():.2f} px; rotasi maks: {df.ecc_rotation_deg.abs().max():.3f}°; "
          f"pergeseran blok maks: {df.block_shift_max_px.max():.2f} px (skala {config.WORK_SCALE})")
    return df


if __name__ == "__main__":
    main()
