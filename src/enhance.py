"""Tahap 2-4: reduksi derau, perbaikan kontras, pasca-proses, dan koreksi warna.

Setiap fungsi kontras/pemertajaman bekerja pada satu kanal uint8 (H, W).
Fungsi reduksi derau bekerja pada citra RGB uint8 (H, W, 3).
"""

import cv2
import numpy as np

from src import from_scratch


def _to_u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ---------------- reduksi derau (RGB) ----------------

def denoise_none(rgb):
    return np.array(rgb)


def denoise_median(rgb, ksize=3, impl="cv2"):
    if impl == "scratch":
        return from_scratch.median_filter(rgb, ksize)
    return cv2.medianBlur(np.ascontiguousarray(rgb), ksize)


def denoise_gaussian(rgb, ksize=5, sigma=0):
    return cv2.GaussianBlur(rgb, (ksize, ksize), sigma)


def denoise_bilateral(rgb, d=7, sigma_color=25, sigma_space=5):
    return cv2.bilateralFilter(np.ascontiguousarray(rgb), d, sigma_color, sigma_space)


def denoise_nlm(rgb, h=5, h_color=5, template=7, search=21):
    """Non-local means (Buades dkk., 2005) versi OpenCV; h = kekuatan filter luminans."""
    return cv2.fastNlMeansDenoisingColored(np.ascontiguousarray(rgb), None, h, h_color, template, search)


DENOISE = {
    "none": denoise_none,
    "median": denoise_median,
    "gaussian": denoise_gaussian,
    "bilateral": denoise_bilateral,
    "nlm": denoise_nlm,
}


# ---------------- perbaikan kontras (satu kanal) ----------------

def gamma(ch, g=0.5):
    """Koreksi gamma: out = 255 * (in/255)^g. g < 1 mencerahkan."""
    lut = _to_u8(255.0 * (np.arange(256) / 255.0) ** g)
    return lut[ch]


def adaptive_gamma(ch, target=0.4, g_min=0.1, g_max=1.5):
    """Gamma dipilih per citra agar rata-rata luminans (skala 0-1) mendekati `target`.

    g = log(target) / log(mean), sehingga mean^g = target.
    """
    m = np.clip(ch.mean() / 255.0, 1e-3, 0.999)
    g = float(np.clip(np.log(target) / np.log(m), g_min, g_max))
    return gamma(ch, g)


def hist_eq(ch, impl="scratch"):
    """Ekualisasi histogram global; impl='scratch' memakai versi NumPy sendiri."""
    return from_scratch.equalize_hist(ch) if impl == "scratch" else cv2.equalizeHist(ch)


def clahe(ch, clip=2.0, tile=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(ch)


def _stretch(x, low_pct=1.0, high_pct=99.0):
    lo, hi = np.percentile(x, [low_pct, high_pct])
    return _to_u8((x - lo) / max(hi - lo, 1e-6) * 255.0)


def ssr(ch, sigma=80, low_pct=1.0, high_pct=99.0):
    """Single-Scale Retinex: log(I) - log(G_σ * I), lalu direntangkan ke 0-255."""
    i = ch.astype(np.float32) + 1.0
    r = np.log(i) - np.log(cv2.GaussianBlur(i, (0, 0), sigma))
    return _stretch(r, low_pct, high_pct)


def msr(ch, sigmas=(15, 80, 250), low_pct=1.0, high_pct=99.0):
    """Multi-Scale Retinex: rata-rata SSR pada beberapa skala."""
    i = ch.astype(np.float32) + 1.0
    log_i = np.log(i)
    r = sum(log_i - np.log(cv2.GaussianBlur(i, (0, 0), s)) for s in sigmas) / len(sigmas)
    return _stretch(r, low_pct, high_pct)


def illum_flatten(ch, sigma=60, alpha=0.3):
    """Koreksi iluminasi lokal parsial: out = I · (mean(L) / L)^alpha, L = G_σ * I.

    alpha = 0 tidak mengubah apa pun; alpha = 1 meratakan iluminasi penuh (mirip SSR
    tanpa log). Nilai kecil meredam gradien cahaya tanpa membuat citra datar.
    """
    i = ch.astype(np.float32) + 1.0
    light = cv2.GaussianBlur(i, (0, 0), sigma)
    return _to_u8(i * (light.mean() / light) ** alpha - 1.0)


CONTRAST = {
    "illum_flatten": illum_flatten,
    "gamma": gamma,
    "adaptive_gamma": adaptive_gamma,
    "hist_eq": hist_eq,
    "clahe": clahe,
    "ssr": ssr,
    "msr": msr,
}


# ---------------- pasca-proses (satu kanal) ----------------

def unsharp(ch, sigma=1.0, amount=0.5):
    """Unsharp mask: out = in + amount * (in - G_σ * in)."""
    if amount == 0:
        return ch.copy()
    f = ch.astype(np.float32)
    return _to_u8(f + amount * (f - cv2.GaussianBlur(f, (0, 0), sigma)))


# ---------------- koreksi warna (RGB) ----------------

def gray_world(rgb):
    """White balance gray-world: samakan rata-rata tiap kanal dengan rata-rata keseluruhan."""
    f = rgb.astype(np.float32)
    means = f.reshape(-1, 3).mean(0)
    return _to_u8(f * (means.mean() / np.maximum(means, 1e-6)))


def saturation_boost(rgb, factor=1.2):
    """Kalikan saturasi HSV dengan `factor` (untuk mengimbangi warna pudar)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
