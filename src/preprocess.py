"""Tahap 1: validasi citra dan konversi ke ruang luminans-kromatik."""

import cv2
import numpy as np


def validate(rgb):
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"citra harus RGB uint8 (H, W, 3), bukan {rgb.dtype} {rgb.shape}")
    return rgb


def split_luminance(rgb, space):
    """Pisahkan kanal luminans. Kembalikan (lum uint8, state) untuk merge_luminance.

    - ycrcb: Y dari YCrCb; Cr/Cb dipertahankan apa adanya.
    - lab:   L dari CIELAB 8-bit; a/b dipertahankan.
    - hsv:   V dari HSV; H/S dipertahankan.
    - ratio: luma BT.601; saat digabung, setiap kanal RGB dikalikan rasio
             Y_baru / Y_lama sehingga rona dan saturasi relatif tetap.
    """
    rgb = validate(rgb)
    if space == "ycrcb":
        conv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        return conv[..., 0].copy(), (space, conv)
    if space == "lab":
        conv = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        return conv[..., 0].copy(), (space, conv)
    if space == "hsv":
        conv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        return conv[..., 2].copy(), (space, conv)
    if space == "ratio":
        y = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return y, (space, rgb)
    raise ValueError(f"ruang warna tidak dikenal: {space}")


def merge_luminance(lum, state):
    space, conv = state
    if space == "ratio":
        y_old = cv2.cvtColor(conv, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gain = (lum.astype(np.float32) + 1.0) / (y_old + 1.0)
        out = conv.astype(np.float32) * gain[..., None]
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)
    conv = conv.copy()
    idx = 2 if space == "hsv" else 0
    conv[..., idx] = lum
    back = {"ycrcb": cv2.COLOR_YCrCb2RGB, "lab": cv2.COLOR_LAB2RGB, "hsv": cv2.COLOR_HSV2RGB}[space]
    return cv2.cvtColor(conv, back)


def apply_on_luminance(rgb, fn, space):
    """Terapkan fn(uint8 2D) -> uint8 2D ke kanal luminans, atau ke tiap kanal bila space='rgb'."""
    if space == "rgb":
        return np.dstack([fn(np.ascontiguousarray(rgb[..., c])) for c in range(3)])
    lum, state = split_luminance(rgb, space)
    return merge_luminance(fn(lum), state)
