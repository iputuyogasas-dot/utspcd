"""Unit test metrik (T13) dan penjaga data uji pada runner eksperimen (T28)."""

import numpy as np
import pytest
from skimage.metrics import peak_signal_noise_ratio

from src.metrics import delta_e, entropy, no_reference, psnr, ssim

rng = np.random.default_rng(0)


def _img(shape=(48, 64, 3)):
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def test_identical_images():
    a = _img()
    assert psnr(a, a) == float("inf")
    assert ssim(a, a) == pytest.approx(1.0)
    assert delta_e(a, a) == pytest.approx(0.0, abs=1e-9)


def test_psnr_known_noise():
    a = np.full((50, 50, 3), 100, np.uint8)
    b = a + np.uint8(10)  # MSE = 100
    assert psnr(b, a) == pytest.approx(10 * np.log10(255 ** 2 / 100))


def test_psnr_matches_skimage():
    a, b = _img(), _img()
    assert psnr(a, b) == pytest.approx(peak_signal_noise_ratio(b, a, data_range=255))


def test_metrics_order_and_range():
    a = _img()
    noisy = np.clip(a.astype(int) + rng.normal(0, 10, a.shape), 0, 255).astype(np.uint8)
    noisier = np.clip(a.astype(int) + rng.normal(0, 40, a.shape), 0, 255).astype(np.uint8)
    assert psnr(noisy, a) > psnr(noisier, a)
    assert ssim(noisy, a) > ssim(noisier, a)
    assert delta_e(noisy, a) < delta_e(noisier, a)
    assert -1 <= ssim(noisier, a) <= 1


def test_mask_restricts_region():
    a = np.full((40, 40, 3), 100, np.uint8)
    b = a.copy()
    b[:, 20:] = 200  # hanya separuh kanan yang berbeda
    mask = np.zeros((40, 40), bool)
    mask[:, :10] = True
    assert psnr(b, a, mask) == float("inf")
    assert delta_e(b, a, mask) == pytest.approx(0.0, abs=1e-9)
    assert ssim(b, a, mask) == pytest.approx(1.0)


def test_mask_errors():
    a = _img()
    with pytest.raises(ValueError):
        psnr(a, a, np.zeros(a.shape[:2], bool))
    with pytest.raises(ValueError):
        psnr(a, a, np.ones((3, 3), bool))
    with pytest.raises(ValueError):
        psnr(a, _img((10, 10, 3)))


def test_no_reference():
    flat = np.full((10, 10, 3), 80, np.uint8)
    nr = no_reference(flat)
    assert nr["mean_lum"] == pytest.approx(80, abs=0.5)
    assert nr["std_lum"] == pytest.approx(0, abs=1e-6)
    assert nr["entropy"] == 0
    assert entropy(np.arange(256, dtype=np.uint8)) == pytest.approx(8.0)


def test_runner_refuses_test_scenarios():
    from src import config
    if not config.DATA_DIR.is_dir():
        pytest.skip(f"dataset tidak ada di {config.DATA_DIR} (atur PCD_DATA_DIR)")
    from experiments.common import TestDataAccessError, guarded_pairs
    from src.split import load_split

    dev = guarded_pairs("dev")
    assert {p.scenario for p in dev} == set(load_split()["dev"])
    with pytest.raises(TestDataAccessError):
        guarded_pairs("test")
    with pytest.raises(TestDataAccessError):
        guarded_pairs("all")
