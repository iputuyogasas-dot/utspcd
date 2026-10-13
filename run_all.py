"""Satu perintah untuk menjalankan seluruh pipeline dan mencetak tabel metrik (F9).

  python run_all.py                # validasi, audit, verifikasi dari nol, evaluasi akhir (uji), analisis kegagalan
  python run_all.py --quick        # demo cepat: baseline vs pipeline final di citra uji + contoh keluaran
  python run_all.py --experiments  # tambah: ulang semua eksperimen & tuning di dev (lama)

Konfigurasi final dibaca dari configs/final.json (sudah dikunci setelah tuning di dev).
"""

import argparse
import json
import time

import numpy as np

from src import config


def header(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def quick():
    from experiments.common import run_configs
    from experiments.final_eval import load_final
    from src.dataset import list_pairs, load_image, save_rgb
    from src.evaluate import print_table
    from src.pipeline import Pipeline
    from src.split import load_split

    final = load_final()
    df = run_configs({"baseline": None, "final": final}, subset="test", allow_test=True)
    print_table(df, title="Baseline vs pipeline final (uji, n=20)")

    test = set(load_split()["test"])
    pipe = Pipeline(final)
    out_dir = config.DEMO_DIR
    for p in [p for p in list_pairs() if p.scenario in test][:5]:
        dark = np.array(load_image(p.dark_path))
        ref = np.array(load_image(p.ref_path))
        save_rgb(out_dir / f"{p.name}.jpg", np.hstack([dark, pipe(dark), ref]))
    print(f"\nContoh gelap | hasil | acuan disimpan di {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--experiments", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    from src.dataset import list_pairs
    from src.split import load_split

    header("1. Validasi dataset dan split")
    pairs = list_pairs()
    split = load_split()
    print(f"{len(pairs)} pasangan valid di {config.DATA_DIR}")
    print(f"Split (seed {split['seed']}): dev {split['dev']} | uji {split['test']}")

    if args.quick:
        header("Demo cepat")
        quick()
        print(f"\nSelesai dalam {time.time() - t0:.0f} s")
        return

    import audit_dataset
    from experiments import failure_cases, final_eval, verify_from_scratch

    header("2. Audit keselarasan pasangan")
    audit = audit_dataset.main()

    header("3. Verifikasi operasi dari nol vs OpenCV")
    verify_from_scratch.main([])

    if args.experiments:
        from experiments import dataset_stats, exp_contrast, exp_denoise, exp_order_colorspace, tune
        for name, mod in [("Statistik dataset", dataset_stats), ("T29 kontras", exp_contrast),
                          ("T30 reduksi derau", exp_denoise), ("T31 urutan & ruang warna", exp_order_colorspace),
                          ("T32 tuning", tune)]:
            header(f"Eksperimen dev: {name}")
            mod.main([]) if mod is exp_denoise else mod.main()

    header("4. Evaluasi akhir pada citra uji")
    print("Konfigurasi final:\n" + json.dumps(final_eval.load_final(), indent=2))
    final_eval.main()

    header("5. Analisis kegagalan (citra uji terburuk)")
    failure_cases.main()

    header("Ringkasan")
    print((config.FINAL_DIR / "test.md").read_text(encoding="utf-8").split("## Per kondisi")[0])
    print(f"Audit: {(audit.status == 'LOLOS').sum()}/{len(audit)} pasangan lolos")
    print(f"Selesai dalam {time.time() - t0:.0f} s. Semua hasil di {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
