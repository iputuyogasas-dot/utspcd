"""Runner eksperimen bersama (T28).

Menjalankan beberapa konfigurasi pipeline secara paralel dan menyimpan hasil per
citra ke CSV. Secara default hanya citra dev yang boleh dimuat: skenario uji
menimbulkan error kecuali allow_test=True, yang hanya dipakai oleh evaluasi akhir
(experiments/final_eval.py) setelah konfigurasi dikunci.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from src import config
from src.dataset import load_pair
from src.evaluate import pairs_for
from src.metrics import full_reference, no_reference
from src.pipeline import Pipeline
from src.split import load_split


class TestDataAccessError(RuntimeError):
    pass


def guarded_pairs(subset="dev", allow_test=False, conditions=None):
    """Pasangan untuk subset (opsional: hanya kondisi tertentu); tolak skenario uji bila allow_test=False."""
    if subset != "dev" and not allow_test:
        raise TestDataAccessError(f"subset '{subset}' butuh allow_test=True (hanya untuk evaluasi akhir)")
    pairs = pairs_for(subset)
    if conditions:
        pairs = [p for p in pairs if p.condition in set(conditions)]
    if not allow_test:
        test = set(load_split()["test"])
        leaked = sorted({p.scenario for p in pairs} & test)
        if leaked:
            raise TestDataAccessError(f"skenario uji ikut dimuat: {leaked}")
    return pairs


def _eval_pair(job):
    """Satu pekerja memproses satu citra untuk semua konfigurasi (citra dimuat sekali)."""
    configs, pair, scale, extra = job
    dark, ref = load_pair(pair, scale)
    rows = []
    for name, cfg in configs:
        pipe = Pipeline(cfg)
        t0 = time.perf_counter()
        out = pipe(dark)
        elapsed = time.perf_counter() - t0
        row = {"method": name, "image": pair.name, "scenario": pair.scenario,
               "condition": pair.condition, **full_reference(out, ref), "time_s": elapsed}
        if extra:
            row.update(no_reference(out))
        rows.append(row)
    return rows


def run_configs(configs, subset="dev", allow_test=False, scale=config.WORK_SCALE,
                workers=None, out_csv=None, extra_metrics=False, verbose=True, conditions=None):
    """configs: dict nama -> config pipeline (None = baseline citra mentah).

    Jumlah pekerja dibatasi (default 6) karena tiap proses memuat OpenCV,
    scikit-image, dan SciPy (±300 MB).
    """
    pairs = guarded_pairs(subset, allow_test, conditions)
    items = [(name, cfg or {}) for name, cfg in configs.items()]
    jobs = [(items, p, scale, extra_metrics) for p in pairs]
    workers = workers or int(os.environ.get("PCD_WORKERS", min(6, os.cpu_count() or 2)))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        per_pair = list(ex.map(_eval_pair, jobs))
    order = {name: i for i, (name, _) in enumerate(items)}
    df = pd.DataFrame([r for rows in per_pair for r in rows])
    df = df.sort_values("method", key=lambda c: c.map(order), kind="stable").reset_index(drop=True)
    if verbose:
        print(f"[{len(configs)} konfigurasi × {len(pairs)} citra ({subset}) selesai dalam {time.time() - t0:.0f} s]")
    if out_csv:
        out_csv = config.RESULTS_DIR / out_csv
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return df


def ranking(df, metrics=("psnr", "ssim", "delta_e")):
    """Rata-rata per metode, diurutkan berdasarkan SSIM lalu PSNR."""
    agg = df.groupby("method", sort=False)[list(metrics) + ["time_s"]].mean()
    return agg.sort_values(["ssim", "psnr"], ascending=False)


def report(df, name, title=None, by="method"):
    """Simpan ringkasan per kondisi (CSV + Markdown) dan cetak tabelnya."""
    from src.evaluate import format_table, summarize, to_markdown

    summary = summarize(df, by=by)
    summary.to_csv(config.RESULTS_DIR / f"{name}_summary.csv", index=False)
    table = format_table(summary, by=by)
    overall = table[table.kondisi == "Semua"].drop(columns="kondisi")
    md = [f"# {title or name}", "", "## Keseluruhan", "", to_markdown(overall), "",
          "## Per kondisi", "", to_markdown(table[table.kondisi != "Semua"]), ""]
    (config.RESULTS_DIR / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n=== {title or name} (keseluruhan) ===")
    print(overall.to_string(index=False))
    return summary
