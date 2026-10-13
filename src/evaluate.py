"""Pemilihan pasangan per subset dan agregasi metrik per kondisi (F5).

Eksekusi metode pada citra ada di experiments/common.run_configs.
"""

import pandas as pd

from src import config
from src.dataset import list_pairs
from src.split import load_split

METRICS = ["psnr", "ssim", "delta_e"]


def pairs_for(subset):
    """Pasangan untuk 'dev', 'test', atau 'all' berdasarkan split.json."""
    if subset == "all":
        return list_pairs()
    split = load_split()
    if subset not in ("dev", "test"):
        raise ValueError(f"subset tidak dikenal: {subset}")
    return list_pairs(scenarios=set(split[subset]))


def summarize(df, metrics=METRICS + ["time_s"], by="method"):
    """Rata-rata dan simpangan baku: keseluruhan + per kondisi K1-K5 (format panjang)."""
    groups = [("Semua", df)] + [(k, df[df.condition == k]) for k in config.CONDITIONS]
    rows = []
    for label, part in groups:
        if part.empty:
            continue
        for key, g in part.groupby(by, sort=False):
            row = {by: key, "kondisi": label, "n": len(g)}
            for m in metrics:
                row[f"{m}_mean"] = g[m].mean()
                row[f"{m}_std"] = g[m].std(ddof=1) if len(g) > 1 else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def format_table(summary, metrics=METRICS, by="method", decimals=None):
    """Tabel siap cetak dengan sel 'mean ± std'."""
    decimals = decimals or {"psnr": 2, "ssim": 3, "delta_e": 2, "time_s": 3}
    out = summary[[by, "kondisi", "n"]].copy()
    for m in metrics:
        d = decimals.get(m, 3)
        out[m.upper() if m != "delta_e" else "ΔE"] = [
            f"{a:.{d}f} ± {b:.{d}f}" for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_std"])]
    return out


def print_table(df, metrics=METRICS, by="method", title=None):
    if title:
        print(f"\n=== {title} ===")
    table = format_table(summarize(df, metrics, by), metrics, by)
    print(table.to_string(index=False))
    return table


def to_markdown(table):
    cols = list(table.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in table.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)
