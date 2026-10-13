"""Gabungkan bab laporan menjadi laporan.html lalu cetak ke laporan.pdf dengan Chrome headless.

  python docs/report/build_report.py
"""

import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PARTS = ["01_pendahuluan_teori.md", "02_isi.md"]
APPENDIX = ["03_lampiran.md"]

CSS = """
@page { size: A4; margin: 17mm 17mm 16mm 17mm; }
body { font-family: "Segoe UI", "DejaVu Sans", Arial, sans-serif; font-size: 10pt; line-height: 1.42;
       color: #0b0b0b; }
h1 { font-size: 16pt; margin: 0 0 2pt; }
h2 { font-size: 13pt; margin: 14pt 0 5pt; border-bottom: 1px solid #c3c2b7; padding-bottom: 2pt;
     break-after: avoid; }
h3 { font-size: 11pt; margin: 10pt 0 4pt; break-after: avoid; }
p, li { margin: 3pt 0; text-align: justify; }
ul, ol { padding-left: 18pt; margin: 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 8.6pt; break-inside: avoid; }
th, td { border: 1px solid #c3c2b7; padding: 2.5pt 4pt; vertical-align: top; text-align: left; }
th { background: #f1f0ec; }
figure { margin: 8pt 0; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figcaption { font-size: 8.6pt; color: #52514e; margin-top: 2pt; }
code { font-family: Consolas, "DejaVu Sans Mono", monospace; font-size: 8.8pt; background: #f1f0ec;
       padding: 0 2pt; }
pre { background: #f1f0ec; padding: 6pt; font-size: 8.6pt; }
.meta { color: #52514e; font-size: 9.5pt; margin-bottom: 8pt; }
.meta td, .meta th { border: none; padding: 1pt 8pt 1pt 0; }
.pipe { display: flex; gap: 6pt; margin: 8pt 0; break-inside: avoid; }
.pipe > div { flex: 1; border: 1.5px solid #2a78d6; border-radius: 4pt; padding: 5pt; font-size: 8.6pt;
              text-align: center; background: #f4f8fd; position: relative; }
.pipe > div:not(:last-child)::after { content: "→"; position: absolute; right: -9pt; top: 40%;
                                      color: #2a78d6; font-weight: bold; }
.appendix { break-before: page; }
"""

HEADER = """
<h1>Pipeline Peningkatan Kualitas Citra Minim Cahaya dan Berderau</h1>
<table class="meta">
<tr><td>Mata kuliah</td><td>Pengolahan Citra Digital, Teknik Informatika - UTS (take-home)</td></tr>
<tr><td>Topik</td><td>No. 6: Peningkatan kualitas citra minim cahaya atau berderau</td></tr>
<tr><td>Kelompok</td><td>[isi nama kelompok dan anggota]</td></tr>
<tr><td>Tanggal</td><td>{date}</td></tr>
<tr><td>Repository</td><td>[isi tautan]</td></tr>
</table>
"""


def md(paths):
    text = "\n\n".join((HERE / p).read_text(encoding="utf-8") for p in paths)
    return markdown.markdown(text, extensions=["tables", "fenced_code", "md_in_html"])


def find_browser():
    for c in [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              shutil.which("chromium"), shutil.which("google-chrome"), shutil.which("chrome")]:
        if c and Path(c).exists():
            return c
    return None


def main():
    html = (f"<!doctype html><html lang='id'><head><meta charset='utf-8'><title>Laporan UTS PCD</title>"
            f"<style>{CSS}</style></head><body>"
            f"{HEADER.format(date=date.today().strftime('%d-%m-%Y'))}{md(PARTS)}"
            f"<div class='appendix'>{md(APPENDIX)}</div></body></html>")
    out_html = HERE / "laporan.html"
    out_html.write_text(html, encoding="utf-8")
    print("HTML:", out_html)

    browser = find_browser()
    if not browser:
        print("Chrome/Edge tidak ditemukan; buka laporan.html di browser lalu Cetak → Simpan sebagai PDF.")
        return
    out_pdf = ROOT / "laporan.pdf"
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out_pdf}", out_html.as_uri()], check=True, capture_output=True)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", out_pdf.read_bytes()))
    print(f"PDF: {out_pdf} ({pages} halaman termasuk lampiran)")


if __name__ == "__main__":
    main()
