"""Copy the final tables and figures from results/rerun/** into the repository's reports/ folders, with stable names.

  reports/tables/   R-*.md, R-*.csv, R-*.tex (one file per table, prefixed by the folder they came from)
  reports/figures/  *.png and *.pdf figures (600 dpi PNG + vector PDF)
Only copies (never moves or deletes); an existing file is overwritten only if the source is newer. Run from the repository root:
  python rerun/scripts/collect_reports.py
"""
import shutil
from pathlib import Path

R = Path("results/rerun"); T = Path("reports/tables"); F = Path("reports/figures"); T.mkdir(parents=True, exist_ok=True); F.mkdir(parents=True, exist_ok=True); n = 0
def cp(src, dst):
    global n
    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime: shutil.copy2(src, dst); n += 1
for f in R.rglob("*"):
    if not f.is_file() or "rater_sheets" in f.parts or "blinding" in f.name: continue
    tag = "_".join(f.relative_to(R).parts[:-1])
    if f.name.startswith("R-") and f.suffix in (".md", ".csv", ".tex"): cp(f, T / f"{tag}__{f.name}" if tag else T / f.name)
    elif f.suffix in (".png", ".pdf") and (f.name.startswith(("fig", "fig")) or "figures" in f.parts or "stats" in f.parts): cp(f, F / f.name)
print(f"{n} files copied to reports/tables and reports/figures")
