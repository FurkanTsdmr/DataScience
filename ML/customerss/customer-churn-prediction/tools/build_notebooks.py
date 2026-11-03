"""Tüm notebook'ları üretir ve çalıştırır:  python tools/build_notebooks.py"""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.nbbuild import build
from tools import nb01_02, nb03_04, nb05_06, nb07_09

NB = {
    "01_business_understanding": nb01_02.NB01, "02_data_understanding": nb01_02.NB02,
    "03_data_cleaning": nb03_04.NB03, "04_eda": nb03_04.NB04,
    "05_statistical_analysis": nb05_06.NB05, "06_feature_engineering": nb05_06.NB06,
    "07_machine_learning": nb07_09.NB07, "08_model_evaluation": nb07_09.NB08, "09_sql_analysis": nb07_09.NB09,
}
only = sys.argv[1:]
for name, cells in NB.items():
    if only and not any(o in name for o in only): continue
    t = time.time()
    n, errs = build(ROOT / "notebooks" / f"{name}.ipynb", cells)
    print(f"{name}: {n} hücre, hata={errs}, {time.time()-t:.0f}s", flush=True)
