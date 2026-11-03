"""SQL dosyasındaki isimli sorguları SQLite üzerinde çalıştırır."""
from __future__ import annotations

import re
import sqlite3

import pandas as pd

from src import config as C

SQL_PATH = C.ROOT / "sql" / "churn_analysis.sql"


def load_queries(path=SQL_PATH) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^-- name:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    # parts = [önsöz, ad1, gövde1, ad2, gövde2, ...]
    return {name: body.strip() for name, body in zip(parts[1::2], parts[2::2])}


def build_connection(db_path: str = ":memory:") -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    pd.read_csv(C.CLEAN_PATH).to_sql("customers", con, if_exists="replace", index=False)
    if C.SCORED_PATH.exists():
        pd.read_csv(C.SCORED_PATH).to_sql("scored_customers", con, if_exists="replace", index=False)
    return con


def run_all(con: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    return {name: pd.read_sql(q, con) for name, q in load_queries().items()}
