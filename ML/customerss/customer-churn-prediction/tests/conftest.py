import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def cleaning():
    from src.data_cleaning import clean_data, read_raw
    return clean_data(read_raw())


@pytest.fixture(scope="session")
def reference():
    import pandas as pd
    from src import config as C
    return pd.read_csv(C.REFERENCE_PATH).set_index("customer_id")
