"""Notebook üretme/çalıştırma yardımcıları (tools/build_notebooks.py kullanır)."""
import nbformat
from nbclient import NotebookClient
from pathlib import Path

def md(text): return nbformat.v4.new_markdown_cell(text.strip("\n"))
def code(text): return nbformat.v4.new_code_cell(text.strip("\n"))

SETUP = '''
import sys, warnings
from pathlib import Path
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

pd.set_option("display.max_columns", None, "display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 100
from src import config as C
'''

def build(path: Path, cells, execute=True, timeout=900):
    nb = nbformat.v4.new_notebook()
    nb.cells = [code(SETUP)] + cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    if execute:
        NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    nbformat.write(nb, path)
    errs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.output_type == "error"]
    return len(nb.cells), len(errs)
