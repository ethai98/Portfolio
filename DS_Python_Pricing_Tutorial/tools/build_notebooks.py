"""
build_notebooks.py
==================
Converts the `# %%`-cell lesson scripts into Jupyter notebooks (.ipynb) so you
can use whichever you prefer — interactive scripts in VSCode, or classic
notebooks in Jupyter/Lab/Colab. The `.py` files remain the source of truth
(they're the git-friendly version); regenerate notebooks any time with:

    python tools/build_notebooks.py

Notebooks are written next to the scripts (tutorial root) so the relative
`data/freight_quotes.csv` path resolves the same way.
"""

from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

ROOT = Path(__file__).resolve().parent.parent
LESSONS = [
    "01_python_pandas_foundations",
    "02_data_visualization",
    "03_regression",
    "04_causal_inference",
    "05_machine_learning",
]


def split_cells(text):
    """Yield (kind, source) for each # %% cell. kind is 'code' or 'markdown'."""
    lines = text.splitlines()
    cells, current, kind = [], [], "code"
    started = False
    for line in lines:
        if line.startswith("# %%"):
            if started:
                cells.append((kind, current))
            started = True
            current = []
            kind = "markdown" if "[markdown]" in line else "code"
        else:
            current.append(line)
    if started:
        cells.append((kind, current))
    return cells


def clean_markdown(src_lines):
    """Strip the leading '# ' comment prefix from markdown cell lines."""
    out = []
    for ln in src_lines:
        if ln.startswith("# "):
            out.append(ln[2:])
        elif ln == "#":
            out.append("")
        else:
            out.append(ln)
    # trim leading/trailing blank lines
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def build(name):
    src = (ROOT / f"{name}.py").read_text()
    nb = new_notebook()
    for kind, lines in split_cells(src):
        if kind == "markdown":
            md = clean_markdown(lines)
            if md.strip():
                nb.cells.append(new_markdown_cell(md))
        else:
            code = "\n".join(lines).strip("\n")
            if code.strip():
                nb.cells.append(new_code_cell(code))
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    out_path = ROOT / f"{name}.ipynb"
    nbformat.write(nb, out_path)
    return out_path, len(nb.cells)


if __name__ == "__main__":
    for name in LESSONS:
        path, n = build(name)
        print(f"Wrote {path.name}  ({n} cells)")
