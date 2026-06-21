# Python Data Science for Pricing — A Hands-On Tutorial

A self-contained, pricing-themed course that takes you from pandas basics to
regression, causal inference, and machine learning — using a synthetic
freight-brokerage quoting dataset modeled on the kind of work a pricing team
(e.g., **Uber Freight Pricing**) actually does.

Every concept is framed around a real pricing question: *What does a lane cost to
cover? How does price affect win rate? Did our pricing change actually work? What
price should we quote?*

## What's inside

| File | Topic |
|------|-------|
| [`00_VSCode_AI_Setup_Guide.md`](00_VSCode_AI_Setup_Guide.md) | Set up VSCode for corporate DS work + how to use AI tools responsibly on a team |
| [`data/generate_pricing_data.py`](data/generate_pricing_data.py) | Builds the synthetic `freight_quotes.csv` (run this first) |
| [`01_python_pandas_foundations.py`](01_python_pandas_foundations.py) | Loading, cleaning, filtering, group-by, time series |
| [`02_data_visualization.py`](02_data_visualization.py) | matplotlib + seaborn; the price–response curve |
| [`03_regression.py`](03_regression.py) | OLS, multiple regression, logits, elasticities, robust SEs |
| [`04_causal_inference.py`](04_causal_inference.py) | A/B tests, difference-in-differences, confounding |
| [`05_machine_learning.py`](05_machine_learning.py) | Pipelines, random forests, classifiers, a price recommender |
| `solutions/` | Worked answers to every exercise |
| `*.ipynb` | Jupyter-notebook versions of each lesson (for Jupyter/Lab/Colab) |
| [`tools/build_notebooks.py`](tools/build_notebooks.py) | Regenerates the notebooks from the `.py` files |

## Quick start

```bash
cd DS_Python_Pricing_Tutorial
python -m venv .venv && source .venv/bin/activate   # or use conda
pip install -r requirements.txt
python data/generate_pricing_data.py                # writes data/freight_quotes.csv
```

Open `01_python_pandas_foundations.py` in VSCode, pick your interpreter
(`Cmd+Shift+P` → "Python: Select Interpreter"), and run cells with `Shift+Enter`.
Work through `01` → `05` in order.

## How to use this

- The lesson files use `# %%` cell markers — run them interactively in VSCode,
  no Jupyter notebook needed. See the setup guide for why this matters on a team.
  Prefer classic notebooks (or Colab)? Open the matching `.ipynb` instead — same
  content, regenerated from the scripts via `python tools/build_notebooks.py`.
- **Do the exercises** at the bottom of each lesson before opening `solutions/`.
- The data generator prints the *true* coefficients it used. In lessons 03–05,
  check whether your models recover them — that's how you know your code is right.

## A note on the data

It's **synthetic**. No real Uber Freight data is used or implied. The numbers are
designed to behave realistically (seasonality, fuel pass-through, price
elasticity, a baked-in pricing experiment) so the methods you practice transfer
directly to real work.
