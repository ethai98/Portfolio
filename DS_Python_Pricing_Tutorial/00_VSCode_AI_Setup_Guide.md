# Setting Up VSCode for Corporate Data Science (with AI)

A practical setup guide aimed at the workflow you'll use on a pricing/analytics
team like **Uber Freight Pricing**. The goal isn't just "make Python run" — it's
to work the way a *team* works: reproducible environments, version control,
clean code, and AI tools used responsibly.

---

## 0. Mental model: how a data science team actually works

Before tools, the workflow. On a corporate pricing team you'll typically:

1. **Pull data** from a warehouse (Snowflake / BigQuery / Databricks) via SQL.
2. **Explore & prototype** in notebooks or scripts (pandas, plotting).
3. **Model** something — a rate predictor, an elasticity estimate, an experiment readout.
4. **Productionize** the parts that matter into versioned, reviewed `.py` modules.
5. **Share** results: a deck, a dashboard, a PR, or a scheduled job.

Notebooks are for *exploration*; `.py` files under version control are for
anything *others depend on*. This tutorial uses `.py` scripts (run cell-by-cell
in VSCode) because that's the habit that scales on a team. More on this below.

---

## 1. The corporate cloud data stack (where the work actually runs)

Your laptop and VSCode are the *cockpit*. The data, and most of the compute,
live in the **cloud** — this is the single biggest difference between a school
project and a corporate pricing job. You won't `pd.read_csv` a 2 TB table; you'll
query it where it lives. Expect a stack roughly like this:

```
   ┌─────────────┐     SQL      ┌──────────────────┐    Python/pandas    ┌──────────┐
   │  Warehouse  │ ───────────▶ │  Notebook /      │ ──────────────────▶ │  Deck /  │
   │ (the data)  │              │  compute layer   │                     │ dashboard│
   └─────────────┘              └──────────────────┘                     └──────────┘
   Snowflake / BigQuery /       Databricks / Hex /                       Tableau /
   Databricks / Redshift        SageMaker / local VSCode                 Looker / PPT
```

**The warehouse — where the data lives (and where you'll write a LOT of SQL):**

| Platform | What it is | You'll hear it called |
|----------|-----------|------------------------|
| **Snowflake** | Cloud SQL warehouse, hugely popular | "Snowflake", "the warehouse" |
| **Google BigQuery** | Google's serverless warehouse | "BQ" |
| **Databricks** | Spark-based lakehouse; notebooks + warehouse + ML in one | "Databricks", "the lakehouse" |
| **Amazon Redshift** | AWS's warehouse | "Redshift" |

The job-one skill here is **SQL**, not Python. On a pricing team you'll spend a
large chunk of your day writing SQL to pull and aggregate lane/quote/rate data,
*then* a smaller slice doing the pandas/modeling work this tutorial teaches. Your
existing `A4_SQL` practice is more directly job-relevant than any single Python
trick — keep it sharp. (Once data is small enough — an aggregated query result —
you pull it into pandas exactly as in these lessons.)

**The compute/notebook layer — where you analyze:**
- **Databricks notebooks** — extremely common; the notebook runs *on* a Spark
  cluster sitting next to the data, so you can crunch huge tables.
- **Hex / Deepnote / SageMaker Studio / Snowflake Notebooks** — cloud notebooks
  that connect straight to the warehouse.
- **Local VSCode** (what we set up) — for development, productionized `.py`
  modules, and analyses on data small enough to pull down.

**The transformation/orchestration layer (you'll meet it within months):**
- **dbt** — defines reusable, version-controlled SQL models (the modern way
  teams build the clean tables you'll query). Increasingly a core DS skill.
- **Airflow / Dagster** — schedule pipelines (e.g., "refresh the rate-prediction
  features every morning at 6am").
- **Git + CI** — the same version-control discipline from section 5, applied to
  SQL models and pipelines, not just Python.

**How this changes your day-to-day vs. this tutorial:**
1. Data comes from a **SQL query against the warehouse**, not a local CSV.
2. Heavy lifting may run on a **cluster** (Spark/Snowflake), not your laptop.
3. Credentials live in a **secrets manager / SSO**, never hard-coded (see §5).
4. Results ship as a **PR, dashboard, or scheduled job**, not a file on your desk.

**The one thing to confirm in week one:** ask your manager *"What's our warehouse,
and what notebook/compute platform does the team use?"* The answer (e.g.,
"Snowflake + Databricks" or "BigQuery + Hex") tells you exactly which SQL dialect
and notebook tool to get fluent in. Everything in this tutorial then runs inside
whichever notebook layer they name — the pandas/regression/causal/ML skills are
identical; only where you *get* the data and *run* the code changes.

---

## 2. Install the essentials

| Tool | Why | Link |
|------|-----|------|
| **VSCode** | The editor | code.visualstudio.com |
| **Python extension** (Microsoft) | IntelliSense, debugging, env selection | Extensions panel |
| **Pylance** | Fast type-checking / autocomplete (ships with Python ext) | bundled |
| **Jupyter extension** | Run `.py` files cell-by-cell + open `.ipynb` | Extensions panel |
| **Ruff** | Ultra-fast linter + formatter (replaces flake8/black/isort) | Extensions panel |
| **GitLens** | See who changed what, when, why | Extensions panel |

You already have **Anaconda Python 3.9**, which is what this tutorial targets.

> **Tip:** Open the Command Palette with `Cmd+Shift+P`. Almost everything in
> VSCode is reachable from there ("Python: Select Interpreter", "Format
> Document", etc.). Learning the palette is the single biggest productivity win.

---

## 3. Virtual environments (the #1 habit that marks a pro)

**Never install packages into your base/global Python.** Each project gets its
own isolated environment so that Project A's `pandas 1.4` can't break Project B.
This is also how your code stays reproducible for teammates and production.

Two common options — your team will standardize on one:

### Option A: `venv` (built-in, lightweight, what most prod teams use)
```bash
# from the project folder
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: `conda` (you have Anaconda; common in research-y DS teams)
```bash
conda create -n pricing python=3.11 -y
conda activate pricing
pip install -r requirements.txt
```

Then tell VSCode to use it: `Cmd+Shift+P` → **Python: Select Interpreter** →
pick the `.venv` / `pricing` environment. The chosen interpreter shows in the
bottom-right status bar; confirm it's the project env, not "base".

> **Corporate reality:** commit `requirements.txt` (or `environment.yml`), never
> commit the `.venv/` folder itself. Add it to `.gitignore`. A teammate should
> be able to clone your repo, create the env from that one file, and reproduce
> your results exactly.

---

## 4. Scripts vs. notebooks — and the "interactive script" sweet spot

- **Notebooks (`.ipynb`)**: great for storytelling and one-off exploration. Bad
  for version control (they're JSON, diffs are unreadable) and for code reuse.
- **Plain scripts (`.py`)**: reviewable, testable, importable. Less interactive.

**The sweet spot, used by many DS teams:** write `.py` files but split them into
cells with `# %%` markers. VSCode's Jupyter extension then lets you run each cell
with `Shift+Enter` and see plots/tables inline — notebook interactivity, but the
file is clean Python that git can diff and you can `import`.

```python
# %% [markdown]
# ## Load the data
# %%
import pandas as pd
df = pd.read_csv("data/freight_quotes.csv")
df.head()        # renders as a table in the interactive window
```

**Every lesson in this tutorial uses `# %%` cells** — open one, click "Run Cell"
above the marker (or `Shift+Enter`), and watch output appear in the interactive
window. This is the exact muscle memory you want for the job.

---

## 5. Git & code review (non-negotiable on a team)

Your team lives in pull requests. Minimum workflow:

```bash
git checkout -b ethai/elasticity-model     # branch per piece of work
# ... edit, run, commit ...
git add 03_regression.py
git commit -m "Add price-elasticity regression for dry van lanes"
git push -u origin ethai/elasticity-model
# open a PR, get a review, merge
```

- **Small, focused commits** with clear messages.
- **Never commit secrets** (warehouse passwords, API keys) or data files with PII.
  Use environment variables / a secrets manager, and `.gitignore` data dumps.
- GitLens lets you hover any line to see the commit that introduced it — gold
  when you inherit a teammate's pricing model.

---

## 6. Integrating AI into the workflow (the part you asked about)

AI coding assistants are now standard on data teams. The three tiers you'll see:

### (a) Inline autocomplete — **GitHub Copilot** / **Claude in your IDE**
Suggests the next line(s) as you type. Best for boilerplate: `groupby`
aggregations, plotting scaffolds, docstrings, test stubs. Accept with `Tab`.

### (b) Chat / agentic assistants — **Claude Code**, **Cursor**, **Copilot Chat**
You describe a task ("read freight_quotes.csv, fit a logit of `won` on
`realized_margin`, plot the elasticity curve") and it writes/edits across files,
runs code, and iterates. This is the highest-leverage tier — it's what generated
*this* tutorial. Use it for scaffolding analyses, refactoring, explaining
unfamiliar code, and debugging stack traces.

### (c) Notebook-native AI — Jupyter AI, Databricks Assistant, etc.
Often what's available inside a corporate warehouse environment.

**How to actually use AI well on a pricing team:**

1. **Treat AI like a fast junior analyst, not an oracle.** It drafts; *you* are
   accountable for correctness. A wrong elasticity number that ships into a
   pricing decision is *your* mistake, not the model's. Always sanity-check
   coefficients against domain intuition and the data.

2. **Give it context.** Point it at the actual schema/columns, paste the real
   error, state the business question. "Why is my reefer margin negative?" beats
   "fix my code."

3. **Verify numbers independently.** Re-derive a key stat a second way (e.g.,
   check a regression coefficient against a simple groupby mean). This tutorial's
   data generator literally prints the "true" coefficients so you can audit the
   model's output.

4. **Mind data confidentiality — the corporate dealbreaker.** Many companies
   forbid pasting proprietary data, customer info, or internal code into
   *consumer* AI tools. Use the **enterprise/approved** version (enterprise
   Copilot, enterprise Claude, an internal gateway) which contractually doesn't
   train on your inputs. **Ask your manager / security team what's sanctioned
   before pasting anything from the warehouse.** When in doubt, share schema and
   synthetic examples, not real rows.

5. **Read every line before you commit it.** AI loves to hallucinate plausible
   column names and subtly wrong stats formulas. Code review applies to
   AI-written code exactly as to human-written code.

6. **Use it to learn, not to skip learning.** Ask it *why* it chose a robust
   standard error, or to explain a confidence interval. The point of this
   tutorial is that you understand the methods — AI is the tutor, not the
   substitute.

---

## 7. Recommended `settings.json` for a DS workflow

`Cmd+Shift+P` → "Preferences: Open User Settings (JSON)":
```jsonc
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "charliermarsh.ruff",
  "editor.rulers": [88],                       // PEP-8-ish line length
  "python.analysis.typeCheckingMode": "basic", // catch bugs early
  "jupyter.interactiveWindow.textEditor.executeSelection": true,
  "files.trimTrailingWhitespace": true
}
```

---

## 8. Quick-start for THIS tutorial

```bash
cd DS_Python_Pricing_Tutorial
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/generate_pricing_data.py        # creates data/freight_quotes.csv
```
Then open `01_python_pandas_foundations.py`, select your interpreter, and start
running cells with `Shift+Enter`. Work through `01` → `05` in order. Try the
exercises at the bottom of each file *before* peeking at `solutions/`.

**Keyboard shortcuts worth memorizing on day one:**

| Action | Mac | Windows/Linux |
|--------|-----|---------------|
| Command Palette | `Cmd+Shift+P` | `Ctrl+Shift+P` |
| Run current cell | `Shift+Enter` | `Shift+Enter` |
| Quick file open | `Cmd+P` | `Ctrl+P` |
| Toggle terminal | `Ctrl+` ` | `Ctrl+` ` |
| Format document | `Shift+Alt+F` | `Shift+Alt+F` |
| Accept AI suggestion | `Tab` | `Tab` |
| Go to definition | `F12` | `F12` |

Happy modeling — now go to `01_python_pandas_foundations.py`.
