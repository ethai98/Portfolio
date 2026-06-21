# %% [markdown]
# # Solutions — Lesson 02 (visualization)
#   python solutions/02_solutions.py
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
df = pd.read_csv("data/freight_quotes.csv", parse_dates=["quote_date"])
df["equipment_type"] = df["equipment_type"].str.title().str.strip()
df = df.drop_duplicates().reset_index(drop=True)
df["gross_profit_usd"] = (df["quoted_price_usd"] - df["carrier_cost_usd"]) * df["won"]

# %% E1 — quoted price histogram, linear vs log x
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["quoted_price_usd"], bins=50, ax=axes[0])
axes[0].set_title("Quoted price (linear x) — right-skewed")
sns.histplot(df["quoted_price_usd"], bins=50, ax=axes[1])
axes[1].set_xscale("log")
axes[1].set_title("Quoted price (log x) — more symmetric")
plt.tight_layout(); plt.show()

# %% E2 — rate per mile by haul band
df["haul_type"] = pd.cut(df["distance_miles"], [0, 250, 500, 1000, np.inf],
                         labels=["Local", "Short", "Mid", "Long"])
fig, ax = plt.subplots()
sns.boxplot(data=df, x="haul_type", y="linehaul_rate_per_mile", ax=ax)
ax.set_title("Shorter hauls cost MORE per mile (fixed costs spread over fewer miles)")
plt.show()

# %% E3 — price-response curve by segment
df["margin_bin"] = pd.cut(df["realized_margin"], bins=np.arange(0, 0.31, 0.02))
curve = (df.groupby(["margin_bin", "customer_segment"], observed=True)["won"]
         .mean().reset_index())
curve["mid"] = curve["margin_bin"].apply(lambda b: b.mid).astype(float)
fig, ax = plt.subplots()
sns.lineplot(data=curve, x="mid", y="won", hue="customer_segment", marker="o", ax=ax)
ax.set_title("Win rate vs margin by segment (steeper = more price-sensitive)")
ax.set_xlabel("Realized margin"); ax.set_ylabel("Win rate")
plt.show()

# %% E4 — monthly market tightness
monthly_t = df.set_index("quote_date")["market_tightness"].resample("MS").mean()
fig, ax = plt.subplots()
monthly_t.plot(marker="o", ax=ax)
ax.set_title("Market tightness over time (peaks ≈ produce/holiday season)")
plt.show()

# %% E5 — total win-adjusted profit by equipment, labeled bars
e5 = df.groupby("equipment_type")["gross_profit_usd"].sum().sort_values(ascending=False)
fig, ax = plt.subplots()
bars = ax.bar(e5.index, e5.values, color="steelblue")
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height(),
            f"${b.get_height()/1e6:.1f}M", ha="center", va="bottom", fontweight="bold")
ax.set_title("Total Gross Profit by Equipment Type")
plt.tight_layout(); plt.show()
