# %% [markdown]
# # Lesson 02 — Data Visualization for Pricing
#
# Plots are how you (a) explore data and (b) persuade stakeholders. A pricing
# director won't read your code — they'll look at one chart. This lesson covers
# `matplotlib` (the engine) and `seaborn` (high-level statistical plots).
#
# **Rule of thumb:** seaborn for fast statistical exploration; matplotlib when
# you need fine control for a polished, presentation-ready figure.
#
# Run cells with `Shift+Enter`. Figures render in the interactive window.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")          # clean default styling
plt.rcParams["figure.figsize"] = (8, 5)

df = pd.read_csv("data/freight_quotes.csv", parse_dates=["quote_date"])
df["equipment_type"] = df["equipment_type"].str.title().str.strip()
df = df.drop_duplicates()

# %% [markdown]
# ## 1. Distributions — histogram & KDE
# Always look at the shape of your key variables first. Is rate-per-mile skewed?
# Are there outliers that will wreck a regression?

# %%
fig, ax = plt.subplots()
sns.histplot(df["linehaul_rate_per_mile"], bins=50, kde=True, ax=ax)
ax.set_title("Distribution of Linehaul Rate per Mile")
ax.set_xlabel("$ / mile")
plt.show()

# %% [markdown]
# ## 2. Comparing groups — boxplot & violin
# How does margin differ across customer segments? Boxplots show median, spread,
# and outliers at a glance.

# %%
fig, ax = plt.subplots()
sns.boxplot(data=df, x="customer_segment", y="realized_margin",
            order=["Enterprise", "Mid-Market", "SMB"], ax=ax)
ax.set_title("Realized Margin by Customer Segment")
ax.set_ylabel("Realized margin")
plt.show()

# %% [markdown]
# ## 3. Relationships — scatter with a trend
# The core pricing relationship: longer hauls cost more. A regression line
# (`regplot`) previews lesson 03. We sample 2,000 points so the plot isn't a blob.

# %%
sample = df.sample(2000, random_state=0)
fig, ax = plt.subplots()
sns.regplot(data=sample, x="distance_miles", y="carrier_cost_usd",
            scatter_kws={"alpha": 0.2, "s": 12}, line_kws={"color": "red"}, ax=ax)
ax.set_title("Carrier Cost vs Distance (with OLS fit)")
plt.show()

# %% [markdown]
# ## 4. The elasticity picture — does a higher price lose the load?
# Bin realized margin and plot the win rate per bin. This is the single most
# important chart for a pricing team: the **price–response curve**.

# %%
df["margin_bin"] = pd.cut(df["realized_margin"], bins=np.arange(0, 0.31, 0.02))
curve = df.groupby("margin_bin", observed=True)["won"].mean()

fig, ax = plt.subplots()
curve.plot(marker="o", ax=ax)
ax.set_title("Win Rate Falls as We Raise Margin (Price Elasticity)")
ax.set_xlabel("Realized margin bucket")
ax.set_ylabel("Win rate")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Time series — trends over the 2-year window
# Plot monthly rate-per-mile against fuel to show co-movement.

# %%
monthly = (
    df.set_index("quote_date")
    .resample("MS")
    .agg(rate=("linehaul_rate_per_mile", "mean"),
         fuel=("fuel_price_usd_gal", "mean"))
)

fig, ax1 = plt.subplots()
ax1.plot(monthly.index, monthly["rate"], color="navy", label="Rate/mile")
ax1.set_ylabel("Rate per mile ($)", color="navy")

ax2 = ax1.twinx()                          # second y-axis sharing the x-axis
ax2.plot(monthly.index, monthly["fuel"], color="darkorange", label="Diesel $/gal")
ax2.set_ylabel("Diesel ($/gal)", color="darkorange")
ax1.set_title("Rate per Mile and Diesel Price Move Together")
plt.show()

# %% [markdown]
# ## 6. Correlation heatmap — what moves with what
# A fast scan for multicollinearity and candidate predictors before modeling.

# %%
num_cols = ["distance_miles", "weight_lbs", "fuel_price_usd_gal",
            "market_tightness", "carrier_cost_usd", "quoted_price_usd",
            "realized_margin", "won"]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Matrix")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Small multiples — `FacetGrid` / `relplot`
# One chart per equipment type so trends aren't muddled together.

# %%
g = sns.relplot(
    data=df.sample(3000, random_state=1),
    x="distance_miles", y="quoted_price_usd",
    col="equipment_type", hue="won",
    alpha=0.4, height=4, palette={0: "red", 1: "green"},
)
g.fig.suptitle("Quoted Price vs Distance by Equipment (won=green)", y=1.03)
plt.show()

# %% [markdown]
# ## 8. Making a chart presentation-ready
# The difference between an exploratory plot and one you put in a deck: title,
# labeled axes with units, no chartjunk, a clear takeaway annotation.

# %%
seg_win = df.groupby("customer_segment")["won"].mean().reindex(
    ["Enterprise", "Mid-Market", "SMB"]
)
fig, ax = plt.subplots()
bars = ax.bar(seg_win.index, seg_win.values,
              color=["#1f77b4", "#ff7f0e", "#2ca02c"])
ax.set_ylim(0, 1)
ax.set_ylabel("Win rate")
ax.set_title("SMB Quotes Win Most Often — but Recall They Carry Higher Margin")
for bar in bars:                           # annotate each bar with its value
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{bar.get_height():.0%}", ha="center", fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# # EXERCISES
# Solutions in `solutions/02_solutions.py`.
#
# **E1.** Plot a histogram of `quoted_price_usd`. Is it skewed? Try a log scale
# on the x-axis (`ax.set_xscale("log")`) and compare.
#
# **E2.** Make a boxplot of `linehaul_rate_per_mile` by `haul_type` band (create
# the band with `pd.cut` like in lesson 01). Do shorter hauls cost more per mile?
#
# **E3.** Recreate the price–response curve (section 4) but make ONE line per
# `customer_segment` on the same axes (hint: loop over segments, or use seaborn
# `lineplot` with `hue`). Which segment is most price-sensitive?
#
# **E4.** Plot monthly average `market_tightness` over time. When is capacity
# tightest (highest)? Does that line up with peak season months?
#
# **E5.** Build a bar chart of total `gross_profit` (win-adjusted) by
# `equipment_type`, sorted descending, with value labels on each bar.

# %%
# Your answers here:
