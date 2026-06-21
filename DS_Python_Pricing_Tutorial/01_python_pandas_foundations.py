# %% [markdown]
# # Lesson 01 — Python & pandas Foundations for Pricing Analytics
#
# **Goal:** get fluent with the daily-driver skill of a pricing data scientist —
# loading, cleaning, slicing, grouping, and aggregating quote data.
#
# Run each cell with `Shift+Enter`. Read the comments — they explain the *why*,
# not just the *what*. Exercises are at the bottom; solutions in `solutions/`.
#
# **Business context:** every row of `freight_quotes.csv` is one price we quoted
# a shipper to move a truckload. We want to understand cost, price, margin, and
# whether we won the load.

# %%
import numpy as np
import pandas as pd

# Show more columns/rows when printing, and don't use scientific notation for $.
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

df = pd.read_csv("data/freight_quotes.csv", parse_dates=["quote_date"])
print(df.shape)          # (rows, columns)
df.head()

# %% [markdown]
# ## 1. Know your data before you touch it
# The first thing any analyst does: understand types, ranges, and missingness.
# Cheaper to find a problem here than in a model later.

# %%
df.info()          # dtypes + non-null counts -> spot missing data & wrong types

# %%
df.describe()      # summary stats for numeric columns -> spot impossible values

# %%
# Categorical columns: what are the distinct values and counts?
print(df["equipment_type"].value_counts())
print("\nSegments:\n", df["customer_segment"].value_counts())

# %% [markdown]
# ## 2. Data cleaning — the unglamorous 80% of the job
# Notice three planted problems: missing weights, messy equipment casing
# ("REEFER" vs "Reefer"), and duplicate rows. Let's fix each.

# %%
# 2a. Duplicates: the generator double-logged 50 quotes. quote_id should be unique.
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("After dedupe:", df.shape)

# %%
# 2b. Inconsistent categories: standardize casing so "REEFER" == "Reefer".
# .str accessor applies string methods element-wise to a column.
df["equipment_type"] = df["equipment_type"].str.title().str.strip()
print(df["equipment_type"].value_counts())

# %%
# 2c. Missing weights. Options: drop rows, or impute. For ~3% missing, imputing
# with the MEDIAN weight *within each equipment type* is more defensible than a
# global mean (a flatbed load weighs differently than a reefer).
print("Missing weights before:", df["weight_lbs"].isna().sum())

df["weight_lbs"] = df.groupby("equipment_type")["weight_lbs"].transform(
    lambda s: s.fillna(s.median())
)
print("Missing weights after:", df["weight_lbs"].isna().sum())

# %% [markdown]
# ## 3. Selecting & filtering — `.loc`, boolean masks
# `.loc[rows, cols]` is the explicit, recommended way to select.

# %%
# Single column -> Series; list of columns -> DataFrame
rpm = df["linehaul_rate_per_mile"]
subset = df[["lane", "quoted_price_usd", "won"]]

# Boolean filtering: high-value reefer loads we WON
big_reefer_wins = df.loc[
    (df["equipment_type"] == "Reefer")
    & (df["quoted_price_usd"] > 3000)
    & (df["won"] == 1)
]
print(f"{len(big_reefer_wins):,} big reefer wins")
big_reefer_wins[["lane", "distance_miles", "quoted_price_usd", "realized_margin"]].head()

# %% [markdown]
# ## 4. Creating columns (feature engineering)
# New, derived metrics are where pricing insight lives.

# %%
# Profit per load (only realized when we WIN — a lost quote earns $0).
df["gross_profit_usd"] = (df["quoted_price_usd"] - df["carrier_cost_usd"]) * df["won"]

# Bucket distance into haul-length bands — a very common pricing segmentation.
df["haul_type"] = pd.cut(
    df["distance_miles"],
    bins=[0, 250, 500, 1000, np.inf],
    labels=["Local (<250)", "Short (250-500)", "Mid (500-1000)", "Long (1000+)"],
)
df[["distance_miles", "haul_type", "gross_profit_usd"]].head()

# %% [markdown]
# ## 5. Group-by: the workhorse of analytics
# "Split → apply → combine." Answer business questions one aggregation at a time.

# %%
# Win rate and average margin by customer segment
seg = df.groupby("customer_segment").agg(
    quotes=("quote_id", "count"),
    win_rate=("won", "mean"),
    avg_margin=("realized_margin", "mean"),
    total_profit=("gross_profit_usd", "sum"),
).sort_values("total_profit", ascending=False)
print(seg)

# %%
# Multi-key group-by: win rate by equipment AND haul length.
# unstack() pivots the last index level into columns -> a readable matrix.
win_matrix = (
    df.groupby(["equipment_type", "haul_type"], observed=True)["won"]
    .mean()
    .unstack()
)
print(win_matrix)

# %% [markdown]
# ## 6. Time series basics — resampling
# Pricing is seasonal. Roll the data up to monthly to see trends.

# %%
monthly = (
    df.set_index("quote_date")
    .resample("MS")                       # MS = month start
    .agg(avg_rate_per_mile=("linehaul_rate_per_mile", "mean"),
         avg_fuel=("fuel_price_usd_gal", "mean"),
         win_rate=("won", "mean"))
)
print(monthly.head(12))

# %% [markdown]
# ## 7. Top-N analysis — which lanes make us money?
# A classic "where should pricing focus" question.

# %%
lane_perf = (
    df.groupby("lane")
    .agg(quotes=("quote_id", "count"),
         win_rate=("won", "mean"),
         total_profit=("gross_profit_usd", "sum"),
         avg_margin=("realized_margin", "mean"))
    .query("quotes >= 50")                 # ignore thin lanes (noisy)
    .sort_values("total_profit", ascending=False)
)
print("Top 10 lanes by profit:\n", lane_perf.head(10))

# %% [markdown]
# ---
# # EXERCISES
# Try these before checking `solutions/01_solutions.py`. Write your answer in the
# blank cell under each prompt and run it.
#
# **E1.** What is the overall company win rate, and the win rate *only* on
# Dynamic-Pricing lanes (`is_dynamic_pricing_lane == 1`)? Are they different?
#
# **E2.** Build a table of average `realized_margin` by `equipment_type`. Which
# trailer type is most profitable on a margin basis?
#
# **E3.** Create a column `is_long_haul` (1 if distance_miles > 800 else 0), then
# compare average `quoted_price_usd` for long vs short hauls.
#
# **E4.** For *won* loads only, find the 5 shippers (`shipper_id`) that generated
# the most total `gross_profit_usd`.
#
# **E5.** Compute monthly average `carrier_cost_usd` and `fuel_price_usd_gal`.
# Eyeball whether cost rises with fuel (you'll quantify this with regression in
# lesson 03).

# %%
# Your E1 answer here:


# %%
# Your E2 answer here:


# %%
# Your E3 answer here:


# %%
# Your E4 answer here:


# %%
# Your E5 answer here:
