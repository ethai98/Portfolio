# %% [markdown]
# # Solutions — Lesson 01 (pandas foundations)
# Run from the tutorial root so the data path resolves:
#   python solutions/01_solutions.py
# %%
import numpy as np
import pandas as pd

df = pd.read_csv("data/freight_quotes.csv", parse_dates=["quote_date"])
df["equipment_type"] = df["equipment_type"].str.title().str.strip()
df = df.drop_duplicates().reset_index(drop=True)
df["weight_lbs"] = df.groupby("equipment_type")["weight_lbs"].transform(
    lambda s: s.fillna(s.median())
)
df["gross_profit_usd"] = (df["quoted_price_usd"] - df["carrier_cost_usd"]) * df["won"]

# %% E1 — overall vs dynamic-pricing-lane win rate
overall = df["won"].mean()
by_lane = df.groupby("is_dynamic_pricing_lane")["won"].mean()
print(f"E1: overall win rate = {overall:.1%}")
print(by_lane)
print("They're close — the pilot lifted MARGIN, not win rate (see lesson 04).\n")

# %% E2 — average realized margin by equipment type
e2 = df.groupby("equipment_type")["realized_margin"].mean().sort_values(ascending=False)
print("E2: avg realized margin by equipment:\n", e2)
print("Margin is driven by customer segment mix, not trailer type, so these are"
      " similar; the difference is small.\n")

# %% E3 — long vs short haul average quoted price
df["is_long_haul"] = (df["distance_miles"] > 800).astype(int)
e3 = df.groupby("is_long_haul")["quoted_price_usd"].mean()
print("E3: avg quoted price by haul length (0=short, 1=long):\n", e3, "\n")

# %% E4 — top 5 shippers by total gross profit (won loads only)
e4 = (df[df["won"] == 1]
      .groupby("shipper_id")["gross_profit_usd"].sum()
      .sort_values(ascending=False).head(5))
print("E4: top 5 shippers by profit:\n", e4, "\n")

# %% E5 — monthly cost vs fuel
e5 = (df.set_index("quote_date").resample("MS")
      .agg(avg_cost=("carrier_cost_usd", "mean"),
           avg_fuel=("fuel_price_usd_gal", "mean")))
print("E5: monthly cost & fuel (corr = "
      f"{e5['avg_cost'].corr(e5['avg_fuel']):.2f}):")
print(e5.head(12))
print("Positive correlation -> cost rises with fuel, as expected.")
