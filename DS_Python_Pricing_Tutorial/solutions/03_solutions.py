# %% [markdown]
# # Solutions — Lesson 03 (regression)
#   python solutions/03_solutions.py
# %%
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("data/freight_quotes.csv", parse_dates=["quote_date"])
df["equipment_type"] = df["equipment_type"].str.title().str.strip()
df = df.drop_duplicates().reset_index(drop=True)
df["weight_lbs"] = df.groupby("equipment_type")["weight_lbs"].transform(
    lambda s: s.fillna(s.median())
)

# %% E1 — price by distance + segment
e1 = smf.ols("quoted_price_usd ~ distance_miles + C(customer_segment)", data=df).fit()
print("E1 coefficients:\n", e1.params)
print("Baseline segment is 'Enterprise' (alphabetical). Positive SMB/Mid-Market"
      " dummies => they're quoted higher prices at the same distance, matching"
      " the higher-margin story from lesson 01.\n")

# %% E2 — fuel pass-through
e2 = smf.ols("carrier_cost_usd ~ fuel_price_usd_gal + distance_miles", data=df).fit()
print(f"E2: cost rises ${e2.params['fuel_price_usd_gal']:.0f} per +$1 diesel"
      " (averaged across haul lengths). Note this single coefficient blends short"
      " and long hauls; the interaction model in lesson 03 sec.4 is sharper.\n")

# %% E3 — win logit with segment + odds ratio
e3 = smf.logit("won ~ realized_margin + market_tightness + C(customer_segment)",
               data=df).fit(disp=0)
print("E3 params:\n", e3.params)
odds = np.exp(e3.params["realized_margin"] * 0.01)  # per +1 percentage point
print(f"Odds ratio per +1pt margin = {odds:.3f}: each extra point of margin"
      f" multiplies the odds of winning by ~{odds:.2f} (i.e. lowers them ~{(1-odds)*100:.0f}%).\n")

# %% E4 — margin x tightness interaction
e4 = smf.logit("won ~ realized_margin * market_tightness", data=df).fit(disp=0)
print("E4 params:\n", e4.params)
print("A positive interaction would mean shippers are LESS price-sensitive when"
      " capacity is tight (fewer outside options). Check the sign of"
      " realized_margin:market_tightness.\n")

# %% E5 — optimal margin function
win_model = smf.logit("won ~ realized_margin + market_tightness", data=df).fit(disp=0)

def optimal_margin(cost, tightness):
    margins = np.linspace(0.02, 0.35, 200)
    grid = pd.DataFrame({"realized_margin": margins, "market_tightness": tightness})
    p = win_model.predict(grid).values
    price = cost * margins / (1 - margins)          # profit per won load = price-cost
    exp_profit = p * price
    return margins[np.argmax(exp_profit)]

print("E5:")
print(f"  cheap, loose market (cost=$1500, tight=0.3): {optimal_margin(1500,0.3):.1%}")
print(f"  pricey, tight market (cost=$4000, tight=1.4): {optimal_margin(4000,1.4):.1%}")
print("  Tighter markets support higher margins (higher baseline win prob).")
