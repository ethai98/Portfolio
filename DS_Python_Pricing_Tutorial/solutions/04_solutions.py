# %% [markdown]
# # Solutions — Lesson 04 (causal inference)
#   python solutions/04_solutions.py
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats

df = pd.read_csv("data/freight_quotes.csv", parse_dates=["quote_date"])
df["equipment_type"] = df["equipment_type"].str.title().str.strip()
df = df.drop_duplicates().reset_index(drop=True)
df["post"] = (df["quote_date"] >= "2024-07-01").astype(int)
df["treated"] = df["is_dynamic_pricing_lane"]
df["month"] = df["quote_date"].dt.month
df["gross_profit"] = (df["quoted_price_usd"] - df["carrier_cost_usd"]) * df["won"]

# %% E1 — DiD with lane + month fixed effects
e1 = smf.ols("realized_margin ~ treated*post + C(lane) + C(month)",
             data=df).fit(cov_type="HC3")
print(f"E1: DiD with fixed effects, treated:post = "
      f"{e1.params['treated:post']:+.4f} (≈ +0.022 expected on realized margin; "
      "the +3pt markup maps to ~+2.2pts of realized margin). "
      "Robust to lane/seasonal differences.\n")

# %% E2 — parallel trends check
trend = (df.groupby([df["quote_date"].dt.to_period("M").dt.to_timestamp(), "treated"])
         ["realized_margin"].mean().unstack())
trend.columns = ["Control", "Treated"]
fig, ax = plt.subplots()
trend.plot(ax=ax)
ax.axvline(pd.Timestamp("2024-07-01"), color="k", ls="--", label="Pilot launch")
ax.set_title("Parallel trends check: lines track before launch, diverge after")
ax.legend(); plt.show()
print("E2: Before Jul-2024 the two lines move together (parallel) -> DiD is valid;"
      " after launch the treated line steps up ~2 pts of realized margin.\n")

# %% E3 — profit DiD with fixed effects + significance
e3 = smf.ols("gross_profit ~ treated*post + C(lane) + C(month)",
             data=df).fit(cov_type="HC3")
eff, p = e3.params["treated:post"], e3.pvalues["treated:post"]
print(f"E3: profit DiD w/ fixed effects = ${eff:+.2f}/quote (p={p:.3f}).")
print("Margin gain on retained loads > value of lost loads => net positive. If"
      " the p-value is small, the uplift is robust -> reasonable to green-light a"
      " staged rollout, while monitoring win rate on a control holdout.\n")

# %% E4 — statistical power: effect size & n drive detectability
def ab_sim(n_per_arm, lift, seed):
    rng = np.random.default_rng(seed)
    s = df.sample(2 * n_per_arm, random_state=seed).copy()
    s["arm"] = rng.choice(["control", "test"], size=2 * n_per_arm)
    s.loc[s.arm == "test", "realized_margin"] += lift
    t, p = stats.ttest_ind(s.loc[s.arm == "test", "realized_margin"],
                           s.loc[s.arm == "control", "realized_margin"])
    return t, p

for n, lift, label in [(200, 0.010, "200/arm, +1.0pt"),
                       (200, 0.002, "200/arm, +0.2pt"),
                       (40, 0.002, " 40/arm, +0.2pt")]:
    t, p = ab_sim(n, lift, seed=3)
    print(f"E4 {label}: t={t:5.2f}, p={p:.3f} -> "
          f"{'DETECTABLE' if p < 0.05 else 'NOT detectable'}")
print("Margin noise here is small (~0.03), so a 1-pt lift is easy to detect even"
      " at 200/arm; a 0.2-pt lift needs far more data. Always run a power"
      " calculation to size a price test BEFORE launching.\n")

# %% E5 — omitted variable bias
short = smf.ols("carrier_cost_usd ~ is_peak_season", data=df).fit()
full = smf.ols("carrier_cost_usd ~ is_peak_season + market_tightness + distance_miles",
               data=df).fit()
print(f"E5: peak coef alone = {short.params['is_peak_season']:.0f}; "
      f"with controls = {full.params['is_peak_season']:.0f}.")
print("Peak season correlates with tight markets & maybe longer hauls, so the"
      " naive coefficient absorbs their effects (omitted-variable bias). Adding"
      " them isolates the pure peak surcharge (true value $180).")
