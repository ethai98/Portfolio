# %% [markdown]
# # Lesson 04 — Causal Inference for Pricing
#
# Prediction asks "what *will* happen?" Causal inference asks "what happens *if
# we change the price?*" — which is the actual job of a pricing team. You can't
# answer that with correlation alone, because price is *chosen*, not random.
#
# **The core problem (confounding):** we tend to charge higher margins on tight-
# capacity lanes where we'd win anyway. A naive "high margin → still high win
# rate" correlation would *wrongly* suggest raising prices is free. We need
# methods that isolate the causal effect.
#
# We cover: (1) why naive comparisons mislead, (2) the gold standard — A/B tests,
# (3) difference-in-differences for a real rollout, (4) controlling for
# confounders via regression. The generator baked in a real pricing trade-off:
# the Dynamic-Pricing pilot raised our **markup by 3 points** on treated lanes
# after 2024-07-01. That lifts realized margin ~2.2 pts (margin = markup/(1+markup))
# but — because charging more loses some loads — also costs a little win rate. The
# job of this lesson is to measure both and decide whether it was NET profitable.

# %%
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv("data/freight_quotes.csv", parse_dates=["quote_date"])
df["equipment_type"] = df["equipment_type"].str.title().str.strip()
df = df.drop_duplicates().reset_index(drop=True)
df["post"] = (df["quote_date"] >= "2024-07-01").astype(int)
df["treated"] = df["is_dynamic_pricing_lane"]      # lanes in the pilot

# %% [markdown]
# ## 1. The naive (wrong) comparison
# Just compare margins of treated vs untreated lanes. The problem: treated and
# control lanes may have differed *before* the pilot for other reasons. This
# number conflates the treatment effect with pre-existing differences.

# %%
naive = df.groupby("treated")["realized_margin"].mean()
print(naive)
print(f"Naive 'effect' = {naive[1] - naive[0]:+.4f}")
print("^ Don't trust this — it ignores time and pre-existing lane differences.")

# %% [markdown]
# ## 2. Why randomization is the gold standard (A/B tests)
# If we had RANDOMLY assigned which lanes get dynamic pricing, treated and
# control groups would be identical on average *except* for the treatment, so a
# simple difference in means would be causal. Most rigorous pricing decisions
# come from experiments (price tests). Quick simulated illustration:

# %%
rng = np.random.default_rng(7)
ab = df.sample(4000, random_state=7).copy()
ab["arm"] = rng.choice(["control", "test"], size=len(ab))   # random assignment
# Pretend the 'test' arm got a 2-pt margin bump:
ab.loc[ab["arm"] == "test", "realized_margin"] += 0.02
diff = (ab.loc[ab.arm == "test", "realized_margin"].mean()
        - ab.loc[ab.arm == "control", "realized_margin"].mean())
print(f"A/B estimated lift: {diff:+.4f} (we injected +0.0200)")
# A t-test tells us if it's distinguishable from noise:
from scipy import stats
t, p = stats.ttest_ind(ab.loc[ab.arm == "test", "realized_margin"],
                       ab.loc[ab.arm == "control", "realized_margin"])
print(f"t = {t:.2f}, p = {p:.4f}")

# %% [markdown]
# ## 3. Difference-in-Differences (DiD) — the workhorse for rollouts
# We rarely get a clean randomized test for an org-wide rollout. DiD recovers a
# causal effect from a staggered launch by comparing the **change** in treated
# lanes to the **change** in control lanes over the same period. The key
# assumption: *parallel trends* — absent the pilot, both groups would have moved
# together.
#
# The DiD estimate = (Treated_post − Treated_pre) − (Control_post − Control_pre).
# The "differencing" cancels out fixed lane differences AND common time shocks.

# %%
cell = df.groupby(["treated", "post"])["realized_margin"].mean().unstack()
cell.index = ["Control lanes", "Treated lanes"]
cell.columns = ["Pre (before Jul-24)", "Post"]
print(cell, "\n")

did_manual = ((cell.loc["Treated lanes", "Post"] - cell.loc["Treated lanes", "Pre (before Jul-24)"])
              - (cell.loc["Control lanes", "Post"] - cell.loc["Control lanes", "Pre (before Jul-24)"]))
print(f"Manual DiD estimate: {did_manual:+.4f}")
print("(We injected a +3pt MARKUP; on realized margin that's ~+0.022, which DiD"
      " recovers. Good reminder that 'margin' depends on how you define it.)")

# %% [markdown]
# ## 4. DiD as a regression (the standard way to run it)
# The interaction `treated:post` coefficient IS the DiD estimate — and you get a
# standard error / p-value for free. This is how it's reported in practice.

# %%
did_model = smf.ols("realized_margin ~ treated + post + treated:post",
                    data=df).fit(cov_type="HC3")
print(did_model.summary().tables[1])
print(f"\nDiD effect (treated:post) = {did_model.params['treated:post']:+.4f}")
print("Interpretation: dynamic pricing lifted realized margin ~2 pts, causally.")

# %% [markdown]
# **Reading the DiD regression:**
# - `treated` = baseline difference between pilot and control lanes (pre-period).
# - `post` = common time trend affecting everyone.
# - `treated:post` = the *causal* extra change for treated lanes = the answer.

# %% [markdown]
# ## 5. Did it cost us volume? (the critical pricing follow-up)
# A margin lift means nothing if we lose the loads. Run the SAME DiD on `won`.
# This is *the* discipline that separates a pricing scientist from someone who
# only looks at the headline margin number.

# %%
did_win = smf.ols("won ~ treated + post + treated:post", data=df).fit(cov_type="HC3")
print(did_win.summary().tables[1])
print(f"\nEffect on win rate: {did_win.params['treated:post']:+.4f}")
print("Negative & significant: charging more DID lose us a few points of win"
      " rate, exactly as price elasticity predicts. So is the pilot worth it?")

# %% [markdown]
# ## 5b. The verdict: net effect on PROFIT
# We have a classic pricing trade-off — higher margin per won load, but fewer won
# loads. Neither number alone settles it. The right metric is causal effect on
# **gross profit per quote** = (price − cost) × won. Run DiD on that.

# %%
df["gross_profit"] = (df["quoted_price_usd"] - df["carrier_cost_usd"]) * df["won"]
did_profit = smf.ols("gross_profit ~ treated + post + treated:post",
                     data=df).fit(cov_type="HC3")
print(did_profit.summary().tables[1])
eff = did_profit.params["treated:post"]
pval = did_profit.pvalues["treated:post"]
print(f"\nDiD effect on profit/quote = ${eff:+,.2f} (p = {pval:.3f})")
print("Positive: the extra margin on retained loads OUTWEIGHS the lost loads, so"
      " the pilot was net profitable. THIS is the number you take to leadership —"
      " not the margin lift in isolation. (Always check the win-rate cost too.)")

# %% [markdown]
# ## 6. Controlling for confounders (selection-on-observables)
# When you have neither an experiment nor a clean before/after, the fallback is
# to *adjust* for the confounders in a regression. Recall the naive story: margin
# and win rate are positively correlated in raw data because we charge more on
# tight markets where we'd win anyway. Watch the sign of `realized_margin` FLIP
# once we control for `market_tightness`.

# %%
naive_win = smf.logit("won ~ realized_margin", data=df).fit(disp=0)
ctrl_win = smf.logit("won ~ realized_margin + market_tightness",
                     data=df).fit(disp=0)
print(f"Margin coef, NO controls:   {naive_win.params['realized_margin']:+.2f}")
print(f"Margin coef, WITH tightness:{ctrl_win.params['realized_margin']:+.2f}")
print("\nControlling for the confounder reveals the true negative price effect.")
print("This is 'omitted variable bias' in action — the heart of causal thinking.")

# %% [markdown]
# **Caveat:** controlling-for-observables only works if you've measured the
# confounders. Unobserved confounders (a salesperson's relationship, say) still
# bias you — which is why experiments and DiD are preferred when feasible. Always
# ask: "what else could explain this besides the price change?"
#
# ---
# # EXERCISES
# Solutions in `solutions/04_solutions.py`.
#
# **E1.** Run the DiD regression on `realized_margin` but add lane and
# month fixed effects: `... + C(lane) + C(quote_date.dt.month)`. (Hint: create a
# `month` column first.) Does the `treated:post` estimate stay ~+0.03? Fixed
# effects are how teams make DiD robust.
#
# **E2.** Check the **parallel-trends assumption**: compute monthly average
# `realized_margin` for treated vs control lanes and plot both lines. Before the
# July-2024 launch, do they move in parallel? (If not, DiD is suspect.)
#
# **E3.** Re-run the profit DiD from section 5b but add lane + month fixed
# effects, and look at the t-stat / p-value on `treated:post`. Is the +$/quote
# uplift statistically robust? Given the margin gain, win-rate loss, and profit
# result together, would you green-light the company-wide rollout?
#
# **E4.** Power matters. Simulate an A/B test with 200 quotes/arm and a true 1-pt
# margin lift, then repeat with a tiny 0.2-pt lift (and/or 40 quotes/arm). Which
# are detectable (p < 0.05)? Lesson: detectability depends on effect size
# relative to noise AND sample size — size your price tests before launching.
#
# **E5.** Demonstrate omitted-variable bias yourself: regress `carrier_cost_usd`
# on `is_peak_season` alone, then add `market_tightness` and `distance_miles`.
# Explain why the peak-season coefficient changes.

# %%
# Your answers here:
