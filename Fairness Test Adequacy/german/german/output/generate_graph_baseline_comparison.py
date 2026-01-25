import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Load and prepare CSVs
# ----------------------------------------------------
dp_file  = "final_correlation_ALL_TESTSETS_demographic_parity.csv"
eo_file  = "final_correlation_ALL_TESTSETS_equal_oppurtunity.csv"
eod_file = "final_correlation_ALL_TESTSETS_equalized_odd.csv"

dp  = pd.read_csv(dp_file)
eo  = pd.read_csv(eo_file)
eod = pd.read_csv(eod_file)

dp  = dp[["Adequacy Criterion", "Spearman ρ"]].rename(columns={"Spearman ρ": "Spearman_ρ_DP"})
eo  = eo[["Adequacy Criterion", "Spearman ρ"]].rename(columns={"Spearman ρ": "Spearman_ρ_EO"})
eod = eod[["Adequacy Criterion", "Spearman ρ"]].rename(columns={"Spearman ρ": "Spearman_ρ_EOd"})

merged = dp.merge(eo, on="Adequacy Criterion", how="outer") \
           .merge(eod, on="Adequacy Criterion", how="outer")

preferred_order = [
    "Mutation Based",
    "Combinatorial Coverage",
    "Distributional Balance",
    "Distance to Training",
    "Individual Discrimination Index",
    "Decision Boundary Coverage",
]

order = [c for c in preferred_order if c in merged["Adequacy Criterion"].values]
merged = merged.set_index("Adequacy Criterion").loc[order].reset_index()

# ----------------------------------------------------
# 2. JSA-Compliant Figure Setup
# ----------------------------------------------------
x = np.arange(len(merged))
width = 0.25

fig_width_in = 7.0     # 178 mm → compliant with JSA two-column width
fig_height_in = 3.5    # good visual balance

fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=600)

# ----------------------------------------------------
# 3. Plot Bars
# ----------------------------------------------------
ax.bar(x - width, merged["Spearman_ρ_DP"],  width, label="Demographic Parity")
ax.bar(x,          merged["Spearman_ρ_EO"], width, label="Equal Opportunity")
ax.bar(x + width,  merged["Spearman_ρ_EOd"], width, label="Equalized Odds")

# ----------------------------------------------------
# 4. Axis Formatting
# ----------------------------------------------------
ax.set_ylabel("Spearman correlation (ρ)", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(merged["Adequacy Criterion"], rotation=45, ha="right", fontsize=7)
ax.axhline(0, color="gray", linewidth=0.8)

ax.legend(fontsize=7)
plt.tight_layout()

plt.savefig("rq2_correlation_all_metrics.png", dpi=600, bbox_inches="tight")
plt.close()

print("Saved JSA-compliant figure: rq2_correlation_all_metrics.png")
