import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Load the three correlation files
# ----------------------------------------------------
dp_file  = "final_correlation_results_demographic_parity.csv"
eo_file  = "final_correlation_results_equal_oppurtunity.csv"
eod_file = "final_correlation_results_equalized_odd.csv"

dp  = pd.read_csv(dp_file)
eo  = pd.read_csv(eo_file)
eod = pd.read_csv(eod_file)

# Keep only the columns we need
dp  = dp[["Adequacy Criterion", "Spearman ρ"]].rename(columns={"Spearman ρ": "Spearman_ρ_DP"})
eo  = eo[["Adequacy Criterion", "Spearman ρ"]].rename(columns={"Spearman ρ": "Spearman_ρ_EO"})
eod = eod[["Adequacy Criterion", "Spearman ρ"]].rename(columns={"Spearman ρ": "Spearman_ρ_EOd"})

# ----------------------------------------------------
# 2. Merge into a single table
# ----------------------------------------------------
merged = dp.merge(eo, on="Adequacy Criterion", how="outer") \
           .merge(eod, on="Adequacy Criterion", how="outer")

# Optional: define a nicer order (Mutation Score first)
preferred_order = [
    "Mutation Based",
    "Combinatorial Coverage",
 
    "Distributional Balance",
    "Distance to Training",
   
    "Individual Discrimination Index",
    "Decision Boundary Coverage",
]

# Keep only criteria that exist and preserve this order
order = [c for c in preferred_order if c in merged["Adequacy Criterion"].values]
merged = merged.set_index("Adequacy Criterion").loc[order].reset_index()

# ----------------------------------------------------
# 3. Build the grouped bar chart
# ----------------------------------------------------
x = np.arange(len(merged))  # positions for each adequacy criterion
width = 0.25                # bar width

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(x - width, merged["Spearman_ρ_DP"],  width, label="Demographic Parity")
ax.bar(x,          merged["Spearman_ρ_EO"], width, label="Equal Opportunity")
ax.bar(x + width,  merged["Spearman_ρ_EOd"], width, label="Equalized Odds")

# Axes labels and formatting
ax.set_ylabel("Spearman correlation (ρ) with fairness faults")
ax.set_xticks(x)
ax.set_xticklabels(merged["Adequacy Criterion"], rotation=45, ha="right")
ax.axhline(0, color="gray", linewidth=0.8)

ax.legend()
plt.tight_layout()

out_file = "rq2_correlation_all_metrics.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved figure: {out_file}")
