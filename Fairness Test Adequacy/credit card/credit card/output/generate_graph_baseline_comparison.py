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

# ----------------------------------------------------
# 2. Keep required columns and rename
# ----------------------------------------------------
dp = dp[["Adequacy Criterion", "Spearman ρ"]] \
        .rename(columns={"Spearman ρ": "Spearman_ρ_DP"})

eo = eo[["Adequacy Criterion", "Spearman ρ"]] \
        .rename(columns={"Spearman ρ": "Spearman_ρ_EO"})

eod = eod[["Adequacy Criterion", "Spearman ρ"]] \
        .rename(columns={"Spearman ρ": "Spearman_ρ_EOd"})

# ----------------------------------------------------
# 3. Merge into a single table
# ----------------------------------------------------
merged = (
    dp.merge(eo, on="Adequacy Criterion", how="outer")
      .merge(eod, on="Adequacy Criterion", how="outer")
)

# ----------------------------------------------------
# 4. Order adequacy criteria
# ----------------------------------------------------
preferred_order = [
    "Mutation Based",
    "Combinatorial Coverage",
    "Distributional Balance",
    "Distance to Training",
    "Individual Discrimination Index",
    "Decision Boundary Coverage",
]

order = [c for c in preferred_order if c in merged["Adequacy Criterion"].values]

merged = (
    merged.set_index("Adequacy Criterion")
          .loc[order]
          .reset_index()
)

# ----------------------------------------------------
# 5. Build grouped bar chart (journal-safe)
# ----------------------------------------------------
x = np.arange(len(merged))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    x - width,
    merged["Spearman_ρ_DP"],
    width,
    label="Demographic Parity"
)

ax.bar(
    x,
    merged["Spearman_ρ_EO"],
    width,
    label="Equal Opportunity"
)

ax.bar(
    x + width,
    merged["Spearman_ρ_EOd"],
    width,
    label="Equalized Odds"
)

# ----------------------------------------------------
# 6. Axis formatting (consistent styling)
# ----------------------------------------------------
ax.set_ylabel(
    "Spearman correlation (ρ) with fairness faults",
    fontsize=12
)

ax.set_xticks(x)
ax.set_xticklabels(
    merged["Adequacy Criterion"],
    rotation=40,
    ha="right",
    fontsize=11
)

ax.set_ylim(0.0, 0.85)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8,0.9, 1.0])
ax.tick_params(axis="y", labelsize=11)

ax.axhline(0, color="gray", linewidth=0.8)
ax.grid(axis="y", linestyle="--", alpha=0.4)

ax.legend(fontsize=10)

plt.tight_layout()

# ----------------------------------------------------
# 7. Save
# ----------------------------------------------------
out_file = "rq2_correlation_all_metrics.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved figure: {out_file}")
