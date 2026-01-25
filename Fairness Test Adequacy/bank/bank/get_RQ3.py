import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Configuration
# --------------------------------------------------
FILES = {
    "Demographic Parity": "master_summary_demographic_parity.csv",
    "Equal Opportunity": "master_summary_equal_oppurtunity.csv",
    "Equalized Odds": "master_summary_equalized_odd.csv",
}

OUTPUT_FILE = "rq3_mutation_vs_test_strength_all_metrics.png"

# Order of test-set strength (x-axis)
ORDER = ["40%", "60%", "80%", "Removed", "Full"]

# Plot styles (journal-safe)
STYLES = {
    "Demographic Parity": dict(marker="o", linestyle="-"),
    "Equal Opportunity": dict(marker="s", linestyle="--"),
    "Equalized Odds": dict(marker="^", linestyle=":"),
}

# --------------------------------------------------
# Helper: Map test set names to strength categories
# --------------------------------------------------
def map_test_strength(test_name: str) -> str:
    test_name = test_name.lower()

    if test_name.startswith("no_"):
        return "Removed"
    if "40" in test_name:
        return "40%"
    if "60" in test_name:
        return "60%"
    if "80" in test_name:
        return "80%"
    if test_name == "full":
        return "Full"

    return None  # ignore anything unexpected


# --------------------------------------------------
# Load and aggregate results
# --------------------------------------------------
results = {}

for metric, file in FILES.items():
    df = pd.read_csv(file)

    # Map test sets to strength categories
    df["TestStrength"] = df["TestSet"].apply(map_test_strength)
    df = df.dropna(subset=["TestStrength"])

    # Compute mean mutation score per strength
    grouped = (
        df.groupby("TestStrength")["MutationScore"]
        .mean()
        .reindex(ORDER)
    )

    # Convert to percentage
    results[metric] = grouped * 100


# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure(figsize=(8, 5))

x = np.arange(len(ORDER))

for metric, scores in results.items():
    plt.plot(
        x,
        scores.values,
        label=metric,
        linewidth=2,
        markersize=8,
        **STYLES[metric],
    )

# Axes and labels
plt.xticks(x, ORDER, fontsize=11)
plt.ylabel("Mean Mutation Score (%)", fontsize=12)
plt.xlabel("Test Set Strength", fontsize=12)

plt.ylim(0, 100)
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.legend(fontsize=10)
plt.tight_layout()

# Save
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved RQ3 figure → {OUTPUT_FILE}")
