import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------
FILES = {
    "Demographic Parity": "master_summary_demographic_parity.csv",
    "Equal Opportunity": "master_summary_equal_oppurtunity.csv",
    "Equalized Odds": "master_summary_equalized_odd.csv",
}

ORDER = ["40%", "60%", "80%", "Removed", "Full"]

OUTPUT_CSV = "rq3_mutation_vs_test_strength_all_metrics.csv"

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

    return None  # ignore unexpected cases


# --------------------------------------------------
# Load, aggregate, and store raw values
# --------------------------------------------------
tables = []

for metric, file in FILES.items():
    df = pd.read_csv(file)

    df["TestStrength"] = df["TestSet"].apply(map_test_strength)
    df = df.dropna(subset=["TestStrength"])

    grouped = (
        df.groupby("TestStrength")["MutationScore"]
        .mean()
        .reindex(ORDER)
        * 100  # convert to percentage
    )

    metric_df = grouped.reset_index()
    metric_df["Metric"] = metric
    metric_df.rename(columns={"MutationScore": "MeanMutationScore"}, inplace=True)

    tables.append(metric_df)

# Combine all metrics into one table
final_df = pd.concat(tables, ignore_index=True)

# Pivot for wide-format table (paper-friendly)
pivot_df = final_df.pivot(
    index="TestStrength",
    columns="Metric",
    values="MeanMutationScore"
).reindex(ORDER)

# --------------------------------------------------
# Output
# --------------------------------------------------
pivot_df.to_csv(OUTPUT_CSV, float_format="%.2f")

print("\nRQ3: Mean Mutation Score (%) vs Test Set Strength\n")
print(pivot_df.round(2))
print(f"\nSaved raw RQ3 table → {OUTPUT_CSV}")
