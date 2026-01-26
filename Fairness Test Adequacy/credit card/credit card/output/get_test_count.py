import pandas as pd
import re

def analyze_test_sets(csv_path, test_set_col="TestSet"):
    df = pd.read_csv(csv_path)

    if test_set_col not in df.columns:
        raise ValueError(f"Column '{test_set_col}' not found in CSV")

    test_sets = df[test_set_col].astype(str).unique()

    counts = {
        "full": 0,
        "stratified_40": 0,
        "stratified_60": 0,
        "stratified_80": 0,
        "group_removed": 0,
        "intersection_group_removed": 0  # expected to be 0
    }

    suspicious_intersectional = []

    for ts in test_sets:
        ts_lower = ts.lower()

        # Full test set
        if ts_lower == "full":
            counts["full"] += 1

        # Stratified test sets
        elif ts_lower == "stratified_40":
            counts["stratified_40"] += 1
        elif ts_lower == "stratified_60":
            counts["stratified_60"] += 1
        elif ts_lower == "stratified_80":
            counts["stratified_80"] += 1

        # Group-removed test sets (single attribute only)
        elif ts_lower.startswith("no_"):
            num_equals = ts_lower.count("=")

            if num_equals == 1:
                counts["group_removed"] += 1
            else:
                # This should NOT happen in your pipeline
                counts["intersection_group_removed"] += 1
                suspicious_intersectional.append(ts)

    return counts, suspicious_intersectional


if __name__ == "__main__":
    csv_file = "master_summary_equal_oppurtunity.csv"  # update path if needed
    counts, suspicious = analyze_test_sets(csv_file)

    print("Test Set Counts:")
    for k, v in counts.items():
        print(f"{k}: {v}")

    if suspicious:
        print("\n⚠ WARNING: Unexpected intersectional-looking test sets detected:")
        for s in suspicious:
            print(f"  - {s}")
    else:
        print("\n✓ No intersectional group-removed test sets detected (as expected).")
