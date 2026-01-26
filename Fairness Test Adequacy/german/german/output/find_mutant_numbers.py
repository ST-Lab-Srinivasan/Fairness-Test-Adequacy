import pandas as pd

def count_mutants_per_operator(csv_path,
                               operator_col="Operator",
                               mutant_col=None):
    """
    Counts number of mutants generated per operator from a CSV file.

    If mutant_col is provided, counts UNIQUE mutants per operator.
    Otherwise, counts rows per operator.
    """

    df = pd.read_csv(csv_path)

    if operator_col not in df.columns:
        raise ValueError(f"Column '{operator_col}' not found in CSV file")

    if mutant_col and mutant_col not in df.columns:
        raise ValueError(f"Column '{mutant_col}' not found in CSV file")

    if mutant_col:
        # Count unique mutants per operator
        counts = (
            df[[operator_col, mutant_col]]
            .drop_duplicates()
            .groupby(operator_col)
            .size()
            .sort_values(ascending=False)
        )
    else:
        # Each row corresponds to one mutant
        counts = (
            df.groupby(operator_col)
            .size()
            .sort_values(ascending=False)
        )

    print("\nMutants per operator:")
    print(counts)

    print(f"\nTotal operators: {counts.shape[0]}")
    print(f"Total mutants: {counts.sum()}")

    return counts


if __name__ == "__main__":
    csv_file = "master_summary_demographic_parity.csv"  # <-- update path

    # Case 1: one row = one mutant
    count_mutants_per_operator(
        csv_path=csv_file,
        operator_col="Operator"
    )

    # Case 2: if mutants repeat and you want unique counts
    # count_mutants_per_operator(
    #     csv_path=csv_file,
    #     operator_col="Operator",
    #     mutant_col="mutant"
    # )
