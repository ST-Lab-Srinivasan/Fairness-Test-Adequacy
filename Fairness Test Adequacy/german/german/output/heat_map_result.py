import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_heatmap(metric_name, df, output_file):
    # =====================================================
    # Aggregate mean mutation score per (Model, Operator)
    # =====================================================
    g = df.groupby(["Model", "Operator"])["MutationScore"].mean().reset_index()

    # Convert to percentage (0–100)
    g["MutationScore"] = (g["MutationScore"] * 100).round(1)

    # Pivot → heatmap matrix
    pivot = g.pivot(index="Model", columns="Operator", values="MutationScore")

    # Sort rows and columns
    pivot = pivot.sort_index().sort_index(axis=1)

    # Replace NaN values for visual clarity
    pivot = pivot.fillna(0)

    # Matrix for plotting
    data = pivot.values

    # =====================================================
    # JSA-Compliant Figure Size and DPI
    # Width: 7.0 inches (≈178 mm) → within 80–180 mm requirement
    # Height: 4.5 inches → readable aspect ratio
    # DPI: 600 (recommended for journal print quality)
    # =====================================================
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=600)

    # Heatmap
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=100)

    # =====================================================
    # Annotate exact values inside cells
    # =====================================================
    max_val = np.nanmax(data)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = "white" if value > max_val * 0.6 else "black"
            ax.text(j, i, f"{value:.1f}",
                    ha="center", va="center",
                    fontsize=7, color=color)  # ≥6pt as required

    # =====================================================
    # Axis ticks (models and operators)
    # =====================================================
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=7, rotation=90)

    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    # Keep labels aligned tightly
    ax.set_xlim(-0.5, len(pivot.columns) - 0.5)
    ax.set_ylim(len(pivot.index) - 0.5, -0.5)

    # =====================================================
    # Colorbar
    # =====================================================
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Mutation Score (%)", fontsize=8)

    plt.tight_layout()

    # Save with bbox_inches for clean borders
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap → {output_file}")


# =====================================================
# Run for all metrics
# =====================================================
paths = {
    "Demographic Parity": "master_summary_demographic_parity.csv",
    "Equal Opportunity": "master_summary_equal_oppurtunity.csv",
    "Equalized Odds": "master_summary_equalized_odd.csv",
}

for metric, file in paths.items():
    print(f"Processing: {metric}  from  {file}")

    df = pd.read_csv(file)

    outfile = (
        "heatmap_demographic_parity_fixed.png" if metric == "Demographic Parity" else
        "heatmap_equal_opportunity_fixed.png" if metric == "Equal Opportunity" else
        "heatmap_equalized_odds_fixed.png"
    )

    create_heatmap(metric, df, outfile)
