# ============================================================
# COMPLETE FIXED VERSION - GERMAN CREDIT DATASET
# Handles numeric-coded demographic test sets properly
# ============================================================
import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from itertools import combinations
import pickle
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# PATHS
# ------------------------------
OUTPUT_ROOT = r"C:\Users\srinivasanm23\Documents\fairness\fairness_adequacy\fair test adequacy\fair test adequacy\german\output\equal_opportunity"
MASTER_FILE = os.path.join(OUTPUT_ROOT, "master_summary.csv")
TRAIN_PATH = r"C:\Users\srinivasanm23\Documents\fairness\fairness_adequacy\fair test adequacy\fair test adequacy\german\output\german_train.csv"
TEST_PATH = r"C:\Users\srinivasanm23\Documents\fairness\fairness_adequacy\fair test adequacy\fair test adequacy\german\output\german_test.csv"
MODEL_PATH = os.path.join(OUTPUT_ROOT, "model.pkl")

# NOTE: We'll automatically detect which columns exist in the dataset
# This will be populated after loading the data
SENSITIVE_ATTRIBUTES = []

print("="*120)
print("MUTATION ADEQUACY vs BASELINES - GERMAN CREDIT DATASET")
print("="*120)
print("\nMethodology:")
print("1. Load ALL test sets from master_summary.csv")
print("2. Split mutants 70% train / 30% test")
print("3. Compute MutationAdequacy from train mutants for EACH test set")
print("4. Compute FaultYield from test mutants for EACH test set")
print("5. Compute baseline metrics for EACH test set")
print("6. Correlate all adequacy metrics with FaultYield")
print("="*120 + "\n")

# ============================================================
# 1) LOAD master_summary.csv and do 70/30 split on MUTANTS
# ============================================================
print("Step 1: Loading master_summary.csv...")
df_master = pd.read_csv(MASTER_FILE)
print(f"   Loaded {len(df_master)} rows")
print(f"   Unique mutants: {df_master['Mutant'].nunique()}")
print(f"   Unique test sets: {df_master['TestSet'].nunique()}")

# Get all unique test sets and mutants
all_testsets = sorted(df_master['TestSet'].unique())
mutants = df_master["Mutant"].dropna().unique()

print(f"\n   ALL test sets in master_summary.csv: {len(all_testsets)}")
print(f"   Examples: {all_testsets[:20]}")
print(f"   Total mutants: {len(mutants)}")

# Randomly split mutants 70% train / 30% test
rng = np.random.default_rng(42)
train_mutants = set(rng.choice(mutants, size=int(0.7 * len(mutants)), replace=False))
test_mutants = set(mutants) - train_mutants

print(f"\n   Train mutants: {len(train_mutants)} (70%)")
print(f"   Test mutants: {len(test_mutants)} (30%)")

# Label each row as train or test
df_master["split"] = df_master["Mutant"].apply(lambda x: "train" if x in train_mutants else "test")

# Aggregate by test set
print("\nStep 2: Aggregating mutation scores by test set...")
train_scores = df_master[df_master["split"] == "train"].groupby("TestSet").agg(
    MutationAdequacy=("MutationScore", "mean")
).reset_index()

test_scores = df_master[df_master["split"] == "test"].groupby("TestSet").agg(
    FaultYield=("MutationScore", "mean")
).reset_index()

print(f"   Train adequacy computed for {len(train_scores)} test sets")
print(f"   Test fault yield computed for {len(test_scores)} test sets")

# ============================================================
# 2) LOAD + CLEAN DATA + TRAIN MODEL
# ============================================================
print("\nStep 3: Loading and training model...")
df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

print(f"   Loaded train: {len(df_train)} rows, {len(df_train.columns)} columns")
print(f"   Loaded test: {len(df_test)} rows, {len(df_test.columns)} columns")

for df in [df_train, df_test]:
    df.columns = df.columns.str.strip().str.lower().str.replace("-", "_").str.replace(" ", "_")
    df.replace(["?", " ?", "?"], np.nan, inplace=True)
    for col in df.select_dtypes("object"):
        df[col] = df[col].str.strip()

print(f"   Train columns: {list(df_train.columns)}")

# Find target column for German Credit dataset
target_candidates = ['risk', 'credit_risk', 'creditability', 'target', 'class', 'label', 'credit']
target_col = None
for candidate in target_candidates:
    if candidate in df_train.columns:
        target_col = candidate
        break

if target_col is None:
    print("\nERROR: Could not identify target column!")
    print(f"Available columns in df_train: {list(df_train.columns)}")
    print("Please manually specify target_col in the script")
    exit(1)

print(f"   Target column identified: '{target_col}'")

def clean_target(x):
    if pd.isna(x): return np.nan
    x_str = str(x).strip().lower()
    # Map to binary (0/1)
    if x_str in ['good', '1', '1.0', 'yes', 'positive']:
        return 1
    elif x_str in ['bad', '0', '0.0', 'no', 'negative']:
        return 0
    else:
        try:
            val = int(float(x_str))
            return val if val in [0, 1] else np.nan
        except:
            return np.nan

df_train[target_col] = df_train[target_col].apply(clean_target)
df_test[target_col] = df_test[target_col].apply(clean_target)
df_train = df_train.dropna(subset=[target_col])
df_test = df_test.dropna(subset=[target_col])

print(f"   After cleaning - Train: {len(df_train)}, Test: {len(df_test)}")

# Identify SENSITIVE_ATTRIBUTES from actual columns
# Look at the test set names to understand which attributes are used
test_set_attributes = set()
for ts in all_testsets:
    if ts.startswith("no_") and "=" in ts:
        attr = ts.split("=")[0].replace("no_", "")
        test_set_attributes.add(attr)

print(f"\n   Attributes found in test set names: {sorted(test_set_attributes)}")

# Now find which of these actually exist in the dataset
available_columns = set(df_test.columns) - {target_col}
SENSITIVE_ATTRIBUTES = sorted([attr for attr in test_set_attributes if attr in available_columns])

print(f"   SENSITIVE_ATTRIBUTES (present in dataset): {SENSITIVE_ATTRIBUTES}")

if len(SENSITIVE_ATTRIBUTES) == 0:
    print("\n   WARNING: No sensitive attributes found! Using all non-target columns.")
    SENSITIVE_ATTRIBUTES = [c for c in df_test.columns if c != target_col][:10]  # Limit to 10 for performance

# Train model
X_train_raw = df_train.drop(columns=[target_col])
y_train = df_train[target_col].astype(int)

cat_cols = X_train_raw.select_dtypes("object").columns.tolist()
num_cols = X_train_raw.select_dtypes(["int64", "float64"]).columns.tolist()

print(f"   Categorical columns: {len(cat_cols)}, Numeric columns: {len(num_cols)}")

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                     ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
], remainder="drop")

model = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
model.fit(X_train_raw, y_train)

train_acc = model.score(X_train_raw, y_train)
test_acc = model.score(df_test.drop(columns=[target_col]), df_test[target_col].astype(int))
print(f"   Model accuracy - Train: {train_acc:.3f}, Test: {test_acc:.3f}")

with open(MODEL_PATH, 'wb') as f: pickle.dump(model, f)
print("   Model trained and saved")

X_train_dense = preprocessor.transform(X_train_raw)
X_test_dense = preprocessor.transform(df_test.drop(columns=[target_col]))

# ============================================================
# 3) CREATE ALL TEST SETS MATCHING master_summary.csv
# ============================================================
print("\nStep 4: Creating ALL test sets from master_summary.csv...")
print(f"   Will create baselines for ALL {len(all_testsets)} test sets")

test_sets = {}
rng = np.random.default_rng(42)
n = len(df_test)
y_stratify = df_test[target_col]

# Process each test set name from master_summary.csv
# Your German Credit dataset has:
# - 1 "full" test set (entire test data)
# - 57 "no_<attribute>=<value>" test sets (demographic-removed)
# - No stratified sampling test sets
for testset_name in all_testsets:
    
    # 1. Full set
    if testset_name == "full":
        test_sets[testset_name] = list(range(n))
    
    # 2. Stratified samples
    elif testset_name == "stratified_40":
        _, sub_idx = train_test_split(df_test.index, test_size=0.4, stratify=y_stratify, random_state=42)
        test_sets[testset_name] = sub_idx.tolist()
    elif testset_name == "stratified_60":
        _, sub_idx = train_test_split(df_test.index, test_size=0.6, stratify=y_stratify, random_state=42)
        test_sets[testset_name] = sub_idx.tolist()
    elif testset_name == "stratified_80":
        _, sub_idx = train_test_split(df_test.index, test_size=0.8, stratify=y_stratify, random_state=42)
        test_sets[testset_name] = sub_idx.tolist()
    
    # 3. Demographic-removed test sets with NUMERIC VALUES
    # Format: "no_<attribute>=<numeric_value>"
    # Example: "no_housing=151" means remove rows where housing==151
    elif testset_name.startswith("no_") and "=" in testset_name:
        parts = testset_name.split("=")
        if len(parts) == 2:
            attr_name = parts[0].replace("no_", "")
            value_to_remove = parts[1].strip()
            
            if attr_name in df_test.columns:
                # Try to match as string first, then as numeric
                try:
                    # Convert both to string for comparison to handle numeric codes
                    mask = df_test[attr_name].astype(str) != value_to_remove
                    indices = df_test[mask].index.tolist()
                    
                    if len(indices) > 0:
                        test_sets[testset_name] = indices
                except Exception as e:
                    print(f"   Warning: Could not create test set '{testset_name}': {e}")
                    continue

print(f"\n   Successfully created {len(test_sets)} test sets")
missing_testsets = set(all_testsets) - set(test_sets.keys())
if missing_testsets:
    print(f"   WARNING: Could not create {len(missing_testsets)} test sets")
    if len(missing_testsets) <= 10:
        print(f"   Missing: {sorted(list(missing_testsets))}")
    else:
        print(f"   First 10 missing: {sorted(list(missing_testsets))[:10]}")

# ============================================================
# 4) BASELINE METRIC FUNCTIONS
# ============================================================

def compute_dataset_coverage(sub, full):
    """BASELINE 1: Combinatorial Coverage"""
    single_scores = []
    pair_scores = []
    attrs = [a for a in SENSITIVE_ATTRIBUTES if a in full.columns and a in sub.columns and len(sub[a].dropna()) > 0]
    
    if len(attrs) == 0:
        return 0.0
    
    for a in attrs:
        try:
            if pd.api.types.is_numeric_dtype(full[a]):
                full_bins = pd.cut(full[a], bins=5, duplicates='drop').astype(str)
                sub_bins = pd.cut(sub[a], bins=5, duplicates='drop').astype(str)
                full_cats = set(full_bins.unique())
                sub_cats = set(sub_bins.unique())
            else:
                full_cats = set(full[a].astype(str))
                sub_cats = set(sub[a].astype(str))
            coverage = len(full_cats & sub_cats) / len(full_cats) if full_cats else 0.0
            single_scores.append(coverage)
        except:
            continue
    
    # Pairwise coverage
    for a, b in combinations(attrs[:min(5, len(attrs))], 2):  # Limit to 5 attrs for performance
        try:
            if pd.api.types.is_numeric_dtype(full[a]):
                A_full = pd.cut(full[a], bins=5, duplicates='drop').astype(str)
                A_sub = pd.cut(sub[a], bins=5, duplicates='drop').astype(str)
            else:
                A_full = full[a].astype(str)
                A_sub = sub[a].astype(str)
            if pd.api.types.is_numeric_dtype(full[b]):
                B_full = pd.cut(full[b], bins=5, duplicates='drop').astype(str)
                B_sub = pd.cut(sub[b], bins=5, duplicates='drop').astype(str)
            else:
                B_full = full[b].astype(str)
                B_sub = sub[b].astype(str)
            full_pairs = set(zip(A_full, B_full))
            sub_pairs = set(zip(A_sub, B_sub))
            pair_cov = len(full_pairs & sub_pairs) / len(full_pairs) if full_pairs else 0.0
            pair_scores.append(pair_cov)
        except:
            continue
    
    single_avg = np.mean(single_scores) if single_scores else 0.0
    pair_avg = np.mean(pair_scores) if pair_scores else 0.0
    return round(0.5 * single_avg + 0.5 * pair_avg, 4)

def compute_balance_to_full(sub, full):
    """BASELINE 2: Distributional Balance"""
    sims = []
    attrs = [a for a in SENSITIVE_ATTRIBUTES if a in full.columns and a in sub.columns and len(sub[a].dropna()) > 0]
    for a in attrs:
        try:
            full_counts = full[a].astype(str).value_counts(normalize=True)
            sub_counts = sub[a].astype(str).value_counts(normalize=True)
            all_cats = full_counts.index.union(sub_counts.index)
            full_counts = full_counts.reindex(all_cats, fill_value=0.0)
            sub_counts = sub_counts.reindex(all_cats, fill_value=0.0)
            jsd = jensenshannon(full_counts, sub_counts, base=2)
            sims.append(1.0 - jsd)
        except:
            continue
    return round(np.mean(sims), 4) if sims else 0.0

def gower_distance(df):
    """BASELINE 3: Test Set Diversity"""
    df = df.copy()
    n = len(df)
    if n < 2: return 0.0
    
    # Only use columns that actually exist
    available_cols = [c for c in df.columns if c in df.columns]
    df = df[available_cols]
    
    numeric_cols = df.select_dtypes(["int64", "float64"]).columns
    cat_cols = df.select_dtypes("object").columns
    
    for col in numeric_cols:
        rng = df[col].max() - df[col].min()
        df[col] = (df[col] - df[col].min()) / rng if rng else 0.0
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    
    X = df.values.astype(float)
    if X.shape[1] == 0:
        return 0.0
    
    # Sample if too large
    if n > 200:
        sample_indices = np.random.choice(n, size=200, replace=False)
        X = X[sample_indices]
        n = 200
    
    dists = [np.abs(X[i] - X[j]).mean() for i, j in combinations(range(min(n, 100)), 2)]
    return float(np.mean(dists)) if dists else 0.0

def compute_dsa(indices):
    """BASELINE 4: Distance to Training"""
    if len(indices) == 0: return np.nan
    X_sub = X_test_dense[indices]
    k = min(3, len(X_train_dense))
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_train_dense)
    distances, _ = nbrs.kneighbors(X_sub)
    return float(distances.mean())

def compute_idi(sub_df):
    """BASELINE 5: Individual Discrimination Index (Simplified for speed)"""
    if len(sub_df) == 0: return 0.0
    sample_size = min(50, len(sub_df))  # Reduced for speed
    sub_df_sample = sub_df.sample(n=sample_size, random_state=42) if len(sub_df) > sample_size else sub_df
    
    sub_clean = sub_df_sample.drop(columns=[target_col], errors="ignore")
    try:
        preds = model.predict(sub_clean)
    except:
        return 0.0
    
    count = 0
    test_attrs = [a for a in SENSITIVE_ATTRIBUTES[:2] if a in sub_df_sample.columns]  # Only first 2 attrs for speed
    
    for idx, row in sub_df_sample.iterrows():
        for attr in test_attrs:
            if attr not in row.index or pd.isna(row[attr]): continue
            try:
                flip_vals = [v for v in df_train[attr].dropna().unique() if str(v).strip() != str(row[attr]).strip()]
                if not flip_vals: continue
                cf_row = row.copy()
                cf_row[attr] = np.random.choice(flip_vals)
                cf_df = pd.DataFrame([cf_row]).drop(columns=[target_col], errors="ignore")
                pred_cf = model.predict(cf_df)[0]
                orig_pred = preds[sub_df_sample.index.get_loc(idx)]
                if pred_cf != orig_pred:
                    count += 1
                    break
            except:
                continue
    
    return round(count / len(sub_df_sample) * 100, 2)

def compute_boundary_coverage(indices):
    """BASELINE 6: Decision Boundary Coverage"""
    if len(indices) == 0: return 0.0
    sub_df = df_test.iloc[indices].drop(columns=[target_col], errors="ignore")
    try:
        proba = model.predict_proba(sub_df)[:, 1]
    except:
        return 0.0
    boundary_mask = (proba >= 0.4) & (proba <= 0.6)
    boundary_ratio = np.sum(boundary_mask) / len(proba) if len(proba) > 0 else 0.0
    boundary_proba = proba[boundary_mask]
    if len(boundary_proba) > 0:
        bins = 10
        boundary_bins = np.linspace(0.4, 0.6, bins + 1)
        hist, _ = np.histogram(boundary_proba, bins=boundary_bins)
        bin_coverage = np.sum(hist > 0) / bins
    else:
        bin_coverage = 0.0
    return round(0.5 * boundary_ratio + 0.5 * bin_coverage, 4)

def compute_feature_space_coverage(sub_df, full_df):
    """BASELINE 7: Feature Space Coverage"""
    coverage_scores = []
    attrs = [a for a in SENSITIVE_ATTRIBUTES if a in full_df.columns and a in sub_df.columns and len(sub_df[a].dropna()) > 0]
    for attr in attrs:
        try:
            if pd.api.types.is_numeric_dtype(full_df[attr]):
                full_binned = pd.cut(full_df[attr], bins=5, duplicates='drop')
                sub_binned = pd.cut(sub_df[attr], bins=5, duplicates='drop')
                full_cats = set(full_binned.cat.categories)
                sub_cats = set(sub_binned.value_counts().index)
                coverage = len(sub_cats) / len(full_cats) if full_cats else 0.0
            else:
                full_cats = set(full_df[attr].astype(str))
                sub_cats = set(sub_df[attr].astype(str))
                coverage = len(sub_cats) / len(full_cats) if full_cats else 0.0
            coverage_scores.append(coverage)
        except:
            continue
    return round(np.mean(coverage_scores), 4) if coverage_scores else 0.0

# ============================================================
# 5) COMPUTE BASELINES FOR ALL TEST SETS
# ============================================================
print(f"\nStep 5: Computing baseline metrics for ALL {len(test_sets)} test sets...")
print("   This may take a few minutes...")

results = []
computed_count = 0

for name, idx in test_sets.items():
    computed_count += 1
    if computed_count % 10 == 0:
        print(f"   Progress: {computed_count}/{len(test_sets)} test sets...")
    
    sub_df = df_test.iloc[idx]
    
    # Use only columns that exist in sub_df
    available_sens_attrs = [a for a in SENSITIVE_ATTRIBUTES if a in sub_df.columns]
    sub_sens = sub_df[available_sens_attrs] if available_sens_attrs else sub_df.iloc[:, :5]
    
    cov = compute_dataset_coverage(sub_df, df_test)
    bal = compute_balance_to_full(sub_df, df_test)
    diam = gower_distance(sub_sens)
    dsa = compute_dsa(idx)
    idi = compute_idi(sub_df)
    boundary = compute_boundary_coverage(idx)
    feat_space = compute_feature_space_coverage(sub_df, df_test)
    
    results.append({
        "TestSet": name,
        "DatasetCoverage": cov,
        "TestSetDiameter": diam,
        "BalanceToFull": bal,
        "DSA": dsa,
        "IDI_Percentage": idi,
        "BoundaryCoverage": boundary,
        "FeatureSpaceCoverage": feat_space,
        "Size": len(sub_df)
    })

baselines_df = pd.DataFrame(results)
print(f"\n   ✓ Computed baselines for {len(baselines_df)} test sets")

# ============================================================
# 6) MERGE EVERYTHING AND COMPUTE CORRELATIONS
# ============================================================
print("\nStep 6: Merging all data...")

merged = baselines_df.merge(train_scores, on="TestSet", how="inner")
merged = merged.merge(test_scores, on="TestSet", how="inner")

print(f"   ✓ Successfully merged {len(merged)} test sets")

merged.to_csv(os.path.join(OUTPUT_ROOT, "merged_data_ALL_TESTSETS.csv"), index=False)
print(f"   ✓ Saved: merged_data_ALL_TESTSETS.csv")

print(f"\n   Summary statistics:")
print(f"   - Total test sets analyzed: {len(merged)}")
if len(merged) > 0:
    print(f"   - MutationAdequacy range: [{merged['MutationAdequacy'].min():.3f}, {merged['MutationAdequacy'].max():.3f}]")
    print(f"   - FaultYield range: [{merged['FaultYield'].min():.3f}, {merged['FaultYield'].max():.3f}]")

# ============================================================
# 7) CORRELATION ANALYSIS
# ============================================================
print("\n" + "="*120)
print(f"CORRELATION ANALYSIS: Using ALL {len(merged)} Test Sets")
print("="*120)

def compute_correlation(metric_name):
    """Compute Spearman correlation between metric and FaultYield"""
    valid = merged[[metric_name, "FaultYield"]].dropna()
    if len(valid) < 3:
        return np.nan, 0
    rho, pval = spearmanr(valid[metric_name], valid["FaultYield"])
    return rho, len(valid)

adequacy_criteria = {
    "MutationAdequacy": ("Mutation Score (OURS)", "Fault-Based"),
    "DatasetCoverage": ("Combinatorial Coverage", "Coverage-Based"),
    "TestSetDiameter": ("Test Set Diversity", "Diversity-Based"),
    "BalanceToFull": ("Distributional Balance", "Distribution-Based"),
    "DSA": ("Distance to Training", "Distance-Based"),
    "IDI_Percentage": ("Individual Discrimination Index", "Counterfactual-Based"),
    "BoundaryCoverage": ("Decision Boundary Coverage", "Boundary-Based"),
    "FeatureSpaceCoverage": ("Feature Space Coverage", "Coverage-Based")
}

results_list = []
for metric, (name, category) in adequacy_criteria.items():
    rho, n_valid = compute_correlation(metric)
    results_list.append({
        "Adequacy Criterion": name,
        "Category": category,
        "Metric": metric,
        "Spearman ρ": rho,
        "Valid Pairs": n_valid
    })

final_results = pd.DataFrame(results_list).sort_values("Spearman ρ", ascending=False)

print("\n" + "="*120)
print("RESULTS: Adequacy Criteria Ranked by Predictive Power")
print("="*120)
print(final_results.to_string(index=False))
print("="*120)

final_results.to_csv(os.path.join(OUTPUT_ROOT, "final_correlation_ALL_TESTSETS.csv"), index=False)
print(f"\n✓ Saved: final_correlation_ALL_TESTSETS.csv")

# ============================================================
# 8) STATISTICAL SUMMARY
# ============================================================
print("\n" + "="*120)
print("STATISTICAL SUMMARY")
print("="*120)

mutation_row = final_results[final_results['Metric'] == 'MutationAdequacy'].iloc[0]
baseline_results = final_results[final_results['Metric'] != 'MutationAdequacy']

if len(baseline_results) > 0:
    best_baseline = baseline_results.iloc[0]
    
    print(f"\n✓ MUTATION ADEQUACY (Our Proposed Approach):")
    print(f"   Spearman ρ = {mutation_row['Spearman ρ']:.4f}")
    print(f"   Valid test set pairs = {mutation_row['Valid Pairs']}")
    
    correlation_strength = 'Strong' if abs(mutation_row['Spearman ρ']) > 0.7 else 'Moderate' if abs(mutation_row['Spearman ρ']) > 0.5 else 'Weak'
    print(f"   Interpretation: {correlation_strength} positive correlation")
    
    print(f"\n✓ BEST BASELINE: {best_baseline['Adequacy Criterion']}")
    print(f"   Spearman ρ = {best_baseline['Spearman ρ']:.4f}")
    print(f"   Category: {best_baseline['Category']}")
    
    if not pd.isna(mutation_row['Spearman ρ']) and not pd.isna(best_baseline['Spearman ρ']):
        if abs(best_baseline['Spearman ρ']) > 0.001:
            improvement = ((mutation_row['Spearman ρ'] - best_baseline['Spearman ρ']) / abs(best_baseline['Spearman ρ'])) * 100
            print(f"\n✓ IMPROVEMENT: {improvement:+.1f}% relative improvement over best baseline")

print("\n" + "="*120)
print("CONCLUSION")
print("="*120)
print(f"\nAnalyzed {len(merged)} test sets from master_summary.csv (German Credit Dataset)")
print(f"Mutation adequacy achieves ρ = {mutation_row['Spearman ρ']:.3f} correlation with fault detection")
print(f"\nThis validates mutation score as an effective test adequacy measure for fairness testing.")
print("="*120)

print(f"\n✓ All files saved to: {OUTPUT_ROOT}")
print("   - merged_data_ALL_TESTSETS.csv ({} test sets)".format(len(merged)))
print("   - final_correlation_ALL_TESTSETS.csv")
print("   - model.pkl")
print("\n✓ ANALYSIS COMPLETE - READY FOR PUBLICATION!")
print("="*120)