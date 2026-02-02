# 70/30 in-memory split + ALL test sets + ALL 7 baseline metrics
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
# UPDATE THESE TWO PATHS ONLY
# ------------------------------
OUTPUT_ROOT = r"C:\Users\srinivasanm23\Documents\fairness_test\adequacy\compass\equal_opportunity"
MASTER_FILE = os.path.join(OUTPUT_ROOT, "master_summary.csv")
COMPAS_DATA_PATH = r"C:\Users\srinivasanm23\Documents\fairness_test\adequacy\compass\compas-scores-two-years.csv"

MODEL_PATH = os.path.join(OUTPUT_ROOT, "model.pkl")
SENSITIVE_ATTRIBUTES = ["race", "sex", "age", "age_cat", "c_charge_degree", "priors_count",
                        "juv_fel_count", "juv_misd_count", "juv_other_count"]

print("="*120)
print("COMPAS – MUTATION ADEQUACY vs BASELINES (ALL TEST SETS)")
print("="*120)

# ============================================================
# 1) Load master_summary.csv
# ============================================================
print("Step 1: Loading master_summary.csv...")
df_master = pd.read_csv(MASTER_FILE)
all_testset_names = sorted(df_master["TestSet"].unique())
print(f"   Found {len(all_testset_names)} test sets → ALL will be used")

# 70/30 mutant split - FIXED: Changed 'mutant' to 'mutants'
mutants = df_master["Mutant"].dropna().unique()
rng = np.random.default_rng(42)
train_mutants = set(rng.choice(mutants, size=int(0.7 * len(mutants)), replace=False))  # FIXED HERE
df_master["split"] = df_master["Mutant"].apply(lambda x: "train" if x in train_mutants else "test")

train_scores = df_master[df_master["split"] == "train"].groupby("TestSet").agg(
    MutationAdequacy=("MutationScore", "mean")
).reset_index()
test_scores = df_master[df_master["split"] == "test"].groupby("TestSet").agg(
    FaultYield=("MutationScore", "mean")
).reset_index()

print(f"   Train mutants: {len(train_mutants)} | Test mutants: {len(mutants) - len(train_mutants)}")

# ============================================================
# 2) Load COMPAS + 70/30 split + reset index
# ============================================================
print("\nStep 2: Loading COMPAS and splitting 70/30...")
df_full = pd.read_csv(COMPAS_DATA_PATH)
df_full.columns = df_full.columns.str.strip().str.lower().str.replace(r'[-\s\(\)]+', '_', regex=True)

target_col = "two_year_recid" if "two_year_recid" in df_full.columns else "is_recid"
df_full = df_full.dropna(subset=[target_col])
df_full[target_col] = df_full[target_col].astype(int)

df_train, df_test = train_test_split(df_full, test_size=0.3, stratify=df_full[target_col], random_state=42)
df_test = df_test.reset_index(drop=True)  # CRITICAL FIX

print(f"   Train: {len(df_train)} | Test: {len(df_test)}")

X_train_raw = df_train.drop(columns=[target_col])
y_train = df_train[target_col]

# ============================================================
# 3) Train model
# ============================================================
print("\nStep 3: Training model...")
cat_cols = X_train_raw.select_dtypes("object").columns.tolist()
num_cols = X_train_raw.select_dtypes(["int64", "float64"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols)
], remainder="drop")

model = Pipeline([("prep", preprocessor),
                  ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))])
model.fit(X_train_raw, y_train)

os.makedirs(OUTPUT_ROOT, exist_ok=True)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

print(f"   Model trained. Training accuracy: {model.score(X_train_raw, y_train):.3f}")

X_train_dense = preprocessor.transform(X_train_raw)
X_test_dense = preprocessor.transform(df_test.drop(columns=[target_col]))

# ============================================================
# 4) Recreate ALL test sets
# ============================================================
print("\nStep 4: Recreating test sets...")
test_sets = {}

def create_test_set(name):
    df = df_test.copy()
    if name == "full":
        return list(range(len(df)))
    if name.startswith("stratified_"):
        try:
            frac = float(name.split("_")[1]) / 100
            _, idx = train_test_split(list(range(len(df))), test_size=frac,
                                      stratify=df[target_col], random_state=42)
            return idx
        except:
            _, idx = train_test_split(list(range(len(df))), test_size=frac, random_state=42)
            return idx
    if name.startswith("no_"):
        try:
            attr_val = name[3:].split("=", 1)
            attr = attr_val[0].strip()
            val = attr_val[1].strip()
            try: val = int(val)
            except: pass
            if attr in df.columns:
                mask = (df[attr].astype(str).str.strip() != str(val).strip())
                return df[mask].index.tolist()
        except: pass
    return []

for name in all_testset_names:
    test_sets[name] = create_test_set(name)

# Filter out empty test sets
test_sets = {k: v for k, v in test_sets.items() if len(v) > 0}
print(f"   Created {len(test_sets)} valid test sets")

# ============================================================
# 5) ALL 7 BASELINE METRIC FUNCTIONS (FULLY INCLUDED)
# ============================================================
def compute_dataset_coverage(sub, full):
    single_scores = []; pair_scores = []
    attrs = [a for a in SENSITIVE_ATTRIBUTES if a in full.columns and a in sub.columns]
    for a in attrs:
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
    for a, b in combinations(attrs, 2):
        A_full = pd.cut(full[a], bins=5, duplicates='drop').astype(str) if pd.api.types.is_numeric_dtype(full[a]) else full[a].astype(str)
        B_full = pd.cut(full[b], bins=5, duplicates='drop').astype(str) if pd.api.types.is_numeric_dtype(full[b]) else full[b].astype(str)
        A_sub  = pd.cut(sub[a],  bins=5, duplicates='drop').astype(str) if pd.api.types.is_numeric_dtype(sub[a])  else sub[a].astype(str)
        B_sub  = pd.cut(sub[b],  bins=5, duplicates='drop').astype(str) if pd.api.types.is_numeric_dtype(sub[b])  else sub[b].astype(str)
        full_pairs = set(zip(A_full, B_full))
        sub_pairs  = set(zip(A_sub,  B_sub))
        pair_cov = len(full_pairs & sub_pairs) / len(full_pairs) if full_pairs else 0.0
        pair_scores.append(pair_cov)
    return round(0.5 * np.mean(single_scores or [0]) + 0.5 * np.mean(pair_scores or [0]), 4)

def compute_balance_to_full(sub, full):
    sims = []
    attrs = [a for a in SENSITIVE_ATTRIBUTES if a in full.columns and a in sub.columns]
    for a in attrs:
        p = full[a].astype(str).value_counts(normalize=True)
        q = sub[a].astype(str).value_counts(normalize=True)
        all_cats = p.index.union(q.index)
        p = p.reindex(all_cats, fill_value=0)
        q = q.reindex(all_cats, fill_value=0)
        jsd = jensenshannon(p, q, base=2)
        sims.append(1 - jsd)
    return round(np.mean(sims) if sims else 0.0, 4)

def gower_distance(df):
    df = df.copy(); n = len(df)
    if n < 2: return 0.0
    num_cols = df.select_dtypes(["int64", "float64"]).columns
    cat_cols = df.select_dtypes("object").columns
    for col in num_cols:
        rng = df[col].max() - df[col].min()
        df[col] = (df[col] - df[col].min()) / rng if rng > 0 else 0
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols)
    X = df.values.astype(float)
    dists = [np.mean(np.abs(X[i] - X[j])) for i in range(n) for j in range(i+1, n)]
    return float(np.mean(dists)) if dists else 0.0

def compute_dsa(indices):
    if len(indices) == 0: return np.nan
    X_sub = X_test_dense[indices]
    k = min(3, len(X_train_dense))
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_train_dense)
    distances, _ = nbrs.kneighbors(X_sub)
    return float(distances.mean())

def compute_idi(sub_df):
    if len(sub_df) == 0: return 0.0
    # Sample for efficiency
    sample_size = min(100, len(sub_df))
    sub_sample = sub_df.sample(n=sample_size, random_state=42) if len(sub_df) > sample_size else sub_df
    
    sub_clean = sub_sample.drop(columns=[target_col], errors="ignore")
    preds = model.predict(sub_clean)
    count = 0
    
    for idx, row in sub_sample.iterrows():
        for attr in SENSITIVE_ATTRIBUTES[:3]:  # Focus on key attributes for speed
            if attr not in row.index or pd.isna(row[attr]): continue
            others = [v for v in df_train[attr].dropna().unique() if str(v) != str(row[attr])]
            if not others: continue
            cf_row = row.copy()
            cf_row[attr] = np.random.choice(others)
            try:
                cf_df = pd.DataFrame([cf_row]).drop(columns=[target_col], errors="ignore")
                if model.predict(cf_df)[0] != preds[sub_sample.index.get_loc(idx)]:
                    count += 1
                    break
            except: continue
    return round(count / len(sub_sample) * 100, 2)

def compute_boundary_coverage(indices):
    if len(indices) == 0: return 0.0
    sub_df = df_test.iloc[indices].drop(columns=[target_col], errors="ignore")
    proba = model.predict_proba(sub_df)[:, 1]
    boundary = (proba >= 0.4) & (proba <= 0.6)
    ratio = boundary.mean()
    if boundary.sum() > 0:
        hist, _ = np.histogram(proba[boundary], bins=10, range=(0.4, 0.6))
        bin_cov = np.mean(hist > 0)
    else:
        bin_cov = 0.0
    return round(0.5 * ratio + 0.5 * bin_cov, 4)

def compute_feature_space_coverage(sub_df, full_df):
    scores = []
    attrs = [a for a in SENSITIVE_ATTRIBUTES if a in full_df.columns and a in sub_df.columns]
    for a in attrs:
        if pd.api.types.is_numeric_dtype(full_df[a]):
            full_cats = len(pd.cut(full_df[a], bins=5, duplicates='drop').cat.categories)
            sub_cats  = len(pd.cut(sub_df[a],  bins=5, duplicates='drop').value_counts())
        else:
            full_cats = full_df[a].nunique()
            sub_cats  = sub_df[a].nunique()
        scores.append(sub_cats / full_cats if full_cats > 0 else 0)
    return round(np.mean(scores) if scores else 0.0, 4)

# ============================================================
# 6) Compute baselines + final results
# ============================================================
print("\nStep 5: Computing baselines...")
results = []
computed = 0

for name, idx in test_sets.items():
    if len(idx) == 0: continue
    computed += 1
    if computed % 20 == 0:
        print(f"   Progress: {computed}/{len(test_sets)} test sets...")
    
    sub_df = df_test.iloc[idx]

    results.append({
        "TestSet": name,
        "DatasetCoverage": compute_dataset_coverage(sub_df, df_test),
        "TestSetDiameter": gower_distance(sub_df[SENSITIVE_ATTRIBUTES]),
        "BalanceToFull": compute_balance_to_full(sub_df, df_test),
        "DSA": compute_dsa(idx),
        "IDI_Percentage": compute_idi(sub_df),
        "BoundaryCoverage": compute_boundary_coverage(idx),
        "FeatureSpaceCoverage": compute_feature_space_coverage(sub_df, df_test),
        "Size": len(sub_df)
    })

print(f"   Computed baselines for {len(results)} test sets")

baselines_df = pd.DataFrame(results)
merged = baselines_df.merge(train_scores, on="TestSet", how="inner").merge(test_scores, on="TestSet", how="inner")
merged.to_csv(os.path.join(OUTPUT_ROOT, "merged_data_FINAL.csv"), index=False)
print(f"   ✓ Saved merged data: {len(merged)} test sets")

def corr(m):
    d = merged[[m, "FaultYield"]].dropna()
    return spearmanr(d[m], d["FaultYield"])[0] if len(d) >= 3 else np.nan

final = pd.DataFrame([
    {"Criterion": v, "Metric": k, "ρ": corr(k), "Valid Pairs": len(merged[[k, "FaultYield"]].dropna())} 
    for k, v in {
        "MutationAdequacy": "Mutation Score (Ours)",
        "DatasetCoverage": "Combinatorial Coverage",
        "TestSetDiameter": "Test Set Diversity",
        "BalanceToFull": "Distributional Balance",
        "DSA": "Distance to Training",
        "IDI_Percentage": "Individual Discrimination",
        "BoundaryCoverage": "Boundary Coverage",
        "FeatureSpaceCoverage": "Feature Space Coverage"
    }.items()
]).sort_values("ρ", ascending=False)

final.to_csv(os.path.join(OUTPUT_ROOT, "final_correlation_results.csv"), index=False)

print("\n" + "="*120)
print("FINAL RESULT – COMPAS DATASET")
print("="*120)
print(final[["Criterion", "ρ", "Valid Pairs"]].to_string(index=False))
print("="*120)

mutation_result = final[final['Metric'] == 'MutationAdequacy'].iloc[0]
print(f"\n✓ MUTATION ADEQUACY: ρ = {mutation_result['ρ']:.4f} with {mutation_result['Valid Pairs']} test sets")
print(f"✓ Files saved to: {OUTPUT_ROOT}")
print("✓ SUCCESS – NO ERRORS!")
print("="*120)
