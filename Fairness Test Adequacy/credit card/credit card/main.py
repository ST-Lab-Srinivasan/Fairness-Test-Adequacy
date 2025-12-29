import os
import pandas as pd
import numpy as np
import traceback

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG — CREDIT CARD CLIENTS DATASET (XLSX VERSION)
# ============================================================
TRAIN_FILE = r"C:\Users\srinivasanm23\Documents\fairness_test\adequacy\credit card\train.xlsx"
TEST_FILE  = r"C:\Users\srinivasanm23\Documents\fairness_test\adequacy\credit card\test.xlsx"

OUTPUT_ROOT = r"C:\Users\srinivasanm23\Documents\fairness_test\adequacy\credit card\output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

LABEL_COL = "default_payment_next_month"

# Sensitive attributes for fairness
SENSITIVE_ATTRIBUTES = [
    "sex",
    "education",
    "marriage",
    "age"
]

# Numeric columns used in corruption operators
LINEAR_NUMERIC_CANDIDATES = [
    "limit_bal",
    "age",
    "bill_amt1", "bill_amt2", "bill_amt3",
    "bill_amt4", "bill_amt5", "bill_amt6",
    "pay_amt1", "pay_amt2", "pay_amt3",
    "pay_amt4", "pay_amt5", "pay_amt6"
]

MIN_GROUP_SIZE = 10
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Threshold params for fairness kill/alive
EPS_EQUIV = 0.01
K_DYNAMIC = 1.0

# ---- Option D controls ----
MAX_MUTANTS = 2000          # global hard cap across all operators
HIGH_SHIFT_THRESHOLD = 5.0  # mutants with shift > this are discarded as unrealistic

# ------------------------------------------------------------
# (Kept from German version for completeness; not used here)
# ------------------------------------------------------------
def convert_A_code(value):
    val = str(value).strip()
    if val.startswith("A") and val[1:].isdigit():
        return int(val[1:])
    return value

# ============================================================
# LOAD DATA — CREDIT CARD CLIENTS FROM XLSX
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    # Requires openpyxl: pip install openpyxl
    df = pd.read_excel(path, engine="openpyxl")

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Ensure label column matches LABEL_COL
    if LABEL_COL not in df.columns:
        for cand in ["default_payment_next_month", "default payment next month"]:
            cand_std = cand.strip().lower().replace(" ", "_")
            if cand_std in df.columns:
                df.rename(columns={cand_std: LABEL_COL}, inplace=True)
                break

    # Make sure label is int 0/1
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df.dropna(subset=[LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    return df

# ============================================================
# PREPARE X, y  (with float32 to save memory)
# ============================================================
def prepare_xy(df: pd.DataFrame, reference_columns=None):
    # shallow copy to reduce memory pressure
    df = df.copy(deep=False)
    y = df[LABEL_COL].astype(int).values
    X = df.drop(columns=[LABEL_COL])

    # categorical columns (object dtype)
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]

    # numeric cleaning
    for c in X.columns:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # imputation and clipping for numeric
    for c in X.columns:
        if c not in cat_cols:
            X[c] = X[c].replace([np.inf, -np.inf], np.nan)
            X[c] = X[c].fillna(X[c].median())
            X[c] = X[c].clip(upper=X[c].quantile(0.99))

    # categorical cleanup
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("Unknown")

    # one-hot encode categorical
    X_enc = pd.get_dummies(X, columns=cat_cols)

    if reference_columns is not None:
        # align to reference (train) columns
        for col in reference_columns:
            if col not in X_enc.columns:
                X_enc[col] = 0
        extra = [c for c in X_enc.columns if c not in reference_columns]
        if extra:
            X_enc = X_enc.drop(columns=extra)
        X_enc = X_enc[reference_columns]
        ref = reference_columns
    else:
        ref = X_enc.columns.tolist()

    # Convert numeric to float32 to reduce memory
    num_cols = [c for c in X_enc.columns if pd.api.types.is_numeric_dtype(X_enc[c])]
    X_enc[num_cols] = X_enc[num_cols].astype(np.float32)

    return X_enc, y, ref

# ============================================================
# FAIRNESS METRICS
# ============================================================
def compute_equalized_odds(y_true, y_pred, groups):
    """
    Returns per-group TPR and FPR.
    """
    metrics = {}
    uniq = pd.unique(groups)

    for g in uniq:
        mask = (groups == g)
        if mask.sum() < MIN_GROUP_SIZE:
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        pos = (y_t == 1)
        neg = (y_t == 0)

        tpr = float(np.mean(y_p[pos])) if pos.sum() > 0 else 0.0
        fpr = float(np.mean(y_p[neg])) if neg.sum() > 0 else 0.0

        metrics[g] = {"TPR": tpr, "FPR": fpr}

    return metrics


def compute_equal_opportunity(y_true, y_pred, groups):
    """
    Returns per-group TPR only.
    """
    metrics = {}
    uniq = pd.unique(groups)

    for g in uniq:
        mask = (groups == g)
        if mask.sum() < MIN_GROUP_SIZE:
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        pos = (y_t == 1)
        tpr = float(np.mean(y_p[pos])) if pos.sum() > 0 else 0.0

        metrics[g] = {"TPR": tpr}

    return metrics


def compute_demographic_parity(y_pred, groups):
    """
    Demographic Parity via Positive Prediction Rate (PPR).
    """
    metrics = {}
    uniq = pd.unique(groups)

    for g in uniq:
        mask = (groups == g)
        if mask.sum() < MIN_GROUP_SIZE:
            continue

        y_p = y_pred[mask]
        ppr = float(np.mean(y_p))  # P(ŷ = 1 | A=g)

        metrics[g] = {"PPR": ppr}

    return metrics


def determine_status_per_groups(orig_metrics, mut_metrics, value_keys, eps=EPS_EQUIV, k=K_DYNAMIC):
    """
    Generic threshold-based kill/alive decision using:
    threshold = mean(diff) + k * std(diff)
    where diff are absolute metric differences over all groups and keys.
    No 'Equivalent' category: only Killed / Alive.
    """
    diffs = []

    for g in orig_metrics:
        if g not in mut_metrics:
            continue
        for key in value_keys:
            if key in orig_metrics[g] and key in mut_metrics[g]:
                d = abs(orig_metrics[g][key] - mut_metrics[g][key])
                diffs.append(d)

    if not diffs:
        return "Alive"  # can't decide, treat as Alive

    diffs = np.array(diffs, dtype=float)
    thresh = float(np.mean(diffs) + k * np.std(diffs))

    # Any group beyond threshold → Killed
    any_killed = np.any(diffs > thresh)

    return "Killed" if any_killed else "Alive"

# ============================================================
# DISTRIBUTION SHIFT CHECK (TRAIN) — Option D
# ============================================================
def compute_distribution_shift_score(df_orig: pd.DataFrame, df_mut: pd.DataFrame) -> float:
    """
    Simple shift score: numeric mean/std distance + L1 difference in
    sensitive attribute distributions. Larger = farther from original.
    Used only to drop *unrealistic* mutants (very large shifts).
    """
    diffs = []

    # Numeric columns
    numeric_cols = [
        c for c in df_orig.columns
        if pd.api.types.is_numeric_dtype(df_orig[c]) and c != LABEL_COL
    ]
    for col in numeric_cols:
        if col not in df_mut.columns:
            continue
        mean_orig = df_orig[col].mean()
        mean_mut  = df_mut[col].mean()
        std_orig  = df_orig[col].std()
        if std_orig == 0 or np.isnan(std_orig):
            continue
        diffs.append(abs(mean_orig - mean_mut) / (std_orig + 1e-8))

    # Sensitive categorical distribution L1 differences
    for attr in SENSITIVE_ATTRIBUTES:
        if attr not in df_orig.columns or attr not in df_mut.columns:
            continue
        p_orig = df_orig[attr].astype(str).value_counts(normalize=True)
        p_mut  = df_mut[attr].astype(str).value_counts(normalize=True)
        p_mut  = p_mut.reindex(p_orig.index, fill_value=0.0)
        l1 = float(np.sum(np.abs(p_orig.values - p_mut.values)))
        diffs.append(l1)

    if not diffs:
        return 0.0

    return float(np.mean(diffs))

# ============================================================
# MUTATION OPERATORS (ALL KEPT, INCLUDING OVER/UNDER-SAMPLE)
# ============================================================

# ---------- Helper functions ----------
def oversample_group(df, y, mask, factor=1.5):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return df.copy(), y.copy()

    extra_n = max(1, int(len(idx) * (factor - 1.0)))
    chosen = np.random.choice(idx, extra_n, replace=True)

    df_new = pd.concat([df, df.iloc[chosen]], ignore_index=True)
    y_new = np.concatenate([y, y[chosen]])

    return df_new, y_new


def undersample_group(df, y, mask, factor=0.7):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return df.copy(), y.copy()

    keep_n = int(len(idx) * factor)
    if keep_n < 1:
        return df.copy(), y.copy()

    keep_idx = np.random.choice(idx, keep_n, replace=False)
    keep_mask = np.ones(len(df), dtype=bool)
    drop_idx = set(idx) - set(keep_idx)
    for di in drop_idx:
        keep_mask[di] = False

    return df.iloc[keep_mask].copy(), y[keep_mask].copy()

# ---------- Operators ----------
def op_label_flip(df, y, flip_rate=0.3):
    muts = {}
    for attr in SENSITIVE_ATTRIBUTES:
        if attr not in df.columns:
            continue
        for g in pd.unique(df[attr].astype(str)):
            mask = (df[attr].astype(str) == g) & (y == 1)
            idx = np.where(mask)[0]
            if len(idx) < 10:
                continue
            y_mut = y.copy()
            flip_n = max(1, int(len(idx) * flip_rate))
            chosen = np.random.choice(idx, flip_n, replace=False)
            y_mut[chosen] = 0
            muts[f"label_flip__{attr}={g}"] = ("label_flip", df.copy(), y_mut)
    return muts


def op_column_removal(df):
    muts = {}
    for col in df.columns:
        if col == LABEL_COL:
            continue
        dfm = df.drop(columns=[col]).copy()
        muts[f"col_remove__{col}"] = ("column_removal", dfm, None)
    return muts


def op_instance_removal(df, y):
    muts = {}
    for attr in SENSITIVE_ATTRIBUTES:
        if attr not in df.columns:
            continue
        for g in pd.unique(df[attr].astype(str)):
            mask = (df[attr].astype(str) == g)
            if mask.sum() < 20:
                continue
            muts[f"inst_remove__{attr}={g}"] = (
                "instance_removal",
                df.loc[~mask].copy(),
                y[~mask]
            )
    return muts


def op_permutation(df):
    muts = {}
    for col in df.columns:
        if col == LABEL_COL:
            continue
        dfm = df.copy()
        dfm[col] = np.random.permutation(dfm[col].values)
        muts[f"permute__{col}"] = ("permutation", dfm, None)
    return muts


def op_noise_injection(df):
    muts = {}
    for col in df.columns:
        if col == LABEL_COL:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            dfm = df.copy()
            std_val = dfm[col].std() if dfm[col].std() > 0 else 1.0
            dfm[col] = dfm[col] + np.random.normal(0, 0.1 * std_val, size=len(dfm))
            muts[f"noise__{col}"] = ("noise_injection", dfm, None)
    return muts


def op_distribution_drift(df):
    muts = {}
    df_age = df.copy()
    if "age" in df.columns:
        df_age["age"] = df_age["age"] + np.random.randint(3, 7)
        muts["drift_age_shift"] = ("drift", df_age, None)

    df_lb = df.copy()
    if "limit_bal" in df.columns:
        df_lb["limit_bal"] = df_lb["limit_bal"] * 1.1
        muts["drift_limit_bal_up10"] = ("drift", df_lb, None)

    df_skew = df.copy()
    numerical = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in numerical:
        df_skew[c] = df_skew[c] * (1.0 + np.random.normal(0.02, 0.01))
    muts["drift_numeric_skew"] = ("drift", df_skew, None)

    return muts


def op_intersectional_instance_removal(df, y):
    muts = {}
    sens = SENSITIVE_ATTRIBUTES

    for i in range(len(sens)):
        for j in range(i + 1, len(sens)):
            a1, a2 = sens[i], sens[j]
            if a1 not in df.columns or a2 not in df.columns:
                continue

            for v1 in pd.unique(df[a1].astype(str)):
                for v2 in pd.unique(df[a2].astype(str)):
                    mask = (df[a1].astype(str) == v1) & (df[a2].astype(str) == v2)
                    if mask.sum() < 15:
                        continue
                    muts[f"intersect_remove__{a1}={v1}__{a2}={v2}"] = (
                        "intersectional_instance_removal",
                        df.loc[~mask].copy(),
                        y[~mask]
                    )
    return muts


def op_intersectional_label_flip(df, y, flip_rate=0.3):
    muts = {}
    sens = SENSITIVE_ATTRIBUTES

    for i in range(len(sens)):
        for j in range(i + 1, len(sens)):
            a1, a2 = sens[i], sens[j]
            if a1 not in df.columns or a2 not in df.columns:
                continue

            vals1 = pd.unique(df[a1].astype(str))
            vals2 = pd.unique(df[a2].astype(str))

            for v1 in vals1:
                for v2 in vals2:
                    mask = (
                        (df[a1].astype(str) == v1) &
                        (df[a2].astype(str) == v2) &
                        (y == 1)
                    )
                    idx = np.where(mask)[0]
                    if len(idx) < 10:
                        continue

                    y_mut = y.copy()
                    flip_n = max(1, int(len(idx) * flip_rate))
                    chosen = np.random.choice(idx, flip_n, replace=False)
                    y_mut[chosen] = 0

                    muts[f"intersect_label_flip__{a1}={v1}__{a2}={v2}"] = (
                        "intersectional_label_flip",
                        df.copy(),
                        y_mut
                    )

    return muts



def op_minority_resample(df, y):
    muts = {}
    total = len(df)
    for attr in SENSITIVE_ATTRIBUTES:
        if attr not in df.columns:
            continue
        for v in pd.unique(df[attr].astype(str)):
            mask = (df[attr].astype(str) == v)
            group_size = mask.sum()
            frac = group_size / total

            if group_size < 10:
                continue

            if frac < 0.2:
                df_o, y_o = oversample_group(df, y, mask, factor=1.5)
                muts[f"minority_over__{attr}={v}"] = (
                    "minority_oversample", df_o, y_o
                )

            if frac > 0.5:
                df_u, y_u = undersample_group(df, y, mask, factor=0.7)
                muts[f"majority_under__{attr}={v}"] = (
                    "majority_undersample", df_u, y_u
                )

    return muts


def op_group_specific_corruption(df):
    muts = {}
    numeric = [c for c in LINEAR_NUMERIC_CANDIDATES
               if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if not numeric:
        return muts

    for attr in SENSITIVE_ATTRIBUTES:
        if attr not in df.columns:
            continue
        for v in pd.unique(df[attr].astype(str)):
            mask = (df[attr].astype(str) == v)
            if mask.sum() < 10:
                continue
            dfm = df.copy()
            for col in numeric:
                median_val = dfm.loc[mask, col].median()
                jitter = np.random.normal(
                    0, 0.05 * (dfm[col].std() if dfm[col].std() > 0 else 1),
                    size=mask.sum()
                )
                dfm.loc[mask, col] = median_val + jitter
            muts[f"group_corrupt__{attr}={v}"] = (
                "group_specific_corruption", dfm, None
            )
    return muts


def op_counterfactual_swap(df):
    muts = {}
    numeric_cols = [
        c for c in df.columns
        if c != LABEL_COL and pd.api.types.is_numeric_dtype(df[c])
    ]

    for attr in SENSITIVE_ATTRIBUTES:
        if attr not in df.columns:
            continue

        vals = pd.unique(df[attr].astype(str))
        if len(vals) < 2:
            continue

        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                g1, g2 = vals[i], vals[j]

                dfm = df.copy()
                # swap sensitive groups
                dfm[attr] = dfm[attr].replace({g1: g2, g2: g1})

                # numeric drift for affected rows
                mask = df[attr].astype(str).isin([g1, g2])
                for col in numeric_cols:
                    std = df[col].std()
                    if std == 0 or pd.isna(std):
                        continue
                    drift = np.random.uniform(-0.05 * std, 0.05 * std, size=mask.sum())
                    dfm.loc[mask, col] += drift

                muts[f"cf_swap__{attr}_{g1}<->{g2}"] = (
                    "counterfactual_swap",
                    dfm,
                    None,
                )

    return muts

# ============================================================
# COMBINE ALL OPERATORS + GLOBAL MUTANT CAP (Option D)
# ============================================================
def build_all_mutants(df, y):
    M = {}
    M.update(op_label_flip(df, y))
    M.update(op_column_removal(df))
    M.update(op_instance_removal(df, y))
    M.update(op_permutation(df))
    M.update(op_noise_injection(df))
    M.update(op_distribution_drift(df))
    M.update(op_intersectional_instance_removal(df, y))
    M.update(op_intersectional_label_flip(df, y))
    M.update(op_minority_resample(df, y))
    M.update(op_group_specific_corruption(df))
    M.update(op_counterfactual_swap(df))

    # Global cap to keep number of mutants manageable
    if len(M) > MAX_MUTANTS:
        keys = list(M.keys())
        np.random.shuffle(keys)
        selected = keys[:MAX_MUTANTS]
        M = {k: M[k] for k in selected}

    return M

# ============================================================
# MODELS
# ============================================================
def fit_and_predict(model_name, X_train, y_train, X_test):
    if model_name == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    elif model_name == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
    elif model_name == "knn":
        clf = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        )
    elif model_name == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
    else:
        raise ValueError("Unknown model")

    clf.fit(X_train, y_train)
    return clf, clf.predict(X_test)

# ============================================================
# CREATE TEST SUBSETS
# ============================================================
def create_test_sets(df_test_raw, y_test, fractions=(0.4, 0.6, 0.8)):
    sets = {"full": {"indices": df_test_raw.index.to_list(), "adequacy": "full"}}

    for frac in fractions:
        try:
            _, subset = train_test_split(
                df_test_raw, test_size=frac, stratify=y_test, random_state=RANDOM_STATE
            )
            sets[f"stratified_{int(frac * 100)}"] = {
                "indices": subset.index.to_list(),
                "adequacy": f"stratified_{int(frac * 100)}"
            }
        except Exception:
            pass

    for frac in fractions:
        subset = df_test_raw.sample(frac=frac, random_state=RANDOM_STATE)
        sets[f"random_{int(frac * 100)}"] = {
            "indices": subset.index.to_list(),
            "adequacy": f"random_{int(frac * 100)}"
        }

    return sets

# ============================================================
# RUN MODEL BLOCK – ONE PIPELINE, 3 METRICS + OPTION D FILTER
# ============================================================
def run_model_block(model_name, df_train_raw, df_test_raw):
    # Prepare encoded data
    X_train, y_train, ref = prepare_xy(df_train_raw)
    X_test,  y_test, _    = prepare_xy(df_test_raw, ref)

    # Scale numeric features
    scaler = StandardScaler().fit(X_train.values)
    X_train_s = scaler.transform(X_train.values)
    X_test_s  = scaler.transform(X_test.values)

    # Fit base model & predictions
    _, base_pred = fit_and_predict(model_name, X_train_s, y_train, X_test_s)

    # Build test subsets
    test_sets = create_test_sets(df_test_raw, y_test)

    # Build mutants
    mutants = build_all_mutants(df_train_raw, y_train)
    print(f"[{model_name}] Total mutants generated (before shift filter): {len(mutants)}")

    rows_eqodds = []
    rows_dp     = []
    rows_eopp   = []

    kept_mutants = 0

    # Iterate through mutants
    for mname, (op_name, df_mut, y_mut) in mutants.items():

        # Option D: distribution-shift filter (only drop unrealistic mutants)
        shift_score = compute_distribution_shift_score(df_train_raw, df_mut)
        if shift_score > HIGH_SHIFT_THRESHOLD:
            # too far → unrealistic mutant, skip
            continue

        kept_mutants += 1

        # Prepare mutant X,y
        X_mut, y_auto, _ = prepare_xy(df_mut, ref)
        y_used = y_mut if y_mut is not None else y_auto

        if len(X_mut) != len(y_used):
            continue

        X_mut_s = scaler.transform(X_mut.values)

        # Fit mutant model
        try:
            _, mut_pred = fit_and_predict(model_name, X_mut_s, y_used, X_test_s)
        except Exception:
            continue

        # Evaluate fairness for each test subset
        for ts_name, ts_info in test_sets.items():
            mask = X_test.index.isin(ts_info["indices"])
            if mask.sum() < MIN_GROUP_SIZE:
                continue

            y_true = y_test[mask]
            y_base = base_pred[mask]
            y_mutp = mut_pred[mask]

            killed_eq = alive_eq = 0
            killed_dp = alive_dp = 0
            killed_eo = alive_eo = 0

            # Evaluate attribute-wise metrics
            for attr in SENSITIVE_ATTRIBUTES:
                if attr not in df_test_raw.columns:
                    continue

                groups = df_test_raw.loc[mask, attr].astype(str).values

                # Equalized Odds
                orig_eq = compute_equalized_odds(y_true, y_base, groups)
                mut_eq  = compute_equalized_odds(y_true, y_mutp, groups)
                status_eq = determine_status_per_groups(orig_eq, mut_eq, value_keys=("TPR", "FPR"))
                if status_eq == "Killed":
                    killed_eq += 1
                else:
                    alive_eq += 1

                # Demographic Parity
                orig_dp = compute_demographic_parity(y_base, groups)
                mut_dp  = compute_demographic_parity(y_mutp, groups)
                status_dp = determine_status_per_groups(orig_dp, mut_dp, value_keys=("PPR",))
                if status_dp == "Killed":
                    killed_dp += 1
                else:
                    alive_dp += 1

                # Equal Opportunity
                orig_eo = compute_equal_opportunity(y_true, y_base, groups)
                mut_eo  = compute_equal_opportunity(y_true, y_mutp, groups)
                status_eo = determine_status_per_groups(orig_eo, mut_eo, value_keys=("TPR",))
                if status_eo == "Killed":
                    killed_eo += 1
                else:
                    alive_eo += 1

            # Append results per metric
            if killed_eq + alive_eq > 0:
                mscore_eq = killed_eq / (killed_eq + alive_eq)
                rows_eqodds.append({
                    "Operator": op_name,
                    "Mutant": mname,
                    "TestSet": ts_name,
                    "Adequacy": ts_info["adequacy"],
                    "Killed": killed_eq,
                    "Alive": alive_eq,
                    "MutationScore": round(mscore_eq, 4)
                })

            if killed_dp + alive_dp > 0:
                mscore_dp = killed_dp / (killed_dp + alive_dp)
                rows_dp.append({
                    "Operator": op_name,
                    "Mutant": mname,
                    "TestSet": ts_name,
                    "Adequacy": ts_info["adequacy"],
                    "Killed": killed_dp,
                    "Alive": alive_dp,
                    "MutationScore": round(mscore_dp, 4)
                })

            if killed_eo + alive_eo > 0:
                mscore_eo = killed_eo / (killed_eo + alive_eo)
                rows_eopp.append({
                    "Operator": op_name,
                    "Mutant": mname,
                    "TestSet": ts_name,
                    "Adequacy": ts_info["adequacy"],
                    "Killed": killed_eo,
                    "Alive": alive_eo,
                    "MutationScore": round(mscore_eo, 4)
                })

    print(f"[{model_name}] Mutants kept after distribution-based filtering: {kept_mutants}")

    # Convert lists to DataFrames
    df_eq = pd.DataFrame(rows_eqodds)
    df_dp = pd.DataFrame(rows_dp)
    df_eo = pd.DataFrame(rows_eopp)

    # Save results to metric-specific folders per model
    metric_map = {
        "equalized_odds": df_eq,
        "demographic_parity": df_dp,
        "equal_opportunity": df_eo,
    }

    for metric_name, df_metric in metric_map.items():
        metric_dir = os.path.join(OUTPUT_ROOT, metric_name, model_name)
        os.makedirs(metric_dir, exist_ok=True)
        if not df_metric.empty:
            df_metric.to_csv(os.path.join(metric_dir, "detailed.csv"), index=False)

    return df_eq, df_dp, df_eo

# ============================================================
# MASTER SUMMARY PER METRIC
# ============================================================
def write_master_tables(frames, metric_name):
    if not frames:
        return
    master = pd.concat(frames, ignore_index=True)
    out_dir = os.path.join(OUTPUT_ROOT, metric_name)
    os.makedirs(out_dir, exist_ok=True)

    master.to_csv(os.path.join(out_dir, "master_summary.csv"), index=False)

    op_summary = (
        master.groupby(["Operator", "TestSet"])
        .agg(
            TotalMutants=("Mutant", "nunique"),
            TotalKilled=("Killed", "sum"),
            TotalAlive=("Alive", "sum")
        )
        .reset_index()
    )

    op_summary["MutationScore"] = (
        op_summary["TotalKilled"] /
        (op_summary["TotalKilled"] + op_summary["TotalAlive"])
    )

    op_summary.to_csv(
        os.path.join(out_dir, "master_operator_summary.csv"),
        index=False
    )

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    try:
        print("Loading credit card train/test data...")
        df_train_raw = load_data(TRAIN_FILE)
        df_test_raw  = load_data(TEST_FILE)

        print("Label distribution (train):")
        print(df_train_raw[LABEL_COL].value_counts())

        models = [
            "logistic_regression",
            "decision_tree",
            "knn",
            "random_forest",
            "gradient_boosting"
        ]

        frames_eq_all = []
        frames_dp_all = []
        frames_eo_all = []

        for m in models:
            print(f"\n=== Running model: {m} ===")
            df_eq, df_dp, df_eo = run_model_block(m, df_train_raw, df_test_raw)

            if not df_eq.empty:
                df_eq["Model"] = m
                frames_eq_all.append(df_eq)

            if not df_dp.empty:
                df_dp["Model"] = m
                frames_dp_all.append(df_dp)

            if not df_eo.empty:
                df_eo["Model"] = m
                frames_eo_all.append(df_eo)

        # write master tables per metric
        write_master_tables(frames_eq_all, "equalized_odds")
        write_master_tables(frames_dp_all, "demographic_parity")
        write_master_tables(frames_eo_all, "equal_opportunity")

        print("✅ Done. Metrics: Equalized Odds, Demographic Parity, Equal Opportunity.")
        print("✅ Option D: global mutant cap + distribution-shift filter, all operators kept.")
        print("✅ Results stored in:", OUTPUT_ROOT)

    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()
