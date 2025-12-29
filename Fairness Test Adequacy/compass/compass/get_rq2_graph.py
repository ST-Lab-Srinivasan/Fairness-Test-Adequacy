import os
import pandas as pd
import numpy as np
import traceback

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ============================================================
# CONFIG — COMPAS DATASET
# ============================================================

COMPAS_FILE = r"C:\Users\srinivasanm23\Documents\fairness_test\adequacy\compass\compas-scores-two-years.csv"

OUTPUT_ROOT = r"C:\Users\srinivasanm23\Documents\fairness_test\adequacy\compass"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

LABEL_COL = "two_year_recid"

# Sensitive attributes (you can adjust this list if needed)
SENSITIVE_ATTRIBUTES = [
    "race",
    "sex",
    "age",
    "age_cat",
    "c_charge_degree",
    "priors_count",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
]

# Numeric columns used by some operators
LINEAR_NUMERIC_CANDIDATES = [
    "age",
    "priors_count",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "decile_score",
    "days_b_screening_arrest",
]

MIN_GROUP_SIZE = 10
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Mutation caps
MAX_MUTANTS = 5000         # global hard cap
MAX_PER_OPERATOR = 20      # cap per operator

# Distribution shift threshold (filter unrealistic mutants)
HIGH_SHIFT_THRESHOLD = 5.0

# Heavy operators on/off
ENABLE_INTERSECTIONAL_LABEL_FLIP = True
ENABLE_GROUP_CORRUPTION = True
ENABLE_COUNTERFACTUAL_SWAP = True

# ============================================================
# LOAD COMPAS & TRAIN/TEST SPLIT
# ============================================================

def load_compas(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("-", "_")
        .str.replace(" ", "_")
    )

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in COMPAS file.")

    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df.dropna(subset=[LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # sanity: two classes
    if len(df[LABEL_COL].unique()) < 2:
        raise ValueError("Label column has fewer than 2 classes after cleaning.")

    return df


def create_train_test(df: pd.DataFrame, test_size: float = 0.3):
    y = df[LABEL_COL].values
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


# ============================================================
# PREPARE X, y (NaN-safe)
# ============================================================

def prepare_xy(df: pd.DataFrame, reference_columns=None):
    """
    - Drops LABEL_COL from X
    - Converts numeric columns, imputes median, clips 99% tail
    - One-hot encodes categoricals
    - Aligns to training columns if reference_columns provided
    - Ensures no NaNs/inf in X
    """
    df = df.copy()
    y = df[LABEL_COL].astype(int).values
    X = df.drop(columns=[LABEL_COL])

    # identify categoricals
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]

    # numeric conversion
    for c in X.columns:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # numeric cleanup
    for c in X.columns:
        if c not in cat_cols:
            col = X[c]
            col = col.replace([np.inf, -np.inf], np.nan)
            # if column is fully NaN, fill with 0
            if col.isna().all():
                col = col.fillna(0.0)
            else:
                col = col.fillna(col.median())
                col = col.clip(upper=col.quantile(0.99))
            X[c] = col

    # categorical cleanup
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("Unknown")

    # one-hot
    X_enc = pd.get_dummies(X, columns=cat_cols)

    # align to reference columns (train)
    if reference_columns is not None:
        for col in reference_columns:
            if col not in X_enc.columns:
                X_enc[col] = 0
        extra = [c for c in X_enc.columns if c not in reference_columns]
        if extra:
            X_enc = X_enc.drop(columns=extra)
        X_enc = X_enc[reference_columns]
        ref_cols = reference_columns
    else:
        ref_cols = X_enc.columns.tolist()

    # final safety
    X_enc = X_enc.replace([np.inf, -np.inf], np.nan)
    # fill *any* remaining NaNs with 0 (very conservative, but safe for LR)
    X_enc = X_enc.fillna(0.0)

    return X_enc, y, ref_cols


# ============================================================
# FAIRNESS METRICS
# ============================================================

def compute_equalized_odds(y_true, y_pred, groups):
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
    metrics = {}
    uniq = pd.unique(groups)
    for g in uniq:
        mask = (groups == g)
        if mask.sum() < MIN_GROUP_SIZE:
            continue
        y_p = y_pred[mask]
        ppr = float(np.mean(y_p))
        metrics[g] = {"PPR": ppr}
    return metrics


def determine_status_per_groups(orig_m, mut_m, keys):
    """
    Threshold = mean(diff) + 1*std(diff) over all groups & keys.
    """
    diffs = []
    for g in orig_m:
        if g not in mut_m:
            continue
        for k in keys:
            if k in orig_m[g] and k in mut_m[g]:
                diffs.append(abs(orig_m[g][k] - mut_m[g][k]))
    if not diffs:
        return "Alive"
    diffs = np.array(diffs, dtype=float)
    mean = float(diffs.mean())
    std = float(diffs.std())
    thresh = 0.0 if std == 0 else mean + std
    return "Killed" if np.any(diffs > thresh) else "Alive"


# ============================================================
# DISTRIBUTION SHIFT SCORE
# ============================================================

def compute_distribution_shift_score(df_orig: pd.DataFrame, df_mut: pd.DataFrame) -> float:
    diffs = []

    # numeric
    numeric_cols = [
        c for c in df_orig.columns
        if pd.api.types.is_numeric_dtype(df_orig[c]) and c != LABEL_COL
    ]
    for col in numeric_cols:
        if col not in df_mut.columns:
            continue
        mean_o = df_orig[col].mean()
        mean_m = df_mut[col].mean()
        std_o = df_orig[col].std()
        if std_o == 0 or np.isnan(std_o):
            continue
        diffs.append(abs(mean_o - mean_m) / (std_o + 1e-8))

    # sensitive categorical L1 difference
    for attr in SENSITIVE_ATTRIBUTES:
        if attr not in df_orig.columns or attr not in df_mut.columns:
            continue
        p_o = df_orig[attr].astype(str).value_counts(normalize=True)
        p_m = df_mut[attr].astype(str).value_counts(normalize=True)
        p_m = p_m.reindex(p_o.index, fill_value=0.0)
        l1 = float(np.sum(np.abs(p_o.values - p_m.values)))
        diffs.append(l1)

    if not diffs:
        return 0.0
    return float(np.mean(diffs))


# ============================================================
# MUTATION OPERATORS
# ============================================================

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
            dfm = df.copy()
            muts[f"label_flip__{attr}={g}"] = ("label_flip", dfm, y_mut)
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
            dfm = df.loc[~mask].copy()
            ym = y[~mask]
            muts[f"inst_remove__{attr}={g}"] = ("instance_removal", dfm, ym)
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
            noise = np.random.normal(0, 0.1 * std_val, size=len(dfm))
            dfm[col] = dfm[col] + noise
            dfm[col] = dfm[col].replace([np.inf, -np.inf], np.nan)
            dfm[col] = dfm[col].fillna(dfm[col].median())
            muts[f"noise__{col}"] = ("noise_injection", dfm, None)
    return muts


def op_distribution_drift(df):
    muts = {}

    # age shift
    df_age = df.copy()
    if "age" in df_age.columns:
        df_age["age"] = df_age["age"] + np.random.randint(3, 7)
        df_age["age"] = df_age["age"].replace([np.inf, -np.inf], np.nan)
        df_age["age"] = df_age["age"].fillna(df_age["age"].median())
        muts["drift_age_shift"] = ("drift", df_age, None)

    # decile_score up 10%
    if "decile_score" in df.columns:
        df_ds = df.copy()
        df_ds["decile_score"] = df_ds["decile_score"] * 1.1
        df_ds["decile_score"] = df_ds["decile_score"].replace([np.inf, -np.inf], np.nan)
        df_ds["decile_score"] = df_ds["decile_score"].fillna(df_ds["decile_score"].median())
        muts["drift_decile_score_up10"] = ("drift", df_ds, None)

    # global numeric skew
    df_skew = df.copy()
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric:
        df_skew[c] = df_skew[c] * (1.0 + np.random.normal(0.02, 0.01))
        df_skew[c] = df_skew[c].replace([np.inf, -np.inf], np.nan)
        df_skew[c] = df_skew[c].fillna(df_skew[c].median())
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
                    dfm = df.loc[~mask].copy()
                    ym = y[~mask]
                    muts[f"intersect_remove__{a1}={v1}__{a2}={v2}"] = (
                        "intersectional_instance_removal",
                        dfm,
                        ym,
                    )
    return muts


def op_intersectional_label_flip(df, y, flip_rate=0.3):
    if not ENABLE_INTERSECTIONAL_LABEL_FLIP:
        return {}
    muts = {}
    sens = SENSITIVE_ATTRIBUTES
    for i in range(len(sens)):
        for j in range(i + 1, len(sens)):
            a1, a2 = sens[i], sens[j]
            if a1 not in df.columns or a2 not in df.columns:
                continue
            for v1 in pd.unique(df[a1].astype(str)):
                for v2 in pd.unique(df[a2].astype(str)):
                    mask = (
                        (df[a1].astype(str) == v1)
                        & (df[a2].astype(str) == v2)
                        & (y == 1)
                    )
                    idx = np.where(mask)[0]
                    if len(idx) < 10:
                        continue
                    y_mut = y.copy()
                    flip_n = max(1, int(len(idx) * flip_rate))
                    chosen = np.random.choice(idx, flip_n, replace=False)
                    y_mut[chosen] = 0
                    dfm = df.copy()
                    muts[f"intersect_label_flip__{a1}={v1}__{a2}={v2}"] = (
                        "intersectional_label_flip",
                        dfm,
                        y_mut,
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
                    "minority_oversample",
                    df_o,
                    y_o,
                )
            if frac > 0.5:
                df_u, y_u = undersample_group(df, y, mask, factor=0.7)
                muts[f"majority_under__{attr}={v}"] = (
                    "majority_undersample",
                    df_u,
                    y_u,
                )
    return muts


def op_group_specific_corruption(df):
    if not ENABLE_GROUP_CORRUPTION:
        return {}
    muts = {}
    numeric = [
        c
        for c in LINEAR_NUMERIC_CANDIDATES
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
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
                std_val = dfm[col].std() if dfm[col].std() > 0 else 1.0
                jitter = np.random.normal(0, 0.05 * std_val, size=mask.sum())
                dfm.loc[mask, col] = median_val + jitter
                dfm[col] = dfm[col].replace([np.inf, -np.inf], np.nan)
                dfm[col] = dfm[col].fillna(dfm[col].median())
            muts[f"group_corrupt__{attr}={v}"] = (
                "group_specific_corruption",
                dfm,
                None,
            )
    return muts


def op_counterfactual_swap(df):
    if not ENABLE_COUNTERFACTUAL_SWAP:
        return {}
    muts = {}
    numeric_cols = [
        c for c in df.columns if c != LABEL_COL and pd.api.types.is_numeric_dtype(df[c])
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
                dfm[attr] = dfm[attr].replace({g1: g2, g2: g1})
                mask = df[attr].astype(str).isin([g1, g2])
                for col in numeric_cols:
                    std = df[col].std()
                    if std == 0 or pd.isna(std):
                        continue
                    drift = np.random.uniform(-0.05 * std, 0.05 * std, size=mask.sum())
                    dfm.loc[mask, col] = dfm.loc[mask, col] + drift
                    dfm[col] = dfm[col].replace([np.inf, -np.inf], np.nan)
                    dfm[col] = dfm[col].fillna(dfm[col].median())
                muts[f"cf_swap__{attr}_{g1}<->{g2}"] = (
                    "counterfactual_swap",
                    dfm,
                    None,
                )
    return muts


def build_all_mutants(df, y):
    M = {}

    def add_capped(op_dict):
        if not op_dict:
            return
        subset = dict(list(op_dict.items())[:MAX_PER_OPERATOR])
        M.update(subset)

    add_capped(op_label_flip(df, y))
    add_capped(op_column_removal(df))
    add_capped(op_instance_removal(df, y))
    add_capped(op_permutation(df))
    add_capped(op_noise_injection(df))
    add_capped(op_distribution_drift(df))
    add_capped(op_intersectional_instance_removal(df, y))
    add_capped(op_intersectional_label_flip(df, y))
    add_capped(op_minority_resample(df, y))
    add_capped(op_group_specific_corruption(df))
    add_capped(op_counterfactual_swap(df))

    if len(M) > MAX_MUTANTS:
        M = dict(list(M.items())[:MAX_MUTANTS])

    return M


# ============================================================
# MODELS (CPU, NaN-safe)
# ============================================================

def fit_and_predict(model_name, X_train, y_train, X_test):
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Train labels for {model_name} have fewer than 2 classes.")

    if model_name == "logistic_regression":
        clf = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            solver="lbfgs",
        )
    elif model_name == "decision_tree":
        clf = DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
    elif model_name == "knn":
        clf = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_name == "gradient_boosting":
        clf = GradientBoostingClassifier(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred


# ============================================================
# TEST SETS
# ============================================================

def create_test_sets(df_test_raw, y_test, fractions=(0.4, 0.6, 0.8)):
    sets = {}

    # full test
    sets["full"] = {
        "indices": df_test_raw.index.to_list(),
        "adequacy": "full",
    }

    # stratified subsets
    for frac in fractions:
        try:
            _, subset = train_test_split(
                df_test_raw,
                test_size=frac,
                stratify=y_test,
                random_state=RANDOM_STATE,
            )
            sets[f"stratified_{int(frac*100)}"] = {
                "indices": subset.index.to_list(),
                "adequacy": f"stratified_{int(frac*100)}",
            }
        except Exception:
            pass

    # group-removed sets
    for attr in SENSITIVE_ATTRIBUTES:
        if attr not in df_test_raw.columns:
            continue
        groups = pd.unique(df_test_raw[attr].astype(str))
        for g in groups:
            subset = df_test_raw[df_test_raw[attr].astype(str) != g]
            if len(subset) < MIN_GROUP_SIZE:
                continue
            key = f"no_{attr}={g}"
            sets[key] = {
                "indices": subset.index.to_list(),
                "adequacy": key,
            }

    return sets


# ============================================================
# CORE PIPELINE
# ============================================================

def run_model_block(model_name, df_train_raw, df_test_raw):
    X_train, y_train, ref = prepare_xy(df_train_raw)
    X_test, y_test, _ = prepare_xy(df_test_raw, ref)

    scaler = StandardScaler().fit(X_train.values)
    X_train_s = scaler.transform(X_train.values)
    X_test_s = scaler.transform(X_test.values)

    # extra safety: kill any NaN/inf that survived scaler
    X_train_s = np.nan_to_num(X_train_s, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

    # base model
    _, base_pred = fit_and_predict(model_name, X_train_s, y_train, X_test_s)

    test_sets = create_test_sets(df_test_raw, y_test)
    mutants = build_all_mutants(df_train_raw, y_train)
    print(f"[{model_name}] Total mutants generated (before shift filter): {len(mutants)}")

    rows_eq = []
    rows_dp = []
    rows_eo = []

    kept_mutants = 0

    for mname, (op_name, df_mut, y_mut) in mutants.items():
        # 1) distribution shift
        dist_shift = compute_distribution_shift_score(df_train_raw, df_mut)
        if dist_shift > HIGH_SHIFT_THRESHOLD:
            continue

        kept_mutants += 1

        # 2) prepare mutant train
        X_mut, y_auto, _ = prepare_xy(df_mut, ref)
        y_used = y_mut if y_mut is not None else y_auto

        if len(X_mut) != len(y_used):
            continue

        X_mut_s = scaler.transform(X_mut.values)
        X_mut_s = np.nan_to_num(X_mut_s, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            _, mut_pred_test = fit_and_predict(model_name, X_mut_s, y_used, X_test_s)
        except Exception:
            continue

        # 3) evaluate fairness on all test sets
        for ts_name, ts_info in test_sets.items():
            mask = X_test.index.isin(ts_info["indices"])
            if mask.sum() < MIN_GROUP_SIZE:
                continue

            y_true = y_test[mask]
            y_base = base_pred[mask]
            y_mutp = mut_pred_test[mask]

            killed_eq = alive_eq = 0
            killed_dp = alive_dp = 0
            killed_eo = alive_eo = 0

            for attr in SENSITIVE_ATTRIBUTES:
                if attr not in df_test_raw.columns:
                    continue
                groups = df_test_raw.loc[mask, attr].astype(str).values

                # Equalized odds
                orig_eq = compute_equalized_odds(y_true, y_base, groups)
                mut_eq = compute_equalized_odds(y_true, y_mutp, groups)
                status_eq = determine_status_per_groups(
                    orig_eq, mut_eq, ("TPR", "FPR")
                )
                if status_eq == "Killed":
                    killed_eq += 1
                else:
                    alive_eq += 1

                # Demographic parity
                orig_dp = compute_demographic_parity(y_base, groups)
                mut_dp = compute_demographic_parity(y_mutp, groups)
                status_dp = determine_status_per_groups(
                    orig_dp, mut_dp, ("PPR",)
                )
                if status_dp == "Killed":
                    killed_dp += 1
                else:
                    alive_dp += 1

                # Equal opportunity
                orig_eo = compute_equal_opportunity(y_true, y_base, groups)
                mut_eo = compute_equal_opportunity(y_true, y_mutp, groups)
                status_eo = determine_status_per_groups(
                    orig_eo, mut_eo, ("TPR",)
                )
                if status_eo == "Killed":
                    killed_eo += 1
                else:
                    alive_eo += 1

            if killed_eq + alive_eq > 0:
                ms_eq = killed_eq / (killed_eq + alive_eq)
                rows_eq.append(
                    {
                        "Operator": op_name,
                        "Mutant": mname,
                        "TestSet": ts_name,
                        "Adequacy": ts_info["adequacy"],
                        "Killed": killed_eq,
                        "Alive": alive_eq,
                        "MutationScore": round(ms_eq, 4),
                    }
                )

            if killed_dp + alive_dp > 0:
                ms_dp = killed_dp / (killed_dp + alive_dp)
                rows_dp.append(
                    {
                        "Operator": op_name,
                        "Mutant": mname,
                        "TestSet": ts_name,
                        "Adequacy": ts_info["adequacy"],
                        "Killed": killed_dp,
                        "Alive": alive_dp,
                        "MutationScore": round(ms_dp, 4),
                    }
                )

            if killed_eo + alive_eo > 0:
                ms_eo = killed_eo / (killed_eo + alive_eo)
                rows_eo.append(
                    {
                        "Operator": op_name,
                        "Mutant": mname,
                        "TestSet": ts_name,
                        "Adequacy": ts_info["adequacy"],
                        "Killed": killed_eo,
                        "Alive": alive_eo,
                        "MutationScore": round(ms_eo, 4),
                    }
                )

    print(f"[{model_name}] Mutants kept after shift filter: {kept_mutants}")

    df_eq = pd.DataFrame(rows_eq)
    df_dp = pd.DataFrame(rows_dp)
    df_eo = pd.DataFrame(rows_eo)

    metric_map = {
        "equalized_odds": df_eq,
        "demographic_parity": df_dp,
        "equal_opportunity": df_eo,
    }

    for metric_name, df_metric in metric_map.items():
        metric_dir = os.path.join(OUTPUT_ROOT, metric_name, model_name)
        os.makedirs(metric_dir, exist_ok=True)
        if not df_metric.empty:
            df_metric.to_csv(
                os.path.join(metric_dir, "detailed.csv"), index=False
            )

    return df_eq, df_dp, df_eo


# ============================================================
# MASTER TABLES
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
            TotalAlive=("Alive", "sum"),
        )
        .reset_index()
    )
    op_summary["MutationScore"] = (
        op_summary["TotalKilled"]
        / (op_summary["TotalKilled"] + op_summary["TotalAlive"])
    )

    op_summary.to_csv(
        os.path.join(out_dir, "master_operator_summary.csv"),
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    try:
        print("Loading COMPAS data...")
        df_all = load_compas(COMPAS_FILE)

        print("Label distribution (all):")
        print(df_all[LABEL_COL].value_counts())

        df_train_raw, df_test_raw = create_train_test(df_all, test_size=0.3)
        print("Train size:", len(df_train_raw), " Test size:", len(df_test_raw))

        models = [
            "logistic_regression",
            "decision_tree",
            "knn",
            "random_forest",
            "gradient_boosting",
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

        write_master_tables(frames_eq_all, "equalized_odds")
        write_master_tables(frames_dp_all, "demographic_parity")
        write_master_tables(frames_eo_all, "equal_opportunity")

        print("\nDone. COMPAS fairness mutation analysis complete.")
        print("Results stored under:", OUTPUT_ROOT)

    except Exception as e:
        print("❌ Error:", repr(e))
        traceback.print_exc()
