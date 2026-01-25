import os
import pandas as pd
import numpy as np
import traceback

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ----------------------------------------
# GPU LIBRARIES
# ----------------------------------------
import torch
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# ============================================================
# CONFIG — ADULT DATASET (CSV)
# ============================================================

TRAIN_FILE = r"C:\Users\srinivasanm23\Documents\fairness\fairness_adequacy\fair test adequacy\fair test adequacy\equalized_odd\Adult Dataset\linear_regression\equalized odd\adult_cleaned_correction.csv"
TEST_FILE  = r"C:\Users\srinivasanm23\Documents\fairness\fairness_adequacy\fair test adequacy\fair test adequacy\equalized_odd\Adult Dataset\linear_regression\equalized odd\test_data.csv"

OUTPUT_ROOT = r"C:\Users\srinivasanm23\Documents\fairness\fairness_adequacy\fair test adequacy\fair test adequacy\adult\output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

LABEL_COL = "income"

# Use EXACTLY 5 sensitive attributes
SENSITIVE_ATTRIBUTES = [
    "sex",
    "race",
    "age",
    "marital_status",
    "education"
]

LINEAR_NUMERIC_CANDIDATES = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week"
]

MIN_GROUP_SIZE = 10
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Mutant explosion control
MAX_MUTANTS = 5000           # hard cap on total mutants
MAX_PER_OPERATOR = 20        # OPTION A: cap mutants per operator

# Distribution-based weak mutant thresholds
# We currently only remove "unrealistic" mutants (too far)
HIGH_SHIFT_THRESHOLD = 5.0   # if shift score > 5, skip mutant

# Heavy-operator flags
ENABLE_INTERSECTIONAL_LABEL_FLIP = True
ENABLE_GROUP_CORRUPTION = True
ENABLE_COUNTERFACTUAL_SWAP = True

# ============================================================
# GPU CONFIG
# ============================================================

USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
print(f"GPU available: {USE_GPU}  |  DEVICE = {DEVICE}")

# ============================================================
# LOAD DATA — ADULT
# ============================================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("-", "_")
        .str.replace(" ", "_")
    )

    # Ensure label column exists
    if LABEL_COL not in df.columns and "income" in df.columns:
        df.rename(columns={"income": LABEL_COL}, inplace=True)

    def clean_label(x):
        if isinstance(x, str):
            s = x.strip().replace(".", "")
            if s == "<=50K":
                return 0
            if s == ">50K":
                return 1
        return np.nan

    df[LABEL_COL] = df[LABEL_COL].apply(clean_label)

    # Drop rows with missing labels – do NOT impute labels
    df = df.dropna(subset=[LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # Sanity check: must have both classes
    if len(df[LABEL_COL].unique()) < 2:
        raise ValueError("Label column has fewer than 2 classes after cleaning.")

    return df

# ============================================================
# PREPARE X, y
# ============================================================

def prepare_xy(df: pd.DataFrame, reference_columns=None):
    df = df.copy()
    y = df[LABEL_COL].astype(int).values
    X = df.drop(columns=[LABEL_COL])

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]

    # Numeric conversion
    for c in X.columns:
        if c not in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Numeric cleanup
    for c in X.columns:
        if c not in cat_cols:
            X[c] = X[c].replace([np.inf, -np.inf], np.nan)
            X[c] = X[c].fillna(X[c].median())
            X[c] = X[c].clip(upper=X[c].quantile(0.99))

    # Categorical cleanup
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("Unknown")

    # One-hot encode
    X_enc = pd.get_dummies(X, columns=cat_cols)

    # Align to reference (train) columns if provided
    if reference_columns is not None:
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

    # Final safety cleaning
    X_enc = X_enc.replace([np.inf, -np.inf], np.nan)
    X_enc = X_enc.fillna(X_enc.median(numeric_only=True))

    return X_enc, y, ref

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
        ppr = float(np.mean(y_p))  # P(ŷ = 1 | A=g)

        metrics[g] = {"PPR": ppr}

    return metrics

def determine_status_per_groups(orig_metrics, mut_metrics, value_keys):
    """
    Threshold = mean(diff) + 1*std(diff) over all groups & keys.
    """
    diffs = []

    for g in orig_metrics:
        if g not in mut_metrics:
            continue
        for key in value_keys:
            if key in orig_metrics[g] and key in mut_metrics[g]:
                diffs.append(abs(orig_metrics[g][key] - mut_metrics[g][key]))

    if not diffs:
        return "Alive"

    diffs = np.array(diffs, dtype=float)
    mean = float(np.mean(diffs))
    std  = float(np.std(diffs))
    thresh = 0.0 if std == 0 else mean + std

    any_killed = np.any(diffs > thresh)
    return "Killed" if any_killed else "Alive"

# ============================================================
# DISTRIBUTION SHIFT CHECK (TRAIN)
# ============================================================

def compute_distribution_shift_score(df_orig: pd.DataFrame, df_mut: pd.DataFrame) -> float:
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
# MUTATION OPERATORS + CAPPING (OPTION A)
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
            ym  = y[~mask]
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

    df_age = df.copy()
    if "age" in df_age.columns:
        df_age["age"] = df_age["age"] + np.random.randint(3, 7)
        df_age["age"] = df_age["age"].replace([np.inf, -np.inf], np.nan)
        df_age["age"] = df_age["age"].fillna(df_age["age"].median())
        muts["drift_age_shift"] = ("drift", df_age, None)

    df_cg = df.copy()
    if "capital_gain" in df_cg.columns:
        df_cg["capital_gain"] = df_cg["capital_gain"] * 1.1
        df_cg["capital_gain"] = df_cg["capital_gain"].replace([np.inf, -np.inf], np.nan)
        df_cg["capital_gain"] = df_cg["capital_gain"].fillna(df_cg["capital_gain"].median())
        muts["drift_capital_gain_up10"] = ("drift", df_cg, None)

    df_skew = df.copy()
    numerical = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for c in numerical:
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
                    ym  = y[~mask]
                    muts[f"intersect_remove__{a1}={v1}__{a2}={v2}"] = (
                        "intersectional_instance_removal",
                        dfm,
                        ym
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
                    mask = (df[a1].astype(str) == v1) & (df[a2].astype(str) == v2) & (y == 1)
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
    if not ENABLE_GROUP_CORRUPTION:
        return {}
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
                dfm[col] = dfm[col].replace([np.inf, -np.inf], np.nan)
                dfm[col] = dfm[col].fillna(dfm[col].median())
            muts[f"group_corrupt__{attr}={v}"] = (
                "group_specific_corruption", dfm, None
            )
    return muts

def op_counterfactual_swap(df):
    if not ENABLE_COUNTERFACTUAL_SWAP:
        return {}
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

# ---------- OPTION A: cap mutants per operator ----------

def cap_operator(m_dict):
    """Randomly keep at most MAX_PER_OPERATOR mutants from a given operator dict."""
    if len(m_dict) <= MAX_PER_OPERATOR:
        return m_dict
    keys = list(m_dict.keys())
    np.random.shuffle(keys)
    keys = keys[:MAX_PER_OPERATOR]
    return {k: m_dict[k] for k in keys}

def build_all_mutants(df, y):
    M = {}

    # For each operator, generate all, then cap to MAX_PER_OPERATOR
    M.update(cap_operator(op_label_flip(df, y)))
    M.update(cap_operator(op_column_removal(df)))
    M.update(cap_operator(op_instance_removal(df, y)))
    M.update(cap_operator(op_permutation(df)))
    M.update(cap_operator(op_noise_injection(df)))
    M.update(cap_operator(op_distribution_drift(df)))
    M.update(cap_operator(op_intersectional_instance_removal(df, y)))
    M.update(cap_operator(op_intersectional_label_flip(df, y)))
    M.update(cap_operator(op_minority_resample(df, y)))
    M.update(cap_operator(op_group_specific_corruption(df)))
    M.update(cap_operator(op_counterfactual_swap(df)))

    # Hard cap on total mutants (after per-operator caps)
    if len(M) > MAX_MUTANTS:
        M = dict(list(M.items())[:MAX_MUTANTS])

    return M

# ============================================================
# GPU / CPU MODELS
# ============================================================

def fit_and_predict(model_name, X_train, y_train, X_test):
    """
    GPU-enabled for logistic_regression (PyTorch).
    Tree-based models via XGBoost on CPU (hist), since GPU build
    of xgboost may not be available on this system.
    """
    if len(np.unique(y_train)) < 2:
        raise ValueError(f"Train labels for {model_name} have fewer than 2 classes.")

    # ------- PyTorch Logistic Regression -------
    if model_name == "logistic_regression":
        in_features = X_train.shape[1]

        model = torch.nn.Sequential(
            torch.nn.Linear(in_features, 1),
            torch.nn.Sigmoid()
        ).to(DEVICE)

        X_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(DEVICE)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=2048, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = torch.nn.BCELoss()

        model.train()
        for _ in range(30):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            probs = model(X_test_tensor).cpu().numpy().ravel()
        y_pred = (probs >= 0.5).astype(int)
        return model, y_pred

    # ------- XGBoost (CPU hist) -------
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",   # 'gpu_hist' removed to avoid error
        "predictor": "auto",
    }

    if model_name == "decision_tree":
        params.update({"max_depth": 8, "eta": 1.0})
        num_round = 1  # single tree

    elif model_name == "gradient_boosting":
        params.update({
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        })
        num_round = 150

    elif model_name == "random_forest":
        params.update({
            "max_depth": 8,
            "eta": 1.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "num_parallel_tree": 100,
        })
        num_round = 1  # RF-style: many trees in parallel

    else:
        raise ValueError(f"Unknown model: {model_name}")

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test)

    booster = xgb.train(params, dtrain, num_boost_round=num_round, verbose_eval=False)
    probs = booster.predict(dtest)
    y_pred = (probs >= 0.5).astype(int)

    return booster, y_pred

# ============================================================
# TEST SETS: full + stratified + group-removed
# ============================================================

def create_test_sets(df_test_raw, y_test, fractions=(0.4, 0.6, 0.8)):
    sets = {}

    sets["full"] = {
        "indices": df_test_raw.index.to_list(),
        "adequacy": "full"
    }

    # stratified adequacy levels
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

    # group-removed test sets
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
                "adequacy": key
            }

    return sets

# ============================================================
# CORE PIPELINE WITH DISTRIBUTION-BASED FILTERING + OPTION A
# ============================================================

def run_model_block(model_name, df_train_raw, df_test_raw):
    X_train, y_train, ref = prepare_xy(df_train_raw)
    X_test,  y_test, _    = prepare_xy(df_test_raw, ref)

    scaler = StandardScaler().fit(X_train.values)
    X_train_s = scaler.transform(X_train.values)
    X_test_s  = scaler.transform(X_test.values)

    # base model on TEST
    _, base_pred = fit_and_predict(model_name, X_train_s, y_train, X_test_s)

    test_sets = create_test_sets(df_test_raw, y_test)

    mutants = build_all_mutants(df_train_raw, y_train)
    print(f"[{model_name}] Total mutants generated (after per-operator cap, before shift filter): {len(mutants)}")

    rows_eqodds = []
    rows_dp     = []
    rows_eopp   = []

    kept_mutants = 0

    for mname, (op_name, df_mut, y_mut) in mutants.items():
        # 1) Distribution shift on TRAIN
        dist_shift_score = compute_distribution_shift_score(df_train_raw, df_mut)

        # Remove only unrealistic mutants (too big shift)
        if dist_shift_score > HIGH_SHIFT_THRESHOLD:
            continue

        kept_mutants += 1

        # 2) Prepare mutant data (aligned to train reference columns)
        X_mut, y_auto, _ = prepare_xy(df_mut, ref)
        y_used = y_mut if y_mut is not None else y_auto

        if len(X_mut) != len(y_used):
            continue

        X_mut_s = scaler.transform(X_mut.values)

        try:
            _, mut_pred_test = fit_and_predict(model_name, X_mut_s, y_used, X_test_s)
        except Exception:
            continue

        # 3) Evaluate fairness on TEST subsets
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

                groups_test = df_test_raw.loc[mask, attr].astype(str).values

                # Equalized Odds
                orig_eq = compute_equalized_odds(y_true, y_base, groups_test)
                mut_eq  = compute_equalized_odds(y_true, y_mutp, groups_test)
                status_eq = determine_status_per_groups(orig_eq, mut_eq, ("TPR", "FPR"))
                if status_eq == "Killed":
                    killed_eq += 1
                else:
                    alive_eq += 1

                # Demographic Parity
                orig_dp = compute_demographic_parity(y_base, groups_test)
                mut_dp  = compute_demographic_parity(y_mutp, groups_test)
                status_dp = determine_status_per_groups(orig_dp, mut_dp, ("PPR",))
                if status_dp == "Killed":
                    killed_dp += 1
                else:
                    alive_dp += 1

                # Equal Opportunity
                orig_eo = compute_equal_opportunity(y_true, y_base, groups_test)
                mut_eo  = compute_equal_opportunity(y_true, y_mutp, groups_test)
                status_eo = determine_status_per_groups(orig_eo, mut_eo, ("TPR",))
                if status_eo == "Killed":
                    killed_eo += 1
                else:
                    alive_eo += 1

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

    df_eq = pd.DataFrame(rows_eqodds)
    df_dp = pd.DataFrame(rows_dp)
    df_eo = pd.DataFrame(rows_eopp)

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
# MASTER TABLES + MAIN
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

if __name__ == "__main__":
    try:
        print("Loading Adult train/test data...")
        df_train_raw = load_data(TRAIN_FILE)
        df_test_raw  = load_data(TEST_FILE)

        print("Label distribution (train):")
        print(df_train_raw[LABEL_COL].value_counts())

        models = [
            "logistic_regression",
            "decision_tree",
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

        write_master_tables(frames_eq_all, "equalized_odds")
        write_master_tables(frames_dp_all, "demographic_parity")
        write_master_tables(frames_eo_all, "equal_opportunity")

        print("\nDone. Results stored under:", OUTPUT_ROOT)

    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()
