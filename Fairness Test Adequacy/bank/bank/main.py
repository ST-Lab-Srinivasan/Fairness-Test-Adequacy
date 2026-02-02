import os
import pandas as pd
import numpy as np
import traceback
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ----------------------------------------
# GPU LIBRARIES
# ----------------------------------------
import torch
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# ============================================================
# CONFIG — BANK DATASET (CSV)
# ============================================================

# TODO: change this to your local bank-full.csv path
BANK_FILE = r"C:\Users\srinivasanm23\Documents\fairness_test\Fairness_Test_Adequacy\Fairness Test Adequacy\bank\bank\bank-full.csv"

# TODO: change this to your desired output directory
OUTPUT_ROOT = r"C:\Users\srinivasanm23\Documents\fairness testing\adequacy\bank\output"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Label column: yes/no
LABEL_COL = "y"

# Choose 5 sensitive attributes for BANK
SENSITIVE_ATTRIBUTES = [
    "age",
    "job",
    "marital",
    "education",
    "housing"
]

# Numeric columns for some operators
LINEAR_NUMERIC_CANDIDATES = [
    "age",
    "balance",
    "duration",
    "campaign",
    "pdays",
    "previous"
]

MIN_GROUP_SIZE = 10
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Mutant explosion control
MAX_MUTANTS = 5000  # hard cap on total mutants
MAX_PER_OPERATOR = 20      # cap mutants per operator

# Distribution-based unrealistic mutant threshold
HIGH_SHIFT_THRESHOLD = 5.0  # mutants with huge distribution shift are skipped

# Heavy-operator flags (all enabled here)
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
# LOAD BANK & CREATE TRAIN/TEST SPLIT
# ============================================================

def load_bank(path: str) -> pd.DataFrame:
    """
    Load original Bank CSV, standardize column names,
    and clean label y (yes/no -> 1/0).
    """
    df = pd.read_csv(path, sep=";")

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("-", "_")
        .str.replace(" ", "_")
    )

    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in Bank file.")

    def clean_label(x):
        if isinstance(x, str):
            s = x.strip().lower()
            if s == "yes":
                return 1
            if s == "no":
                return 0
        return np.nan

    df[LABEL_COL] = df[LABEL_COL].apply(clean_label)

    df = df.dropna(subset=[LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    if len(df[LABEL_COL].unique()) < 2:
        raise ValueError("Label column has fewer than 2 classes after cleaning.")

    return df

# ============================================================
# PREPARE X, y
# ============================================================

def prepare_xy(df: pd.DataFrame, reference_columns=None):
    """
    Prepares encoded features X and label y.
    - Cleans numeric columns.
    - One-hot encodes categoricals.
    - Aligns to reference_columns (if provided).
    """
    df = df.copy()
    y = df[LABEL_COL].astype(int).values
    X = df.drop(columns=[LABEL_COL])

    # Categorical columns
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
        ppr = float(np.mean(y_p))

        metrics[g] = {"PPR": ppr}

    return metrics

def determine_status_with_deltas(orig_metrics, mut_metrics, value_keys, multiplier=1.5):
    """
    UPDATED: Returns status AND all delta values for sensitivity analysis.
    
    Threshold = mean(diff) + multiplier*std(diff) over all groups & keys.
    """
    diffs = []
    delta_records = []  # NEW: Store all deltas with group/metric info

    for g in orig_metrics:
        if g not in mut_metrics:
            continue
        for key in value_keys:
            if key in orig_metrics[g] and key in mut_metrics[g]:
                delta = abs(orig_metrics[g][key] - mut_metrics[g][key])
                diffs.append(delta)
                # NEW: Track each delta with metadata
                delta_records.append({
                    'group': g,
                    'metric': key,
                    'delta': delta,
                    'orig_value': orig_metrics[g][key],
                    'mut_value': mut_metrics[g][key]
                })

    if not diffs:
        return "Alive", [], 0.0, 0.0, 0.0

    diffs = np.array(diffs, dtype=float)
    mean_delta = float(np.mean(diffs))
    std_delta = float(np.std(diffs))
    max_delta = float(np.max(diffs))
    
    thresh = 0.0 if std_delta == 0 else mean_delta + multiplier * std_delta

    any_killed = np.any(diffs > thresh)
    status = "Killed" if any_killed else "Alive"
    
    return status, delta_records, max_delta, mean_delta, thresh

# ============================================================
# DISTRIBUTION SHIFT CHECK (TRAIN)
# ============================================================

def compute_distribution_shift_score(df_orig: pd.DataFrame, df_mut: pd.DataFrame) -> float:
    """
    Simple distribution shift score:
    - For numeric columns: normalized mean-diff
    - For sensitive categoricals: L1 distance on group proportions
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

    return float(np.mean(diffs)) if diffs else 0.0

# ============================================================
# MODEL TRAINING & PREDICTION (GPU-enabled)
# ============================================================

def fit_and_predict(model_name, X_train, y_train, X_test):
    """
    Train model on X_train, y_train.
    Return (train predictions, test predictions).
    """
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred  = model.predict(X_test)
        return train_pred, test_pred

    elif model_name == "decision_tree":
        model = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred  = model.predict(X_test)
        return train_pred, test_pred

    elif model_name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=50, max_depth=10, 
            random_state=RANDOM_STATE, n_jobs=-1
        )
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred  = model.predict(X_test)
        return train_pred, test_pred

    elif model_name == "gradient_boosting":
        if USE_GPU:
            # GPU-enabled XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest  = xgb.DMatrix(X_test)
            
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'device': 'cuda',
                'max_depth': 6,
                'learning_rate': 0.1,
                'seed': RANDOM_STATE
            }
            
            bst = xgb.train(params, dtrain, num_boost_round=100)
            train_pred = (bst.predict(dtrain) > 0.5).astype(int)
            test_pred = (bst.predict(dtest) > 0.5).astype(int)
        else:
            # CPU fallback
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=6, 
                learning_rate=0.1, random_state=RANDOM_STATE
            )
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred  = model.predict(X_test)
        
        return train_pred, test_pred

    elif model_name == "knn":
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        test_pred  = model.predict(X_test)
        return train_pred, test_pred

    else:
        raise ValueError(f"Unknown model: {model_name}")

# ============================================================
# MUTATION OPERATORS (simplified for brevity - use your full code)
# ============================================================

def build_all_mutants(df_train_raw, y_train):
    """
    Build mutation operators for Bank dataset.
    Returns {mutant_name: (operator_name, df_mut, y_mut)}
    """
    mutants = {}
    counts = {}
    
    # Initialize counts for all operators
    operator_names = [
        "instance_removal",
        "majority_undersample", 
        "minority_oversample",
        "counterfactual_swap",
        "label_flip",
        "intersectional_instance_removal",
        "feature_removal",
        "group_specific_corruption",
        "drift",
        "noise_injection",
        "permutation"
    ]
    
    if ENABLE_INTERSECTIONAL_LABEL_FLIP:
        operator_names.append("intersectional_label_flip")
    
    for op in operator_names:
        counts[op] = 0
    
    # INSTANCE REMOVAL
    for attr in SENSITIVE_ATTRIBUTES:
        if attr not in df_train_raw.columns:
            continue
        if counts["instance_removal"] >= MAX_PER_OPERATOR:
            break
            
        groups = pd.unique(df_train_raw[attr].astype(str))
        for g in groups:
            if counts["instance_removal"] >= MAX_PER_OPERATOR:
                break
                
            for frac in [0.1, 0.2, 0.3]:
                if counts["instance_removal"] >= MAX_PER_OPERATOR:
                    break
                    
                subset = df_train_raw[df_train_raw[attr].astype(str) == g]
                if len(subset) < MIN_GROUP_SIZE:
                    continue
                    
                n_remove = int(len(subset) * frac)
                if n_remove > 0:
                    indices_remove = subset.sample(n=n_remove, random_state=RANDOM_STATE).index
                    df_mut = df_train_raw.drop(indices_remove).reset_index(drop=True)
                    
                    name = f"instance_removal_{attr}={g}_frac={frac:.1f}"
                    mutants[name] = ("instance_removal", df_mut, None)
                    counts["instance_removal"] += 1
    
    # MAJORITY UNDERSAMPLE
    for frac in [0.2, 0.3, 0.4]:
        if counts["majority_undersample"] >= MAX_PER_OPERATOR:
            break
        
        vc = df_train_raw[LABEL_COL].value_counts()
        if len(vc) < 2:
            continue
        maj_class = vc.idxmax()
        
        idx_maj = df_train_raw[df_train_raw[LABEL_COL] == maj_class].index
        n_remove = int(len(idx_maj) * frac)
        if n_remove > 0:
            remove_idx = np.random.choice(idx_maj, n_remove, replace=False)
            df_mut = df_train_raw.drop(remove_idx).reset_index(drop=True)
            
            name = f"majority_undersample_frac={frac:.1f}"
            mutants[name] = ("majority_undersample", df_mut, None)
            counts["majority_undersample"] += 1
    
    # MINORITY OVERSAMPLE
    for mult in [1.5, 2.0]:
        if counts["minority_oversample"] >= MAX_PER_OPERATOR:
            break
        
        vc = df_train_raw[LABEL_COL].value_counts()
        if len(vc) < 2:
            continue
        min_class = vc.idxmin()
        
        minority_data = df_train_raw[df_train_raw[LABEL_COL] == min_class]
        n_dup = int(len(minority_data) * mult)
        if n_dup > 0:
            dup = minority_data.sample(n=n_dup, replace=True, random_state=RANDOM_STATE)
            df_mut = pd.concat([df_train_raw, dup], ignore_index=True)
            
            name = f"minority_oversample_mult={mult:.1f}"
            mutants[name] = ("minority_oversample", df_mut, None)
            counts["minority_oversample"] += 1
    
    # LABEL FLIP
    for frac in [0.1, 0.2, 0.3]:
        if counts["label_flip"] >= MAX_PER_OPERATOR:
            break
        
        n_flip = int(len(df_train_raw) * frac)
        if n_flip > 0:
            df_mut = df_train_raw.copy()
            flip_idx = df_mut.sample(n=n_flip, random_state=RANDOM_STATE).index
            df_mut.loc[flip_idx, LABEL_COL] = 1 - df_mut.loc[flip_idx, LABEL_COL]
            
            name = f"label_flip_frac={frac:.1f}"
            mutants[name] = ("label_flip", df_mut, None)
            counts["label_flip"] += 1
    
    # PERMUTATION
    for attr in SENSITIVE_ATTRIBUTES:
        if attr not in df_train_raw.columns:
            continue
        if counts["permutation"] >= MAX_PER_OPERATOR:
            break
        
        df_mut = df_train_raw.copy()
        df_mut[attr] = np.random.permutation(df_mut[attr].values)
        
        name = f"permutation_{attr}"
        mutants[name] = ("permutation", df_mut, None)
        counts["permutation"] += 1
    
    # NOISE INJECTION
    for sigma_mult in [1.0, 2.0, 3.0]:
        if counts["noise_injection"] >= MAX_PER_OPERATOR:
            break
        
        df_mut = df_train_raw.copy()
        for col in LINEAR_NUMERIC_CANDIDATES:
            if col in df_mut.columns:
                col_std = df_mut[col].std()
                if col_std > 0:
                    noise = np.random.normal(0, col_std * sigma_mult, len(df_mut))
                    df_mut[col] = df_mut[col] + noise
        
        name = f"noise_injection_sigma={sigma_mult:.1f}"
        mutants[name] = ("noise_injection", df_mut, None)
        counts["noise_injection"] += 1
    
    # FEATURE REMOVAL
    for attr in SENSITIVE_ATTRIBUTES:
        if attr not in df_train_raw.columns:
            continue
        if counts["feature_removal"] >= MAX_PER_OPERATOR:
            break
        
        df_mut = df_train_raw.drop(columns=[attr])
        
        name = f"feature_removal_{attr}"
        mutants[name] = ("feature_removal", df_mut, None)
        counts["feature_removal"] += 1
    
    # DRIFT
    for attr in SENSITIVE_ATTRIBUTES:
        if attr not in df_train_raw.columns:
            continue
        if counts["drift"] >= MAX_PER_OPERATOR:
            break
        
        groups = pd.unique(df_train_raw[attr].astype(str))
        if len(groups) < 2:
            continue
        
        g = groups[0]
        half_data = df_train_raw.sample(frac=0.5, random_state=RANDOM_STATE)
        drift_data = df_train_raw[df_train_raw[attr].astype(str) == g]
        df_mut = pd.concat([half_data, drift_data], ignore_index=True)
        
        name = f"drift_{attr}={g}"
        mutants[name] = ("drift", df_mut, None)
        counts["drift"] += 1
    
    # COUNTERFACTUAL SWAP (if enabled)
    if ENABLE_COUNTERFACTUAL_SWAP:
        for attr in SENSITIVE_ATTRIBUTES:
            if attr not in df_train_raw.columns:
                continue
            if counts.get("counterfactual_swap", 0) >= MAX_PER_OPERATOR:
                break
            
            groups = pd.unique(df_train_raw[attr].astype(str))
            if len(groups) < 2:
                continue
            
            for frac in [0.3, 0.5]:
                if counts.get("counterfactual_swap", 0) >= MAX_PER_OPERATOR:
                    break
                
                n_swap = int(len(df_train_raw) * frac)
                df_mut = df_train_raw.copy()
                swap_idx = df_mut.sample(n=n_swap, random_state=RANDOM_STATE).index
                df_mut.loc[swap_idx, attr] = np.random.permutation(
                    df_mut.loc[swap_idx, attr].values
                )
                
                name = f"counterfactual_swap_{attr}_frac={frac:.1f}"
                mutants[name] = ("counterfactual_swap", df_mut, None)
                counts["counterfactual_swap"] = counts.get("counterfactual_swap", 0) + 1
    
    return mutants

# ============================================================
# TEST SETS
# ============================================================

def create_test_sets(df_test_raw, y_test):
    """
    Build test sets: full, group-specific, and group-removed.
    """
    sets = {}

    # full test
    sets["full_test"] = {
        "indices": df_test_raw.index.to_list(),
        "adequacy": "full_test"
    }

    # group-specific test sets
    for attr in SENSITIVE_ATTRIBUTES:
        if attr not in df_test_raw.columns:
            continue

        groups = pd.unique(df_test_raw[attr].astype(str))

        for g in groups:
            subset = df_test_raw[df_test_raw[attr].astype(str) == g]
            if len(subset) < MIN_GROUP_SIZE:
                continue
            key = f"{attr}={g}"
            sets[key] = {
                "indices": subset.index.to_list(),
                "adequacy": key
            }

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
# CORE PIPELINE WITH DELTA TRACKING
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
    print(f"[{model_name}] Total mutants generated: {len(mutants)}")

    rows_eqodds = []
    rows_dp     = []
    rows_eopp   = []
    
    # NEW: Track all delta values for sensitivity analysis
    delta_tracking_rows = []

    kept_mutants = 0

    for mname, (op_name, df_mut, y_mut) in mutants.items():
        # 1) Distribution shift on TRAIN
        dist_shift_score = compute_distribution_shift_score(df_train_raw, df_mut)

        # Remove only unrealistic mutants
        if dist_shift_score > HIGH_SHIFT_THRESHOLD:
            continue

        kept_mutants += 1

        # 2) Prepare mutant data
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
                status_eq, deltas_eq, max_delta_eq, mean_delta_eq, thresh_eq = determine_status_with_deltas(
                    orig_eq, mut_eq, ("TPR", "FPR"), multiplier=1.5
                )
                
                # NEW: Save delta values
                for delta_info in deltas_eq:
                    delta_tracking_rows.append({
                        'model': model_name,
                        'mutant': mname,
                        'operator': op_name,
                        'test_set': ts_name,
                        'sensitive_attr': attr,
                        'group': delta_info['group'],
                        'fairness_metric': 'equalized_odds',
                        'metric_component': delta_info['metric'],
                        'delta': delta_info['delta'],
                        'orig_value': delta_info['orig_value'],
                        'mut_value': delta_info['mut_value'],
                        'max_delta': max_delta_eq,
                        'mean_delta': mean_delta_eq,
                        'threshold': thresh_eq,
                        'status': status_eq
                    })
                
                if status_eq == "Killed":
                    killed_eq += 1
                else:
                    alive_eq += 1

                # Demographic Parity
                orig_dp = compute_demographic_parity(y_base, groups_test)
                mut_dp  = compute_demographic_parity(y_mutp, groups_test)
                status_dp, deltas_dp, max_delta_dp, mean_delta_dp, thresh_dp = determine_status_with_deltas(
                    orig_dp, mut_dp, ("PPR",), multiplier=1.5
                )
                
                # NEW: Save delta values
                for delta_info in deltas_dp:
                    delta_tracking_rows.append({
                        'model': model_name,
                        'mutant': mname,
                        'operator': op_name,
                        'test_set': ts_name,
                        'sensitive_attr': attr,
                        'group': delta_info['group'],
                        'fairness_metric': 'demographic_parity',
                        'metric_component': delta_info['metric'],
                        'delta': delta_info['delta'],
                        'orig_value': delta_info['orig_value'],
                        'mut_value': delta_info['mut_value'],
                        'max_delta': max_delta_dp,
                        'mean_delta': mean_delta_dp,
                        'threshold': thresh_dp,
                        'status': status_dp
                    })
                
                if status_dp == "Killed":
                    killed_dp += 1
                else:
                    alive_dp += 1

                # Equal Opportunity
                orig_eo = compute_equal_opportunity(y_true, y_base, groups_test)
                mut_eo  = compute_equal_opportunity(y_true, y_mutp, groups_test)
                status_eo, deltas_eo, max_delta_eo, mean_delta_eo, thresh_eo = determine_status_with_deltas(
                    orig_eo, mut_eo, ("TPR",), multiplier=1.5
                )
                
                # NEW: Save delta values
                for delta_info in deltas_eo:
                    delta_tracking_rows.append({
                        'model': model_name,
                        'mutant': mname,
                        'operator': op_name,
                        'test_set': ts_name,
                        'sensitive_attr': attr,
                        'group': delta_info['group'],
                        'fairness_metric': 'equal_opportunity',
                        'metric_component': delta_info['metric'],
                        'delta': delta_info['delta'],
                        'orig_value': delta_info['orig_value'],
                        'mut_value': delta_info['mut_value'],
                        'max_delta': max_delta_eo,
                        'mean_delta': mean_delta_eo,
                        'threshold': thresh_eo,
                        'status': status_eo
                    })
                
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

    print(f"[{model_name}] Mutants kept after filtering: {kept_mutants}")

    df_eq = pd.DataFrame(rows_eqodds)
    df_dp = pd.DataFrame(rows_dp)
    df_eo = pd.DataFrame(rows_eopp)
    
    # NEW: Save delta tracking data
    df_deltas = pd.DataFrame(delta_tracking_rows)

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
    
    # NEW: Save delta tracking for sensitivity analysis
    if not df_deltas.empty:
        delta_dir = os.path.join(OUTPUT_ROOT, "delta_tracking")
        os.makedirs(delta_dir, exist_ok=True)
        df_deltas.to_csv(os.path.join(delta_dir, f"{model_name}_deltas.csv"), index=False)
        print(f"[{model_name}] Saved {len(df_deltas)} delta records to delta_tracking/")

    return df_eq, df_dp, df_eo, df_deltas

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
        print("="*80)
        print("BANK MARKETING - MUTATION TESTING WITH DELTA TRACKING")
        print("="*80)
        print()
        
        print("Loading Bank dataset...")
        df_full = load_bank(BANK_FILE)
        print(f"Total records: {len(df_full)}")
        
        print("\nLabel distribution:")
        print(df_full[LABEL_COL].value_counts())
        
        print("\nCreating 70/30 train/test split...")
        df_train_raw, df_test_raw = train_test_split(
            df_full,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=df_full[LABEL_COL]
        )
        
        print(f"Train: {len(df_train_raw)} | Test: {len(df_test_raw)}")

        models = [
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "gradient_boosting",
            "knn"
        ]

        frames_eq_all = []
        frames_dp_all = []
        frames_eo_all = []
        frames_delta_all = []  # NEW

        for m in models:
            print(f"\n{'='*80}")
            print(f"Running model: {m}")
            print(f"{'='*80}")
            
            start_time = time.time()
            df_eq, df_dp, df_eo, df_deltas = run_model_block(m, df_train_raw, df_test_raw)
            elapsed = time.time() - start_time
            
            print(f"Completed {m} in {elapsed:.2f} seconds")

            if not df_eq.empty:
                df_eq["Model"] = m
                frames_eq_all.append(df_eq)

            if not df_dp.empty:
                df_dp["Model"] = m
                frames_dp_all.append(df_dp)

            if not df_eo.empty:
                df_eo["Model"] = m
                frames_eo_all.append(df_eo)
            
            if not df_deltas.empty:
                frames_delta_all.append(df_deltas)

        write_master_tables(frames_eq_all, "equalized_odds")
        write_master_tables(frames_dp_all, "demographic_parity")
        write_master_tables(frames_eo_all, "equal_opportunity")
        
        # NEW: Write combined delta tracking file
        if frames_delta_all:
            all_deltas = pd.concat(frames_delta_all, ignore_index=True)
            delta_path = os.path.join(OUTPUT_ROOT, "delta_tracking", "all_models_deltas.csv")
            all_deltas.to_csv(delta_path, index=False)
            print(f"\n{'='*80}")
            print(f"✓ Saved combined delta tracking: {delta_path}")
            print(f"  Total delta records: {len(all_deltas):,}")
            print(f"{'='*80}")

        print("\n" + "="*80)
        print("✓ MUTATION TESTING COMPLETE!")
        print("="*80)
        print(f"\nResults stored in: {OUTPUT_ROOT}")
        print("\nNext step: Run threshold_sensitivity_analysis_bank.py")
        print("="*80)

    except Exception as e:
        print("❌ Error:", e)
        traceback.print_exc()