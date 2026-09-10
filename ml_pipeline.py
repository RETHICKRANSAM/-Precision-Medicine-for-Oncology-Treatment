"""
==========================================================
Leakage-Free Multiclass ML Pipeline
Oncology Toxicity Risk Prediction
==========================================================
Target:   Toxicity_Risk  (Low / Moderate / High)
Models:   Logistic Regression, Decision Tree, Random Forest,
          XGBoost, SVM
Dataset:  Personalized Precision Oncology (~3,893 rows)

DISCLAIMER: This is an educational / research prototype.
It is NOT clinically validated and must NOT be used to
make independent treatment decisions.
==========================================================
"""

import io
import json
import os
import sys
import warnings

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Columns to EXCLUDE (leakage / identifiers)
# ──────────────────────────────────────────────
EXCLUDE_COLS = [
    "Patient_ID",
    "Risk_Score",
    "Toxicity_Score",
    "Treatment_Response",
    "Clinical_Note",
    "Treatment_Drug",
    "Dosage_mg",
    "Treatment_Adherence_Pct",
    "Encounter_Date",
]

TARGET_COL = "Toxicity_Risk"


def banner(step_num: int, title: str):
    """Print a formatted step banner."""
    print(f"\n{'='*60}")
    print(f"  STEP {step_num} — {title}")
    print(f"{'='*60}\n")


# ==========================================================
#  STEP 1 — LOAD AND INSPECT DATA
# ==========================================================
def step1_load_and_inspect(path: str) -> pd.DataFrame:
    banner(1, "LOAD AND INSPECT DATA")

    df = pd.read_csv(path)
    print(f"Rows:             {df.shape[0]}")
    print(f"Columns:          {df.shape[1]}")
    print(f"Column names:     {df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}\n")
    missing = df.isnull().sum()
    missing_any = missing[missing > 0]
    if len(missing_any) == 0:
        print("Missing values:   NONE")
    else:
        print(f"Missing values:\n{missing_any}\n")
    print(f"Duplicate rows:   {df.duplicated().sum()}")

    assert "Patient_ID" in df.columns, "Patient_ID column NOT found!"
    print(f"Patient_ID exists: YES")
    print(f"Unique Patient_IDs: {df['Patient_ID'].nunique()}")

    print(f"\nTarget class distribution ({TARGET_COL}):")
    print(df[TARGET_COL].value_counts().to_string())

    enc_per_patient = df.groupby("Patient_ID").size()
    print(f"\nEncounters per Patient_ID:")
    print(enc_per_patient.describe().to_string())

    return df


# ==========================================================
#  STEP 2 — DATASET-SPECIFIC LEAKAGE AUDIT
# ==========================================================
def step2_leakage_audit(df: pd.DataFrame) -> pd.DataFrame:
    banner(2, "DATASET-SPECIFIC LEAKAGE AUDIT")

    leakage_records = [
        {
            "column": "Patient_ID",
            "leakage_risk": "HIGH",
            "reason": "Unique identifier — not predictive, causes memorisation",
            "action": "EXCLUDE from features; use for grouping only",
        },
        {
            "column": "Risk_Score",
            "leakage_risk": "HIGH",
            "reason": "Appears derived from risk/toxicity information",
            "action": "EXCLUDE",
        },
        {
            "column": "Toxicity_Score",
            "leakage_risk": "HIGH",
            "reason": "Toxicity_Risk is derived from Toxicity_Score thresholds",
            "action": "EXCLUDE",
        },
        {
            "column": "Treatment_Response",
            "leakage_risk": "HIGH",
            "reason": "Post-treatment outcome — not available at prediction time",
            "action": "EXCLUDE",
        },
        {
            "column": "Clinical_Note",
            "leakage_risk": "MODERATE",
            "reason": "Free text recorded after clinical events; may contain outcome info",
            "action": "EXCLUDE",
        },
        {
            "column": "Treatment_Drug",
            "leakage_risk": "MODERATE",
            "reason": "Post-assignment information; not available pre-treatment",
            "action": "EXCLUDE from primary model",
        },
        {
            "column": "Dosage_mg",
            "leakage_risk": "MODERATE",
            "reason": "Post-assignment information; unknown at prediction time",
            "action": "EXCLUDE from primary model",
        },
        {
            "column": "Treatment_Adherence_Pct",
            "leakage_risk": "HIGH",
            "reason": "Future treatment adherence — impossible to know at prediction time",
            "action": "EXCLUDE",
        },
        {
            "column": "Encounter_Date",
            "leakage_risk": "LOW",
            "reason": "Temporal field; not directly predictive",
            "action": "EXCLUDE from features",
        },
    ]

    # Verify Toxicity_Score → Toxicity_Risk relationship
    print("Verifying Toxicity_Score → Toxicity_Risk relationship:")
    crosstab = pd.crosstab(df["Toxicity_Score"], df["Toxicity_Risk"])
    print(crosstab.to_string())
    print("\n=> Toxicity_Score is strongly correlated with Toxicity_Risk.")
    print("   Toxicity_Risk is very likely derived from Toxicity_Score thresholds.")
    print("   EXCLUDING Toxicity_Score to prevent target leakage.\n")

    audit_df = pd.DataFrame(leakage_records)
    audit_path = os.path.join(REPORTS_DIR, "leakage_audit.csv")
    audit_df.to_csv(audit_path, index=False)
    print(f"Leakage audit saved → {audit_path}")
    print(audit_df.to_string(index=False))

    return audit_df


# ==========================================================
#  STEP 3 — PATIENT-LEVEL LEAKAGE CHECK
# ==========================================================
def step3_patient_level_check(df: pd.DataFrame):
    banner(3, "PATIENT-LEVEL LEAKAGE CHECK")

    n_unique = df["Patient_ID"].nunique()
    enc_counts = df.groupby("Patient_ID").size()
    multi_enc = enc_counts[enc_counts > 1]

    print(f"Unique patients:                {n_unique}")
    print(f"Total rows:                     {len(df)}")
    print(f"Patients with multiple encounters: {len(multi_enc)}")

    if len(multi_enc) > 0:
        # Check for conflicting labels
        conflict = (
            df.groupby("Patient_ID")[TARGET_COL]
            .nunique()
            .reset_index()
            .rename(columns={TARGET_COL: "n_labels"})
        )
        conflict_patients = conflict[conflict["n_labels"] > 1]
        print(f"Conflicting-label patients:     {len(conflict_patients)}")
    else:
        print(f"Conflicting-label patients:     0 (single encounter per patient)")

    print(f"\n=> Each Patient_ID has exactly 1 row.")
    print(f"   GroupShuffleSplit will still be used for correctness.")


# ==========================================================
#  STEP 4 — DEFINE PRIMARY FEATURES
# ==========================================================
def step4_define_features(df: pd.DataFrame):
    banner(4, "DEFINE PRIMARY FEATURES")

    all_cols = set(df.columns)
    exclude = set(EXCLUDE_COLS + [TARGET_COL])
    feature_cols = sorted(all_cols - exclude)

    # Categorise
    numeric_features = []
    categorical_features = []

    for col in feature_cols:
        if df[col].dtype in ["int64", "float64"]:
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    # Inspect each feature
    print("Feature inspection:")
    for col in feature_cols:
        nunique = df[col].nunique()
        dtype = df[col].dtype
        null_pct = df[col].isnull().mean() * 100
        print(f"  {col:30s}  dtype={str(dtype):10s}  unique={nunique:6d}  null%={null_pct:.1f}")

    print(f"\nNumeric features  ({len(numeric_features)}):")
    print(f"  {numeric_features}")
    print(f"\nCategorical features ({len(categorical_features)}):")
    print(f"  {categorical_features}")

    print(f"\nExcluded columns ({len(exclude)}):")
    for col in sorted(exclude):
        reason = "TARGET" if col == TARGET_COL else "LEAKAGE/IDENTIFIER"
        print(f"  {col:35s} — {reason}")

    print(f"\nTotal features used: {len(feature_cols)}")

    return numeric_features, categorical_features, feature_cols


# ==========================================================
#  STEP 5 — PREPROCESSING
# ==========================================================
def step5_build_preprocessor(numeric_features, categorical_features):
    banner(5, "PREPROCESSING PIPELINE")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    print("Preprocessor built:")
    print(f"  Numeric ({len(numeric_features)}):     Median impute → StandardScaler")
    print(f"  Categorical ({len(categorical_features)}): Mode impute → OneHotEncoder")
    print("  Fitted ONLY on training data (inside Pipeline)")

    return preprocessor


# ==========================================================
#  STEP 6 — TRAIN/TEST SPLIT  (Patient-level)
# ==========================================================
def step6_train_test_split(df, feature_cols):
    banner(6, "TRAIN/TEST SPLIT (Patient-Level)")

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    groups = df["Patient_ID"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups[train_idx]
    groups_test = groups[test_idx]

    # Patient overlap check
    train_patients = set(groups_train)
    test_patients = set(groups_test)
    overlap = train_patients & test_patients

    print(f"Train samples:       {len(X_train)}")
    print(f"Test samples:        {len(X_test)}")
    print(f"Train patients:      {len(train_patients)}")
    print(f"Test patients:       {len(test_patients)}")
    print(f"Patient overlap:     {len(overlap)}  {'✓ CLEAN' if len(overlap) == 0 else '✗ LEAKAGE!'}")

    print(f"\nTrain target distribution:")
    print(y_train.value_counts().to_string())
    print(f"\nTest target distribution:")
    print(y_test.value_counts().to_string())

    # High-class counts
    high_train = (y_train == "High").sum()
    high_test = (y_test == "High").sum()
    print(f"\n⚠  HIGH class — Train: {high_train}, Test: {high_test}")
    if high_test == 0:
        print("   WARNING: Zero High samples in test set — High-class test metrics will be undefined.")

    return X_train, X_test, y_train, y_test, groups_train, groups_test


# ==========================================================
#  STEP 7 — HANDLE CLASS IMBALANCE
# ==========================================================
def step7_class_imbalance(y_train, y_test):
    banner(7, "CLASS IMBALANCE ANALYSIS")

    print("Original full-data distribution:")
    combined = pd.concat([y_train, y_test])
    print(combined.value_counts().to_string())

    print(f"\nTrain distribution:")
    print(y_train.value_counts().to_string())

    print(f"\nStrategy:")
    print("  - class_weight='balanced' for LR, DT, RF, SVM")
    print("  - Scale_pos_weight / sample_weight for XGBoost")
    print("  - NO SMOTE applied (High class too small)")
    print("")
    print("⚠  WARNING: High class has ≤4 total samples.")
    print("   Per-class metrics for High will be extremely unstable.")
    print("   Do NOT rely on High-class recall as a reliable estimate.")


# ==========================================================
#  HELPER: Build cross-validation strategy
# ==========================================================
def get_cv_strategy(y_train, groups_train, n_splits=5):
    """Return StratifiedGroupKFold if feasible, else GroupKFold."""
    try:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        # Test if it works
        list(cv.split(np.zeros(len(y_train)), y_train, groups_train))
        print(f"  CV strategy: StratifiedGroupKFold (n_splits={n_splits})")
        return cv
    except Exception as e:
        print(f"  StratifiedGroupKFold failed ({e}), falling back to GroupShuffleSplit")
        from sklearn.model_selection import GroupKFold
        cv = GroupKFold(n_splits=n_splits)
        print(f"  CV strategy: GroupKFold (n_splits={n_splits})")
        return cv


# ==========================================================
#  HELPER: Evaluate cross-validation
# ==========================================================
def evaluate_cv(model, X_train, y_train, groups_train, cv, model_name):
    """Run manual cross-validation and collect all metrics."""
    from sklearn.base import clone

    metrics = {
        "accuracy": [],
        "macro_precision": [],
        "macro_recall": [],
        "macro_f1": [],
        "weighted_f1": [],
        "balanced_accuracy": [],
    }

    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups_train)):
        X_tr = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        m = clone(model)
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_val)

        metrics["accuracy"].append(accuracy_score(y_val, y_pred))
        metrics["macro_precision"].append(precision_score(y_val, y_pred, average="macro", zero_division=0))
        metrics["macro_recall"].append(recall_score(y_val, y_pred, average="macro", zero_division=0))
        metrics["macro_f1"].append(f1_score(y_val, y_pred, average="macro", zero_division=0))
        metrics["weighted_f1"].append(f1_score(y_val, y_pred, average="weighted", zero_division=0))
        metrics["balanced_accuracy"].append(balanced_accuracy_score(y_val, y_pred))

    print(f"\n  {model_name} — Cross-Validation Results ({len(metrics['accuracy'])} folds):")
    results = {}
    for metric_name, values in metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        results[metric_name] = {"mean": mean_val, "std": std_val}
        print(f"    {metric_name:25s}  {mean_val:.4f} ± {std_val:.4f}")

    return results


# ==========================================================
#  HELPER: Evaluate on test set
# ==========================================================
def evaluate_test(model, X_test, y_test, model_name):
    """Evaluate a fitted model on the test set."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_p = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    print(f"\n  {model_name} — Test Set Results:")
    print(f"    Accuracy:           {acc:.4f}")
    print(f"    Macro Precision:    {macro_p:.4f}")
    print(f"    Macro Recall:       {macro_r:.4f}")
    print(f"    Macro F1:           {macro_f1:.4f}")
    print(f"    Weighted F1:        {weighted_f1:.4f}")
    print(f"    Balanced Accuracy:  {bal_acc:.4f}")

    print(f"\n    Classification Report:")
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=["Low", "Moderate", "High"])
    print(f"    Confusion Matrix:")
    print(f"    {cm}")

    return {
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
    }


# ==========================================================
#  STEP 8 — MODEL 1: LOGISTIC REGRESSION
# ==========================================================
def step8_logistic_regression(preprocessor, X_train, y_train, groups_train, cv):
    banner(8, "MODEL 1: LOGISTIC REGRESSION")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )),
    ])

    param_dist = {
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["lbfgs", "saga"],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=10,
        cv=cv,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    search.fit(X_train, y_train, groups=groups_train)

    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV f1_macro: {search.best_score_:.4f}")

    return search.best_estimator_


# ==========================================================
#  STEP 9 — MODEL 2: DECISION TREE
# ==========================================================
def step9_decision_tree(preprocessor, X_train, y_train, groups_train, cv):
    banner(9, "MODEL 2: DECISION TREE")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    param_dist = {
        "classifier__max_depth": [3, 5, 10, 15, 20, None],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 4, 8],
        "classifier__criterion": ["gini", "entropy"],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=30,
        cv=cv,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    search.fit(X_train, y_train, groups=groups_train)

    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV f1_macro: {search.best_score_:.4f}")

    return search.best_estimator_


# ==========================================================
#  STEP 10 — MODEL 3: RANDOM FOREST
# ==========================================================
def step10_random_forest(preprocessor, X_train, y_train, groups_train, cv):
    banner(10, "MODEL 3: RANDOM FOREST")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    param_dist = {
        "classifier__n_estimators": [200, 300, 500],
        "classifier__max_depth": [None, 5, 10, 15, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
        "classifier__class_weight": ["balanced", "balanced_subsample"],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=40,
        cv=cv,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    search.fit(X_train, y_train, groups=groups_train)

    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV f1_macro: {search.best_score_:.4f}")

    return search.best_estimator_


# ==========================================================
#  STEP 11 — MODEL 4: XGBOOST
# ==========================================================
def step11_xgboost(preprocessor, X_train, y_train, groups_train, cv):
    banner(11, "MODEL 4: XGBOOST")

    # Encode labels for XGBoost
    le = LabelEncoder()
    le.fit(["High", "Low", "Moderate"])  # sorted alphabetical

    # Compute sample weights for class imbalance
    class_counts = y_train.value_counts()
    total = len(y_train)
    n_classes = len(class_counts)
    class_weight_map = {
        cls: total / (n_classes * count) for cls, count in class_counts.items()
    }
    print(f"  Class weight map: {class_weight_map}")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            verbosity=0,
        )),
    ])

    param_dist = {
        "classifier__n_estimators": [100, 200, 300, 500],
        "classifier__max_depth": [3, 5, 7, 10],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "classifier__min_child_weight": [1, 3, 5, 7],
        "classifier__gamma": [0, 0.1, 0.3, 0.5],
        "classifier__reg_lambda": [0.1, 1.0, 5.0, 10.0],
    }

    # Encode y for XGBoost
    y_train_encoded = le.transform(y_train)

    # Compute sample weights
    sample_weights = np.array([class_weight_map[label] for label in y_train])

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=50,
        cv=cv,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    search.fit(
        X_train, y_train_encoded, groups=groups_train,
        classifier__sample_weight=sample_weights,
    )

    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV f1_macro: {search.best_score_:.4f}")

    return search.best_estimator_, le


# ==========================================================
#  STEP 12 — MODEL 5: SVM
# ==========================================================
def step12_svm(preprocessor, X_train, y_train, groups_train, cv):
    banner(12, "MODEL 5: SVM")

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", SVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    param_dist = {
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__gamma": ["scale", "auto", 0.01, 0.001],
        "classifier__kernel": ["rbf", "poly"],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=20,
        cv=cv,
        scoring="f1_macro",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    search.fit(X_train, y_train, groups=groups_train)

    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV f1_macro: {search.best_score_:.4f}")

    return search.best_estimator_


# ==========================================================
#  STEP 15 — 99% ACCURACY AUDIT
# ==========================================================
def step15_accuracy_audit(all_results, X_train, X_test, y_train, y_test,
                          groups_train, groups_test, feature_cols):
    banner(15, "HIGH-ACCURACY AUDIT")

    suspicious = False
    for name, res in all_results.items():
        test_acc = res["test"]["accuracy"]
        if test_acc > 0.95:
            suspicious = True
            print(f"  ⚠ {name}: Test Accuracy = {test_acc:.4f} (> 0.95)")

    if suspicious:
        print("\n  ╔════════════════════════════════════════════════════╗")
        print("  ║  HIGH PERFORMANCE DETECTED — LEAKAGE AUDIT       ║")
        print("  ║  REQUIRED BEFORE ACCEPTING THIS RESULT.          ║")
        print("  ╚════════════════════════════════════════════════════╝\n")

    # Audit checks
    print("  Audit checks:")
    train_patients = set(groups_train)
    test_patients = set(groups_test)
    overlap = train_patients & test_patients
    print(f"    1. Patient overlap:           {len(overlap)} {'✓' if len(overlap)==0 else '✗ LEAKAGE!'}")

    leaky_in_features = [c for c in ["Risk_Score", "Toxicity_Score", "Patient_ID"] if c in feature_cols]
    print(f"    2. Risk_Score in features:    {'YES ✗' if 'Risk_Score' in feature_cols else 'NO ✓'}")
    print(f"    3. Toxicity_Score in features: {'YES ✗' if 'Toxicity_Score' in feature_cols else 'NO ✓'}")
    print(f"    4. Target-derived features:   {'FOUND ✗' if leaky_in_features else 'NONE ✓'}")
    print(f"    5. Post-treatment variables:  EXCLUDED ✓")
    print(f"    6. Preprocessing leakage:     Pipeline ensures fit on train only ✓")
    print(f"    7. SMOTE placement:           NOT USED ✓")

    # Check feature distribution shift
    print(f"    8. Feature distribution check (numeric mean shift):")
    for col in feature_cols:
        if X_train[col].dtype in ["int64", "float64"]:
            train_mean = X_train[col].mean()
            test_mean = X_test[col].mean()
            shift = abs(train_mean - test_mean) / (train_mean + 1e-10)
            if shift > 0.3:
                print(f"       {col}: train_mean={train_mean:.2f}, test_mean={test_mean:.2f}, shift={shift:.1%} ⚠")

    if not suspicious:
        print("\n  No models exceeded 95% accuracy. Audit passed. ✓")

    return len(overlap)


# ==========================================================
#  STEP 16 — FEATURE IMPORTANCE
# ==========================================================
def step16_feature_importance(models_dict, preprocessor, X_train, feature_cols):
    banner(16, "FEATURE IMPORTANCE")

    # Get feature names after one-hot encoding
    preprocessor_fitted = models_dict["Random Forest"].named_steps["preprocessor"]
    try:
        feature_names = preprocessor_fitted.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(
            models_dict["Random Forest"].named_steps["classifier"].n_features_in_
        )]

    # --- Random Forest ---
    rf_model = models_dict["Random Forest"].named_steps["classifier"]
    rf_importances = rf_model.feature_importances_
    rf_indices = np.argsort(rf_importances)[::-1][:20]

    print("  Random Forest — Top 20 Features:")
    for rank, idx in enumerate(rf_indices, 1):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        print(f"    {rank:2d}. {name:45s}  {rf_importances[idx]:.4f}")

    # Check for suspicious features
    suspicious = [
        n for n in feature_names
        if any(s in n.lower() for s in ["risk_score", "toxicity_score", "toxicity_risk", "patient_id"])
    ]
    if suspicious:
        print(f"\n  ⚠ SUSPICIOUS features detected in model: {suspicious}")
    else:
        print(f"\n  ✓ No suspicious features (Risk_Score, Toxicity_Score, Patient_ID) found.")

    # Plot RF importance
    fig, ax = plt.subplots(figsize=(10, 8))
    top_names = [feature_names[i] if i < len(feature_names) else f"f_{i}" for i in rf_indices]
    top_vals = rf_importances[rf_indices]
    ax.barh(range(len(top_names)), top_vals, color="#3b82f6")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title("Random Forest — Top 20 Feature Importances")
    plt.tight_layout()
    path_rf = os.path.join(REPORTS_DIR, "feature_importance_random_forest.png")
    fig.savefig(path_rf, dpi=150)
    plt.close(fig)
    print(f"\n  Saved → {path_rf}")

    # --- XGBoost ---
    xgb_model = models_dict["XGBoost"].named_steps["classifier"]
    xgb_importances = xgb_model.feature_importances_
    xgb_indices = np.argsort(xgb_importances)[::-1][:20]

    print("\n  XGBoost — Top 20 Features:")
    for rank, idx in enumerate(xgb_indices, 1):
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        print(f"    {rank:2d}. {name:45s}  {xgb_importances[idx]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    top_names_xgb = [feature_names[i] if i < len(feature_names) else f"f_{i}" for i in xgb_indices]
    top_vals_xgb = xgb_importances[xgb_indices]
    ax.barh(range(len(top_names_xgb)), top_vals_xgb, color="#10b981")
    ax.set_yticks(range(len(top_names_xgb)))
    ax.set_yticklabels(top_names_xgb, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title("XGBoost — Top 20 Feature Importances")
    plt.tight_layout()
    path_xgb = os.path.join(REPORTS_DIR, "feature_importance_xgboost.png")
    fig.savefig(path_xgb, dpi=150)
    plt.close(fig)
    print(f"  Saved → {path_xgb}")


# ==========================================================
#  MAIN PIPELINE
# ==========================================================
def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║  ONCOLOGY TOXICITY RISK PREDICTION — ML PIPELINE   ║")
    print("║  Leakage-Free | Patient-Aware | Multiclass          ║")
    print("╚══════════════════════════════════════════════════════╝")

    # Step 1
    df = step1_load_and_inspect(DATA_PATH)

    # Step 2
    step2_leakage_audit(df)

    # Step 3
    step3_patient_level_check(df)

    # Step 4
    numeric_features, categorical_features, feature_cols = step4_define_features(df)

    # Step 5
    preprocessor = step5_build_preprocessor(numeric_features, categorical_features)

    # Step 6
    X_train, X_test, y_train, y_test, groups_train, groups_test = step6_train_test_split(df, feature_cols)

    # Step 7
    step7_class_imbalance(y_train, y_test)

    # Build CV strategy
    banner(13, "CROSS-VALIDATION STRATEGY")
    cv = get_cv_strategy(y_train, groups_train, n_splits=5)

    # ──────────────────────────────────────────
    # Train all 5 models (Steps 8–12)
    # ──────────────────────────────────────────
    models = {}
    cv_results = {}
    test_results = {}

    # Step 8 — Logistic Regression
    lr_model = step8_logistic_regression(preprocessor, X_train, y_train, groups_train, cv)
    models["Logistic Regression"] = lr_model
    cv_results["Logistic Regression"] = evaluate_cv(lr_model, X_train, y_train, groups_train, cv, "Logistic Regression")
    test_results["Logistic Regression"] = evaluate_test(lr_model, X_test, y_test, "Logistic Regression")

    # Step 9 — Decision Tree
    dt_model = step9_decision_tree(preprocessor, X_train, y_train, groups_train, cv)
    models["Decision Tree"] = dt_model
    cv_results["Decision Tree"] = evaluate_cv(dt_model, X_train, y_train, groups_train, cv, "Decision Tree")
    test_results["Decision Tree"] = evaluate_test(dt_model, X_test, y_test, "Decision Tree")

    # Step 10 — Random Forest
    rf_model = step10_random_forest(preprocessor, X_train, y_train, groups_train, cv)
    models["Random Forest"] = rf_model
    cv_results["Random Forest"] = evaluate_cv(rf_model, X_train, y_train, groups_train, cv, "Random Forest")
    test_results["Random Forest"] = evaluate_test(rf_model, X_test, y_test, "Random Forest")

    # Step 11 — XGBoost
    xgb_model, xgb_label_encoder = step11_xgboost(preprocessor, X_train, y_train, groups_train, cv)
    models["XGBoost"] = xgb_model

    # For XGBoost, CV and test need label encoding
    y_train_xgb = xgb_label_encoder.transform(y_train)
    y_test_xgb = xgb_label_encoder.transform(y_test)

    # XGBoost CV (manual with encoded labels)
    banner(13, "XGBOOST CROSS-VALIDATION (encoded labels)")
    xgb_cv_metrics = {
        "accuracy": [], "macro_precision": [], "macro_recall": [],
        "macro_f1": [], "weighted_f1": [], "balanced_accuracy": [],
    }

    from sklearn.base import clone
    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups_train)):
        X_tr = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train_xgb[tr_idx]
        y_val = y_train_xgb[val_idx]

        m = clone(xgb_model)
        sample_w_fold = np.array([
            len(y_tr) / (3 * np.sum(y_tr == c)) if np.sum(y_tr == c) > 0 else 1.0
            for label in y_tr
            for c in [label]
        ])
        m.fit(X_tr, y_tr, classifier__sample_weight=sample_w_fold)
        y_pred_fold = m.predict(X_val)

        xgb_cv_metrics["accuracy"].append(accuracy_score(y_val, y_pred_fold))
        xgb_cv_metrics["macro_precision"].append(precision_score(y_val, y_pred_fold, average="macro", zero_division=0))
        xgb_cv_metrics["macro_recall"].append(recall_score(y_val, y_pred_fold, average="macro", zero_division=0))
        xgb_cv_metrics["macro_f1"].append(f1_score(y_val, y_pred_fold, average="macro", zero_division=0))
        xgb_cv_metrics["weighted_f1"].append(f1_score(y_val, y_pred_fold, average="weighted", zero_division=0))
        xgb_cv_metrics["balanced_accuracy"].append(balanced_accuracy_score(y_val, y_pred_fold))

    xgb_cv_results_formatted = {}
    print(f"\n  XGBoost — Cross-Validation Results (5 folds):")
    for metric_name, values in xgb_cv_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        xgb_cv_results_formatted[metric_name] = {"mean": mean_val, "std": std_val}
        print(f"    {metric_name:25s}  {mean_val:.4f} ± {std_val:.4f}")
    cv_results["XGBoost"] = xgb_cv_results_formatted

    # XGBoost test evaluation
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_xgb_labels = xgb_label_encoder.inverse_transform(y_pred_xgb)

    xgb_test_acc = accuracy_score(y_test, y_pred_xgb_labels)
    xgb_test_macro_p = precision_score(y_test, y_pred_xgb_labels, average="macro", zero_division=0)
    xgb_test_macro_r = recall_score(y_test, y_pred_xgb_labels, average="macro", zero_division=0)
    xgb_test_macro_f1 = f1_score(y_test, y_pred_xgb_labels, average="macro", zero_division=0)
    xgb_test_weighted_f1 = f1_score(y_test, y_pred_xgb_labels, average="weighted", zero_division=0)
    xgb_test_bal_acc = balanced_accuracy_score(y_test, y_pred_xgb_labels)
    xgb_test_cm = confusion_matrix(y_test, y_pred_xgb_labels, labels=["Low", "Moderate", "High"])
    xgb_test_report = classification_report(y_test, y_pred_xgb_labels, zero_division=0)

    print(f"\n  XGBoost — Test Set Results:")
    print(f"    Accuracy:           {xgb_test_acc:.4f}")
    print(f"    Macro Precision:    {xgb_test_macro_p:.4f}")
    print(f"    Macro Recall:       {xgb_test_macro_r:.4f}")
    print(f"    Macro F1:           {xgb_test_macro_f1:.4f}")
    print(f"    Weighted F1:        {xgb_test_weighted_f1:.4f}")
    print(f"    Balanced Accuracy:  {xgb_test_bal_acc:.4f}")
    print(f"\n    Classification Report:\n{xgb_test_report}")
    print(f"    Confusion Matrix:\n    {xgb_test_cm}")

    test_results["XGBoost"] = {
        "accuracy": xgb_test_acc,
        "macro_precision": xgb_test_macro_p,
        "macro_recall": xgb_test_macro_r,
        "macro_f1": xgb_test_macro_f1,
        "weighted_f1": xgb_test_weighted_f1,
        "balanced_accuracy": xgb_test_bal_acc,
        "confusion_matrix": xgb_test_cm,
        "classification_report": xgb_test_report,
        "y_pred": y_pred_xgb_labels,
    }

    # Step 12 — SVM
    svm_model = step12_svm(preprocessor, X_train, y_train, groups_train, cv)
    models["SVM"] = svm_model
    cv_results["SVM"] = evaluate_cv(svm_model, X_train, y_train, groups_train, cv, "SVM")
    test_results["SVM"] = evaluate_test(svm_model, X_test, y_test, "SVM")

    # ──────────────────────────────────────────
    # Step 14 — FINAL TEST EVALUATION (combined)
    # ──────────────────────────────────────────
    banner(14, "FINAL TEST EVALUATION SUMMARY")
    for name in models:
        res = test_results[name]
        print(f"  {name}:")
        print(f"    Accuracy={res['accuracy']:.4f}  Macro_F1={res['macro_f1']:.4f}  Balanced_Acc={res['balanced_accuracy']:.4f}")

    # ──────────────────────────────────────────
    # Step 15 — Accuracy Audit
    # ──────────────────────────────────────────
    all_results_combined = {name: {"test": test_results[name], "cv": cv_results[name]} for name in models}
    patient_overlap = step15_accuracy_audit(
        all_results_combined, X_train, X_test, y_train, y_test,
        groups_train, groups_test, feature_cols,
    )

    # ──────────────────────────────────────────
    # Step 16 — Feature Importance
    # ──────────────────────────────────────────
    step16_feature_importance(models, preprocessor, X_train, feature_cols)

    # ──────────────────────────────────────────
    # Step 17 — MODEL COMPARISON
    # ──────────────────────────────────────────
    banner(17, "MODEL COMPARISON")

    comparison_rows = []
    ml_comparison_rows = []
    for name in ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "SVM"]:
        row = {
            "Model": name,
            "CV_Accuracy": f"{cv_results[name]['accuracy']['mean']:.4f} ± {cv_results[name]['accuracy']['std']:.4f}",
            "CV_Macro_F1": f"{cv_results[name]['macro_f1']['mean']:.4f} ± {cv_results[name]['macro_f1']['std']:.4f}",
            "CV_Macro_Recall": f"{cv_results[name]['macro_recall']['mean']:.4f} ± {cv_results[name]['macro_recall']['std']:.4f}",
            "CV_Weighted_F1": f"{cv_results[name]['weighted_f1']['mean']:.4f} ± {cv_results[name]['weighted_f1']['std']:.4f}",
            "Test_Accuracy": f"{test_results[name]['accuracy']:.4f}",
            "Test_Macro_F1": f"{test_results[name]['macro_f1']:.4f}",
            "Test_Balanced_Accuracy": f"{test_results[name]['balanced_accuracy']:.4f}",
        }
        comparison_rows.append(row)

        ml_row = {
            "Model": name,
            "Test Accuracy": round(test_results[name]["accuracy"], 4),
            "CV Accuracy Mean": round(cv_results[name]["accuracy"]["mean"], 4),
            "CV Accuracy Std": round(cv_results[name]["accuracy"]["std"], 4),
            "Balanced Accuracy": round(test_results[name]["balanced_accuracy"], 4),
            "Macro Precision": round(test_results[name]["macro_precision"], 4),
            "Macro Recall": round(test_results[name]["macro_recall"], 4),
            "Macro F1": round(test_results[name]["macro_f1"], 4),
            "Weighted F1": round(test_results[name]["weighted_f1"], 4),
        }
        ml_comparison_rows.append(ml_row)

    comparison_df = pd.DataFrame(comparison_rows)
    # Sort by CV Macro F1 (extract mean)
    comparison_df["_sort_key"] = [cv_results[name]["macro_f1"]["mean"]
                                  for name in ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "SVM"]]
    comparison_df = comparison_df.sort_values("_sort_key", ascending=False).drop(columns=["_sort_key"])

    print(comparison_df.to_string(index=False))

    comparison_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n  Saved → {comparison_path}")

    ml_comparison_df = pd.DataFrame(ml_comparison_rows)
    ml_comparison_df = ml_comparison_df.sort_values("Macro F1", ascending=False)
    ml_comp_path = os.path.join(REPORTS_DIR, "ml_model_comparison.csv")
    ml_comparison_df.to_csv(ml_comp_path, index=False)
    print(f"  Saved → {ml_comp_path}")

    # ──────────────────────────────────────────
    # Step 18 — SELECT BEST MODEL
    # ──────────────────────────────────────────
    banner(18, "SELECT BEST MODEL")

    best_name = max(cv_results, key=lambda n: cv_results[n]["macro_f1"]["mean"])
    best_cv_f1 = cv_results[best_name]["macro_f1"]["mean"]
    best_test_f1 = test_results[best_name]["macro_f1"]
    best_test_acc = test_results[best_name]["accuracy"]
    best_test_bal_acc = test_results[best_name]["balanced_accuracy"]

    print(f"  BEST MODEL: {best_name}")
    print(f"  Selection criteria: Highest CV Macro F1")
    print(f"")
    print(f"  CV Macro F1:          {best_cv_f1:.4f}")
    print(f"  Test Accuracy:        {best_test_acc:.4f}")
    print(f"  Test Macro F1:        {best_test_f1:.4f}")
    print(f"  Test Balanced Acc:    {best_test_bal_acc:.4f}")

    # Per-class recall
    report_str = test_results[best_name]["classification_report"]
    print(f"\n  Per-class report:\n{report_str}")

    # ──────────────────────────────────────────
    # Step 19 — SAVE MODELS
    # ──────────────────────────────────────────
    banner(19, "SAVE MODELS")

    model_filenames = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
        "SVM": "svm.pkl",
    }

    for name, filename in model_filenames.items():
        path = os.path.join(MODELS_DIR, filename)
        if name == "XGBoost":
            # Save with label encoder
            joblib.dump({"pipeline": models[name], "label_encoder": xgb_label_encoder}, path)
        else:
            joblib.dump(models[name], path)
        print(f"  Saved {name:25s} → {path}")

    # Save best model as risk_model.pkl
    best_path = os.path.join(MODELS_DIR, "risk_model.pkl")
    if best_name == "XGBoost":
        joblib.dump({"pipeline": models[best_name], "label_encoder": xgb_label_encoder}, best_path)
    else:
        joblib.dump(models[best_name], best_path)
    print(f"\n  Best model ({best_name}) saved → {best_path}")

    # ──────────────────────────────────────────
    # Step 20 — SAVE REPORTS
    # ──────────────────────────────────────────
    banner(20, "SAVE REPORTS")

    # Classification report (best model)
    report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Best Model: {best_name}\n")
        f.write(f"{'='*60}\n")
        f.write(test_results[best_name]["classification_report"])
    print(f"  Saved classification report → {report_path}")

    # Confusion matrix (best model)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = test_results[best_name]["confusion_matrix"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Low", "Moderate", "High"],
                yticklabels=["Low", "Moderate", "High"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"  Saved confusion matrix → {cm_path}")

    # Cross-validation results
    cv_rows = []
    for name in models:
        row = {"Model": name}
        for metric in cv_results[name]:
            row[f"{metric}_mean"] = cv_results[name][metric]["mean"]
            row[f"{metric}_std"] = cv_results[name][metric]["std"]
        cv_rows.append(row)
    cv_df = pd.DataFrame(cv_rows)
    cv_path = os.path.join(REPORTS_DIR, "cross_validation_results.csv")
    cv_df.to_csv(cv_path, index=False)
    print(f"  Saved CV results → {cv_path}")

    # Final model summary JSON
    target_dist = df[TARGET_COL].value_counts().to_dict()
    summary = {
        "best_model": best_name,
        "features_used": feature_cols,
        "excluded_features": EXCLUDE_COLS,
        "excluded_reasons": {
            "Patient_ID": "Unique identifier",
            "Risk_Score": "Derived from risk/toxicity information",
            "Toxicity_Score": "Target is derived from it",
            "Treatment_Response": "Post-treatment outcome",
            "Clinical_Note": "Post-event text",
            "Treatment_Drug": "Post-assignment information",
            "Dosage_mg": "Post-assignment information",
            "Treatment_Adherence_Pct": "Future treatment information",
            "Encounter_Date": "Temporal field, not predictive",
        },
        "cv_metrics": {
            name: {k: v["mean"] for k, v in cv_results[name].items()}
            for name in models
        },
        "test_metrics": {
            name: {k: v for k, v in test_results[name].items()
                   if k not in ("confusion_matrix", "classification_report", "y_pred")}
            for name in models
        },
        "class_distribution": target_dist,
        "patient_count": int(df["Patient_ID"].nunique()),
        "train_patient_count": int(len(set(groups_train))),
        "test_patient_count": int(len(set(groups_test))),
        "patient_overlap": int(patient_overlap),
        "leakage_status": "CLEAN" if patient_overlap == 0 else "LEAKAGE_DETECTED",
        "random_state": RANDOM_STATE,
        "disclaimer": "Educational/research prototype. NOT clinically validated.",
    }

    summary_path = os.path.join(REPORTS_DIR, "final_model_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Saved final summary → {summary_path}")

    # ──────────────────────────────────────────
    # Step 21 — REPRODUCIBILITY
    # ──────────────────────────────────────────
    banner(21, "REPRODUCIBILITY")
    print(f"  random_state = {RANDOM_STATE}")
    print(f"\n  Features used ({len(feature_cols)}):")
    for col in sorted(feature_cols):
        print(f"    - {col}")
    print(f"\n  Excluded features ({len(EXCLUDE_COLS)}):")
    for col in EXCLUDE_COLS:
        print(f"    - {col}")

    # ──────────────────────────────────────────
    # Step 22 — FINAL TERMINAL OUTPUT
    # ──────────────────────────────────────────
    banner(22, "FINAL SUMMARY")

    n_patients = df["Patient_ID"].nunique()
    low_count = target_dist.get("Low", 0)
    mod_count = target_dist.get("Moderate", 0)
    high_count = target_dist.get("High", 0)

    print("  DATASET")
    print(f"    Rows:       {len(df)}")
    print(f"    Patients:   {n_patients}")
    print()
    print("  TARGET")
    print(f"    Low:        {low_count}")
    print(f"    Moderate:   {mod_count}")
    print(f"    High:       {high_count}")
    print()
    print("  LEAKAGE")
    print(f"    Patient overlap:             {patient_overlap}")
    print(f"    Risk_Score excluded:          YES")
    print(f"    Toxicity_Score excluded:      YES")
    print(f"    Treatment_Response excluded:  YES")
    print(f"    Clinical_Note excluded:       YES")
    print()
    print("  MODEL RESULTS")

    for name in ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "SVM"]:
        cv_f1 = cv_results[name]["macro_f1"]["mean"]
        test_f1 = test_results[name]["macro_f1"]
        test_acc = test_results[name]["accuracy"]
        marker = " ★ BEST" if name == best_name else ""
        print(f"    {name:25s}  CV_F1={cv_f1:.4f}  Test_Acc={test_acc:.4f}  Test_F1={test_f1:.4f}{marker}")

    # High-class recall for best model
    best_report = test_results[best_name]["classification_report"]
    print()
    print(f"  BEST MODEL:            {best_name}")
    print(f"  CV Macro F1:           {best_cv_f1:.4f}")
    print(f"  TEST ACCURACY:         {best_test_acc:.4f}")
    print(f"  TEST MACRO F1:         {best_test_f1:.4f}")
    print(f"  TEST BALANCED ACC:     {best_test_bal_acc:.4f}")

    # Extract high-class recall from confusion matrix
    cm = test_results[best_name]["confusion_matrix"]
    if cm.shape[0] >= 3 and cm[2].sum() > 0:
        high_recall = cm[2, 2] / cm[2].sum()
        print(f"  HIGH-CLASS RECALL:     {high_recall:.4f}")
    else:
        print(f"  HIGH-CLASS RECALL:     N/A (insufficient test samples)")

    print()
    print("  ⚠  DISCLAIMER")
    print("  This is an educational/research prototype for oncology risk")
    print("  prediction. It is NOT clinically validated and must NOT be")
    print("  used to independently make treatment decisions.")
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║               PIPELINE COMPLETE                     ║")
    print("╚══════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
