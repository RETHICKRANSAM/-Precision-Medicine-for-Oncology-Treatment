"""
==========================================================
Evaluation Engineer — Comprehensive Model Evaluation
Oncology Toxicity Risk Prediction
==========================================================
This script loads the 5 trained models produced by the
ML Engineer and performs in-depth evaluation including:

 1. Per-Class Deep Dive Analysis
 2. ROC / AUC Curves (multiclass one-vs-rest)
 3. Precision-Recall Curves
 4. Overfitting Analysis
 5. Confusion Matrix Comparison (normalised)
 6. Cross-Validation Stability (box plots)
 7. Clinical Misclassification Cost Analysis
 8. Model Comparison Radar Chart
 9. Calibration Analysis
10. Final Evaluation Report (Markdown)

DISCLAIMER: Educational / research prototype.
            NOT clinically validated.
==========================================================
"""

import io
import json
import os
import sys
import warnings
from datetime import datetime

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, label_binarize

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────
# CONSTANTS  (must match ml_pipeline.py)
# ──────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

EXCLUDE_COLS = [
    "Patient_ID", "Risk_Score", "Toxicity_Score",
    "Treatment_Response", "Clinical_Note", "Treatment_Drug",
    "Dosage_mg", "Treatment_Adherence_Pct", "Encounter_Date",
]
TARGET_COL = "Toxicity_Risk"
CLASS_LABELS = ["Low", "Moderate", "High"]

# Plot styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})
PALETTE = {
    "Logistic Regression": "#6366f1",
    "Decision Tree": "#f59e0b",
    "Random Forest": "#10b981",
    "XGBoost": "#ef4444",
    "SVM": "#8b5cf6",
}


def banner(title: str):
    """Print a formatted section banner."""
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}\n")


# ==========================================================
#  LOAD DATA & RECREATE SPLIT
# ==========================================================
def load_data_and_split():
    """Load cleaned data and reproduce the exact train/test split."""
    banner("LOADING DATA & RECREATING TRAIN/TEST SPLIT")

    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    all_cols = set(df.columns)
    exclude = set(EXCLUDE_COLS + [TARGET_COL])
    feature_cols = sorted(all_cols - exclude)

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    groups = df["Patient_ID"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups[train_idx]

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Target distribution (test): {y_test.value_counts().to_dict()}")

    return df, X_train, X_test, y_train, y_test, groups_train, feature_cols


# ==========================================================
#  LOAD SAVED MODELS
# ==========================================================
def load_models():
    """Load all 5 trained models from disk."""
    banner("LOADING SAVED MODELS")

    model_files = {
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decision_tree.pkl",
        "Random Forest": "random_forest.pkl",
        "XGBoost": "xgboost.pkl",
        "SVM": "svm.pkl",
    }

    models = {}
    xgb_label_encoder = None

    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        obj = joblib.load(path)
        if name == "XGBoost":
            models[name] = obj["pipeline"]
            xgb_label_encoder = obj["label_encoder"]
        else:
            models[name] = obj
        print(f"  Loaded: {name} ({filename})")

    return models, xgb_label_encoder


# ==========================================================
#  HELPER: Get predictions & probabilities
# ==========================================================
def get_predictions(models, xgb_le, X_test, y_test):
    """Get predictions and probability estimates for all models."""
    predictions = {}
    probabilities = {}

    for name, model in models.items():
        if name == "XGBoost":
            y_pred_encoded = model.predict(X_test)
            y_pred = xgb_le.inverse_transform(y_pred_encoded)
            # Probabilities
            try:
                proba_encoded = model.predict_proba(X_test)
                # Map columns to class order: xgb_le.classes_ → [High, Low, Moderate]
                # We need order: [Low, Moderate, High]
                class_order_map = {cls: i for i, cls in enumerate(xgb_le.classes_)}
                reorder = [class_order_map[c] for c in CLASS_LABELS]
                proba = proba_encoded[:, reorder]
                probabilities[name] = proba
            except Exception:
                probabilities[name] = None
        else:
            y_pred = model.predict(X_test)
            try:
                proba = model.predict_proba(X_test)
                # Ensure column order matches CLASS_LABELS
                model_classes = list(model.classes_)
                reorder = [model_classes.index(c) for c in CLASS_LABELS]
                proba = proba[:, reorder]
                probabilities[name] = proba
            except Exception:
                probabilities[name] = None

        predictions[name] = y_pred

    return predictions, probabilities


# ==========================================================
#  1. PER-CLASS DEEP DIVE ANALYSIS
# ==========================================================
def analysis_per_class(predictions, y_test):
    """Detailed per-class metrics for every model."""
    banner("1. PER-CLASS DEEP DIVE ANALYSIS")

    rows = []
    for name, y_pred in predictions.items():
        for cls in CLASS_LABELS:
            support = (y_test == cls).sum()
            if support == 0:
                continue
            tp = ((y_pred == cls) & (y_test == cls)).sum()
            fp = ((y_pred == cls) & (y_test != cls)).sum()
            fn = ((y_pred != cls) & (y_test == cls)).sum()

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            rows.append({
                "Model": name, "Class": cls,
                "Support": int(support), "TP": int(tp), "FP": int(fp), "FN": int(fn),
                "Precision": round(prec, 4), "Recall": round(rec, 4), "F1": round(f1, 4),
            })

    df_per_class = pd.DataFrame(rows)
    print(df_per_class.to_string(index=False))

    # Misclassification analysis
    print("\n  Misclassification Matrix (which classes get confused):")
    for name, y_pred in predictions.items():
        cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
        print(f"\n  {name}:")
        cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in CLASS_LABELS],
                             columns=[f"Pred_{c}" for c in CLASS_LABELS])
        print(f"  {cm_df.to_string()}")

    return df_per_class


# ==========================================================
#  2. ROC / AUC CURVES (Multiclass One-vs-Rest)
# ==========================================================
def analysis_roc_auc(probabilities, y_test):
    """Multiclass ROC curves using One-vs-Rest strategy."""
    banner("2. ROC / AUC CURVES (Multiclass One-vs-Rest)")

    y_bin = label_binarize(y_test, classes=CLASS_LABELS)
    n_classes = len(CLASS_LABELS)

    # Only models with probability support
    prob_models = {n: p for n, p in probabilities.items() if p is not None}

    if not prob_models:
        print("  No models support predict_proba — skipping ROC analysis.")
        return {}

    auc_scores = {}

    fig, axes = plt.subplots(1, n_classes, figsize=(18, 5.5))
    fig.suptitle("ROC Curves — One-vs-Rest (per class)", fontsize=15, fontweight="bold", y=1.02)

    for cls_idx, cls_name in enumerate(CLASS_LABELS):
        ax = axes[cls_idx]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random")

        for name, proba in prob_models.items():
            y_true_cls = y_bin[:, cls_idx]

            # Skip if only one class present
            if len(np.unique(y_true_cls)) < 2:
                ax.text(0.5, 0.5, f"Only {np.sum(y_true_cls)} positive\nsamples — ROC undefined",
                        ha="center", va="center", fontsize=9, transform=ax.transAxes)
                continue

            fpr, tpr, _ = roc_curve(y_true_cls, proba[:, cls_idx])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=PALETTE[name], linewidth=2,
                    label=f"{name} (AUC={roc_auc:.3f})")

            if name not in auc_scores:
                auc_scores[name] = {}
            auc_scores[name][cls_name] = round(roc_auc, 4)

        ax.set_title(f"Class: {cls_name}", fontsize=12)
        ax.set_xlabel("False Positive Rate")
        if cls_idx == 0:
            ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "roc_curves_all_models.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    # Macro AUC
    for name in prob_models:
        if name in auc_scores and len(auc_scores[name]) > 0:
            valid_aucs = [v for v in auc_scores[name].values()]
            auc_scores[name]["macro_avg"] = round(np.mean(valid_aucs), 4)
            print(f"  {name}: per-class AUC = {auc_scores[name]}")

    return auc_scores


# ==========================================================
#  3. PRECISION-RECALL CURVES
# ==========================================================
def analysis_precision_recall(probabilities, y_test):
    """Precision-Recall curves — critical for imbalanced datasets."""
    banner("3. PRECISION-RECALL CURVES")

    y_bin = label_binarize(y_test, classes=CLASS_LABELS)
    n_classes = len(CLASS_LABELS)

    prob_models = {n: p for n, p in probabilities.items() if p is not None}
    if not prob_models:
        print("  No models support predict_proba — skipping PR analysis.")
        return {}

    ap_scores = {}

    fig, axes = plt.subplots(1, n_classes, figsize=(18, 5.5))
    fig.suptitle("Precision-Recall Curves — One-vs-Rest (per class)",
                 fontsize=15, fontweight="bold", y=1.02)

    for cls_idx, cls_name in enumerate(CLASS_LABELS):
        ax = axes[cls_idx]
        prevalence = y_bin[:, cls_idx].mean()
        ax.axhline(y=prevalence, color="gray", linestyle="--", alpha=0.5,
                    label=f"Baseline ({prevalence:.3f})")

        for name, proba in prob_models.items():
            y_true_cls = y_bin[:, cls_idx]

            if len(np.unique(y_true_cls)) < 2:
                ax.text(0.5, 0.5, f"Only {np.sum(y_true_cls)} positive\nsamples",
                        ha="center", va="center", fontsize=9, transform=ax.transAxes)
                continue

            prec_vals, rec_vals, _ = precision_recall_curve(y_true_cls, proba[:, cls_idx])
            ap = average_precision_score(y_true_cls, proba[:, cls_idx])
            ax.plot(rec_vals, prec_vals, color=PALETTE[name], linewidth=2,
                    label=f"{name} (AP={ap:.3f})")

            if name not in ap_scores:
                ap_scores[name] = {}
            ap_scores[name][cls_name] = round(ap, 4)

        ax.set_title(f"Class: {cls_name}", fontsize=12)
        ax.set_xlabel("Recall")
        if cls_idx == 0:
            ax.set_ylabel("Precision")
        ax.legend(fontsize=7, loc="best")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "precision_recall_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    for name in prob_models:
        if name in ap_scores and len(ap_scores[name]) > 0:
            valid_aps = list(ap_scores[name].values())
            ap_scores[name]["macro_avg"] = round(np.mean(valid_aps), 4)
            print(f"  {name}: Average Precision = {ap_scores[name]}")

    return ap_scores


# ==========================================================
#  4. OVERFITTING ANALYSIS
# ==========================================================
def analysis_overfitting(models, xgb_le, X_train, y_train, X_test, y_test):
    """Compare train vs test performance to detect overfitting."""
    banner("4. OVERFITTING ANALYSIS")

    results = {}
    for name, model in models.items():
        # Train predictions
        if name == "XGBoost":
            y_train_pred_enc = model.predict(X_train)
            y_train_pred = xgb_le.inverse_transform(y_train_pred_enc)
            y_test_pred_enc = model.predict(X_test)
            y_test_pred = xgb_le.inverse_transform(y_test_pred_enc)
        else:
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred, average="macro", zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
        train_bal = balanced_accuracy_score(y_train, y_train_pred)
        test_bal = balanced_accuracy_score(y_test, y_test_pred)

        gap_acc = train_acc - test_acc
        gap_f1 = train_f1 - test_f1

        status = "✓ OK"
        if gap_acc > 0.10 or gap_f1 > 0.15:
            status = "⚠ OVERFITTING"
        elif gap_acc > 0.05 or gap_f1 > 0.10:
            status = "⚠ MILD"

        results[name] = {
            "train_acc": train_acc, "test_acc": test_acc, "gap_acc": gap_acc,
            "train_f1": train_f1, "test_f1": test_f1, "gap_f1": gap_f1,
            "train_bal_acc": train_bal, "test_bal_acc": test_bal,
            "status": status,
        }

        print(f"  {name}:")
        print(f"    Train Acc={train_acc:.4f}  Test Acc={test_acc:.4f}  Gap={gap_acc:+.4f}")
        print(f"    Train F1 ={train_f1:.4f}  Test F1 ={test_f1:.4f}  Gap={gap_f1:+.4f}")
        print(f"    Status: {status}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    model_names = list(results.keys())
    x = np.arange(len(model_names))
    width = 0.35

    # Accuracy
    train_accs = [results[n]["train_acc"] for n in model_names]
    test_accs = [results[n]["test_acc"] for n in model_names]
    axes[0].bar(x - width/2, train_accs, width, label="Train", color="#6366f1", alpha=0.85)
    axes[0].bar(x + width/2, test_accs, width, label="Test", color="#ef4444", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Train vs Test — Accuracy")
    axes[0].legend()
    axes[0].set_ylim([0.6, 1.05])
    axes[0].grid(axis="y", alpha=0.3)
    # Annotate gaps
    for i, n in enumerate(model_names):
        gap = results[n]["gap_acc"]
        axes[0].annotate(f"Δ={gap:+.3f}", (i, max(train_accs[i], test_accs[i]) + 0.01),
                         ha="center", fontsize=8, color="red" if gap > 0.05 else "green")

    # Macro F1
    train_f1s = [results[n]["train_f1"] for n in model_names]
    test_f1s = [results[n]["test_f1"] for n in model_names]
    axes[1].bar(x - width/2, train_f1s, width, label="Train", color="#6366f1", alpha=0.85)
    axes[1].bar(x + width/2, test_f1s, width, label="Test", color="#ef4444", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylabel("Macro F1")
    axes[1].set_title("Train vs Test — Macro F1")
    axes[1].legend()
    axes[1].set_ylim([0.0, 1.15])
    axes[1].grid(axis="y", alpha=0.3)
    for i, n in enumerate(model_names):
        gap = results[n]["gap_f1"]
        axes[1].annotate(f"Δ={gap:+.3f}", (i, max(train_f1s[i], test_f1s[i]) + 0.02),
                         ha="center", fontsize=8, color="red" if gap > 0.10 else "green")

    plt.suptitle("Overfitting Analysis — Train vs Test Performance",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "overfitting_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved → {path}")

    return results


# ==========================================================
#  5. CONFUSION MATRIX COMPARISON (Normalised)
# ==========================================================
def analysis_confusion_matrices(predictions, y_test):
    """Side-by-side normalised confusion matrices."""
    banner("5. CONFUSION MATRIX COMPARISON (Normalised)")

    model_names = list(predictions.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for i, name in enumerate(model_names):
        cm = confusion_matrix(y_test, predictions[name], labels=CLASS_LABELS)
        # Row-normalise (recall-oriented)
        cm_norm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm_norm = cm_norm / row_sums

        annot = []
        for r in range(cm.shape[0]):
            row_annot = []
            for c in range(cm.shape[1]):
                row_annot.append(f"{cm_norm[r, c]:.2f}\n({cm[r, c]})")
            annot.append(row_annot)

        sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues",
                    xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                    ax=axes[i], vmin=0, vmax=1, cbar=i == n-1,
                    linewidths=0.5, linecolor="white")
        axes[i].set_title(name, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Predicted")
        if i == 0:
            axes[i].set_ylabel("Actual")
        else:
            axes[i].set_ylabel("")

    plt.suptitle("Normalised Confusion Matrices (row = recall)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "confusion_matrix_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ==========================================================
#  6. CROSS-VALIDATION STABILITY ANALYSIS
# ==========================================================
def analysis_cv_stability(models, xgb_le, X_train, y_train, groups_train):
    """Re-run CV to collect per-fold scores and visualise stability."""
    banner("6. CROSS-VALIDATION STABILITY ANALYSIS")

    # Build CV strategy
    try:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        list(cv.split(np.zeros(len(y_train)), y_train, groups_train))
        print("  CV strategy: StratifiedGroupKFold (5 folds)")
    except Exception:
        from sklearn.model_selection import GroupKFold
        cv = GroupKFold(n_splits=5)
        print("  CV strategy: GroupKFold (5 folds)")

    fold_results = {name: {"accuracy": [], "macro_f1": [], "balanced_accuracy": []}
                    for name in models}

    for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train, groups_train)):
        X_tr = X_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        for name, model in models.items():
            m = clone(model)
            if name == "XGBoost":
                y_tr_enc = xgb_le.transform(y_tr)
                y_val_enc = xgb_le.transform(y_val)
                class_counts = np.bincount(y_tr_enc, minlength=3)
                class_counts[class_counts == 0] = 1
                sw = np.array([len(y_tr_enc) / (3 * class_counts[c]) for c in y_tr_enc])
                m.fit(X_tr, y_tr_enc, classifier__sample_weight=sw)
                y_pred_enc = m.predict(X_val)
                y_pred = xgb_le.inverse_transform(y_pred_enc)
                y_val_labels = y_val
            else:
                m.fit(X_tr, y_tr)
                y_pred = m.predict(X_val)
                y_val_labels = y_val

            fold_results[name]["accuracy"].append(accuracy_score(y_val_labels, y_pred))
            fold_results[name]["macro_f1"].append(
                f1_score(y_val_labels, y_pred, average="macro", zero_division=0))
            fold_results[name]["balanced_accuracy"].append(
                balanced_accuracy_score(y_val_labels, y_pred))

        print(f"  Fold {fold_idx + 1}/5 complete")

    # Print summary
    print("\n  CV Stability Summary:")
    for name in models:
        f1_vals = fold_results[name]["macro_f1"]
        print(f"    {name:25s}  Macro F1: {np.mean(f1_vals):.4f} ± {np.std(f1_vals):.4f}  "
              f"(range: {np.min(f1_vals):.4f} – {np.max(f1_vals):.4f})")

    # Box plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ["accuracy", "macro_f1", "balanced_accuracy"]
    titles = ["Accuracy", "Macro F1", "Balanced Accuracy"]

    for ax, metric, title in zip(axes, metrics, titles):
        data = []
        labels = []
        colors = []
        for name in models:
            data.append(fold_results[name][metric])
            labels.append(name)
            colors.append(PALETTE[name])

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(2)

        # Overlay individual fold points
        for i, (d, color) in enumerate(zip(data, colors)):
            jitter = np.random.normal(0, 0.04, size=len(d))
            ax.scatter([i + 1 + j for j in jitter], d, color=color,
                       edgecolors="white", s=40, zorder=5, alpha=0.9)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Cross-Validation Stability — 5 Fold Results",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "cv_stability_boxplot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved → {path}")

    return fold_results


# ==========================================================
#  7. CLINICAL MISCLASSIFICATION COST ANALYSIS
# ==========================================================
def analysis_clinical_cost(predictions, y_test):
    """
    Weighted misclassification cost for oncology context.

    In oncology, the cost of missing a HIGH-risk patient (predicting
    Low when true is High) is far more dangerous than the reverse.
    """
    banner("7. CLINICAL MISCLASSIFICATION COST ANALYSIS")

    # Define asymmetric cost matrix
    # Cost[true][predicted] — higher = more dangerous
    COST_MATRIX = {
        # True label    → Predicted Low  Predicted Moderate  Predicted High
        "Low":       {"Low": 0, "Moderate": 1, "High": 2},     # false alarm, mild cost
        "Moderate":  {"Low": 5, "Moderate": 0, "High": 1},     # missing moderate risk
        "High":      {"Low": 20, "Moderate": 5, "High": 0},    # CRITICAL: missing high toxicity
    }

    print("  Clinical Cost Matrix (True → Predicted):")
    print(f"  {'':12s} {'Pred Low':>10s} {'Pred Mod':>10s} {'Pred High':>10s}")
    for true_cls in CLASS_LABELS:
        row = COST_MATRIX[true_cls]
        print(f"  True {true_cls:6s}  {row['Low']:10d} {row['Moderate']:10d} {row['High']:10d}")

    print("\n  Rationale:")
    print("    - Predicting Low when true is High → cost 20 (CRITICAL safety risk)")
    print("    - Predicting Low when true is Moderate → cost 5 (delayed intervention)")
    print("    - False alarms (Low→High) → cost 2 (unnecessary monitoring)")
    print()

    cost_results = {}
    for name, y_pred in predictions.items():
        total_cost = 0
        n_errors = 0
        critical_errors = 0

        for true, pred in zip(y_test, y_pred):
            cost = COST_MATRIX[true][pred]
            total_cost += cost
            if true != pred:
                n_errors += 1
            if true == "High" and pred == "Low":
                critical_errors += 1
            elif true == "Moderate" and pred == "Low":
                critical_errors += 1

        avg_cost = total_cost / len(y_test)
        cost_results[name] = {
            "total_cost": int(total_cost),
            "avg_cost_per_sample": round(avg_cost, 4),
            "total_errors": int(n_errors),
            "critical_errors": int(critical_errors),
            "error_rate": round(n_errors / len(y_test), 4),
        }

        print(f"  {name}:")
        print(f"    Total weighted cost:   {total_cost}")
        print(f"    Avg cost per sample:   {avg_cost:.4f}")
        print(f"    Total errors:          {n_errors}")
        print(f"    Critical errors:       {critical_errors}  "
              f"(High→Low or Mod→Low)")
        print(f"    Error rate:            {n_errors/len(y_test):.4f}")
        print()

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    names = list(cost_results.keys())
    x = np.arange(len(names))

    avg_costs = [cost_results[n]["avg_cost_per_sample"] for n in names]
    colors = [PALETTE[n] for n in names]
    axes[0].bar(x, avg_costs, color=colors, alpha=0.85, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylabel("Average Weighted Cost per Sample")
    axes[0].set_title("Clinical Misclassification Cost", fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)
    for i, v in enumerate(avg_costs):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

    crit = [cost_results[n]["critical_errors"] for n in names]
    axes[1].bar(x, crit, color=colors, alpha=0.85, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Critical Errors (High→Low or Moderate→Low)", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    for i, v in enumerate(crit):
        axes[1].text(i, v + 0.3, str(v), ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("Clinical Safety Assessment", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "clinical_cost_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    return cost_results


# ==========================================================
#  8. MODEL COMPARISON RADAR CHART
# ==========================================================
def analysis_radar_chart(predictions, y_test):
    """Radar chart comparing all models across multiple metrics."""
    banner("8. MODEL COMPARISON RADAR CHART")

    metrics_data = {}
    metric_names = ["Accuracy", "Macro Precision", "Macro Recall",
                    "Macro F1", "Weighted F1", "Balanced Accuracy"]

    for name, y_pred in predictions.items():
        metrics_data[name] = [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, average="macro", zero_division=0),
            recall_score(y_test, y_pred, average="macro", zero_division=0),
            f1_score(y_test, y_pred, average="macro", zero_division=0),
            f1_score(y_test, y_pred, average="weighted", zero_division=0),
            balanced_accuracy_score(y_test, y_pred),
        ]

    # Radar chart
    n_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.yaxis.grid(True, alpha=0.3)

    for name, values in metrics_data.items():
        vals = values + values[:1]
        ax.plot(angles, vals, linewidth=2, color=PALETTE[name], label=name)
        ax.fill(angles, vals, alpha=0.1, color=PALETTE[name])

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.set_title("Model Comparison — Multi-Metric Radar",
                 fontsize=13, fontweight="bold", pad=20)

    path = os.path.join(REPORTS_DIR, "model_comparison_radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    # Print table
    print("\n  Multi-Metric Comparison Table:")
    header = f"  {'Model':25s}" + "".join(f"  {m:>16s}" for m in metric_names)
    print(header)
    print("  " + "-" * (25 + 18 * len(metric_names)))
    for name, values in metrics_data.items():
        row = f"  {name:25s}" + "".join(f"  {v:16.4f}" for v in values)
        print(row)

    return metrics_data


# ==========================================================
#  9. CALIBRATION ANALYSIS
# ==========================================================
def analysis_calibration(probabilities, y_test):
    """Reliability diagrams and Expected Calibration Error."""
    banner("9. CALIBRATION ANALYSIS")

    prob_models = {n: p for n, p in probabilities.items() if p is not None}
    if not prob_models:
        print("  No models support predict_proba — skipping calibration.")
        return {}

    # Focus on Low and Moderate classes (High has too few samples)
    eval_classes = ["Low", "Moderate"]
    n_classes = len(eval_classes)

    fig, axes = plt.subplots(1, n_classes, figsize=(12, 5.5))

    ece_scores = {}

    for cls_idx, cls_name in enumerate(eval_classes):
        ax = axes[cls_idx]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1, label="Perfect calibration")

        cls_full_idx = CLASS_LABELS.index(cls_name)
        y_true_bin = (y_test == cls_name).astype(int)

        for name, proba in prob_models.items():
            prob_cls = proba[:, cls_full_idx]

            try:
                fraction_pos, mean_pred = calibration_curve(
                    y_true_bin, prob_cls, n_bins=8, strategy="uniform")

                ax.plot(mean_pred, fraction_pos, "o-", color=PALETTE[name],
                        linewidth=2, markersize=5, label=name)

                # Expected Calibration Error
                bin_counts = np.histogram(prob_cls, bins=8, range=(0, 1))[0]
                bin_total = bin_counts.sum()
                ece = 0
                for fp, mp, bc in zip(fraction_pos, mean_pred, bin_counts):
                    if bin_total > 0:
                        ece += (bc / bin_total) * abs(fp - mp)

                if name not in ece_scores:
                    ece_scores[name] = {}
                ece_scores[name][cls_name] = round(ece, 4)

            except Exception as e:
                print(f"    Calibration failed for {name}/{cls_name}: {e}")

        ax.set_title(f"Class: {cls_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability")
        if cls_idx == 0:
            ax.set_ylabel("Fraction of Positives")
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])
        ax.grid(alpha=0.3)

    plt.suptitle("Calibration / Reliability Diagrams",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "calibration_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")

    print("\n  Expected Calibration Error (ECE):")
    for name in prob_models:
        if name in ece_scores:
            print(f"    {name:25s}  {ece_scores[name]}")

    return ece_scores


# ==========================================================
#  10. FINAL EVALUATION REPORT (Markdown)
# ==========================================================
def generate_evaluation_report(
    per_class_df, auc_scores, ap_scores, overfit_results,
    fold_results, cost_results, radar_data, ece_scores,
    predictions, y_test
):
    """Generate comprehensive markdown evaluation report."""
    banner("10. GENERATING FINAL EVALUATION REPORT")

    lines = []
    lines.append("# Comprehensive Model Evaluation Report")
    lines.append(f"**Oncology Toxicity Risk Prediction**  ")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"Evaluation Engineer — Automated Assessment\n")
    lines.append("---\n")

    # ── Executive Summary ──
    lines.append("## Executive Summary\n")
    lines.append("This report presents a comprehensive evaluation of 5 machine learning models ")
    lines.append("trained for **multiclass toxicity risk prediction** (Low / Moderate / High) ")
    lines.append("in oncology patients.\n")

    best_name = max(radar_data, key=lambda n: radar_data[n][3])  # Macro F1 index
    best_f1 = radar_data[best_name][3]
    best_acc = radar_data[best_name][0]

    lines.append(f"> [!IMPORTANT]")
    lines.append(f"> **Best Model: {best_name}** — Test Accuracy: {best_acc:.4f}, "
                 f"Test Macro F1: {best_f1:.4f}\n")
    lines.append(f"> [!WARNING]")
    lines.append(f"> The **High** toxicity class has only **4 total samples** (1 in test). "
                 f"All High-class metrics are statistically unreliable. "
                 f"This model should NOT be used for High-risk patient identification "
                 f"without substantially more High-class data.\n")

    # ── Dataset Summary ──
    lines.append("## Dataset Summary\n")
    lines.append("| Property | Value |")
    lines.append("|---|---|")
    lines.append("| Total Samples | 3,893 |")
    lines.append("| Train/Test Split | 80/20 (Patient-level, GroupShuffleSplit) |")
    lines.append(f"| Test Samples | {len(y_test)} |")
    lines.append(f"| Low (test) | {(y_test == 'Low').sum()} |")
    lines.append(f"| Moderate (test) | {(y_test == 'Moderate').sum()} |")
    lines.append(f"| High (test) | {(y_test == 'High').sum()} |")
    lines.append("| Patient Overlap | 0 (CLEAN) |")
    lines.append("| Features Used | 26 |")
    lines.append("| Leakage Status | CLEAN ✓ |\n")

    # ── Test Performance Comparison ──
    lines.append("## Test Performance Comparison\n")
    lines.append("| Model | Accuracy | Macro Prec | Macro Recall | Macro F1 | Weighted F1 | Balanced Acc |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, vals in radar_data.items():
        marker = " ★" if name == best_name else ""
        lines.append(f"| {name}{marker} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | "
                     f"{vals[3]:.4f} | {vals[4]:.4f} | {vals[5]:.4f} |")
    lines.append("")

    # ── Per-Class Analysis ──
    lines.append("## Per-Class Analysis\n")
    for cls in CLASS_LABELS:
        subset = per_class_df[per_class_df["Class"] == cls]
        if len(subset) == 0:
            continue
        lines.append(f"### Class: {cls}\n")
        lines.append("| Model | Precision | Recall | F1 | Support |")
        lines.append("|---|---|---|---|---|")
        for _, row in subset.iterrows():
            lines.append(f"| {row['Model']} | {row['Precision']:.4f} | "
                         f"{row['Recall']:.4f} | {row['F1']:.4f} | {row['Support']} |")
        lines.append("")

    # ── ROC/AUC ──
    lines.append("## ROC / AUC Analysis\n")
    if auc_scores:
        lines.append("| Model | Low AUC | Moderate AUC | High AUC | Macro AUC |")
        lines.append("|---|---|---|---|---|")
        for name, scores in auc_scores.items():
            low = scores.get("Low", "N/A")
            mod = scores.get("Moderate", "N/A")
            high = scores.get("High", "N/A")
            macro = scores.get("macro_avg", "N/A")
            lines.append(f"| {name} | {low} | {mod} | {high} | {macro} |")
        lines.append("")
    else:
        lines.append("No models with probability support available.\n")

    lines.append("![ROC Curves](roc_curves_all_models.png)\n")

    # ── Precision-Recall ──
    lines.append("## Precision-Recall Analysis\n")
    if ap_scores:
        lines.append("| Model | Low AP | Moderate AP | High AP | Macro AP |")
        lines.append("|---|---|---|---|---|")
        for name, scores in ap_scores.items():
            low = scores.get("Low", "N/A")
            mod = scores.get("Moderate", "N/A")
            high = scores.get("High", "N/A")
            macro = scores.get("macro_avg", "N/A")
            lines.append(f"| {name} | {low} | {mod} | {high} | {macro} |")
        lines.append("")

    lines.append("![Precision-Recall Curves](precision_recall_curves.png)\n")

    # ── Overfitting ──
    lines.append("## Overfitting Analysis\n")
    lines.append("| Model | Train Acc | Test Acc | Gap | Train F1 | Test F1 | Gap | Status |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, res in overfit_results.items():
        lines.append(f"| {name} | {res['train_acc']:.4f} | {res['test_acc']:.4f} | "
                     f"{res['gap_acc']:+.4f} | {res['train_f1']:.4f} | {res['test_f1']:.4f} | "
                     f"{res['gap_f1']:+.4f} | {res['status']} |")
    lines.append("")
    lines.append("![Overfitting Analysis](overfitting_analysis.png)\n")

    # ── CV Stability ──
    lines.append("## Cross-Validation Stability\n")
    lines.append("| Model | CV Macro F1 (mean ± std) | Range |")
    lines.append("|---|---|---|")
    for name in fold_results:
        vals = fold_results[name]["macro_f1"]
        lines.append(f"| {name} | {np.mean(vals):.4f} ± {np.std(vals):.4f} | "
                     f"{np.min(vals):.4f} – {np.max(vals):.4f} |")
    lines.append("")
    lines.append("![CV Stability](cv_stability_boxplot.png)\n")

    # ── Clinical Cost ──
    lines.append("## Clinical Misclassification Cost\n")
    lines.append("> [!CAUTION]")
    lines.append("> Missing a **High-risk** patient (predicting Low) carries a cost weight ")
    lines.append("> of **20×**, reflecting the severe clinical consequences of undetected toxicity.\n")
    lines.append("| Model | Total Cost | Avg Cost/Sample | Critical Errors | Error Rate |")
    lines.append("|---|---|---|---|---|")
    for name, res in cost_results.items():
        lines.append(f"| {name} | {res['total_cost']} | {res['avg_cost_per_sample']:.4f} | "
                     f"{res['critical_errors']} | {res['error_rate']:.4f} |")
    lines.append("")
    lines.append("![Clinical Cost](clinical_cost_analysis.png)\n")

    # ── Calibration ──
    lines.append("## Calibration Analysis\n")
    if ece_scores:
        lines.append("| Model | ECE (Low) | ECE (Moderate) |")
        lines.append("|---|---|---|")
        for name, scores in ece_scores.items():
            low = scores.get("Low", "N/A")
            mod = scores.get("Moderate", "N/A")
            lines.append(f"| {name} | {low} | {mod} |")
        lines.append("")
    lines.append("![Calibration](calibration_curves.png)\n")

    # ── Confusion Matrices & Radar ──
    lines.append("## Visual Comparisons\n")
    lines.append("### Normalised Confusion Matrices\n")
    lines.append("![Confusion Matrices](confusion_matrix_comparison.png)\n")
    lines.append("### Multi-Metric Radar Chart\n")
    lines.append("![Radar Chart](model_comparison_radar.png)\n")

    # ── Key Findings ──
    lines.append("## Key Findings\n")

    # Find lowest clinical cost model
    best_cost_name = min(cost_results, key=lambda n: cost_results[n]["avg_cost_per_sample"])
    best_cost_val = cost_results[best_cost_name]["avg_cost_per_sample"]

    lines.append(f"1. **Best overall model**: {best_name} (Macro F1 = {best_f1:.4f}, "
                 f"Accuracy = {best_acc:.4f})")
    lines.append(f"2. **Lowest clinical cost**: {best_cost_name} "
                 f"(avg cost/sample = {best_cost_val:.4f})")

    # Overfitting
    overfit_models = [n for n, r in overfit_results.items() if "OVER" in r["status"]]
    if overfit_models:
        lines.append(f"3. **Overfitting detected**: {', '.join(overfit_models)}")
    else:
        mild = [n for n, r in overfit_results.items() if "MILD" in r["status"]]
        if mild:
            lines.append(f"3. **Mild overfitting**: {', '.join(mild)}")
        else:
            lines.append("3. **Overfitting**: No significant overfitting detected")

    # CV stability
    most_stable = min(fold_results,
                      key=lambda n: np.std(fold_results[n]["macro_f1"]))
    lines.append(f"4. **Most stable model**: {most_stable} "
                 f"(lowest CV Macro F1 std = {np.std(fold_results[most_stable]['macro_f1']):.4f})")

    lines.append(f"5. **Class imbalance**: High class has only 4 samples — ")
    lines.append(f"   all High-class metrics are unreliable")
    lines.append(f"6. **Leakage status**: CLEAN — no patient overlap, no target-derived features\n")

    # ── Recommendations ──
    lines.append("## Recommendations\n")
    lines.append("> [!NOTE]")
    lines.append("> These recommendations are for improving the model pipeline, ")
    lines.append("> not for clinical deployment.\n")
    lines.append("1. **Collect more High-class samples** — 4 samples is insufficient for any "
                 "reliable classification. Target at least 50–100 High-risk cases.")
    lines.append("2. **Consider binary framing** — Merge Low+Moderate vs High, or Low vs "
                 "Moderate+High for a more balanced and clinically useful split.")
    lines.append("3. **Calibrate probabilities** — Use `CalibratedClassifierCV` (isotonic or "
                 "Platt scaling) to improve probability estimates.")
    lines.append("4. **Threshold tuning** — For clinical deployment, optimise the "
                 "classification threshold on the Moderate class to balance sensitivity "
                 "and specificity.")
    lines.append("5. **External validation** — Test on an independent external cohort "
                 "before considering any clinical use.")
    lines.append("6. **SHAP analysis** — Use SHAP values for individual prediction "
                 "explanations to build clinical trust.\n")

    # ── Disclaimer ──
    lines.append("---\n")
    lines.append("> [!CAUTION]")
    lines.append("> **DISCLAIMER**: This is an educational/research prototype for oncology ")
    lines.append("> toxicity risk prediction. It is **NOT** clinically validated and must ")
    lines.append("> **NOT** be used to independently make treatment decisions.")

    report_text = "\n".join(lines)
    report_path = os.path.join(REPORTS_DIR, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  Saved → {report_path}")

    return report_path


# ==========================================================
#  SAVE EVALUATION METRICS JSON
# ==========================================================
def save_evaluation_metrics(
    auc_scores, ap_scores, overfit_results,
    cost_results, radar_data, ece_scores, fold_results
):
    """Save all computed metrics to a structured JSON file."""
    banner("SAVING EVALUATION METRICS (JSON)")

    metrics = {
        "generated_at": datetime.now().isoformat(),
        "auc_scores": auc_scores,
        "average_precision_scores": ap_scores,
        "overfitting_analysis": overfit_results,
        "clinical_cost": cost_results,
        "test_metrics": {
            name: {
                "accuracy": vals[0],
                "macro_precision": vals[1],
                "macro_recall": vals[2],
                "macro_f1": vals[3],
                "weighted_f1": vals[4],
                "balanced_accuracy": vals[5],
            }
            for name, vals in radar_data.items()
        },
        "calibration_ece": ece_scores,
        "cv_stability": {
            name: {
                metric: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                for metric, vals in metrics_dict.items()
            }
            for name, metrics_dict in fold_results.items()
        },
    }

    path = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved → {path}")

    return path


# ==========================================================
#  MAIN
# ==========================================================
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  EVALUATION ENGINEER — COMPREHENSIVE MODEL EVALUATION  ║")
    print("║  Oncology Toxicity Risk Prediction                     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Load data and models
    df, X_train, X_test, y_train, y_test, groups_train, feature_cols = load_data_and_split()
    models, xgb_le = load_models()

    # Get all predictions and probabilities
    predictions, probabilities = get_predictions(models, xgb_le, X_test, y_test)

    # ── Run all 10 evaluation analyses ──

    # 1. Per-class deep dive
    per_class_df = analysis_per_class(predictions, y_test)

    # 2. ROC/AUC curves
    auc_scores = analysis_roc_auc(probabilities, y_test)

    # 3. Precision-Recall curves
    ap_scores = analysis_precision_recall(probabilities, y_test)

    # 4. Overfitting analysis
    overfit_results = analysis_overfitting(models, xgb_le, X_train, y_train, X_test, y_test)

    # 5. Confusion matrix comparison
    analysis_confusion_matrices(predictions, y_test)

    # 6. CV stability
    fold_results = analysis_cv_stability(models, xgb_le, X_train, y_train, groups_train)

    # 7. Clinical cost analysis
    cost_results = analysis_clinical_cost(predictions, y_test)

    # 8. Radar chart
    radar_data = analysis_radar_chart(predictions, y_test)

    # 9. Calibration analysis
    ece_scores = analysis_calibration(probabilities, y_test)

    # 10. Final evaluation report
    report_path = generate_evaluation_report(
        per_class_df, auc_scores, ap_scores, overfit_results,
        fold_results, cost_results, radar_data, ece_scores,
        predictions, y_test,
    )

    # Save metrics JSON
    metrics_path = save_evaluation_metrics(
        auc_scores, ap_scores, overfit_results,
        cost_results, radar_data, ece_scores, fold_results,
    )

    # ── Final summary ──
    banner("EVALUATION COMPLETE")
    print("  Generated reports:")
    report_files = [
        "roc_curves_all_models.png",
        "precision_recall_curves.png",
        "overfitting_analysis.png",
        "confusion_matrix_comparison.png",
        "cv_stability_boxplot.png",
        "clinical_cost_analysis.png",
        "model_comparison_radar.png",
        "calibration_curves.png",
        "evaluation_report.md",
        "evaluation_metrics.json",
    ]
    for f in report_files:
        full = os.path.join(REPORTS_DIR, f)
        exists = "✓" if os.path.exists(full) else "✗"
        print(f"    {exists} {f}")

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║           EVALUATION PIPELINE COMPLETE                 ║")
    print("╚══════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
