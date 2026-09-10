"""
Stage 03 NLP Pipeline - Unified Evaluation & Model Comparison
Evaluates all models on the exact same test set across:
- Accuracy
- Macro Precision
- Macro Recall
- Macro F1 (Primary Metric)
- Weighted F1
- Balanced Accuracy
- Per-class Precision, Recall, F1 with High Recall highlighted
Generates confusion matrices and model comparison artifacts.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

CLASSES = ["Low", "Moderate", "High"]


def calculate_metrics(y_true: List[str], y_pred: List[str], model_name: str) -> Dict[str, Any]:
    """Computes comprehensive metrics for multiclass classification."""
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0
    )
    
    # Per class metrics
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=CLASSES, average=None, zero_division=0
    )
    
    per_class = {}
    for idx, cls_name in enumerate(CLASSES):
        per_class[cls_name] = {
            "precision": float(round(p_per[idx], 4)),
            "recall": float(round(r_per[idx], 4)),
            "f1": float(round(f1_per[idx], 4)),
            "support": int(sup_per[idx])
        }
        
    metrics = {
        "Model": model_name,
        "Accuracy": float(round(acc, 4)),
        "Macro_Precision": float(round(macro_p, 4)),
        "Macro_Recall": float(round(macro_r, 4)),
        "Macro_F1": float(round(macro_f1, 4)),
        "Weighted_F1": float(round(weighted_f1, 4)),
        "Balanced_Accuracy": float(round(bal_acc, 4)),
        "High_Recall": per_class["High"]["recall"],
        "Per_Class": per_class
    }
    
    return metrics


def plot_confusion_matrix(y_true: List[str], y_pred: List[str], model_name: str, save_path: str):
    """Generates and saves a clean confusion matrix heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=CLASSES, 
        yticklabels=CLASSES,
        cbar=True
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Urgency Label", fontsize=10, labelpad=8)
    plt.ylabel("Actual Urgency Label", fontsize=10, labelpad=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[EVALUATE] Confusion matrix for {model_name} saved to: {save_path}")


def create_model_comparison_table(
    all_metrics: List[Dict[str, Any]], 
    output_csv: str = "stage03_nlp/reports/model_comparison.csv",
    output_md: str = "stage03_nlp/reports/model_comparison.md"
) -> pd.DataFrame:
    """Generates comparison table sorted strictly by Macro F1."""
    rows = []
    for m in all_metrics:
        rows.append({
            "Model": m["Model"],
            "Accuracy": m["Accuracy"],
            "Macro_Precision": m["Macro_Precision"],
            "Macro_Recall": m["Macro_Recall"],
            "Macro_F1": m["Macro_F1"],
            "Weighted_F1": m["Weighted_F1"],
            "Balanced_Accuracy": m["Balanced_Accuracy"]
        })
        
    df_comp = pd.DataFrame(rows).sort_values(by="Macro_F1", ascending=False)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_comp.to_csv(output_csv, index=False)
    print(f"[EVALUATE] Model comparison CSV saved to: {output_csv}")
    
    # Generate Markdown table
    md_lines = [
        "# Model Comparison: Clinical Urgency Classification",
        "",
        "Evaluation performed on identical test set. Primary ranking metric: **Macro F1**.",
        "",
        df_comp.to_markdown(index=False),
        "",
        "## Per-Class Breakdown & High-Urgency Recall",
        "| Model | Low F1 | Moderate F1 | High F1 | **High Recall** |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]
    for m in all_metrics:
        pc = m["Per_Class"]
        md_lines.append(
            f"| **{m['Model']}** | {pc['Low']['f1']:.4f} | {pc['Moderate']['f1']:.4f} | {pc['High']['f1']:.4f} | **{pc['High']['recall']:.4f}** |"
        )
        
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[EVALUATE] Model comparison Markdown saved to: {output_md}")
    
    return df_comp
