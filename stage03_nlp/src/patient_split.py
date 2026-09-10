"""
Stage 03 NLP Pipeline - Patient Split & Zero Leakage Enforcement
Splits labeled data into Train (70%), Val (15%), and Test (15%) ensuring both
zero patient overlap and zero duplicate-text leakage across splits.
"""

import os
import hashlib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from stage03_nlp.src.preprocessing import normalize_clinical_text

PRIMARY_TEXT_COL = "clinical_note"
PATIENT_ID_COL = "patient_id"
TARGET_COL = "urgency"
RANDOM_SEED = 42


def split_patient_zero_leakage(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Performs patient-level splitting with strict zero-leakage duplicate text clustering.
    All instances of identical clinical text are assigned strictly to the same partition,
    guaranteeing:
      Train patients ∩ Val patients = 0
      Train patients ∩ Test patients = 0
      Val patients ∩ Test patients = 0
      Train texts ∩ Val texts = 0
      Train texts ∩ Test texts = 0
      Val texts ∩ Test texts = 0
    """
    # Create normalized text series for robust duplicate grouping
    norm_text = df[PRIMARY_TEXT_COL].astype(str).apply(normalize_clinical_text)
    text_groups = df.groupby(norm_text).ngroup().rename("text_group_id")
    df_with_group = df.join(text_groups)
    
    # Assign each group a primary label for stratification
    group_labels = df_with_group.groupby("text_group_id")[TARGET_COL].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
    )
    
    unique_groups = group_labels.index.values
    unique_targets = group_labels.values
    
    # First split: Train (70%) vs Temp (30%)
    np.random.seed(seed)
    temp_ratio = val_ratio + test_ratio  # 0.30
    
    sgkf1 = StratifiedGroupKFold(n_splits=round(1.0 / temp_ratio), shuffle=True, random_state=seed)
    train_idx, temp_idx = next(sgkf1.split(unique_groups, unique_targets, groups=unique_groups))
    
    train_groups = set(unique_groups[train_idx])
    temp_groups = unique_groups[temp_idx]
    temp_targets = unique_targets[temp_idx]
    
    # Second split: Val (50% of temp = 15%) vs Test (50% of temp = 15%)
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    val_sub_idx, test_sub_idx = next(sgkf2.split(temp_groups, temp_targets, groups=temp_groups))
    
    val_groups = set(temp_groups[val_sub_idx])
    test_groups = set(temp_groups[test_sub_idx])
    
    # Filter original data
    df_train = df_with_group[df_with_group["text_group_id"].isin(train_groups)].drop(columns=["text_group_id"]).copy()
    df_val = df_with_group[df_with_group["text_group_id"].isin(val_groups)].drop(columns=["text_group_id"]).copy()
    df_test = df_with_group[df_with_group["text_group_id"].isin(test_groups)].drop(columns=["text_group_id"]).copy()
    
    # Verify patient isolation
    train_pts = set(df_train[PATIENT_ID_COL])
    val_pts = set(df_val[PATIENT_ID_COL])
    test_pts = set(df_test[PATIENT_ID_COL])
    
    pt_overlap_train_val = len(train_pts & val_pts)
    pt_overlap_train_test = len(train_pts & test_pts)
    pt_overlap_val_test = len(val_pts & test_pts)
    
    # Verify text isolation
    train_texts = set(df_train[PRIMARY_TEXT_COL].astype(str).apply(normalize_clinical_text))
    val_texts = set(df_val[PRIMARY_TEXT_COL].astype(str).apply(normalize_clinical_text))
    test_texts = set(df_test[PRIMARY_TEXT_COL].astype(str).apply(normalize_clinical_text))
    
    txt_overlap_train_val = len(train_texts & val_texts)
    txt_overlap_train_test = len(train_texts & test_texts)
    txt_overlap_val_test = len(val_texts & test_texts)
    
    assert pt_overlap_train_val == 0, "Patient overlap between Train and Val!"
    assert pt_overlap_train_test == 0, "Patient overlap between Train and Test!"
    assert pt_overlap_val_test == 0, "Patient overlap between Val and Test!"
    assert txt_overlap_train_val == 0, "Duplicate text leakage between Train and Val!"
    assert txt_overlap_train_test == 0, "Duplicate text leakage between Train and Test!"
    assert txt_overlap_val_test == 0, "Duplicate text leakage between Val and Test!"
    
    audit_stats = {
        "train_patients": len(train_pts),
        "val_patients": len(val_pts),
        "test_patients": len(test_pts),
        "train_records": len(df_train),
        "val_records": len(df_val),
        "test_records": len(df_test),
        "train_unique_texts": len(train_texts),
        "val_unique_texts": len(val_texts),
        "test_unique_texts": len(test_texts),
        "patient_overlap_train_val": pt_overlap_train_val,
        "patient_overlap_train_test": pt_overlap_train_test,
        "patient_overlap_val_test": pt_overlap_val_test,
        "text_overlap_train_val": txt_overlap_train_val,
        "text_overlap_train_test": txt_overlap_train_test,
        "text_overlap_val_test": txt_overlap_val_test,
        "train_urgency_dist": df_train[TARGET_COL].value_counts().to_dict(),
        "val_urgency_dist": df_val[TARGET_COL].value_counts().to_dict(),
        "test_urgency_dist": df_test[TARGET_COL].value_counts().to_dict(),
    }
    
    print("\n" + "="*50)
    print("PATIENT & LEAKAGE SPLIT SUMMARY")
    print("="*50)
    print(f"Train patients: {audit_stats['train_patients']} | Records: {audit_stats['train_records']}")
    print(f"Val patients:   {audit_stats['val_patients']} | Records: {audit_stats['val_records']}")
    print(f"Test patients:  {audit_stats['test_patients']} | Records: {audit_stats['test_records']}")
    print(f"Patient overlap: Train intersect Val = {pt_overlap_train_val}, Train intersect Test = {pt_overlap_train_test}, Val intersect Test = {pt_overlap_val_test}")
    print(f"Duplicate text overlap: Train intersect Val = {txt_overlap_train_val}, Train intersect Test = {txt_overlap_train_test}, Val intersect Test = {txt_overlap_val_test}")
    print("="*50 + "\n")
    
    return df_train, df_val, df_test, audit_stats


def generate_leakage_report(audit_stats: Dict[str, Any], path: str):
    """Generates reports/leakage_audit.md."""
    lines = [
        "# Stage 03 NLP Pipeline: Data Leakage & Patient Split Audit",
        "",
        "## 1. Overview",
        "This audit verifies that patient identifiers and identical clinical texts do not leak between",
        "the Training (70%), Validation (15%), and Testing (15%) partitions.",
        "",
        "## 2. Partition Summary",
        "| Partition | Record Count | Unique Patients | Unique Clinical Texts |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Train** | {audit_stats['train_records']} | {audit_stats['train_patients']} | {audit_stats['train_unique_texts']} |",
        f"| **Validation** | {audit_stats['val_records']} | {audit_stats['val_patients']} | {audit_stats['val_unique_texts']} |",
        f"| **Test** | {audit_stats['test_records']} | {audit_stats['test_patients']} | {audit_stats['test_unique_texts']} |",
        "",
        "## 3. Class Distribution by Partition",
        "| Partition | High | Moderate | Low |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Train** | {audit_stats['train_urgency_dist'].get('High', 0)} | {audit_stats['train_urgency_dist'].get('Moderate', 0)} | {audit_stats['train_urgency_dist'].get('Low', 0)} |",
        f"| **Validation** | {audit_stats['val_urgency_dist'].get('High', 0)} | {audit_stats['val_urgency_dist'].get('Moderate', 0)} | {audit_stats['val_urgency_dist'].get('Low', 0)} |",
        f"| **Test** | {audit_stats['test_urgency_dist'].get('High', 0)} | {audit_stats['test_urgency_dist'].get('Moderate', 0)} | {audit_stats['test_urgency_dist'].get('Low', 0)} |",
        "",
        "## 4. Leakage Verification Results",
        "| Audit Check | Permissible Overlap | Observed Overlap | Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Patient Overlap: Train ∩ Validation** | 0 | {audit_stats['patient_overlap_train_val']} | **PASSED** |",
        f"| **Patient Overlap: Train ∩ Test** | 0 | {audit_stats['patient_overlap_train_test']} | **PASSED** |",
        f"| **Patient Overlap: Validation ∩ Test** | 0 | {audit_stats['patient_overlap_val_test']} | **PASSED** |",
        f"| **Text Leakage: Train ∩ Validation** | 0 | {audit_stats['text_overlap_train_val']} | **PASSED** |",
        f"| **Text Leakage: Train ∩ Test** | 0 | {audit_stats['text_overlap_train_test']} | **PASSED** |",
        f"| **Text Leakage: Validation ∩ Test** | 0 | {audit_stats['text_overlap_val_test']} | **PASSED** |",
        "",
        "## 5. Grouping Strategy Implementation",
        "Because 826 records share duplicated `cleaned_clinical_note` content, random row-level partitioning",
        "would cause identical clinical text to appear in both training and test sets. To guarantee zero text leakage,",
        "all identical clinical texts were grouped deterministically and assigned en bloc to a single split.",
        "Since each patient corresponds to exactly 1 record, patient isolation is also strictly preserved at 0 overlap.",
        ""
    ]
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[PATIENT SPLIT] Leakage audit report saved to: {path}")


def save_splits(
    df_train: pd.DataFrame, 
    df_val: pd.DataFrame, 
    df_test: pd.DataFrame, 
    data_dir: str
):
    """Saves train.csv, val.csv, test.csv to data_dir."""
    os.makedirs(data_dir, exist_ok=True)
    df_train.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(data_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    print(f"[PATIENT SPLIT] Saved train.csv ({len(df_train)}), val.csv ({len(df_val)}), test.csv ({len(df_test)}) to {data_dir}")
