"""
Stage 03 NLP Pipeline - Duplicate Text Audit
Audits duplicate cleaned_clinical_note entries, analyzes label consistency, and flags conflicts.
"""

import os
import pandas as pd
from typing import Tuple, Dict, Any

PRIMARY_TEXT_COL = "cleaned_clinical_note"
TARGET_COL = "urgency"


def run_duplicate_audit(
    df: pd.DataFrame, 
    output_csv_path: str = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Identifies all duplicate cleaned_clinical_note entries, examines label consistency,
    and produces reports/text_duplicate_audit.csv.
    """
    dup_mask = df.duplicated(subset=[PRIMARY_TEXT_COL], keep=False)
    df_dups = df[dup_mask].copy()
    
    total_dup_rows = len(df_dups)
    unique_dup_texts = df_dups[PRIMARY_TEXT_COL].nunique()
    
    records = []
    conflict_count = 0
    consistent_count = 0
    
    for text, group in df_dups.groupby(PRIMARY_TEXT_COL):
        count = len(group)
        labels = sorted(group[TARGET_COL].unique())
        has_conflict = len(labels) > 1
        
        if has_conflict:
            conflict_count += 1
        else:
            consistent_count += 1
            
        records.append({
            "cleaned_clinical_note": text,
            "number_of_occurrences": count,
            "unique_urgency_labels": "; ".join(labels),
            "label_count": len(labels),
            "has_label_conflict": has_conflict,
            "patient_ids": "; ".join(group["patient_id"].astype(str).tolist()[:5])  # sample IDs
        })
        
    audit_df = pd.DataFrame(records).sort_values(
        by=["has_label_conflict", "number_of_occurrences"], ascending=[False, False]
    )
    
    summary = {
        "total_dataset_rows": len(df),
        "total_duplicated_rows": total_dup_rows,
        "unique_duplicated_text_groups": unique_dup_texts,
        "conflicting_label_groups": conflict_count,
        "consistent_label_groups": consistent_count,
    }
    
    print(f"[DUPLICATE AUDIT] Found {unique_dup_texts} distinct duplicated text groups across {total_dup_rows} rows.")
    print(f"[DUPLICATE AUDIT] Groups with conflicting labels: {conflict_count}")
    print(f"[DUPLICATE AUDIT] Groups with consistent labels: {consistent_count}")
    
    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        audit_df.to_csv(output_csv_path, index=False)
        print(f"[DUPLICATE AUDIT] Report saved to: {output_csv_path}")
        
    return audit_df, summary


if __name__ == "__main__":
    from data_loader import load_raw_dataset
    df = load_raw_dataset(r"c:\Users\rethi\OneDrive\เอกสาร\Desktop\DS team pro\STAGE_3\nlp_cleaned_data.csv")
    run_duplicate_audit(df, "stage03_nlp/reports/text_duplicate_audit.csv")
