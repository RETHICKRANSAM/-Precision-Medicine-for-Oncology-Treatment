"""
Stage 03 NLP Pipeline - Data Loader & Schema Validator
Handles schema inspection, data quality auditing, and isolation of Unknown records.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

EXPECTED_COLUMNS = [
    "patient_id",
    "note_type",
    "clinical_note",
    "urgency",
    "gene_mutation",
    "drug_name",
    "dosage_level",
    "adverse_event",
    "symptom_text",
    "annotation_status",
    "has_missing_fields",
    "clinical_note_placeholder",
    "cleaned_clinical_note",
    "text_word_count",
    "text_char_count"
]

TARGET_COL = "urgency"
PRIMARY_TEXT_COL = "cleaned_clinical_note"
ENRICHED_TEXT_COL = "enriched_clinical_note"
PATIENT_ID_COL = "patient_id"
KNOWN_CLASSES = ["Low", "Moderate", "High"]
UNKNOWN_LABEL = "Unknown"

# Additional columns added by feature engineering
FEATURE_ENG_COLUMNS = [
    "symptom_severity", "ae_grade", "mutation_risk",
    "urgency_score", "original_urgency",
    "symptom_tag", "ae_tag", "mutation_tag",
    "enriched_clinical_note"
]


def load_raw_dataset(csv_path: str) -> pd.DataFrame:
    """Loads and validates the actual schema of nlp_cleaned_data.csv."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Schema check
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Schema mismatch! Missing expected columns: {missing_cols}")
    
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        print(f"[WARNING] Extra columns found in dataset: {extra_cols}")
        
    return df[EXPECTED_COLUMNS].copy()


def audit_data_quality(df: pd.DataFrame, output_report_path: str = None) -> Dict[str, Any]:
    """
    Performs comprehensive data quality audit:
    - shape
    - columns & dtypes
    - missing values
    - duplicate rows
    - duplicate cleaned_clinical_note
    - unique patient count
    - urgency distribution
    - note_type distribution
    - annotation_status distribution
    - has_missing_fields distribution
    - text length statistics (min, max, mean, median, 95th percentile)
    """
    # Text length stats on cleaned_clinical_note
    word_counts = df[PRIMARY_TEXT_COL].astype(str).apply(lambda s: len(s.strip().split()))
    char_counts = df[PRIMARY_TEXT_COL].astype(str).apply(len)
    
    dup_rows = int(df.duplicated().sum())
    dup_texts = int(df.duplicated(subset=[PRIMARY_TEXT_COL], keep=False).sum())
    unique_texts = int(df[PRIMARY_TEXT_COL].nunique())
    unique_patients = int(df[PATIENT_ID_COL].nunique())
    
    urgency_counts = df[TARGET_COL].value_counts(dropna=False).to_dict()
    note_type_counts = df["note_type"].value_counts(dropna=False).to_dict()
    annotation_counts = df["annotation_status"].value_counts(dropna=False).to_dict()
    missing_fields_counts = df["has_missing_fields"].value_counts(dropna=False).to_dict()
    
    text_stats = {
        "min_word_count": int(word_counts.min()),
        "max_word_count": int(word_counts.max()),
        "mean_word_count": float(round(word_counts.mean(), 2)),
        "median_word_count": float(round(word_counts.median(), 2)),
        "p95_word_count": float(round(np.percentile(word_counts, 95), 2)),
        "min_char_count": int(char_counts.min()),
        "max_char_count": int(char_counts.max()),
        "mean_char_count": float(round(char_counts.mean(), 2)),
    }
    
    audit_results = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        "duplicate_full_rows": dup_rows,
        "duplicate_cleaned_notes_count": dup_texts,
        "unique_cleaned_notes_count": unique_texts,
        "unique_patient_count": unique_patients,
        "urgency_distribution": urgency_counts,
        "note_type_distribution": note_type_counts,
        "annotation_status_distribution": annotation_counts,
        "has_missing_fields_distribution": missing_fields_counts,
        "text_length_stats": text_stats,
    }
    
    if output_report_path:
        os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
        generate_markdown_report(audit_results, output_report_path)
        
    return audit_results


def generate_markdown_report(audit: Dict[str, Any], path: str):
    """Generates a detailed markdown report for data quality."""
    lines = [
        "# Clinical NLP Data Quality Audit Report",
        "",
        "## 1. Executive Summary",
        f"- **Dataset Shape**: {audit['shape'][0]} rows, {audit['shape'][1]} columns",
        f"- **Unique Patients**: {audit['unique_patient_count']}",
        f"- **Duplicate Full Rows**: {audit['duplicate_full_rows']}",
        f"- **Unique Cleaned Clinical Notes**: {audit['unique_cleaned_notes_count']}",
        f"- **Rows Sharing Duplicate Text**: {audit['duplicate_cleaned_notes_count']}",
        "",
        "## 2. Missing Value Analysis",
        "| Column | Missing Count | Missing Percentage |",
        "| :--- | :--- | :--- |"
    ]
    for col, count in audit["missing_values"].items():
        pct = (count / audit["shape"][0]) * 100
        lines.append(f"| `{col}` | {count} | {pct:.2f}% |")
        
    lines.extend([
        "",
        "## 3. Target Distribution (`urgency`)",
        "| Urgency Class | Record Count | Percentage |",
        "| :--- | :--- | :--- |"
    ])
    for cls, count in audit["urgency_distribution"].items():
        pct = (count / audit["shape"][0]) * 100
        lines.append(f"| **{cls}** | {count} | {pct:.2f}% |")
        
    lines.extend([
        "",
        "## 4. Text Length Statistics (`cleaned_clinical_note`)",
        f"- **Minimum Word Count**: {audit['text_length_stats']['min_word_count']}",
        f"- **Maximum Word Count**: {audit['text_length_stats']['max_word_count']}",
        f"- **Mean Word Count**: {audit['text_length_stats']['mean_word_count']}",
        f"- **Median Word Count**: {audit['text_length_stats']['median_word_count']}",
        f"- **95th Percentile Word Count**: {audit['text_length_stats']['p95_word_count']}",
        f"- **Mean Character Count**: {audit['text_length_stats']['mean_char_count']}",
        "",
        "## 5. Clinical Attributes Distribution",
        "### Note Type",
        "| Note Type | Count |",
        "| :--- | :--- |"
    ])
    for nt, cnt in audit["note_type_distribution"].items():
        lines.append(f"| {nt} | {cnt} |")
        
    lines.extend([
        "",
        "### Annotation Status",
        "| Annotation Status | Count |",
        "| :--- | :--- |"
    ])
    for st, cnt in audit["annotation_status_distribution"].items():
        lines.append(f"| {st} | {cnt} |")
        
    lines.extend([
        "",
        "### Missing Fields Indicator (`has_missing_fields`)",
        "| Has Missing Fields | Count |",
        "| :--- | :--- |"
    ])
    for mf, cnt in audit["has_missing_fields_distribution"].items():
        lines.append(f"| {mf} | {cnt} |")
        
    lines.extend([
        "",
        "## 6. Treatment of Unknown Urgency Records",
        "- Total Unknown records: **" + str(audit["urgency_distribution"].get("Unknown", 0)) + "**",
        "- **Policy**: Excluded from supervised training and validation/test evaluation.",
        "- Preserved in `data/unlabeled_unknown_data.csv` for semi-supervised / active learning workflows.",
        ""
    ])
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[DATA LOADER] Data quality report saved to: {path}")


def separate_unknown_records(
    df: pd.DataFrame, 
    unlabeled_out_path: str = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separates Unknown urgency records from the labeled dataset.
    Returns (df_labeled, df_unknown).
    """
    unknown_mask = df[TARGET_COL] == UNKNOWN_LABEL
    df_unknown = df[unknown_mask].copy()
    df_labeled = df[~unknown_mask].copy()
    
    print(f"[DATA LOADER] Total records: {len(df)}")
    print(f"[DATA LOADER] Labeled records (Low/Moderate/High): {len(df_labeled)}")
    print(f"[DATA LOADER] Excluded Unknown records: {len(df_unknown)} ({len(df_unknown)/len(df)*100:.2f}%)")
    
    if unlabeled_out_path:
        os.makedirs(os.path.dirname(unlabeled_out_path), exist_ok=True)
        df_unknown.to_csv(unlabeled_out_path, index=False)
        print(f"[DATA LOADER] Saved unlabeled Unknown records to: {unlabeled_out_path}")
        
    return df_labeled, df_unknown


if __name__ == "__main__":
    raw_path = r"c:\Users\rethi\OneDrive\เอกสาร\Desktop\DS team pro\STAGE_3\nlp_cleaned_data.csv"
    df = load_raw_dataset(raw_path)
    audit = audit_data_quality(df, "stage03_nlp/reports/data_quality_report.md")
    df_labeled, df_unknown = separate_unknown_records(df, "stage03_nlp/data/unlabeled_unknown_data.csv")
