"""
Stage 03 NLP Pipeline - Clinical Bias Audits & NER Preparation
Performs:
1. Annotation Quality Audit (annotation_status vs urgency)
2. Missing Fields Audit (has_missing_fields vs urgency)
3. Note Type Bias Audit (note_type vs urgency)
4. NER Preparation & Reference Export (gene_mutation, drug_name, dosage_level, adverse_event)
"""

import os
import pandas as pd
from typing import Dict, Any, Tuple

TARGET_COL = "urgency"


def audit_annotation_quality(
    df: pd.DataFrame, 
    output_csv: str = "stage03_nlp/reports/annotation_quality_audit.csv"
) -> pd.DataFrame:
    """
    Analyzes urgency distribution across annotation_status values.
    Checks if reviewed/annotated vs pending/unreviewed have systematic differences.
    """
    ct = pd.crosstab(
        df["annotation_status"], 
        df[TARGET_COL], 
        margins=True, 
        margins_name="Total"
    )
    
    # Normalized row percentages
    ct_pct = pd.crosstab(
        df["annotation_status"], 
        df[TARGET_COL], 
        normalize="index"
    ) * 100
    
    # Combine counts and percentages
    result_rows = []
    for status in ct.index:
        row = {"annotation_status": status, "total_records": ct.loc[status, "Total"]}
        for col in [c for c in ct.columns if c != "Total"]:
            count = ct.loc[status, col]
            pct = ct_pct.loc[status, col] if status in ct_pct.index and col in ct_pct.columns else 0.0
            row[f"{col}_count"] = int(count)
            row[f"{col}_pct"] = float(round(pct, 2))
        result_rows.append(row)
        
    df_result = pd.DataFrame(result_rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_result.to_csv(output_csv, index=False)
    print(f"[BIAS AUDIT] Annotation quality audit saved to: {output_csv}")
    return df_result


def audit_missing_fields(
    df: pd.DataFrame,
    output_csv: str = "stage03_nlp/reports/missing_field_urgency_audit.csv"
) -> pd.DataFrame:
    """
    Analyzes has_missing_fields vs urgency.
    Checks whether incomplete records correlate with specific urgency levels.
    """
    ct = pd.crosstab(df["has_missing_fields"], df[TARGET_COL], margins=True, margins_name="Total")
    ct_pct = pd.crosstab(df["has_missing_fields"], df[TARGET_COL], normalize="index") * 100
    
    rows = []
    for mf in ct.index:
        row = {"has_missing_fields": mf, "total_records": ct.loc[mf, "Total"]}
        for col in [c for c in ct.columns if c != "Total"]:
            row[f"{col}_count"] = int(ct.loc[mf, col])
            row[f"{col}_pct"] = float(round(ct_pct.loc[mf, col], 2)) if mf in ct_pct.index else 0.0
        rows.append(row)
        
    df_result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_result.to_csv(output_csv, index=False)
    print(f"[BIAS AUDIT] Missing fields urgency audit saved to: {output_csv}")
    return df_result


def audit_note_type_bias(
    df: pd.DataFrame,
    output_csv: str = "stage03_nlp/reports/note_type_urgency_distribution.csv"
) -> pd.DataFrame:
    """
    Calculates percentage of Low, Moderate, High, Unknown across note types.
    """
    ct = pd.crosstab(df["note_type"], df[TARGET_COL], margins=True, margins_name="Total")
    ct_pct = pd.crosstab(df["note_type"], df[TARGET_COL], normalize="index") * 100
    
    rows = []
    for nt in ct.index:
        row = {"note_type": nt, "total_records": ct.loc[nt, "Total"]}
        for col in [c for c in ct.columns if c != "Total"]:
            row[f"{col}_count"] = int(ct.loc[nt, col])
            row[f"{col}_pct"] = float(round(ct_pct.loc[nt, col], 2)) if nt in ct_pct.index else 0.0
        rows.append(row)
        
    df_result = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_result.to_csv(output_csv, index=False)
    print(f"[BIAS AUDIT] Note type bias audit saved to: {output_csv}")
    return df_result


def prepare_ner_dataset(
    df: pd.DataFrame,
    output_csv: str = "stage03_nlp/data/ner_preparation_dataset.csv"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts entity reference fields:
    - gene_mutation (GENE)
    - drug_name (DRUG)
    - dosage_level (DOSAGE)
    - adverse_event (ADVERSE_EVENT)
    
    Analyzes whether values represent normalized categorical values vs raw text spans.
    Documents that token-level BIO tags and character offsets are NOT present in the raw table,
    and requires dictionary/span alignment for supervised NER token tagging.
    """
    ner_cols = [
        "patient_id",
        "cleaned_clinical_note",
        "gene_mutation",
        "drug_name",
        "dosage_level",
        "adverse_event",
        "symptom_text"
    ]
    df_ner = df[ner_cols].copy()
    
    # Check exact span matches in cleaned_clinical_note
    def check_presence(row, col):
        val = str(row[col]).lower().strip()
        if not val or val in ["unknown", "none reported", "nan"]:
            return False
        return val in str(row["cleaned_clinical_note"]).lower()
    
    df_ner["gene_in_text"] = df_ner.apply(lambda r: check_presence(r, "gene_mutation"), axis=1)
    df_ner["drug_in_text"] = df_ner.apply(lambda r: check_presence(r, "drug_name"), axis=1)
    df_ner["dosage_in_text"] = df_ner.apply(lambda r: check_presence(r, "dosage_level"), axis=1)
    df_ner["ae_in_text"] = df_ner.apply(lambda r: check_presence(r, "adverse_event"), axis=1)
    
    summary = {
        "total_records": len(df_ner),
        "gene_exact_span_match_rate": float(round(df_ner["gene_in_text"].mean() * 100, 2)),
        "drug_exact_span_match_rate": float(round(df_ner["drug_in_text"].mean() * 100, 2)),
        "dosage_exact_span_match_rate": float(round(df_ner["dosage_in_text"].mean() * 100, 2)),
        "ae_exact_span_match_rate": float(round(df_ner["ae_in_text"].mean() * 100, 2)),
        "entity_types": ["GENE", "DRUG", "DOSAGE", "ADVERSE_EVENT"],
        "annotation_nature": "Normalized categorical labels (not token-level BIO tags)"
    }
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_ner.to_csv(output_csv, index=False)
    print(f"[NER PREP] NER preparation dataset exported to: {output_csv}")
    print(f"[NER PREP] Exact text match rates: Drug={summary['drug_exact_span_match_rate']}%, Gene={summary['gene_exact_span_match_rate']}%, AE={summary['ae_exact_span_match_rate']}%, Dosage={summary['dosage_exact_span_match_rate']}%")
    
    return df_ner, summary
