"""
Stage 03 NLP Pipeline - Clinical Error Analysis
Performs detailed directional error analysis on test predictions:
- Identifies confusion pairs (Low -> Mod, Mod -> High, High -> Mod, etc.)
- Detects clinical linguistic patterns: negation, severity indicators, adverse events, abbreviations
- Exports reports/error_analysis.csv
"""

import os
import re
import pandas as pd
from typing import List, Dict, Any

# Domain lexicons for clinical error pattern mining
NEGATION_WORDS = {"no", "not", "none", "denies", "without", "negative", "never", "resolved"}
SEVERITY_WORDS = {"mild", "moderate", "severe", "critical", "acute", "worsening", "worse", "intense", "stable"}
AE_WORDS = {"hepatotoxicity", "thrombocytopenia", "neutropenia", "mucositis", "rash", "toxicity", "diarrhea", "nausea", "fatigue"}
ABBREVIATIONS = {"sob", "f/u", "nsclc", "tx", "ae", "pt", "bid", "c/o", "specimen", "req", "req'd", "intake"}


def profile_clinical_error_text(text: str) -> Dict[str, Any]:
    """Analyzes presence of clinical phenomena in error text."""
    lower = text.lower()
    tokens = set(re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", lower))
    
    found_neg = tokens.intersection(NEGATION_WORDS)
    found_sev = tokens.intersection(SEVERITY_WORDS)
    found_ae = tokens.intersection(AE_WORDS)
    found_abbr = tokens.intersection(ABBREVIATIONS)
    
    return {
        "has_negation": len(found_neg) > 0,
        "negation_terms": "; ".join(found_neg) if found_neg else "None",
        "has_severity_word": len(found_sev) > 0,
        "severity_terms": "; ".join(found_sev) if found_sev else "None",
        "has_ae_terminology": len(found_ae) > 0,
        "ae_terms": "; ".join(found_ae) if found_ae else "None",
        "has_abbreviation": len(found_abbr) > 0,
        "abbreviations": "; ".join(found_abbr) if found_abbr else "None"
    }


def perform_error_analysis(
    df_test: pd.DataFrame,
    predictions_by_model: Dict[str, List[str]],
    output_csv: str = "stage03_nlp/reports/error_analysis.csv"
) -> pd.DataFrame:
    """
    Consolidates errors across all evaluated models on the test set.
    """
    y_true = df_test["urgency"].tolist()
    patient_ids = df_test["patient_id"].tolist()
    note_types = df_test["note_type"].tolist()
    text_col = "clinical_note" if "clinical_note" in df_test.columns else (
        "cleaned_clinical_note" if "cleaned_clinical_note" in df_test.columns else df_test.columns[0]
    )
    clinical_texts = df_test[text_col].tolist()
    
    all_error_rows = []
    
    for model_name, preds in predictions_by_model.items():
        for i in range(len(y_true)):
            act = y_true[i]
            pred = preds[i]
            if act != pred:
                text = clinical_texts[i]
                profile = profile_clinical_error_text(text)
                
                transition = f"{act} -> {pred}"
                is_underprediction = (act == "High" and pred in ["Moderate", "Low"]) or (act == "Moderate" and pred == "Low")
                is_overprediction = (act == "Low" and pred in ["Moderate", "High"]) or (act == "Moderate" and pred == "High")
                
                row = {
                    "patient_id": patient_ids[i],
                    "note_type": note_types[i],
                    "clinical_text": text,
                    "actual_label": act,
                    "predicted_label": pred,
                    "model": model_name,
                    "error_transition": transition,
                    "risk_type": "Under-triaged (High Risk)" if is_underprediction else ("Over-triaged" if is_overprediction else "Other"),
                    "has_negation": profile["has_negation"],
                    "negation_terms": profile["negation_terms"],
                    "has_severity_word": profile["has_severity_word"],
                    "severity_terms": profile["severity_terms"],
                    "has_ae_terminology": profile["has_ae_terminology"],
                    "ae_terms": profile["ae_terms"],
                    "has_abbreviation": profile["has_abbreviation"]
                }
                all_error_rows.append(row)
                
    df_errors = pd.DataFrame(all_error_rows)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_errors.to_csv(output_csv, index=False)
    print(f"[ERROR ANALYSIS] Total errors logged across models: {len(df_errors)}")
    print(f"[ERROR ANALYSIS] Saved detailed report to: {output_csv}")
    
    # Print distribution of transitions
    if not df_errors.empty:
        print("\n--- Error Transitions Distribution ---")
        print(df_errors.groupby(["model", "error_transition"]).size())
        
    return df_errors
