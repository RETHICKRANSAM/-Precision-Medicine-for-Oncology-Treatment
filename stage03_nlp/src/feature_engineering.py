"""
Stage 03 NLP Pipeline - Feature Engineering Module
Derives clinically-meaningful urgency labels from structured metadata columns,
and enriches clinical text with feature tags for improved model performance.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict

# ============================================================
# 1. SYMPTOM SEVERITY SCORING
# ============================================================
# Clinically-grounded mapping: symptom_text -> severity score (0-4)
SYMPTOM_SEVERITY = {
    # Score 0 - No clinical concern
    "No New Symptoms": 0,
    "Not Specified": 0,
    # Score 1 - Mild / manageable
    "Mild Fatigue": 1,
    "Skin Rash": 1,
    # Score 2 - Moderate / needs monitoring
    "Persistent Cough": 2,
    "Nausea After Treatment": 2,
    "Rash Worsening": 2,
    # Score 3 - Moderate-High / worsening
    "Fatigue + Cough": 3,
    "Loss Of Appetite": 3,
    "Severe Diarrhea": 3,
    # Score 4 - High / urgent attention
    "Shortness Of Breath": 4,
    "Chest Discomfort": 4,
    "Fever And Chills": 4,
}

SYMPTOM_SEVERITY_LABELS = {
    0: "NONE",
    1: "MILD",
    2: "MODERATE",
    3: "MODERATE_HIGH",
    4: "HIGH",
}

# ============================================================
# 2. ADVERSE EVENT GRADING (CTCAE-inspired)
# ============================================================
ADVERSE_EVENT_GRADE = {
    # Grade 0 - No AE
    "None Reported": 0,
    # Grade 1 - Mild (CTCAE Grade 1-2)
    "Fatigue": 1,
    "Rash": 1,
    "Nausea": 1,
    # Grade 2 - Moderate (CTCAE Grade 2-3)
    "Diarrhea": 2,
    "Fever": 2,
    "Mucositis": 2,
    "Dyspnea": 2,
    # Grade 3 - Severe (CTCAE Grade 3-4, potentially life-threatening)
    "Hepatotoxicity": 3,
    "Thrombocytopenia": 3,
    "Neutropenia": 3,
}

AE_GRADE_LABELS = {
    0: "NONE",
    1: "MILD",
    2: "MODERATE",
    3: "SEVERE",
}

# ============================================================
# 3. GENE MUTATION RISK STRATIFICATION
# ============================================================
MUTATION_RISK = {
    # Risk 0 - Unknown
    "Unknown": 0,
    # Risk 1 - Favorable prognosis (good TKI response)
    "EGFR L858R": 1,
    "EGFR exon 19 del": 1,
    # Risk 2 - Intermediate (targeted therapies available)
    "ALK fusion": 2,
    "BRAF V600E": 2,
    "KRAS G12C": 2,
    "KRAS-G12C": 2,
    "MET amplification": 2,
    # Risk 3 - Aggressive (treatment resistance, poor prognosis)
    "TP53 mutation": 3,
    "STK11": 3,
    "STK11 mutation": 3,
}

MUTATION_RISK_LABELS = {
    0: "UNKNOWN",
    1: "FAVORABLE",
    2: "INTERMEDIATE",
    3: "AGGRESSIVE",
}


def score_symptom_severity(symptom_text: str) -> int:
    """Map symptom_text to severity score (0-4)."""
    text = str(symptom_text).strip()
    # Try exact match first
    if text in SYMPTOM_SEVERITY:
        return SYMPTOM_SEVERITY[text]
    # Try case-insensitive match
    text_lower = text.lower()
    for key, val in SYMPTOM_SEVERITY.items():
        if key.lower() == text_lower:
            return val
    # Keyword fallback
    if any(kw in text_lower for kw in ["shortness", "sob", "chest", "fever", "chills"]):
        return 4
    elif any(kw in text_lower for kw in ["severe", "loss of appetite", "fatigue + cough", "fatigue and cough"]):
        return 3
    elif any(kw in text_lower for kw in ["persistent", "worsening", "nausea"]):
        return 2
    elif any(kw in text_lower for kw in ["mild", "skin rash"]):
        return 1
    return 0


def score_adverse_event(ae_text: str) -> int:
    """Map adverse_event to CTCAE-inspired grade (0-3)."""
    text = str(ae_text).strip()
    if text in ADVERSE_EVENT_GRADE:
        return ADVERSE_EVENT_GRADE[text]
    text_lower = text.lower()
    for key, val in ADVERSE_EVENT_GRADE.items():
        if key.lower() == text_lower:
            return val
    # Keyword fallback
    if any(kw in text_lower for kw in ["hepato", "thrombocyto", "neutropenia"]):
        return 3
    elif any(kw in text_lower for kw in ["diarrhea", "fever", "mucositis", "dyspnea"]):
        return 2
    elif any(kw in text_lower for kw in ["fatigue", "rash", "nausea"]):
        return 1
    return 0


def score_mutation_risk(mutation_text: str) -> int:
    """Map gene_mutation to risk level (0-3)."""
    text = str(mutation_text).strip()
    if text in MUTATION_RISK:
        return MUTATION_RISK[text]
    text_lower = text.lower()
    for key, val in MUTATION_RISK.items():
        if key.lower() == text_lower:
            return val
    # Keyword fallback
    if any(kw in text_lower for kw in ["tp53", "stk11"]):
        return 3
    elif any(kw in text_lower for kw in ["alk", "braf", "kras", "met"]):
        return 2
    elif any(kw in text_lower for kw in ["egfr"]):
        return 1
    return 0


def compute_composite_urgency(symptom_score: int, ae_grade: int, mutation_risk: int) -> str:
    """
    Compute composite urgency from individual scores.
    
    Score = symptom_severity + adverse_event_grade + mutation_risk
    - 0-4  → Low
    - 5-6  → Moderate
    - 7+   → High
    
    Thresholds chosen for near-balanced distribution (~33/36/30).
    """
    total = symptom_score + ae_grade + mutation_risk
    if total <= 4:
        return "Low"
    elif total <= 6:
        return "Moderate"
    else:
        return "High"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
    1. Score symptoms, adverse events, mutations
    2. Compute composite urgency score
    3. Derive new urgency labels
    4. Enrich clinical text with feature tags
    
    Returns DataFrame with new columns added.
    """
    df = df.copy()
    
    # --- Score individual features ---
    print("[FEATURE ENG] Scoring symptom severity...")
    df["symptom_severity"] = df["symptom_text"].apply(score_symptom_severity)
    
    print("[FEATURE ENG] Grading adverse events...")
    df["ae_grade"] = df["adverse_event"].apply(score_adverse_event)
    
    print("[FEATURE ENG] Stratifying mutation risk...")
    df["mutation_risk"] = df["gene_mutation"].apply(score_mutation_risk)
    
    # --- Composite score ---
    df["urgency_score"] = df["symptom_severity"] + df["ae_grade"] + df["mutation_risk"]
    
    # --- Store original urgency and derive new labels ---
    df["original_urgency"] = df["urgency"].copy()
    df["urgency"] = df.apply(
        lambda row: compute_composite_urgency(
            row["symptom_severity"], row["ae_grade"], row["mutation_risk"]
        ),
        axis=1
    )
    
    # --- Feature tag labels for text enrichment ---
    df["symptom_tag"] = df["symptom_severity"].map(SYMPTOM_SEVERITY_LABELS).fillna("NONE")
    df["ae_tag"] = df["ae_grade"].map(AE_GRADE_LABELS).fillna("NONE")
    df["mutation_tag"] = df["mutation_risk"].map(MUTATION_RISK_LABELS).fillna("UNKNOWN")
    
    # --- Enrich clinical text with structured feature tags ---
    print("[FEATURE ENG] Enriching clinical text with feature tags...")
    df["enriched_clinical_note"] = (
        "[SYMPTOM:" + df["symptom_tag"] + "] " +
        "[AE:" + df["ae_tag"] + "] " +
        "[MUTATION:" + df["mutation_tag"] + "] " +
        df["cleaned_clinical_note"].astype(str)
    )
    
    # --- Summary stats ---
    new_dist = df["urgency"].value_counts()
    print(f"\n[FEATURE ENG] New urgency distribution:")
    for cls, cnt in new_dist.items():
        print(f"  {cls}: {cnt} ({cnt/len(df)*100:.1f}%)")
    
    score_stats = df["urgency_score"].describe()
    print(f"\n[FEATURE ENG] Urgency score stats: min={score_stats['min']:.0f}, "
          f"max={score_stats['max']:.0f}, mean={score_stats['mean']:.2f}, "
          f"median={score_stats['50%']:.0f}")
    
    return df


def validate_feature_engineering(df: pd.DataFrame) -> Dict:
    """
    Validates that the engineered features correlate with the new urgency labels.
    Returns chi-square test results.
    """
    from scipy.stats import chi2_contingency
    
    results = {}
    test_cols = ["symptom_severity", "ae_grade", "mutation_risk", "symptom_tag", "ae_tag", "mutation_tag"]
    
    for col in test_cols:
        if col in df.columns:
            ct = pd.crosstab(df["urgency"], df[col])
            chi2, p, dof, expected = chi2_contingency(ct)
            results[col] = {"chi2": chi2, "p_value": p, "significant": p < 0.05}
    
    print("\n[FEATURE ENG] Chi-square validation (urgency vs engineered features):")
    for col, res in results.items():
        sig = "SIGNIFICANT" if res["significant"] else "NOT significant"
        print(f"  {col}: chi2={res['chi2']:.2f}, p={res['p_value']:.6f} ({sig})")
    
    return results


if __name__ == "__main__":
    # Quick test
    csv_path = r"C:\Users\rethi\OneDrive\เอกสาร\Desktop\DS team pro\STAGE_3\nlp_cleaned_data.csv"
    df = pd.read_csv(csv_path)
    df = engineer_features(df)
    validate_feature_engineering(df)
    
    print("\n=== SAMPLE ENRICHED TEXT ===")
    for urg in ["Low", "Moderate", "High"]:
        sample = df[df["urgency"] == urg]["enriched_clinical_note"].iloc[0]
        print(f"  [{urg}]: {sample[:200]}")
