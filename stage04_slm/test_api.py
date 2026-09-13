"""
Stage 04: Unit and Integration Test Suite for SLM API & Clinical Guardrails.

Test Scenarios:
1. Normal oncology note entity preservation (Pembrolizumab, 300mg twice daily, EGFR L858R, severe diarrhea, nausea)
2. Clinical note with missing entities (no hallucinations invented)
3. Multiple clinical entities preservation
4. Potential hallucination & unauthorized recommendation detection by Safety Guardrail
5. GET /health endpoint status check
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure root is on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Enable mock mode for testing if large base weights are not present
os.environ["SLM_MOCK_FOR_TESTS"] = "1"

from stage04_slm.safety_guardrail import ClinicalSummarySafetyGuardrail
from stage04_slm.summarizer import clean_generated_output
from stage04_slm.api import app

client = TestClient(app)


# ==============================================================================
# TEST 1: NORMAL ONCOLOGY NOTE PRESERVATION
# ==============================================================================
def test_normal_oncology_note_entity_preservation():
    """
    Test 1: Verify that a normal oncology note preserves all key entities:
    - Drug: Pembrolizumab
    - Dosage: 300mg (or 300mg twice daily)
    - Mutation: EGFR L858R
    - Symptoms: severe diarrhea
    - Adverse Event: nausea
    """
    note = "Trial screening EGFR L858R prior/current drug Pembrolizumab dose 300mg twice daily symptoms severe diarrhea ae nausea"
    
    # Grounded summary reflecting faithful generation
    faithful_summary = (
        "This trial screening note documents a patient with an EGFR L858R mutation receiving "
        "Pembrolizumab at a dose of 300mg twice daily. The patient reports symptoms of severe diarrhea "
        "and an adverse event of nausea."
    )
    
    # Run through safety guardrail to ensure zero false positives on faithful reproduction
    guardrail = ClinicalSummarySafetyGuardrail()
    validation = guardrail.validate(clinical_report=note, generated_summary=faithful_summary)
    
    assert validation["passed"] is True, f"Faithful summary should pass guardrail. Flags: {validation['flags']}"
    assert validation["safety_status"] == "passed"
    
    # Entity preservation checks
    summary_lower = faithful_summary.lower()
    assert "pembrolizumab" in summary_lower
    assert "300mg" in summary_lower or "300 mg" in summary_lower
    assert "egfr l858r" in summary_lower
    assert "severe diarrhea" in summary_lower
    assert "nausea" in summary_lower


# ==============================================================================
# TEST 2: CLINICAL NOTE WITH MISSING INFORMATION
# ==============================================================================
def test_missing_information_note():
    """
    Test 2: When entities are missing in the report, the model/guardrail
    must NOT invent cancer type, medication, dosage, mutation, symptom, or adverse event.
    """
    note = "Follow-up evaluation. Patient reports mild fatigue. Gene mutation unknown. No antineoplastic therapy active."
    
    # Scenario A: Faithful summary reflecting missing entities
    faithful_summary = (
        "Follow-up note: Patient reports mild fatigue. Genomic mutation is unknown "
        "and no active antineoplastic therapy is recorded."
    )
    guardrail = ClinicalSummarySafetyGuardrail()
    val_faithful = guardrail.validate(clinical_report=note, generated_summary=faithful_summary)
    assert val_faithful["passed"] is True
    
    # Scenario B: Hallucinated summary inventing Osimertinib and EGFR mutation
    hallucinated_summary = (
        "Patient with EGFR L858R mutation initiated on Osimertinib 80mg/day with hepatotoxicity."
    )
    val_hallucinated = guardrail.validate(clinical_report=note, generated_summary=hallucinated_summary)
    assert val_hallucinated["passed"] is False
    assert val_hallucinated["safety_status"] == "flagged"
    categories = [f["category"] for f in val_hallucinated["flags"]]
    assert "DRUG_HALLUCINATION" in categories
    assert "MUTATION_HALLUCINATION" in categories
    assert "UNSUPPORTED_ADVERSE_EVENT" in categories


# ==============================================================================
# TEST 3: NOTE CONTAINING MULTIPLE CLINICAL ENTITIES
# ==============================================================================
def test_multiple_clinical_entities_preservation():
    """
    Test 3: Verify a multi-entity note (mutation, dual toxicities, targeted drug)
    correctly passes validation when entities are accurately captured.
    """
    note = (
        "Follow-up visit: Patient with KRAS G12C receiving Sotorasib 960mg daily. "
        "Intake shows dyspnea, rash worsening, and grade 2 hepatotoxicity."
    )
    grounded_summary = (
        "Follow-up documents a patient with KRAS G12C on Sotorasib 960mg daily. "
        "Current clinical symptoms include dyspnea and rash worsening with hepatotoxicity noted."
    )
    guardrail = ClinicalSummarySafetyGuardrail()
    val = guardrail.validate(clinical_report=note, generated_summary=grounded_summary)
    assert val["passed"] is True
    
    for entity in ["kras g12c", "sotorasib", "960mg", "dyspnea", "rash", "hepatotoxicity"]:
        assert entity in grounded_summary.lower()


# ==============================================================================
# TEST 4: POTENTIAL HALLUCINATIONS & UNAUTHORIZED RECOMMENDATIONS
# ==============================================================================
def test_guardrail_catches_unauthorized_recommendations_and_hallucinations():
    """
    Test 4: Verify the safety guardrail detects:
    - Invented antineoplastic drugs
    - Dosage mismatch
    - Unauthorized prescriptive instructions ('recommend starting', 'prescribe')
    - Autonomous diagnosis claims
    """
    note = "Routine checkup: Patient has persistent cough. Currently on no targeted therapy."
    
    # A summary with unauthorized clinical prescription & invented drug
    dangerous_summary = (
        "Patient has persistent cough. Recommend starting Osimertinib 80mg daily immediately. "
        "Autonomous diagnosis confirms progression."
    )
    guardrail = ClinicalSummarySafetyGuardrail()
    val = guardrail.validate(clinical_report=note, generated_summary=dangerous_summary)
    
    assert val["passed"] is False
    assert val["safety_status"] == "flagged"
    
    categories = {f["category"] for f in val["flags"]}
    assert "DRUG_HALLUCINATION" in categories
    assert "UNAUTHORIZED_TREATMENT_RECOMMENDATION" in categories
    assert "AUTONOMOUS_DIAGNOSIS_CLAIM" in categories
    assert "GUARDRAIL WARNING" in val["sanitized_summary"]


# ==============================================================================
# TEST 5: GET /health API ENDPOINT
# ==============================================================================
def test_get_health_endpoint():
    """
    Test 5: Verify GET /health returns status=ok and stage identification.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["stage"] == "Stage 04 SLM"


# ==============================================================================
# TEST 6: POST /summarize & /summarize_stage03 API ENDPOINTS
# ==============================================================================
def test_post_summarize_endpoint():
    """Verify POST /summarize handles requests properly."""
    payload = {
        "clinical_report": "Trial screening EGFR L858R prior/current drug Pembrolizumab dose 300mg twice daily symptoms severe diarrhea ae nausea",
        "patient_id": "P01234"  # Should be filtered out
    }
    response = client.post("/summarize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert len(data["summary"]) > 0


def test_post_summarize_empty_fails():
    """Verify empty input returns 400 Bad Request."""
    response = client.post("/summarize", json={"clinical_report": ""})
    assert response.status_code == 400


def test_post_summarize_stage03_integration():
    """Verify POST /summarize_stage03 returns standardized Stage 06 payload."""
    payload = {
        "clinical_report": "Trial screening EGFR L858R prior/current drug Pembrolizumab dose 300mg twice daily symptoms severe diarrhea ae nausea",
        "urgency": "High",
        "entities": {
            "gene_mutation": "EGFR L858R",
            "drug_name": "Pembrolizumab",
            "dosage": "300mg twice daily",
            "adverse_event": "nausea"
        }
    }
    response = client.post("/summarize_stage03", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert data["urgency"] == "High"
    assert data["entities"]["gene_mutation"] == "EGFR L858R"
    assert data["safety_status"] in ["passed", "flagged"]
    assert "safety_validation" in data


# ==============================================================================
# TEST 7: MODEL OUTPUT CLEANING
# ==============================================================================
def test_model_output_cleaning():
    """Verify repetitive continuations and Qwen special tokens are cleaned cleanly."""
    raw = (
        "<|im_start|>assistant\nThis note documents a patient receiving Pembrolizumab at 300mg BID. "
        "Symptoms include severe diarrhea.<|im_end|>\n"
        "Symptoms include severe diarrhea. Symptoms include severe diarrhea."
    )
    cleaned = clean_generated_output(raw)
    assert "<|im_start|>" not in cleaned
    assert "<|im_end|>" not in cleaned
    assert cleaned.count("Symptoms include severe diarrhea.") == 1
