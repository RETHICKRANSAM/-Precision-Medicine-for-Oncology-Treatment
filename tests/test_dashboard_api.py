"""
Integration Test Suite for OncoNexus Dashboard REST API.

Tests:
1. /api/health
2. /api/stages
3. /api/records
4. /api/pipeline/trace/{record_id}
5. /api/stage/01_ml
6. /api/stage/02_dl
7. /api/stage/03_nlp
8. /api/stage/04_slm
9. /api/stage/05_genai
10. /api/supabase/scenarios
11. /api/analytics
12. Static root index.html serving
"""

import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from onconexu_api.app import app

client = TestClient(app)


def test_api_health():
    """Verify health endpoint reports system status and 5 stages."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Healthy"
    assert "pipeline" in data
    assert "stage_01_ml" in data["pipeline"]
    assert "stage_02_dl" in data["pipeline"]
    assert "stage_03_nlp" in data["pipeline"]
    assert "stage_04_slm" in data["pipeline"]
    assert "stage_05_genai" in data["pipeline"]
    assert "supabase" in data


def test_api_stages():
    """Verify high-level metadata for all 5 stages."""
    response = client.get("/api/stages")
    assert response.status_code == 200
    data = response.json()
    stages = data.get("stages", [])
    assert len(stages) == 5
    codes = [s["code"] for s in stages]
    assert codes == ["ML", "DL", "NLP", "SLM", "GENAI"]


def test_api_records():
    """Verify record indexing and filtering functionality."""
    response = client.get("/api/records")
    assert response.status_code == 200
    data = response.json()
    records = data.get("records", [])
    assert len(records) > 0

    # Test search filter
    search_resp = client.get("/api/records?search=SCEN")
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert all("SCEN" in r["id"] for r in search_data.get("records", []))


def test_api_pipeline_trace():
    """Verify end-to-end 5-stage trace for a scenario."""
    response = client.get("/api/pipeline/trace/SCEN-BATCH-0001")
    assert response.status_code == 200
    data = response.json()
    assert data["record_id"] == "SCEN-BATCH-0001"
    stages = data.get("stages", [])
    assert len(stages) == 5

    for stg in stages:
        assert "stage_id" in stg
        assert "code" in stg
        assert "input" in stg
        assert "processing" in stg
        assert "output" in stg


def test_api_stage_01_ml():
    """Verify Stage 01 ML returns authentic XGBoost metrics and patient samples."""
    response = client.get("/api/stage/01_ml")
    assert response.status_code == 200
    data = response.json()
    assert data["stage_code"] == "ML"
    assert data["best_model"] == "XGBoost"
    assert "features_used" in data
    assert len(data["features_used"]) >= 20
    assert "sample_patients" in data


def test_api_stage_02_dl():
    """Verify Stage 02 DL returns multi-modal benchmarks."""
    response = client.get("/api/stage/02_dl")
    assert response.status_code == 200
    data = response.json()
    assert data["stage_code"] == "DL"
    assert len(data.get("models", [])) == 3
    assert "sample_encounters" in data


def test_api_stage_03_nlp():
    """Verify Stage 03 NLP returns separated source notes and extracted NER."""
    response = client.get("/api/stage/03_nlp")
    assert response.status_code == 200
    data = response.json()
    assert data["stage_code"] == "NLP"
    notes = data.get("sample_notes", [])
    assert len(notes) > 0
    first_note = notes[0]
    assert "source_clinical_note" in first_note
    assert "nlp_extracted" in first_note
    ext = first_note["nlp_extracted"]
    assert "urgency" in ext
    assert "gene_mutation" in ext
    assert "drug_name" in ext


def test_api_stage_04_slm():
    """Verify Stage 04 SLM returns summaries and safety guardrail status."""
    response = client.get("/api/stage/04_slm")
    assert response.status_code == 200
    data = response.json()
    assert data["stage_code"] == "SLM"
    assert data["safety_pass_rate"] >= 85.0
    preds = data.get("predictions", [])
    assert len(preds) > 0
    first_pred = preds[0]
    assert "source_report" in first_pred
    assert "generated_summary" in first_pred
    assert "safety_status" in first_pred


def test_api_stage_05_genai():
    """Verify Stage 05 GenAI returns validated scenarios and evaluation rates."""
    response = client.get("/api/stage/05_genai")
    assert response.status_code == 200
    data = response.json()
    assert data["stage_code"] == "GENAI"
    assert data["model_name"] == "Qwen/Qwen2.5-0.5B-Instruct"
    scenarios = data.get("scenarios", [])
    assert len(scenarios) == 20
    assert data["evaluation_metrics"]["validation_pass_rate"] == 100.0
    assert data["evaluation_metrics"]["seed_preservation_rate"] == 95.43


def test_api_supabase_scenarios():
    """Verify Supabase live query endpoint responds gracefully."""
    response = client.get("/api/supabase/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "connected" in data
    if data["connected"]:
        assert data["total_records"] >= 20
        assert len(data["records"]) >= 20


def test_api_analytics():
    """Verify analytics endpoint returns distributions across all 5 stages."""
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "ml_toxicity_risk" in data
    assert "dl_progression_risk" in data
    assert "nlp_urgency" in data
    assert "genai_severity" in data
    assert "benchmarks" in data


def test_static_index():
    """Verify root / serves the OncoNexus dashboard HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "ONCONEXUS" in response.text
    assert "Multi-Stage Precision Oncology Intelligence" in response.text


def test_api_pipeline_run():
    """Verify pipeline execution endpoint runs through 5 stages."""
    response = client.post("/api/pipeline/run/SCEN-BATCH-0001")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Success"
    assert data["record_id"] == "SCEN-BATCH-0001"
    assert len(data["pipeline_stages"]) == 5


def test_api_activity():
    """Verify activity timeline endpoint returns real audit logs."""
    response = client.get("/api/activity")
    assert response.status_code == 200
    data = response.json()
    assert "activities" in data
    assert len(data["activities"]) >= 4

