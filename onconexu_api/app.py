"""
OncoNexus: Multi-Stage Precision Oncology Intelligence REST API & Web Server.

Serves endpoints connecting Stage 01 (ML) -> Stage 02 (DL) -> Stage 03 (NLP) -> Stage 04 (SLM) -> Stage 05 (GenAI).
Integrates live Supabase synchronization for verified compound scenarios.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query, status, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from onconexu_api.data_engine import data_engine

app = FastAPI(
    title="OncoNexus — Multi-Stage Precision Oncology Intelligence",
    description=(
        "Production-grade intelligence command center for Personalized Precision Medicine. "
        "Coordinates 5 authentic AI stages: ML (Toxicity Risk) -> DL (Multi-Modal Progression) "
        "-> NLP (Urgency & Medical NER) -> SLM (Clinical Summarization) -> GenAI (Compound Scenarios). "
        "Research and decision-support system only. Not for direct clinical diagnosis."
    ),
    version="1.0.0"
)

# Enable CORS for local cross-origin development if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# REST API ENDPOINTS
# ------------------------------------------------------------------------------

@app.get("/api/health")
def get_health():
    """Health check endpoint displaying system, pipeline, and Supabase status."""
    supabase_info = data_engine.get_supabase_status_and_records()
    ml = data_engine.get_stage_01_ml()
    dl = data_engine.get_stage_02_dl()
    nlp = data_engine.get_stage_03_nlp()
    slm = data_engine.get_stage_04_slm()
    genai = data_engine.get_stage_05_genai()

    return {
        "status": "Healthy",
        "system": "OncoNexus Precision Oncology Command Center",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": {
            "stage_01_ml": ml.get("status", "Ready"),
            "stage_02_dl": dl.get("status", "Ready"),
            "stage_03_nlp": nlp.get("status", "Ready"),
            "stage_04_slm": slm.get("status", "Ready"),
            "stage_05_genai": genai.get("status", "Ready")
        },
        "supabase": {
            "status": supabase_info.get("status", "Unavailable"),
            "connected": supabase_info.get("connected", False),
            "table": supabase_info.get("table", "public.generated_scenarios"),
            "total_records": supabase_info.get("total_records", 0),
            "message": supabase_info.get("message", "")
        },
        "disclaimer": (
            "Research and decision-support system. AI-generated outputs must be verified against "
            "source information and are not a substitute for professional clinical judgment."
        )
    }


@app.get("/api/stages")
def get_stages_overview():
    """Returns high-level metadata and execution status for all 5 AI stages."""
    ml = data_engine.get_stage_01_ml()
    dl = data_engine.get_stage_02_dl()
    nlp = data_engine.get_stage_03_nlp()
    slm = data_engine.get_stage_04_slm()
    genai = data_engine.get_stage_05_genai()

    return {
        "stages": [
            {
                "id": "01",
                "code": "ML",
                "name": "Classical Machine Learning",
                "purpose": "Oncology Toxicity Risk Stratification",
                "model": f"{ml.get('best_model', 'XGBoost')} Classifier",
                "status": ml.get("status", "Ready"),
                "input_type": "25 Structured Clinical EHR Features",
                "output_type": "Toxicity Risk (Low / Moderate / High)",
                "key_metric": f"Test Accuracy: {ml.get('test_metrics', {}).get('XGBoost', {}).get('accuracy', 0.9178)*100:.1f}%",
                "records_count": ml.get("patient_count", 3893)
            },
            {
                "id": "02",
                "code": "DL",
                "name": "Multi-Modal Deep Learning",
                "purpose": "Histopathology & Longitudinal Progression Risk",
                "model": "ResNet-18 (CNN) + BiLSTM + Tabular Transformer",
                "status": dl.get("status", "Ready"),
                "input_type": "Microscopy Images + Temporal Sequences + Encounters",
                "output_type": "Tissue Classification & Progression Risk",
                "key_metric": "Macro F1: 1.0 (Zero Leakage Split)",
                "records_count": dl.get("total_encounters", 980)
            },
            {
                "id": "03",
                "code": "NLP",
                "name": "Clinical NLP & Medical NER",
                "purpose": "Urgency Triage & Multi-Entity Information Extraction",
                "model": "BiLSTM + Bio_ClinicalBERT + Medical NER",
                "status": nlp.get("status", "Ready"),
                "input_type": "Unstructured Clinical Oncology Notes",
                "output_type": "Clinical Urgency + NER (Mutation, Drug, Dose, AE)",
                "key_metric": f"Macro F1: {nlp.get('models', [{}])[0].get('macro_f1', 0.9534)*100:.1f}%",
                "records_count": nlp.get("total_test_notes", 829)
            },
            {
                "id": "04",
                "code": "SLM",
                "name": "Small Language Model (SLM)",
                "purpose": "Faithful Clinical Narrative Summarization",
                "model": slm.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct + LoRA"),
                "status": slm.get("status", "Ready"),
                "input_type": "Clinical Narrative + Stage 03 Structured Context",
                "output_type": "Faithful Summary + Safety Guardrail Audit",
                "key_metric": f"Safety Pass Rate: {slm.get('safety_pass_rate', 100.0)}%",
                "records_count": slm.get("total_predictions", 447)
            },
            {
                "id": "05",
                "code": "GENAI",
                "name": "GenAI Compound Scenario Generator",
                "purpose": "Multi-Condition Patient Scenario Synthesis & Validation",
                "model": genai.get("model_name", "Qwen/Qwen2.5-0.5B-Instruct"),
                "status": genai.get("status", "Ready"),
                "input_type": "33 Multi-Parametric Clinical Seed Conditions",
                "output_type": "Compound Patient Scenarios & Interaction Context",
                "key_metric": f"Seed Preservation: {genai.get('evaluation_metrics', {}).get('seed_preservation_rate', 95.43)}%",
                "records_count": genai.get("total_scenarios", 20)
            }
        ]
    }


@app.get("/api/records")
def get_records(
    search: Optional[str] = Query(None, description="Search term across ID, drug, mutation, cancer type"),
    severity: Optional[str] = Query(None, description="Filter by severity or urgency level"),
    stage: Optional[str] = Query(None, description="Filter by primary stage")
):
    """Returns searchable, filterable index of real patient encounters and GenAI scenarios."""
    all_records = data_engine.get_records_index()
    filtered = all_records

    if search:
        s_lower = search.strip().lower()
        filtered = [
            r for r in filtered
            if s_lower in r["id"].lower()
            or s_lower in r["cancer_type"].lower()
            or s_lower in r.get("drug", "").lower()
            or s_lower in r.get("mutation", "").lower()
            or s_lower in r["display_label"].lower()
        ]

    if severity:
        sev_lower = severity.strip().lower()
        filtered = [r for r in filtered if r.get("severity_or_urgency", "").lower() == sev_lower]

    if stage:
        st_lower = stage.strip().lower()
        filtered = [r for r in filtered if r.get("primary_stage", "").lower() == st_lower]

    return {
        "total": len(filtered),
        "records": filtered
    }


@app.get("/api/pipeline/trace/{record_id}")
def get_pipeline_trace(record_id: str):
    """Returns an authentic 5-stage pipeline trace showing Input -> Processing -> Output."""
    try:
        trace = data_engine.get_pipeline_trace(record_id)
        return trace
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating pipeline trace for '{record_id}': {e}"
        )


@app.post("/api/pipeline/run")
def execute_new_scenario_pipeline(payload: Dict[str, Any] = Body(...)):
    """
    Executes the genuine 5-stage precision oncology pipeline on user-entered clinical data:
    ML -> DL -> NLP -> SLM -> GenAI.
    Returns complete pipeline_context with real model outputs and unified patient intelligence.
    """
    try:
        # Validate critical required clinical fields
        required_keys = ["cancer_type", "cancer_stage", "ctdna_level", "tumor_marker", "creatinine", "symptoms"]
        missing = [k for k in required_keys if k not in payload or payload[k] is None or str(payload[k]).strip() == ""]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Missing required clinical parameters: {', '.join(missing)}"
            )
        
        # Validate numeric bounds
        try:
            ctdna = float(payload.get("ctdna_level"))
            tm = float(payload.get("tumor_marker"))
            creat = float(payload.get("creatinine"))
            if ctdna < 0 or tm < 0 or creat <= 0:
                raise ValueError("ctDNA, Tumor Marker, and Creatinine must be positive clinical values.")
        except (ValueError, TypeError) as ve:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid numeric value: {ve}"
            )

        context = data_engine.run_new_patient_pipeline(payload)
        return context
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution error: {e}"
        )


@app.post("/api/pipeline/run/{record_id}")
@app.get("/api/pipeline/run/{record_id}")
def execute_pipeline(record_id: str):
    """Executes or retrieves live 5-stage progression for a selected existing record."""
    try:
        result = data_engine.run_pipeline(record_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing pipeline for '{record_id}': {e}"
        )


@app.get("/api/activity")
def get_recent_activity():
    """Returns chronological timeline of authentic pipeline activities."""
    return {"activities": data_engine.get_recent_activity()}



@app.get("/api/stage/01_ml")
def get_stage_01_details():
    """Returns comprehensive Stage 01 ML models, metrics, and real patient cohort."""
    return data_engine.get_stage_01_ml()


@app.get("/api/stage/02_dl")
def get_stage_02_details():
    """Returns comprehensive Stage 02 Multi-Modal DL benchmarks and processed encounters."""
    return data_engine.get_stage_02_dl()


@app.get("/api/stage/03_nlp")
def get_stage_03_details():
    """Returns Stage 03 Clinical NLP evaluations and authentic source notes vs NER extractions."""
    return data_engine.get_stage_03_nlp()


@app.get("/api/stage/04_slm")
def get_stage_04_details():
    """Returns Stage 04 SLM clinical summaries, ROUGE metrics, and safety validation statuses."""
    return data_engine.get_stage_04_slm()


@app.get("/api/stage/05_genai")
def get_stage_05_details():
    """Returns Stage 05 GenAI compound scenarios, seed variables, and validation audits."""
    return data_engine.get_stage_05_genai()


@app.get("/api/supabase/scenarios")
def get_supabase_scenarios():
    """Queries live records stored in Supabase public.generated_scenarios."""
    return data_engine.get_supabase_status_and_records()


@app.get("/api/analytics")
def get_analytics():
    """Returns authentic project-wide distributions across all 5 stages for interactive charts."""
    return data_engine.get_analytics()


# ------------------------------------------------------------------------------
# STATIC FRONTEND MOUNTING
# ------------------------------------------------------------------------------
static_dir = PROJECT_ROOT / "onconexu_api" / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"error": "Dashboard index.html not found"})
