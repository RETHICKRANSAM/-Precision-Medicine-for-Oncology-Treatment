"""
Lightweight FastAPI REST API for Stage 04 SLM Clinical Summarization.

Endpoint:
  POST /summarize
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
import sys

# Ensure stage4_slm is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stage4_slm.safety_guardrail import ClinicalSummarySafetyGuardrail
from stage4_slm.stage03_adapter import Stage03ToStage04Adapter
from stage4_slm.inference import ClinicalSummarizer

app = FastAPI(
    title="Personalized Precision Medicine - Stage 04 SLM Clinical Summarizer API",
    description="Clinical summarization of oncology reports using Qwen2.5-0.5B-Instruct with safety and hallucination guardrails.",
    version="1.0.0"
)

# Global summarizer instance (lazy loaded)
_summarizer: Optional[ClinicalSummarizer] = None
_adapter: Optional[Stage03ToStage04Adapter] = None


def get_summarizer() -> ClinicalSummarizer:
    global _summarizer, _adapter
    if _summarizer is None:
        _summarizer = ClinicalSummarizer()
        _adapter = Stage03ToStage04Adapter(summarizer_pipeline=_summarizer.generate_summary)
    return _summarizer


class SummarizeRequest(BaseModel):
    clinical_report: str = Field(..., description="Unstructured or semi-structured oncology clinical report text.")
    urgency: Optional[str] = Field(None, description="Optional Stage 03 urgency classification (High, Moderate, Low).")
    entities: Optional[Dict[str, Any]] = Field(None, description="Optional Stage 03 extracted clinical entities.")


class SummarizeResponse(BaseModel):
    clinical_summary: str
    model: str
    source_stage: Optional[str] = "Stage_04_SLM"
    safety_validation: Dict[str, Any]


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "stage": "Stage 04 — SLM Clinical Summarization",
        "architecture": "Qwen2.5-0.5B-Instruct (CPU)"
    }


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    if not request.clinical_report or not request.clinical_report.strip():
        raise HTTPException(status_code=400, detail="clinical_report cannot be empty.")

    summarizer = get_summarizer()
    adapter = Stage03ToStage04Adapter()

    payload = {
        "clinical_report": request.clinical_report,
        "urgency": request.urgency,
        "entities": request.entities or {}
    }

    try:
        response = adapter.process(payload, generate_fn=summarizer.generate_summary)
        return SummarizeResponse(
            clinical_summary=response["clinical_summary"],
            model=response.get("model", "Qwen2.5-0.5B-Instruct"),
            source_stage=response.get("source_stage", "Stage_03_NLP"),
            safety_validation=response["safety_validation"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
