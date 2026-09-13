"""
Stage 04: Small Language Model (SLM) Faithful Clinical Summarization.
Personalized Precision Medicine for Oncology Treatment Optimization.
"""

from stage04_slm.model_loader import load_slm_model, SLMModelBundle
from stage04_slm.safety_guardrail import ClinicalSummarySafetyGuardrail
from stage04_slm.summarizer import ClinicalSummarizer, summarize_clinical_note

__all__ = [
    "load_slm_model",
    "SLMModelBundle",
    "ClinicalSummarySafetyGuardrail",
    "ClinicalSummarizer",
    "summarize_clinical_note"
]
