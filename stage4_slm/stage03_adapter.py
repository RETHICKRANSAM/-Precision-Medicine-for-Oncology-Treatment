"""
Stage 03 NLP to Stage 04 SLM Integration Adapter.

Provides seamless interfacing between Stage 03 NLP extracted structures
(Urgency Classification, Medical NER) and Stage 04 SLM summarization.
Ensures zero feature leakage (patient_id, labels) into the generative model.
"""

from typing import Dict, Any, Optional
from stage4_slm.safety_guardrail import ClinicalSummarySafetyGuardrail


class Stage03ToStage04Adapter:
    """Adapter to ingest Stage 03 NLP structured outputs into Stage 04 SLM."""

    def __init__(self, summarizer_pipeline=None):
        self.summarizer = summarizer_pipeline
        self.guardrail = ClinicalSummarySafetyGuardrail()

    def format_input_for_slm(self, stage03_payload: Dict[str, Any]) -> str:
        """
        Formats raw clinical report and verified Stage 03 context.
        Strictly filters out patient_id, internal targets, or leaked outcomes.
        """
        raw_report = stage03_payload.get("clinical_report", "").strip()
        if not raw_report:
            raise ValueError("stage03_payload must contain non-empty 'clinical_report'.")

        # Optional genuine context from Stage 03 NLP
        urgency = stage03_payload.get("urgency", None)
        entities = stage03_payload.get("entities", {})

        context_parts = []
        if urgency and urgency not in ["Unknown", "None"]:
            context_parts.append(f"Triage Urgency: {urgency}")

        if isinstance(entities, dict):
            for k, v in entities.items():
                if v and v not in ["Unknown", "None", "None Reported"]:
                    clean_k = k.replace("_", " ").title()
                    context_parts.append(f"{clean_k}: {v}")

        if context_parts:
            structured_context = "; ".join(context_parts)
            formatted = f"{raw_report}\n[Stage 03 Clinical Context: {structured_context}]"
        else:
            formatted = raw_report

        return formatted

    def process(self, stage03_payload: Dict[str, Any], generate_fn=None) -> Dict[str, Any]:
        """
        Takes Stage 03 payload, generates clinical summary via SLM, and runs safety guardrail.
        """
        # Ensure patient_id is not passed to the generation pipeline
        patient_id = stage03_payload.get("patient_id") or stage03_payload.get("Patient_ID")

        prompt_input = self.format_input_for_slm(stage03_payload)

        # Generate summary
        if generate_fn is not None:
            summary = generate_fn(prompt_input)
        elif self.summarizer is not None:
            summary = self.summarizer(prompt_input)
        else:
            raise RuntimeError("No summarization function or pipeline available in adapter.")

        # Post-generation safety & hallucination audit
        validation = self.guardrail.validate(
            clinical_report=stage03_payload.get("clinical_report", ""),
            generated_summary=summary
        )

        response = {
            "clinical_summary": summary,
            "source_stage": "Stage_03_NLP",
            "model": "Qwen2.5-0.5B-Instruct",
            "safety_validation": validation
        }

        return response
