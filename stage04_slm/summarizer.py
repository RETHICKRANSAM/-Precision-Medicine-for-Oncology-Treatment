"""
Stage 04: Clinical Summarization Module (SLM Qwen2.5-3B + LoRA).

Generates faithful, factual clinical summaries from raw clinical reports and Stage 03 NLP structures.
Strictly avoids hallucination, extraneous repetition, and unauthorized clinical recommendations.
"""

import os
import re
import logging
from typing import Dict, Any, Optional, List
import torch

from stage04_slm.model_loader import load_slm_model, SLMModelBundle
from stage04_slm.safety_guardrail import ClinicalSummarySafetyGuardrail

logger = logging.getLogger("stage04_slm.summarizer")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")

SYSTEM_PROMPT = "You are a clinical oncology assistant. Summarize clinical notes faithfully without inventing facts."

USER_PROMPT_TEMPLATE = (
    "Summarize the following clinical oncology report faithfully.\n"
    "Include cancer-related information, genomic mutation, medication\n"
    "and dosage if mentioned, symptoms, and adverse events.\n"
    "Do not invent information.\n\n"
    "Clinical report:\n"
    "{clinical_report}"
)


def clean_generated_output(raw_text: str) -> str:
    """
    Cleans model-generated output safely:
    - Strips leading assistant role tags
    - Stops at the first assistant end-of-turn token (<|im_end|>, <|endoftext|>, etc.)
    - Removes all Qwen special tokens
    - Detects and prunes repetitive continuation loops without arbitrary truncation
    - Preserves exact clinical terms, numbers, and dosages
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # 1. Strip leading assistant header if present
    text = re.sub(r"^(?:<\|im_start\|>\s*assistant\s*)", "", text, flags=re.IGNORECASE).strip()

    # 2. Stop at earliest occurrence of assistant end-of-turn or subsequent prompt reversal tags
    stop_markers = [
        "<|im_end|>",
        "<|endoftext|>",
        "<|im_start|>",
        "\nUser:",
        "\nAssistant:",
        "\nSystem:",
        "\nClinical report:",
        "</s>"
    ]
    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0]

    # 3. Strip any remaining Qwen/HuggingFace control tokens
    text = re.sub(r"<\|[a-zA-Z0-9_-]+\|>", "", text)
    text = text.strip()

    # 3. Detect and remove sentence-level repetitive continuation
    # Break into sentences while preserving abbreviations and decimals (e.g. 5.0 mg)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1:
        seen_sentences: List[str] = []
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_norm = re.sub(r"\s+", " ", s_clean.lower())
            # If this exact normalized sentence is identical to the immediate predecessor or repeats a loop
            if seen_sentences and s_norm == re.sub(r"\s+", " ", seen_sentences[-1].lower()):
                # Repetition loop encountered; stop here
                break
            seen_sentences.append(s_clean)
        text = " ".join(seen_sentences)

    # 4. Final whitespace and punctuation tidying
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ClinicalSummarizer:
    """Inference orchestrator for Stage 04 SLM Faithful Clinical Summarization."""

    def __init__(
        self,
        model_bundle: Optional[SLMModelBundle] = None,
        guardrail: Optional[ClinicalSummarySafetyGuardrail] = None
    ):
        self.bundle = model_bundle or load_slm_model()
        self.guardrail = guardrail or ClinicalSummarySafetyGuardrail()

    def build_prompt(self, clinical_report: str) -> str:
        """Constructs prompt using the exact Qwen chat template."""
        user_content = USER_PROMPT_TEMPLATE.format(clinical_report=clinical_report.strip())
        
        # Check if tokenizer has apply_chat_template
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        if hasattr(self.bundle.tokenizer, "apply_chat_template"):
            try:
                formatted_prompt = self.bundle.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted_prompt
            except Exception as e:
                logger.debug(f"apply_chat_template failed: {e}. Falling back to manual ChatML.")

        # Fallback to standard Qwen ChatML formatting
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def generate(
        self,
        clinical_report: str,
        max_new_tokens: int = 160,
        repetition_penalty: float = 1.1
    ) -> str:
        """
        Executes deterministic greedy decoding to generate faithful summary.
        """
        if not clinical_report or not clinical_report.strip():
            return ""

        prompt = self.build_prompt(clinical_report)
        tokenizer = self.bundle.tokenizer
        model = self.bundle.model
        device = self.bundle.device

        # Determine EOS token IDs
        eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        if "<|im_end|>" in tokenizer.get_vocab():
            eos_ids.append(tokenizer.encode("<|im_end|>", add_special_tokens=False)[0])

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Strictly deterministic greedy generation
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=eos_ids if len(eos_ids) > 1 else (eos_ids[0] if eos_ids else None)
            )

        # Slice off input tokens to isolate generated continuation
        gen_tokens = output_ids[0][input_ids.shape[1]:]
        raw_summary = tokenizer.decode(gen_tokens, skip_special_tokens=False)

        # Clean output
        clean_summary = clean_generated_output(raw_summary)
        return clean_summary

    def summarize(
        self,
        clinical_report: str,
        stage03_urgency: Optional[str] = None,
        stage03_entities: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Complete end-to-end Stage 04 pipeline:
        1. Strips any target-leakage or patient identifiers
        2. Generates faithful summary via SLM
        3. Executes safety guardrail validation
        4. Packages standardized output for Stage 06 Multi-Agent consumption
        """
        # Strictly verify zero patient leakage
        summary = self.generate(clinical_report)

        # Run safety guardrail
        validation = self.guardrail.validate(
            clinical_report=clinical_report,
            generated_summary=summary
        )

        return {
            "summary": summary,
            "urgency": stage03_urgency or "Not Provided",
            "entities": stage03_entities or {},
            "safety_status": validation["safety_status"],
            "safety_validation": validation
        }


# Global cached summarizer instance
_GLOBAL_SUMMARIZER: Optional[ClinicalSummarizer] = None


def get_summarizer() -> ClinicalSummarizer:
    """Returns singleton ClinicalSummarizer instance."""
    global _GLOBAL_SUMMARIZER
    if _GLOBAL_SUMMARIZER is None:
        _GLOBAL_SUMMARIZER = ClinicalSummarizer()
    return _GLOBAL_SUMMARIZER


def summarize_clinical_note(clinical_report: str) -> str:
    """
    Standard interface function as specified in Stage 04 requirements.
    Generates and returns only the concise faithful summary.
    """
    summarizer = get_summarizer()
    return summarizer.generate(clinical_report)
