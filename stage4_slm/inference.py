"""
Stage 04: Clinical Inference Pipeline for SLM Summarization (Qwen/Qwen2.5-0.5B-Instruct).

CPU-friendly local inference:
- Model: Qwen/Qwen2.5-0.5B-Instruct
- Device: CPU (torch_dtype=torch.float32)
- Zero CUDA / GPU / flash-attn requirements
- Ignores old Qwen2.5-3B LoRA adapters for clean base model inference
- Preserves clinical summarization system prompt & faithful synthesis
- Post-generation safety and hallucination guardrail verification
- Interactive CLI prompt and automated test mode (--test)
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stage4_slm.safety_guardrail import ClinicalSummarySafetyGuardrail

# Safe UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class ClinicalSummarizer:
    """Production CPU inference wrapper for Stage 04 SLM summarizer."""

    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        adapter_path: str = None
    ):
        self.base_model_name = base_model_name
        self.device = device
        self.dtype = torch_dtype
        self.guardrail = ClinicalSummarySafetyGuardrail()

        # Check default trained 0.5B CPU adapter if none specified
        if adapter_path is None:
            default_cpu_adapter = "stage4_slm/models/qwen2.5_0.5b_lora_cpu"
            if os.path.exists(os.path.join(default_cpu_adapter, "adapter_model.safetensors")):
                adapter_path = default_cpu_adapter

        # Explicitly ignore legacy 3B LoRA adapters to prevent architecture mismatches
        if adapter_path and "3b" in adapter_path.lower():
            adapter_path = None
        self.adapter_path = adapter_path

        # Load Tokenizer
        tok_source = self.adapter_path if (self.adapter_path and os.path.exists(os.path.join(self.adapter_path, "tokenizer.json"))) else self.base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load Pretrained Base Model on CPU
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=self.dtype,
            device_map=None,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True
        )
        base_model.to(self.device)

        # Mount trained LoRA adapter if found
        if self.adapter_path and os.path.exists(os.path.join(self.adapter_path, "adapter_model.safetensors")):
            print(f"[INIT] Mounting fine-tuned LoRA adapter from: {self.adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        else:
            self.model = base_model

        self.model.eval()

    def generate_summary(
        self,
        clinical_report: str,
        instruction: str = "Summarize the following oncology clinical report faithfully.",
        max_new_tokens: int = 128
    ) -> str:
        """
        Generates a concise clinical summary from an oncology clinical report.
        Strictly preserves factual statements and prohibits hallucinations.
        """
        prompt = (
            f"<|im_start|>system\n"
            f"You are a clinical oncology assistant. Summarize clinical notes faithfully without inventing facts.<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{instruction}\n\nClinical Report:\n{clinical_report}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic greedy generation for factual fidelity
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.encode("<|im_end|>")[0] if "<|im_end|>" in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id
            )

        gen_tokens = output_ids[0][input_ids.shape[1]:]
        summary = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 04: Clinical Summarizer Inference (Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--test", action="store_true", default=False, help="Run sample oncology test report")
    parser.add_argument("--text", type=str, default=None, help="Direct clinical report text string")
    parser.add_argument("--file", type=str, default=None, help="Path to text file containing clinical report")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model identifier")
    parser.add_argument("--adapter", type=str, default=None, help="Optional LoRA adapter path (ignored if 3B)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (default: cpu)")
    return parser.parse_args()


def display_result(report_text: str, summary: str, guardrail: ClinicalSummarySafetyGuardrail):
    val = guardrail.validate(clinical_report=report_text, generated_summary=summary)
    print("\nGenerated Summary:")
    print(summary)
    print("\n" + "-" * 50)
    print(f"Safety Validation: {val['status']}")
    if not val['is_safe']:
        print("Flags:")
        for flag in val['flags']:
            print(f"  - [{flag['category']}] {flag['message']}")
    print(f"\nDisclaimer: {val['disclaimer']}")
    print("-" * 50 + "\n")


def main():
    args = parse_args()

    summarizer = ClinicalSummarizer(
        base_model_name=args.base_model,
        device=args.device,
        torch_dtype=torch.float32,
        adapter_path=args.adapter
    )

    # Terminal Header
    print("Stage 04 SLM")
    print(f"Model: {args.base_model}")
    print(f"Device: {args.device.upper()}")
    print("Status: Ready\n")

    # 1. Simple Test Mode (--test)
    if args.test:
        sample_report = (
            "Nurse intake: pt feeling nausea after treatment; current med Erlotinib 80 mg daily; "
            "mutation noted as EGFR L858R; adverse event none reported."
        )
        print("Running sample oncology clinical report:")
        print(f"> {sample_report}\n")
        summary = summarizer.generate_summary(sample_report)
        display_result(sample_report, summary, summarizer.guardrail)
        return

    # 2. Text provided via --text flag
    if args.text:
        summary = summarizer.generate_summary(args.text)
        display_result(args.text, summary, summarizer.guardrail)
        return

    # 3. File provided via --file flag
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File not found '{args.file}'")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            report_text = f.read().strip()
        summary = summarizer.generate_summary(report_text)
        display_result(report_text, summary, summarizer.guardrail)
        return

    # 4. Interactive Manual Input Mode
    print("Enter clinical report:")
    try:
        user_input = input("> ").strip()
        if not user_input:
            print("[INFO] No text entered. Exiting.")
            return

        summary = summarizer.generate_summary(user_input)
        display_result(user_input, summary, summarizer.guardrail)

    except (KeyboardInterrupt, EOFError):
        print("\nSession ended.")


if __name__ == "__main__":
    main()
