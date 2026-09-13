"""
Stage 04: Model Loader for SLM Clinical Summarization (Qwen2.5-3B + LoRA).

Provides robust, cached singleton loading of:
1. Base Model: Qwen/Qwen2.5-3B via Hugging Face Transformers
2. Trained LoRA Adapter: PEFT LoRA weights mounted on the base model

Hardware Acceleration:
- Automatic CUDA GPU detection (FP16 / BF16)
- Transparent CPU fallback (FP32)
- Zero model re-loading on consecutive requests (singleton instance)
"""

import os
import sys
import logging
from typing import Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from peft import PeftModel

logger = logging.getLogger("stage04_slm.model_loader")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")

# Default configuration paths (overridable via environment variables)
DEFAULT_BASE_MODEL = os.environ.get("SLM_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEFAULT_ADAPTER_PATH = os.environ.get("SLM_ADAPTER_PATH", "")


class SLMModelBundle:
    """Container holding loaded model, tokenizer, and runtime device specifications."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        dtype: torch.dtype,
        base_model_name: str,
        adapter_path: str,
        is_mock: bool = False
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.is_mock = is_mock


class MockSLMModel(torch.nn.Module):
    """
    Lightweight mock model for fast CI/schema verification or environments
    without access to the 6GB Qwen base weights.
    """
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def eval(self):
        return self

    def generate(self, input_ids, **kwargs):
        # Generate a standard faithful summary response format for test validation
        mock_text = (
            "This clinical report summarizes an oncology patient evaluation. "
            "Prescribed medications, reported symptoms, genomic alterations, "
            "and adverse events were recorded faithfully."
        )
        tokens = self.tokenizer.encode(mock_text, return_tensors="pt")
        return torch.cat([input_ids, tokens.to(input_ids.device)], dim=1)


# Global singleton instance cache
_GLOBAL_MODEL_BUNDLE: Optional[SLMModelBundle] = None


def get_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    """Determines the optimal execution device and floating-point precision."""
    forced_device = os.environ.get("SLM_DEVICE", "").strip().lower()
    if forced_device in ["cuda", "gpu"] and torch.cuda.is_available():
        device = torch.device("cuda")
    elif forced_device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32

    return device, dtype


def load_slm_model(
    base_model_name: Optional[str] = None,
    adapter_path: Optional[str] = None,
    force_reload: bool = False
) -> SLMModelBundle:
    """
    Loads Qwen/Qwen2.5-3B and applies the trained PEFT LoRA adapter.
    Caches the bundle in memory so the model is loaded only once across the application lifecycle.

    Args:
        base_model_name: HuggingFace model identifier or local path.
        adapter_path: Path to the trained LoRA adapter weights directory.
        force_reload: If True, disregards cached singleton and reloads.

    Returns:
        SLMModelBundle containing model, tokenizer, device, and metadata.
    """
    global _GLOBAL_MODEL_BUNDLE

    if _GLOBAL_MODEL_BUNDLE is not None and not force_reload:
        logger.debug("Returning cached SLMModelBundle singleton.")
        return _GLOBAL_MODEL_BUNDLE

    base_model = base_model_name or os.environ.get("SLM_BASE_MODEL", DEFAULT_BASE_MODEL)
    adapter = adapter_path or os.environ.get("SLM_ADAPTER_PATH", DEFAULT_ADAPTER_PATH)

    # Check for mock testing mode (e.g. in test suites or CI without 3B weights downloaded)
    if os.environ.get("SLM_MOCK_FOR_TESTS", "").lower() in ["true", "1", "yes"]:
        logger.info("[MOCK MODE] Initializing mock SLM bundle for lightweight testing.")
        tok_source = adapter if os.path.exists(os.path.join(adapter, "tokenizer.json")) else "Qwen/Qwen2.5-0.5B"
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_source, use_fast=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
        device, dtype = get_device_and_dtype()
        mock_model = MockSLMModel(tokenizer).to(device)
        _GLOBAL_MODEL_BUNDLE = SLMModelBundle(
            model=mock_model,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
            base_model_name=base_model,
            adapter_path=adapter,
            is_mock=True
        )
        return _GLOBAL_MODEL_BUNDLE

    device, dtype = get_device_and_dtype()
    logger.info(f"Loading SLM base model '{base_model}' on {device} ({dtype})...")

    # 1. Load Tokenizer (prefer local adapter directory if tokenizer files exist)
    tok_dir = adapter if os.path.exists(os.path.join(adapter, "tokenizer.json")) else base_model
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
    except Exception as e:
        logger.warning(f"Failed to load tokenizer from '{tok_dir}': {e}. Falling back to '{base_model}'.")
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model
    try:
        if device.type == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=dtype,
                device_map="auto",
                attn_implementation="sdpa",
                low_cpu_mem_usage=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.float32,
                device_map=None,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True
            )
            model = model.to(device)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load base model '{base_model}': {exc}\n"
            "Ensure internet access is available to download Qwen/Qwen2.5-3B weights, "
            "or pre-download weights to the Hugging Face cache, "
            "or set SLM_MOCK_FOR_TESTS=1 for offline testing."
        ) from exc

    # 3. Apply LoRA Adapter
    adapter_weights_safetensors = os.path.join(adapter, "adapter_model.safetensors") if adapter else ""
    adapter_weights_bin = os.path.join(adapter, "adapter_model.bin") if adapter else ""
    has_adapter = bool(adapter) and os.path.exists(adapter) and (
        os.path.exists(adapter_weights_safetensors) or os.path.exists(adapter_weights_bin)
    )

    # Ignore legacy 3B adapters when running 0.5B base models
    if has_adapter and "0.5b" in base_model.lower() and "3b" in adapter.lower():
        logger.warning(f"Ignoring incompatible legacy 3B LoRA adapter '{adapter}' for 0.5B base model.")
        has_adapter = False

    if has_adapter:
        logger.info(f"Mounting trained LoRA adapter from '{adapter}'...")
        model = PeftModel.from_pretrained(model, adapter)
    else:
        logger.info(f"Operating with base model '{base_model}' weights (no adapter mounted).")

    # 4. Set model to evaluation mode
    model.eval()

    _GLOBAL_MODEL_BUNDLE = SLMModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        base_model_name=base_model,
        adapter_path=adapter,
        is_mock=False
    )
    logger.info("SLM Model Loader initialized successfully. Model is in evaluation mode.")
    return _GLOBAL_MODEL_BUNDLE


def reset_model_loader():
    """Clears the singleton cache, freeing memory."""
    global _GLOBAL_MODEL_BUNDLE
    _GLOBAL_MODEL_BUNDLE = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
