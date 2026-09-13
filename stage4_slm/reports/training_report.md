# Stage 04 SLM Training Report: Qwen2.5 + LoRA Fine-Tuning

**Execution Date:** 2026-09-10  
**Model Name:** `Qwen/Qwen2.5-3B`  
**Tuning Method:** PEFT / LoRA (Supervised Fine-Tuning)  
**Target Architecture:** Qwen2.5 Causal Language Model  

---

## 1. Hardware & Environment
- **Device Used:** `CPU` (CPU Fallback)
- **Total System VRAM / Memory:** `0.00 GB`
- **Precision:** `torch.float32`

---

## 2. LoRA Adapter Architecture & Parameter Efficiency
- **Target Modules:** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **LoRA Rank ($r$):** `16`
- **LoRA Alpha ($lpha$):** `32`
- **LoRA Dropout:** `0.05`
- **Bias Term:** `none`

### Parameter Counts
| Metric | Value |
| :--- | :--- |
| **Total Base Model Parameters** | 87,219,712 |
| **Trainable LoRA Parameters** | 507,904 |
| **Trainable Parameter Percentage** | **0.5823%** |
| **Base Model Status** | **100% Frozen** |

---

## 3. Dataset Configuration
- **Train File:** `stage4_slm/data/train.jsonl` (80 utilized samples)
- **Validation File:** `stage4_slm/data/validation.jsonl` (20 utilized samples)
- **Test File (Untouched):** `stage4_slm/data/test.jsonl`
- **Max Sequence Length:** `96` tokens
- **Loss Masking:** Loss computed exclusively on target clinical summary tokens.

---

## 4. Hyperparameters
- **Epochs:** `2`
- **Per-Device Batch Size:** `4`
- **Gradient Accumulation Steps:** `2`
- **Effective Batch Size:** `8`
- **Learning Rate:** `0.0002`
- **Warmup Steps:** `1`
- **Optimizer:** `AdamW (weight_decay=0.01)`
- **Random Seed:** `42`

---

## 5. Training Progression & Loss History
| Epoch | Training Loss | Validation Loss | Duration |
| :---: | :---: | :---: | :---: |
| 1 | 11.7909 | 11.4037 | 35.3s |
| 2 | 11.2726 | 11.1840 | 35.2s |

**Best Validation Loss:** `11.1840`  
**Saved Adapter Path:** `stage4_slm/models/qwen2.5_3b_lora`  
