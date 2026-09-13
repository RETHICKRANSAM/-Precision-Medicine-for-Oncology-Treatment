# Stage 04 SLM CPU Training Report: Qwen2.5-0.5B-Instruct + LoRA

**Model:** `Qwen/Qwen2.5-0.5B-Instruct`  
**Adapter Output Directory:** `stage4_slm/models/qwen2.5_0.5b_lora_cpu`  
**Hardware:** Host CPU (`torch.float32`, 12 cores, 0 CUDA/GPU dependencies)  
**Total Steps Completed:** `31` optimizer steps (124 training batches processed)  
**Total Training Time:** `485.50s` (`8.09 minutes`)  
**Initial Baseline Validation Loss:** `2.0989`  
**Step 25 Validation Loss:** `0.2062` (Improved by 90.18%)  
**Best Validation Loss:** `0.2062`  
**Latest Step 31 Training Loss:** `0.1570`  

---

## 1. LoRA Architecture & Parameters
- **Base Pretrained Model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Total Base Parameters:** `494,032,768` (100% Frozen)
- **Trainable LoRA Parameters:** `8,798,208` (1.7497% trainable)
- **Target Projection Modules:** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- **LoRA Hyperparameters:**
  - Rank ($r$): 16
  - Alpha ($\alpha$): 32
  - Dropout: 0.05
  - Bias: `"none"`
  - Task Type: `CAUSAL_LM`

---

## 2. Dataset Isolation & Integrity
- **Training Set:** `stage4_slm/data/train.jsonl` (3,576 records)
- **Validation Set:** `stage4_slm/data/val.jsonl` (447 records)
- **Test Set:** `stage4_slm/data/test.jsonl` (447 records — 100% held out, untouched)
- **Loss Masking:** Target-only token loss masking active (prompt tokens labeled `-100`, loss computed solely on assistant summary tokens).

---

## 3. Training Dynamics & Loss Curve
| Step | Training Loss | Validation Loss | Learning Rate | Elapsed Time |
| :---: | :---: | :---: | :---: | :---: |
| Baseline | — | **2.0989** | — | 0.0s |
| 1 | 2.0639 | — | 4.55e-06 | 67.9s |
| 5 | 1.6761 | — | 2.27e-05 | 110.4s |
| 10 | 1.1861 | — | 4.55e-05 | 169.0s |
| 15 | 0.6485 | — | 6.82e-05 | 227.9s |
| 20 | 0.5661 | — | 9.09e-05 | 285.8s |
| 25 | 0.2294 | **0.2062** | 1.14e-04 | 346.3s |
| 30 | 0.1953 | — | 1.36e-04 | 473.1s |
| 31 | **0.1570** | — | 1.41e-04 | 485.5s |

> [!NOTE]
> Training was safely paused after 31 optimizer steps (124 forward/backward batches) as permitted by the safe CPU runtime guidelines. The best model weights were saved at Step 25 where validation loss reached `0.2062`.

---

## 4. Saved Checkpoint & Adapter Artifacts
- LoRA Adapter: `stage4_slm/models/qwen2.5_0.5b_lora_cpu/adapter_model.safetensors` (35.2 MB)
- Tokenizer: `stage4_slm/models/qwen2.5_0.5b_lora_cpu/tokenizer.json`
- Config: `stage4_slm/models/qwen2.5_0.5b_lora_cpu/adapter_config.json`
- Step Checkpoint: `stage4_slm/models/qwen2.5_0.5b_lora_cpu/checkpoint-25/`
- JSON Logs: `stage4_slm/reports/training_log_cpu.json`
- CSV Logs: `stage4_slm/reports/training_log_cpu.csv`
