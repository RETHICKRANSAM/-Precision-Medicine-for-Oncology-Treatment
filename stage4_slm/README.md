# Stage 04 — Small Language Model (SLM) Pipeline

## Personalized Precision Medicine for Oncology Treatment Optimization

---

## 1. Stage 04 Objective

Stage 04 implements a domain-adapted Small Language Model (SLM) pipeline designed to ingest complex oncology clinical reports and synthesize concise, faithful, and hallucination-free clinical summaries.

### Architecture & Logical Role in the End-to-End System
The pipeline bridges Stage 03 (Natural Language Processing & Information Extraction) with Stage 06 (Autonomous Clinical Agent):

```
Stage 03 NLP: Clinical Report
       ↓
Urgency Classification + Medical NER
       ↓
Structured Clinical Payload (Urgency, Gene Mutation, Drug, Dosage, Adverse Event)
       ↓
Stage 04 SLM: Qwen2.5-3B + LoRA (Instruction Fine-Tuning)
       ↓
Faithful Clinical Summary + Post-Generation Safety Guardrail
       ↓
Stage 06 Agent: Decision Support & Treatment Optimization
```

> [!IMPORTANT]
> **Clinical Non-Prescriptive Posture:**
> The Stage 04 SLM is strictly a **decision-support summarization system**. It synthesizes only information explicitly documented in the input narrative and Stage 03 NLP context. It is prohibited from formulating autonomous diagnoses, prescribing medications, altering dosages, or inventing clinical findings.

---

## 2. Dataset & Integrity

- **Master Clean Dataset:** `stage_4/slm_master_dataset_deduplicated_clean.csv`
- **Total Valid Records:** 4,470 (100% verified complete records, 0 NaN/null values)
- **Deduplication Audit:** Eliminated 530 duplicate clinical reports present in the raw 5,000-record intake (`slm_master_dataset_with_ner.csv (1).xls`), eliminating identical clinical notes across distinct patient IDs.
- **Partitioning Strategy:** Patient-level stratified split (80% Train, 10% Validation, 10% Holdout Test) stratified by triage urgency:
  - `stage4_slm/data/train.jsonl`: 3,576 records
  - `stage4_slm/data/validation.jsonl`: 447 records
  - `stage4_slm/data/test.jsonl`: 447 records (strictly locked until final evaluation)

---

## 3. Data Leakage Prevention

Prior artifacts suffered from cross-split contamination due to random splitting prior to clinical report deduplication (51 train/val overlap, 57 train/test overlap, 20 val/test overlap). 

The Stage 04 pipeline enforces strict zero-leakage guarantees:
- **Patient Overlap:** 0 patients shared across Train, Validation, or Test splits.
- **Report Overlap:** 0 clinical reports shared across splits.
- **Target Leakage:** The target summary is completely excluded from the instruction and input prompt.
- **Identifier Masking:** `Patient_ID` is strictly barred from model inputs.
- See detailed audit: [`stage4_slm/reports/data_leakage_audit.md`](reports/data_leakage_audit.md).

---

## 4. Input & Output Format

Instruction tuning utilizes the standard ChatML / SFT structure:

```json
{
  "instruction": "Summarize the following oncology clinical report faithfully.",
  "input": "Nurse intake pt feeling nausea after treatment current med Erlotinib 80 mg daily mutation noted as EGFR L858R ae none",
  "output": "This nurse intake documents a patient receiving Erlotinib at 80 mg/day with an EGFR L858R mutation. Reported symptoms include nausea after treatment with no adverse events reported."
}
```

Prompt token masking (`labels = -100`) ensures loss is computed exclusively on the target summary tokens during training.

---

## 5. Model Architecture & LoRA Configuration

- **Base Architecture:** `Qwen2.5-3B` (`Qwen/Qwen2.5-3B`)
- **Fine-Tuning Method:** Low-Rank Adaptation (PEFT / LoRA)
- **Why LoRA:** 
  Full parameter fine-tuning on a 3B parameter model requires >24 GB VRAM, creates catastrophic forgetting risks, and requires storing full model checkpoints. LoRA freezes all 3 billion base parameters and trains low-rank decomposition matrices ($r=16$) on key linear projection layers, reducing trainable weights to <0.5% while achieving state-of-the-art domain adaptation.

### LoRA Hyperparameters
```python
LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
```

---

## 6. Hardware Safety & Resource Management

- **Automatic Environment Detection:** Script dynamically detects CUDA availability, VRAM allocation, and provides CPU fallback mode.
- **Memory Optimization:**
  - Gradient accumulation (`gradient_accumulation_steps = 4`)
  - Sequence truncation (`max_length = 256`, safely encompassing 100% of clinical notes whose average length is ~55 tokens)
  - Mixed-precision (`bfloat16`/`float16` when GPU-enabled)
  - Out-of-memory exception catching with clear error remediation suggestions.

---

## 7. Evaluation & Verification

Evaluation is executed exclusively on the untouched holdout test set (`stage4_slm/data/test.jsonl`, 447 records).

### Text Generation Metrics
- **ROUGE-1 / ROUGE-2 / ROUGE-L**
- Average token length tracking (Input vs Reference vs Generated)
- Empty output and failure rates

### Factual & Entity Preservation Checks
The evaluation tracks whether key clinical entities in the source note are faithfully represented in the summary:
- Prescribed Oncology Drugs
- Genomic Mutations / Biomarkers
- Dosage Regimens
- Adverse Events / Toxicities

Reports generated:
- [`stage4_slm/reports/evaluation_report.md`](reports/evaluation_report.md)
- [`stage4_slm/reports/model_comparison.csv`](reports/model_comparison.csv)
- [`stage4_slm/outputs/predictions.csv`](outputs/predictions.csv)
- [`stage4_slm/outputs/sample_predictions.json`](outputs/sample_predictions.json)

---

## 8. Post-Generation Safety & Hallucination Guardrail

The module [`stage4_slm/safety_guardrail.py`](safety_guardrail.py) implements automated verification:
1. **Drug Hallucination Check:** Flags any antineoplastic agent mentioned in the summary that does not appear in the source report.
2. **Mutation Alteration Check:** Flags any genomic variant not documented in the input.
3. **Dosage Consistency Check:** Checks numerical dosage values against source report.
4. **Prescription Prohibition:** Flags any unauthorized prescriptive triggers (`"should start"`, `"increase dose"`, `"prescribe"`).
5. **Mandatory Disclaimer:** Appends clinical decision-support warning to all outputs.

---

## 9. How to Run

### 1. Run Data Preparation & Leakage Audit
```bash
python scratch/prepare_clean_sft.py
```

### 2. Run LoRA Fine-Tuning
```bash
python stage4_slm/train_slm.py \
    --model_name "Qwen/Qwen2.5-3B" \
    --train_file "stage4_slm/data/train.jsonl" \
    --val_file "stage4_slm/data/validation.jsonl" \
    --output_dir "stage4_slm/models/qwen2.5_3b_lora" \
    --epochs 2 \
    --batch_size 2 \
    --gradient_accumulation 4 \
    --learning_rate 2e-4
```

### 3. Run Holdout Test Evaluation
```bash
python stage4_slm/evaluate_slm.py \
    --base_model "Qwen/Qwen2.5-3B" \
    --adapter_dir "stage4_slm/models/qwen2.5_3b_lora" \
    --test_file "stage4_slm/data/test.jsonl"
```

### 4. Run CLI Inference
```bash
# Single text input
python stage4_slm/inference.py --text "Biopsy notes KRAS G12C mutation. Patient started on Sotorasib 960 mg daily. Reports grade 2 diarrhea."

# Or from file
python stage4_slm/inference.py --file "path/to/report.txt"
```

### 5. Launch REST API
```bash
uvicorn stage4_slm.api:app --host 0.0.0.0 --port 8000
```
Endpoint: `POST /summarize`  
Payload: `{"clinical_report": "...", "urgency": "High", "entities": {...}}`

---

## 10. Stage 03 and Stage 06 Integration

The integration adapter [`stage4_slm/stage03_adapter.py`](stage03_adapter.py) ingests Stage 03 NLP predictions (urgency classification and NER dictionary) and structures them into the prompt without patient identifier leakage, returning structured JSON directly consumable by Stage 06 Agent.
