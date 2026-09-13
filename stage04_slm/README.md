# Stage 04 — Small Language Model (SLM) Faithful Clinical Summarization

Part of the **Personalized Precision Medicine for Oncology Treatment Optimization** system.

---

## 1. Overview & Purpose

Stage 04 implements a domain-adapted Small Language Model (SLM) fine-tuned to produce **concise, faithful clinical summaries** of unstructured oncology notes. 

In precision oncology, clinical documentation contains dense, heterogeneous narrative text regarding tumor genomics, targeted antineoplastic drugs, dosage regimens, toxicity profiles, and patient-reported symptoms. 

### Why an SLM?
- **High Local Deployability:** 3B parameters run efficiently on standard workstations, clinical edge nodes, and cloud GPUs without requiring massive cluster infrastructure.
- **Strict Factual Fidelity:** By training with targeted supervised fine-tuning (SFT) and constrained greedy generation, the SLM minimizes generative hallucinations compared to general-purpose foundation LLMs.
- **Privacy Compliance:** Enables on-premise execution with zero third-party API data exposure, adhering to HIPAA and clinical data confidentiality standards.

> [!WARNING]
> **Clinical Research Disclaimer**: This system is clinical decision-support infrastructure intended for retrospective research and queue prioritization. It does NOT autonomously diagnose patients, prescribe therapy, alter medication dosages, or replace oncologist clinical judgment.

---

## 2. Model Architecture & LoRA Adaptation

- **Base Model:** `Qwen/Qwen2.5-3B` (Hugging Face Transformers)
  - Architecture: Decoder-only Transformer with RoPE, GQA, and SwiGLU activations.
  - Context Window: 32,768 tokens (compact sequence usage ~60 tokens).
- **Fine-Tuning Technique:** Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA).
  - LoRA Rank ($r$): 16
  - LoRA Alpha ($\alpha$): 32
  - LoRA Dropout: 0.05
  - Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - Trainable Parameters: ~0.06% of total network parameters (~2 MB adapter footprint).
- **Weight Storage:**
  - LoRA Adapter: Saved locally in `stage04_slm/model/adapter/` (`adapter_model.safetensors`).
  - Base Model: Downloaded on demand or loaded from cache (`Qwen/Qwen2.5-3B`).

---

## 3. Dataset & Supervised Fine-Tuning (SFT) Split

- **Dataset:** `slm_master_dataset_with_ner.csv.xls` (5,000 clinically grounded oncology records).
- **Entities Covered:**
  - 10 Actionable Genomic Alterations: `EGFR L858R`, `EGFR exon 19 del`, `KRAS G12C`, `BRAF V600E`, `ALK fusion`, `MET amplification`, `TP53`, `STK11`, etc.
  - 12 Antineoplastic Agents: `Osimertinib`, `Sotorasib`, `Pembrolizumab`, `Cisplatin`, `Carboplatin`, `Trametinib`, `Erlotinib`, etc.
  - Common Dosages: `80 mg/day`, `150 mg/day`, `300mg twice daily`, `5 mg BID`, etc.
  - Toxicities & CTCAE Grades: `Hepatotoxicity`, `Severe Diarrhea`, `Mucositis`, `Thrombocytopenia`, `Nausea`, `Rash`.
- **Deduplication & Zero-Leakage Split:**
  - 530 duplicate report narratives were consolidated to prevent train-test contamination.
  - Stratified 80% / 10% / 10% split by urgency and mutation profile:
    - **Train:** 4,000 samples (`slm_train_sft.jsonl`)
    - **Validation:** 500 samples (`slm_validation_sft.jsonl`)
    - **Test:** 500 samples (`slm_test_sft.jsonl`, strictly held-out)

---

## 4. Training vs. Inference Separation

| Attribute | Training Environment | Inference / API Environment |
| :--- | :--- | :--- |
| **Platform** | Google Colab (T4 / L4 GPU) | Local / Workstation / Server API |
| **Compute** | NVIDIA T4 16GB VRAM | CUDA GPU (FP16) or CPU Fallback (FP32) |
| **Objective** | SFT with Cross-Entropy Loss Masking | Deterministic Greedy Generation (`do_sample=False`) |
| **Weights** | Full gradient computation with AdamW | Frozen `eval()` mode with SDPA attention |

---

## 5. Pipeline Flow: Stage 03 → Stage 04 → Stage 06

```
Stage 03 NLP Pipeline
  ├── Clinical Urgency Classification (SVM / BiLSTM / Bio_ClinicalBERT)
  ├── Medical Named Entity Recognition (Gene, Drug, Dose, Adverse Event)
  └── Raw Clinical Narrative
        ↓  (Zero Patient_ID / Target Leakage)
Stage 04 SLM Module
  ├── Qwen2.5-3B + PEFT LoRA Adapter
  ├── Faithful Prompt Format (ChatML Template)
  ├── Anti-Repetition Output Cleaner
  └── Clinical Safety & Anti-Hallucination Guardrail
        ↓
Stage 06 Multi-Agent System
  └── Consumes verified JSON payload for downstream clinical workflow coordination
```

### Stage 06 Downstream JSON Schema:
```json
{
  "summary": "This progress note documents a patient receiving Osimertinib at 80 mg/day with an EGFR L858R mutation. Reported symptoms include mild fatigue with an adverse event of rash noted.",
  "urgency": "High",
  "entities": {
    "gene_mutation": "EGFR L858R",
    "drug_name": "Osimertinib",
    "dosage": "80 mg/day",
    "adverse_event": "rash"
  },
  "safety_status": "passed",
  "safety_validation": {
    "passed": true,
    "flags": [],
    "metrics": {
      "total_flags": 0,
      "critical_flags": 0
    },
    "disclaimer": "RESEARCH PROTOTYPE NOTICE: ..."
  }
}
```

---

## 6. Safety Guardrail Rules

The safety guardrail (`safety_guardrail.py`) verifies the generated summary against the source text:
1. **Drug Hallucination Audit:** Flags any antineoplastic medication in the summary not present in the note.
2. **Dosage Grounding Audit:** Verifies numerical dosage specifications (`300mg`, `80 mg/day`, `BID`).
3. **Genomic Alteration Grounding:** Confirms mentioned mutations exist in the note.
4. **Adverse Event Verification:** Audits toxicities against documented adverse events.
5. **Prescription Directive Blocker:** Flags unauthorized recommendations (`"recommend starting"`, `"prescribe"`, `"increase dose"`, `"switch to"`).
6. **Autonomous Diagnosis Blocker:** Flags claims of definitive diagnosis or disease staging.

---

## 7. REST API Endpoints

### `GET /health`
Returns service readiness and active compute device.
```json
{
  "status": "ok",
  "stage": "Stage 04 SLM",
  "model": "Qwen/Qwen2.5-3B + LoRA (stage04_slm/model/adapter)",
  "device": "cpu"
}
```

### `POST /summarize`
Direct raw text summarization endpoint.
```json
// Request
{
  "clinical_report": "Trial screening EGFR L858R prior/current drug Pembrolizumab dose 300mg twice daily symptoms severe diarrhea ae nausea"
}

// Response
{
  "summary": "This trial screening note documents a patient receiving Pembrolizumab at 300mg twice daily with an EGFR L858R mutation. Symptoms include severe diarrhea with an adverse event of nausea."
}
```

### `POST /summarize_stage03`
Integrates Stage 03 structured context and outputs verified Stage 06 payload.

---

## 8. Quickstart & Execution Commands

### Step 1: Install Dependencies
```powershell
pip install -r stage04_slm/requirements.txt
```

### Step 2: Run Unit & Integration Tests
```powershell
python -m pytest stage04_slm/test_api.py -v
```

### Step 3: Start REST API
```powershell
uvicorn stage04_slm.api:app --host 127.0.0.1 --port 8004 --reload
```

### Step 4: Run Streamlit Clinical Dashboard (with Stage 04 Tab)
```powershell
streamlit run stage03_nlp/app.py
```

### Environment Variable Overrides
```powershell
$env:SLM_BASE_MODEL = "Qwen/Qwen2.5-3B"
$env:SLM_ADAPTER_PATH = "stage04_slm/model/adapter"
$env:SLM_DEVICE = "cuda" # or "cpu"
```
