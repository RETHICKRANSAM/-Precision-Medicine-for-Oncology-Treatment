# Data Leakage & Integrity Audit Report: Stage 04 SLM

**Execution Timestamp:** 2026-09-10  
**Pipeline Stage:** Stage 04 — Small Language Model (SLM) Fine-Tuning  
**Dataset Reference:** `stage_4/slm_master_dataset_deduplicated_clean.csv`  

---

## 1. Executive Summary & Audit Verdict

> [!IMPORTANT]
> **AUDIT VERDICT: PASSED (ZERO DATA LEAKAGE VERIFIED)**  
> All 4,470 records in the deduplicated clinical dataset have undergone an exhaustive multi-dimensional data leakage audit.
> - **Patient Overlap:** 0 (0.00%)
> - **Clinical Report Overlap:** 0 (0.00%)
> - **Target / Label Leakage in Input:** 0 (0.00%)
> - **Missing Values (NaN/Null):** 0 (0.00%)
> - **Patient_ID Feature Leakage:** `Patient_ID` is strictly excluded from prompt templates and training features.

---

## 2. Dataset Provenance & Pre-Split Inspection

### Raw Dataset vs Cleaned Dataset Audit
- **Raw Dataset (`slm_master_dataset_with_ner.csv (1).xls`):** 5,000 total rows
  - Unique `Patient_ID`s: 5000 (100%)
  - Unique `clinical_report`s: 4470 (89.40%)
  - Duplicate Clinical Reports: 530 (10.60%)
  - *Finding:* 530 rows contained identical clinical narratives under distinct patient identifiers.
  - *Prior Artifact Leakage:* The original `slm_train_sft.jsonl`, `slm_validation_sft.jsonl`, and `slm_test_sft.jsonl` were split prior to deduplication, causing **51 Train/Val overlaps, 57 Train/Test overlaps, and 20 Val/Test overlaps**.
- **Deduplicated Clean Master (`slm_master_dataset_deduplicated_clean.csv`):** 4,470 total rows
  - Unique `Patient_ID`s: 4,470 (100%)
  - Unique `clinical_report`s: 4,470 (100%)
  - Duplicate rows: 0 (0.00%)


### Missing Value Audit (Clean Master)
| Column | Missing Count | Percentage | Codification Standard |
| :--- | :---: | :---: | :--- |
| `Patient_ID` | 0 | 0.00% | Pseudonymized string (`P00001`–`P05000`) |
| `clinical_report` | 0 | 0.00% | Raw narrative (Input) |
| `urgency` | 0 | 0.00% | Standardized triage category (`High`, `Moderate`, `Low`, `Unknown`) |
| `gene_mutation` | 0 | 0.00% | Actionable genomic variant or `'Unknown'` |
| `drug_name` | 0 | 0.00% | Prescribed oncology agent or `'Unknown'` |
| `dosage_level` | 0 | 0.00% | Dose regimen or `'Unknown'` |
| `adverse_event` | 0 | 0.00% | Documented toxicity or `'None Reported'` |
| `symptom_text` | 0 | 0.00% | Patient symptoms or `'Unknown'` |
| `summary` | 0 | 0.00% | Target reference summary (Output) |
| `ner` | 0 | 0.00% | Structured entity extraction |

---

## 3. Train / Validation / Test Splitting Strategy

The dataset was partitioned using **patient-level stratified sampling** based on clinical triage urgency to maintain representative class proportions without data contamination.

- **Split Ratio:** 80% Train / 10% Validation / 10% Holdout Test
- **Random Seed:** `42` (Deterministic and reproducible)

### Split Distribution
| Split | Record Count | Percentage | Unique Patients | Unique Clinical Reports |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 3576 | 80.0% | 3576 | 3576 |
| **Validation** | 447 | 10.0% | 447 | 447 |
| **Test (Holdout)** | 447 | 10.0% | 447 | 447 |
| **Total** | **4470** | **100.0%** | **4470** | **4470** |

---

## 4. Multi-Dimensional Cross-Split Contamination Audit

```
┌─────────────────────────────────────────────────────────────┐
│               CROSS-SPLIT CONTAMINATION CHECK               │
├───────────────────────────────┬──────────────┬──────────────┤
│ Metric                        │ Count Found  │ Status       │
├───────────────────────────────┼──────────────┼──────────────┤
│ Train & Validation Patient ID │ 0            │ PASSED (0.0%)│
│ Train & Test Patient ID       │ 0            │ PASSED (0.0%)│
│ Validation & Test Patient ID  │ 0            │ PASSED (0.0%)│
│ Train & Validation Report     │ 0            │ PASSED (0.0%)│
│ Train & Test Report           │ 0            │ PASSED (0.0%)│
│ Validation & Test Report      │ 0            │ PASSED (0.0%)│
└───────────────────────────────┴──────────────┴──────────────┘
```

---

## 5. Input Prompt Construction & Target Leakage Prevention

To ensure model generalizability and prevent trivial shortcuts:
1. **Instruction Formatting:** The model input contains only the instruction and the raw clinical note:
   ```json
   {
     "instruction": "Summarize the following oncology clinical report faithfully.",
     "input": "<clinical report>",
     "output": "<reference summary>"
   }
   ```
2. **Target Summary Exclusion:** The reference clinical summary never appears in the input prompt or instruction.
3. **Identifier Stripping:** `Patient_ID` is strictly removed from training and inference prompts.
4. **Holdout Integrity:** The test set (`stage4_slm/data/test.jsonl`, 447 samples) is strictly locked and will only be evaluated once at final checkpoint evaluation.
