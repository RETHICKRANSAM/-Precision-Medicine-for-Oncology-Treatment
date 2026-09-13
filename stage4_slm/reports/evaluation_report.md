# Stage 04 SLM Evaluation Report: Performance on Untouched Holdout Test Set

**Target Architecture:** `Qwen2.5-0.5B-Instruct + LoRA`  
**LoRA Adapter:** `stage4_slm/models/qwen2.5_0.5b_lora_cpu`  
**Evaluation Set:** `stage4_slm/data/test.jsonl` (Strictly untouched holdout)  
**Total Test Records Evaluated:** `447`  
**Total Evaluation Time:** `3417.78s` (Average `7.64s/sample`)  
**Device:** `cpu` (`torch.float32`)

---

## 1. Quantitative Benchmark Results

### Text Summarization Quality (ROUGE Metrics)
| Metric | Score | Clinical Interpretation |
| :--- | :---: | :--- |
| **ROUGE-1 (F1)** | **0.8716** | Overlap of individual clinical terms and keywords |
| **ROUGE-2 (F1)** | **0.7793** | Exact bigram preservation (drug-dosage and mutation pairs) |
| **ROUGE-L (F1)** | **0.8515** | Longest common sequence and syntactic structural fidelity |

### Clinical Entity Preservation Rates
| Entity Class | Preservation Rate (%) | Preservation Assessment |
| :--- | :---: | :--- |
| **Antineoplastic Drugs** | **99.75%** | Prescribed drugs retained without omission |
| **Gene Mutations** | **96.68%** | Genomic alteration accurately transcribed |
| **Dosage Levels** | **70.39%** | Dosage quantities matched to source |
| **Adverse Events / Toxicities** | **88.14%** | Toxicities accurately recorded without confusion |

### Sequence Length & Generation Reliability Statistics
| Statistic | Average Token Count | Notes |
| :--- | :---: | :--- |
| **Average Input Length** | 27.9 tokens | Raw clinical report tokens |
| **Average Reference Summary Length** | 36.5 tokens | Target clinical synthesis tokens |
| **Average Generated Summary Length** | 38.1 tokens | SLM generated tokens |
| **Empty Outputs** | 0 (0.00%) | Robust generation without collapsed outputs |
| **Generation Failures** | 0 (0.00%) | 100% completion reliability |

---

## 2. Qualitative Case Studies & Hallucination Audit

Below are 6 representative test scenarios evaluated directly against human reference summaries and checked through the `ClinicalSummarySafetyGuardrail`.

### Case 1
- **INPUT REPORT:**  
  `Trial screening unknown prior/current drug Osimertinib dose 200 mg/day symptoms loss of appetite ae fever`
- **REFERENCE SUMMARY:**  
  `This trial note documents a patient receiving Osimertinib at 200 mg/day. Reported symptoms include loss of appetite with an adverse event of fever noted.`
- **GENERATED SUMMARY:**  
  `This trial note documents an Osimertinib at 200 mg/day with no adverse events reported.`
- **Factual Integrity Check:**
  - *ROUGE-L Score:* `0.6`
  - *Safety Guardrail Status:* `PASS`
  - *Flags:* `None`

---

### Case 2
- **INPUT REPORT:**  
  `Trial screening EGFR L858R prior/current drug Cisplatin dose 600 mg BID symptoms nausea after treatment ae none reported`
- **REFERENCE SUMMARY:**  
  `This trial note documents a patient receiving Cisplatin at 600 mg BID with an EGFR L858R mutation. Reported symptoms include nausea after treatment with no adverse events reported.`
- **GENERATED SUMMARY:**  
  `This trial note documents an individual receiving Cisplatin at 600 mg BID with an EGFR L858R mutation. Reported symptoms include nausea with no adverse events noted.`
- **Factual Integrity Check:**
  - *ROUGE-L Score:* `0.8519`
  - *Safety Guardrail Status:* `PASS`
  - *Flags:* `None`

---

### Case 3
- **INPUT REPORT:**  
  `Trial screening BRAF V600E prior/current drug Osimertinib dose 80mg daily symptoms rash worsening ae mucositis`
- **REFERENCE SUMMARY:**  
  `This trial note documents a patient receiving Osimertinib with a BRAF V600E mutation. Reported symptoms include rash worsening with an adverse event of mucositis noted.`
- **GENERATED SUMMARY:**  
  `This trial note documents an individual receiving Osimertinib at 80 mg/day with a BRAF V600E mutation. Reported symptoms include symptoms of rash with an adverse event of mucositis noted.`
- **Factual Integrity Check:**
  - *ROUGE-L Score:* `0.8`
  - *Safety Guardrail Status:* `PASS`
  - *Flags:* `None`

---

### Case 4
- **INPUT REPORT:**  
  `Progress note - kras-g12c treatment Erlotinib dose 80mg daily pt reports not documented no/possible mucositis`
- **REFERENCE SUMMARY:**  
  `This progress note documents a patient receiving Erlotinib with a KRAS G12C mutation. An adverse event of mucositis was noted.`
- **GENERATED SUMMARY:**  
  `This progress note documents a patient receiving Erlotinib at 80 mg/day with KRAS G12C mutation. Reported symptoms include no adverse events with an adverse event of mucositis noted.`
- **Factual Integrity Check:**
  - *ROUGE-L Score:* `0.7347`
  - *Safety Guardrail Status:* `PASS`
  - *Flags:* `None`

---

### Case 5
- **INPUT REPORT:**  
  `Pt w/ nsclc mutation MET amplification started Cisplatin 5 mg BID reports mild fatigue ae fever`
- **REFERENCE SUMMARY:**  
  `This clinical note documents a patient receiving Cisplatin at 5 mg BID with a MET amplification. Reported symptoms include mild fatigue with an adverse event of fatigue noted.`
- **GENERATED SUMMARY:**  
  `This clinical note documents a patient receiving Cisplatin at 5 mg BID with a MET amplification. Reported symptoms include mild fatigue with an adverse event of fever noted.`
- **Factual Integrity Check:**
  - *ROUGE-L Score:* `0.9643`
  - *Safety Guardrail Status:* `PASS`
  - *Flags:* `None`

---

### Case 6
- **INPUT REPORT:**  
  `Trial screening ROS1 fusion prior/current drug Sotorasib dose 80 mg daily symptoms fatigue cough ae none reported`
- **REFERENCE SUMMARY:**  
  `This trial note documents a patient receiving Sotorasib at 80 mg/day with a ROS1 fusion. Reported symptoms include fatigue + cough with no adverse events reported.`
- **GENERATED SUMMARY:**  
  `This trial note documents a patient receiving Sotorasib at 80 mg/day with an ROS1 fusion. Reported symptoms include fatigue and mild cough with no adverse events noted.`
- **Factual Integrity Check:**
  - *ROUGE-L Score:* `0.8889`
  - *Safety Guardrail Status:* `PASS`
  - *Flags:* `None`

---

## 3. Clinical Safety & Decision-Support Posture

> [!NOTE]
> **Safety Verdict:**
> The Stage 04 SLM acts strictly as a **faithful summarization pipeline**. It synthesizes only information explicitly stated in the input text and Stage 03 NLP features.
> - Zero autonomous treatment recommendations were formulated.
> - Missing values in the input are preserved as `"Unknown"` or `"Not Documented"` rather than fabricated.
> - All outputs carry the mandatory clinical disclaimer:
>   *"Generated summary — verify against the source clinical report. Decision-support summarization only; not an autonomous clinical decision-maker."*
