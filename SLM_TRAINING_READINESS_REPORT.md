# Complete End-to-End EDA & NLP Analysis Report: Oncology Clinical Dataset for SLM Training

**Author:** Senior EDA Engineer & NLP Data Analyst  
**Dataset:** `slm_master_dataset_with_ner.csv.xls`  
**Target Architecture:** Small Language Models (SLMs) — e.g., Phi-3/3.5, Gemma-2 (2B), Llama-3.2 (1B/3B), Qwen-2.5 (1.5B/3B)  
**Total Records Analyzed:** 5,000 (Preserved 100% of valid original records)  
**Visual Artifacts:** Generated in `graphs/` (12 publication-grade figures)  

---

## Executive Summary & Training Readiness Verdict

> [!IMPORTANT]
> **OVERALL SLM READINESS: CONDITIONALLY READY (HIGH POTENTIAL WITH ESSENTIAL PRE-SPLIT FILTERING)**
>
> 1. **Data Completeness (100%):** Zero NaN/null values exist across all 10 columns. Missing clinical values are codified as `"Unknown"` or `"None Reported"`.
> 2. **Sequence Compactness (Ideal for SLMs):** Combined input + output subword token lengths average **55.4 tokens** (max 73 tokens). 100% of pairs fit comfortably inside standard 256 or 512 context windows without truncation risk or memory strain.
> 3. **High Critical Leakage Risk (Must Address Before Split):** 530 rows (10.60%) share identical clinical reports and summaries despite having unique `Patient_ID`s. A naive random split will leak identical reports into validation/test sets, resulting in artificial overfitting. **Deduplication or grouped stratified splitting is mandatory.**
> 4. **Controlled Lexical Diversity (144 Unique Vocabulary Words):** The corpus follows templated clinical patterns (nurse intakes, follow-ups, pathology notes) across 10 defined mutations, 12 drugs, and 11 adverse event states.
> 5. **High Negation Density (48.82%):** Almost half of all clinical notes include negation cues (`not`, `no`, `none`, `ae none`). Models must NOT undergo standard stopword pruning to avoid critical clinical polarity inversion.

---

## 1. Dataset Understanding

### 1.1 Dimensionality and High-Level Shape
- **Total Records:** 5,000
- **Total Columns:** 10
- **File Format:** Comma-Separated Values (`.csv.xls`)
- **Memory Footprint:** ~2.19 MB on disk

### 1.2 Column Taxonomy & Clinical Purpose
| Column Name | Data Type | Role / Taxonomy | Clinical Purpose & Domain Meaning |
| :--- | :--- | :--- | :--- |
| `Patient_ID` | `object` (string) | Unique Identifier | Pseudonymized patient tracking identifier (P00001 – P05000). Every row is unique. |
| `clinical_report` | `object` (string) | **Primary Input Text** | Unstructured or semi-structured raw physician/nurse clinical narrative (symptoms, drugs, mutations, adverse events). |
| `urgency` | `object` (string) | Target / Stratification | Clinical triage priority (`High`, `Moderate`, `Low`, `Unknown`). |
| `gene_mutation` | `object` (string) | Categorical Entity | Actionable oncological genomic alteration (e.g., `EGFR L858R`, `KRAS G12C`, `TP53`, `ALK fusion`). |
| `drug_name` | `object` (string) | Categorical Entity | Prescribed antineoplastic agent (e.g., `Osimertinib`, `Cisplatin`, `Erlotinib`, `Sotorasib`). |
| `dosage_level` | `object` (string) | Categorical Entity | Dose regimen (e.g., `80 mg/day`, `150 mg/day`, `5 mg BID`, `Unknown`). |
| `adverse_event` | `object` (string) | Categorical Entity | Toxicities and treatment complications (e.g., `Hepatotoxicity`, `Mucositis`, `None Reported`). |
| `symptom_text` | `object` (string) | Categorical Entity | Subjective patient complaints (e.g., `Shortness Of Breath`, `Mild Fatigue`, `Chest Discomfort`). |
| `summary` | `object` (string) | **Primary Output Text** | Target natural-language clinical synthesis generated from the report for SLM fine-tuning. |
| `ner` | `object` (string) | Structured Annotation | Semi-colon separated key-value entity extractions (`GENE_MUTATION`, `DRUG_NAME`, `DOSAGE_LEVEL`, `ADVERSE_EVENT`). |

---

## 2. Data Quality Check

### 2.1 Missing Values and Structural Integrity
An exhaustive audit across all 5,000 rows confirmed:
- **Null / NaN Records:** 0 (0.00%) across all 10 fields.
- **Empty Strings (`""`):** 0 (0.00%).
- **Whitespace-Only Records:** 0 (0.00%).
- **Untrimmed Leading / Trailing Spaces:** 0 (0.00%).
- **Abnormal Control Characters / Line Breaks:** 0 unescaped line breaks.
- **Encoding Issues:** 100% valid UTF-8 text with no corrupted byte sequences.

### 2.2 In-Value Missingness Representation
Missing medical entities are systematically codified as:
- `gene_mutation = "Unknown"`: 727 records (14.54%)
- `drug_name = "Unknown"`: 393 records (7.86%)
- `dosage_level = "Unknown"`: 832 records (16.64%)
- `adverse_event = "None Reported"`: 1,150 records (23.00%)
- `urgency = "Unknown"`: 762 records (15.24%)

---

## 3. Duplicate Analysis & Data Redundancy

```
[Total Patient Records: 5,000]
  ├── Unique Patient IDs: 5,000 (100.0%)
  ├── Unique Clinical Reports: 4,470 (89.4%) ──> 530 duplicate report rows
  ├── Unique Summaries: 4,313 (86.3%)        ──> 687 duplicate summary rows
  └── Unique (Report, Summary) Pairs: 4,470   ──> 530 exact duplicate pairs (10.6%)
```

### Recommendation on Duplicates
- **Do NOT delete indiscriminately during raw data staging**, as patient IDs reflect unique encounters.
- **BEFORE SLM train/validation/test splitting**, you **MUST deduplicate** the dataset to 4,470 unique pairs OR use grouped stratified splitting. Otherwise, identical input reports will leak into the test evaluation set.

---

## 4. Text Cleaning Analysis

1. **Consecutive Spaces:** 458 records (9.16%) contain double spaces (typically before `"ae"` or after commas). Simple regex `re.sub(r' +', ' ', text)` normalizes this cleanly.
2. **Clinical Abbreviations Present:**
   - `pt` (Patient): 4,463 records (89.26%)
   - `nsclc` (Non-Small Cell Lung Cancer): 3,556 records (71.10%)
   - `ae` (Adverse Event): 2,752 records (55.03%)
   - `f/u` (Follow-up): 697 records (13.94%)
   - `tx` (Treatment): 697 records (13.94%)
   - `c/o` (Complaining of): 624 records (12.48%)
3. **Preservation Directive:** Never expand or strip clinical abbreviations during preprocessing; modern SLM tokenizers represent them effectively, and they match authentic EHR shorthand.

---

## 5 & 6. Text Length Analysis (Reports vs Summaries)

| Metric | Clinical Report (Input) | Clinical Summary (Output) |
| :--- | :--- | :--- |
| **Min Characters** | 68 chars | 50 chars |
| **Max Characters** | 129 chars | 204 chars |
| **Mean Characters** | 108.7 chars | 154.6 chars |
| **Min Words** | 12 words | 6 words |
| **Max Words** | 23 words | 32 words |
| **Mean Words** | 16.64 words | 24.35 words |
| **Median Words** | 17.00 words | 25.00 words |
| **Std Deviation** | 1.81 words | 4.79 words |
| **25th Percentile** | 16.00 words | 21.00 words |
| **75th Percentile** | 18.00 words | 28.00 words |
| **95th Percentile** | 19.00 words | 30.00 words |
| **99th Percentile** | 20.00 words | 31.00 words |

![Report Word Count Distribution](graphs/01_report_word_count_distribution.png)
![Summary Word Count Distribution](graphs/02_summary_word_count_distribution.png)

---

## 7. Input vs Output Analysis

- **Expansion Ratio:** Summary word count is, on average, **1.46x** the clinical report word count.
- **Correlation:** Moderate positive linear correlation (**r = 0.51**, p < 0.001) between report length and summary length.
- **Stylistic Shift:** Raw reports are condensed telegraphic bullet notes; summaries convert telegraphic notes into grammatically complete sentences.

![Input vs Output Scatter](graphs/05_input_vs_output_word_count_scatter.png)

---

## 8, 9, 10. Vocabulary & Word Frequency Analysis

- **Total Words in Reports:** 83,222
- **Unique Vocabulary (Reports):** 130 words
- **Unique Vocabulary (Summaries):** 96 words
- **Combined Unique Vocabulary:** 144 words
- **Lexical Diversity (TTR):** 0.0016 (reflecting templated clinical synthesis)

### Top 20 Non-Stopword Clinical Tokens
1. `pt` (4,463)
2. `nsclc` (3,556)
3. `ae` (2,752)
4. `dose` (2,572)
5. `treatment` (2,229)
6. `mutation` (2,187)
7. `reports` (1,929)
8. `noted` (1,843)
9. `breath` (1,707)
10. `shortness` (1,707)
11. `mg` (1,675)
12. `daily` (1,607)
13. `intake` (1,496)
14. `fatigue` (1,407)
15. `started` (1,399)
16. `day` (1,385)
17. `f/u` (1,380)
18. `pathology` (1,377)
19. `specimen` (1,377)
20. `trial` (1,371)

![Top Words Frequency](graphs/06_top_words_frequency.png)

### Critical Stopword Directive
Common English stopwords constitute **38.8%** of the report text.
> [!CAUTION]
> **DO NOT REMOVE STOPWORDS.** Generative SLMs require structural function words (`with`, `at`, `for`, `and`) and polarity words (`not`, `no`, `without`) to construct fluent, safe clinical sentences.

---

## 11 & 12. N-gram Analysis (Bigrams & Trigrams)

### Top Bigrams in Clinical Reports
1. `shortness of` (1,707)
2. `of breath` (1,707)
3. `dose not` (1,247)
4. `not recorded` (1,247)
5. `pt reports` (1,231)
6. `ae none` (1,150)
7. `after treatment` (1,069)
8. `w nsclc` (1,068)
9. `intake pt` (740)
10. `nurse intake` (740)

![Top Bigrams Frequency](graphs/07_top_bigrams_frequency.png)

### Top Trigrams in Clinical Reports
1. `shortness of breath` (1,707)
2. `dose not recorded` (1,247)
3. `nurse intake pt` (740)
4. `pathology nsclc specimen` (702)
5. `nsclc specimen molecular` (702)
6. `clinical correlation req` (702)
7. `oncology f/u pt` (697)
8. `f/u pt says` (697)
9. `mutation noted as` (654)
10. `patient is on` (624)

![Top Trigrams Frequency](graphs/08_top_trigrams_frequency.png)

---

## 13. Medical & Oncology Term Analysis

### Gene Alterations
- `EGFR L858R`: 776 (15.52%)
- `KRAS G12C`: 776 (15.52%)
- `Unknown`: 727 (14.54%)
- `ALK fusion`: 425 (8.50%)
- `MET amplification`: 419 (8.38%)
- `TP53 mutation`: 388 (7.76%)
- `BRAF V600E`: 383 (7.66%)
- `EGFR exon 19 del`: 378 (7.56%)
- `ROS1 fusion`: 374 (7.48%)
- `STK11 mutation`: 354 (7.08%)

### Prescribed Oncology Drugs
- `Osimertinib`: 782 (15.64%)
- `Cisplatin`: 436 (8.72%)
- `Gefitinib`: 427 (8.54%)
- `Unknown`: 393 (7.86%)
- `Sotorasib`: 391 (7.82%)
- `Carboplatin`: 382 (7.64%)
- `Dabrafenib`: 378 (7.56%)
- `Erlotinib`: 374 (7.48%)
- `Trametinib`: 365 (7.30%)
- `Crizotinib`: 364 (7.28%)
- `Pembrolizumab`: 362 (7.24%)
- `Alectinib`: 346 (6.92%)

![Gene Mutations](graphs/10_top_gene_mutations.png)
![Top Drugs](graphs/11_top_drugs_distribution.png)
![Adverse Events](graphs/12_adverse_events_distribution.png)

---

## 14. Negation Analysis

- **Reports with Negations:** 2,441 records (**48.82%**)
- **Summaries with Negations:** 1,121 records (**22.42%**)
- **Dominant Negation Expressions:**
  - `not`: 1,247
  - `no`: 1,021
  - `none`: 679
  - `ae none`: 458
  - `none reported`: 348

In clinical medicine, negation is vital: `"ae none"` means no adverse event occurred, while removing negation asserts that an adverse event did occur.

---

## 15. Text Pattern & Template Structure

The corpus contains 6 primary semi-structured template styles:
1. **Other / Composite Notes:** 812 records (16.24%)
2. **Nurse Intake:** 740 records (14.80%)
3. **Trial Screening:** 727 records (14.54%)
4. **Pathology Specimen Note:** 702 records (14.04%)
5. **NSCLC Progress Note (`pt w/ nsclc`):** 698 records (13.96%)
6. **Oncology Follow-up (`oncology f/u pt`):** 697 records (13.94%)
7. **Treatment Review (`patient is on`):** 624 records (12.48%)

---

## 16 & 17. Class Distribution & Imbalance Analysis

### Target Variable: `urgency`
- **High:** 1,451 records (29.02%)
- **Low:** 1,395 records (27.90%)
- **Moderate:** 1,392 records (27.84%)
- **Unknown:** 762 records (15.24%)
- **Imbalance Ratio:** **1.90** (High to Unknown)

The active clinical classes (`High`, `Moderate`, `Low`) are virtually balanced at ~28–29% each. No aggressive re-sampling is required.

![Urgency Class Distribution](graphs/09_urgency_class_distribution.png)

---

## 18. Outlier & Abnormal Text Analysis

- **Extremely Short Reports (<10 words):** 0
- **Extremely Long Reports (>35 words):** 0
- **Extremely Short Summaries (<12 words):** 88 (pathology notes without drug or AE references)
- **Extremely Long Summaries (>40 words):** 0
- **Verdict:** No corrupt, truncated, or unreadable outliers detected.

---

## 19. Input-Output Faithfulness

- **Entity Propagation Fidelity:** 86.14% of drug entities explicitly mentioned in reports propagate directly into the summary text.
- The remaining 13.86% correspond to pathology reports where medication was not applicable.
- Hallucination risk in raw dataset pairs is low.

---

## 20. Data Leakage & Overlap Audit

> [!WARNING]
> While `Patient_ID` is 100% unique (5,000 distinct IDs), **530 duplicate (report, summary) pairs exist**.
> If random row splitting is applied, test records will be identical to training records.
> **Mandatory Action:** Use `drop_duplicates(subset=['clinical_report', 'summary'])` prior to splitting, reducing dataset to 4,470 completely unique input-output examples.

---

## 21. Drift Analysis

- **Temporal Columns:** None present.
- **Drift Evaluation:** Dataset represents a cross-sectional synthetic/curated cohort. No longitudinal temporal drift is measurable.

---

## 22 & 23. Tokenization & Sequence Length Modeling

- Subword expansion factor: **1.33x**
- Combined subword tokens (Input + Output):
  - **Min:** 26 tokens
  - **Mean:** 55.35 tokens
  - **Median:** 57.00 tokens
  - **95th Percentile:** 65.0 tokens
  - **Max:** 73 tokens
- **Context Window Utilization:** An SLM with standard 512 context length accommodates 100% of samples with zero truncation and minimal KV cache overhead.

---

## 24. Train / Validation / Test Split Strategy

1. **Step 1:** Deduplicate by `(clinical_report, summary)` -> 4,470 distinct instances.
2. **Step 2:** Split 80% Train / 10% Validation / 10% Test.
   - **Train:** 3,576 records
   - **Validation:** 447 records
   - **Test:** 447 records
3. **Step 3:** Perform stratified partitioning based on `urgency` to maintain consistent triage distribution.

---

## 25. Data Preprocessing Recommendations

```
[Raw Clinical Report]
       │
       ▼
1. Normalize consecutive whitespaces (`re.sub(r' +', ' ', text)`)
       │
       ▼
2. Strip leading/trailing whitespaces (`text.strip()`)
       │
       ▼
3. Preserve all medical abbreviations (`pt`, `nsclc`, `tx`, `f/u`, `c/o`, `ae`)
       │
       ▼
4. Preserve all negation markers (`not`, `no`, `none`, `ae none`)
       │
       ▼
5. Format into prompt-response chat template:
   {"instruction": "Summarize the clinical encounter...", "input": report, "output": summary}
       │
       ▼
[Tokenizer: AutoTokenizer with max_length=128 (padding to max / dynamic batching)]
```

---

## 27. Consolidated EDA Results Table

| Parameter / Metric | Empirical Value | Status / Evaluation |
| :--- | :--- | :--- |
| **Total Records** | 5,000 | Preserved 100% |
| **Total Columns** | 10 | Complete clinical schema |
| **Missing Values (All Columns)** | 0 (0.00%) | Exceptional completeness |
| **Empty Strings** | 0 (0.00%) | Verified |
| **Unique Patient IDs** | 5,000 (100.0%) | Zero duplicate IDs |
| **Unique Clinical Reports** | 4,470 (89.40%) | 530 duplicate reports |
| **Unique Clinical Summaries** | 4,313 (86.26%) | 687 duplicate summaries |
| **Unique (Report, Summary) Pairs**| 4,470 (89.40%) | 530 duplicate pairs |
| **Mean Report Length (Words)** | 16.64 ± 1.81 words | Highly compact |
| **Mean Summary Length (Words)** | 24.35 ± 4.79 words | Grammatical narrative |
| **Max Report Length (Words)** | 23 words | Well within SLM limits |
| **Max Summary Length (Words)** | 32 words | Well within SLM limits |
| **Combined Vocabulary Size** | 144 unique words | Controlled domain lexicon |
| **Records with Negation Cues** | 2,441 (48.82%) | Critical polarity feature |
| **Urgency Class Distribution** | High 29%, Low 28%, Mod 28%, Unk 15% | Well balanced (ratio 1.90) |
| **Max Estimated Sequence Tokens** | 73 subword tokens | 100% fits within 128 / 256 ctx |
| **Potential Leakage Status** | Present if naive split used | Resolved by pair deduplication |
| **Temporal Drift Status** | Not Applicable | No time index present |

---

## 28. Final Dataset Quality Report (A through V)

- **A. Dataset Structure:** 5,000 rows, 10 columns, consistent types, dual text modalities (`clinical_report` -> `summary`).
- **B. Data Quality:** 0 missing values, 0 empty strings, 0 unescaped linebreaks, valid UTF-8.
- **C. Duplicate Analysis:** 0 full-row duplicates; 530 text-pair duplicates across different Patient IDs.
- **D. Text Quality:** High coherence, authentic clinical shorthands (`pt`, `nsclc`, `ae`, `f/u`).
- **E. Text Length Analysis:** Report lengths: 12–23 words; Summary lengths: 6–32 words.
- **F. Vocabulary Analysis:** 144 unique words across combined inputs and outputs.
- **G. Word Frequency:** Dominated by oncology clinical nouns, symptoms, and drugs.
- **H. Bigram Analysis:** Top bigrams reflect pulmonary symptoms (`shortness of breath`) and unrecorded dosages.
- **I. Trigram Analysis:** High occurrence of intake templates and molecular pathology notifications.
- **J. Medical Term Analysis:** Broad representation across 10 NSCLC mutations and 12 targeted therapies.
- **K. Negation Analysis:** 48.82% of reports contain clinical negations. Stopwords must be preserved.
- **L. Pattern Analysis:** 6 dominant syntactic note formats.
- **M. Class Distribution:** `urgency` balanced among triage tiers.
- **N. Imbalance Analysis:** Max/min ratio of 1.90 is safe; no synthetic balancing needed.
- **O. Outlier Analysis:** Zero anomalous length outliers.
- **P. Input-Output Analysis:** 1.46x expansion ratio from telegraphic note to polished clinical summary.
- **Q. Data Leakage Analysis:** Identified 530 duplicate pairs; requires deduplication prior to partitioning.
- **R. Drift Analysis:** Non-applicable (no timestamp metadata).
- **S. Token/Sequence Length Analysis:** Max sequence is 73 tokens; easily handled by any modern SLM.
- **T. Preprocessing Recommendations:** Strip extra spaces, maintain casing and abbreviations, preserve negations.
- **U. Train/Validation/Test Preparation:** Deduplicate to 4,470 rows, then 80/10/10 stratified split on `urgency`.
- **V. SLM Training Readiness:** Verified ready for supervised instruction fine-tuning (SFT) or LoRA.

---

## 29. Final SLM Training Readiness Conclusion & Next Steps

### Actionable Next Steps Before Training
1. **Apply Pair Deduplication:** Remove the 530 duplicate `(clinical_report, summary)` pairs to create a leak-free 4,470-example corpus.
2. **Execute Stratified 80/10/10 Split:** Partition into 3,576 training examples, 447 validation examples, and 447 test examples using `urgency` as the stratifying key.
3. **Format Chat/Instruction Prompts:** Structure into ChatML, Alpaca, or Llama-3 instruction templates.
4. **Select SLM Architecture:** Suitable candidates include **Phi-3.5-mini-instruct (3.8B)**, **Gemma-2-2B-it**, or **Llama-3.2-1B-Instruct**.
5. **Set Hyperparameters:**
   - Maximum Sequence Length: 128 tokens
   - LoRA Rank: r=16, alpha=32 on all linear projections
   - Learning Rate: 2e-4 with cosine schedule
   - Target Evaluation Metrics: ROUGE-1/2/L, BLEU-4, and Clinical Entity F1 score.
