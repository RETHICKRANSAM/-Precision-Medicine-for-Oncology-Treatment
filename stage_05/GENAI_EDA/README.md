# Precision Oncology GenAI Clinical Text Summarization — EDA Pipeline

## 1. Project Purpose
This repository contains the complete, production-grade Exploratory Data Analysis (EDA), data profiling, leakage audit, and dataset preparation pipeline for the **GenAI Clinical Text Summarization** project.

The system is designed to summarize unstructured, shorthand clinical oncology notes into formal, standardized clinical summaries:
- **INPUT Feature:** `clinical_report` (noisy clinical consultation notes, nurse intake notes, pathology reports)
- **TARGET Feature:** `summary` (complete, synthesized medical narrative)

---

## 2. Dataset Overview
- **Source File:** Precision Oncology Master Dataset (`genai_cleaned_master.csv`)
- **Total Records:** 4,749 rows
- **Raw Features:** 9 columns
  1. `clinical_report` (Text, Model Input)
  2. `summary` (Text, Model Target)
  3. `ner` (Semi-structured, Audit Only)
  4. `gene_mutation` (Categorical, Audit Only)
  5. `drug_name` (Categorical, Audit Only)
  6. `dosage_level` (Text/Categorical, Audit Only)
  7. `adverse_event` (Categorical, Audit Only)
  8. `symptom_text` (Categorical, Audit Only)
  9. `urgency` (Categorical, Audit Only)
- **Patient_ID Status:** **NOT PRESENT** in this dataset and must not be fabricated. Grouped train/validation/test isolation is performed on `clinical_report`.

---

## 3. Mandatory Model Input & Target Rules
### Model Input
Only `clinical_report` may be fed into the GenAI summarization model.

### Excluded Model Features
The following 7 auxiliary columns are strictly **EXCLUDED FROM GENAI MODEL INPUT**:
- `ner`
- `gene_mutation`
- `drug_name`
- `dosage_level`
- `adverse_event`
- `symptom_text`
- `urgency`

> [!CAUTION]
> **Data Leakage Warning:** Audit findings prove that 100% of known drug names and gene mutations appear verbatim inside the target `summary`. These auxiliary fields represent an upstream decomposition of the target text itself. Passing them as model input causes catastrophic target leakage, enabling models to perform trivial slot-filling rather than true clinical language summarization.

---

## 4. How to Execute the EDA Pipeline
To install dependencies and run the complete pipeline end-to-end:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Execute the complete EDA pipeline
python eda.py
```

---

## 5. Folder Structure
```
GENAI_EDA/
│
├── data/
│   ├── raw/                           # Untouched working copy of source data
│   ├── processed/                     # Grouped train/val/test splits
│   │   ├── genai_train.csv            # 80% train split (3,800 rows)
│   │   ├── genai_validation.csv       # 10% validation split (471 rows)
│   │   └── genai_test.csv             # 10% held-out test split (478 rows)
│   └── eda_ready/                     # Standardized dataset for EDA ingestion
│       └── genai_eda_ready.csv
│
├── graphs/
│   ├── overview/                      # High-level dataset summary cards
│   ├── data_quality/                  # Missing values and duplicate breakdown
│   ├── text_analysis/                 # Length dists, ratio dists, input vs target
│   ├── categorical_analysis/          # Distributions for mutations, drugs, AEs
│   ├── relationship_analysis/         # Text length vs categorical strata
│   ├── outlier_analysis/              # Non-parametric IQR boxplots
│   └── leakage_analysis/              # Overlap percentages and target leakage
│
├── reports/
│   ├── EDA_Report.md                  # Master Comprehensive 19-Section Report
│   ├── Data_Profile_Report.md         # Schema, dimensions, memory, and roles
│   ├── Data_Quality_Report.md         # Nulls, formatting, and duplicates
│   ├── Text_Analysis_Report.md        # Lengths, percentiles, and vocabulary
│   ├── Categorical_Analysis_Report.md  # Categorical frequencies and anomalies
│   ├── Relationship_Analysis_Report.md# Length vs attribute correlation analysis
│   ├── Outlier_Analysis_Report.md     # IQR outlier detection and classifications
│   ├── Leakage_Report.md              # Target leakage and exclusion rationale
│   ├── Split_Report.md                # Grouped partitioning verification
│   └── GenAI_Handoff_Report.md        # Technical guidance for GenAI Engineer
│
├── outputs/
│   ├── dataset_summary.csv            # High-level metadata metrics
│   ├── column_summary.csv             # Field-by-field schema, nulls, and roles
│   ├── missing_value_report.csv       # Actual nulls vs explicit Unknowns
│   ├── duplicate_report.csv           # 6-way duplicate analysis
│   ├── categorical_summary.csv        # Top/least common categories
│   ├── text_statistics.csv            # Text metric quantiles (min to p99)
│   ├── vocabulary_statistics.csv      # TTR, hapax legomena, top tokens
│   ├── relationship_summary.csv       # Mean/median length per category
│   ├── outlier_report.csv             # IQR outlier record details
│   ├── leakage_report.csv             # Exact and substring target overlap
│   ├── low_information_records.csv    # Flagged sparse/template records
│   └── suspicious_records.csv         # Whitespace and punctuation anomalies
│
├── eda.py                             # Master pipeline execution script
├── requirements.txt                   # Dependency specifications
└── README.md                          # Project documentation
```

---

## 6. Grouped Split Methodology
- **Grouping Key:** `clinical_report`
- **Random Seed:** `42`
- **Ratio:** 80% Train / 10% Validation / 10% Test
- **Guaranteed Isolation:** Zero overlapping `clinical_report` strings across splits (`groupby('clinical_report')['split'].nunique() == 1`).

---

## 7. Next Stage: GenAI / ML Model Development
The dataset is prepared and certified. Proceed to:
1. Fine-tuning Seq2Seq LLMs (e.g. Flan-T5, BART, LLaMA-3, Mistral, Med-Gemma) using `clinical_report` as the input prompt and `summary` as the target generation.
2. Evaluating with ROUGE-1/2/L, BERTScore, and clinical factuality checks.
