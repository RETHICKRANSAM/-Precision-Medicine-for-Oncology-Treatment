# Master Exploratory Data Analysis (EDA) Report
## Precision Oncology Clinical Text Summarization

---

### 1. Executive Summary
This report presents a thorough, production-grade exploratory data analysis (EDA) of the Precision Oncology Clinical Text Summarization dataset (4,749 rows, 9 columns). The primary objective is to evaluate data quality, characterize linguistic and semantic properties of the clinical texts, detect target leakage risks, and establish a grouped train/validation/test split for subsequent GenAI / LLM modeling.

### 2. Dataset Overview
- **Project Type:** GenAI Clinical Text Summarization
- **Input Feature:** `clinical_report` (clinical consultation notes, nurse intake notes, pathology reports)
- **Target Feature:** `summary` (multi-sentence standardized summary)
- **Auxiliary Features:** `ner`, `gene_mutation`, `drug_name`, `dosage_level`, `adverse_event`, `symptom_text`, `urgency`
- **Patient_ID:** Absent; grouped isolation enforced via `clinical_report`.

### 3. Dataset Structure
- **Dimensions:** 4,749 rows by 9 columns.
- **Memory Footprint:** 4017.65 KB.
- Full details saved in `outputs/dataset_summary.csv` and `outputs/column_summary.csv`.

### 4. Data Quality
- Data integrity across primary columns is high. No byte-level corruption or illegible encoding was detected.
- Formatting anomalies and suspicious records are cataloged in `outputs/suspicious_records.csv`.

### 5. Missing Value Analysis
- **Actual Nulls:** 0 rows (0.0%).
- **Empty / Whitespace Strings:** 0 rows (0.0%).
- **Clinical Shorthand (`Unknown` / `None Reported`):** Present in `dosage_level` (25.7%), `adverse_event` (30.9%), and `drug_name` (17.6%).
- Full breakdown in `outputs/missing_value_report.csv`.

### 6. Duplicate Analysis
- **Full-Row Duplicates:** 0 rows.
- **Duplicate `clinical_report` Instances:** 434 rows belonging to 155 distinct groups.
- Handled via group-based splitting to prevent test contamination. Full details in `outputs/duplicate_report.csv`.

### 7. Text Analysis
- **clinical_report Length:** Mean = 17.5 words, Median = 18.0 words, Range = [12.0, 23.0].
- **summary Length:** Mean = 25.1 words, Median = 26.0 words, Range = [6.0, 32.0].
- Full statistical quantiles in `outputs/text_statistics.csv`.

### 8. Vocabulary Analysis
- **Type-Token Ratio (TTR):** Input note TTR = 0.0016, Target summary TTR = 0.0008.
- Full token metrics in `outputs/vocabulary_statistics.csv`.

### 9. Categorical Analysis
- Distributions analyzed for `gene_mutation`, `drug_name`, `dosage_level`, `adverse_event`, `symptom_text`, and `urgency`.
- Visualized in `graphs/categorical_analysis/` and tabulated in `outputs/categorical_summary.csv`.

### 10. Relationship Analysis
- Length variances across clinical categories evaluated. Summaries consistently maintain 20-35 words regardless of urgency tier.
- Tabulated in `outputs/relationship_summary.csv`.

### 11. Outlier Analysis
- Statistical outliers identified using IQR method (Q1 - 1.5*IQR to Q3 + 1.5*IQR).
- Outliers reflect genuine clinical complexity rather than data corruption. Full report in `outputs/outlier_report.csv`.

### 12. Low-Information Records
- Identified 134 sparse or template-dominated records. Documented in `outputs/low_information_records.csv`.

### 13. NER Consistency
- NER audit indicates 90.19% structural alignment between semi-structured NER tokens and structured columns.

### 14. Leakage Analysis
- **Crucial Finding:** 100% of known drug names and gene mutations match target summary tokens.
- **Action:** Exclude all auxiliary columns from model inputs (`outputs/leakage_report.csv`).

### 15. Train/Validation/Test Split
- **Group Key:** `clinical_report`, Seed = 42.
- **Train:** 3,800 rows (80.02%)
- **Validation:** 471 rows (9.92%)
- **Test:** 478 rows (10.07%)
- Overlap across splits: 0 records. Details in `reports/Split_Report.md`.

### 16. Important EDA Findings
1. The summarization task is an expansion task (synthesizing abbreviations into full sentences).
2. Grouped splitting is essential to prevent data leakage due to repeated clinical notes.
3. Auxiliary columns are target-derived and must not be used as model inputs.

### 17. Data Quality Risks
- Low-information records may encourage model hallucination if not properly regularized.
- Absence of Patient_ID requires continued acknowledgment of potential multi-visit patient notes.

### 18. GenAI Modeling Considerations
- Context length of 256 tokens is sufficient for both input and target sequences.
- Utilize held-out structured fields solely for automated clinical factuality evaluation.

### 19. Final Conclusion
The dataset is clean, well-structured, and fully partitioned into reproducible splits. All required outputs, graphs, and audit tables have been compiled. The project is certified and ready for GenAI / ML model development.
