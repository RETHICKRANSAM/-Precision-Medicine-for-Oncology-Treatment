# Data Quality and Integrity Report

## 1. Missing Value and Shorthand Audit
| column_name | actual_null_count | actual_null_pct | empty_string_count | whitespace_only_count | explicit_unknown_count | explicit_unknown_pct | explicit_none_reported_count | explicit_none_reported_pct | total_sparse_or_unknown_count | total_sparse_or_unknown_pct | nature_of_shorthand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clinical_report | 0 | 0.0 | 0 | 0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | Fully populated |
| summary | 0 | 0.0 | 0 | 0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | Fully populated |
| ner | 0 | 0.0 | 0 | 0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | Fully populated |
| gene_mutation | 0 | 0.0 | 0 | 0 | 759 | 15.98 | 0 | 0.0 | 759 | 15.98 | Legitimate clinical shorthand / intentional absence |
| drug_name | 0 | 0.0 | 0 | 0 | 835 | 17.58 | 0 | 0.0 | 835 | 17.58 | Legitimate clinical shorthand / intentional absence |
| dosage_level | 0 | 0.0 | 0 | 0 | 1221 | 25.71 | 0 | 0.0 | 1221 | 25.71 | Legitimate clinical shorthand / intentional absence |
| adverse_event | 0 | 0.0 | 0 | 0 | 0 | 0.0 | 1469 | 30.93 | 1469 | 30.93 | Legitimate clinical shorthand / intentional absence |
| symptom_text | 0 | 0.0 | 0 | 0 | 0 | 0.0 | 0 | 0.0 | 0 | 0.0 | Fully populated |
| urgency | 0 | 0.0 | 0 | 0 | 730 | 15.37 | 0 | 0.0 | 730 | 15.37 | Legitimate clinical shorthand / intentional absence |

### Key Takeaway:
- Actual `NaN`, `None`, empty string, and whitespace-only counts are **0 across all columns**.
- Values labeled `Unknown` or `None Reported` represent intentional clinical shorthand (e.g. adverse event not present), not ingestion corruption.

## 2. Duplicate Analysis
| check_category | duplicate_row_count | unique_duplicate_groups | percentage | examples |
| --- | --- | --- | --- | --- |
| A. Exact Full-Row Duplicates | 0 | 0 | 0.0 | None (0 rows) |
| B. Duplicate clinical_report | 434 | 155 | 9.14 | Pathology nsclc specimen molecular MET a... \| Pathology nsclc specimen molecular ALK f... |
| C. Duplicate summary | 666 | 230 | 14.02 | This pathology report documents a patien... \| This pathology report documents a patien... |
| D. Duplicate clinical_report + summary | 434 | 155 | 9.14 | Pathology nsclc specimen molecular MET a... \| Pathology nsclc specimen molecular ALK f... |
| E. Duplicate clinical_report with different summary | 0 | 0 | 0.0 | None (100% deterministic) |
| F. Duplicate clinical_report with different structured fields | 434 | 155 | 9.14 | 155 groups |

### Duplicate Policy:
- Exact full-row duplicates: **0**
- Duplicate `clinical_report` texts: **434 rows across 155 unique groups**.
- **Recommendation:** Do not delete repeated clinical reports; they represent distinct patient presentations or follow-ups. Handled strictly via grouped train/test splitting.

## 3. Suspicious Formatting Records
- Identified **134** records with unusual text formatting or potential low-information content.
- See `outputs/suspicious_records.csv` and `outputs/low_information_records.csv`.
