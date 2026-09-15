# Data Profile Report — Precision Oncology Clinical Dataset

## 1. Dataset Source & Dimensions
- **Source File:** `genai_cleaned_master.csv`
- **Dimensions:** `4,749 rows x 9 columns`
- **Memory Footprint:** `4017.65 KB`

## 2. Column Schema and Roles
| column_name | data_type | row_count | unique_count | unique_percentage | null_count | null_percentage | empty_string_count | whitespace_only_count | sample_values | role | GenAI_input_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clinical_report | object | 4749 | 4470 | 94.13 | 0 | 0.0 | 0 | 0 | Nurse intake pt feeling nausea after treatment current me... | INPUT | MODEL_INPUT |
| summary | object | 4749 | 4313 | 90.82 | 0 | 0.0 | 0 | 0 | This nurse intake documents a patient receiving Erlotinib... | TARGET | MODEL_TARGET |
| ner | object | 4749 | 3843 | 80.92 | 0 | 0.0 | 0 | 0 | GENE_MUTATION: egfr L858R; DRUG_NAME: Erlotinib; DOSAGE_L... | AUDIT_ONLY | EXCLUDED FROM GENAI MODEL INPUT |
| gene_mutation | object | 4749 | 10 | 0.21 | 0 | 0.0 | 0 | 0 | EGFR L858R \| STK11 mutation \| KRAS G12C | AUDIT_ONLY | EXCLUDED FROM GENAI MODEL INPUT |
| drug_name | object | 4749 | 12 | 0.25 | 0 | 0.0 | 0 | 0 | Erlotinib \| Cisplatin \| Unknown | AUDIT_ONLY | EXCLUDED FROM GENAI MODEL INPUT |
| dosage_level | object | 4749 | 9 | 0.19 | 0 | 0.0 | 0 | 0 | 80 mg/day \| Unknown \| 150 mg/day | AUDIT_ONLY | EXCLUDED FROM GENAI MODEL INPUT |
| adverse_event | object | 4749 | 11 | 0.23 | 0 | 0.0 | 0 | 0 | None Reported \| Hepatotoxicity \| Mucositis | AUDIT_ONLY | EXCLUDED FROM GENAI MODEL INPUT |
| symptom_text | object | 4749 | 13 | 0.27 | 0 | 0.0 | 0 | 0 | Nausea After Treatment \| Mild Fatigue \| Fever And Chills | AUDIT_ONLY | EXCLUDED FROM GENAI MODEL INPUT |
| urgency | object | 4749 | 4 | 0.08 | 0 | 0.0 | 0 | 0 | Moderate \| Low \| High | AUDIT_ONLY | EXCLUDED FROM GENAI MODEL INPUT |

## 3. High-Level Characteristics
- **Input Text:** Unstructured, shorthand clinical consultation and intake notes.
- **Target Text:** Formalized, syntactically complete clinical summary narratives.
- **Auxiliary Columns:** 7 semi-structured and categorical annotations used for clinical quality audits.
