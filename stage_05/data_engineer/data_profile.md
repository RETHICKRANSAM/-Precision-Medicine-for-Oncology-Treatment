# Data Profile Report

**Source file:** `genai_dataset_from_slm_corrected.xlsx` (sheet: `corrected_dataset`)
**Note:** The original file was not modified. This profile was generated from a copy loaded into memory.

## Shape

- Rows: **4,749**
- Columns: **9**

## Columns and data types

| Column | Dtype | Notes |
|---|---|---|
| clinical_report | text | Free-text shorthand clinical note (input) |
| summary | text | Full-sentence generated summary (target) |
| ner | text | Semi-structured string: `GENE_MUTATION: ...; DRUG_NAME: ...; DOSAGE_LEVEL: ...; ADVERSE_EVENT: ...` |
| gene_mutation | categorical (text) | 10 unique values |
| drug_name | categorical (text) | 12 unique values |
| dosage_level | categorical/text | 9 unique values (free-text dosage strings) |
| adverse_event | categorical (text) | 11 unique values |
| symptom_text | categorical (text) | reported symptom label |
| urgency | categorical (text) | Low / Moderate / High / Unknown |

`Patient_ID` is **not present** in this file (it was already excluded upstream). No other direct patient identifier column exists.

## Missing / null values

Zero nulls in every column (`.isnull().sum() == 0` across the board).

## Empty strings

Zero empty strings (after stripping whitespace) in any column.

## Duplicates

- Exact full-row duplicates: **0**
- Duplicate `clinical_report` values: **279** duplicate rows, spanning **155 distinct report strings** that each appear more than once (434 rows total involved). See `cleaning_report.md` and `leakage_audit.md` for what this means and how it's handled.

## Unique value counts (categorical fields)

| Field | Unique values | Top value | Share Unknown/None |
|---|---|---|---|
| gene_mutation | 10 | Unknown (759) | 16.0% |
| drug_name | 12 | Unknown (835) | 17.6% |
| dosage_level | 9 | Unknown (1,221) | 25.7% |
| adverse_event | 11 | None Reported (1,469) | 30.9% |
| urgency | 4 | High (1,372) | — (Unknown = 730, 15.4%) |

## Text length profile

| Field | Min | Mean | Max |
|---|---|---|---|
| clinical_report | 73 chars | 106 chars | 137 chars |
| summary | 42 chars | 163 chars | 210 chars |

7 rows have a minimal, generic summary ("This pathology report documents a patient.") paired with all-Unknown/None structured fields — flagged in `cleaning_report.md` as low-information-content records, not removed.
