# Data Dictionary — GenAI Dataset (Precision Oncology Summarization)

## Task definition

**Task type:** Clinical text generation / summarization.

**INPUT (model feature):**
- `clinical_report` — the raw, shorthand clinical note. This is the *only* field used as model
  input.

**TARGET (label to generate):**
- `summary` — the full-sentence clinical summary.

**Fields present in the file but explicitly EXCLUDED from model input:**
`ner`, `gene_mutation`, `drug_name`, `dosage_level`, `adverse_event`, `symptom_text`, `urgency`.

**Why excluded:** `leakage_audit.md` Finding 4 shows these fields are effectively a decomposition of
the `summary` target (100% verbatim overlap for known drug names and gene mutations) — including
them as input would let a model shortcut the task instead of learning to summarize. They are
retained in the cleaned master file for:
- auditing / data quality tracking (this pipeline),
- potential future use as **evaluation-only** signals (e.g., checking whether a generated summary
  mentions the correct entities) — never as training input,
- a possible *separate* structured-extraction (NER) modeling task, which is out of scope here.

`Patient_ID` is not present in this file and must never be reintroduced as a feature if a future
merge with the master dataset occurs.

## Column reference

| Column | Type | Used in GenAI task as | Description |
|---|---|---|---|
| `clinical_report` | text | **INPUT** | Shorthand/noisy clinical note (nurse intake, oncology f/u, pathology, trial screening, progress note, etc.) |
| `summary` | text | **TARGET** | Full-sentence rewritten clinical summary |
| `ner` | text | excluded from input (audit-only) | Semi-structured extraction string: `GENE_MUTATION: ...; DRUG_NAME: ...; DOSAGE_LEVEL: ...; ADVERSE_EVENT: ...` |
| `gene_mutation` | categorical | excluded from input | Flat gene mutation label, one of 10 values incl. `Unknown` |
| `drug_name` | categorical | excluded from input | Flat drug name, one of 12 values incl. `Unknown` |
| `dosage_level` | text/categorical | excluded from input | Free-text dosage string or `Unknown` |
| `adverse_event` | categorical | excluded from input | Adverse event label or `None Reported` |
| `symptom_text` | categorical | excluded from input | Reported symptom label |
| `urgency` | categorical | excluded (not used by this task; noisy — see cleaning_report.md) | Low / Moderate / High / Unknown |

## Files produced

| File | Contents |
|---|---|
| `data/genai_cleaned_master.csv` | Full cleaned dataset, all 9 columns, 4,749 rows |
| `data/genai_train.csv` | Training split (80% by clinical_report group) |
| `data/genai_validation.csv` | Validation split (~10%) |
| `data/genai_test.csv` | Test split (~10%), completely held out |
| `reports/data_profile.md` | Raw data profiling |
| `reports/cleaning_report.md` | Every cleaning action taken and why |
| `reports/ner_consistency_audit.csv` | Row-level NER-vs-structured-field mismatch audit |
| `reports/leakage_audit.md` | Full leakage investigation and findings |
| `reports/data_dictionary.md` | This file |

## Split methodology

- Grouping key: `clinical_report` (best available proxy for patient-level isolation — no
  `Patient_ID` present in this file).
- Method: shuffle unique `clinical_report` values with a fixed seed (`numpy.random.RandomState(42)`),
  then allocate ~80% of groups to train, ~10% to validation, ~10% to test; every row belonging to a
  group goes to that group's split.
- Verified: 0 groups span more than one split.
