# Cleaning Report

**Principle followed throughout:** clean formatting, never clinical meaning. No missing value was fabricated or guessed. No medical value was silently changed.

## 1. Whitespace / formatting normalization

Checked for leading/trailing spaces, repeated internal spaces, and tabs across every text column.
**Result: none found.** The uploaded file was already clean on this dimension. Defensive `strip()` +
whitespace-collapse was still applied as a no-op safety pass (0 values changed).

## 2. Repeated-token artifact in the `ner` field

Found a text-duplication bug in the `ADVERSE_EVENT` sub-value of the `ner` field: values like
`rash | rash`, `diarrhea | diarrhea`, `fever | fever` (same token repeated with a pipe separator).

- **Rows affected:** 158
- **Action:** collapsed `X | X` → `X` (e.g. `rash | rash` → `rash`).
- **Why this is safe:** this is a duplicated *string token*, not a different clinical value — the
  adverse event named (e.g. "rash") is unchanged. This is a technical/text-formatting fix, not a
  medical correction.

## 3. Exact duplicate rows

0 fully-identical rows found. Nothing removed.

## 4. Repeated `clinical_report` values

155 distinct `clinical_report` strings appear more than once (434 rows total, 279 "extra" beyond
first occurrence).

**These rows were NOT removed or merged.** Investigation showed:

- The **target `summary`** is identical across every duplicate group (0/155 groups have a differing
  summary) — so these are not conflicting input→target pairs.
- However, **`urgency` disagrees within 153 of the 155 groups** — the same clinical text is labeled
  with different urgency levels in different rows. This means `urgency` is **not a deterministic
  function of the clinical text** in this dataset and should be treated with caution as a label
  (see Final Report, "remaining data-quality issues").
- A smaller number of groups (51/155) also show minor differences in `ner`/`adverse_event`/
  `dosage_level`/`symptom_text` — all traced to the same kind of formatting variation documented in
  the NER audit (not new information).

**Handling:** rows were kept as-is (each may represent a distinct patient encounter that happens to
produce identical shorthand text), but the train/validation/test split groups all rows sharing a
`clinical_report` value together, so no duplicate report text crosses a split boundary (see
`leakage_audit.md`).

## 5. "n/a" tokens in `clinical_report`

344 rows contain `n/a` inside the clinical shorthand text (e.g. *"...started Erlotinib n/a
reports..."*). This is **legitimate clinical shorthand for "dose not recorded"**, not placeholder
junk — confirmed by the fact that these rows consistently have `dosage_level == Unknown`. Left
untouched.

## 6. Low-information summaries

7 rows have the generic summary "This pathology report documents a patient." paired with
`gene_mutation = Unknown`, `drug_name = Unknown`, `adverse_event = None Reported`,
`symptom_text = Not Specified`. These are not broken, but they carry almost no learning signal for
a summarization model. **Flagged, not removed** — a judgment call left to the modeling team on
whether to filter these for training.

## 7. Net effect on row count

| Stage | Rows |
|---|---|
| Original (uploaded file) | 4,749 |
| After cleaning | 4,749 |

**0 rows removed.** All cleaning was cosmetic (a text-duplication fix inside `ner`); no records were
dropped because no exact duplicates or corrupted rows were found.
