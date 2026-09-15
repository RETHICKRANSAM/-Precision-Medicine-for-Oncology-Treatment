# Data Leakage Audit

**Conclusion up front: leakage risk exists, and it is structural, not accidental — see Finding 3.
It is fully avoidable by the INPUT/TARGET design specified in the GenAI dataset definition.**
No claim of "no leakage" is made without the evidence below.

## 1. Duplicate clinical reports across splits

- 155 distinct `clinical_report` values repeat (434 rows).
- The train/validation/test split (see below) was built by **grouping on `clinical_report`** before
  splitting, so every row sharing the same report text is guaranteed to land in the same split.
- **Verified:** 0 `clinical_report` values appear in more than one split (checked programmatically:
  `groupby('clinical_report')['split'].nunique()` has a max of 1 across all 4,749 rows).

## 2. Duplicate input-target pairs

- Because `summary` is identical within every duplicated-`clinical_report` group, the 279 "extra"
  duplicate rows are also duplicate `(clinical_report, summary)` pairs.
- These are confined to a single split by the same grouping (see #1), so they cannot appear as
  "seen in training, re-appears identically in test."
- They do mean the **effective number of unique training examples is smaller than the row count**
  (3,576 unique clinical_report groups vs 4,749 rows) — noted as a quality issue, not a leakage
  issue, since it doesn't cross split boundaries.

## 3. Patient overlap

- `Patient_ID` is **not present** in this file (already removed upstream). Patient-level grouping
  therefore could not be verified directly.
- **`clinical_report` grouping was used as the best available proxy** for patient-level isolation.
  This is a limitation: if the same patient could genuinely generate two *different* clinical_report
  strings (e.g., two different visits), true patient-level leakage across splits **cannot be ruled
  out from this file alone**. Flagged as a residual risk — see Final Report.

## 4. Target information accidentally included in the input — ⚠️ MOST IMPORTANT FINDING

This is the central leakage risk in this dataset, and it is why the GenAI dataset definition
(`data_dictionary.md` / task definition in the Final Report) restricts model **INPUT** to
`clinical_report` only.

Evidence:
- **100% of known drug names (3,914/3,914 rows with a non-Unknown `drug_name`) appear verbatim
  inside the `summary` text.**
- **100% of known gene mutation tokens (3,990/3,990 rows with a non-Unknown `gene_mutation`) appear
  verbatim inside the `summary` text.**
- In other words, `gene_mutation`, `drug_name`, `dosage_level`, `adverse_event`, `symptom_text`, and
  `ner` are not independent context — they are effectively **a decomposition of the `summary`
  target itself**, produced by the same upstream generation step.

**Implication:** if any of these structured/NER fields were fed to a summarization model as
additional INPUT context, the model could trivially reconstruct the target by template-filling
rather than by learning genuine clinical language understanding — this would silently inflate
apparent performance metrics. **These fields must not be used as INPUT features for the
summarization task.** They remain useful for: (a) the NER-consistency audit itself, (b) a
*separate* structured-extraction task if one is ever built, and (c) held-out evaluation of whether
a generated summary mentions the right entities.

## 5. Summary copied into clinical_report (or vice versa)

- Checked whether `summary` appears verbatim inside `clinical_report`: **0 rows**.
- Checked whether `clinical_report` appears verbatim inside `summary`: **0 rows**.
- **No copy-paste leakage between input and target text.** The clinical shorthand and the full-
  sentence summary are genuinely different textual representations — real summarization/generation
  is required, not extraction.

## 6. NER/structured fields directly revealing the target

Covered under Finding 4 above — yes, they do, by design of the upstream data generation. Excluding
them from INPUT (Finding 4) is the mitigation.

## 7. Post-outcome information used incorrectly

- `urgency` is not treated as an input feature or target in the current GenAI (summarization) task
  definition, so this risk is currently moot.
- If `urgency` is ever used as a downstream label, note the inconsistency found in
  `cleaning_report.md` (§4): the same `clinical_report` text carries different `urgency` values in
  153/155 duplicate groups — this label is noisy and would need its own remediation before use.

## Summary table

| Check | Result | Leakage confirmed? |
|---|---|---|
| Duplicate clinical reports across splits | 0 after group-based split | No (mitigated) |
| Duplicate input-target pairs across splits | 0 after group-based split | No (mitigated) |
| Patient overlap across splits | Cannot fully verify (no Patient_ID in this file) | **Unresolved / residual risk** |
| Target info leaking into input text | 0 verbatim copies found | No |
| Structured/NER fields as answer key for target | **Confirmed (100% verbatim overlap)** | **Yes — mitigated by INPUT/TARGET design, not by the data itself** |
| Urgency label consistency | 153/155 duplicate groups disagree | N/A to current task, flagged for future use |
