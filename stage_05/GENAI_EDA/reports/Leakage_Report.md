# Data Leakage and Target Contamination Report

## 1. Context & Task Definition
The GenAI summarization task is strictly defined as:
- **INPUT:** `clinical_report`
- **TARGET:** `summary`

All other 7 columns (`ner`, `gene_mutation`, `drug_name`, `dosage_level`, `adverse_event`, `symptom_text`, `urgency`) are auxiliary audit fields.

## 2. Leakage Audit Findings
| feature_name | non_null_known_rows | rows_verbatim_in_summary | percentage_of_known | percentage_of_total_rows | risk_level | leakage_mechanism | recommended_action | example |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ner | 4749 | 0 | 0.0 | 0.0 | HIGH RISK | Direct entity / verbatim token match inside generated target summary | MANDATORY EXCLUSION: Serves as a target decomposition. If included as input, model shortcuts genuine summarization. | None |
| gene_mutation | 3990 | 3990 | 100.0 | 84.02 | HIGH RISK | Direct entity / verbatim token match inside generated target summary | MANDATORY EXCLUSION: Serves as a target decomposition. If included as input, model shortcuts genuine summarization. | 'EGFR L858R' in summary |
| drug_name | 3914 | 3914 | 100.0 | 82.42 | HIGH RISK | Direct entity / verbatim token match inside generated target summary | MANDATORY EXCLUSION: Serves as a target decomposition. If included as input, model shortcuts genuine summarization. | 'Erlotinib' in summary |
| dosage_level | 3528 | 2767 | 78.43 | 58.26 | HIGH RISK | Direct entity / verbatim token match inside generated target summary | MANDATORY EXCLUSION: Serves as a target decomposition. If included as input, model shortcuts genuine summarization. | '80 mg/day' in summary |
| adverse_event | 3280 | 2929 | 89.3 | 61.68 | HIGH RISK | Direct entity / verbatim token match inside generated target summary | MANDATORY EXCLUSION: Serves as a target decomposition. If included as input, model shortcuts genuine summarization. | 'Hepatotoxicity' in summary |
| symptom_text | 4749 | 4368 | 91.98 | 91.98 | HIGH RISK | Direct entity / verbatim token match inside generated target summary | MANDATORY EXCLUSION: Serves as a target decomposition. If included as input, model shortcuts genuine summarization. | 'Nausea After Treatment' in summary |
| urgency | 4019 | 0 | 0.0 | 0.0 | LOW RISK | Direct entity / verbatim token match inside generated target summary | EXCLUDED: Kept out of model input to prevent distributional skew. | None |

## 3. Deep-Dive Risk Analysis
1. **Verbatim Entity Duplication (HIGH RISK):**
   - 100% of known drug names and gene mutations in the structured fields appear verbatim within the target `summary`.
   - The structured fields represent an upstream decomposition of the target text itself rather than independent clinical context.
2. **Shortcutting Summarization:**
   - If an LLM is fed `drug_name`, `gene_mutation`, and `adverse_event` as input prompts, it will trivially learn template slot-filling rather than clinical comprehension and summarization.
3. **Copy-Paste Verbatim Audit:**
   - Summary appears inside clinical_report: `0 rows`
   - Clinical report appears inside summary: `0 rows`
   - Exact identical pairs: `0 rows`

## 4. Policy for Model Development
> [!CAUTION]
> Under no circumstances should `ner`, `gene_mutation`, `drug_name`, `dosage_level`, `adverse_event`, `symptom_text`, or `urgency` be concatenated into the model prompt or input embeddings. They are designated as `EXCLUDED FROM GENAI MODEL INPUT`.
