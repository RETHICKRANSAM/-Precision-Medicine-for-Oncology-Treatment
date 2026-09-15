# GenAI Engineer Handoff Report

---

## PROJECT
**Clinical Text Summarization (Precision Oncology)**

---

## DATASET
- **Total Master Rows:** 4,749
- **Columns:** 9 (Raw master schema: 9 columns)
- **Train Set:** `3,800 rows` (`data/processed/genai_train.csv`)
- **Validation Set:** `471 rows` (`data/processed/genai_validation.csv`)
- **Test Set:** `478 rows` (`data/processed/genai_test.csv`)

---

## INPUT
- `clinical_report` (shorthand, noisy clinical consultation notes)

---

## TARGET
- `summary` (standardized, grammatically complete clinical summary)

---

## EXCLUDED FEATURES
- `ner`
- `gene_mutation`
- `drug_name`
- `dosage_level`
- `adverse_event`
- `symptom_text`
- `urgency`

### Why They Are Excluded:
1. **Target Leakage:** 100% of known drug names and gene mutations appear verbatim inside the target `summary`.
2. **Task Integrity:** Feeding structured entities allows the model to perform trivial dictionary lookup or slot-filling instead of learning clinical language comprehension and abstraction.

---

## DATA QUALITY
- **Missing Values:** 0 actual nulls/empty strings; explicit `Unknown` and `None Reported` represent valid medical status.
- **Duplicates:** 0 full-row duplicates; 434 rows share clinical reports, isolated 100% by grouped splitting.
- **Low-Information Records:** Flagged in `outputs/low_information_records.csv`.
- **Outliers:** Documented in `outputs/outlier_report.csv`; all appear medically plausible.

---

## TEXT CHARACTERISTICS
- **Input Note Length:** Mean `17.5` words (Max `23`, 99th percentile `21`)
- **Target Summary Length:** Mean `25.1` words (Max `32`, 99th percentile `32`)
- **Input/Target Relationship:** Target summaries are expansions of shorthand notes (mean ratio `1.54x`).

---

## LEAKAGE
- No verbatim copy between raw note and target summary.
- Complete train/val/test group isolation verified (0 overlapping `clinical_report` strings).

---

## SPLIT
- Grouped split key: `clinical_report`
- Random Seed: `42`
- Proportion: 80% Train / 10% Validation / 10% Test

---

## MODELING RECOMMENDATIONS
1. **Model Architecture:** Encoder-Decoder (e.g. T5, BART, BioLinkBERT-Seq2Seq) or instruction-tuned decoder LLM (e.g. LLaMA, Mistral, Med-PaLM/Gemma).
2. **Tokenizer & Sequence Length:**
   - Max input tokens: 128 - 256 tokens covers > 99.5% of input notes.
   - Max target tokens: 128 - 256 tokens covers > 99.5% of summaries.
3. **Evaluation Metrics:**
   - Primary: ROUGE-1, ROUGE-2, ROUGE-L, BERTScore (BioBERT-based).
   - Clinical Factuality: Entity presence check using held-out `gene_mutation`, `drug_name`, and `adverse_event` to verify factual consistency without prompt contamination.
4. **Reproducibility:** Load strictly from `GENAI_EDA/data/processed/` splits.
