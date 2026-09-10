# Stage 03 NLP Pipeline: Clinical Urgency Classification & Clinical Audit

## Overview
This repository implements an end-to-end, production-grade NLP engineering pipeline for oncology clinical notes. The primary objective is **Clinical Note → Urgency Classification**, triaging patient documentation into **Low**, **Moderate**, or **High** priority, alongside deep data quality auditing, duplicate text leakage prevention, annotation bias auditing, and preparation for Medical Named Entity Recognition (NER).

---

## 1. Actual Dataset & Schema
- **Source**: `STAGE_3/nlp_cleaned_data.csv`
- **Total Records**: 5,000 rows, 15 columns
- **Unique Patients**: 5,000 patients (1 record per patient)
- **Primary Text Feature**: `cleaned_clinical_note` (normalized, lowercased clinical text)
- **Target Variable**: `urgency`
  - `High`: 1,451 records (29.0%)
  - `Low`: 1,395 records (27.9%)
  - `Moderate`: 1,392 records (27.8%)
  - `Unknown`: 762 records (15.2%)

### Handling of `Unknown` Records
Per strict clinical NLP standards, `Unknown` represents unannotated or clinically uncertain data rather than a distinct medical urgency tier. 
- All 762 `Unknown` records are **excluded** from supervised model training and validation/test evaluation.
- They are preserved in `data/unlabeled_unknown_data.csv` for downstream semi-supervised learning or active clinical review.
- All supervised modeling is executed strictly on the remaining 4,238 labeled cases.

---

## 2. Duplicate Text Audit & Zero-Leakage Splitting
- **Duplicate Rows**: 826 records share duplicate `cleaned_clinical_note` text across 226 distinct text groups.
- **Label Consistency**: 192 groups contain conflicting urgency labels across encounters; 34 groups exhibit identical labels.
- **Patient-Level Split with Strict Zero Leakage (70% Train / 15% Val / 15% Test, Seed 42)**:
  - Patients and duplicate text groups are partitioned en bloc.
  - **Guaranteed Isolation**:
    $$\text{Train Patients} \cap \text{Val Patients} = 0$$
    $$\text{Train Patients} \cap \text{Test Patients} = 0$$
    $$\text{Val Patients} \cap \text{Test Patients} = 0$$
    $$\text{Train Texts} \cap \text{Val Texts} = 0$$
    $$\text{Train Texts} \cap \text{Test Texts} = 0$$
    $$\text{Val Texts} \cap \text{Test Texts} = 0$$
  - Fully documented in [`reports/leakage_audit.md`](file:///reports/leakage_audit.md).

---

## 3. NLP Architecture & Models
1. **Model 1 — Linear SVM + TF-IDF**:
   - `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2)` fit strictly on training texts.
   - `LinearSVC(class_weight='balanced')` with validation tuning for regularization parameter $C$.
   - Saved to `models/svm_tfidf_model.joblib` and `models/tfidf_vectorizer.joblib`.

2. **Model 2 — PyTorch BiLSTM**:
   - Word vocabulary constructed strictly from training texts.
   - Architecture: `Embedding(128) -> BiLSTM(128, 2 layers, dropout=0.3) -> Dropout -> Linear(256, 3)`.
   - Loss: Class-weighted `CrossEntropyLoss`.
   - Optimizer: `Adam(lr=1e-3, weight_decay=1e-4)`.
   - Early stopping monitored strictly on **Validation Macro F1**.
   - Checkpoint saved to `models/bilstm_model.pt`.

3. **Model 3 — Bio_ClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`)**:
   - Domain-specific clinical transformer pretrained on MIMIC-III clinical text.
   - Sequence classification head with 3 output logits (`Low: 0`, `Moderate: 1`, `High: 2`).
   - Optimizer: `AdamW(lr=2e-5, weight_decay=0.01)`.
   - Early stopping monitored on **Validation Macro F1**.
   - Saved to `models/clinicalbert_model/`.

---

## 4. Directory Structure
```
stage03_nlp/
├── data/
│   ├── nlp_cleaned_data.csv          # Source cleaned dataset
│   ├── unlabeled_unknown_data.csv    # 762 excluded Unknown records
│   ├── train.csv                     # Training split (70%)
│   ├── val.csv                       # Validation split (15%)
│   ├── test.csv                      # Test split (15%)
│   └── ner_preparation_dataset.csv   # NER reference dataset
│
├── src/
│   ├── data_loader.py                # Schema validation & quality audit
│   ├── duplicate_audit.py            # Duplicate detection & conflict analysis
│   ├── patient_split.py              # Zero-leakage patient/text grouping
│   ├── preprocessing.py              # Clinical tokenization & vocabulary builder
│   ├── train_svm_tfidf.py            # Linear SVM + TF-IDF pipeline
│   ├── train_bilstm.py               # PyTorch BiLSTM classifier
│   ├── train_clinicalbert.py         # Bio_ClinicalBERT fine-tuning
│   ├── evaluate.py                   # Multi-metric evaluation & comparison
│   ├── error_analysis.py             # Directional error mining & linguistic profiling
│   ├── bias_audit.py                 # Annotation status, missing fields, note type bias
│   └── inference.py                  # Predictor interface with safety disclosures
│
├── models/
│   ├── svm_tfidf_model.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── bilstm_model.pt
│   ├── bilstm_vocab.json
│   └── clinicalbert_model/
│
├── reports/
│   ├── data_quality_report.md
│   ├── leakage_audit.md
│   ├── text_duplicate_audit.csv
│   ├── model_comparison.csv
│   ├── model_comparison.md
│   ├── error_analysis.csv
│   ├── annotation_quality_audit.csv
│   ├── missing_field_urgency_audit.csv
│   ├── note_type_urgency_distribution.csv
│   └── figures/
│       ├── svm_confusion_matrix.png
│       ├── bilstm_confusion_matrix.png
│       ├── clinicalbert_confusion_matrix.png
│       └── bilstm_training_curves.png
│
├── run_nlp_pipeline.py               # Master pipeline running all 20 stages
└── README.md                         # Pipeline documentation & governance
```

---

## 5. Execution Instructions
To run the entire pipeline end-to-end:
```powershell
python -u stage03_nlp/run_nlp_pipeline.py
```

To run inference on new clinical notes:
```python
from stage03_nlp.src.inference import ClinicalUrgencyPredictor

predictor = ClinicalUrgencyPredictor("svm")  # or "bilstm" / "clinicalbert"
result = predictor.predict("pt w/ nsclc mutation kras g12c started unknown med 5 mg bid reports nausea after treatment ae thrombocytopenia")
print(result)
```

---

## 6. Clinical & Research Safety Disclaimers
> [!CAUTION]
> **Research Prototype Only**:
> 1. This pipeline is an experimental research prototype intended for decision-support methodology exploration.
> 2. It does **not** provide autonomous diagnostic or triage capabilities.
> 3. Urgency scores must not supersede licensed clinical judgement.
> 4. Performance on prototype/synthetic distributions does not infer real-world clinical validity.
