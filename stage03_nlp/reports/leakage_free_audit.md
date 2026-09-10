# Stage 03 NLP Pipeline: Leakage-Free Audit & Validation Report

## 1. Executive Summary
This audit report provides formal verification that the Stage 03 NLP Pipeline adheres
to strict zero-leakage protocols. All target-derived feature engineering tags (`[SYMPTOM:...]`,
`[AE:...]`, `[MUTATION:...]`) have been completely removed from model inputs. All models
are trained strictly on genuine clinical note text using ground-truth urgency labels.

## 2. 10-Point Leakage-Free Verification Checklist
| Check # | Requirement | Observed Status | Audit Result |
| :---: | :--- | :--- | :---: |
| 1 | No patient overlap across splits | Train ∩ Val: 0, Train ∩ Test: 0, Val ∩ Test: 0 | **PASSED** |
| 2 | No duplicate text overlap across splits | Train ∩ Val: 0, Train ∩ Test: 0, Val ∩ Test: 0 | **PASSED** |
| 3 | No urgency-derived tags in model input | Input column is strictly `clinical_note` without prefix tags | **PASSED** |
| 4 | No composite-score-derived labels | Trained strictly on original labels (High, Moderate, Low) | **PASSED** |
| 5 | Isolation of Unknown records | 762 records saved to `originally_unknown_data.csv` and excluded | **PASSED** |
| 6 | TF-IDF fitted strictly on training data | Vectorizer `fit_transform` called only on Train split | **PASSED** |
| 7 | BiLSTM vocabulary built strictly from train | Vocabulary counter initialized only from Train split | **PASSED** |
| 8 | Class weights calculated strictly from train | Weights computed solely from Train label distribution | **PASSED** |
| 9 | Validation split used solely for model selection | Early stopping & hyperparameter tuning on Val Macro F1 | **PASSED** |
| 10 | Holdout test set used solely for final evaluation | Test data never seen during preprocessing fit or training | **PASSED** |

## 3. Dataset Partition Breakdown
- **Training Partition (70%)**: 3342 records (3342 unique patients)
- **Validation Partition (15%)**: 829 records (829 unique patients)
- **Holdout Test Partition (15%)**: 829 records (829 unique patients)

## 4. Class Distribution Across Partitions
| Partition | High | Moderate | Low | Total |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 1010 | 1219 | 1113 | 3342 |
| **Validation** | 252 | 300 | 277 | 829 |
| **Test** | 248 | 304 | 277 | 829 |
