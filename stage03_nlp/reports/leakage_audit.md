# Stage 03 NLP Pipeline: Data Leakage & Patient Split Audit

## 1. Overview
This audit verifies that patient identifiers and identical clinical texts do not leak between
the Training (70%), Validation (15%), and Testing (15%) partitions.

## 2. Partition Summary
| Partition | Record Count | Unique Patients | Unique Clinical Texts |
| :--- | :--- | :--- | :--- |
| **Train** | 3342 | 3342 | 3032 |
| **Validation** | 829 | 829 | 758 |
| **Test** | 829 | 829 | 759 |

## 3. Class Distribution by Partition
| Partition | High | Moderate | Low |
| :--- | :--- | :--- | :--- |
| **Train** | 1010 | 1219 | 1113 |
| **Validation** | 252 | 300 | 277 |
| **Test** | 248 | 304 | 277 |

## 4. Leakage Verification Results
| Audit Check | Permissible Overlap | Observed Overlap | Status |
| :--- | :--- | :--- | :--- |
| **Patient Overlap: Train ∩ Validation** | 0 | 0 | **PASSED** |
| **Patient Overlap: Train ∩ Test** | 0 | 0 | **PASSED** |
| **Patient Overlap: Validation ∩ Test** | 0 | 0 | **PASSED** |
| **Text Leakage: Train ∩ Validation** | 0 | 0 | **PASSED** |
| **Text Leakage: Train ∩ Test** | 0 | 0 | **PASSED** |
| **Text Leakage: Validation ∩ Test** | 0 | 0 | **PASSED** |

## 5. Grouping Strategy Implementation
Because 826 records share duplicated `cleaned_clinical_note` content, random row-level partitioning
would cause identical clinical text to appear in both training and test sets. To guarantee zero text leakage,
all identical clinical texts were grouped deterministically and assigned en bloc to a single split.
Since each patient corresponds to exactly 1 record, patient isolation is also strictly preserved at 0 overlap.
