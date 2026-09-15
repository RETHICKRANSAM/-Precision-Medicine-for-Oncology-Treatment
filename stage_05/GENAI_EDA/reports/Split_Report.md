# Grouped Train / Validation / Test Split Report

## 1. Executive Summary
- **Grouping Key:** `clinical_report` (exact text match)
- **Random Seed:** `42`
- **Target Distribution:** 80% Train, 10% Validation, 10% Test
- **Total Dataset Rows:** 4,749
- **Unique `clinical_report` Groups:** 4,470

## 2. Split Partitioning Results
| Split Partition | Row Count | Percentage of Rows | Unique Group Count | Percentage of Groups |
|---|---|---|---|---|
| **Train** | 3,800 | 80.02% | 3,576 | 80.0% |
| **Validation** | 471 | 9.92% | 447 | 10.0% |
| **Test** | 478 | 10.07% | 447 | 10.0% |
| **Total** | **4,749** | **100.0%** | **4,470** | **100.0%** |

## 3. Cross-Split Isolation Verification
- **Train & Validation Overlap:** 0 records
- **Train & Test Overlap:** 0 records
- **Validation & Test Overlap:** 0 records
- **Group Isolation Status:** `PASSED - 0 LEAKAGE ACROSS SPLITS`

## 4. Known Patient_ID Limitation
> [!IMPORTANT]
> `Patient_ID` is **not present** in the dataset and must not be fabricated. Grouping by `clinical_report` ensures that identical clinical shorthand notes are never split across train and evaluation sets. However, if a single patient generated two distinct notes across different visits, true patient-level cross-visit isolation cannot be mathematically proven without an upstream patient identifier. This is documented as an acknowledged residual limitation for modeling teams.
