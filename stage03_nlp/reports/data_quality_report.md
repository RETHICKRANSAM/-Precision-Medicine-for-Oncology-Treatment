# Clinical NLP Data Quality Audit Report

## 1. Executive Summary
- **Dataset Shape**: 5000 rows, 24 columns
- **Unique Patients**: 5000
- **Duplicate Full Rows**: 0
- **Unique Cleaned Clinical Notes**: 4400
- **Rows Sharing Duplicate Text**: 826

## 2. Missing Value Analysis
| Column | Missing Count | Missing Percentage |
| :--- | :--- | :--- |
| `patient_id` | 0 | 0.00% |
| `note_type` | 0 | 0.00% |
| `clinical_note` | 0 | 0.00% |
| `urgency` | 0 | 0.00% |
| `gene_mutation` | 0 | 0.00% |
| `drug_name` | 0 | 0.00% |
| `dosage_level` | 0 | 0.00% |
| `adverse_event` | 0 | 0.00% |
| `symptom_text` | 0 | 0.00% |
| `annotation_status` | 0 | 0.00% |
| `has_missing_fields` | 0 | 0.00% |
| `clinical_note_placeholder` | 0 | 0.00% |
| `cleaned_clinical_note` | 0 | 0.00% |
| `text_word_count` | 0 | 0.00% |
| `text_char_count` | 0 | 0.00% |
| `symptom_severity` | 0 | 0.00% |
| `ae_grade` | 0 | 0.00% |
| `mutation_risk` | 0 | 0.00% |
| `urgency_score` | 0 | 0.00% |
| `original_urgency` | 0 | 0.00% |
| `symptom_tag` | 0 | 0.00% |
| `ae_tag` | 0 | 0.00% |
| `mutation_tag` | 0 | 0.00% |
| `enriched_clinical_note` | 0 | 0.00% |

## 3. Target Distribution (`urgency`)
| Urgency Class | Record Count | Percentage |
| :--- | :--- | :--- |
| **Moderate** | 1823 | 36.46% |
| **Low** | 1667 | 33.34% |
| **High** | 1510 | 30.20% |

## 4. Text Length Statistics (`cleaned_clinical_note`)
- **Minimum Word Count**: 12
- **Maximum Word Count**: 23
- **Mean Word Count**: 16.64
- **Median Word Count**: 17.0
- **95th Percentile Word Count**: 20.0
- **Mean Character Count**: 105.46

## 5. Clinical Attributes Distribution
### Note Type
| Note Type | Count |
| :--- | :--- |
| Progress Note | 740 |
| Trial Note | 731 |
| Surgical Pathology | 721 |
| Clinical Note | 719 |
| Nurse Intake | 705 |
| Oncology Consultation | 702 |
| Pathology Report | 682 |

### Annotation Status
| Annotation Status | Count |
| :--- | :--- |
| pending | 1686 |
| unreviewed | 1685 |
| reviewed | 843 |
| annotated | 786 |

### Missing Fields Indicator (`has_missing_fields`)
| Has Missing Fields | Count |
| :--- | :--- |
| True | 2832 |
| False | 2168 |

## 6. Treatment of Unknown Urgency Records
- Total Unknown records: **0**
- **Policy**: Excluded from supervised training and validation/test evaluation.
- Preserved in `data/unlabeled_unknown_data.csv` for semi-supervised / active learning workflows.
