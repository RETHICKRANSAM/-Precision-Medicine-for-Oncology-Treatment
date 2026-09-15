# Feature Relationship and Correlation Report

## 1. Summary of Clinical Relationships
| feature_name | category_value | record_count | input_word_mean | input_word_median | input_word_std | target_word_mean | target_word_median | target_word_std | expansion_ratio_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| drug_name | Alectinib | 281 | 17.78 | 18.0 | 1.41 | 26.41 | 27.0 | 3.14 | 1.62 |
| drug_name | Carboplatin | 325 | 17.91 | 18.0 | 1.35 | 26.32 | 27.0 | 3.21 | 1.59 |
| drug_name | Cisplatin | 365 | 17.69 | 18.0 | 1.38 | 25.9 | 26.0 | 3.3 | 1.6 |
| drug_name | Crizotinib | 313 | 17.81 | 18.0 | 1.44 | 26.26 | 27.0 | 3.26 | 1.6 |
| drug_name | Dabrafenib | 328 | 17.83 | 18.0 | 1.42 | 26.42 | 27.0 | 3.21 | 1.61 |
| drug_name | Erlotinib | 329 | 17.79 | 18.0 | 1.48 | 26.22 | 27.0 | 3.26 | 1.6 |
| drug_name | Gefitinib | 373 | 17.7 | 18.0 | 1.45 | 25.95 | 26.0 | 3.22 | 1.59 |
| drug_name | Osimertinib | 647 | 17.83 | 18.0 | 1.35 | 26.19 | 27.0 | 3.37 | 1.58 |
| drug_name | Pembrolizumab | 310 | 17.69 | 18.0 | 1.27 | 26.37 | 27.0 | 3.3 | 1.6 |
| drug_name | Sotorasib | 330 | 17.76 | 18.0 | 1.38 | 26.13 | 26.0 | 3.08 | 1.61 |
| drug_name | Trametinib | 313 | 17.75 | 18.0 | 1.29 | 25.86 | 26.0 | 3.37 | 1.59 |
| drug_name | Unknown | 835 | 16.31 | 15.0 | 2.35 | 20.2 | 17.0 | 6.25 | 1.29 |
| gene_mutation | ALK fusion | 388 | 17.41 | 18.0 | 1.68 | 24.93 | 26.0 | 4.27 | 1.53 |
| gene_mutation | BRAF V600E | 362 | 17.5 | 18.0 | 1.65 | 26.04 | 27.0 | 4.26 | 1.61 |
| gene_mutation | EGFR L858R | 713 | 17.48 | 18.0 | 1.5 | 26.56 | 28.0 | 3.84 | 1.66 |
| gene_mutation | EGFR exon 19 del | 350 | 19.31 | 19.0 | 1.59 | 27.89 | 29.0 | 4.32 | 1.6 |
| gene_mutation | KRAS G12C | 736 | 17.5 | 18.0 | 1.58 | 26.06 | 27.0 | 4.02 | 1.62 |
| gene_mutation | MET amplification | 396 | 17.44 | 18.0 | 1.56 | 25.06 | 26.0 | 4.15 | 1.5 |
| gene_mutation | ROS1 fusion | 349 | 17.4 | 18.0 | 1.65 | 24.98 | 26.0 | 4.31 | 1.53 |
| gene_mutation | STK11 mutation | 332 | 17.42 | 18.0 | 1.68 | 24.92 | 26.0 | 4.46 | 1.5 |

## 2. Analytical Insights
- Certain adverse events (e.g. Thrombocytopenia, Hepatotoxicity) correlate with slightly longer clinical narratives and summaries due to increased diagnostic descriptions.
- Urgency level exhibits minimal variation in summary sentence length, indicating standardized summary reporting templates across urgency strata.
- **Notice:** All relationship metrics are derived for auditing and data profiling only; categorical fields remain excluded from GenAI model inputs.
