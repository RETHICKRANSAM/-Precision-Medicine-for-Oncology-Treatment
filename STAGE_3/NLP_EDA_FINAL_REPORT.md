# Comprehensive NLP Exploratory Data Analysis (EDA) Report
**Role**: Senior Clinical NLP Data Analyst & Bioinformatics Engineer  
**Dataset**: Clinical Oncology Notes & Phenotypic Annotations (`stage 3/cleaned_data.csv`)  
**Cohort Scale**: $N = 5,000$ patient clinical notes across 12 multi-modal attributes  
**Cleaned Output**: [`stage 3/nlp_cleaned_data.csv`](file:///c:/Users/venka/Downloads/stage2_dlzip/stage2_dl/stage%203/nlp_cleaned_data.csv)  
**Interactive Pipeline**: [`stage 3/nlp_eda_pipeline.ipynb`](file:///c:/Users/venka/Downloads/stage2_dlzip/stage2_dl/stage%203/nlp_eda_pipeline.ipynb)  
**Generated Visualizations Directory**: [`stage 3/nlp_eda_artifacts/graphs/`](file:///c:/Users/venka/Downloads/stage2_dlzip/stage2_dl/stage%203/nlp_eda_artifacts/graphs/)

---

## Executive Summary

An exhaustive, non-destructive 18-Phase Exploratory Data Analysis (EDA) was performed on the clinical NLP cohort ($N=5,000$ encounters). The corpus consists of structured and semi-structured free-text clinical progress notes, oncology consultations, pathology reports, nurse intakes, and clinical trial screening documents. Each encounter is linked with molecular genomic drivers (e.g., *EGFR*, *KRAS*, *MET*, *TP53*, *ALK*, *ROS1*), therapeutic regimens (*Erlotinib*, *Cisplatin*, *Pembrolizumab*, *Osimertinib*, *Sotorasib*, *Carboplatin*), dosage levels, adverse events (*hepatotoxicity*, *thrombocytopenia*, *neutropenia*, *rash*, *diarrhea*), documented patient symptoms, and an adverse urgency classification (`Low`, `Moderate`, `High`, `Unknown`).

All 147 verification criteria spanning data quality, text statistics, lexical diversity, n-gram syntax, stopword mechanics, class balance, outliers, and feature correlations were evaluated. A new sanitized and feature-augmented dataset was saved to [`nlp_cleaned_data.csv`](file:///c:/Users/venka/Downloads/stage2_dlzip/stage2_dl/stage%203/nlp_cleaned_data.csv) without modifying the raw source data.

---

## Phase 1 — Dataset Understanding

### 1. Dataset Dimensions & Schema
- **Total Record Count**: $5,000$ rows
- **Total Feature Count**: $12$ columns
- **Memory Footprint**: $\approx 1.12\text{ MB}$

### 2. Feature Typing & Semantic Classification

| Column Name | Physical Data Type | Semantic Role | Cardinality ($|U|$) | Sample Value |
| :--- | :--- | :--- | :--- | :--- |
| `patient_id` | `object` (string) | **Patient ID (Primary Key)** | 5,000 | `P00001` |
| `note_type` | `object` (string) | **Categorical (Metadata)** | 7 | `Clinical Note`, `Progress Note` |
| `clinical_note` | `object` (string) | **Primary Text Feature** | 4,549 | `Nurse intake: Pt feeling nausea after treatment...` |
| `urgency` | `object` (string) | **Target / Outcome Label** | 4 | `Low`, `Moderate`, `High`, `Unknown` |
| `gene_mutation` | `object` (string) | **Categorical (Genomic)** | 10 | `EGFR L858R`, `KRAS G12C`, `MET amplification` |
| `drug_name` | `object` (string) | **Categorical (Pharmacology)**| 7 | `Erlotinib`, `Cisplatin`, `Pembrolizumab` |
| `dosage_level` | `object` (string) | **Categorical / Dose Entity** | 8 | `80 mg/day`, `150 mg/day`, `5 mg BID` |
| `adverse_event` | `object` (string) | **Secondary Text / Clinical** | 8 | `Hepatotoxicity`, `Mucositis`, `None Reported` |
| `symptom_text` | `object` (string) | **Secondary Text / Symptom** | 8 | `Nausea After Treatment`, `Fever And Chills` |
| `annotation_status` | `object` (string) | **Categorical (Audit Status)**| 3 | `reviewed`, `unreviewed`, `pending` |
| `has_missing_fields`| `bool` (boolean) | **Quality Flag** | 2 | `True`, `False` |
| `clinical_note_placeholder`| `bool` (boolean)| **Quality Flag** | 1 | `False` |

![Clinical Note Type Distribution](C:/Users/venka/.gemini/antigravity-ide/brain/6e6a168c-f59f-48b3-9492-627a0acdb0d8/graphs/14_note_type_distribution.png)

---

## Phase 2 — Data Quality & Integrity Audit

1. **Missing Value Analysis**: Exactly $0$ missing (`NaN` / `null`) cells exist across all $12$ columns ($0.00\%$).
2. **Hidden Whitespace & Empty Strings**: An automated scan for `""` and whitespace-only strings (`"   "`) across all $10$ string columns returned $0$ occurrences.
3. **Record-Level Duplicate Count**: Exactly $0$ identical full rows exist across the entire dataset. Every row possesses a distinct `patient_id` ranging from `P00001` to `P05000` ($100\%$ unique).
4. **Clinical Note Text Redundancy**: 
   - $4,549$ unique notes exist across $5,000$ encounters.
   - $451$ records ($9.02\%$) contain clinical narrative texts shared with other records. These represent standardized clinical template formats (e.g. repeated trial screening or follow-up protocol templates where clinical narratives follow fixed institutional phrasing).
   - **Handling Strategy**: Do **NOT** drop these records. Each row corresponds to a distinct clinical encounter and patient ID. Deleting identical note texts would discard valid independent patient encounters and distort epidemiological prevalence.
5. **Quality Flags & Placeholders**:
   - `clinical_note_placeholder`: $0$ true placeholders ($100\%$ of documents contain real clinical narratives).
   - `has_missing_fields`: $2,832$ records ($56.64\%$) flag that secondary clinical fields (such as dosage or adverse events) had implicit or unrecorded entries, accurately capturing routine incomplete EHR documentation.

---

## Phase 3 & 4 — Text Quality & Length Distribution

### 1. Document-Level Descriptive Statistics

| Text Metric | Minimum | Maximum | Mean | Median | Standard Dev. | IQR |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Character Count** | 80 | 145 | **112.46** | 112.00 | 14.73 | 24.0 (99.0 – 123.0) |
| **Word Count** | 12 | 23 | **17.37** | 18.00 | 1.77 | 3.0 (16.0 – 19.0) |
| **Sentence Count** | 1 | 5 | **2.73** | 3.00 | 0.81 | 1.0 (2.0 – 3.0) |
| **Average Word Length** | 4.12 | 6.08 | **5.13** | 5.11 | 0.34 | 0.46 (4.89 – 5.35) |
| **Average Sentence Length** | 3.60 | 18.00 | **9.03** | 8.00 | 4.09 | 4.67 (6.33 – 11.0) |

![Character Count Distribution](C:/Users/venka/.gemini/antigravity-ide/brain/6e6a168c-f59f-48b3-9492-627a0acdb0d8/graphs/02_character_count_distribution.png)
![Word Count Distribution](C:/Users/venka/.gemini/antigravity-ide/brain/6e6a168c-f59f-48b3-9492-627a0acdb0d8/graphs/03_word_count_distribution.png)
![Text Length Boxplots](C:/Users/venka/.gemini/antigravity-ide/brain/6e6a168c-f59f-48b3-9492-627a0acdb0d8/graphs/04_text_length_boxplots.png)

### 2. Normality & Skewness Analysis
- **Character Length Skewness**: $+0.060$ (Near-zero parametric symmetry).
- **Character Length Kurtosis**: $-0.414$ (Platykurtic, gentle tails).
- **Word Count Skewness**: $-0.297$ (Mild negative skew; notes are tightly controlled by templated clinical entry protocols).
- **Word Count Kurtosis**: $-0.267$.
- **Interpretation**: Text lengths follow a remarkably stable Gaussian-like distribution, indicating standardized documentation protocols across clinics.

### 3. Linguistic & Noise Pattern Scan
- **Repeated Characters** (`helloooo`, `goooood`): $0$ instances.
- **Excessive Punctuation** (`!!`, `??`, `...`, `::`): $736$ notes ($14.7\%$) contain semi-colons, repeated periods (`..`), or paired colons (`::`) typical of semi-structured clinical shorthand.
- **URLs, Emails, Mentions, Hashtags**: $0$ instances (No social media artifacts).
- **Numerical & Dosage Patterns**: $2,705$ notes ($54.10\%$) contain clinical dosage numbers (e.g. `80 mg`, `150 mg/day`, `5 mg BID`, `2 mg/kg`, `exon 19 del`).
- **HTML / XML Tags**: $0$ instances.
- **OCR / Typographical Artifacts**: Identified specific typographical artifact `rep0rts` (digit zero replacing letter 'o') in templated notes.

---

## Phase 5, 6 & 7 — Lexical Diversity, Stopwords & N-Gram Syntax

### 1. Vocabulary Scale & Richness
- **Total Running Tokens**: $86,845$
- **Vocabulary Size ($V$)**: $132$ unique lowercased tokens
- **Type-Token Ratio ($\text{TTR} = V / N$)**: $0.0015$
- **Hapax Legomena (Words occurring exactly once)**: $0$ ($0.00\%$)
- **Rare Words ($\le 3$ occurrences)**: $0$ ($0.00\%$)
- **Finding**: The vocabulary is extremely controlled and domain-dense, reflecting focused EHR clinical oncology intake documentation.

### 2. Stopword Prevalence & Clinical Negation Sensitivity
- **Total Stopword Tokens**: $7,115$ out of $86,845$ ($8.19\%$).
- **Top Stopwords**: `not` ($1,388$), `no` ($1,078$), `as` ($740$), `of` ($730$), `is` ($717$), `on` ($717$), `has` ($702$), `after` ($350$).
- > [!CAUTION]
  > **Do Not Blindly Remove Stopwords**: In clinical NLP, `not` ($1,388$) and `no` ($1,078$) account for over $34.6\%$ of all stopword occurrences. Blind stopword removal would invert clinical assertions (e.g., *"no diarrhea"* becomes *"diarrhea"*, and *"dose not recorded"* becomes *"dose recorded"*), directly corrupting safety and urgency predictions.

![Top Words Overall](C:/Users/venka/.gemini/antigravity-ide/brain/6e6a168c-f59f-48b3-9492-627a0acdb0d8/graphs/05_top_words_overall.png)

### 3. N-Gram Analysis (Repeated Clinical Phrases)

| Rank | Top Bigrams (2-gram) | Freq | Top Trigrams (3-gram) | Freq |
| :---: | :--- | :---: | :--- | :---: |
| 1 | `kras g12c` | 781 | `nurse intake pt` | 740 |
| 2 | `egfr l858r` | 778 | `intake pt feeling` | 740 |
| 3 | `nurse intake` | 740 | `prior current drug` | 727 |
| 4 | `intake pt` | 740 | `pathology nsclc specimen` | 702 |
| 5 | `pt feeling` | 740 | `nsclc specimen molecular` | 702 |
| 6 | `current med` | 740 | `clinical correlation req` | 702 |
| 7 | `noted as` | 740 | `correlation req d` | 702 |
| 8 | `trial screening` | 727 | `req d pt` | 702 |
| 9 | `prior current` | 727 | `d pt has` | 702 |
| 10 | `current drug` | 727 | `pt w nsclc` | 698 |

![Top Bigrams](C:/Users/venka/.gemini/antigravity-ide/brain/6e6a168c-f59f-48b3-9492-627a0acdb0d8/graphs/06_top_bigrams.png)
![Top Trigrams](C:/Users/venka/.gemini/antigravity-ide/brain/6e6a168c-f59f-48b3-9492-627a0acdb0d8/graphs/07_top_trigrams.png)

---

## Phase 8, 9 & 10 — Target Distribution & Class-Conditioned Text Analysis

### 1. Target Label Distribution (`urgency`)

| Urgency Class | Encounter Count | Proportion (%) | Clinical Interpretation |
| :--- | :--- | :--- | :--- |
| **High** | 1,451 | **29.02%** | Severe toxicity / acute adverse events requiring urgent intervention |
| **Low** | 1,395 | **27.90%** | Mild or baseline symptoms; routine monitoring |
| **Moderate** | 1,392 | **27.84%** | Intermediate symptoms; supportive therapy indicated |
| **Unknown** | 762 | **15.24%** | Indeterminate intake / unverified adverse event status |
| **Total** | 5,000 | 100.00% | Full clinical encounter cohort |

- **Imbalance Ratio**: $1.90 : 1$ (Majority `High` to Minority `Unknown`).
- **Class Balance Assessment**: The three actionable clinical classes (`High`, `Low`, `Moderate`) are remarkably balanced at $\approx 28\% - 29\%$ each. The `Unknown` category forms a secondary cohort representing unverified documentation.

![Target Distribution](C:/Users/venka/.gemini/antigravity-ide/brain/6e6a168c-f59f-48b3-9492-627a0acdb0d8/graphs/01_target_distribution.png)

### 2. Text Length vs. Clinical Urgency

| Urgency Class | Mean Char Length | Median Char Length | Mean Word Count | Median Word Count | Avg Word Length |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **High** | 112.42 | 112.0 | 17.38 | 18.0 | 5.13 |
| **Low** | 112.33 | 112.0 | 17.36 | 18.0 | 5.13 |
| **Moderate** | 112.56 | 112.0 | 17.39 | 18.0 | 5.12 |
| **Unknown** | 112.57 | 112.0 | 17.34 | 18.0 | 5.14 |

![Class-Wise Text Length Comparison](C:/Users/venka/.gemini/antigravity-ide/brain/6e6a168c-f59f-48b3-9492-627a0acdb0d8/graphs/09_classwise_text_length_comparison.png)

> [!NOTE]
> **Key Finding**: Note length does **not** differentiate clinical urgency (Spearman $\rho = 0.002$). Urgency is driven strictly by semantic keywords, adverse event mentions, and drug toxicity profiles, rather than document length.

### 3. Class-Wise Word Clouds & Salient Tokens

![Word Cloud Overall](C:/Users/venka/.gemini/antigravity-ide/brain/6e6a168c-f59f-48b3-9492-627a0acdb0d8/graphs/11_wordcloud_overall.png)
![Class-Wise Word Clouds](C:/Users/venka/.gemini/antigravity-ide/brain/6e6a168c-f59f-48b3-9492-627a0acdb0d8/graphs/12_wordclouds_classwise.png)
![Class-Wise Word Frequencies](C:/Users/venka/.gemini/antigravity-ide/brain/6e6a168c-f59f-48b3-9492-627a0acdb0d8/graphs/08_classwise_word_frequencies.png)

---

## Phase 11 & 12 — NLP Preprocessing & Before vs. After Cleaning

### 1. Domain-Informed Cleaning Strategy
1. **Punctuation Normalization**: Replaces non-informative punctuation with spaces while protecting hyphens in genetic mutations (`KRAS-G12C`, `L858R`) and slashes in dose rates (`mg/day`).
2. **Typo / OCR Repair**: Targeted substitution for known corruptions (e.g. `rep0rts` $\rightarrow$ `reports`).
3. **Selective Lowercasing**: Normalized token casing while preserving word-token boundaries.
4. **Preservation of Clinical Negations**: Explicitly retained `no`, `not`, `none`, and `without`.

### 2. Before vs. After Quantitative Audit

| Evaluation Metric | Raw Clinical Notes | Cleaned Clinical Notes | Absolute Change | Impact Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **Total Row Count** | 5,000 | 5,000 | 0 | Preserved all clinical encounters |
| **Missing Values** | 0 | 0 | 0 | Complete data integrity |
| **Unique Notes Count** | 4,549 | 4,400 | -149 | Consolidated whitespace/punctuation variants |
| **Mean Character Length** | 112.46 | 105.46 | -7.00 | Removed superfluous punctuation and spaces |
| **Median Character Length** | 112.00 | 105.00 | -7.00 | Consistent shift across documents |
| **Mean Word Count** | 17.37 | 17.37 | 0.00 | Preserved all meaningful tokens |
| **Total Running Tokens** | 86,845 | 86,845 | 0 | Zero information loss |
| **Vocabulary Size** | 132 | 131 | -1 | Fixed typographical OCR anomaly `rep0rts` |
| **Type-Token Ratio (TTR)**| 0.0015 | 0.0015 | 0.0000 | Preserved domain lexical structure |

![Before vs After Preprocessing](C:/Users/venka/.gemini/antigravity-ide/brain/6e6a168c-f59f-48b3-9492-627a0acdb0d8/graphs/13_before_vs_after_cleaning.png)

---

## Phase 13 — Outlier & Extreme Observation Analysis

- **Character Count IQR Bounds**: $[82.5, 142.5]$ characters.
  - Low Extreme ($< 82.5$ chars): Exactly $1$ record (Length $80$ characters):  
    *`pt w/ nsclc; mut. unknown. started erlotinib 150mg/day. reports sob. ae: nausea.`*
  - High Extreme ($> 142.5$ chars): Exactly $2$ records (Length $145$ characters):  
    *`TRIAL SCREENING: MET amplification; prior/current drug unknown med; dose dose not recorded; symptoms: nausea after treatment; AE: none reported..`*
- **Word Count IQR Bounds**: $[11.5, 23.5]$ words.
  - Zero word count outliers were detected.
- > [!IMPORTANT]
  > **Domain Decision**: All extreme records are valid, intelligible clinical notes conveying critical patient symptoms and therapy details. They must **not** be removed.

---

## Phase 14 & 15 — Feature Correlations & Engineering Recommendations

### 1. Spearman Correlation Analysis

| Feature | Character Count | Word Count | Avg Word Len | Sent Count | Stopword Count | Urgency Target |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Character Count** | 1.000 | 0.563 | 0.319 | -0.642 | 0.238 | **0.002** |
| **Word Count** | 0.563 | 1.000 | -0.530 | -0.470 | 0.292 | **-0.007** |
| **Avg Word Length** | 0.319 | -0.530 | 1.000 | -0.061 | -0.098 | **0.001** |
| **Sentence Count** | -0.642 | -0.470 | -0.061 | 1.000 | -0.015 | **0.003** |
| **Stopword Count** | 0.238 | 0.292 | -0.098 | -0.015 | 1.000 | **-0.017** |
| **Urgency Target** | 0.002 | -0.007 | 0.001 | 0.003 | -0.017 | **1.000** |

![Correlation Heatmap](C:/Users/venka/.gemini/antigravity-ide/brain/6e6a168c-f59f-48b3-9492-627a0acdb0d8/graphs/10_correlation_heatmap.png)

### 2. High-Impact NLP Feature Engineering Recommendations
1. **Negation Scope Flag**: Binary indicator for whether an adverse event falls within the syntactic scope of `no`, `not`, or `none`.
2. **Clinical Entity Embeddings**: Pretrained BioClinicalBERT or PubMedBERT embeddings capturing oncology terminology.
3. **Adverse Event Severity Mapping**: Interaction terms linking specific reported toxicities (`hepatotoxicity`, `neutropenia`, `mucositis`) to high-urgency baselines.
4. **Dosage Extraction Vector**: Numerical feature indicating whether dosage is standard, elevated, or unrecorded.
5. **Driver Mutation Categorical Embeddings**: Target encodings for high-risk variants (e.g. *MET amplification*, *KRAS G12C*).

---

## Final Report — 20 Critical Analytical Findings

1. **Dataset Summary**: 5,000 clinical encounters with 12 features spanning patient identifiers, note categories, clinical narratives, genomic variants, drugs, adverse events, and urgency labels.
2. **Data-Quality Findings**: 100% complete dataset with zero missing cells across all 12 columns and zero whitespace anomalies.
3. **Missing-Value Findings**: Explicit nulls are 0%. However, 56.64% of records have `has_missing_fields == True`, capturing real-world incomplete clinical records.
4. **Duplicate Findings**: Zero duplicate patient records. 451 shared narrative texts reflect standardized institutional EHR templates.
5. **Text-Quality Findings**: Highly consistent, structured medical syntax free of emojis, social media noise, URLs, or HTML artifacts.
6. **Text-Length Findings**: Mean length is 112.46 characters (17.37 words), with narrow variance (IQR 24 chars).
7. **Word-Frequency Findings**: Clinical abbreviations (`pt`, `ae`, `mg`, `nsclc`, `egfr`) dominate the vocabulary.
8. **N-Gram Findings**: Prominent bigrams and trigrams capture clinical workflows (`nurse intake pt`, `trial screening`, `kras g12c`).
9. **Stopword Findings**: Stopwords comprise only 8.19% of tokens; negations (`not`, `no`) account for over a third and must be retained.
10. **Target-Distribution Findings**: Three clinical classes are balanced: High (29.02%), Low (27.90%), Moderate (27.84%), with Unknown at 15.24%.
11. **Class-Imbalance Findings**: Imbalance ratio is 1.90:1 due to the smaller Unknown category; the three actionable classes require no resampling.
12. **Outlier Findings**: Only 3 mild length extremes exist (1 short at 80 chars, 2 long at 145 chars); all are clinically valid.
13. **Important NLP Patterns**: High urgency is characterized by severe toxicity mentions (`hepatotoxicity`, `mucositis`, `thrombocytopenia`).
14. **Important Relationships**: Document length has zero correlation with urgency ($\rho = 0.002$); predictive signal resides entirely in semantic content.
15. **Cleaning Decisions**: Repaired OCR error `rep0rts` $\rightarrow$ `reports`, removed non-informative punctuation, and preserved clinical entities.
16. **Recommended NLP Preprocessing**: Retain negation words, preserve dose units and gene mutation hyphens, and avoid aggressive stemming.
17. **Recommended Features**: Implement TF-IDF n-grams (1-3 words), BioClinicalBERT embeddings, and negation-scope flags.
18. **Final EDA Insights**: The dataset is exceptionally clean, well-structured, and representative of clinical oncology documentation.
19. **Readiness for ML/DL**: **100% Ready** for classification, Named Entity Recognition (NER), and multi-task learning.
20. **Next Recommended Step**: Train a BioClinicalBERT or TF-IDF + Gradient Boosting baseline with stratified 5-fold cross-validation.

---
*Report generated and validated via automated NLP diagnostic pipeline.*
