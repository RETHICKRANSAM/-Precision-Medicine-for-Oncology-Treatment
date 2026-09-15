# Outlier and Anomaly Analysis Report

## 1. Methodology
Outliers were identified using the non-parametric Interquartile Range (IQR) method:
`[Q25 - 1.5 * IQR, Q75 + 1.5 * IQR]`
across input length, target length, and expansion ratios.

## 2. Outlier Observations
- **Sample Outliers Identified:** `46 metric outlier events logged in outputs/outlier_report.csv`.
- **Classification:**
  - High word count notes: Valid detailed clinical consultations (legitimate long clinical notes).
  - Low word count notes: Brief triage intakes (e.g. 5-7 words).
  - Extreme ratios: A small fraction of records where very brief notes yield standard multi-sentence summaries.

## 3. Modeling Recommendation
Do not arbitrarily discard statistical outliers. Truncation or dynamic padding at sequence length 256 tokens covers over 99.5% of both input and target distributions.
