"""
Script: clean_dataset.py
Purpose: Deduplicate and clean clinical text dataset for SLM training readiness.
Preserves the original raw dataset intact and exports the cleaned version.
"""

import pandas as pd
import re

ORIGINAL_FILE = "slm_master_dataset_with_ner.csv.xls"
CLEANED_FILE = "slm_master_dataset_deduplicated_clean.csv"

print("==================================================")
print("1. LOADING RAW DATASET")
print("==================================================")
try:
    df_raw = pd.read_csv(ORIGINAL_FILE)
except Exception:
    df_raw = pd.read_excel(ORIGINAL_FILE)

initial_count = len(df_raw)
print(f"Loaded {initial_count:,} records from '{ORIGINAL_FILE}'.")

print("\n==================================================")
print("2. TEXT NORMALIZATION & PREPROCESSING")
print("==================================================")
# Clean multiple consecutive spaces in clinical_report and summary while preserving all clinical terms
df_clean = df_raw.copy()
for col in ["clinical_report", "summary"]:
    df_clean[col] = df_clean[col].astype(str).apply(lambda x: re.sub(r" {2,}", " ", x.strip()))

# Identify exact text duplicates
dup_mask = df_clean.duplicated(subset=["clinical_report", "summary"], keep="first")
duplicate_count = dup_mask.sum()
print(f"Identified {duplicate_count:,} duplicate (clinical_report, summary) records.")

print("\n==================================================")
print("3. DEDUPLICATION")
print("==================================================")
# Deduplicate: keep first valid clinical occurrence
df_deduped = df_clean[~dup_mask].reset_index(drop=True)
final_count = len(df_deduped)

print(f"Original record count:  {initial_count:,}")
print(f"Removed duplicate rows: {duplicate_count:,}")
print(f"Cleaned record count:   {final_count:,} (100% unique input-output pairs)")

# Verification checks
assert df_deduped.duplicated(subset=["clinical_report", "summary"]).sum() == 0, "Error: Duplicate pairs remain!"
assert len(df_deduped) == initial_count - duplicate_count, "Error: Count mismatch!"

print("\nVerification passed:")
print(f"  - Unique clinical reports: {df_deduped['clinical_report'].nunique():,}")
print(f"  - Unique summaries:        {df_deduped['summary'].nunique():,}")
print(f"  - Remaining duplicate pairs: 0")

print("\n==================================================")
print("4. EXPORTING CLEANED DATASET")
print("==================================================")
df_deduped.to_csv(CLEANED_FILE, index=False, encoding="utf-8")
print(f"Saved cleaned dataset to: '{CLEANED_FILE}'")

print("\nSample Cleaned Records:")
print(df_deduped[["Patient_ID", "urgency", "clinical_report", "summary"]].head(3).to_string())
print("\n==================================================")
print("CLEANING COMPLETE!")
print("==================================================")
