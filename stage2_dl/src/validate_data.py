import os
import re
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
RAW_CSV = os.path.join(DATA_RAW, "dl_raw_1000.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
VALIDATION_REPORT_PATH = os.path.join(REPORTS_DIR, "02_validation_report.txt")

EXPECTED_COLUMNS = [
    "Patient_ID", "Encounter_ID", "Encounter_Date", "Cancer_Type", "Cancer_Stage",
    "Age", "Sex", "Organ_Site", "Histopathology_Image_ID", "Histopathology_Image_Path",
    "Tissue_Type", "Histopathology_Label", "Tumor_Grade", "Tumor_Margin_Status",
    "CT_Scan_ID", "CT_Scan_Path", "CT_Slice_Count", "MRI_Scan_ID", "MRI_Scan_Path",
    "MRI_Sequence_Type", "Tumor_Volume_cm3", "Biomarker_Timepoint_Days", "ctDNA_Level",
    "Protein_Marker", "Protein_Marker_Level", "Tumor_Growth_Rate", "Treatment_Drug",
    "Treatment_Response", "Progression_Status", "Progression_Risk", "Image_Quality",
    "Annotation_Status", "Clinical_Notes", "Image_Source", "Image_Label_Confidence"
]

def validate_raw_dataset():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Raw CSV dataset not found at {RAW_CSV}")

    df = pd.read_csv(RAW_CSV, dtype=str).fillna("")
    report_lines = []
    
    report_lines.append("==================================================")
    report_lines.append("STAGE 02 DEEP LEARNING DATASET - VALIDATION REPORT")
    report_lines.append("==================================================")
    report_lines.append("")
    report_lines.append("DISCLAIMER:")
    report_lines.append("This is a SYNTHETIC / MOCK oncology dataset for academic project development only.")
    report_lines.append("Do not use real patient data. Synthetic data is not real clinical data.")
    report_lines.append("")

    # 1. Structural Checks
    report_lines.append("--- 1. STRUCTURAL VALIDATION ---")
    row_count, col_count = len(df), len(df.columns)
    report_lines.append(f"Total Rows: {row_count} (Expected: 1000)")
    report_lines.append(f"Total Columns: {col_count} (Expected: 35)")
    
    col_matches = (list(df.columns) == EXPECTED_COLUMNS)
    report_lines.append(f"Column Schema Matches Expected: {col_matches}")
    if not col_matches:
        missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
        extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
        if missing_cols:
            report_lines.append(f"  Missing columns: {missing_cols}")
        if extra_cols:
            report_lines.append(f"  Unexpected columns: {extra_cols}")
    report_lines.append("")

    # 2. Duplicate Records Check
    report_lines.append("--- 2. DUPLICATE CHECK ---")
    dup_count = df.duplicated().sum()
    report_lines.append(f"Exact Duplicate Rows Found: {dup_count}")
    report_lines.append("")

    # 3. Missing Value Analysis
    report_lines.append("--- 3. MISSING VALUE ANALYSIS ---")
    missing_summary = {}
    for col in df.columns:
        empty_count = (df[col] == "").sum() + (df[col].str.upper().isin(["N/A", "NULL", "NONE"])).sum()
        if empty_count > 0:
            missing_summary[col] = empty_count
            report_lines.append(f"  {col}: {empty_count} missing / incomplete values ({empty_count/row_count*100:.1f}%)")
    report_lines.append(f"Total Columns with Missing Values: {len(missing_summary)}")
    report_lines.append("")

    # 4. Numeric & Data Type Contamination Checks
    report_lines.append("--- 4. NUMERIC CONTAMINATION & RANGE ANOMALIES ---")
    
    # Tumor volume text contamination & negative values
    tv_raw = df["Tumor_Volume_cm3"]
    tv_contaminated = tv_raw.str.contains(r"[a-zA-Z]", regex=True).sum()
    report_lines.append(f"Contaminated Tumor Volume entries (e.g. '25.4 cm3'): {tv_contaminated}")
    
    # ctDNA text contamination
    ctdna_raw = df["ctDNA_Level"]
    ctdna_contaminated = ctdna_raw.str.contains(r"[a-zA-Z]", regex=True).sum()
    report_lines.append(f"Contaminated ctDNA Level entries (e.g. 'copies/mL'): {ctdna_contaminated}")
    
    # Protein Marker Level text contamination
    prot_raw = df["Protein_Marker_Level"]
    prot_contaminated = prot_raw.str.contains(r"[a-zA-Z]", regex=True).sum()
    report_lines.append(f"Contaminated Protein Marker Level entries (e.g. 'ng/mL'): {prot_contaminated}")

    # Age anomalies
    age_clean = df["Age"].str.extract(r"(-?\d+)", expand=False).dropna().astype(int)
    invalid_ages = age_clean[(age_clean < 0) | (age_clean > 120)]
    report_lines.append(f"Invalid Age values (< 0 or > 120): {len(invalid_ages)} (Values: {invalid_ages.tolist()})")
    report_lines.append("")

    # 5. Categorical Consistency Checks
    report_lines.append("--- 5. CATEGORICAL CONSISTENCY CHECKS ---")
    sex_vals = df["Sex"].unique().tolist()
    report_lines.append(f"Unique Sex entries (Inconsistent): {sex_vals}")
    
    stage_vals = df["Cancer_Stage"].unique().tolist()
    report_lines.append(f"Unique Cancer Stage entries (Inconsistent): {stage_vals}")
    
    histo_labels = df["Histopathology_Label"].unique().tolist()
    report_lines.append(f"Unique Histopathology Labels: {histo_labels}")
    report_lines.append("")

    # 6. Image Path & File Existence Validation
    report_lines.append("--- 6. IMAGE FILE & PATH VALIDATION ---")
    
    broken_histo, valid_histo = 0, 0
    missing_histo_paths = 0
    for path in df["Histopathology_Image_Path"]:
        if path == "":
            missing_histo_paths += 1
            continue
        full_img_path = os.path.join(BASE_DIR, path)
        if os.path.exists(full_img_path):
            valid_histo += 1
        else:
            broken_histo += 1

    broken_ct, valid_ct = 0, 0
    for path in df["CT_Scan_Path"]:
        if path == "":
            continue
        full_img_path = os.path.join(BASE_DIR, path)
        if os.path.exists(full_img_path):
            valid_ct += 1
        else:
            broken_ct += 1

    broken_mri, valid_mri = 0, 0
    for path in df["MRI_Scan_Path"]:
        if path == "":
            continue
        full_img_path = os.path.join(BASE_DIR, path)
        if os.path.exists(full_img_path):
            valid_mri += 1
        else:
            broken_mri += 1

    report_lines.append(f"Histopathology Images: Valid on disk = {valid_histo}, Broken paths = {broken_histo}, Missing path refs = {missing_histo_paths}")
    report_lines.append(f"CT Scans: Valid on disk = {valid_ct}, Broken paths = {broken_ct}")
    report_lines.append(f"MRI Scans: Valid on disk = {valid_mri}, Broken paths = {broken_mri}")
    report_lines.append("")

    # 7. Longitudinal Sequence Consistency Check
    report_lines.append("--- 7. LONGITUDINAL SEQUENCE CONSISTENCY ---")
    unique_patients = df["Patient_ID"].nunique()
    report_lines.append(f"Unique Patients in Dataset: {unique_patients}")
    
    pts_with_multiple_tp = (df.groupby("Patient_ID").size() > 1).sum()
    report_lines.append(f"Patients with Multiple Longitudinal Timepoints: {pts_with_multiple_tp}")
    report_lines.append("")
    
    report_lines.append("==================================================")
    report_lines.append("VALIDATION STATUS: ISSUES DETECTED (READY FOR CLEANING)")
    report_lines.append("==================================================")

    report_text = "\n".join(report_lines)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(VALIDATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    print(report_text)
    print(f"\nValidation Report saved to {VALIDATION_REPORT_PATH}")

if __name__ == "__main__":
    validate_raw_dataset()
