import os
import re
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

RAW_CSV = os.path.join(DATA_RAW, "dl_raw_1000.csv")
CLEANED_CSV = os.path.join(DATA_PROC, "dl_cleaned.csv")
IMAGE_META_CSV = os.path.join(DATA_PROC, "image_metadata.csv")
TEMPORAL_CSV = os.path.join(DATA_PROC, "temporal_biomarker_sequences.csv")
CLEANING_REPORT = os.path.join(REPORTS_DIR, "03_cleaning_report.txt")

def clean_dataset():
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Raw file {RAW_CSV} not found.")

    os.makedirs(DATA_PROC, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    df_raw = pd.read_csv(RAW_CSV, dtype=str).fillna("")
    initial_row_count = len(df_raw)

    print(f"Starting cleaning pipeline on {initial_row_count} raw records...")

    # Step 1: Remove exact duplicates
    df = df_raw.drop_duplicates().copy()
    duplicates_removed = initial_row_count - len(df)
    print(f"Removed {duplicates_removed} duplicate rows.")

    # Step 2: Clean Whitespace across all string fields
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    # Step 3: Standardize Categorical Values
    # Sex
    def standardize_sex(val):
        v = str(val).upper()
        if "FEMALE" in v or v == "F":
            return "Female"
        elif "MALE" in v or v == "M":
            return "Male"
        return "Unknown"
    df["Sex"] = df["Sex"].apply(standardize_sex)

    # Cancer Stage
    def standardize_stage(val):
        v = str(val).upper()
        if "IV" in v:
            return "Stage IV"
        elif "III" in v:
            return "Stage III"
        elif "II" in v:
            return "Stage II"
        elif "I" in v:
            return "Stage I"
        return "Stage II" # Default mode
    df["Cancer_Stage"] = df["Cancer_Stage"].apply(standardize_stage)

    # Histopathology Label
    def standardize_histo_label(val):
        v = str(val).upper()
        if "MALIGNANT" in v:
            return "Malignant"
        elif "BENIGN" in v:
            return "Benign"
        elif "ATYPICAL" in v:
            return "Atypical"
        elif "NECROTIC" in v:
            return "Necrotic"
        elif "INFLAMMATORY" in v:
            return "Inflammatory"
        return "Unknown"
    df["Histopathology_Label"] = df["Histopathology_Label"].apply(standardize_histo_label)

    # Progression Status
    def standardize_prog_status(val):
        v = str(val).upper()
        if "PROGRESSED" in v and "NOT" not in v:
            return "Progressed"
        elif "NOT" in v:
            return "Not Progressed"
        elif "STABLE" in v:
            return "Stable"
        return "Stable"
    df["Progression_Status"] = df["Progression_Status"].apply(standardize_prog_status)

    # Progression Risk
    def standardize_prog_risk(val):
        v = str(val).upper()
        if "HIGH" in v:
            return "High"
        elif "LOW" in v:
            return "Low"
        elif "MODERATE" in v:
            return "Moderate"
        return "Moderate"
    df["Progression_Risk"] = df["Progression_Risk"].apply(standardize_prog_risk)

    # Step 4: Extract and clean numeric fields
    def extract_numeric(val):
        if not val or val in ["N/A", "NULL", "NONE", "Unknown"]:
            return np.nan
        # Handle percentages
        if "%" in str(val):
            m = re.search(r"(\d+(\.\d+)?)", str(val))
            if m:
                return float(m.group(1)) / 100.0 if float(m.group(1)) > 1.0 else float(m.group(1))
        # Handle general numbers
        m = re.search(r"(-?\d+(\.\d+)?)", str(val))
        if m:
            return float(m.group(1))
        return np.nan

    num_cols = ["Age", "CT_Slice_Count", "Tumor_Volume_cm3", "Biomarker_Timepoint_Days", 
                "ctDNA_Level", "Protein_Marker_Level", "Tumor_Growth_Rate", "Image_Label_Confidence"]

    for col in num_cols:
        df[col] = df[col].apply(extract_numeric)

    # Clean Age bounds (0 to 110)
    median_age = df["Age"][(df["Age"] >= 0) & (df["Age"] <= 110)].median()
    df["Age"] = df["Age"].apply(lambda x: median_age if (pd.isna(x) or x < 0 or x > 110) else x).astype(int)

    # Clean Tumor Volume (must be positive)
    median_vol = df["Tumor_Volume_cm3"][df["Tumor_Volume_cm3"] > 0].median()
    df["Tumor_Volume_cm3"] = df["Tumor_Volume_cm3"].apply(lambda x: median_vol if (pd.isna(x) or x <= 0) else x)

    # Impute ctDNA & Protein Marker levels with median
    med_ctdna = df["ctDNA_Level"][df["ctDNA_Level"] > 0].median()
    df["ctDNA_Level"] = df["ctDNA_Level"].fillna(med_ctdna)

    med_prot = df["Protein_Marker_Level"][df["Protein_Marker_Level"] > 0].median()
    df["Protein_Marker_Level"] = df["Protein_Marker_Level"].fillna(med_prot)

    # Impute Biomarker_Timepoint_Days
    df["Biomarker_Timepoint_Days"] = df["Biomarker_Timepoint_Days"].fillna(0).astype(int)

    # Impute Confidence
    df["Image_Label_Confidence"] = df["Image_Label_Confidence"].fillna(0.85)

    # Step 5: Date Format Standardization
    def standardize_date(val):
        if not val or str(val).strip() == "":
            return "2025-01-10"
        try:
            dt = pd.to_datetime(val, format="mixed")
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return "2025-01-10"
    df["Encounter_Date"] = df["Encounter_Date"].apply(standardize_date)

    # Step 6: Image Path & File Verification
    def verify_and_clean_path(path):
        if not path or path == "":
            return ""
        clean_path = str(path).replace("\\", "/")
        full_p = os.path.join(BASE_DIR, clean_path)
        if os.path.exists(full_p):
            return clean_path
        return "" # Clear broken path

    df["Histopathology_Image_Path"] = df["Histopathology_Image_Path"].apply(verify_and_clean_path)
    df["CT_Scan_Path"] = df["CT_Scan_Path"].apply(verify_and_clean_path)
    df["MRI_Scan_Path"] = df["MRI_Scan_Path"].apply(verify_and_clean_path)

    # If image path is cleared, clear corresponding image ID
    df.loc[df["Histopathology_Image_Path"] == "", "Histopathology_Image_ID"] = ""
    df.loc[df["CT_Scan_Path"] == "", "CT_Scan_ID"] = ""
    df.loc[df["MRI_Scan_Path"] == "", "MRI_Scan_ID"] = ""

    # Step 7: Sort by Patient_ID and Biomarker_Timepoint_Days
    df = df.sort_values(by=["Patient_ID", "Biomarker_Timepoint_Days"]).reset_index(drop=True)

    # Save cleaned full dataset
    df.to_csv(CLEANED_CSV, index=False)
    print(f"Saved cleaned tabular dataset to {CLEANED_CSV} ({len(df)} rows)")

    # Step 8: Generate Processed Dataset 1 -> image_metadata.csv (for CNN team)
    # Filter rows with valid Histopathology images
    df_histo = df[df["Histopathology_Image_Path"] != ""].copy()
    image_metadata = pd.DataFrame({
        "Image_ID": df_histo["Histopathology_Image_ID"],
        "Patient_ID": df_histo["Patient_ID"],
        "Image_Type": "Histopathology",
        "Image_Path": df_histo["Histopathology_Image_Path"],
        "Label": df_histo["Histopathology_Label"],
        "Cancer_Type": df_histo["Cancer_Type"],
        "Cancer_Stage": df_histo["Cancer_Stage"],
        "Tissue_Type": df_histo["Tissue_Type"],
        "Tumor_Grade": df_histo["Tumor_Grade"],
        "Image_Quality": df_histo["Image_Quality"],
        "Annotation_Status": df_histo["Annotation_Status"]
    })
    # Remove any duplicates in Image_ID
    image_metadata = image_metadata.drop_duplicates(subset=["Image_ID"]).reset_index(drop=True)
    image_metadata.to_csv(IMAGE_META_CSV, index=False)
    print(f"Saved image metadata dataset to {IMAGE_META_CSV} ({len(image_metadata)} records)")

    # Step 9: Generate Processed Dataset 2 -> temporal_biomarker_sequences.csv (for LSTM / Transformer team)
    temporal_sequences = pd.DataFrame({
        "Patient_ID": df["Patient_ID"],
        "Timepoint_Days": df["Biomarker_Timepoint_Days"],
        "ctDNA_Level": df["ctDNA_Level"],
        "Protein_Marker": df["Protein_Marker"],
        "Protein_Marker_Level": df["Protein_Marker_Level"],
        "Tumor_Volume_cm3": df["Tumor_Volume_cm3"],
        "Tumor_Growth_Rate": df["Tumor_Growth_Rate"],
        "Progression_Status": df["Progression_Status"],
        "Progression_Risk": df["Progression_Risk"]
    }).sort_values(by=["Patient_ID", "Timepoint_Days"]).reset_index(drop=True)
    
    temporal_sequences.to_csv(TEMPORAL_CSV, index=False)
    print(f"Saved temporal biomarker sequence dataset to {TEMPORAL_CSV} ({len(temporal_sequences)} records)")

    # Step 10: Write 03_cleaning_report.txt
    with open(CLEANING_REPORT, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("STAGE 02 DEEP LEARNING DATASET - CLEANING REPORT\n")
        f.write("==================================================\n\n")
        f.write("DISCLAIMER:\n")
        f.write("This is a SYNTHETIC / MOCK oncology dataset for academic project development only.\n")
        f.write("Do not use real patient data. Synthetic data is not real clinical data.\n\n")
        f.write(f"Raw Input Records: {initial_row_count}\n")
        f.write(f"Duplicates Removed: {duplicates_removed}\n")
        f.write(f"Cleaned Tabular Records Saved: {len(df)}\n")
        f.write(f"Processed Image Metadata Records: {len(image_metadata)}\n")
        f.write(f"Processed Temporal Sequence Records: {len(temporal_sequences)}\n\n")
        f.write("SUMMARY OF CLEANING OPERATIONS PERFORMED:\n")
        f.write("1. Duplicate Removal: Identified and purged exact duplicate encounter rows.\n")
        f.write("2. Categorical Standardization: Unified case and labels for Sex, Cancer_Stage, Histopathology_Label, Progression_Status, and Progression_Risk.\n")
        f.write("3. Numeric Contamination Stripping: Regex extracted numerical values from text-contaminated strings ('cm3', 'copies/mL', 'slices', 'days', '%').\n")
        f.write("4. Outlier & Range Validation: Corrected negative Age and negative Tumor_Volume values using median imputation.\n")
        f.write("5. Date Format Normalization: Parsed mixed dates (MM/DD/YYYY, DD-MM-YYYY) into standardized ISO 8601 YYYY-MM-DD.\n")
        f.write("6. Physical Image File Verification: Checked every image relative path on disk; cleared non-existent path references (e.g. HIMG99999.png).\n")
        f.write("7. Longitudinal Sequence Sorting: Sorted data by Patient_ID and Biomarker_Timepoint_Days to preserve strict temporal order.\n")
        f.write("8. Handoff Dataset Generation: Created specialized image_metadata.csv for CNN model and temporal_biomarker_sequences.csv for LSTM/Transformer model.\n\n")
        f.write("STATUS: CLEANING COMPLETED SUCCESSFULLY\n")

    print(f"Cleaning report saved to {CLEANING_REPORT}")

if __name__ == "__main__":
    clean_dataset()
