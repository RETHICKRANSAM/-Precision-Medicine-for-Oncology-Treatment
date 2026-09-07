import os
import glob
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "dl_raw_1000.csv")
CLEANED_CSV = os.path.join(BASE_DIR, "data", "processed", "dl_cleaned.csv")
IMAGE_META_CSV = os.path.join(BASE_DIR, "data", "processed", "image_metadata.csv")
TEMPORAL_CSV = os.path.join(BASE_DIR, "data", "processed", "temporal_biomarker_sequences.csv")

HISTO_DIR = os.path.join(BASE_DIR, "data", "raw", "histopathology_images")
CT_DIR = os.path.join(BASE_DIR, "data", "raw", "ct_images")
MRI_DIR = os.path.join(BASE_DIR, "data", "raw", "mri_images")

REPORT_DOCX = os.path.join(BASE_DIR, "reports", "STAGE2_DATA_ENGINEER_REPORT.docx")
CLEANING_REPORT = os.path.join(BASE_DIR, "reports", "03_cleaning_report.txt")
README_FILE = os.path.join(BASE_DIR, "README.md")

def verify_all():
    print("Performing Stage 02 Data Package Final Verification...")
    
    # 1. Raw CSV check
    assert os.path.exists(RAW_CSV), "Raw CSV does not exist!"
    df_raw = pd.read_csv(RAW_CSV, dtype=str)
    raw_record_count = len(df_raw)
    assert raw_record_count == 1000, f"Raw CSV records count is {raw_record_count}, expected 1000!"
    assert len(df_raw.columns) == 35, f"Raw CSV column count is {len(df_raw.columns)}, expected 35!"
    
    # 2. Image files check
    histo_images = glob.glob(os.path.join(HISTO_DIR, "**", "*.png"), recursive=True)
    ct_images = glob.glob(os.path.join(CT_DIR, "**", "*.png"), recursive=True)
    mri_images = glob.glob(os.path.join(MRI_DIR, "**", "*.png"), recursive=True)
    
    histo_count = len(histo_images)
    ct_count = len(ct_images)
    mri_count = len(mri_images)
    
    assert histo_count > 0, "No histopathology images found!"
    assert ct_count > 0, "No CT images found!"
    assert mri_count > 0, "No MRI images found!"
    
    # 3. Processed CSV checks
    assert os.path.exists(CLEANED_CSV), "Cleaned CSV does not exist!"
    assert os.path.exists(IMAGE_META_CSV), "Image metadata CSV does not exist!"
    assert os.path.exists(TEMPORAL_CSV), "Temporal sequence CSV does not exist!"
    
    df_clean = pd.read_csv(CLEANED_CSV)
    df_img = pd.read_csv(IMAGE_META_CSV)
    df_temp = pd.read_csv(TEMPORAL_CSV)
    
    assert len(df_clean) <= raw_record_count, "Cleaned dataset row count is greater than raw!"
    assert len(df_img) > 0, "Image metadata is empty!"
    assert len(df_temp) > 0, "Temporal sequences dataset is empty!"
    
    # Check temporal ordering
    assert df_temp['Patient_ID'].is_monotonic_increasing or list(df_temp['Patient_ID']) == sorted(list(df_temp['Patient_ID'])), "Temporal sequence not sorted by Patient_ID!"
    
    # 4. Reports & README check
    assert os.path.exists(REPORT_DOCX), "STAGE2_DATA_ENGINEER_REPORT.docx does not exist!"
    assert os.path.exists(CLEANING_REPORT), "03_cleaning_report.txt does not exist!"
    assert os.path.exists(README_FILE), "README.md does not exist!"

    print("All automated verification checks passed cleanly!\n")

    # Final Summary Banner
    print("========================================")
    print("STAGE 02 DEEP LEARNING DATA PACKAGE")
    print("===================================")
    print("")
    print(f"Raw Records: {raw_record_count}")
    print("")
    print(f"Histopathology Images: {histo_count}")
    print(f"CT Images: {ct_count}")
    print(f"MRI Images: {mri_count}")
    print("")
    print("Image Metadata File:")
    print("data/processed/image_metadata.csv")
    print("")
    print("Temporal Biomarker File:")
    print("data/processed/temporal_biomarker_sequences.csv")
    print("")
    print("Data Engineer Report:")
    print("reports/STAGE2_DATA_ENGINEER_REPORT.docx")
    print("")
    print("Status:")
    print("SUCCESS")
    print("========================================")

if __name__ == "__main__":
    verify_all()
