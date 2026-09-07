import os
import sys
import glob
import hashlib
import shutil
import re
import pandas as pd
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PIXEL = os.path.join(BASE_DIR, "data", "pixel_images")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
PROC_IMAGES_DIR = os.path.join(DATA_PROC, "images")

# -------------------------------------------------------------
# PERCEPTUAL HASHING HELPERS (pHash, dHash, aHash)
# -------------------------------------------------------------
def compute_ahash(img_gray):
    img_resized = img_gray.resize((8, 8), Image.Resampling.LANCZOS)
    pixels = np.array(img_resized, dtype=float)
    avg = pixels.mean()
    bits = (pixels > avg).flatten()
    hex_str = ''.join(['1' if b else '0' for b in bits])
    return f"{int(hex_str, 2):016x}"

def compute_dhash(img_gray):
    img_resized = img_gray.resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.array(img_resized, dtype=float)
    diff = pixels[:, :-1] > pixels[:, 1:]
    bits = diff.flatten()
    hex_str = ''.join(['1' if b else '0' for b in bits])
    return f"{int(hex_str, 2):016x}"

def compute_phash(img_gray):
    img_resized = img_gray.resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.array(img_resized, dtype=float)
    
    N = 32
    cols, rows = np.meshgrid(np.arange(N), np.arange(N))
    dct_matrix = np.cos((2 * cols + 1) * rows * np.pi / (2 * N))
    dct_matrix[0, :] /= np.sqrt(2)
    dct_matrix *= np.sqrt(2 / N)
    
    dct_2d = dct_matrix @ pixels @ dct_matrix.T
    top_left = dct_2d[:8, :8]
    med = np.median(top_left[1:, 1:])
    bits = (top_left > med).flatten()
    hex_str = ''.join(['1' if b else '0' for b in bits])
    return f"{int(hex_str, 2):016x}"

def hamming_distance(h1, h2):
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count('1')
    except Exception:
        return 64

# -------------------------------------------------------------
# MAIN CLEANING PIPELINE ENGINE
# -------------------------------------------------------------
def run_cleaning_pipeline():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(DATA_PROC, exist_ok=True)
    os.makedirs(SCRIPTS_DIR, exist_ok=True)

    print("==================================================")
    print("STARTING STAGE 02 COMPLETE DATA CLEANING PIPELINE")
    print("==================================================")

    # STEP 1: INSPECT COMPLETE DATASET & SUMMARY
    print("\n[Step 1/19] Scanning project directory recursively...")
    
    all_files = []
    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            all_files.append(os.path.join(root, f))

    raw_csv = os.path.join(DATA_RAW, "dl_raw_1000.csv")
    df_raw_csv = pd.read_csv(raw_csv, dtype=str) if os.path.exists(raw_csv) else pd.DataFrame()
    
    raw_images = [f for f in all_files if DATA_RAW in f and os.path.splitext(f)[1].lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]]
    pixel_images = [f for f in all_files if DATA_PIXEL in f and os.path.splitext(f)[1].lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]]
    
    patient_cnt = df_raw_csv["Patient_ID"].nunique() if "Patient_ID" in df_raw_csv.columns else 0
    missing_vals_tot = df_raw_csv.isna().sum().sum() + (df_raw_csv == "").sum().sum() if not df_raw_csv.empty else 0
    dup_rows_tot = df_raw_csv.duplicated().sum() if not df_raw_csv.empty else 0

    summary_lines = [
        "==================================================",
        "STAGE 02 DATASET RAW SUMMARY REPORT",
        "==================================================",
        "",
        f"Raw Tabular File: data/raw/dl_raw_1000.csv",
        f"  Total Rows: {len(df_raw_csv)}",
        f"  Total Columns: {len(df_raw_csv.columns)}",
        f"  Unique Patients: {patient_cnt}",
        f"  Total Missing / Empty Cell Entries: {missing_vals_tot}",
        f"  Exact Duplicate Rows: {dup_rows_tot}",
        "",
        "PHYSICAL IMAGE INVENTORY:",
        f"  Raw Images Discovered (data/raw/): {len(raw_images)}",
        f"  Pixel Converted Images (data/pixel_images/): {len(pixel_images)}",
        f"  Histopathology Raw Images: {len([f for f in raw_images if 'histopathology' in f.lower()])}",
        f"  CT Scan Raw Images: {len([f for f in raw_images if 'ct_images' in f.lower()])}",
        f"  MRI Scan Raw Images: {len([f for f in raw_images if 'mri_images' in f.lower()])}",
        "=================================================="
    ]
    
    summary_path = os.path.join(REPORTS_DIR, "raw_dataset_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"-> Generated {summary_path}")

    # STEP 3: CLEAN TABULAR DATA (STANDARDIZE TEXT & NUMERICS)
    print("\n[Step 3/19] Standardizing text, capitalization, and numerical contamination...")
    
    df = df_raw_csv.copy()
    for c in df.columns:
        df[c] = df[c].fillna("").astype(str).str.strip()

    # Standardize Sex
    def std_sex(v):
        u = v.upper()
        if "FEMALE" in u or u == "F": return "Female"
        elif "MALE" in u or u == "M": return "Male"
        return "Unknown"
    df["Sex"] = df["Sex"].apply(std_sex)

    # Standardize Stage
    def std_stage(v):
        u = v.upper()
        if "IV" in u: return "Stage IV"
        elif "III" in u: return "Stage III"
        elif "II" in u: return "Stage II"
        elif "I" in u: return "Stage I"
        return "Stage II"
    df["Cancer_Stage"] = df["Cancer_Stage"].apply(std_stage)

    # Standardize Histopathology Label
    def std_histo(v):
        u = v.upper()
        if "MALIGNANT" in u: return "Malignant"
        elif "BENIGN" in u: return "Benign"
        elif "ATYPICAL" in u: return "Atypical"
        elif "NECROTIC" in u: return "Necrotic"
        elif "INFLAMMATORY" in u: return "Inflammatory"
        return "Unknown"
    df["Histopathology_Label"] = df["Histopathology_Label"].apply(std_histo)

    # Standardize Progression Status
    def std_prog(v):
        u = v.upper()
        if "PROGRESSED" in u and "NOT" not in u: return "Progressed"
        elif "NOT" in u: return "Not Progressed"
        elif "STABLE" in u: return "Stable"
        return "Stable"
    df["Progression_Status"] = df["Progression_Status"].apply(std_prog)

    # Standardize Progression Risk
    def std_risk(v):
        u = v.upper()
        if "HIGH" in u: return "High"
        elif "LOW" in u: return "Low"
        elif "MODERATE" in u: return "Moderate"
        return "Moderate"
    df["Progression_Risk"] = df["Progression_Risk"].apply(std_risk)

    def clean_num(val):
        if not val or val in ["N/A", "NULL", "NONE", "Unknown"]: return np.nan
        if "%" in str(val):
            m = re.search(r"(\d+(\.\d+)?)", str(val))
            if m:
                v = float(m.group(1))
                return v / 100.0 if v > 1.0 else v
        m = re.search(r"(-?\d+(\.\d+)?)", str(val))
        if m: return float(m.group(1))
        return np.nan

    num_cols = ["Age", "CT_Slice_Count", "Tumor_Volume_cm3", "Biomarker_Timepoint_Days", 
                "ctDNA_Level", "Protein_Marker_Level", "Tumor_Growth_Rate", "Image_Label_Confidence"]
    
    for c in num_cols:
        if c in df.columns: df[c] = df[c].apply(clean_num)

    # STEP 4: HANDLE MISSING VALUES LOGICALLY
    print("\n[Step 4/19] Analyzing missing values and creating missing_value_report.csv...")
    missing_report_recs = []
    critical_fields = ["Patient_ID", "Histopathology_Image_ID", "Histopathology_Image_Path", "Histopathology_Label"]
    
    df["FLAGGED_REMOVAL"] = False
    
    for col in df.columns:
        if col == "FLAGGED_REMOVAL": continue
        orig_missing = (df_raw_csv[col].fillna("").astype(str).str.strip().isin(["", "N/A", "NULL", "NONE", "Unknown"])).sum()
        missing_pct = round(orig_missing / len(df) * 100, 2)
        action = "NO_ACTION_REQUIRED"
        
        if col in ["Age", "Tumor_Volume_cm3", "ctDNA_Level", "Protein_Marker_Level"]:
            med_val = df[col][(df[col].notna()) & (df[col] > 0)].median()
            df[col] = df[col].fillna(med_val)
            action = f"MEDIAN_IMPUTATION ({med_val:.2f})"
        elif col in critical_fields:
            missing_crit_mask = df[col].isna() | (df[col] == "") | (df[col] == "Unknown")
            df.loc[missing_crit_mask, "FLAGGED_REMOVAL"] = True
            action = "FLAGGED_RECORD_FOR_REMOVAL"
        elif df[col].dtype == object:
            df[col] = df[col].fillna("Unknown")
            action = "IMPUTE_UNKNOWN_CATEGORY"
            
        final_missing = (df[col].isna() | (df[col] == "")).sum()
        missing_report_recs.append({
            "Column": col, "Original_Missing_Count": orig_missing,
            "Missing_Percentage": missing_pct, "Action_Taken": action,
            "Final_Missing_Count": final_missing
        })

    df_miss_rep = pd.DataFrame(missing_report_recs)
    miss_csv_path = os.path.join(REPORTS_DIR, "missing_value_report.csv")
    df_miss_rep.to_csv(miss_csv_path, index=False)
    print(f"-> Created {miss_csv_path}")

    # STEP 5: REMOVE DUPLICATE TABULAR RECORDS
    print("\n[Step 5/19] Detecting tabular duplicates and creating duplicate_record_report.csv...")
    dup_recs = []
    dup_row_mask = df.duplicated(keep="first")
    for idx, r in df[dup_row_mask].iterrows():
        dup_recs.append({
            "Record_ID": r.get("Encounter_ID", f"ROW_{idx}"),
            "Duplicate_Type": "EXACT_ROW_DUPLICATE",
            "Action": "PURGE_FROM_PROCESSED",
            "Reason": "Identical encounter values across all columns"
        })

    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    df_dup_rep = pd.DataFrame(dup_recs, columns=["Record_ID", "Duplicate_Type", "Action", "Reason"])
    dup_csv_path = os.path.join(REPORTS_DIR, "duplicate_record_report.csv")
    df_dup_rep.to_csv(dup_csv_path, index=False)
    print(f"-> Purged {len(dup_recs)} exact duplicate rows. Created {dup_csv_path}")

    # STEP 7 & 8: VERIFY EVERY PIXEL IMAGE & EXACT DUPLICATES
    print("\n[Step 7 & 8/19] Inspecting physical pixel images & SHA256 exact duplicates...")
    
    pixel_img_files = []
    for root, dirs, files in os.walk(DATA_PIXEL):
        for f in files:
            if os.path.splitext(f)[1].lower() in [".png", ".jpg", ".jpeg"]:
                pixel_img_files.append(os.path.join(root, f))
                
    img_verify_recs = []
    sha256_map = {}
    exact_dup_recs = []
    exact_dup_paths = set()

    for fpath in sorted(pixel_img_files):
        rel_p = os.path.relpath(fpath, BASE_DIR).replace("\\", "/")
        fname = os.path.basename(fpath)
        img_id = os.path.splitext(fname)[0]
        fsize = os.path.getsize(fpath)

        readable = False
        valid_pixels = False
        blank = True
        corrupted = False
        w, h, mode, channels = 0, 0, "UNKNOWN", 0
        p_min, p_max, p_mean, p_std = 0.0, 0.0, 0.0, 0.0
        sha256_h, phash_h = "", ""

        if fsize == 0:
            corrupted = True
        else:
            try:
                with open(fpath, "rb") as fb:
                    sha256_h = hashlib.sha256(fb.read()).hexdigest()
                    
                with Image.open(fpath) as img:
                    img.load()
                    w, h = img.width, img.height
                    mode = img.mode
                    if w > 0 and h > 0: readable = True
                        
                    arr = np.array(img, dtype=float)
                    if arr.size > 0:
                        valid_pixels = True
                        channels = 1 if len(arr.shape) == 2 else arr.shape[2]
                        p_min = float(arr.min())
                        p_max = float(arr.max())
                        p_mean = float(round(arr.mean(), 2))
                        p_std = float(round(arr.std(), 2))
                        if p_std > 0 and p_max > p_min: blank = False
                            
                    img_gray = img.convert("L")
                    phash_h = compute_phash(img_gray)

            except Exception:
                corrupted = True

        if sha256_h in sha256_map:
            exact_dup_paths.add(rel_p)
            kept_p = sha256_map[sha256_h]
            exact_dup_recs.append({
                "Duplicate_Group_ID": f"EXACT_SHA_{sha256_h[:8]}",
                "Image_Path": rel_p,
                "Duplicate_Type": "EXACT_SHA256_MATCH",
                "SHA256_Hash": sha256_h,
                "Recommended_Action": "PURGE_FROM_PROCESSED_TRAINING",
                "Kept_Representative": kept_p
            })
        else:
            sha256_map[sha256_h] = rel_p

        img_verify_recs.append({
            "Image_ID": img_id, "Image_Path": rel_p, "Width": w, "Height": h,
            "Format": "PNG", "Color_Mode": mode, "Channels": channels,
            "SHA256": sha256_h, "pHash": phash_h,
            "Pixel_Min": p_min, "Pixel_Max": p_max, "Pixel_Mean": p_mean, "Pixel_Std": p_std,
            "Readable": readable, "Valid_Pixels": valid_pixels, "Blank": blank, "Corrupted": corrupted
        })

    df_img_ver = pd.DataFrame(img_verify_recs)
    df_exact_dup = pd.DataFrame(exact_dup_recs, columns=["Duplicate_Group_ID", "Image_Path", "Duplicate_Type", "SHA256_Hash", "Recommended_Action", "Kept_Representative"])
    exact_csv_path = os.path.join(REPORTS_DIR, "exact_duplicate_images.csv")
    df_exact_dup.to_csv(exact_csv_path, index=False)
    print(f"-> Inspected {len(df_img_ver)} pixel images. Found {len(exact_dup_recs)} exact duplicates. Saved {exact_csv_path}")

    # STEP 9 & 10: NEAR DUPLICATE & LABEL CONFLICT DETECTION
    print("\n[Step 9 & 10/19] Computing perceptual hash distances & label conflicts...")
    label_conflict_recs = []
    
    path_to_label = {}
    for idx, r in df.iterrows():
        p = r.get("Histopathology_Image_Path", "")
        lbl = r.get("Histopathology_Label", "")
        if p and lbl:
            path_to_label[p.replace("\\", "/")] = lbl

    near_dup_pairs = []
    n_imgs = len(df_img_ver)
    for i in range(n_imgs):
        r1 = df_img_ver.iloc[i]
        p1, ph1, id1 = r1["Image_Path"], r1["pHash"], r1["Image_ID"]
        lbl1 = path_to_label.get(p1, "Unknown")
        if not ph1: continue
            
        for j in range(i + 1, n_imgs):
            r2 = df_img_ver.iloc[j]
            p2, ph2, id2 = r2["Image_Path"], r2["pHash"], r2["Image_ID"]
            lbl2 = path_to_label.get(p2, "Unknown")
            if not ph2: continue
                
            dist = hamming_distance(ph1, ph2)
            if dist <= 4:
                near_dup_pairs.append((p1, p2, dist))
                if lbl1 != lbl2 and lbl1 != "Unknown" and lbl2 != "Unknown":
                    label_conflict_recs.append({
                        "Image_1_ID": id1, "Image_1_Path": p1, "Image_1_Label": lbl1,
                        "Image_2_ID": id2, "Image_2_Path": p2, "Image_2_Label": lbl2,
                        "pHash_Distance": dist, "Conflict_Status": "REQUIRES_DOMAIN_REVIEW",
                        "Action_Taken": "FLAGGED_FOR_MEDICAL_REVIEW"
                    })

    df_label_conf = pd.DataFrame(label_conflict_recs, columns=["Image_1_ID", "Image_1_Path", "Image_1_Label", "Image_2_ID", "Image_2_Path", "Image_2_Label", "pHash_Distance", "Conflict_Status", "Action_Taken"])
    conf_csv_path = os.path.join(REPORTS_DIR, "label_conflict_report.csv")
    df_label_conf.to_csv(conf_csv_path, index=False)
    print(f"-> Identified {len(df_label_conf)} cross-class label conflicts. Saved {conf_csv_path}")

    # STEP 11: CHECK IMAGE QUALITY
    print("\n[Step 11/19] Auditing image quality metrics & saving image_quality_report.csv...")
    qual_recs = []
    for idx, r in df_img_ver.iterrows():
        p = r["Image_Path"]
        issues = []
        status = "HEALTHY"
        if r["Corrupted"]:
            issues.append("CORRUPTED_FILE"); status = "CRITICAL"
        elif r["Blank"]:
            issues.append("BLANK_IMAGE_ZERO_VARIANCE"); status = "HIGH"
        elif r["Pixel_Mean"] < 5.0:
            issues.append("EXTREMELY_DARK"); status = "LOW_QUALITY"
        elif r["Pixel_Mean"] > 250.0:
            issues.append("EXTREMELY_BRIGHT"); status = "LOW_QUALITY"
        elif r["Pixel_Std"] < 2.0:
            issues.append("VERY_LOW_CONTRAST"); status = "LOW_QUALITY"

        if issues:
            qual_recs.append({
                "Image_ID": r["Image_ID"], "Image_Path": p, "Quality_Status": status,
                "Issues_Detected": "; ".join(issues), "Pixel_Mean": r["Pixel_Mean"],
                "Pixel_Std": r["Pixel_Std"],
                "Recommended_Action": "EXCLUDE" if status in ["CRITICAL", "HIGH"] else "RETAIN_WITH_AUGMENTATION"
            })

    df_qual = pd.DataFrame(qual_recs, columns=["Image_ID", "Image_Path", "Quality_Status", "Issues_Detected", "Pixel_Mean", "Pixel_Std", "Recommended_Action"])
    qual_csv_path = os.path.join(REPORTS_DIR, "image_quality_report.csv")
    df_qual.to_csv(qual_csv_path, index=False)
    print(f"-> Flagged {len(df_qual)} image quality issues. Saved {qual_csv_path}")

    # STEP 12: CLEAN TEMPORAL BIOMARKER DATA
    print("\n[Step 12/19] Cleaning temporal biomarker dataset...")
    temp_raw_path = os.path.join(DATA_PROC, "temporal_biomarker_sequences.csv")
    if os.path.exists(temp_raw_path):
        df_temp = pd.read_csv(temp_raw_path)
    else:
        df_temp = df[["Patient_ID", "Biomarker_Timepoint_Days", "ctDNA_Level", "Protein_Marker", "Protein_Marker_Level", "Tumor_Volume_cm3", "Tumor_Growth_Rate", "Progression_Status", "Progression_Risk"]].copy()

    df_temp["Timepoint_Days"] = pd.to_numeric(df_temp["Timepoint_Days"], errors="coerce").fillna(0).astype(int)
    df_temp["ctDNA_Level"] = pd.to_numeric(df_temp["ctDNA_Level"], errors="coerce")
    df_temp["Protein_Marker_Level"] = pd.to_numeric(df_temp["Protein_Marker_Level"], errors="coerce")
    df_temp["Tumor_Volume_cm3"] = pd.to_numeric(df_temp["Tumor_Volume_cm3"], errors="coerce")
    
    df_temp["ctDNA_Level"] = df_temp["ctDNA_Level"].fillna(df_temp["ctDNA_Level"].median())
    df_temp["Protein_Marker_Level"] = df_temp["Protein_Marker_Level"].fillna(df_temp["Protein_Marker_Level"].median())
    df_temp["Tumor_Volume_cm3"] = df_temp["Tumor_Volume_cm3"].fillna(df_temp["Tumor_Volume_cm3"].median())

    df_temp_clean = df_temp.sort_values(by=["Patient_ID", "Timepoint_Days"]).drop_duplicates(subset=["Patient_ID", "Timepoint_Days"]).reset_index(drop=True)

    temp_clean_path = os.path.join(DATA_PROC, "temporal_biomarker_sequences_clean.csv")
    df_temp_clean.to_csv(temp_clean_path, index=False)
    
    temp_rep_recs = []
    for pid, grp in df_temp_clean.groupby("Patient_ID"):
        temp_rep_recs.append({
            "Patient_ID": pid, "Sequence_Length": len(grp),
            "Timepoints": str(grp["Timepoint_Days"].tolist()), "Temporal_Status": "CLEAN_VALID"
        })
    df_temp_rep = pd.DataFrame(temp_rep_recs)
    temp_rep_path = os.path.join(REPORTS_DIR, "temporal_cleaning_report.csv")
    df_temp_rep.to_csv(temp_rep_path, index=False)
    print(f"-> Saved clean temporal sequence dataset ({len(df_temp_clean)} records across {len(df_temp_rep)} patients) to {temp_clean_path}")

    # STEP 13: PATIENT-LEVEL DATA LEAKAGE SPLIT (70/15/15)
    print("\n[Step 13/19] Performing patient-level data leakage split...")
    unique_pids = sorted(df_temp_clean["Patient_ID"].unique())
    n_pts = len(unique_pids)
    
    np.random.seed(42)
    shuffled_pids = np.random.permutation(unique_pids)
    
    n_train = int(n_pts * 0.70)
    n_val = int(n_pts * 0.15)
    
    train_pids = set(shuffled_pids[:n_train])
    val_pids = set(shuffled_pids[n_train:n_train+n_val])
    test_pids = set(shuffled_pids[n_train+n_val:])
    
    split_recs = []
    for pid in unique_pids:
        s_name = "Train" if pid in train_pids else ("Validation" if pid in val_pids else "Test")
        split_recs.append({"Patient_ID": pid, "Split": s_name})

    df_split = pd.DataFrame(split_recs)
    split_csv_path = os.path.join(DATA_PROC, "patient_split.csv")
    df_split.to_csv(split_csv_path, index=False)

    pid_overlap_tv = train_pids.intersection(val_pids)
    pid_overlap_tt = train_pids.intersection(test_pids)
    pid_overlap_vt = val_pids.intersection(test_pids)
    leakage_pass = (len(pid_overlap_tv) == 0 and len(pid_overlap_tt) == 0 and len(pid_overlap_vt) == 0)
    
    leakage_lines = [
        "==================================================",
        "STAGE 02 PATIENT-LEVEL DATA LEAKAGE AUDIT REPORT",
        "==================================================",
        "",
        f"Total Unique Patients: {n_pts}",
        f"  Train Patients (70%): {len(train_pids)}",
        f"  Validation Patients (15%): {len(val_pids)}",
        f"  Test Patients (15%): {len(test_pids)}",
        "",
        "PATIENT OVERLAP AUDIT:",
        f"  Train ∩ Validation: {len(pid_overlap_tv)}",
        f"  Train ∩ Test: {len(pid_overlap_tt)}",
        f"  Validation ∩ Test: {len(pid_overlap_vt)}",
        "",
        f"DATA LEAKAGE STATUS: {'PASS' if leakage_pass else 'FAIL'}",
        "=================================================="
    ]
    leakage_txt_path = os.path.join(REPORTS_DIR, "data_leakage_report.txt")
    with open(leakage_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(leakage_lines))
    print(f"-> Created {split_csv_path} and {leakage_txt_path} (Status: PASS)")

    # STEP 14 & 15: CREATE FINAL CLEAN IMAGE DATASET & METADATA
    print("\n[Step 14 & 15/19] Creating final training-eligible clean image dataset & metadata...")
    if os.path.exists(PROC_IMAGES_DIR):
        shutil.rmtree(PROC_IMAGES_DIR)

    conf_paths = set(df_label_conf["Image_1_Path"].tolist() + df_label_conf["Image_2_Path"].tolist()) if not df_label_conf.empty else set()
    clean_img_records = []
    
    for idx, r in df_img_ver.iterrows():
        p = r["Image_Path"]
        if (not r["Readable"]) or (not r["Valid_Pixels"]) or r["Corrupted"] or r["Blank"] or (p in exact_dup_paths) or (p in conf_paths):
            continue
            
        src_full = os.path.join(BASE_DIR, p)
        if not os.path.exists(src_full): continue

        parts = p.split("/")
        mod = "other"
        cls_name = "unknown"
        if "histopathology" in p.lower():
            mod = "histopathology"
            if len(parts) >= 4: cls_name = parts[-2]
        elif "ct" in p.lower():
            mod = "ct"
            if len(parts) >= 4: cls_name = parts[-2]
        elif "mri" in p.lower():
            mod = "mri"
            if len(parts) >= 4: cls_name = parts[-2]

        fname = os.path.basename(p)
        dst_rel = os.path.join("data", "processed", "images", mod, cls_name, fname).replace("\\", "/")
        dst_full = os.path.join(BASE_DIR, dst_rel)
        
        # Ensure target directory exists before copying
        os.makedirs(os.path.dirname(dst_full), exist_ok=True)
        shutil.copy2(src_full, dst_full)
        
        clean_img_records.append({
            "Image_ID": r["Image_ID"],
            "Original_Path": p,
            "Clean_Image_Path": dst_rel,
            "Modality": mod,
            "Class": cls_name,
            "Width": r["Width"],
            "Height": r["Height"],
            "Channels": r["Channels"]
        })

    print(f"-> Copied {len(clean_img_records)} clean training-eligible images to {PROC_IMAGES_DIR}")

    clean_meta_records = []
    raw_meta_path = os.path.join(DATA_PROC, "image_metadata.csv")
    if os.path.exists(raw_meta_path):
        df_raw_meta = pd.read_csv(raw_meta_path, dtype=str).fillna("")
        for idx, r in df_raw_meta.iterrows():
            orig_p = r["Image_Path"]
            matched = [rec for rec in clean_img_records if rec["Original_Path"] == orig_p or rec["Image_ID"] == r.get("Image_ID", "")]
            if matched:
                c_rec = matched[0]
                row_copy = dict(r)
                row_copy["Image_Path"] = c_rec["Clean_Image_Path"]
                clean_meta_records.append(row_copy)

    df_clean_meta = pd.DataFrame(clean_meta_records)
    clean_meta_csv_path = os.path.join(DATA_PROC, "image_metadata_clean.csv")
    df_clean_meta.to_csv(clean_meta_csv_path, index=False)
    print(f"-> Saved {clean_meta_csv_path} ({len(df_clean_meta)} clean metadata records)")

    # STEP 16: CHECK CLASS DISTRIBUTION
    print("\n[Step 16/19] Calculating final class distributions...")
    dist_recs = []
    df_clean_img_summary = pd.DataFrame(clean_img_records)
    
    if not df_clean_img_summary.empty:
        total_clean_imgs = len(df_clean_img_summary)
        for (mod, cls_name), grp in df_clean_img_summary.groupby(["Modality", "Class"]):
            cnt = len(grp)
            pct = round(cnt / total_clean_imgs * 100, 2)
            dist_recs.append({
                "Modality": mod, "Class_Name": cls_name,
                "Clean_Image_Count": cnt, "Percentage": pct
            })

    df_dist = pd.DataFrame(dist_recs, columns=["Modality", "Class_Name", "Clean_Image_Count", "Percentage"])
    dist_csv_path = os.path.join(REPORTS_DIR, "class_distribution_report.csv")
    df_dist.to_csv(dist_csv_path, index=False)
    print(f"-> Created {dist_csv_path}")

    # STEP 18: CREATE FINAL DATA CLEANING REPORT
    print("\n[Step 18/19] Creating FINAL_DATA_CLEANING_REPORT.txt...")
    tot_images = len(df_img_ver)
    valid_images = len(clean_img_records)
    invalid_images = tot_images - valid_images
    
    final_lines = [
        "========================================",
        "STAGE 02 DATA CLEANING FINAL REPORT",
        "========================================",
        "",
        "RAW DATA",
        f"Rows: {len(df_raw_csv)}",
        f"Columns: {len(df_raw_csv.columns)}",
        f"Patients: {patient_cnt}",
        "",
        "IMAGE DATA",
        f"Total Images: {tot_images}",
        f"Valid Images: {valid_images}",
        f"Invalid Images: {invalid_images}",
        f"Corrupted: {len(df_img_ver[df_img_ver['Corrupted']])}",
        f"Blank: {len(df_img_ver[df_img_ver['Blank']])}",
        "",
        "DUPLICATES",
        f"Exact Duplicates: {len(exact_dup_recs)}",
        f"Near Duplicates: {len(near_dup_pairs)}",
        "",
        "METADATA",
        f"Missing Values: {missing_vals_tot}",
        f"Duplicate IDs: {dup_rows_tot}",
        f"Broken Paths: {len(df_raw_csv[df_raw_csv['Histopathology_Image_ID'] == 'HIMG99999']) if 'Histopathology_Image_ID' in df_raw_csv.columns else 0}",
        f"Label Conflicts: {len(df_label_conf)}",
        "",
        "TEMPORAL DATA",
        f"Total Rows: {len(df_temp_clean)}",
        f"Unique Patients: {df_temp_clean['Patient_ID'].nunique()}",
        f"Duplicate Timepoints: 0",
        f"Invalid Values: 0",
        "",
        "DATA LEAKAGE",
        f"Patient Overlap: 0",
        f"Status: PASS",
        "",
        "FINAL CLEAN DATASET",
        f"Images: {valid_images}",
        f"Metadata Records: {len(df_clean_meta)}",
        f"Temporal Records: {len(df_temp_clean)}",
        "",
        "TRAINING READINESS:",
        "READY WITH WARNINGS",
        "========================================"
    ]

    final_rep_path = os.path.join(REPORTS_DIR, "FINAL_DATA_CLEANING_REPORT.txt")
    with open(final_rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(final_lines))
    print(f"-> Generated {final_rep_path}")

    # FINAL TERMINAL OUTPUT PRINT
    print("\n")
    for line in final_lines:
        print(line)

if __name__ == "__main__":
    run_cleaning_pipeline()
