import os
import glob
import pandas as pd
import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

RAW_CSV = os.path.join(DATA_RAW, "dl_raw_1000.csv")
CLEAN_CSV = os.path.join(DATA_PROC, "dl_cleaned.csv")

BODY_PARTS_REPORT = os.path.join(REPORTS_DIR, "current_body_parts_report.csv")
MATRIX_REPORT = os.path.join(REPORTS_DIR, "image_type_body_part_matrix.csv")
SUMMARY_TXT = os.path.join(REPORTS_DIR, "CURRENT_IMAGE_DATASET_SUMMARY.txt")

def inspect_dataset():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    print("==================================================")
    print("STARTING STAGE 02 DATASET BODY PART & MODALITY AUDIT")
    print("==================================================")

    df_raw = pd.read_csv(RAW_CSV, dtype=str).fillna("") if os.path.exists(RAW_CSV) else pd.DataFrame()
    df_clean = pd.read_csv(CLEAN_CSV, dtype=str).fillna("") if os.path.exists(CLEAN_CSV) else pd.DataFrame()
    
    # 1. DISCOVER ALL PHYSICAL IMAGE FILES UNDER DATA/RAW
    raw_images = []
    for root, dirs, files in os.walk(DATA_RAW):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                raw_images.append(os.path.join(root, f))
                
    raw_images = sorted(raw_images)
    total_imgs_cnt = len(raw_images)
    total_pts_cnt = df_clean["Patient_ID"].nunique() if not df_clean.empty else df_raw["Patient_ID"].nunique()

    # Map each relative image path to patient details from cleaned dataset
    path_to_row = {}
    for idx, r in df_clean.iterrows():
        p_histo = str(r.get("Histopathology_Image_Path", "")).replace("\\", "/").strip()
        p_ct = str(r.get("CT_Scan_Path", "")).replace("\\", "/").strip()
        p_mri = str(r.get("MRI_Scan_Path", "")).replace("\\", "/").strip()
        
        if p_histo: path_to_row[p_histo] = r
        if p_ct: path_to_row[p_ct] = r
        if p_mri: path_to_row[p_mri] = r

    modality_info = {
        "Histopathology": {"count": 0, "patients": set(), "formats": set(), "dims": set(), "folder": "data/raw/histopathology_images/", "classes": set(), "body_parts": set()},
        "CT": {"count": 0, "patients": set(), "formats": set(), "dims": set(), "folder": "data/raw/ct_images/", "classes": set(), "body_parts": set()},
        "MRI": {"count": 0, "patients": set(), "formats": set(), "dims": set(), "folder": "data/raw/mri_images/", "sequences": set(), "body_parts": set()}
    }

    image_details = []

    for fpath in raw_images:
        rel_p = os.path.relpath(fpath, BASE_DIR).replace("\\", "/")
        fname = os.path.basename(fpath)
        ext = os.path.splitext(fname)[1].replace(".", "").upper()
        
        parts = rel_p.split("/")
        mod = "Other"
        cls_name = "Unknown"
        
        if "histopathology_images" in rel_p:
            mod = "Histopathology"
            if len(parts) >= 4: cls_name = parts[-2].capitalize()
            modality_info["Histopathology"]["classes"].add(cls_name)
        elif "ct_images" in rel_p:
            mod = "CT"
            if len(parts) >= 4: cls_name = parts[-2].capitalize()
            modality_info["CT"]["classes"].add(cls_name)
        elif "mri_images" in rel_p:
            mod = "MRI"
            if len(parts) >= 4: cls_name = parts[-2].upper()
            modality_info["MRI"]["sequences"].add(cls_name)

        try:
            with Image.open(fpath) as img:
                w, h = img.width, img.height
                fmt = img.format if img.format else ext
                dim_str = f"{w}x{h}"
        except Exception:
            dim_str = "Unknown"
            fmt = ext

        matched_r = path_to_row.get(rel_p)
        if matched_r is not None:
            pid = str(matched_r.get("Patient_ID", "Unknown")).strip()
            organ = str(matched_r.get("Organ_Site", "Unknown")).strip()
            ctype = str(matched_r.get("Cancer_Type", "Unknown")).strip()
        else:
            pid = "Unknown"
            organ = "Unknown"
            ctype = "Unknown"

        if organ == "" or organ == "nan": organ = "Unknown"
        if ctype == "" or ctype == "nan": ctype = "Unknown"

        modality_info[mod]["count"] += 1
        if pid and pid != "Unknown": modality_info[mod]["patients"].add(pid)
        modality_info[mod]["formats"].add(fmt)
        modality_info[mod]["dims"].add(dim_str)
        if organ and organ != "Unknown": modality_info[mod]["body_parts"].add(organ)

        image_details.append({
            "Rel_Path": rel_p,
            "Modality": mod,
            "Class_Seq": cls_name,
            "Format": fmt,
            "Dimension": dim_str,
            "Patient_ID": pid,
            "Organ_Site": organ,
            "Cancer_Type": ctype
        })

    df_details = pd.DataFrame(image_details)

    # 3. IDENTIFY BODY PARTS / ORGAN SITES REPORT
    bp_grp = df_details.groupby(["Organ_Site", "Cancer_Type", "Modality"])
    bp_rows = []
    for (organ, ctype, mod), grp in bp_grp:
        bp_rows.append({
            "Body_Part": organ,
            "Cancer_Type": ctype,
            "Image_Type": mod,
            "Number_of_Images": len(grp),
            "Number_of_Patients": grp[grp["Patient_ID"] != "Unknown"]["Patient_ID"].nunique()
        })
    df_bp_rep = pd.DataFrame(bp_rows, columns=["Body_Part", "Cancer_Type", "Image_Type", "Number_of_Images", "Number_of_Patients"])
    df_bp_rep = df_bp_rep.sort_values(by=["Body_Part", "Image_Type"]).reset_index(drop=True)
    df_bp_rep.to_csv(BODY_PARTS_REPORT, index=False)
    print(f"-> Created {BODY_PARTS_REPORT}")

    # 8. COMPLETE IMAGE MATRIX REPORT
    matrix_rows = []
    unique_organs = sorted(list(df_details["Organ_Site"].unique()))
    
    for organ in unique_organs:
        df_o = df_details[df_details["Organ_Site"] == organ]
        cnt_histo = len(df_o[df_o["Modality"] == "Histopathology"])
        cnt_ct = len(df_o[df_o["Modality"] == "CT"])
        cnt_mri = len(df_o[df_o["Modality"] == "MRI"])
        tot = len(df_o)
        matrix_rows.append({
            "Body_Part": organ,
            "Histopathology": cnt_histo,
            "CT": cnt_ct,
            "MRI": cnt_mri,
            "Total_Images": tot
        })
    df_matrix = pd.DataFrame(matrix_rows, columns=["Body_Part", "Histopathology", "CT", "MRI", "Total_Images"])
    df_matrix = df_matrix.sort_values(by="Total_Images", ascending=False).reset_index(drop=True)
    df_matrix.to_csv(MATRIX_REPORT, index=False)
    print(f"-> Created {MATRIX_REPORT}")

    # 9. HUMAN-READABLE REPORT
    report_lines = []
    report_lines.append("========================================")
    report_lines.append("CURRENT STAGE 02 IMAGE DATASET")
    report_lines.append("========================================")
    report_lines.append("")
    report_lines.append(f"TOTAL IMAGES:\n{total_imgs_cnt}")
    report_lines.append("")
    report_lines.append(f"TOTAL PATIENTS:\n{total_pts_cnt}")
    report_lines.append("")
    report_lines.append(f"IMAGE MODALITIES:\nHistopathology, CT, MRI")
    report_lines.append("")
    report_lines.append(f"BODY PARTS / ORGANS:\n{', '.join([o for o in unique_organs if o != 'Unknown'])}")
    report_lines.append("")
    report_lines.append("----------------------------------------")
    report_lines.append("HISTOPATHOLOGY")
    report_lines.append("----------------------------------------")
    report_lines.append(f"Images: {modality_info['Histopathology']['count']}")
    report_lines.append(f"Patients: {len(modality_info['Histopathology']['patients'])}")
    report_lines.append(f"Body Parts: {', '.join(sorted(list(modality_info['Histopathology']['body_parts'])))}")
    report_lines.append(f"Classes: {', '.join(sorted(list(modality_info['Histopathology']['classes'])))}")
    report_lines.append("")
    report_lines.append("----------------------------------------")
    report_lines.append("CT")
    report_lines.append("----------------------------------------")
    report_lines.append(f"Images: {modality_info['CT']['count']}")
    report_lines.append(f"Patients: {len(modality_info['CT']['patients'])}")
    report_lines.append(f"Body Parts: {', '.join(sorted(list(modality_info['CT']['body_parts'])))}")
    report_lines.append(f"Classes: {', '.join(sorted(list(modality_info['CT']['classes'])))}")
    report_lines.append("")
    report_lines.append("----------------------------------------")
    report_lines.append("MRI")
    report_lines.append("----------------------------------------")
    report_lines.append(f"Images: {modality_info['MRI']['count']}")
    report_lines.append(f"Patients: {len(modality_info['MRI']['patients'])}")
    report_lines.append(f"Body Parts: {', '.join(sorted(list(modality_info['MRI']['body_parts'])))}")
    report_lines.append(f"MRI Sequences: {', '.join(sorted(list(modality_info['MRI']['sequences'])))}")
    report_lines.append("")
    report_lines.append("----------------------------------------")
    report_lines.append("BODY PART SUMMARY")
    report_lines.append("----------------------------------------")

    for organ in unique_organs:
        df_o = df_details[df_details["Organ_Site"] == organ]
        mods_avail = sorted(list(df_o["Modality"].unique()))
        report_lines.append(f"Body Part: {organ}")
        report_lines.append(f"Images: {len(df_o)}")
        report_lines.append(f"Modalities: {', '.join(mods_avail)}")
        report_lines.append("")

    report_lines.append("========================================")

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"-> Created {SUMMARY_TXT}")

    # Print summary to console
    print("\n" + "\n".join(report_lines))

if __name__ == "__main__":
    inspect_dataset()
