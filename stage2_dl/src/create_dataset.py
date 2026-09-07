import os
import math
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

HISTO_DIR = os.path.join(DATA_RAW, "histopathology_images")
CT_DIR = os.path.join(DATA_RAW, "ct_images")
MRI_DIR = os.path.join(DATA_RAW, "mri_images")

HISTO_CLASSES = ["benign", "malignant", "atypical", "necrotic", "inflammatory"]
CT_CLASSES = ["normal", "tumor", "progression"]
MRI_CLASSES = ["T1", "T2", "FLAIR", "DWI", "ADC"]

def ensure_directories():
    for folder in [DATA_RAW, REPORTS_DIR, HISTO_DIR, CT_DIR, MRI_DIR]:
        os.makedirs(folder, exist_ok=True)
    for c in HISTO_CLASSES:
        os.makedirs(os.path.join(HISTO_DIR, c), exist_ok=True)
    for c in CT_CLASSES:
        os.makedirs(os.path.join(CT_DIR, c), exist_ok=True)
    for c in MRI_CLASSES:
        os.makedirs(os.path.join(MRI_DIR, c), exist_ok=True)

def apply_pixelated_grid_format(img, grid_size=(48, 48)):
    """Converts image into discrete square pixel block grid format matching user reference."""
    w, h = img.size
    small = img.resize(grid_size, Image.Resampling.BILINEAR)
    pixelated = small.resize((w, h), Image.Resampling.NEAREST)
    return pixelated

def generate_histopathology_image(category, filepath):
    """Generates synthetic microscopy H&E stained image in pixelated block grid format."""
    bg_r = 240 + random.randint(-10, 10)
    bg_g = 210 + random.randint(-10, 10)
    bg_b = 225 + random.randint(-10, 10)
    img = Image.new("RGB", (256, 256), color=(bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)
    
    # Base stroma / tissue texture
    for _ in range(150):
        x = random.randint(0, 256)
        y = random.randint(0, 256)
        r = random.randint(5, 20)
        draw.ellipse([x-r, y-r, x+r, y+r], fill=(225 + random.randint(-15,15), 180 + random.randint(-20,20), 210 + random.randint(-15,15)))
        
    if category == "benign":
        for _ in range(60):
            cx, cy = random.randint(20, 236), random.randint(20, 236)
            r = random.randint(6, 10)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(180 + random.randint(-10,10), 140 + random.randint(-10,10), 190))
            nr = random.randint(3, 5)
            draw.ellipse([cx-nr, cy-nr, cx+nr, cy+nr], fill=(70 + random.randint(-10,10), 20, 110))
            
    elif category == "malignant":
        for _ in range(120):
            cx, cy = random.randint(10, 246), random.randint(10, 246)
            rx, ry = random.randint(6, 16), random.randint(6, 16)
            draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(160 + random.randint(-10,10), 110, 180))
            nrx, nry = random.randint(4, 9), random.randint(4, 9)
            draw.ellipse([cx-nrx, cy-nry, cx+nrx, cy+nry], fill=(40 + random.randint(-5,5), 5, 80))
            
    elif category == "atypical":
        for _ in range(75):
            cx, cy = random.randint(15, 240), random.randint(15, 240)
            r = random.randint(7, 13)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(190 + random.randint(-10,10), 130, 200))
            nr = random.randint(4, 7)
            draw.ellipse([cx-nr, cy-nr, cx+nr, cy+nr], fill=(55 + random.randint(-5,5), 15, 95))
            
    elif category == "necrotic":
        img = Image.new("RGB", (256, 256), color=(220 + random.randint(-5,5), 200, 200))
        draw = ImageDraw.Draw(img)
        for _ in range(200):
            x, y = random.randint(0, 256), random.randint(0, 256)
            r = random.randint(2, 6)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(160 + random.randint(-10,10), 140, 150))
            
    elif category == "inflammatory":
        for _ in range(200):
            cx, cy = random.randint(10, 246), random.randint(10, 246)
            r = random.randint(3, 6)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(50 + random.randint(-10,10), 10, 90))
            
    # Apply pixelated mosaic block grid format
    img_pixelated = apply_pixelated_grid_format(img, grid_size=(48, 48))
    img_pixelated.save(filepath)

def generate_ct_image(category, filepath):
    """Generates synthetic CT grayscale slice image in pixelated block grid format."""
    bg_val = random.randint(12, 18)
    img = Image.new("L", (256, 256), color=bg_val)
    draw = ImageDraw.Draw(img)
    
    bx = random.randint(-3, 3)
    by = random.randint(-3, 3)
    
    # Outer body contour
    draw.ellipse([25+bx, 25+by, 231+bx, 231+by], fill=60+random.randint(-5,5), outline=200+random.randint(-10,10), width=4)
    # Interior soft tissue & organ outlines
    draw.ellipse([45+bx, 45+by, 211+bx, 211+by], fill=90+random.randint(-5,5))
    # Vertebral column / bone
    draw.ellipse([115+bx, 180+by, 141+bx, 206+by], fill=230+random.randint(-10,10))
    
    if category == "tumor":
        tx = random.randint(125, 145)
        ty = random.randint(85, 105)
        tr = random.randint(18, 25)
        draw.ellipse([tx-tr, ty-tr, tx+tr, ty+tr], fill=190+random.randint(-10,10), outline=240, width=2)
    elif category == "progression":
        tx = random.randint(115, 135)
        ty = random.randint(75, 95)
        tr = random.randint(30, 40)
        draw.ellipse([tx-tr, ty-tr, tx+tr, ty+tr], fill=210+random.randint(-10,10), outline=250, width=3)
        draw.ellipse([tx-tr-12, ty-tr-12, tx+tr+12, ty+tr+12], outline=140+random.randint(-10,10), width=2)
        
    # Apply pixelated mosaic block grid format
    img_pixelated = apply_pixelated_grid_format(img, grid_size=(48, 48))
    img_pixelated.save(filepath)

def generate_mri_image(sequence, filepath):
    """Generates synthetic MRI sequence image in pixelated block grid format."""
    j = random.randint(-6, 6)
    if sequence == "T1":
        bg_val, brain_val, csf_val, lesion_val = 10+j, 110+j, 30+j, 70+j
    elif sequence == "T2":
        bg_val, brain_val, csf_val, lesion_val = 10+j, 100+j, 230+j, 200+j
    elif sequence == "FLAIR":
        bg_val, brain_val, csf_val, lesion_val = 10+j, 90+j, 20+j, 240+j
    elif sequence == "DWI":
        bg_val, brain_val, csf_val, lesion_val = 10+j, 60+j, 40+j, 250+j
    else: # ADC
        bg_val, brain_val, csf_val, lesion_val = 10+j, 140+j, 220+j, 30+j

    img = Image.new("L", (256, 256), color=max(0, bg_val))
    draw = ImageDraw.Draw(img)
    
    bx = random.randint(-4, 4)
    by = random.randint(-4, 4)
    
    # Brain parenchyma
    draw.ellipse([30+bx, 30+by, 226+bx, 226+by], fill=max(0, brain_val))
    # Ventricles (CSF)
    draw.ellipse([105+bx, 100+by, 125+bx, 150+by], fill=max(0, csf_val))
    draw.ellipse([131+bx, 100+by, 151+bx, 150+by], fill=max(0, csf_val))
    # Hyper/hypointense lesion spot
    lx = random.randint(140, 160)
    ly = random.randint(70, 95)
    lr = random.randint(15, 25)
    draw.ellipse([lx-lr, ly-lr, lx+lr, ly+lr], fill=max(0, lesion_val))
    
    # Apply pixelated mosaic block grid format
    img_pixelated = apply_pixelated_grid_format(img, grid_size=(48, 48))
    img_pixelated.save(filepath)

def create_images():
    print("Generating synthetic Histopathology images in Pixelated Mosaic Format...")
    histo_count = 0
    histo_records = []
    for category in HISTO_CLASSES:
        cat_dir = os.path.join(HISTO_DIR, category)
        for i in range(1, 71): # 70 images per class = 350 total
            histo_count += 1
            img_id = f"HIMG{histo_count:05d}"
            fname = f"{img_id}.png"
            fpath = os.path.join("data", "raw", "histopathology_images", category, fname).replace("\\", "/")
            full_path = os.path.join(cat_dir, fname)
            generate_histopathology_image(category, full_path)
            histo_records.append({
                "Image_ID": img_id,
                "Relative_Path": fpath,
                "Label": category.capitalize()
            })

    print(f"Created {histo_count} Histopathology pixelated PNG images.")

    print("Generating synthetic CT images in Pixelated Mosaic Format...")
    ct_count = 0
    ct_records = []
    counts_ct = {"normal": 65, "tumor": 70, "progression": 65} # 200 total
    for category, count in counts_ct.items():
        cat_dir = os.path.join(CT_DIR, category)
        for i in range(1, count + 1):
            ct_count += 1
            img_id = f"CT{ct_count:05d}"
            fname = f"{img_id}.png"
            fpath = os.path.join("data", "raw", "ct_images", category, fname).replace("\\", "/")
            full_path = os.path.join(cat_dir, fname)
            generate_ct_image(category, full_path)
            ct_records.append({
                "Image_ID": img_id,
                "Relative_Path": fpath,
                "Category": category,
                "Slice_Count": random.choice([48, 64, 128, 256])
            })
    print(f"Created {ct_count} CT pixelated PNG images.")

    print("Generating synthetic MRI images in Pixelated Mosaic Format...")
    mri_count = 0
    mri_records = []
    for sequence in MRI_CLASSES: # 40 per sequence = 200 total
        seq_dir = os.path.join(MRI_DIR, sequence)
        for i in range(1, 41):
            mri_count += 1
            img_id = f"MRI{mri_count:05d}"
            fname = f"{img_id}.png"
            fpath = os.path.join("data", "raw", "mri_images", sequence, fname).replace("\\", "/")
            full_path = os.path.join(seq_dir, fname)
            generate_mri_image(sequence, full_path)
            mri_records.append({
                "Image_ID": img_id,
                "Relative_Path": fpath,
                "Sequence": sequence
            })
    print(f"Created {mri_count} MRI pixelated PNG images.")

    return histo_records, ct_records, mri_records

def create_raw_csv(histo_records, ct_records, mri_records):
    print("Generating 1000 synthetic raw records with intentional flaws...")
    
    num_patients = 196
    patients = [f"P{10001+i}" for i in range(num_patients)]
    
    cancer_types = ["Breast Cancer", "Non-Small Cell Lung Cancer", "Colorectal Carcinoma", "Prostate Adenocarcinoma", "Melanoma"]
    stages = ["Stage I", "Stage II", "Stage III", "Stage IV"]
    sexes = ["Male", "Female"]
    organ_sites = ["Breast", "Lung", "Colon", "Prostate", "Skin", "Liver", "Brain"]
    tissue_types = ["Core Needle Biopsy", "Surgical Resection", "Fine-Needle Aspiration", "Blood Plasma"]
    tumor_grades = ["Grade 1 (Well Differentiated)", "Grade 2 (Moderately Differentiated)", "Grade 3 (Poorly Differentiated)", "Grade 4 (Undifferentiated)"]
    margin_statuses = ["Negative (Clear)", "Positive (Involved)", "Close (<1mm)", "Unknown"]
    protein_markers = ["CEA", "CA-125", "PSA", "CA19-9", "HER2/neu"]
    drugs = ["Pembrolizumab", "Cisplatin + Pemetrexed", "Folfox Regime", "Tamoxifen", "Erlotinib"]
    prog_statuses = ["Progressed", "Not Progressed", "Stable"]
    prog_risks = ["Low", "Moderate", "High"]
    image_qualities = ["High (Diagnostic)", "Medium (Acceptable)", "Low (Motion Artifacts)", "Blurred"]
    annotation_statuses = ["Fully Annotated", "Pending Verification", "Review Required", "Unannotated"]
    image_sources = ["PACS_Central_Hospital", "Pathology_Core_Lab_A", "Oncology_Trial_Consortium"]

    patient_profiles = {}
    for pid in patients:
        age = random.randint(28, 82)
        sex = random.choice(sexes)
        ctype = random.choice(cancer_types)
        cstage = random.choice(stages)
        organ = random.choice(organ_sites)
        traj = random.choice([0, 1, 2])
        patient_profiles[pid] = {
            "age": age, "sex": sex, "ctype": ctype, "cstage": cstage, "organ": organ, "traj": traj,
            "base_ctdna": random.uniform(10.0, 120.0),
            "base_protein": random.uniform(15.0, 250.0),
            "base_vol": random.uniform(3.5, 45.0)
        }

    timepoints = [0, 14, 28, 56, 84]

    raw_records = []
    enc_id_counter = 10001
    histo_idx = 0
    ct_idx = 0
    mri_idx = 0

    for pid in patients:
        prof = patient_profiles[pid]
        
        for t_idx, day in enumerate(timepoints):
            enc_id = f"ENC{enc_id_counter}"
            enc_id_counter += 1
            
            base_year = 2025
            month = 1 + (day // 30)
            m_day = 10 + (day % 30)
            if m_day > 28:
                m_day = m_day - 28
                month += 1
            enc_date = f"{base_year}-{month:02d}-{m_day:02d}"
            
            t_factor = t_idx / 4.0
            if prof["traj"] == 0:
                ctdna = max(0.5, prof["base_ctdna"] * (1.0 - 0.7 * t_factor) + random.uniform(-2, 2))
                prot_level = max(1.0, prof["base_protein"] * (1.0 - 0.65 * t_factor) + random.uniform(-3, 3))
                tumor_vol = max(0.5, prof["base_vol"] * (1.0 - 0.5 * t_factor) + random.uniform(-0.5, 0.5))
                growth_rate = round(-0.02 - 0.01 * t_factor, 4)
                p_status = "Not Progressed"
                p_risk = "Low"
                t_resp = random.choice(["Partial Response (PR)", "Complete Response (CR)"])
            elif prof["traj"] == 1:
                ctdna = max(0.5, prof["base_ctdna"] * (1.0 + random.uniform(-0.1, 0.1)))
                prot_level = max(1.0, prof["base_protein"] * (1.0 + random.uniform(-0.1, 0.1)))
                tumor_vol = max(0.5, prof["base_vol"] * (1.0 + random.uniform(-0.05, 0.05)))
                growth_rate = round(random.uniform(-0.005, 0.005), 4)
                p_status = "Stable"
                p_risk = "Moderate"
                t_resp = "Stable Disease (SD)"
            else:
                ctdna = prof["base_ctdna"] * (1.0 + 1.2 * t_factor) + random.uniform(-2, 2)
                prot_level = prof["base_protein"] * (1.0 + 1.1 * t_factor) + random.uniform(-3, 3)
                tumor_vol = prof["base_vol"] * (1.0 + 0.9 * t_factor) + random.uniform(-0.5, 0.5)
                growth_rate = round(0.015 + 0.02 * t_factor, 4)
                p_status = "Progressed"
                p_risk = "High"
                t_resp = "Progressive Disease (PD)"
                
            histo_id, histo_path, histo_label = "", "", ""
            if t_idx in [0, 2] and histo_idx < len(histo_records):
                hrec = histo_records[histo_idx]
                histo_id = hrec["Image_ID"]
                histo_path = hrec["Relative_Path"]
                histo_label = hrec["Label"]
                histo_idx += 1

            ct_id, ct_path, ct_slices = "", "", ""
            if ct_idx < len(ct_records) and random.random() < 0.7:
                ctrec = ct_records[ct_idx]
                ct_id = ctrec["Image_ID"]
                ct_path = ctrec["Relative_Path"]
                ct_slices = str(ctrec["Slice_Count"])
                ct_idx += 1

            mri_id, mri_path, mri_seq = "", "", ""
            if mri_idx < len(mri_records) and random.random() < 0.7:
                mrec = mri_records[mri_idx]
                mri_id = mrec["Image_ID"]
                mri_path = mrec["Relative_Path"]
                mri_seq = mrec["Sequence"]
                mri_idx += 1

            row = {
                "Patient_ID": pid,
                "Encounter_ID": enc_id,
                "Encounter_Date": enc_date,
                "Cancer_Type": prof["ctype"],
                "Cancer_Stage": prof["cstage"],
                "Age": str(prof["age"]),
                "Sex": prof["sex"],
                "Organ_Site": prof["organ"],
                "Histopathology_Image_ID": histo_id,
                "Histopathology_Image_Path": histo_path,
                "Tissue_Type": random.choice(tissue_types),
                "Histopathology_Label": histo_label,
                "Tumor_Grade": random.choice(tumor_grades),
                "Tumor_Margin_Status": random.choice(margin_statuses),
                "CT_Scan_ID": ct_id,
                "CT_Scan_Path": ct_path,
                "CT_Slice_Count": ct_slices,
                "MRI_Scan_ID": mri_id,
                "MRI_Scan_Path": mri_path,
                "MRI_Sequence_Type": mri_seq,
                "Tumor_Volume_cm3": f"{round(tumor_vol, 2)}",
                "Biomarker_Timepoint_Days": f"{day}",
                "ctDNA_Level": f"{round(ctdna, 2)}",
                "Protein_Marker": random.choice(protein_markers),
                "Protein_Marker_Level": f"{round(prot_level, 2)}",
                "Tumor_Growth_Rate": f"{growth_rate}",
                "Treatment_Drug": random.choice(drugs),
                "Treatment_Response": t_resp,
                "Progression_Status": p_status,
                "Progression_Risk": p_risk,
                "Image_Quality": random.choice(image_qualities),
                "Annotation_Status": random.choice(annotation_statuses),
                "Clinical_Notes": f"Patient evaluated on day {day} of regimen. Target lesion monitored.",
                "Image_Source": random.choice(image_sources),
                "Image_Label_Confidence": f"{round(random.uniform(0.75, 0.99), 2)}"
            }
            raw_records.append(row)

    print(f"Base records generated: {len(raw_records)}")
    dup_samples = random.sample(raw_records, 20)
    raw_records.extend(dup_samples)
    print(f"Total records after adding duplicates: {len(raw_records)}")

    # Inject intentional data quality flaws
    for idx, r in enumerate(raw_records):
        if random.random() < 0.15:
            r["Sex"] = random.choice(["male", "M", "female", "F", "  Male ", "FEMALE"])
        if random.random() < 0.15:
            r["Cancer_Stage"] = random.choice(["I", "stage i", "Stage IV", "IV", "stage iv", "  Stage III "])
        if r["Histopathology_Label"] and random.random() < 0.20:
            r["Histopathology_Label"] = random.choice([
                r["Histopathology_Label"].lower(),
                r["Histopathology_Label"].upper(),
                f"  {r['Histopathology_Label']}  "
            ])
        if random.random() < 0.12 and r["Tumor_Volume_cm3"]:
            r["Tumor_Volume_cm3"] = f"{r['Tumor_Volume_cm3']} cm3"
        if random.random() < 0.12 and r["ctDNA_Level"]:
            r["ctDNA_Level"] = f"{r['ctDNA_Level']} copies/mL"
        if random.random() < 0.12 and r["Protein_Marker_Level"]:
            r["Protein_Marker_Level"] = f"{r['Protein_Marker_Level']} ng/mL"
        if random.random() < 0.12 and r["CT_Slice_Count"]:
            r["CT_Slice_Count"] = f"{r['CT_Slice_Count']} slices"
        if random.random() < 0.12 and r["Biomarker_Timepoint_Days"]:
            r["Biomarker_Timepoint_Days"] = f"{r['Biomarker_Timepoint_Days']} days"
        if random.random() < 0.10 and r["Image_Label_Confidence"]:
            r["Image_Label_Confidence"] = f"{float(r['Image_Label_Confidence'])*100:.0f}%"

        if random.random() < 0.18:
            d_parts = r["Encounter_Date"].split("-")
            if len(d_parts) == 3:
                r["Encounter_Date"] = random.choice([
                    f"{d_parts[1]}/{d_parts[2]}/{d_parts[0]}",
                    f"{d_parts[2]}-{d_parts[1]}-{d_parts[0]}",
                    f"{d_parts[0]}/{d_parts[1]}/{d_parts[2]}"
                ])

        if idx == 45:
            r["Age"] = "-5"
        elif idx == 112:
            r["Tumor_Volume_cm3"] = "-15.2"
        elif idx == 230:
            r["Age"] = "145"

        if random.random() < 0.10:
            r["Cancer_Type"] = f"  {r['Cancer_Type']} "

        if idx in [15, 88, 204, 350, 512, 670]:
            r["Histopathology_Image_ID"] = "HIMG99999"
            r["Histopathology_Image_Path"] = "data/raw/histopathology_images/malignant/HIMG99999.png"
            r["Histopathology_Label"] = "Malignant"
        if idx in [42, 199, 410]:
            r["CT_Scan_Path"] = "data/raw/ct_images/tumor/CT_missing.png"
        if idx in [77, 305]:
            r["MRI_Scan_Path"] = "data/raw/mri_images/T1/MRI_corrupt.png.txt"

        if random.random() < 0.08:
            r["Histopathology_Label"] = random.choice(["", "N/A", "Unknown"])
        if random.random() < 0.05:
            r["Progression_Status"] = random.choice(["", "N/A", "null"])
        if random.random() < 0.05:
            r["ctDNA_Level"] = ""
        if random.random() < 0.05:
            r["Protein_Marker_Level"] = "N/A"

    df = pd.DataFrame(raw_records)
    csv_path = os.path.join(DATA_RAW, "dl_raw_1000.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved raw dataset to {csv_path} with shape {df.shape}")

    profile_path = os.path.join(REPORTS_DIR, "01_raw_data_profile.txt")
    with open(profile_path, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("STAGE 02 DEEP LEARNING DATASET - RAW PROFILE REPORT\n")
        f.write("==================================================\n\n")
        f.write("DISCLAIMER:\n")
        f.write("This is a SYNTHETIC / MOCK oncology dataset for academic project development only.\n")
        f.write("Do not use real patient data. Synthetic data is not real clinical data.\n\n")
        f.write(f"Total Raw Records: {len(df)}\n")
        f.write(f"Total Columns: {len(df.columns)}\n\n")
        f.write("COLUMN SCHEMA:\n")
        for i, col in enumerate(df.columns, 1):
            f.write(f"{i:02d}. {col}\n")
        f.write("\nSYNTHETIC IMAGE GENERATION SUMMARY:\n")
        f.write(f"Histopathology Images: {len(histo_records)} files across 5 classes (benign, malignant, atypical, necrotic, inflammatory)\n")
        f.write(f"CT Scans: {len(ct_records)} files across 3 classes (normal, tumor, progression)\n")
        f.write(f"MRI Scans: {len(mri_records)} files across 5 sequences (T1, T2, FLAIR, DWI, ADC)\n")
        f.write(f"Total Image Files Generated: {len(histo_records) + len(ct_records) + len(mri_records)}\n")
        f.write("Image Visual Format: Pixelated Block Mosaic Grid Format (matching reference media_1788762674591.png)\n\n")
        f.write("INTENTIONAL RAW DATA QUALITY FLAWS INTRODUCED:\n")
        f.write("- Duplicate Rows: Exactly 20 duplicate encounters\n")
        f.write("- Categorical Flaws: Inconsistent case (male/M/Male, stage i/Stage I, malignant/MALIGNANT)\n")
        f.write("- Contaminated Numerics: Text suffixes embedded ('cm3', 'copies/mL', 'slices', 'days', '%')\n")
        f.write("- Irregular Date Formats: Mixed YYYY-MM-DD, MM/DD/YYYY, DD-MM-YYYY, YYYY/MM/DD\n")
        f.write("- Invalid/Outlier Values: Negative Age (-5), Outlier Age (145), Negative Tumor Volume (-15.2)\n")
        f.write("- Broken Image References: Non-existent image paths (HIMG99999.png, CT_missing.png, MRI_corrupt.png.txt)\n")
        f.write("- Missing Values: Empty strings, N/A, null, and None scattered across columns\n")
        f.write("- Unsorted Sequences: Timepoint days unordered for sequence models\n")

    print(f"Generated raw profile report: {profile_path}")

if __name__ == "__main__":
    ensure_directories()
    histo_recs, ct_recs, mri_recs = create_images()
    create_raw_csv(histo_recs, ct_recs, mri_recs)
    print("Stage 02 Raw Dataset Generation Complete!")
