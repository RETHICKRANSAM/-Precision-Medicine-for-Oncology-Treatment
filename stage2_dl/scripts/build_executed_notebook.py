"""
Builds and executes stage2_data_exploration.ipynb with embedded plots and outputs.
"""

import os
import sys
import io
import json
import base64
import traceback
import contextlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Ensure UTF-8 output encoding on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_PATH = os.path.join(BASE_DIR, "notebooks", "stage2_data_exploration.ipynb")

print("[*] Generating and executing master Stage 2 EDA Notebook...")

# Shared execution environment namespace
execution_env = {
    'np': np,
    'pd': pd,
    'plt': plt,
    'sns': sns,
    'os': os,
    'Image': Image,
    'BASE_DIR': BASE_DIR
}

cells_data = []

def add_markdown(source_text):
    cells_data.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_text.strip().split("\n")]
    })

def add_and_execute_code(code_text, cell_id=1):
    print(f"    Executing Cell {cell_id}...")
    stdout_capture = io.StringIO()
    outputs = []
    
    # Reset pyplot before cell
    plt.close('all')
    
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code_text, execution_env)
    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"[-] Cell {cell_id} execution error:\n{err_msg}")
        stdout_capture.write(f"\nError: {err_msg}")

    # Check for printed stdout
    stdout_val = stdout_capture.getvalue()
    if stdout_val:
        outputs.append({
            "output_type": "stream",
            "name": "stdout",
            "text": [line + "\n" for line in stdout_val.split("\n") if line or line == ""]
        })

    # Check if any matplotlib figures were generated
    fig_nums = plt.get_fignums()
    for fnum in fig_nums:
        fig = plt.figure(fnum)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        outputs.append({
            "output_type": "display_data",
            "data": {
                "image/png": img_b64,
                "text/plain": [f"<Figure size {fig.get_size_inches()[0]*100}x{fig.get_size_inches()[1]*100} with {len(fig.axes)} Axes>"]
            },
            "metadata": {}
        })
    plt.close('all')

    cells_data.append({
        "cell_type": "code",
        "execution_count": cell_id,
        "metadata": {},
        "outputs": outputs,
        "source": [line + "\n" for line in code_text.strip().split("\n")]
    })

# ==============================================================
# NOTEBOOK STRUCTURE & CONTENT
# ==============================================================

add_markdown("""# Stage 02 Deep Learning Multi-Modal EDA Notebook
## Personalized Precision Medicine for Oncology Treatment Optimization
**Author:** EDA Engineering Team  
**Dataset:** Academic Synthetic Oncology Multi-Modal Dataset (Unstructured Imaging + Longitudinal Biomarkers)  

---

### Objectives:
1. **Multi-Modal Asset Profiling:** Tabular clinical records, Histopathology microscopy, CT scans, MRI sequences, and longitudinal biomarker series.
2. **Computer Vision Characterization:** Class distributions, RGB stain characteristics, spatial projection maps, sharpness (Laplacian variance), and Shannon entropy.
3. **Longitudinal Trajectories:** Temporal dynamics of ctDNA, protein biomarkers, and tumor volume across clinical timepoints.
4. **Data Leakage Verification:** Zero patient ID overlap verification across Train, Validation, and Test splits.
5. **Deep Learning Handoff:** Input preprocessing, tensor shaping, and architectural recommendations.""")

# Cell 1: Setup
add_and_execute_code("""import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 1000)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

print("Libraries and visualization styling configured successfully.")
""", 1)

# Cell 2: Data Loading
add_and_execute_code("""# Robust path resolution whether run from root or notebooks directory
data_dir = None
for candidate in ['../data', 'data', 'stage2_dl/data', os.path.join(BASE_DIR, 'data')]:
    if os.path.exists(os.path.join(candidate, 'processed', 'dl_cleaned.csv')):
        data_dir = candidate
        break

if data_dir is None:
    raise FileNotFoundError("Could not locate stage2_dl data directory.")

df_clean = pd.read_csv(os.path.join(data_dir, 'processed', 'dl_cleaned.csv'))
df_temporal = pd.read_csv(os.path.join(data_dir, 'processed', 'temporal_biomarker_sequences.csv'))
df_split = pd.read_csv(os.path.join(data_dir, 'processed', 'patient_split.csv'))
df_pixel = pd.read_csv(os.path.join(data_dir, 'pixel_images', 'pixel_image_metadata.csv'))

print("=== DATASET INVENTORY LOADED ===")
print(f"Cleaned Encounters:   {df_clean.shape}")
print(f"Temporal Sequences:   {df_temporal.shape}")
print(f"Patient Splits:       {df_split.shape}")
print(f"Pixel Image Metadata: {df_pixel.shape}")
""", 2)

# Section 1: Demographics & Clinical Profile
add_markdown("""## 1. Clinical Demographics & Encounter Profile
Explore patient demographics, tumor staging, and primary cancer sites.""")

add_and_execute_code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Age Distribution
sns.histplot(df_clean['Age'], bins=20, kde=True, color='#3498db', ax=axes[0])
axes[0].set_title("Age Distribution (Mean: 55.9, Std: 16.0)", fontweight='bold')
axes[0].set_xlabel("Age (Years)")

# Cancer Stage
stage_counts = df_clean['Cancer_Stage'].value_counts()
axes[1].bar(stage_counts.index, stage_counts.values, color=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'], edgecolor='black')
axes[1].set_title("Cancer Stage Representation", fontweight='bold')
axes[1].set_ylabel("Encounters")

# Primary Organ Site
site_counts = df_clean['Organ_Site'].value_counts()
axes[2].pie(site_counts, labels=site_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=140)
axes[2].set_title("Primary Organ Site Distribution", fontweight='bold')

plt.tight_layout()
plt.show()

print("Summary of Numerical Clinical Features:")
print(df_clean[['Age', 'Tumor_Volume_cm3', 'ctDNA_Level', 'Protein_Marker_Level', 'Tumor_Growth_Rate']].describe().round(2))
""", 3)

# Section 2: Unstructured Imaging & Class Distributions
add_markdown("""## 2. Unstructured Imaging Modality Breakdown & Visual Gallery
Visual and distributional analysis across Histopathology (5 classes), CT (3 classes), and MRI (5 sequences).""")

add_and_execute_code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histopathology Classes
histo_classes = ['benign', 'malignant', 'atypical', 'necrotic', 'inflammatory']
h_counts = [len(glob.glob(os.path.join(data_dir, 'raw', 'histopathology_images', c, '*.png'))) for c in histo_classes]
axes[0].bar(histo_classes, h_counts, color='#9b59b6', edgecolor='black')
axes[0].set_title("Histopathology Classes (5-Class CNN Target)", fontweight='bold')
axes[0].tick_params(axis='x', rotation=25)

# CT Classes
ct_classes = ['normal', 'tumor', 'progression']
c_counts = [len(glob.glob(os.path.join(data_dir, 'raw', 'ct_images', c, '*.png'))) for c in ct_classes]
axes[1].bar(ct_classes, c_counts, color='#2980b9', edgecolor='black')
axes[1].set_title("CT Radiologic Tissue Classes", fontweight='bold')

# MRI Sequences
mri_seqs = ['T1', 'T2', 'FLAIR', 'DWI', 'ADC']
m_counts = [len(glob.glob(os.path.join(data_dir, 'raw', 'mri_images', s, '*.png'))) for s in mri_seqs]
axes[2].bar(mri_seqs, m_counts, color='#16a085', edgecolor='black')
axes[2].set_title("MRI Multi-Sequence Modalities", fontweight='bold')

plt.tight_layout()
plt.show()
""", 4)

# Visual Gallery Code
add_and_execute_code("""fig, axs = plt.subplots(3, 5, figsize=(16, 10))

# Row 1: Histopathology
for i, c in enumerate(histo_classes):
    imgs = glob.glob(os.path.join(data_dir, 'raw', 'histopathology_images', c, '*.png'))
    if imgs:
        im = Image.open(imgs[0])
        axs[0, i].imshow(im)
        axs[0, i].set_title(f"Histo: {c.capitalize()}", fontweight='bold')
    axs[0, i].axis('off')

# Row 2: CT
for i, c in enumerate(ct_classes):
    imgs = glob.glob(os.path.join(data_dir, 'raw', 'ct_images', c, '*.png'))
    if imgs:
        axs[1, i].imshow(Image.open(imgs[0]), cmap='gray')
        axs[1, i].set_title(f"CT: {c.capitalize()}", fontweight='bold')
    axs[1, i].axis('off')
axs[1, 3].axis('off')
axs[1, 4].axis('off')

# Row 3: MRI
for i, s in enumerate(mri_seqs):
    imgs = glob.glob(os.path.join(data_dir, 'raw', 'mri_images', s, '*.png'))
    if imgs:
        axs[2, i].imshow(Image.open(imgs[0]), cmap='gray')
        axs[2, i].set_title(f"MRI: {s}", fontweight='bold')
    axs[2, i].axis('off')

plt.suptitle("Multi-Modal Image Samples: Histopathology (Top), CT (Middle), MRI (Bottom)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
""", 5)

# Section 3: Histopathology Stain Analysis
add_markdown("""## 3. Histopathology Chromatic & H&E Stain Profiling
RGB channel decomposition representing Hematoxylin (nuclei) and Eosin (cytoplasm) optical densities.""")

add_and_execute_code("""r_vals, g_vals, b_vals = [], [], []
histo_files = glob.glob(os.path.join(data_dir, 'raw', 'histopathology_images', '**', '*.png'), recursive=True)[:40]

for f in histo_files:
    arr = np.array(Image.open(f).convert('RGB'))
    r_vals.extend(arr[:, :, 0].flatten()[::64])
    g_vals.extend(arr[:, :, 1].flatten()[::64])
    b_vals.extend(arr[:, :, 2].flatten()[::64])

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.kdeplot(r_vals, ax=axs[0], color='#e74c3c', label='Red (Eosin)', fill=True, alpha=0.3)
sns.kdeplot(g_vals, ax=axs[0], color='#27ae60', label='Green', fill=True, alpha=0.3)
sns.kdeplot(b_vals, ax=axs[0], color='#2980b9', label='Blue (Hematoxylin)', fill=True, alpha=0.3)
axs[0].set_title("RGB Channel Intensity Distribution", fontweight='bold')
axs[0].set_xlabel("Pixel Value (0-255)")
axs[0].legend()

# Scatter sample
sub_idx = np.random.choice(len(r_vals), size=min(1500, len(r_vals)), replace=False)
axs[1].scatter(np.array(r_vals)[sub_idx], np.array(b_vals)[sub_idx], color='#8e44ad', alpha=0.4, s=15)
axs[1].set_title("Red vs. Blue Color Space (H&E Stain Balance)", fontweight='bold')
axs[1].set_xlabel("Red (Eosin Component)")
axs[1].set_ylabel("Blue (Hematoxylin Component)")

plt.tight_layout()
plt.show()
""", 6)

# Section 4: CT Spatial Maps
add_markdown("""## 4. Radiologic Spatial Mean & Standard Deviation Slices
Mean and spatial variation maps identifying structural anatomy and focal tumor masses.""")

add_and_execute_code("""def compute_spatial_stats(img_paths):
    arrays = [np.array(Image.open(p).convert('L'), dtype=float) for p in img_paths[:30]]
    stack = np.stack(arrays, axis=0)
    return stack.mean(axis=0), stack.std(axis=0)

ct_norm = glob.glob(os.path.join(data_dir, 'raw', 'ct_images', 'normal', '*.png'))
ct_prog = glob.glob(os.path.join(data_dir, 'raw', 'ct_images', 'progression', '*.png'))

mean_norm, std_norm = compute_spatial_stats(ct_norm)
mean_prog, std_prog = compute_spatial_stats(ct_prog)

fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))
axs[0].imshow(mean_norm, cmap='bone'); axs[0].set_title("CT Normal: Mean"); axs[0].axis('off')
axs[1].imshow(std_norm, cmap='magma'); axs[1].set_title("CT Normal: Std Map"); axs[1].axis('off')
axs[2].imshow(mean_prog, cmap='bone'); axs[2].set_title("CT Progression: Mean"); axs[2].axis('off')
axs[3].imshow(std_prog, cmap='magma'); axs[3].set_title("CT Progression: Std Map"); axs[3].axis('off')
plt.tight_layout()
plt.show()
""", 7)

# Section 5: Temporal Sequences (LSTM/Transformer)
add_markdown("""## 5. Longitudinal Biomarker Sequences (LSTM / Transformer Targets)
Trajectories of ctDNA, tumor volume, and protein markers across 5 timepoints (Days 0, 14, 28, 56, 84).""")

add_and_execute_code("""fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# ctDNA Trajectory
sns.lineplot(data=df_temporal, x='Timepoint_Days', y='ctDNA_Level', hue='Progression_Risk',
             palette={'Low': '#27ae60', 'Moderate': '#f39c12', 'High': '#c0392b'}, marker='o', ax=axs[0])
axs[0].set_title("ctDNA Dynamics Across Timepoints", fontweight='bold')
axs[0].set_ylabel("ctDNA Level (copies/mL)")

# Tumor Volume Trajectory
sns.lineplot(data=df_temporal, x='Timepoint_Days', y='Tumor_Volume_cm3', hue='Progression_Risk',
             palette={'Low': '#27ae60', 'Moderate': '#f39c12', 'High': '#c0392b'}, marker='s', ax=axs[1])
axs[1].set_title("Tumor Volume Dynamics Across Timepoints", fontweight='bold')
axs[1].set_ylabel("Volume (cm³)")

# Protein Marker Trajectory
sns.lineplot(data=df_temporal, x='Timepoint_Days', y='Protein_Marker_Level', hue='Progression_Status',
             palette={'Not Progressed': '#27ae60', 'Stable': '#2980b9', 'Progressed': '#c0392b'}, marker='^', ax=axs[2])
axs[2].set_title("Serum Protein Marker Dynamics", fontweight='bold')
axs[2].set_ylabel("Protein Marker (ng/mL)")

plt.tight_layout()
plt.show()
""", 8)

# Section 6: Feature Correlations
add_markdown("""## 6. Multi-Modal Correlation Analysis
Correlation heatmap examining relationships between clinical biomarkers and progression risk.""")

add_and_execute_code("""num_cols = ['Age', 'CT_Slice_Count', 'Tumor_Volume_cm3', 'Biomarker_Timepoint_Days', 
            'ctDNA_Level', 'Protein_Marker_Level', 'Tumor_Growth_Rate', 'Image_Label_Confidence']
corr = df_clean[num_cols].corr()

plt.figure(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0, square=True)
plt.title("Pearson Correlation Heatmap: Clinical & Molecular Features", fontweight='bold')
plt.tight_layout()
plt.show()
""", 9)

# Section 7: Patient-Level Data Leakage
add_markdown("""## 7. Patient-Level Data Leakage Verification
Auditing patient-level grouped splits (GroupShuffleSplit on Patient_ID) to ensure zero data leakage.""")

add_and_execute_code("""train_pts = set(df_split[df_split['Split'] == 'Train']['Patient_ID'])
val_pts = set(df_split[df_split['Split'] == 'Validation']['Patient_ID'])
test_pts = set(df_split[df_split['Split'] == 'Test']['Patient_ID'])

overlap_tr_val = len(train_pts.intersection(val_pts))
overlap_tr_te = len(train_pts.intersection(test_pts))
overlap_val_te = len(val_pts.intersection(test_pts))

print(f"=== PATIENT-LEVEL LEAKAGE AUDIT ===")
print(f"Total Unique Patients: {len(df_split)}")
print(f"Train Patients:        {len(train_pts)} ({len(train_pts)/len(df_split)*100:.1f}%)")
print(f"Validation Patients:   {len(val_pts)} ({len(val_pts)/len(df_split)*100:.1f}%)")
print(f"Test Patients:         {len(test_pts)} ({len(test_pts)/len(df_split)*100:.1f}%)")
print(f"Train ∩ Validation:    {overlap_tr_val}")
print(f"Train ∩ Test:          {overlap_tr_te}")
print(f"Validation ∩ Test:     {overlap_val_te}")
print(f"STATUS: {'PASS - ZERO LEAKAGE DETECTED' if (overlap_tr_val + overlap_tr_te + overlap_val_te == 0) else 'FAIL - DATA LEAKAGE'}")
""", 10)

# Section 8: Deep Learning Handoff Summary
add_markdown("""## 8. Deep Learning Architecture Recommendations
- **CNN / ViT Team:** Use 350 Histopathology images (`data/processed/image_metadata.csv`). Resize to 224x224, apply ImageNet normalization and rotation/flip augmentations.
- **LSTM / Transformer Team:** Use 980 longitudinal rows (`temporal_biomarker_sequences.csv`). Shape into 3D tensors `(Batch, Timepoints=5, Features=4)`.
- **Loss Strategy:** Employ Focal Loss or Class-Weighted Cross-Entropy to handle the clinical imbalance in high-risk progression.""")

# Build final notebook JSON
notebook_json = {
    "cells": cells_data,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.13"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=1)

print(f"[+] Successfully wrote fully-executed Jupyter notebook: {NOTEBOOK_PATH}")
