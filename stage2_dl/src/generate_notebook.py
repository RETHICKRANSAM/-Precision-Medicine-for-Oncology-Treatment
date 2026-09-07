import os
import nbformat as nbf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
NOTEBOOK_PATH = os.path.join(NOTEBOOKS_DIR, "stage2_data_exploration.ipynb")

def create_notebook():
    os.makedirs(NOTEBOOKS_DIR, exist_ok=True)
    nb = nbf.v4.new_notebook()

    cells = []

    # Markdown Title Cell
    cells.append(nbf.v4.new_markdown_cell("""# Stage 02 Deep Learning Data Exploration Notebook
## Personalized Precision Medicine for Oncology Treatment Optimization

**Academic Disclaimer:**
> **IMPORTANT:** This notebook analyzes a **SYNTHETIC / MOCK** oncology dataset created for academic project development, demonstration, and viva presentation. No real patient data or clinical imagery is used.

---

### Objectives
1. Profile Raw vs. Cleaned Tabular Datasets (`dl_raw_1000.csv` vs `dl_cleaned.csv`).
2. Inspect Image Metadata (`image_metadata.csv`) and sample synthetic Histopathology, CT, and MRI images.
3. Analyze Temporal Biomarker Sequences (`temporal_biomarker_sequences.csv`) for LSTM/Transformer sequence modeling.
4. Verify Patient-Level Data Leakage Prevention strategy.
"""))

    # Code Cell 1: Environment Setup
    cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Display configuration
pd.set_option('display.max_columns', 35)
plt.style.use('ggplot')
print("Environment and libraries loaded successfully.")
"""))

    # Code Cell 2: Load Datasets
    cells.append(nbf.v4.new_code_cell("""raw_csv = '../data/raw/dl_raw_1000.csv'
cleaned_csv = '../data/processed/dl_cleaned.csv'
image_meta_csv = '../data/processed/image_metadata.csv'
temporal_csv = '../data/processed/temporal_biomarker_sequences.csv'

df_raw = pd.read_csv(raw_csv, dtype=str)
df_clean = pd.read_csv(cleaned_csv)
df_img = pd.read_csv(image_meta_csv)
df_temp = pd.read_csv(temporal_csv)

print(f"Raw Records: {df_raw.shape}")
print(f"Cleaned Master Records: {df_clean.shape}")
print(f"Image Metadata Records: {df_img.shape}")
print(f"Temporal Sequence Records: {df_temp.shape}")
"""))

    # Markdown Section: Raw vs Cleaned Analysis
    cells.append(nbf.v4.new_markdown_cell("""### 1. Data Cleaning Verification & Quality Profile"""))

    cells.append(nbf.v4.new_code_cell("""print("Cleaned Dataset Head:")
display(df_clean[['Patient_ID', 'Encounter_Date', 'Age', 'Sex', 'Cancer_Stage', 'Tumor_Volume_cm3', 'ctDNA_Level']].head())

print("\nCategorical Distributions (Cleaned):")
print("Sex:", df_clean['Sex'].value_counts().to_dict())
print("Cancer Stage:", df_clean['Cancer_Stage'].value_counts().to_dict())
print("Histopathology Label:", df_clean['Histopathology_Label'].value_counts().to_dict())
"""))

    # Markdown Section: Image Visualizations
    cells.append(nbf.v4.new_markdown_cell("""### 2. Synthetic Image Inspection (Histopathology, CT, MRI)"""))

    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Histopathology sample
histo_sample = df_img.iloc[0]['Image_Path']
full_histo_p = os.path.join('..', histo_sample)
if os.path.exists(full_histo_p):
    img_h = Image.open(full_histo_p)
    axes[0].imshow(img_h)
    axes[0].set_title(f"Histopathology: {df_img.iloc[0]['Label']}")
    axes[0].axis('off')

# CT sample
ct_sample = '../data/raw/ct_images/tumor/CT00066.png'
if os.path.exists(ct_sample):
    img_c = Image.open(ct_sample)
    axes[1].imshow(img_c, cmap='gray')
    axes[1].set_title("CT Scan: Tumor Mass")
    axes[1].axis('off')

# MRI sample
mri_sample = '../data/raw/mri_images/FLAIR/MRI00081.png'
if os.path.exists(mri_sample):
    img_m = Image.open(mri_sample)
    axes[2].imshow(img_m, cmap='gray')
    axes[2].set_title("MRI: FLAIR Sequence")
    axes[2].axis('off')

plt.tight_layout()
plt.show()
"""))

    # Markdown Section: Temporal Biomarker Sequences
    cells.append(nbf.v4.new_markdown_cell("""### 3. Temporal Biomarker Trajectory Analysis"""))

    cells.append(nbf.v4.new_code_cell("""# Select 3 sample patients to plot longitudinal ctDNA trajectories
sample_patients = df_temp['Patient_ID'].unique()[:3]

plt.figure(figsize=(10, 5))
for pid in sample_patients:
    p_data = df_temp[df_temp['Patient_ID'] == pid].sort_values('Timepoint_Days')
    plt.plot(p_data['Timepoint_Days'], p_data['ctDNA_Level'], marker='o', label=f"Patient {pid} ({p_data['Progression_Risk'].iloc[0]} Risk)")

plt.title("Longitudinal ctDNA Level Trajectories Across Timepoints")
plt.xlabel("Timepoint (Days)")
plt.ylabel("ctDNA Level (copies/mL)")
plt.legend()
plt.grid(True)
plt.show()
"""))

    # Markdown Section: Summary
    cells.append(nbf.v4.new_markdown_cell("""### 4. Summary & Model Handoff Ready
- **CNN Team:** Use `image_metadata.csv` (346 records) to train Histopathology classifier.
- **LSTM/Transformer Team:** Use `temporal_biomarker_sequences.csv` (980 sequence rows) grouped by `Patient_ID`.
- **Data Leakage:** Ensure splits are grouped by `Patient_ID` (GroupKFold).
"""))

    nb['cells'] = cells

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Generated exploration notebook at {NOTEBOOK_PATH}")

if __name__ == '__main__':
    create_notebook()
