# Personalized Precision Medicine for Oncology Treatment Optimization

## Stage 02 — Deep Learning Data Engineering Package

> [!IMPORTANT]
> **ACADEMIC SYNTHETIC DATASET DISCLAIMER**
> This repository contains a **SYNTHETIC / MOCK** oncology dataset engineered specifically for academic project development, technical demonstration, and viva presentation. No real patient health information (PHI) or actual clinical diagnostic images were used. Do not claim that synthetic data represents real clinical patient data.

---

## 1. Project Overview & Stage 02 Objectives

Stage 02 focuses on **Deep Learning Data Engineering** to support multi-modal neural network architectures. The data pipeline integrates unstructured medical imagery with longitudinal molecular biomarkers to enable downstream modeling by the Deep Learning team:

1. **Histopathology Image Classification (CNN):** 5-class tissue classification (*Benign, Malignant, Atypical, Necrotic, Inflammatory*).
2. **CT Radiologic Image Analysis:** 3-class cross-sectional tissue assessment (*Normal, Tumor, Progression*).
3. **MRI Multi-Sequence Scan Analysis:** 5 sequence representations (*T1, T2, FLAIR, DWI, ADC*).
4. **Temporal Biomarker Sequence Modeling (LSTM / Transformer):** Longitudinal ctDNA and serum protein marker trajectories across 5 clinical timepoints (*Day 0, Day 14, Day 28, Day 56, Day 84*).
5. **Tumor Progression Prediction:** Multi-task risk stratification (*Progression Risk: Low / Moderate / High*; *Progression Status: Progressed / Stable / Not Progressed*).

---

## 2. Directory Structure

```text
stage2_dl/
│
├── data/
│   ├── raw/
│   │   ├── dl_raw_1000.csv                    # Raw dataset (1,000 records, 35 columns, intentional flaws)
│   │   ├── histopathology_images/             # 350 synthetic H&E microscopy PNG images (5 classes)
│   │   │   ├── benign/
│   │   │   ├── malignant/
│   │   │   ├── atypical/
│   │   │   ├── necrotic/
│   │   │   └── inflammatory/
│   │   ├── ct_images/                         # 200 synthetic CT scan PNG images (3 classes)
│   │   │   ├── normal/
│   │   │   ├── tumor/
│   │   │   └── progression/
│   │   └── mri_images/                        # 200 synthetic MRI scan PNG images (5 sequences)
│   │       ├── T1/
│   │       ├── T2/
│   │       ├── FLAIR/
│   │       ├── DWI/
│   │       └── ADC/
│   │
│   └── processed/
│       ├── dl_cleaned.csv                     # 980 master cleaned encounter records
│       ├── image_metadata.csv                 # 346 verified image records for CNN team
│       └── temporal_biomarker_sequences.csv   # 980 temporal sequence records for LSTM/Transformer team
│
├── src/
│   ├── create_dataset.py                      # Raw synthetic dataset & image generator script
│   ├── validate_data.py                      # Data validation & physical disk audit engine
│   ├── clean_data.py                         # Data cleaning & specialized export pipeline
│   ├── generate_docx_report.py                # Word document report generator script
│   └── generate_notebook.py                   # Jupyter notebook generator script
│
├── reports/
│   ├── 01_raw_data_profile.txt                # Raw data profile & synthetic image summary
│   ├── 02_validation_report.txt               # Data quality audit report
│   ├── 03_cleaning_report.txt                 # Cleaning transformation report
│   └── STAGE2_DATA_ENGINEER_REPORT.docx       # Publication-grade Word documentation report
│
├── notebooks/
│   └── stage2_data_exploration.ipynb          # Exploratory data analysis notebook
│
└── README.md                                  # Complete Stage 02 documentation & guide
```

---

## 3. Dataset Summary

| Asset Category | Record / File Count | Details / Specifications |
| :--- | :--- | :--- |
| **Raw Tabular Dataset** | 1,000 rows | `dl_raw_1000.csv` (35 columns, intentional flaws) |
| **Histopathology Images** | 350 files | 256x256 RGB PNG microscopy images (70 per class across 5 classes) |
| **CT Radiologic Images** | 200 files | 256x256 Grayscale PNG slice representations (*Normal, Tumor, Progression*) |
| **MRI Sequence Images** | 200 files | 256x256 Grayscale PNG representations (*T1, T2, FLAIR, DWI, ADC*) |
| **Total Synthetic Images** | **750 files** | Generated on disk under `data/raw/` |
| **Cleaned Tabular Master**| 980 rows | `dl_cleaned.csv` (20 exact duplicate rows purged) |
| **Image Metadata Handoff**| 346 records | `image_metadata.csv` (Verified disk paths & validated labels) |
| **Temporal Sequence Handoff**| 980 records | `temporal_biomarker_sequences.csv` (Sorted by Patient_ID & timepoints) |

---

## 4. Raw Data Flaws & Cleaning Pipeline

The raw dataset intentionally incorporates realistic clinical data defects:
* **Duplicates:** 20 duplicate encounter rows.
* **Categorical Inconsistencies:** Mixed casing (`male`/`M`/`Female`, `stage i`/`IV`, `MALIGNANT`/`benign`).
* **Contaminated Numerics:** Embedded unit text (`25.4 cm3`, `12.8 copies/mL`, `120 slices`, `28 days`, `95%`).
* **Irregular Date Formats:** Mixed dates (`MM/DD/YYYY`, `DD-MM-YYYY`, `YYYY/MM/DD`).
* **Outlier / Invalid Values:** Negative age (`-5`), invalid age (`145`), negative tumor volume (`-15.2 cm3`).
* **Broken Image References:** Non-existent file path references (`HIMG99999.png`, `CT_missing.png`).
* **Unsorted Sequences:** Timepoint days unordered for longitudinal sequence models.

### Execution Instructions

To execute the end-to-end data engineering pipeline:

```bash
# 1. Generate Raw Data & Synthetic Images
python src/create_dataset.py

# 2. Run Data Quality & Image Disk Validation Audit
python src/validate_data.py

# 3. Run Cleaning Engine & Export Processed Datasets
python src/clean_data.py

# 4. Generate Publication-Grade Word Report
python src/generate_docx_report.py

# 5. Generate Exploration Jupyter Notebook
python src/generate_notebook.py
```

---

## 5. Deep Learning Team Handoff Protocol

### 1. CNN / Computer Vision Team
* **Handoff Files:** `data/processed/image_metadata.csv` and image directories (`data/raw/histopathology_images/`, `ct_images/`, `mri_images/`).
* **Input Features:** `Image_Path`, `Cancer_Type`, `Cancer_Stage`, `Tissue_Type`, `Tumor_Grade`.
* **Primary Target:** `Label` (*Benign, Malignant, Atypical, Necrotic, Inflammatory*).
* **Recommended Pipeline:** Use PyTorch `DataLoader` with `ImageFolder` or custom Dataset reading `Image_Path`. Apply transformations: Resize (224x224), ImageNet normalization, Random Flip, Color Jitter.

### 2. LSTM / Transformer Sequence Modeling Team
* **Handoff File:** `data/processed/temporal_biomarker_sequences.csv`.
* **Input Features:** `ctDNA_Level`, `Protein_Marker_Level`, `Tumor_Volume_cm3`, `Tumor_Growth_Rate`.
* **Primary Targets:** `Progression_Risk` (*Low, Moderate, High*) & `Progression_Status` (*Progressed, Stable, Not Progressed*).
* **Recommended Pipeline:** Group sequences by `Patient_ID`, shape into 3D tensors `(Batch_Size, Timepoints, Features)`, apply `MinMaxScaler` fitted exclusively on training set.

---

## 6. Data Leakage Prevention Strategy

> [!WARNING]
> **PATIENT-LEVEL SPLITTING IS MANDATORY**
> Each patient has multiple longitudinal timepoint encounters and multiple image scans. **Do NOT use naive random row-level train/test splits.**

1. **Patient-Grouped Splitting:** Use `GroupKFold` or `GroupShuffleSplit` on `Patient_ID`. This guarantees that 100% of a patient's images, timepoints, and clinical records reside strictly in either Train, Validation, or Test split.
2. **Scaler Isolation:** Calculate feature normalization parameters (mean, std, min, max) strictly on the Training split. Apply fitted scalers to Validation and Test splits without re-computing statistics.

---

## 7. Viva Questions Quick Reference

1. **Why is raw data kept untouched?** For auditability, reproducibility, and lineage tracing.
2. **Why separate image metadata from temporal sequence datasets?** To decouple Computer Vision model streaming from Recurrent/Transformer sequence model pipelines.
3. **How are broken images detected?** By verifying physical file existence on disk via `os.path.exists()` and attempting image header decoding.
4. **How do you prevent data leakage?** By performing patient-level grouped splits (`GroupKFold` on `Patient_ID`).

---

## 8. Simple Language Explanation

> *"I prepared and organized the synthetic oncology image and biomarker data required for the Deep Learning stage. I validated image paths and labels, handled metadata quality issues, prepared temporal biomarker sequences, kept raw data untouched, and produced structured datasets that the CNN and LSTM/Transformer teams can directly use."*
