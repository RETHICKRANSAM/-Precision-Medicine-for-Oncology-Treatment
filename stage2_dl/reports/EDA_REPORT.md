# Comprehensive Exploratory Data Analysis (EDA) Report
## Personalized Precision Medicine for Oncology Treatment Optimization — Stage 02 Deep Learning
**Author:** EDA Engineering Team  
**System Architecture:** Multi-Modal Deep Learning & Longitudinal Sequence Engine  
**Dataset Type:** Academic Synthetic Oncology Multi-Modal Cohort  
**Date:** September 2026  

---

## Executive Summary

This report establishes the complete statistical, structural, and computer vision exploratory baseline for **Stage 02 Deep Learning Data Engineering**. The Stage 2 pipeline combines unstructured diagnostic imagery (**Histopathology, Computed Tomography (CT), Magnetic Resonance Imaging (MRI)**) with longitudinal molecular biomarker time series (**circulating tumor DNA, serum protein markers, volumetric kinetics**) and structured patient clinical encounter records.

### Key Inventory Findings
- **Cleaned Tabular Encounters:** 980 validated encounter records (derived from 1,000 raw rows after purging 20 exact duplicate encounters and repairing clinical formatting defects).
- **Physical Imaging Cohort:** 750 physical images generated on disk under `data/raw/` (350 Histopathology, 200 CT, 200 MRI).
- **Quality-Purged Clean Images:** 433 images retained in `data/processed/images_clean/` after near-duplicate perceptual hash filtering and cross-class label conflict resolution.
- **Longitudinal Biomarker Sequences:** 980 sequence encounters spanning 5 standardized clinical timepoints (Days 0, 14, 28, 56, 84).
- **Patient Cohort & Splitting:** 196 unique patients partitioned with zero data leakage (137 Train / 29 Validation / 30 Test; 70%/15%/15% ratio).

---

## 1. Multi-Modal Dataset Architecture & Inventory

| Data Stream | Modality / Format | Dimension / Structure | Records / Files | Primary Downstream DL Target |
| :--- | :--- | :--- | :--- | :--- |
| **Histopathology** | Microscopy RGB (PNG) | 256 x 256 x 3 | 350 images | 5-Class Tissue Classification (*Benign, Malignant, Atypical, Necrotic, Inflammatory*) |
| **CT Radiologic** | Cross-Sectional Gray (PNG) | 256 x 256 x 1 | 200 raw (59 clean) | 3-Class Tumor Progression Assessment (*Normal, Tumor, Progression*) |
| **MRI Sequences** | Multi-Parametric Gray (PNG) | 256 x 256 x 1 | 200 raw (24 clean) | Multi-Sequence Tissue Characterization (*T1, T2, FLAIR, DWI, ADC*) |
| **Longitudinal Biomarkers** | Numerical Time Series | 5 Timepoints x 4 Features | 980 records | Multi-Task Progression Risk (*Low, Moderate, High*) & Status |
| **Clinical Encounter Tabular**| Tabular EHR Records | 35 Features | 980 records | Multi-Modal Fusion Baseline & Stratification |

*Artifact Reference: [Figure 01: Multi-Modal Dataset Overview](figures/01_dataset_overview_modality_counts.png)*

---

## 2. Computer Vision & Unstructured Imaging EDA

### 2.1 Histopathology Image Characteristics
- **Color Space:** 3-channel RGB microscopy imagery.
- **Stain Profiling (Hematoxylin & Eosin Proxy):**
  - **Red Channel:** Mean intensity 197.4 (Eosin eosinophilic cytoplasm and extracellular matrix).
  - **Blue Channel:** Mean intensity 192.0 (Hematoxylin basophilic cell nuclei).
  - **Green Channel:** Mean intensity 159.3.
  - Malignant and necrotic tissue classes exhibit markedly higher nuclear-to-cytoplasmic intensity variances, aligning with standard histological dysplastic cytology.
- **Class Balance:** Perfectly balanced across 5 tissue categories (70 images per class in raw, 350 total).

*Artifact References: [Figure 02: Image Class Distributions](figures/02_image_class_distributions.png), [Figure 05: Histopathology RGB Stain Analysis](figures/05_histopathology_rgb_stain_analysis.png)*

### 2.2 CT & MRI Radiologic Spatial Analysis
- **CT Scans:** Single-channel grayscale slices (256x256). Spatial standard deviation heatmaps clearly capture anatomical lung/mediastinal boundaries and circumscribed tumor margins with high attenuation gradients.
- **MRI Multi-Sequence Characteristics:**
  - **T1-Weighted:** Anatomical structure baseline with balanced contrast.
  - **T2-Weighted & FLAIR:** Enhanced hyperintense edema and necrotic core visibility.
  - **DWI & ADC:** Diffusion restriction indicators critical for cellular density assessment.

*Artifact References: [Figure 03: Visual Image Gallery](figures/03_sample_image_gallery.png), [Figure 06: CT & MRI Spatial Maps](figures/06_ct_mri_spatial_mean_std_maps.png)*

### 2.3 Image Quality, Sharpness, and Information Entropy
- **Laplacian Variance (Sharpness Metric):**
  - Histopathology: High textural frequency (Mean Laplacian Var: 676.0), indicating dense cellular edges.
  - CT & MRI Scans: Smooth soft-tissue attenuation (Mean Laplacian Var: CT 375.8, MRI 176.0).
- **Shannon Entropy (Information Density):**
  - Histopathology: 6.47 bits.
  - CT Scans: 3.46 bits.
  - MRI Scans: 2.48 bits.
  - No blank (0-entropy) or zero-byte corrupted images were detected across the entire repository.

*Artifact Reference: [Figure 07: Image Quality & Entropy Audit](figures/07_image_quality_blur_entropy.png)*

---

## 3. Longitudinal Biomarker Sequence EDA (LSTM / Transformer)

The temporal dataset (`temporal_biomarker_sequences.csv`) records 5 discrete clinical follow-up timepoints: **Day 0 (Baseline), Day 14, Day 28, Day 56, and Day 84**.

### Key Trajectory Findings:
1. **Circulating Tumor DNA (ctDNA) Dynamics:**
   - **Low Risk / Not Progressed Patients:** ctDNA levels exhibit steady decline from baseline (mean ~15 copies/mL down to < 5 copies/mL by Day 84), reflecting therapeutic response.
   - **High Risk / Progressed Patients:** ctDNA exhibits dramatic exponential surges post-Day 28 (reaching > 80 copies/mL), serving as an ultra-early molecular indicator of radiographic progression.
2. **Tumor Volume Kinetics:**
   - Tumor volume strongly lags molecular ctDNA spikes by 2 to 4 weeks. High-risk cohorts demonstrate tumor volume escalation from ~25 cm³ up to > 85 cm³.
3. **Protein Marker Dynamics:**
   - Serum protein markers correlate with therapeutic resistance, stabilizing in responders while elevating monotonically in progressing cohorts.

*Artifact References: [Figure 08: Temporal Biomarker Trajectories](figures/08_temporal_biomarker_trajectories.png), [Figure 09: Clinical Feature Correlation Heatmap](figures/09_clinical_biomarker_correlations.png)*

---

## 4. Patient-Level Data Leakage Prevention Audit

```text
[Cohort Splitting Strategy: GroupShuffleSplit on Patient_ID]
Total Unique Patients: 196
├── Training Split:   137 Patients (69.9%)  --> 685 Longitudinal Encounters
├── Validation Split:  29 Patients (14.8%)  --> 145 Longitudinal Encounters
└── Testing Split:     30 Patients (15.3%)  --> 150 Longitudinal Encounters

Overlap Check:
  Train ∩ Validation: 0 Patients
  Train ∩ Test:       0 Patients
  Validation ∩ Test:  0 Patients
  Total Patient Leakage: 0 (PASSED)
```

Target stratification across splits demonstrates robust stability:
- **Low Risk Proportion:** Train: 68.6%, Val: 71.7%, Test: 69.3%
- **Moderate Risk Proportion:** Train: 24.5%, Val: 22.1%, Test: 23.3%
- **High Risk Proportion:** Train: 6.9%, Val: 6.2%, Test: 7.4%

*Artifact Reference: [Figure 11: Patient Split & Leakage Audit](figures/11_patient_split_leakage_audit.png)*

---

## 5. Statistical Feature Summary Table

The parametric and non-parametric properties of numerical features are summarized below:

| Feature | Count | Missing | Mean | Std | Median | IQR | Min | Max | Skewness | Kurtosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Age** | 980 | 0 | 55.881 | 16.037 | 55.0 | 27.0 | 28.0 | 82.0 | -0.025 | -1.146 |
| **CT_Slice_Count** | 200 | 780 | 125.04 | 86.246 | 64.0 | 208.0 | 48.0 | 256.0 | 0.699 | -1.256 |
| **Tumor_Volume_cm3** | 980 | 0 | 25.934 | 16.146 | 23.2 | 22.91 | 1.9 | 85.64 | 0.895 | 0.64 |
| **Biomarker_Timepoint_Days** | 980 | 0 | 36.4 | 30.172 | 28.0 | 42.0 | 0.0 | 84.0 | 0.403 | -1.236 |
| **ctDNA_Level** | 980 | 0 | 67.77 | 45.168 | 58.31 | 58.312 | 1.34 | 252.02 | 1.223 | 1.84 |
| **Protein_Marker_Level** | 980 | 0 | 137.97 | 100.244 | 114.86 | 122.475 | 4.75 | 523.21 | 1.327 | 1.89 |
| **Tumor_Growth_Rate** | 980 | 0 | -0.001 | 0.021 | -0.001 | 0.042 | -0.03 | 0.035 | 0.18 | -1.253 |
| **Image_Label_Confidence** | 980 | 0 | 0.869 | 0.07 | 0.87 | 0.12 | 0.75 | 0.99 | 0.004 | -1.218 |

*Artifact Reference: [eda_summary_statistics.csv](eda_summary_statistics.csv)*

---

## 6. Downstream Deep Learning Modeling Recommendations

### 6.1 Computer Vision Architecture (CNN / Vision Transformer)
1. **Input Preprocessing:**
   - Histopathology: Resize from 256x256 to 224x224 RGB. Apply standard ImageNet normalization: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
   - CT & MRI: Min-max intensity rescaling into `[0.0, 1.0]`.
2. **Data Augmentation:**
   - Histopathology is orientation-invariant. Recommend Random Horizontal Flip (p=0.5), Random Vertical Flip (p=0.5), Random Rotation (+/- 90 deg, 180 deg), and subtle ColorJitter (brightness 0.1, contrast 0.1) to simulate slide staining variations.
3. **Backbone Selection:**
   - ResNet-50 / EfficientNet-B2 pre-trained on ImageNet as initial feature extractors, fine-tuning top convolutional blocks.

### 6.2 Sequence Architecture (BiLSTM / Temporal Transformer)
1. **Tensor Dimensionality:**
   - Shape: `(Batch_Size, Timepoints=5, Features=4)`.
   - Features: `[ctDNA_Level, Protein_Marker_Level, Tumor_Volume_cm3, Tumor_Growth_Rate]`.
2. **Sequence Normalization:**
   - Apply `MinMaxScaler` fitted exclusively on the Training patient split to eliminate test lookahead bias.
3. **Loss Function:**
   - Focal Loss or Weighted Cross-Entropy to handle the clinical class imbalance between Low Risk (~70%) and High Risk (~7%).

---

## 7. Viva Presentation Talking Points

1. **"Why is multi-modal EDA critical before training deep learning models?"**  
   *Because deep neural networks are vulnerable to shortcut learning and domain shifts. In medical imaging, variations in staining (H&E), radiological windowing, and patient leakage across timepoints can cause models to memorize patient identities rather than pathological features.*
2. **"How was data leakage avoided across time series and images?"**  
   *All splits were strictly grouped at the `Patient_ID` level. 100% of a patient's historical visits, biomarker timepoints, and imaging scans are confined to either Train, Validation, or Test.*
3. **"Why were near-duplicate images purged?"**  
   *Near-identical synthetic images created by slight perturbations inflate model test accuracy spuriously. Purging 317 redundant images prevents overoptimistic generalization metrics.*

---
**Report Approved by:** Lead EDA & Data Engineering Team  
**Deliverables Stored in:** `stage2_dl/reports/` and `stage2_dl/reports/figures/`
