# Comprehensive Model Evaluation & Audit Report
**Precision Medicine for Oncology Treatment Optimization**  
**System Architecture:** Multi-Modal Precision Oncology Engine (Stage 1 ML + Stage 2 DL)  
**Evaluation Date:** September 2026  
**Auditor:** Machine Learning & Deep Learning Validation Engineering Team  

---

## Executive Summary

This report delivers a complete, scientifically rigorous evaluation and audit across both the **Stage 1 Classical Machine Learning pipeline** (toxicity risk prediction) and the **Stage 2 Multi-Modal Deep Learning pipeline** (histopathology image subtyping, longitudinal biomarker sequence modeling, and clinical encounter profiling).

### Primary Conclusions:
1. **Machine Learning Imbalance**: In Stage 1, high raw Accuracy (**91.78%**) and Weighted F1 (**0.9214**) are driven by severe class imbalance (86.77% `Low` risk). Evaluating with **Macro F1 (0.5513)** and **Balanced Accuracy (57.78%)** provides the true, unbiased indicator of multi-class capability.
2. **Best ML Model**: **XGBoost** is the highest-performing model based primarily on **Macro F1 (0.5513 on test, 0.6912 ± 0.1508 on 5-fold CV)**.
3. **Deep Learning 100% Performance**: The perfect test scores across all three Stage 2 DL models are **not caused by test lookahead or patient leakage** (verified 0 patient overlap). Instead, they are the direct mathematical consequence of **synthetic dataset generation rules**:
   - For **BiLSTM & Tabular Transformer**, the feature `Tumor_Growth_Rate` was generated using completely disjoint, non-overlapping intervals across the risk classes.
   - For **ResNet-18 CNN**, histopathology microscopy images were procedurally generated using deterministic morphological and stain intensity rules per class.
4. **Leakage & Patient Overlap**: All splits across both Stage 1 and Stage 2 have **0 patient overlap** ($\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$). Preprocessing was fitted strictly on training splits.

---

## 1. Stage 1: Classical Machine Learning Pipeline Evaluation

### 1.1 Dataset & Cohort Definition
- **Data Source:** [`data/cleaned_data.csv`](file:///c:/Users/rethi/OneDrive/เอกสาร/Desktop/DS%20team%20pro/data/cleaned_data.csv) (3,893 total patient records)
- **Target Variable:** `Toxicity_Risk` (`Low`, `Moderate`, `High`)
- **Cohort Split:** Strict patient-level `GroupShuffleSplit` (80% Train: 3,114 patients / 20% Test: 779 patients).
- **Patient Overlap:** **0 patients** (Verified CLEAN).

### 1.2 Class Imbalance Breakdown

| Class Label | Cohort Count | Cohort % | Train Count ($N=3,114$) | Test Count ($N=779$) |
| :--- | :---: | :---: | :---: | :---: |
| **Low** | 3,378 | 86.77% | 2,694 | 684 |
| **Moderate** | 511 | 13.13% | 417 | 94 |
| **High** | 4 | 0.10% | 3 | 1 |
| **Total** | 3,893 | 100.00% | 3,114 | 779 |

> [!IMPORTANT]
> **Why Accuracy and Weighted F1 Mislead**:  
> Because 86.77% of samples belong to the `Low` class, a naive majority-class classifier would achieve ~87% accuracy. Weighted F1 is dominated by the `Low` class weight ($\frac{684}{779} \approx 0.88$). Conversely, **Macro F1 treats all three classes equally**:
> $$\text{Macro F1} = \frac{\text{F1}_{\text{Low}} + \text{F1}_{\text{Moderate}} + \text{F1}_{\text{High}}}{3}$$
> Because the `High` class has only 1 test sample (3 train samples), models struggle to identify it with high recall, yielding $\text{F1}_{\text{High}} = 0.00$. Therefore, Macro F1 tops out at ~0.55 despite >91% accuracy.

### 1.3 ML Model Performance Comparison

Evaluated on the isolated 779-patient test set and 5-fold Stratified Cross-Validation on training data:

| Model | Test Accuracy | CV Accuracy Mean | CV Accuracy Std | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 ★ | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** ★ | **0.9178** | **0.9319** | **0.0058** | **0.5778** | **0.5321** | **0.5778** | **0.5513** | **0.9214** |
| **Decision Tree** | 0.9089 | 0.9207 | 0.0055 | 0.5744 | 0.5205 | 0.5744 | 0.5421 | 0.9139 |
| **Random Forest** | 0.9012 | 0.9294 | 0.0033 | 0.5715 | 0.5115 | 0.5715 | 0.5346 | 0.9076 |
| **SVM** | 0.8408 | 0.8613 | 0.0070 | 0.5241 | 0.4549 | 0.5241 | 0.4749 | 0.8576 |
| **Logistic Regression** | 0.7895 | 0.8006 | 0.0109 | 0.5322 | 0.4367 | 0.5322 | 0.4499 | 0.8208 |

*Artifact Reference: [`reports/ml_model_comparison.csv`](file:///c:/Users/rethi/OneDrive/เอกสาร/Desktop/DS%20team%20pro/reports/ml_model_comparison.csv)*

### 1.4 Best Model In-Depth Profile: XGBoost
- **Primary Metric (Macro F1):** **0.5513** (Highest among all 5 models).
- **CV Stability:** $0.6912 \pm 0.1508$ Macro F1 across 5 stratified folds.
- **Classification Report (Held-out Test Set, $N=779$):**
  - `Low`: Precision = 0.97, Recall = 0.94, F1 = 0.95 (Support = 684)
  - `Moderate`: Precision = 0.62, Recall = 0.80, F1 = 0.70 (Support = 94)
  - `High`: Precision = 0.00, Recall = 0.00, F1 = 0.00 (Support = 1)
- **Confusion Matrix:**
  $$\begin{pmatrix} 640 & 44 & 0 \\ 19 & 75 & 0 \\ 0 & 1 & 0 \end{pmatrix}$$
  *(Rows: Actual [Low, Moderate, High]; Columns: Predicted [Low, Moderate, High])*
- **Key Predictors:** `Cancer_Stage` (15.2%), `Hemoglobin` (11.6%), `Systolic_BP` (11.5%), `Age` (6.6%), `Temperature` (4.1%).

---

## 2. Stage 2: Multi-Modal Deep Learning Pipeline Evaluation

### 2.1 Modality Separation Principle
The deep learning models operate across distinct data modalities and solve different clinical prediction tasks. They are reported separately and **must not be directly compared as competing architectures**:

| Model Architecture | Input Modality & Structure | Clinical Task | Cohort Size | Test Samples |
| :--- | :--- | :--- | :---: | :---: |
| **ResNet-18 CNN** | Histopathology Microscopy RGB Images (224x224x3) | 5-Class Tissue Subtyping | 346 images (175 patients) | 49 images (25 patients) |
| **BiLSTM** | Longitudinal Biomarkers (5 Timepoints x 4 Features) | 3-Class Disease Progression Risk | 980 encounters (196 patients) | 30 sequences (30 patients) |
| **Tabular Transformer** | Multi-Parametric Clinical Records (16 Features) | 3-Class Encounter Risk Profiling | 980 encounters (196 patients) | 150 encounters (30 patients) |

### 2.2 Deep Learning Performance Metrics

| Model | Dataset Domain | Accuracy | Macro Precision | Macro Recall | Macro F1 ★ | Weighted F1 | Balanced Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN (ResNet-18)** | Histopathology Microscopy Images (346 samples) | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **BiLSTM** | Longitudinal Biomarker Sequences (196 patients) | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Tabular Transformer**| Clinical Encounters (980 encounters) | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

*Artifact Reference: [`reports/dl_model_comparison.csv`](file:///c:/Users/rethi/OneDrive/เอกสาร/Desktop/DS%20team%20pro/reports/dl_model_comparison.csv)*

---

## 3. Scientific Investigation of the 100% DL Performance

As mandated by scientific validation guidelines, the occurrence of 100% test performance was rigorously investigated to distinguish between genuine generalization, procedural artifacts, and data leakage.

### 3.1 Patient Overlap & Test Set Isolation (VERIFIED CLEAN)
- The authoritative split file (`stage2_dl/data/processed/patient_split.csv`) partitions the 196 patients into 137 Train (70%), 29 Validation (15%), and 30 Test (15%).
- **Patient Overlap:**
  - $\text{Train} \cap \text{Val} = 0$
  - $\text{Train} \cap \text{Test} = 0$
  - $\text{Val} \cap \text{Test} = 0$
- Test sets were held completely untouched until final model inference.

### 3.2 Root Cause 1: Disjoint Interval Leakage in `Tumor_Growth_Rate` (BiLSTM & Transformer)
In [`stage2_dl/src/create_dataset.py`](file:///c:/Users/rethi/OneDrive/เอกสาร/Desktop/DS%20team%20pro/stage2_dl/src/create_dataset.py) (lines 288–312), the simulation engine assigned `Tumor_Growth_Rate` as a direct function of the synthetic progression trajectory:
- **Low Risk:** `growth_rate = round(-0.02 - 0.01 * t_factor, 4)` $\rightarrow$ strictly confined to $[-0.030, -0.020]$.
- **Moderate Risk:** `growth_rate = round(random.uniform(-0.005, 0.005), 4)` $\rightarrow$ strictly confined to $[-0.005, +0.005]$.
- **High Risk:** `growth_rate = round(0.015 + 0.02 * t_factor, 4)` $\rightarrow$ strictly confined to $[+0.015, +0.035]$.

Because these mathematical intervals do not overlap, `Tumor_Growth_Rate` alone provides a 100% separable decision boundary across all timepoints. Any neural network receiving this feature learns a trivial linear threshold and converges to 1.0000 Macro F1.

### 3.3 Root Cause 2: Procedural Geometric Generation (CNN)
In [`stage2_dl/src/create_dataset.py`](file:///c:/Users/rethi/OneDrive/เอกสาร/Desktop/DS%20team%20pro/stage2_dl/src/create_dataset.py) (lines 50–96), the histopathology microscopy images were synthesized using fixed morphological rules:
- `benign`: 60 small circular ellipses with uniform purple nuclei.
- `malignant`: 120 pleomorphic ellipses with larger radii.
- `atypical`: 75 intermediate ellipses.
- `necrotic`: Light background with 200 tiny debris points.
- `inflammatory`: 200 dense purple inflammatory cell dots.

Because each class has distinct, non-overlapping cell counts and color distributions, ResNet-18 easily learns orthogonal spatial filters, yielding 100% test accuracy on the 49 held-out test images.

---

## 4. Summary of Saved Artifacts

### 4.1 Saved Model Weights
- **Classical ML Models (`models/`):**
  - `models/xgboost.pkl` (Best overall ML model, with LabelEncoder)
  - `models/random_forest.pkl`
  - `models/decision_tree.pkl`
  - `models/svm.pkl`
  - `models/logistic_regression.pkl`
  - `models/risk_model.pkl` (Production checkpoint of best model)
- **Deep Learning Checkpoints (`stage2_dl/models/`):**
  - `stage2_dl/models/cnn_model.pt` (ResNet-18 PyTorch state dict, class names, mappings)
  - `stage2_dl/models/bilstm_model.pt` (BiLSTM weights, sequence scaler, feature cols)
  - `stage2_dl/models/tabular_transformer.pt` (FT-Transformer weights, tabular scaler, categorical mappings)

### 4.2 Saved Reports & Visualizations
- [`reports/ml_model_comparison.csv`](file:///c:/Users/rethi/OneDrive/เอกสาร/Desktop/DS%20team%20pro/reports/ml_model_comparison.csv)
- [`reports/dl_model_comparison.csv`](file:///c:/Users/rethi/OneDrive/เอกสาร/Desktop/DS%20team%20pro/reports/dl_model_comparison.csv)
- [`reports/data_leakage_audit.md`](file:///c:/Users/rethi/OneDrive/เอกสาร/Desktop/DS%20team%20pro/reports/data_leakage_audit.md)
- [`reports/model_evaluation_report.md`](file:///c:/Users/rethi/OneDrive/เอกสาร/Desktop/DS%20team%20pro/reports/model_evaluation_report.md)
- Confusion matrix and curve figures in `reports/` and `stage2_dl/reports/figures/`.

---

## 5. Limitations & Regulatory Disclaimer

> [!CAUTION]
> **Research & Academic Prototype Notice**:  
> 1. **High Toxicity Sample Scarcity**: The Stage 1 dataset contains only 4 High-risk toxicity events out of 3,893 patients (0.10%). Predictions for High-risk toxicity cannot be considered statistically reliable until prospective cohorts with higher High-risk representation are collected.
> 2. **Synthetic Boundaries**: The Stage 2 datasets were created via academic simulation algorithms with non-overlapping distributions. Real clinical cohorts contain substantial biological heterogeneity, measurement noise, and class overlap that will result in lower test performance in real-world deployment.
> 3. **Non-Diagnostic Status**: This software is strictly for research and technical demonstration. It has **NOT** been cleared or approved by any regulatory authority (e.g., FDA, CE-IVD) for clinical diagnostic, prognostic, or therapeutic decision-making.
