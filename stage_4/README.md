# Precision Medicine for Oncology Treatment Optimization

AI-powered multi-modal precision oncology system designed to predict patient treatment risk, toxicity outcomes, and disease progression using classical machine learning, deep learning, and longitudinal biomarker tracking.

---

## Project Overview

Modern oncology requires tailoring therapies to the individual patient. This repository implements an end-to-end data science and deep learning architecture for oncology treatment optimization:

1. **Stage 1 — Classical Machine Learning Pipeline:**
   - Multiclass toxicity risk prediction (`Low`, `Moderate`, `High` risk).
   - Trained on clinical cohorts with strict patient-level separation (zero data leakage).
   - Models evaluated: **XGBoost, Random Forest, LightGBM, Decision Trees, and Logistic Regression**.
   - Comprehensive model evaluation suite: ROC/AUC curves, PR curves, calibration curves, clinical misclassification cost analysis, and cross-validation stability.

2. **Stage 2 — Multi-Modal Deep Learning & Longitudinal Sequences (`stage2_dl/`):**
   - **Model 1 (CNN - ResNet18):** Transfer learning on histopathology microscopy images (Benign, Malignant, Atypical, Necrotic, Inflammatory tissue classification).
   - **Model 2 (BiLSTM):** Bidirectional LSTM modeling longitudinal biomarker trajectories across discrete follow-up timepoints (Day 0, Day 14, Day 28, Day 56, Day 84) tracking ctDNA dynamics and tumor volume kinetics to detect progression risk.
   - **Model 3 (Tabular Transformer / FT-Transformer):** Clinical encounter deep learning with multi-head self-attention for complex feature interactions.
   - Rigorous patient-level stratified splitting (70% Train / 15% Val / 15% Test) with guaranteed zero patient overlap.

3. **Risk Monitor & Deployment Guardrails (`risk_monitor/`):**
   - Proactive release and canary phase monitoring for CI/CD deployments.
   - Telemetry health analysis, automated rollback rules, and audit logging.

---

## Directory Structure

```
.
├── .github/                       # GitHub Actions workflows and CI configurations
├── data/                          # Cleaned clinical datasets & raw samples
│   ├── cleaned_data.csv
│   └── data.csv
├── models/                        # Serialized classical ML models (.pkl)
│   ├── xgboost.pkl
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── logistic_regression.pkl
│   └── risk_model.pkl
├── reports/                       # Stage 1 evaluation reports, figures, & metric logs
│   ├── evaluation_report.md       # Comprehensive evaluation report
│   ├── evaluation_metrics.json    # JSON metrics summary
│   ├── model_comparison.csv       # Comparison table across 5 ML models
│   └── *.png                      # ROC, PR, calibration, and cost analysis curves
├── risk_monitor/                  # Deployment risk monitoring engine & tests
│   ├── backend/
│   ├── models/
│   └── scripts/
├── stage2_dl/                     # Stage 2 Deep Learning Multi-Modal Framework
│   ├── data/                      # Processed images, pixel conversions, sequences
│   │   ├── processed/             # Cleaned images, metadata, and patient splits
│   │   └── master/                # Master integrated datasets
│   ├── models/                    # Saved deep learning PyTorch weights
│   ├── notebooks/                 # Executed EDA Jupyter Notebooks
│   │   └── stage2_data_exploration.ipynb
│   ├── reports/                   # Stage 2 EDA, audit, and model comparison reports
│   │   ├── EDA_REPORT.md          # Multi-modal EDA report
│   │   ├── dl_model_comparison.csv# CNN, BiLSTM, and Transformer metrics
│   │   └── figures/               # 11 publication-grade EDA visualizations
│   ├── scripts/                   # Automated pipelines & data engineering
│   └── dl_pipeline.py             # End-to-end PyTorch DL training and evaluation
├── evaluation_report.py           # Stage 1 comprehensive evaluation script
├── ml_pipeline.py                 # Stage 1 ML training and pipeline execution
└── README.md
```

---

## Performance Summary

### Stage 1: Classical Machine Learning (Toxicity Prediction)

| Model | Accuracy | Macro Prec | Macro Recall | Macro F1 | Weighted F1 | Balanced Acc |
|---|---|---|---|---|---|---|
| **XGBoost** | **0.9178** | **0.5513** | **0.5694** | **0.5513** | **0.9182** | **0.5694** |
| Decision Tree | 0.9089 | 0.5205 | 0.5744 | 0.5421 | 0.9139 | 0.5744 |
| Random Forest | 0.9012 | 0.5115 | 0.5715 | 0.5346 | 0.9076 | 0.5715 |
| Logistic Regression | 0.7895 | 0.4367 | 0.5322 | 0.4499 | 0.8208 | 0.5322 |

*Full evaluation details available in [`reports/evaluation_report.md`](reports/evaluation_report.md).*

### Stage 2: Deep Learning Models

| Model Architecture | Modality / Data Domain | Task | Test Accuracy | Macro F1 |
|---|---|---|---|---|
| **ResNet-18 CNN** | Histopathology Microscopy Images | Tissue Subtyping | **1.0000** | **1.0000** |
| **BiLSTM** | Longitudinal Biomarker Sequences | 5-Timepoint Progression Risk | **1.0000** | **1.0000** |
| **Tabular Transformer** | Multi-Feature Clinical Encounters | Multiclass Encounter Profiling | **1.0000** | **1.0000** |

*Metrics exported to [`stage2_dl/reports/dl_model_comparison.csv`](stage2_dl/reports/dl_model_comparison.csv).*

---

## Getting Started

### Prerequisites
- Python 3.9+
- PyTorch & Torchvision
- Scikit-learn, XGBoost, LightGBM
- Pandas, NumPy, Matplotlib, Seaborn

### Running Stage 1 Evaluation
```bash
python evaluation_report.py
```

### Running Stage 2 Deep Learning Pipeline
```bash
python stage2_dl/dl_pipeline.py
```

### Running Automated Multi-Modal EDA
```bash
python stage2_dl/scripts/run_eda_pipeline.py
```

---

## License & Medical Disclaimer
This software is intended for research and educational purposes. It has not been approved for diagnostic or clinical decision-making by any regulatory authority.
