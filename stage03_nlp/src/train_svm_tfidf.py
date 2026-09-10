"""
Stage 03 NLP Pipeline - Model 1: Linear SVM + TF-IDF
Fits TF-IDF strictly on training data, tunes hyperparameter C on validation data,
and evaluates on validation set without touching test data.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, classification_report

from stage03_nlp.src.preprocessing import normalize_clinical_text

PRIMARY_TEXT_COL = "clinical_note"
TARGET_COL = "urgency"
CLASSES = ["Low", "Moderate", "High"]


def train_svm_pipeline(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    models_dir: str = "stage03_nlp/models",
    random_state: int = 42
) -> Tuple[LinearSVC, TfidfVectorizer, Dict[str, Any]]:
    """
    Fits TF-IDF strictly on training texts (pure clinical_note without feature tags),
    performs validation tuning for C, and saves best model and vectorizer.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Safe text normalization without any tags
    X_train_raw = [normalize_clinical_text(t) for t in df_train[PRIMARY_TEXT_COL].tolist()]
    y_train = df_train[TARGET_COL].tolist()
    
    X_val_raw = [normalize_clinical_text(t) for t in df_val[PRIMARY_TEXT_COL].tolist()]
    y_val = df_val[TARGET_COL].tolist()
    
    print("[SVM + TF-IDF] Fitting TF-IDF vectorizer strictly on training data...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_val_vec = vectorizer.transform(X_val_raw)
    
    print(f"[SVM + TF-IDF] Feature matrix shape: {X_train_vec.shape}")
    
    # Validation tuning for C
    candidate_c = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    best_c = 1.0
    best_val_macro_f1 = -1.0
    best_model = None
    
    print("[SVM + TF-IDF] Tuning C on validation data...")
    for c in candidate_c:
        clf = LinearSVC(C=c, class_weight="balanced", random_state=random_state, max_iter=2000, dual="auto")
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_val_vec)
        macro_f1 = f1_score(y_val, preds, average="macro")
        acc = accuracy_score(y_val, preds)
        print(f"  C={c:4.2f} -> Val Macro F1: {macro_f1:.4f}, Val Acc: {acc:.4f}")
        
        if macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = macro_f1
            best_c = c
            best_model = clf
            
    print(f"[SVM + TF-IDF] Selected best C={best_c} with Validation Macro F1: {best_val_macro_f1:.4f}")
    
    # Save artifacts with explicit leakage_free names as specified
    model_save_path_pkl = os.path.join(models_dir, "svm_tfidf_leakage_free.pkl")
    vec_save_path_pkl = os.path.join(models_dir, "tfidf_leakage_free.pkl")
    
    joblib.dump(best_model, model_save_path_pkl)
    joblib.dump(vectorizer, vec_save_path_pkl)
    
    # Also save backwards-compatible aliases if needed
    joblib.dump(best_model, os.path.join(models_dir, "svm_tfidf_model.joblib"))
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
    
    print(f"[SVM + TF-IDF] Saved leakage-free model to: {model_save_path_pkl}")
    print(f"[SVM + TF-IDF] Saved leakage-free vectorizer to: {vec_save_path_pkl}")
    
    tuning_summary = {
        "best_c": best_c,
        "best_val_macro_f1": float(best_val_macro_f1),
        "vocab_size": len(vectorizer.vocabulary_),
        "ngram_range": vectorizer.ngram_range
    }
    
    return best_model, vectorizer, tuning_summary


if __name__ == "__main__":
    train_df = pd.read_csv("stage03_nlp/data/train.csv")
    val_df = pd.read_csv("stage03_nlp/data/val.csv")
    train_svm_pipeline(train_df, val_df)
