"""
Stage 03 NLP Pipeline - Production Inference Module
Provides a unified inference interface for Clinical Urgency Classification.
Includes clinical prototype safety disclaimers.
"""

import os
import joblib
import json
import torch
import numpy as np
from typing import List, Dict, Any, Union

from stage03_nlp.src.preprocessing import ClinicalVocab, ID2LABEL, LABEL2ID, MAX_SEQ_LEN, normalize_clinical_text
from stage03_nlp.src.train_bilstm import ClinicalBiLSTM

SAFETY_DISCLAIMER = (
    "[RESEARCH PROTOTYPE NOTICE] This model is for research and experimental validation only. "
    "Predictions must NOT be used for autonomous clinical decision-making or direct patient care."
)


class ClinicalUrgencyPredictor:
    """Unified predictor supporting leakage-free SVM, BiLSTM, and ClinicalBERT models."""
    def __init__(self, model_type: str = "svm", models_dir: str = "stage03_nlp/models"):
        self.model_type = model_type.lower()
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vectorizer = None
        self.vocab = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        if self.model_type in ["svm", "svm_tfidf"]:
            # Prefer leakage-free models
            model_path = os.path.join(self.models_dir, "svm_tfidf_leakage_free.pkl")
            vec_path = os.path.join(self.models_dir, "tfidf_leakage_free.pkl")
            if not os.path.exists(model_path) or not os.path.exists(vec_path):
                model_path = os.path.join(self.models_dir, "svm_tfidf_model.joblib")
                vec_path = os.path.join(self.models_dir, "tfidf_vectorizer.joblib")
            if not os.path.exists(model_path) or not os.path.exists(vec_path):
                raise FileNotFoundError(f"SVM leakage-free model or vectorizer not found in {self.models_dir}")
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vec_path)
            print(f"[PREDICTOR] Loaded Leakage-Free SVM + TF-IDF model from {model_path}.")
            
        elif self.model_type == "bilstm":
            # Prefer leakage-free models
            model_path = os.path.join(self.models_dir, "bilstm_leakage_free.pt")
            vocab_path = os.path.join(self.models_dir, "bilstm_vocab_leakage_free.json")
            if not os.path.exists(model_path) or not os.path.exists(vocab_path):
                model_path = os.path.join(self.models_dir, "bilstm_model.pt")
                vocab_path = os.path.join(self.models_dir, "bilstm_vocab.json")
            if not os.path.exists(model_path) or not os.path.exists(vocab_path):
                raise FileNotFoundError(f"BiLSTM leakage-free checkpoint or vocab not found in {self.models_dir}")
            self.vocab = ClinicalVocab.load(vocab_path)
            self.model = ClinicalBiLSTM(vocab_size=len(self.vocab)).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            self.model.eval()
            print(f"[PREDICTOR] Loaded Leakage-Free PyTorch BiLSTM model from {model_path}.")
            
        elif self.model_type in ["clinicalbert", "bert"]:
            # Prefer leakage-free models
            bert_dir = os.path.join(self.models_dir, "clinicalbert_leakage_free")
            if not os.path.exists(bert_dir):
                bert_dir = os.path.join(self.models_dir, "clinicalbert_model")
            if not os.path.exists(bert_dir):
                raise FileNotFoundError(f"ClinicalBERT leakage-free directory not found at {bert_dir}")
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(bert_dir).to(self.device)
            self.model.eval()
            print(f"[PREDICTOR] Loaded Leakage-Free ClinicalBERT model from {bert_dir}.")
            
        else:
            raise ValueError(f"Unknown model_type '{self.model_type}'. Choose 'svm', 'bilstm', or 'clinicalbert'.")

    def predict(self, texts: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """Predicts urgency for input clinical note(s) using normalized text."""
        if isinstance(texts, str):
            raw_texts = [texts]
        else:
            raw_texts = list(texts)
            
        clean_texts = [normalize_clinical_text(t) for t in raw_texts]
            
        results = []
        if self.model_type in ["svm", "svm_tfidf"]:
            vecs = self.vectorizer.transform(clean_texts)
            preds = self.model.predict(vecs)
            decision = self.model.decision_function(vecs)
            if len(decision.shape) == 1:
                decision = np.vstack([-decision, decision]).T
            # Softmax on decision scores for probability approximation
            exp_d = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            probs = exp_d / exp_d.sum(axis=1, keepdims=True)
            
            # Map classes
            classes = [str(c) for c in self.model.classes_]
            for raw_t, clean_t, p, pr in zip(raw_texts, clean_texts, preds, probs):
                prob_dict = {cls: float(round(float(pr[idx]), 4)) for idx, cls in enumerate(classes)}
                results.append({
                    "clinical_text": raw_t,
                    "normalized_text": clean_t,
                    "predicted_urgency": str(p),
                    "probabilities": prob_dict,
                    "model_used": "SVM_TFIDF",
                    "disclaimer": SAFETY_DISCLAIMER
                })
                
        elif self.model_type == "bilstm":
            with torch.no_grad():
                encoded_list = [self.vocab.encode(t, max_len=MAX_SEQ_LEN) for t in clean_texts]
                x = torch.tensor(encoded_list, dtype=torch.long).to(self.device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                for raw_t, clean_t, p, pr in zip(raw_texts, clean_texts, preds, probs):
                    results.append({
                        "clinical_text": raw_t,
                        "normalized_text": clean_t,
                        "predicted_urgency": ID2LABEL[p],
                        "probabilities": {ID2LABEL[i]: float(round(pr[i], 4)) for i in range(3)},
                        "model_used": "BiLSTM",
                        "disclaimer": SAFETY_DISCLAIMER
                    })
                    
        elif self.model_type in ["clinicalbert", "bert"]:
            with torch.no_grad():
                for raw_t, clean_t in zip(raw_texts, clean_texts):
                    inputs = self.tokenizer(clean_t, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
                    if probs.ndim == 0:
                        probs = np.array([probs])
                    pred_id = int(np.argmax(probs))
                    results.append({
                        "clinical_text": raw_t,
                        "normalized_text": clean_t,
                        "predicted_urgency": ID2LABEL[pred_id],
                        "probabilities": {ID2LABEL[i]: float(round(float(probs[i]), 4)) for i in range(3)},
                        "model_used": "ClinicalBERT",
                        "disclaimer": SAFETY_DISCLAIMER
                    })
                    
        return results


if __name__ == "__main__":
    sample_text = "oncology f/u pt says loss of appetite genomic panel tp53 mutation tx cisplatin 200 mg/day ae rash"
    try:
        predictor = ClinicalUrgencyPredictor("svm")
        res = predictor.predict(sample_text)
        print("Sample Prediction Result:", json.dumps(res, indent=2))
    except Exception as e:
        print("Model not trained yet:", e)
