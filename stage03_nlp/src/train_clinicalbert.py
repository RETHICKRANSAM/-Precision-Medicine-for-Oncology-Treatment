"""
Stage 03 NLP Pipeline - Model 3: Bio_ClinicalBERT Classifier
Fine-tunes Bio_ClinicalBERT with a sequence classification head on cleaned_clinical_note.
Uses AdamW, class-weighted loss, and early stopping on Validation Macro F1.
"""

import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn.metrics import f1_score, accuracy_score
from typing import Tuple, Dict, Any, List

from stage03_nlp.src.preprocessing import normalize_clinical_text

LABEL2ID = {"Low": 0, "Moderate": 1, "High": 2}
ID2LABEL = {0: "Low", 1: "Moderate", 2: "High"}
MAX_SEQ_LEN = 48
NUM_LABELS = 3
PRIMARY_TEXT_COL = "clinical_note"
TARGET_COL = "urgency"


class ClinicalBertDataset(Dataset):
    """Dataset for Transformer tokenization."""
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = MAX_SEQ_LEN):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_clinicalbert_pipeline(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    pretrained_path: str = "stage03_nlp/models/pretrained_bio_clinicalbert",
    output_dir: str = "stage03_nlp/models/clinicalbert_leakage_free",
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 3e-5,
    patience: int = 2,
    seed: int = 42
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Fine-tunes Bio_ClinicalBERT on training split, monitors Validation Macro F1.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(6)
    print(f"[ClinicalBERT] Training device: {device} (threads={torch.get_num_threads()})")
    
    # 1. Load Pretrained Tokenizer and Model
    print(f"[ClinicalBERT] Loading pretrained model from: {pretrained_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    
    import transformers
    transformers.logging.set_verbosity_error()
    
    config = AutoConfig.from_pretrained(
        pretrained_path,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_path,
        config=config
    ).to(device)
    
    # Efficient transfer learning on CPU: freeze lower representations, fine-tune top layers + classifier
    if device.type == "cpu":
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        for layer in model.bert.encoder.layer[:10]:
            for param in layer.parameters():
                param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[ClinicalBERT] CPU Acceleration: Fine-tuning top transformer layers & head ({trainable:,}/{total:,} params).", flush=True)
        
    print(f"[ClinicalBERT] Model weights loaded successfully. Classification head initialized for {NUM_LABELS} classes.", flush=True)
    
    # 2. Datasets & Loaders (strictly normalized clinical_note, no tags)
    train_texts = [normalize_clinical_text(t) for t in df_train[PRIMARY_TEXT_COL].tolist()]
    val_texts = [normalize_clinical_text(t) for t in df_val[PRIMARY_TEXT_COL].tolist()]
    
    train_labels = [LABEL2ID[lbl] for lbl in df_train[TARGET_COL].tolist()]
    val_labels = [LABEL2ID[lbl] for lbl in df_val[TARGET_COL].tolist()]
    
    train_dataset = ClinicalBertDataset(train_texts, train_labels, tokenizer, max_len=MAX_SEQ_LEN)
    val_dataset = ClinicalBertDataset(val_texts, val_labels, tokenizer, max_len=MAX_SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
    
    # 3. Class weighted loss
    class_counts = [train_labels.count(i) for i in range(NUM_LABELS)]
    total_samples = len(train_labels)
    class_weights = [total_samples / (NUM_LABELS * max(c, 1)) for c in class_counts]
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    # 4. Optimizer & LR schedule
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    
    best_val_macro_f1 = -1.0
    best_model_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_acc": []}
    
    print(f"[ClinicalBERT] Beginning fine-tuning for {epochs} epochs (Early stopping patience={patience})...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * len(targets)
            
            if (step + 1) % 25 == 0 or (step + 1) == len(train_loader):
                print(f"  Epoch {epoch}/{epochs} | Step {step+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}", flush=True)
                
        avg_train_loss = total_train_loss / len(train_dataset)
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, targets)
                total_val_loss += loss.item() * len(targets)
                
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(targets.cpu().numpy())
                
        avg_val_loss = total_val_loss / len(val_dataset)
        val_macro_f1 = f1_score(val_targets, val_preds, average="macro")
        val_acc = accuracy_score(val_targets, val_preds)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_acc"].append(val_acc)
        
        print(f"[ClinicalBERT] Epoch {epoch:02d} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro F1: {val_macro_f1:.4f} | Val Acc: {val_acc:.4f}", flush=True)
        
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[ClinicalBERT] Early stopping triggered at epoch {epoch}!", flush=True)
                break
                
    # Load best state and save
    model.load_state_dict(best_model_state)
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Also ensure saved to explicit clinicalbert_leakage_free directory
    lf_dir = os.path.join(os.path.dirname(output_dir), "clinicalbert_leakage_free")
    if os.path.abspath(lf_dir) != os.path.abspath(output_dir):
        os.makedirs(lf_dir, exist_ok=True)
        model.save_pretrained(lf_dir)
        tokenizer.save_pretrained(lf_dir)
    print(f"[ClinicalBERT] Saved best fine-tuned ClinicalBERT model & tokenizer to: {output_dir}", flush=True)
    
    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_val_macro_f1),
        "history": history
    }
    return model, tokenizer, summary


def predict_clinicalbert(model, tokenizer, texts: List[str], batch_size: int = 64) -> List[str]:
    """Generates predictions for a list of clinical texts."""
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    
    clean_texts = [normalize_clinical_text(t) for t in texts]
    dummy_labels = [0] * len(clean_texts)
    dataset = ClinicalBertDataset(clean_texts, dummy_labels, tokenizer, max_len=MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend([ID2LABEL[p] for p in preds])
            
    return all_preds
