"""
Stage 03 NLP Pipeline - Model 2: PyTorch BiLSTM Classifier
Implements compact Bidirectional LSTM with early stopping based on Validation Macro F1.
"""

import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from typing import Tuple, Dict, Any, List

from stage03_nlp.src.preprocessing import ClinicalVocab, encode_labels, LABEL2ID, ID2LABEL, normalize_clinical_text

PRIMARY_TEXT_COL = "clinical_note"
TARGET_COL = "urgency"
MAX_SEQ_LEN = 64  # Accommodates full clinical notes
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
NUM_CLASSES = 3


class ClinicalNoteDataset(Dataset):
    """PyTorch Dataset for clinical notes."""
    def __init__(self, texts: List[str], labels: List[int], vocab: ClinicalVocab, max_len: int = MAX_SEQ_LEN):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        input_ids = self.vocab.encode(text, max_len=self.max_len)
        label = self.labels[idx]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class ClinicalBiLSTM(nn.Module):
    """Compact Bidirectional LSTM text classifier."""
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int = EMBEDDING_DIM, 
        hidden_dim: int = HIDDEN_DIM, 
        num_layers: int = NUM_LAYERS, 
        dropout: float = DROPOUT, 
        num_classes: int = NUM_CLASSES
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            bidirectional=True, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, (h_n, c_n) = self.lstm(embedded)
        # Concatenate forward and backward final layer states
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        feat = torch.cat((h_forward, h_backward), dim=1)
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits


def train_bilstm_pipeline(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    models_dir: str = "stage03_nlp/models",
    figures_dir: str = "stage03_nlp/reports/figures",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 4,
    seed: int = 42
) -> Tuple[ClinicalBiLSTM, ClinicalVocab, Dict[str, Any]]:
    """
    Trains BiLSTM with early stopping based on Validation Macro F1.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BiLSTM] Training using device: {device}")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Build Vocab strictly on train texts (normalized clinical_note)
    train_texts = [normalize_clinical_text(t) for t in df_train[PRIMARY_TEXT_COL].tolist()]
    val_texts = [normalize_clinical_text(t) for t in df_val[PRIMARY_TEXT_COL].tolist()]
    
    train_labels = encode_labels(df_train[TARGET_COL].tolist())
    val_labels = encode_labels(df_val[TARGET_COL].tolist())
    
    vocab = ClinicalVocab(min_freq=1)
    vocab.build_vocab(train_texts)
    vocab_save_path_lf = os.path.join(models_dir, "bilstm_vocab_leakage_free.json")
    vocab.save(vocab_save_path_lf)
    # Also save standard alias
    vocab.save(os.path.join(models_dir, "bilstm_vocab.json"))
    
    # 2. DataLoaders
    train_dataset = ClinicalNoteDataset(train_texts, train_labels, vocab, max_len=MAX_SEQ_LEN)
    val_dataset = ClinicalNoteDataset(val_texts, val_labels, vocab, max_len=MAX_SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Model setup
    model = ClinicalBiLSTM(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES
    ).to(device)
    
    # Class weights for CrossEntropyLoss
    class_counts = [train_labels.count(i) for i in range(NUM_CLASSES)]
    total_samples = len(train_labels)
    class_weights = [total_samples / (NUM_CLASSES * max(c, 1)) for c in class_counts]
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 4. Training loop with early stopping on Macro F1
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_acc": []
    }
    
    best_macro_f1 = -1.0
    best_weights = None
    patience_counter = 0
    best_epoch = 0
    
    print(f"[BiLSTM] Commencing training for up to {epochs} epochs (Early stopping patience={patience})...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(batch_y)
            
        avg_train_loss = total_train_loss / len(train_dataset)
        
        # Validation evaluation
        model.eval()
        total_val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                total_val_loss += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(batch_y.cpu().numpy())
                
        avg_val_loss = total_val_loss / len(val_dataset)
        val_macro_f1 = f1_score(val_targets, val_preds, average="macro")
        val_acc = accuracy_score(val_targets, val_preds)
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_acc"].append(val_acc)
        
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro F1: {val_macro_f1:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[BiLSTM] Early stopping triggered at epoch {epoch} (Best epoch: {best_epoch} with Val Macro F1: {best_macro_f1:.4f})")
                break
                
    # Load best weights
    model.load_state_dict(best_weights)
    model_save_path_lf = os.path.join(models_dir, "bilstm_leakage_free.pt")
    torch.save(model.state_dict(), model_save_path_lf)
    # Also save standard alias
    torch.save(model.state_dict(), os.path.join(models_dir, "bilstm_model.pt"))
    print(f"[BiLSTM] Best leakage-free model checkpoint saved to: {model_save_path_lf}")
    
    # 5. Generate learning curves
    plot_path_lf = os.path.join(figures_dir, "bilstm_leakage_free_training_curves.png")
    plot_bilstm_curves(history, plot_path_lf)
    plot_bilstm_curves(history, os.path.join(figures_dir, "bilstm_training_curves.png"))
    
    summary = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_macro_f1),
        "vocab_size": len(vocab),
        "history": history
    }
    return model, vocab, summary


def plot_bilstm_curves(history: Dict[str, List[float]], save_path: str):
    """Generates training & validation curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    
    ax1.plot(epochs, history["train_loss"], "b-o", label="Training Loss")
    ax1.plot(epochs, history["val_loss"], "r--s", label="Validation Loss")
    ax1.set_title("BiLSTM Loss Curves")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    ax2.plot(epochs, history["val_macro_f1"], "g-^", label="Val Macro F1")
    ax2.plot(epochs, history["val_acc"], "m--d", label="Val Accuracy")
    ax2.set_title("BiLSTM Validation Metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[BiLSTM] Training curves saved to: {save_path}")
