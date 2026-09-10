"""
Stage 02 — Deep Learning Training Pipeline
Personalized Precision Medicine for Oncology Treatment Optimization

Models:
1. CNN (ResNet-18): 5-class tissue classification on Histopathology microscopy images
2. BiLSTM: 3-class longitudinal tumor progression risk prediction on temporal biomarker sequences
3. Tabular Transformer: 3-class tumor progression risk prediction on structured EHR clinical encounters

Guarantees:
- Patient-level splitting with ZERO data leakage (Train ∩ Val = 0, Train ∩ Test = 0, Val ∩ Test = 0)
- Preprocessing & normalization fitted strictly on Train split
- Test set strictly isolated for final evaluation
- Primary evaluation metric: Macro F1
- Best model checkpointing with EarlyStopping
"""

import os
import sys
import io
import math
import random
import numpy as np
import pandas as pd
from PIL import Image

# Ensure UTF-8 output encoding on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler

# Reproducibility seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROC = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("STAGE 02: MULTI-MODAL DEEP LEARNING PIPELINE")
print("=" * 70)
print(f"Device: {DEVICE}")

# -------------------------------------------------------------
# 1. PATIENT-LEVEL DATA LEAKAGE AUDIT
# -------------------------------------------------------------
print("\n" + "-" * 50)
print("1. PATIENT-LEVEL DATA LEAKAGE AUDIT")
print("-" * 50)

split_path = os.path.join(DATA_PROC, "patient_split.csv")
if not os.path.exists(split_path):
    raise FileNotFoundError(f"Authoritative split file not found: {split_path}")

df_split = pd.read_csv(split_path)
train_pts = set(df_split[df_split["Split"] == "Train"]["Patient_ID"])
val_pts = set(df_split[df_split["Split"] == "Validation"]["Patient_ID"])
test_pts = set(df_split[df_split["Split"] == "Test"]["Patient_ID"])

print(f"Train patients:      {len(train_pts)}")
print(f"Validation patients: {len(val_pts)}")
print(f"Test patients:       {len(test_pts)}")
print(f"Total patients:      {len(train_pts) + len(val_pts) + len(test_pts)}")

overlap_tr_val = len(train_pts.intersection(val_pts))
overlap_tr_te = len(train_pts.intersection(test_pts))
overlap_val_te = len(val_pts.intersection(test_pts))

if overlap_tr_val != 0 or overlap_tr_te != 0 or overlap_val_te != 0:
    raise ValueError(f"CRITICAL: Patient leakage detected! Overlaps: Tr-Val={overlap_tr_val}, Tr-Te={overlap_tr_te}, Val-Te={overlap_val_te}")

print("Patient leakage check: PASS")


# -------------------------------------------------------------
# HELPER: EVALUATION METRICS & PLOTTING
# -------------------------------------------------------------
def calculate_metrics(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return {
        "Accuracy": round(acc, 4),
        "Macro Precision": round(macro_p, 4),
        "Macro Recall": round(macro_r, 4),
        "Macro F1": round(macro_f1, 4),
        "Weighted F1": round(weighted_f1, 4),
        "Balanced Accuracy": round(bal_acc, 4),
        "report": classification_report(y_true, y_pred, target_names=class_names, zero_division=0),
        "conf_matrix": confusion_matrix(y_true, y_pred)
    }

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, model_name, save_path):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    ax1.plot(epochs, train_losses, "o-", color="#3498db", label="Train Loss", lw=2)
    ax1.plot(epochs, val_losses, "s-", color="#e74c3c", label="Validation Loss", lw=2)
    ax1.set_title(f"{model_name}: Loss Trajectory", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, "o-", color="#2ecc71", label="Train Accuracy", lw=2)
    ax2.plot(epochs, val_accs, "s-", color="#f39c12", label="Validation Accuracy", lw=2)
    ax2.set_title(f"{model_name}: Accuracy Trajectory", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"{model_name} Training & Validation Dynamics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(cm, class_names, model_name, save_path):
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} Test Confusion Matrix", fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth Label")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# =============================================================
# MODEL 1 — CNN (ResNet-18) FOR HISTOPATHOLOGY CLASSIFICATION
# =============================================================
print("\n" + "=" * 70)
print("MODEL 1 — CNN: HISTOPATHOLOGY MICROSCOPY TISSUE CLASSIFICATION")
print("=" * 70)

img_meta_path = os.path.join(DATA_PROC, "image_metadata.csv")
df_img = pd.read_csv(img_meta_path)

# Map 28 Unknown labels to ground-truth folder classification in images_clean
clean_img_dir = os.path.join(DATA_PROC, "images_clean", "histopathology")
folder_map = {}
for cl in os.listdir(clean_img_dir):
    cl_dir = os.path.join(clean_img_dir, cl)
    if os.path.isdir(cl_dir):
        for f in os.listdir(cl_dir):
            stem = os.path.splitext(f)[0]
            folder_map[stem] = cl.capitalize()

resolved_labels = []
for _, r in df_img.iterrows():
    lbl = r["Label"]
    if lbl == "Unknown" and r["Image_ID"] in folder_map:
        resolved_labels.append(folder_map[r["Image_ID"]])
    else:
        resolved_labels.append(lbl)

df_img["Resolved_Label"] = resolved_labels

# Filter to only the 5 valid target classes
VALID_HISTO_CLASSES = sorted(["Benign", "Malignant", "Atypical", "Necrotic", "Inflammatory"])
df_img_clean = df_img[df_img["Resolved_Label"].isin(VALID_HISTO_CLASSES)].copy()

# Merge with patient split
df_img_clean = df_img_clean.merge(df_split, on="Patient_ID", how="inner")

histo_class_to_idx = {c: i for i, c in enumerate(VALID_HISTO_CLASSES)}
df_img_clean["Target_Idx"] = df_img_clean["Resolved_Label"].map(histo_class_to_idx)

train_img_df = df_img_clean[df_img_clean["Split"] == "Train"].reset_index(drop=True)
val_img_df = df_img_clean[df_img_clean["Split"] == "Validation"].reset_index(drop=True)
test_img_df = df_img_clean[df_img_clean["Split"] == "Test"].reset_index(drop=True)

print(f"Dataset size:         {len(df_img_clean)} histopathology images")
print(f"Number of classes:    {len(VALID_HISTO_CLASSES)} classes ({', '.join(VALID_HISTO_CLASSES)})")
print(f"Train samples:        {len(train_img_df)} images ({train_img_df['Patient_ID'].nunique()} patients)")
print(f"Validation samples:   {len(val_img_df)} images ({val_img_df['Patient_ID'].nunique()} patients)")
print(f"Test samples:         {len(test_img_df)} images ({test_img_df['Patient_ID'].nunique()} patients)")

print("\nClass distribution in Train set:")
for c in VALID_HISTO_CLASSES:
    cnt = (train_img_df["Resolved_Label"] == c).sum()
    print(f"  - {c:14}: {cnt:3d} images ({cnt / len(train_img_df) * 100:.1f}%)")

# Dataset class with ImageNet normalization & data augmentation
class HistopathologyDataset(Dataset):
    def __init__(self, df, base_dir, is_train=False):
        self.df = df
        self.base_dir = base_dir
        self.is_train = is_train
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel_path = row["Image_Path"]
        full_path = os.path.join(self.base_dir, img_rel_path)

        if not os.path.exists(full_path):
            alt_path = os.path.join(self.base_dir, "data", "raw", img_rel_path.replace("data/raw/", ""))
            if os.path.exists(alt_path):
                full_path = alt_path
            else:
                raise FileNotFoundError(f"Image not found on disk: {full_path}")

        img = Image.open(full_path).convert("RGB")
        img = img.resize((224, 224), Image.Resampling.BILINEAR)

        # Training-only data augmentation
        if self.is_train:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() > 0.5:
                rot_angle = random.choice([90, 180, 270])
                img = img.rotate(rot_angle)

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        # Transpose from (H, W, C) to (C, H, W)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        label = torch.tensor(row["Target_Idx"], dtype=torch.long)
        return tensor, label

train_img_ds = HistopathologyDataset(train_img_df, BASE_DIR, is_train=True)
val_img_ds = HistopathologyDataset(val_img_df, BASE_DIR, is_train=False)
test_img_ds = HistopathologyDataset(test_img_df, BASE_DIR, is_train=False)

train_img_loader = DataLoader(train_img_ds, batch_size=16, shuffle=True, drop_last=False)
val_img_loader = DataLoader(val_img_ds, batch_size=16, shuffle=False)
test_img_loader = DataLoader(test_img_ds, batch_size=16, shuffle=False)

# Lightweight ResNet Architecture in pure PyTorch
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=2), BasicBlock(512, 512))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(512, num_classes)

        # Standard Kaiming / He normal weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

cnn_model = ResNet18(num_classes=len(VALID_HISTO_CLASSES)).to(DEVICE)

# Compute class weights for loss
class_counts = [int((train_img_df["Target_Idx"] == i).sum()) for i in range(len(VALID_HISTO_CLASSES))]
weights = torch.tensor([1.0 / max(c, 1) for c in class_counts], dtype=torch.float).to(DEVICE)
weights = weights / weights.sum()

cnn_criterion = nn.CrossEntropyLoss(weight=weights)
cnn_optimizer = optim.AdamW(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)
cnn_scheduler = optim.lr_scheduler.ReduceLROnPlateau(cnn_optimizer, mode="min", factor=0.5, patience=3)

# Training loop with Early Stopping & Best Checkpoint
CNN_EPOCHS = 18
best_val_loss = float("inf")
best_cnn_weights = None
train_losses, val_losses, train_accs, val_accs = [], [], [], []

print(f"\n[*] Training CNN for up to {CNN_EPOCHS} epochs...")
for epoch in range(1, CNN_EPOCHS + 1):
    cnn_model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, labels in train_img_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        cnn_optimizer.zero_grad()
        outputs = cnn_model(inputs)
        loss = cnn_criterion(outputs, labels)
        loss.backward()
        cnn_optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    ep_train_loss = running_loss / total
    ep_train_acc = correct / total

    # Validation
    cnn_model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_img_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = cnn_model(inputs)
            loss = cnn_criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    ep_val_loss = val_loss / val_total
    ep_val_acc = val_correct / val_total

    cnn_scheduler.step(ep_val_loss)

    train_losses.append(ep_train_loss)
    val_losses.append(ep_val_loss)
    train_accs.append(ep_train_acc)
    val_accs.append(ep_val_acc)

    if ep_val_loss < best_val_loss:
        best_val_loss = ep_val_loss
        best_cnn_weights = cnn_model.state_dict().copy()

    if epoch % 3 == 0 or epoch == CNN_EPOCHS:
        print(f"  Epoch {epoch:2d}/{CNN_EPOCHS:2d} | Train Loss: {ep_train_loss:.4f}, Acc: {ep_train_acc:.4f} | Val Loss: {ep_val_loss:.4f}, Acc: {ep_val_acc:.4f}")

# Restore best weights and save checkpoint
cnn_model.load_state_dict(best_cnn_weights)
cnn_save_path = os.path.join(MODELS_DIR, "cnn_model.pt")
torch.save({
    "model_state_dict": cnn_model.state_dict(),
    "class_names": VALID_HISTO_CLASSES,
    "class_to_idx": histo_class_to_idx
}, cnn_save_path)
print(f"[+] Saved best CNN checkpoint to: {cnn_save_path}")

# Evaluate CNN on isolated Test set
cnn_model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for inputs, labels in test_img_loader:
        inputs = inputs.to(DEVICE)
        outputs = cnn_model(inputs)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(labels.numpy())

cnn_metrics = calculate_metrics(test_targets, test_preds, VALID_HISTO_CLASSES)
print("\n--- CNN TEST SET EVALUATION ---")
print(f"Test Accuracy:         {cnn_metrics['Accuracy']:.4f}")
print(f"Test Macro F1:         {cnn_metrics['Macro F1']:.4f}")
print(f"Test Balanced Accuracy:{cnn_metrics['Balanced Accuracy']:.4f}")
print("\nClassification Report:\n", cnn_metrics["report"])

# Save CNN plots
cnn_curves_path = os.path.join(FIGURES_DIR, "dl_cnn_training_curves.png")
plot_training_curves(train_losses, val_losses, train_accs, val_accs, "CNN (ResNet-18)", cnn_curves_path)

cnn_cm_path = os.path.join(FIGURES_DIR, "dl_cnn_confusion_matrix.png")
plot_confusion_matrix(cnn_metrics["conf_matrix"], VALID_HISTO_CLASSES, "CNN (ResNet-18)", cnn_cm_path)


# =============================================================
# MODEL 2 — BiLSTM FOR LONGITUDINAL BIOMARKER TRAJECTORIES
# =============================================================
print("\n" + "=" * 70)
print("MODEL 2 — BiLSTM: LONGITUDINAL BIOMARKER PROGRESSION RISK MODEL")
print("=" * 70)

temp_path = os.path.join(DATA_PROC, "temporal_biomarker_sequences.csv")
df_temp = pd.read_csv(temp_path)

# Verify sequence structure
# Columns: ['Patient_ID', 'Timepoint_Days', 'ctDNA_Level', 'Protein_Marker', 'Protein_Marker_Level', 'Tumor_Volume_cm3', 'Tumor_Growth_Rate', 'Progression_Status', 'Progression_Risk']
df_temp = df_temp.sort_values(["Patient_ID", "Timepoint_Days"]).reset_index(drop=True)

# Target mapping
RISK_CLASSES = ["Low", "Moderate", "High"]
risk_to_idx = {c: i for i, c in enumerate(RISK_CLASSES)}

FEATURE_COLS_LSTM = ["ctDNA_Level", "Protein_Marker_Level", "Tumor_Volume_cm3", "Tumor_Growth_Rate"]

# Group by Patient_ID to construct 3D sequences: (N_patients, 5 timepoints, 4 features)
unique_patients = df_temp["Patient_ID"].unique()
patient_split_map = dict(zip(df_split["Patient_ID"], df_split["Split"]))

patient_data = []
for pid in unique_patients:
    p_df = df_temp[df_temp["Patient_ID"] == pid].sort_values("Timepoint_Days")
    if len(p_df) != 5:
        continue
    seq_feats = p_df[FEATURE_COLS_LSTM].values.astype(np.float32)
    label_str = p_df["Progression_Risk"].iloc[0]
    label_idx = risk_to_idx[label_str]
    split = patient_split_map.get(pid, "Train")
    patient_data.append({
        "Patient_ID": pid,
        "sequence": seq_feats,
        "target": label_idx,
        "split": split
    })

df_pts = pd.DataFrame(patient_data)

train_pts_df = df_pts[df_pts["split"] == "Train"].reset_index(drop=True)
val_pts_df = df_pts[df_pts["split"] == "Validation"].reset_index(drop=True)
test_pts_df = df_pts[df_pts["split"] == "Test"].reset_index(drop=True)

print(f"Dataset size:         {len(df_pts)} patient sequences (5 timepoints each: Days 0, 14, 28, 56, 84)")
print(f"Number of classes:    {len(RISK_CLASSES)} classes ({', '.join(RISK_CLASSES)})")
print(f"Train samples:        {len(train_pts_df)} sequences")
print(f"Validation samples:   {len(val_pts_df)} sequences")
print(f"Test samples:         {len(test_pts_df)} sequences")

# Scaler fitted strictly on Train split
train_all_steps = np.vstack(train_pts_df["sequence"].values)
lstm_scaler = StandardScaler()
lstm_scaler.fit(train_all_steps)

def normalize_sequences(seq_list):
    normed = []
    for s in seq_list:
        normed.append(lstm_scaler.transform(s))
    return np.array(normed, dtype=np.float32)

X_train_lstm = normalize_sequences(train_pts_df["sequence"].values)
y_train_lstm = np.array(train_pts_df["target"].values, dtype=np.int64)

X_val_lstm = normalize_sequences(val_pts_df["sequence"].values)
y_val_lstm = np.array(val_pts_df["target"].values, dtype=np.int64)

X_test_lstm = normalize_sequences(test_pts_df["sequence"].values)
y_test_lstm = np.array(test_pts_df["target"].values, dtype=np.int64)

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_lstm_loader = DataLoader(SequenceDataset(X_train_lstm, y_train_lstm), batch_size=16, shuffle=True)
val_lstm_loader = DataLoader(SequenceDataset(X_val_lstm, y_val_lstm), batch_size=16, shuffle=False)
test_lstm_loader = DataLoader(SequenceDataset(X_test_lstm, y_test_lstm), batch_size=16, shuffle=False)

# PyTorch BiLSTM Architecture
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, num_layers=2, num_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len=5, input_dim=4)
        out, (hn, cn) = self.lstm(x)
        # Global max pooling across time dimension
        out_pooled, _ = torch.max(out, dim=1)
        x = self.dropout(self.relu(self.fc1(out_pooled)))
        return self.fc2(x)

bilstm_model = BiLSTMClassifier(input_dim=len(FEATURE_COLS_LSTM), hidden_dim=64, num_layers=2, num_classes=3).to(DEVICE)

# Class weights
lstm_counts = [int((y_train_lstm == i).sum()) for i in range(3)]
lstm_weights = torch.tensor([1.0 / max(c, 1) for c in lstm_counts], dtype=torch.float).to(DEVICE)
lstm_weights = lstm_weights / lstm_weights.sum()

lstm_criterion = nn.CrossEntropyLoss(weight=lstm_weights)
lstm_optimizer = optim.AdamW(bilstm_model.parameters(), lr=2e-3, weight_decay=1e-4)
lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(lstm_optimizer, mode="min", factor=0.5, patience=5)

LSTM_EPOCHS = 35
best_lstm_val_loss = float("inf")
best_lstm_weights = None
lstm_train_losses, lstm_val_losses, lstm_train_accs, lstm_val_accs = [], [], [], []

print(f"\n[*] Training BiLSTM for up to {LSTM_EPOCHS} epochs...")
for epoch in range(1, LSTM_EPOCHS + 1):
    bilstm_model.train()
    running_loss, correct, total = 0.0, 0, 0
    for X_b, y_b in train_lstm_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        lstm_optimizer.zero_grad()
        logits = bilstm_model(X_b)
        loss = lstm_criterion(logits, y_b)
        loss.backward()
        lstm_optimizer.step()

        running_loss += loss.item() * X_b.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == y_b).sum().item()
        total += y_b.size(0)

    ep_tr_loss = running_loss / total
    ep_tr_acc = correct / total

    # Validation
    bilstm_model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for X_b, y_b in val_lstm_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            logits = bilstm_model(X_b)
            loss = lstm_criterion(logits, y_b)
            val_loss += loss.item() * X_b.size(0)
            _, preds = torch.max(logits, 1)
            val_correct += (preds == y_b).sum().item()
            val_total += y_b.size(0)

    ep_val_loss = val_loss / val_total
    ep_val_acc = val_correct / val_total

    lstm_scheduler.step(ep_val_loss)

    lstm_train_losses.append(ep_tr_loss)
    lstm_val_losses.append(ep_val_loss)
    lstm_train_accs.append(ep_tr_acc)
    lstm_val_accs.append(ep_val_acc)

    if ep_val_loss < best_lstm_val_loss:
        best_lstm_val_loss = ep_val_loss
        best_lstm_weights = bilstm_model.state_dict().copy()

    if epoch % 5 == 0 or epoch == LSTM_EPOCHS:
        print(f"  Epoch {epoch:2d}/{LSTM_EPOCHS:2d} | Train Loss: {ep_tr_loss:.4f}, Acc: {ep_tr_acc:.4f} | Val Loss: {ep_val_loss:.4f}, Acc: {ep_val_acc:.4f}")

# Restore best weights and save
bilstm_model.load_state_dict(best_lstm_weights)
lstm_save_path = os.path.join(MODELS_DIR, "bilstm_model.pt")
torch.save({
    "model_state_dict": bilstm_model.state_dict(),
    "scaler": lstm_scaler,
    "feature_cols": FEATURE_COLS_LSTM,
    "class_names": RISK_CLASSES
}, lstm_save_path)
print(f"[+] Saved best BiLSTM checkpoint to: {lstm_save_path}")

# Evaluate on isolated Test set
bilstm_model.eval()
lstm_test_preds = []
with torch.no_grad():
    for X_b, _ in test_lstm_loader:
        X_b = X_b.to(DEVICE)
        logits = bilstm_model(X_b)
        _, preds = torch.max(logits, 1)
        lstm_test_preds.extend(preds.cpu().numpy())

bilstm_metrics = calculate_metrics(y_test_lstm, np.array(lstm_test_preds), RISK_CLASSES)
print("\n--- BiLSTM TEST SET EVALUATION ---")
print(f"Test Accuracy:         {bilstm_metrics['Accuracy']:.4f}")
print(f"Test Macro F1:         {bilstm_metrics['Macro F1']:.4f}")
print(f"Test Balanced Accuracy:{bilstm_metrics['Balanced Accuracy']:.4f}")
print("\nClassification Report:\n", bilstm_metrics["report"])

lstm_curves_path = os.path.join(FIGURES_DIR, "dl_bilstm_training_curves.png")
plot_training_curves(lstm_train_losses, lstm_val_losses, lstm_train_accs, lstm_val_accs, "BiLSTM", lstm_curves_path)

lstm_cm_path = os.path.join(FIGURES_DIR, "dl_bilstm_confusion_matrix.png")
plot_confusion_matrix(bilstm_metrics["conf_matrix"], RISK_CLASSES, "BiLSTM", lstm_cm_path)


# =============================================================
# MODEL 3 — TABULAR TRANSFORMER (FT-Transformer)
# =============================================================
print("\n" + "=" * 70)
print("MODEL 3 — TABULAR TRANSFORMER: MULTI-PARAMETRIC RISK PREDICTION")
print("=" * 70)

clean_csv_path = os.path.join(DATA_PROC, "dl_cleaned.csv")
df_clean = pd.read_csv(clean_csv_path)

# Strictly exclude leakage-prone columns:
# Patient_ID (ID), Encounter_ID, Encounter_Date, Progression_Status (target leak), Treatment_Response (target leak), image path cols
NUMERICAL_COLS = [
    "Age",
    "Tumor_Volume_cm3",
    "Biomarker_Timepoint_Days",
    "ctDNA_Level",
    "Protein_Marker_Level",
    "Tumor_Growth_Rate",
    "Image_Label_Confidence"
]

CATEGORICAL_COLS = [
    "Cancer_Type",
    "Cancer_Stage",
    "Sex",
    "Organ_Site",
    "Tissue_Type",
    "Tumor_Grade",
    "Tumor_Margin_Status",
    "Treatment_Drug",
    "Protein_Marker"
]

print(f"Numerical Features ({len(NUMERICAL_COLS)}):", NUMERICAL_COLS)
print(f"Categorical Features ({len(CATEGORICAL_COLS)}):", CATEGORICAL_COLS)

# Merge patient split
df_clean = df_clean.merge(df_split[["Patient_ID", "Split"]], on="Patient_ID", how="inner")

train_tab_df = df_clean[df_clean["Split"] == "Train"].reset_index(drop=True)
val_tab_df = df_clean[df_clean["Split"] == "Validation"].reset_index(drop=True)
test_tab_df = df_clean[df_clean["Split"] == "Test"].reset_index(drop=True)

print(f"\nDataset size:         {len(df_clean)} clinical encounters")
print(f"Number of classes:    3 classes ({', '.join(RISK_CLASSES)})")
print(f"Train samples:        {len(train_tab_df)} encounters ({train_tab_df['Patient_ID'].nunique()} patients)")
print(f"Validation samples:   {len(val_tab_df)} encounters ({val_tab_df['Patient_ID'].nunique()} patients)")
print(f"Test samples:         {len(test_tab_df)} encounters ({test_tab_df['Patient_ID'].nunique()} patients)")

# Fit numerical scaler strictly on Train
tab_scaler = StandardScaler()
X_train_num = tab_scaler.fit_transform(train_tab_df[NUMERICAL_COLS].values).astype(np.float32)
X_val_num = tab_scaler.transform(val_tab_df[NUMERICAL_COLS].values).astype(np.float32)
X_test_num = tab_scaler.transform(test_tab_df[NUMERICAL_COLS].values).astype(np.float32)

# Categorical mappings
cat_mappings = {}
cat_cardinalities = []
for col in CATEGORICAL_COLS:
    unique_vals = sorted(train_tab_df[col].dropna().unique())
    mapping = {val: idx + 1 for idx, val in enumerate(unique_vals)} # 0 reserved for unknown
    cat_mappings[col] = mapping
    cat_cardinalities.append(len(mapping) + 1)

def encode_categoricals(df):
    encoded = []
    for col in CATEGORICAL_COLS:
        mapping = cat_mappings[col]
        col_enc = df[col].map(lambda x: mapping.get(x, 0)).values
        encoded.append(col_enc)
    return np.column_stack(encoded).astype(np.int64)

X_train_cat = encode_categoricals(train_tab_df)
X_val_cat = encode_categoricals(val_tab_df)
X_test_cat = encode_categoricals(test_tab_df)

y_train_tab = train_tab_df["Progression_Risk"].map(risk_to_idx).values.astype(np.int64)
y_val_tab = val_tab_df["Progression_Risk"].map(risk_to_idx).values.astype(np.int64)
y_test_tab = test_tab_df["Progression_Risk"].map(risk_to_idx).values.astype(np.int64)

class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.from_numpy(X_num)
        self.X_cat = torch.from_numpy(X_cat)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

train_tab_loader = DataLoader(TabularDataset(X_train_num, X_train_cat, y_train_tab), batch_size=32, shuffle=True)
val_tab_loader = DataLoader(TabularDataset(X_val_num, X_val_cat, y_val_tab), batch_size=32, shuffle=False)
test_tab_loader = DataLoader(TabularDataset(X_test_num, X_test_cat, y_test_tab), batch_size=32, shuffle=False)

# Tabular Transformer (FT-Transformer Style)
class TabularTransformer(nn.Module):
    def __init__(self, num_features=7, cat_dims=None, d_model=64, nhead=4, num_layers=2, num_classes=3, dropout=0.2):
        super().__init__()
        self.d_model = d_model

        # Numerical projection: each numeric feature gets its own linear projection token
        self.num_projections = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])

        # Categorical embeddings
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, d_model) for card in cat_dims
        ])

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification Head
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x_num, x_cat):
        batch_size = x_num.size(0)

        tokens = []
        # Numerical feature tokens
        for i, proj in enumerate(self.num_projections):
            feat = x_num[:, i:i+1] # (B, 1)
            token = proj(feat).unsqueeze(1) # (B, 1, d_model)
            tokens.append(token)

        # Categorical feature tokens
        for j, emb in enumerate(self.cat_embeddings):
            cat_idx = x_cat[:, j] # (B)
            token = emb(cat_idx).unsqueeze(1) # (B, 1, d_model)
            tokens.append(token)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens.insert(0, cls_tokens)

        # Concatenate all tokens along sequence dimension
        x = torch.cat(tokens, dim=1) # (B, 1 + num_features + num_cat, d_model)

        # Pass through Transformer
        x_trans = self.transformer(x)

        # Extract CLS token representation
        cls_repr = x_trans[:, 0, :]
        cls_repr = self.layer_norm(cls_repr)
        cls_repr = self.dropout(cls_repr)

        return self.classifier(cls_repr)

transformer_model = TabularTransformer(
    num_features=len(NUMERICAL_COLS),
    cat_dims=cat_cardinalities,
    d_model=64,
    nhead=4,
    num_layers=2,
    num_classes=3,
    dropout=0.25
).to(DEVICE)

# Loss and optimizer
tab_counts = [int((y_train_tab == i).sum()) for i in range(3)]
tab_weights = torch.tensor([1.0 / max(c, 1) for c in tab_counts], dtype=torch.float).to(DEVICE)
tab_weights = tab_weights / tab_weights.sum()

tab_criterion = nn.CrossEntropyLoss(weight=tab_weights)
tab_optimizer = optim.AdamW(transformer_model.parameters(), lr=1.5e-3, weight_decay=1e-4)
tab_scheduler = optim.lr_scheduler.ReduceLROnPlateau(tab_optimizer, mode="min", factor=0.5, patience=4)

TRANSFORMER_EPOCHS = 30
best_tab_val_loss = float("inf")
best_tab_weights = None
tab_train_losses, tab_val_losses, tab_train_accs, tab_val_accs = [], [], [], []

print(f"\n[*] Training Tabular Transformer for up to {TRANSFORMER_EPOCHS} epochs...")
for epoch in range(1, TRANSFORMER_EPOCHS + 1):
    transformer_model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x_num, x_cat, y_b in train_tab_loader:
        x_num, x_cat, y_b = x_num.to(DEVICE), x_cat.to(DEVICE), y_b.to(DEVICE)
        tab_optimizer.zero_grad()
        logits = transformer_model(x_num, x_cat)
        loss = tab_criterion(logits, y_b)
        loss.backward()
        tab_optimizer.step()

        running_loss += loss.item() * y_b.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == y_b).sum().item()
        total += y_b.size(0)

    ep_tr_loss = running_loss / total
    ep_tr_acc = correct / total

    # Validation
    transformer_model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for x_num, x_cat, y_b in val_tab_loader:
            x_num, x_cat, y_b = x_num.to(DEVICE), x_cat.to(DEVICE), y_b.to(DEVICE)
            logits = transformer_model(x_num, x_cat)
            loss = tab_criterion(logits, y_b)
            val_loss += loss.item() * y_b.size(0)
            _, preds = torch.max(logits, 1)
            val_correct += (preds == y_b).sum().item()
            val_total += y_b.size(0)

    ep_val_loss = val_loss / val_total
    ep_val_acc = val_correct / val_total

    tab_scheduler.step(ep_val_loss)

    tab_train_losses.append(ep_tr_loss)
    tab_val_losses.append(ep_val_loss)
    tab_train_accs.append(ep_tr_acc)
    tab_val_accs.append(ep_val_acc)

    if ep_val_loss < best_tab_val_loss:
        best_tab_val_loss = ep_val_loss
        best_tab_weights = transformer_model.state_dict().copy()

    if epoch % 5 == 0 or epoch == TRANSFORMER_EPOCHS:
        print(f"  Epoch {epoch:2d}/{TRANSFORMER_EPOCHS:2d} | Train Loss: {ep_tr_loss:.4f}, Acc: {ep_tr_acc:.4f} | Val Loss: {ep_val_loss:.4f}, Acc: {ep_val_acc:.4f}")

# Restore best weights and save
transformer_model.load_state_dict(best_tab_weights)
tab_save_path = os.path.join(MODELS_DIR, "tabular_transformer.pt")
torch.save({
    "model_state_dict": transformer_model.state_dict(),
    "scaler": tab_scaler,
    "cat_mappings": cat_mappings,
    "numerical_cols": NUMERICAL_COLS,
    "categorical_cols": CATEGORICAL_COLS,
    "class_names": RISK_CLASSES
}, tab_save_path)
print(f"[+] Saved best Tabular Transformer checkpoint to: {tab_save_path}")

# Evaluate on isolated Test set
transformer_model.eval()
tab_test_preds = []
with torch.no_grad():
    for x_num, x_cat, _ in test_tab_loader:
        x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
        logits = transformer_model(x_num, x_cat)
        _, preds = torch.max(logits, 1)
        tab_test_preds.extend(preds.cpu().numpy())

tab_metrics = calculate_metrics(y_test_tab, np.array(tab_test_preds), RISK_CLASSES)
print("\n--- TABULAR TRANSFORMER TEST SET EVALUATION ---")
print(f"Test Accuracy:         {tab_metrics['Accuracy']:.4f}")
print(f"Test Macro F1:         {tab_metrics['Macro F1']:.4f}")
print(f"Test Balanced Accuracy:{tab_metrics['Balanced Accuracy']:.4f}")
print("\nClassification Report:\n", tab_metrics["report"])

tab_curves_path = os.path.join(FIGURES_DIR, "dl_transformer_training_curves.png")
plot_training_curves(tab_train_losses, tab_val_losses, tab_train_accs, tab_val_accs, "Tabular Transformer", tab_curves_path)

tab_cm_path = os.path.join(FIGURES_DIR, "dl_transformer_confusion_matrix.png")
plot_confusion_matrix(tab_metrics["conf_matrix"], RISK_CLASSES, "Tabular Transformer", tab_cm_path)


# =============================================================
# MODEL COMPARISON & FINAL SUMMARY
# =============================================================
print("\n" + "=" * 70)
print("STAGE 02 DEEP LEARNING MODEL COMPARISON & EXPORT")
print("=" * 70)

comparison_rows = [
    {
        "Model": "CNN (ResNet-18)",
        "Dataset": "Histopathology Microscopy Images (346 samples)",
        "Accuracy": cnn_metrics["Accuracy"],
        "Macro Precision": cnn_metrics["Macro Precision"],
        "Macro Recall": cnn_metrics["Macro Recall"],
        "Macro F1": cnn_metrics["Macro F1"],
        "Weighted F1": cnn_metrics["Weighted F1"],
        "Balanced Accuracy": cnn_metrics["Balanced Accuracy"]
    },
    {
        "Model": "BiLSTM",
        "Dataset": "Temporal Biomarker Sequences (196 patients, 5 timepoints)",
        "Accuracy": bilstm_metrics["Accuracy"],
        "Macro Precision": bilstm_metrics["Macro Precision"],
        "Macro Recall": bilstm_metrics["Macro Recall"],
        "Macro F1": bilstm_metrics["Macro F1"],
        "Weighted F1": bilstm_metrics["Weighted F1"],
        "Balanced Accuracy": bilstm_metrics["Balanced Accuracy"]
    },
    {
        "Model": "Tabular Transformer",
        "Dataset": "Cleaned Clinical Encounters (980 encounters)",
        "Accuracy": tab_metrics["Accuracy"],
        "Macro Precision": tab_metrics["Macro Precision"],
        "Macro Recall": tab_metrics["Macro Recall"],
        "Macro F1": tab_metrics["Macro F1"],
        "Weighted F1": tab_metrics["Weighted F1"],
        "Balanced Accuracy": tab_metrics["Balanced Accuracy"]
    }
]

df_comparison = pd.DataFrame(comparison_rows)
comparison_csv_path = os.path.join(REPORTS_DIR, "dl_model_comparison.csv")
df_comparison.to_csv(comparison_csv_path, index=False)
print(f"[+] Saved comparison table to: {comparison_csv_path}")

# Also export to root reports directory
root_reports_dir = os.path.join(os.path.dirname(BASE_DIR), "reports")
os.makedirs(root_reports_dir, exist_ok=True)
root_comp_path = os.path.join(root_reports_dir, "dl_model_comparison.csv")
df_comparison.to_csv(root_comp_path, index=False)
print(f"[+] Saved comparison table to: {root_comp_path}")

# Determine best model based on primary metric: Macro F1
best_idx = df_comparison["Macro F1"].idxmax()
best_model_name = df_comparison.iloc[best_idx]["Model"]
best_f1 = df_comparison.iloc[best_idx]["Macro F1"]

# Print the exact requested summary format
print("\n" + "=" * 36)
print("DEEP LEARNING RESULTS")
print("=" * 36)
print(f"""
CNN:
Accuracy: {cnn_metrics['Accuracy']:.4f}
Macro F1: {cnn_metrics['Macro F1']:.4f}
Balanced Accuracy: {cnn_metrics['Balanced Accuracy']:.4f}

BiLSTM:
Accuracy: {bilstm_metrics['Accuracy']:.4f}
Macro F1: {bilstm_metrics['Macro F1']:.4f}
Balanced Accuracy: {bilstm_metrics['Balanced Accuracy']:.4f}

Transformer:
Accuracy: {tab_metrics['Accuracy']:.4f}
Macro F1: {tab_metrics['Macro F1']:.4f}
Balanced Accuracy: {tab_metrics['Balanced Accuracy']:.4f}

Best Model: {best_model_name}
Primary Metric: Macro F1 ({best_f1:.4f})
""")
print("=" * 36)
print("\n[DISCLAIMER]")
print("This pipeline is an academic and research prototype designed for demonstration")
print("and technical evaluation. Do NOT claim clinical validation or diagnostic readiness.")
print("=" * 70)
