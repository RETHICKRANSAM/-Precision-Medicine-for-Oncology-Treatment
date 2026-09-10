"""
Stage 03 NLP Pipeline - Text Preprocessing & Vocabulary Management
Preserves clinical negation, medical abbreviations, gene/drug nomenclature,
and avoids aggressive stopword removal or stemming.
"""

import re
import json
import os
from typing import List, Dict, Tuple, Optional
from collections import Counter

import unicodedata

LABEL2ID = {"Low": 0, "Moderate": 1, "High": 2}
ID2LABEL = {0: "Low", 1: "Moderate", 2: "High"}
MAX_SEQ_LEN = 64

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


def normalize_clinical_text(text: str) -> str:
    """
    Safely normalizes clinical note text:
    - Unicode normalization (NFKC)
    - Collapses multiple whitespace and newlines
    - Preserves all punctuation, medical abbreviations, gene names, drug names,
      negation terms ('no', 'not', 'denies', 'without', 'none'), and numeric dosages.
    - Zero target-derived feature tags.
    """
    if not isinstance(text, str):
        return ""
    # Unicode NFKC normalization
    text = unicodedata.normalize("NFKC", text)
    # Normalize common Unicode quotes and hyphens to ASCII equivalents
    text = re.sub(r"[\u2018\u2019\u201A\u201B]", "'", text)
    text = re.sub(r"[\u201C\u201D\u201E\u201F]", '"', text)
    text = re.sub(r"[\u2013\u2014]", "-", text)
    # Collapse multiple whitespace characters into single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clinical_tokenize(text: str) -> List[str]:
    """
    Tokenizes cleaned clinical text while preserving:
    - Negation terms ("no", "not", "without", "none", "no/possible")
    - Medical abbreviations ("sob", "f/u", "nsclc", "tx", "ae", "pt")
    - Drug & gene alphanumeric strings ("egfr", "l858r", "kras-g12c", "v600e")
    - Dosages and numeric values ("80", "150", "mg/day", "bid")
    """
    if not isinstance(text, str):
        return []
    # Normalize slash-joined terms like "no/possible", "prior/current", "f/u"
    text = text.lower().strip()
    # Replace slashes that are not inside dosage (e.g. mg/day) or keep alphanumeric + dash/slash
    tokens = re.findall(r"[a-z0-9]+(?:[-/][a-z0-9]+)*", text)
    return tokens


class ClinicalVocab:
    """Vocabulary builder and encoder for BiLSTM."""
    def __init__(self, min_freq: int = 1, max_vocab_size: Optional[int] = 5000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.word2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self.idx2word: Dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}
        
    def build_vocab(self, texts: List[str]):
        """Fits vocabulary strictly on training texts."""
        counter = Counter()
        for t in texts:
            counter.update(clinical_tokenize(t))
            
        filtered = [w for w, c in counter.most_common() if c >= self.min_freq]
        if self.max_vocab_size:
            filtered = filtered[: self.max_vocab_size - 2]
            
        for w in filtered:
            if w not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[w] = idx
                self.idx2word[idx] = w
                
        print(f"[VOCAB] Built vocabulary of size {len(self.word2idx)} from {len(texts)} training texts.")
        
    def encode(self, text: str, max_len: int = 32) -> List[int]:
        """Encodes text to padded/truncated list of token IDs."""
        tokens = clinical_tokenize(text)
        ids = [self.word2idx.get(tok, UNK_IDX) for tok in tokens[:max_len]]
        if len(ids) < max_len:
            ids += [PAD_IDX] * (max_len - len(ids))
        return ids

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"word2idx": self.word2idx, "min_freq": self.min_freq}, f, indent=2)
        print(f"[VOCAB] Saved vocabulary to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "ClinicalVocab":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(min_freq=data.get("min_freq", 1))
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(v): k for k, v in vocab.word2idx.items()}
        return vocab

    def __len__(self):
        return len(self.word2idx)


def encode_labels(labels: List[str]) -> List[int]:
    """Encodes string urgency labels to 0, 1, 2."""
    return [LABEL2ID[lbl] for lbl in labels]


def decode_labels(ids: List[int]) -> List[str]:
    """Decodes integer class IDs back to string urgency labels."""
    return [ID2LABEL[i] for i in ids]
