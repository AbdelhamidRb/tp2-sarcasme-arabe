import re
import numpy as np
import torch
from torch.utils.data import Dataset

def clean_str_aravec(text: str) -> str:
    """Nettoyage officiel d'AraVec pour aligner les tokens avec son vocabulaire."""
    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\\", '\n', '\t', '&quot;', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ', ' ! ']
    
    # Suppression du tachkeel (diacritiques)
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel, "", str(text))
    
    # Rduction des allongements (tatweel / lettres rptes)
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    
    for s, r in zip(search, replace):
        text = text.replace(s, r)
        
    return text.strip()

def tokenize(text: str) -> list[str]:
    """Tokenisation simple par espaces."""
    return text.split()

class Vocabulary:
    def __init__(self, pad_token: str = "<PAD>", unk_token: str = "<UNK>"):
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.w2i = {pad_token: 0, unk_token: 1}
        self.i2w = {0: pad_token, 1: unk_token}
        self.pad_idx = 0
        self.unk_idx = 1

    def fit(self, texts: list[str], min_freq: int = 2) -> "Vocabulary":
        freqs = {}
        for t in texts:
            for w in tokenize(t):
                freqs[w] = freqs.get(w, 0) + 1
        for w, c in freqs.items():
            if c >= min_freq and w not in self.w2i:
                idx = len(self.w2i)
                self.w2i[w] = idx
                self.i2w[idx] = w
        return self

    def __len__(self) -> int:
        return len(self.w2i)

    def encode(self, text: str, max_len: int = 64) -> tuple[list[int], int]:
        tokens = tokenize(text)[:max_len]
        length = len(tokens)
        ids = [self.w2i.get(w, self.unk_idx) for w in tokens]
        if len(ids) < max_len:
            ids = ids + [self.pad_idx] * (max_len - len(ids))
        return ids, length

    def oov_rate(self, texts: list[str]) -> float:
        total = 0
        oov = 0
        for t in texts:
            tokens = tokenize(t)
            total += len(tokens)
            oov += sum(1 for w in tokens if w not in self.w2i)
        return (oov / total) if total > 0 else 0.0

def build_embedding_matrix(vocab: Vocabulary, aravec_wv, embed_dim: int = 300) -> torch.Tensor:
    """Construit la matrice d'embedding PyTorch  partir d'AraVec.
    Les mots absents d'AraVec sont initialiss alatoirement (gaussienne)."""
    matrix = np.random.normal(scale=0.1, size=(len(vocab), embed_dim)).astype(np.float32)
    matrix[vocab.pad_idx] = np.zeros(embed_dim, dtype=np.float32)
    
    trouves = 0
    for w, idx in vocab.w2i.items():
        if idx in (vocab.pad_idx, vocab.unk_idx):
            continue
        # Recherche directe ou avec clean_str
        w_clean = clean_str_aravec(w)
        if w in aravec_wv:
            matrix[idx] = aravec_wv[w]
            trouves += 1
        elif w_clean in aravec_wv:
            matrix[idx] = aravec_wv[w_clean]
            trouves += 1
            
    print(f"AraVec couverture vocabulaire : {trouves}/{len(vocab) - 2} ({trouves/(len(vocab)-2):.2%})")
    return torch.tensor(matrix, dtype=torch.float32)

class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int] | None, vocab: Vocabulary, max_len: int = 64):
        self.texts = list(texts)
        self.labels = list(labels) if labels is not None else None
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        ids, length = self.vocab.encode(self.texts[idx], max_len=self.max_len)
        item = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
        }
        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
