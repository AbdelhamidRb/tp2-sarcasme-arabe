import torch
import torch.nn as nn
from transformers import AutoModel

class MonoTaskModel(nn.Module):
    """Architecture E1 : Encodeur Transformer pré-entraîné (dégelé)
    avec une seule tête linéaire pour la classification du sarcasme."""
    def __init__(self, name: str, dropout: float = 0.2):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.sarc = nn.Linear(h, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        cls = self.drop(cls)
        return self.sarc(cls)


class MultiTaskModel(nn.Module):
    """Architecture E2 : Encodeur Transformer partagé (dégelé)
    avec 2 têtes en parallèle :
    - Tête 1 : sarcasme (2 classes : 0 ou 1)
    - Tête 2 : sentiment (3 classes : NEG, NEU, POS)"""
    def __init__(self, name: str, dropout: float = 0.2):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.sarc = nn.Linear(h, 2)
        self.sent = nn.Linear(h, 3)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        cls = self.drop(cls)
        return self.sarc(cls), self.sent(cls)
