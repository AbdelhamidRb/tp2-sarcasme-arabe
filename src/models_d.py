import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models_c import AttentionPooling

def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Calcule la moyenne des vecteurs pondérée par le masque d'attention (ignore le padding).
    last_hidden_state: [batch_size, seq_len, hidden_size]
    attention_mask:    [batch_size, seq_len]
    Sortie:            [batch_size, hidden_size]"""
    input_mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


class AraBertFrozenBiLSTMAttention(nn.Module):
    """Architecture D2 : Encodeur AraBERT gelé (feature extractor)
    suivi d'une tête BiLSTM + pooling attentionnel personnalisé de C2."""
    def __init__(
        self,
        arabert_model: nn.Module,
        hidden_dim: int = 128,
        att_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.arabert = arabert_model
        # Geler tous les paramètres d'AraBERT
        for param in self.arabert.parameters():
            param.requires_grad = False

        bert_dim = self.arabert.config.hidden_size  # 768 pour bert-base

        self.lstm = nn.LSTM(
            input_size=bert_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = AttentionPooling(in_features=hidden_dim * 2, att_dim=att_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        # AraBERT en mode gelé (pas de gradient pour l'encodeur)
        with torch.no_grad():
            outputs = self.arabert(input_ids=input_ids, attention_mask=attention_mask)
            seq_reps = outputs.last_hidden_state  # [batch_size, seq_len, 768]

        # Tête BiLSTM
        lstm_out, _ = self.lstm(seq_reps)  # [batch_size, seq_len, 2*hidden_dim]

        # Masque de padding : True pour les positions à ignorer (attention_mask == 0)
        pad_mask = (attention_mask == 0)
        c, weights = self.attention(lstm_out, mask=pad_mask)

        rep = self.dropout(c)
        logits = self.fc(rep)

        if return_attention:
            return logits, weights
        return logits
