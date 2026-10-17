import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# C1. BiLSTM Classifier
# ==============================================================================

class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 128,
        num_classes: int = 2,
        embedding_matrix: torch.Tensor | None = None,
        freeze_embedding: bool = False,
        padding_idx: int = 0,
        pooling: str = "max",
        dropout: float = 0.3,
    ):
        super().__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, freeze=freeze_embedding, padding_idx=padding_idx
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # [batch, seq_len, embed_dim]
        emb = self.dropout(self.embedding(input_ids))
        out, (h_n, _) = self.lstm(emb)  # out: [batch, seq_len, 2*hidden_dim]

        if self.pooling == "max":
            # Masquer le padding pour le max-pooling
            mask = (input_ids == 0).unsqueeze(-1)  # [batch, seq_len, 1]
            out_masked = out.masked_fill(mask, -1e9)
            rep = torch.max(out_masked, dim=1)[0]
        else:
            # Concaténation des deux derniers états cachés (avant et arrière)
            rep = torch.cat([h_n[-2], h_n[-1]], dim=-1)

        rep = self.dropout(rep)
        return self.fc(rep)


# ==============================================================================
# C2. BiLSTM avec pooling attentionnel custom : alpha_i = softmax(v^T tanh(W h_i))
# ==============================================================================

class AttentionPooling(nn.Module):
    """Pooling attentionnel selon la formule imposée :
    alpha_i = softmax(v^T tanh(W h_i)), c = sum(alpha_i * h_i)"""
    def __init__(self, in_features: int, att_dim: int = 64):
        super().__init__()
        self.W = nn.Linear(in_features, att_dim, bias=True)
        self.v = nn.Linear(att_dim, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        # h: [batch, seq_len, in_features]
        # u: [batch, seq_len, att_dim]
        u = torch.tanh(self.W(h))
        # scores: [batch, seq_len]
        scores = self.v(u).squeeze(-1)

        if mask is not None:
            # mask: [batch, seq_len] (True pour les tokens de padding à ignorer)
            scores = scores.masked_fill(mask, -1e9)

        weights = F.softmax(scores, dim=-1)  # [batch, seq_len]
        # Contexte pondéré c: [batch, in_features]
        c = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
        return c, weights


class AttentionBiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 128,
        att_dim: int = 64,
        num_classes: int = 2,
        embedding_matrix: torch.Tensor | None = None,
        freeze_embedding: bool = False,
        padding_idx: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, freeze=freeze_embedding, padding_idx=padding_idx
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = AttentionPooling(in_features=hidden_dim * 2, att_dim=att_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(
        self, input_ids: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        emb = self.dropout(self.embedding(input_ids))
        out, _ = self.lstm(emb)  # [batch, seq_len, 2*hidden_dim]
        pad_mask = (input_ids == 0)
        c, weights = self.attention(out, mask=pad_mask)
        rep = self.dropout(c)
        logits = self.fc(rep)
        if return_attention:
            return logits, weights
        return logits


# ==============================================================================
# C3. Mini-encodeur Transformer (2 couches, d_model=128, 4 têtes) + Encodage Pos
# ==============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]


class MiniTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        dim_feedforward: int = 256,
        use_positional_encoding: bool = True,
        embedding_matrix: torch.Tensor | None = None,
        freeze_embedding: bool = False,
        padding_idx: int = 0,
        dropout: float = 0.2,
    ):
        super().__init__()
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_matrix, freeze=freeze_embedding, padding_idx=padding_idx
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        self.proj = nn.Linear(embed_dim, d_model) if embed_dim != d_model else nn.Identity()
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model=d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Masque de padding : True pour les positions à masquer (ignorer)
        pad_mask = (input_ids == 0)
        emb = self.embedding(input_ids)
        h = self.proj(emb)

        if self.use_positional_encoding:
            h = self.pos_encoder(h)

        h = self.dropout(h)
        out = self.transformer(h, src_key_padding_mask=pad_mask)  # [batch, seq_len, d_model]

        # Mean-pooling masqué sur les tokens valides
        mask_expanded = (~pad_mask).unsqueeze(-1).float()  # 1 pour valide, 0 pour pad
        sum_embeddings = torch.sum(out * mask_expanded, dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        rep = sum_embeddings / sum_mask

        rep = self.dropout(rep)
        return self.fc(rep)
