import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
from src.train_c import find_best_threshold

SENTIMENT_MAP = {"NEG": 0, "NEU": 1, "POS": 2}

class MultiTaskDataset(Dataset):
    """Dataset gérant les entrées pour le sarcasme et la tâche auxiliaire de sentiment."""
    def __init__(self, texts, sarcasm_labels, sentiment_labels, tokenizer, max_len=64):
        self.encodings = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.sarc_labels = list(sarcasm_labels)
        if sentiment_labels is not None:
            self.sent_labels = [
                SENTIMENT_MAP.get(s, 1) if isinstance(s, str) else int(s) for s in sentiment_labels
            ]
        else:
            self.sent_labels = [0] * len(self.sarc_labels)

    def __len__(self):
        return len(self.sarc_labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["label_sarc"] = torch.tensor(self.sarc_labels[idx], dtype=torch.long)
        item["label_sent"] = torch.tensor(self.sent_labels[idx], dtype=torch.long)
        return item


def evaluate_transformer(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_multitask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Évalue le modèle sur le sarcasme et retourne y_true et y_prob."""
    model.eval()
    val_probs, val_targets = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_sarc"].numpy()

            if is_multitask:
                logits_sarc, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                logits_sarc = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(logits_sarc, dim=-1)[:, 1].cpu().numpy()
            val_probs.extend(probs)
            val_targets.extend(labels)

    return np.array(val_targets), np.array(val_probs)


def train_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    weights_sarc: torch.Tensor,
    is_multitask: bool = False,
    lam: float = 0.5,
    lr: float = 2e-5,
    epochs: int = 4,
    patience: int = 2,
    weight_decay: float = 0.01,
) -> dict:
    """Entraîne un modèle Transformer (Mono ou Multi-tâche) avec fp16 et early stopping sur val F1-sarc."""
    model.to(device)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.10 * total_steps)

    # Paramètres avec et sans weight decay
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    ce_sarc = nn.CrossEntropyLoss(weight=weights_sarc.to(device))
    ce_sent = nn.CrossEntropyLoss()

    use_cuda = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    best_f1 = -1.0
    best_state = None
    best_metrics = {}
    stagnation = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y_sarc = batch["label_sarc"].to(device)
            y_sent = batch["label_sent"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_cuda):
                if is_multitask:
                    logits_sarc, logits_sent = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = ce_sarc(logits_sarc, y_sarc) + lam * ce_sent(logits_sent, y_sent)
                else:
                    logits_sarc = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = ce_sarc(logits_sarc, y_sarc)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        # Évaluation sur val
        val_targets, val_probs = evaluate_transformer(model, val_loader, device, is_multitask=is_multitask)
        best_s, best_f1_epoch = find_best_threshold(val_targets, val_probs)
        pr_auc = float(average_precision_score(val_targets, val_probs))
        f1_05 = float(f1_score(val_targets, val_probs >= 0.5, zero_division=0))
        acc = float(accuracy_score(val_targets, val_probs >= best_s))

        print(
            f"Epoch {epoch:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | "
            f"PR-AUC: {pr_auc:.4f} | Val F1 sarc: {best_f1_epoch:.4f} (seuil {best_s}) | F1@0.5: {f1_05:.4f}"
        )

        if best_f1_epoch > best_f1:
            best_f1 = best_f1_epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "seuil": best_s,
                "f1_sarc_val": best_f1_epoch,
                "f1_sarc_val_seuil_0_5": round(f1_05, 4),
                "pr_auc_val": round(pr_auc, 4),
                "accuracy_val_seuil": round(acc, 4),
                "epoch": epoch,
            }
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= patience:
                print(f"Early stopping déclenché à l'époque {epoch} (meilleur F1: {best_f1:.4f})")
                break

    temps_total = round(time.time() - t0, 2)
    if best_state is not None:
        model.load_state_dict(best_state)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best_metrics["temps_entrainement_s"] = temps_total
    best_metrics["params_entraines"] = n_params
    return best_metrics
