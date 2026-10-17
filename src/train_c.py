import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, f1_score, accuracy_score

def eval_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Retourne les vrais labels et les probabilités prédites pour la classe 1 (sarcasme)."""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].cpu().numpy()
            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            # Probabilité de la classe 1 (softmax sur les 2 logits)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(labels)
    return np.array(all_targets), np.array(all_preds)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Trouve le seuil qui maximise le F1 de la classe sarcastique."""
    seuils = np.arange(0.1, 0.91, 0.02)
    best_thresh = 0.5
    best_f1 = -1.0
    for s in seuils:
        pred = (y_prob >= s).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = s
    return round(float(best_thresh), 2), round(float(best_f1), 4)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 15,
    patience: int = 4,
) -> dict:
    """Entraîne un modèle PyTorch avec early stopping sur le F1-sarcastique validation."""
    model.to(device)
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
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Évaluation sur val
        y_val_true, y_val_prob = eval_model(model, val_loader, device)
        best_s, best_f1_epoch = find_best_threshold(y_val_true, y_val_prob)
        pr_auc = float(average_precision_score(y_val_true, y_val_prob))
        f1_05 = float(f1_score(y_val_true, y_val_prob >= 0.5, zero_division=0))
        acc = float(accuracy_score(y_val_true, y_val_prob >= best_s))

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
    # Recharger les meilleurs poids
    if best_state is not None:
        model.load_state_dict(best_state)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best_metrics["temps_entrainement_s"] = temps_total
    best_metrics["params_entraines"] = n_params
    return best_metrics
