import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from src.models_d import mean_pooling
from src.train_c import find_best_threshold

def extract_arabert_features(
    arabert_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pooling: str = "cls",
) -> tuple[np.ndarray, np.ndarray]:
    """Extrait les représentations vectorielles d'AraBERT gelé (768d).
    pooling : 'cls' pour last_hidden_state[:, 0, :] ou 'mean' pour la moyenne pondérée."""
    arabert_model.eval()
    arabert_model.to(device)
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].numpy()

            outputs = arabert_model(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state

            if pooling == "cls":
                # Token [CLS] à l'indice 0
                feats = hidden_states[:, 0, :].cpu().numpy()
            elif pooling == "mean":
                feats = mean_pooling(hidden_states, attention_mask).cpu().numpy()
            else:
                raise ValueError(f"Pooling inconnu : {pooling}")

            all_features.append(feats)
            all_labels.append(labels)

    return np.vstack(all_features), np.concatenate(all_labels)


def train_d2_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 10,
    patience: int = 3,
) -> dict:
    """Entraîne la tête D2 (BiLSTM + Attention) sur AraBERT gelé avec early stopping sur val F1-sarc."""
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
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Évaluation sur val
        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].numpy()

                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(labels)

        val_targets = np.array(val_targets)
        val_probs = np.array(val_probs)

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
            # On ne sauvegarde que les poids entraînables (la tête, pas AraBERT)
            trainable_state = {k: v.cpu().clone() for k, v in model.state_dict().items() if "arabert" not in k}
            best_state = trainable_state
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
        model.load_state_dict(best_state, strict=False)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    best_metrics["temps_entrainement_s"] = temps_total
    best_metrics["params_entraines"] = n_params
    return best_metrics


def plot_attention_visualization(
    model: nn.Module,
    tokenizer,
    tweets_list: list[str],
    device: torch.device,
    save_path: str = "results/attention_d2_5_tweets.png",
):
    """Génère la figure officielle de l'attention sur 5 tweets sarcastiques demandée pour le rapport."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        def reshape_ar(t):
            return get_display(arabic_reshaper.reshape(t))
    except ImportError:
        def reshape_ar(t):
            return t

    model.eval()
    fig, axes = plt.subplots(len(tweets_list), 1, figsize=(14, 2.5 * len(tweets_list)))
    if len(tweets_list) == 1:
        axes = [axes]

    for idx, (tweet, ax) in enumerate(zip(tweets_list, axes)):
        encoded = tokenizer(
            tweet,
            return_tensors="pt",
            truncation=True,
            max_length=48,
            padding="max_length",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            _, weights = model(input_ids=input_ids, attention_mask=attention_mask, return_attention=True)

        w = weights[0].cpu().numpy()  # [seq_len]
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

        # Filtrer [PAD]
        valid_indices = [i for i, mask_val in enumerate(attention_mask[0].cpu().numpy()) if mask_val == 1]
        tokens_valides = [tokens[i] for i in valid_indices]
        poids_valides = [w[i] for i in valid_indices]

        # Normaliser pour la visualisation
        poids_norm = np.array(poids_valides)
        if poids_norm.max() > 0:
            poids_norm = poids_norm / poids_norm.max()

        labels = [reshape_ar(tok.replace("##", "")) for tok in tokens_valides]

        sns.barplot(x=list(range(len(labels))), y=poids_norm, ax=ax, palette="mako")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel("Poids normalisé")
        ax.set_title(f"Tweet Sarcastique #{idx+1} : Poids d'Attention D2", fontsize=12, fontweight="bold")
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure sauvegardée avec succès : {save_path}")
    plt.show()
