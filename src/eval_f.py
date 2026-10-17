import os
import re
import emoji
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from scipy.stats import chi2

def compute_f1_curve(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """Calcule F1 = f(seuil) sur toute la plage de seuils possibles."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.02)
    f1_scores = []
    for s in thresholds:
        pred = (y_prob >= s).astype(int)
        f1_scores.append(f1_score(y_true, pred, zero_division=0))
    return thresholds, np.array(f1_scores)


def plot_prior_shift_curve(
    y_test_true: np.ndarray,
    y_test_prob: np.ndarray,
    seuil_val: float,
    model_name: str,
    save_path: str = "results/prior_shift_f1_curve.png",
):
    """Trace et sauvegarde la figure officielle du compte-rendu : F1 = f(seuil) sur le test (F1)."""
    thresholds, f1_scores = compute_f1_curve(y_test_true, y_test_prob)
    best_idx = np.argmax(f1_scores)
    best_s_test = thresholds[best_idx]
    best_f1_test = f1_scores[best_idx]

    # F1 obtenu sur test avec le seuil val (sans tricher)
    pred_val_thresh = (y_test_prob >= seuil_val).astype(int)
    f1_at_val_thresh = f1_score(y_test_true, pred_val_thresh, zero_division=0)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, color="#1f77b4", lw=2.5, label="F1(seuil) sur le test officiel")
    
    # Ligne verticale seuil val
    plt.axvline(x=seuil_val, color="#d62728", linestyle="--", lw=2,
                label=f"Seuil choisi sur val = {seuil_val:.2f} (F1 test = {f1_at_val_thresh:.4f})")
    
    # Point seuil optimal a posteriori sur test
    plt.scatter([best_s_test], [best_f1_test], color="#2ca02c", s=100, zorder=5,
                label=f"Seuil optimal test a posteriori = {best_s_test:.2f} (F1 = {best_f1_test:.4f})")

    plt.title(f"F1. Décalage de Prior — Courbe F1 = f(seuil) sur le test ({model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Seuil de décision", fontsize=11)
    plt.ylabel("F1 de la classe sarcastique", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10, loc="lower center")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Figure sauvegardée avec succès : {save_path}")
    plt.show()
    return f1_at_val_thresh, best_s_test, best_f1_test


def ablate_sarcasm_markers(text: str) -> str:
    """Supprime emojis, rires répétés et ponctuations répétées (F4)."""
    # 1. Suppression emojis
    text_clean = emoji.replace_emoji(str(text), replace="")
    # 2. Suppression des rires arabes (au moins 2 'ه')
    text_clean = re.sub(r"ه{2,}", " ", text_clean)
    # 3. Réduction des ponctuations répétées (!, ?, ؟, .)
    text_clean = re.sub(r"[!?؟\.]{2,}", " ", text_clean)
    return " ".join(text_clean.split())


def bootstrap_f1_ci(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, n_bootstraps: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    """Calcule l'intervalle de confiance à 95% par bootstrap (1 000 rééchantillonnages) pour F5."""
    rng = np.random.RandomState(seed)
    f1_boot = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        y_t = y_true[idx]
        pred = (y_prob[idx] >= threshold).astype(int)
        f1_boot.append(f1_score(y_t, pred, zero_division=0))

    ci_low = np.percentile(f1_boot, 100 * (alpha / 2))
    ci_high = np.percentile(f1_boot, 100 * (1 - alpha / 2))
    f1_mean = np.mean(f1_boot)
    return round(float(f1_mean), 4), round(float(ci_low), 4), round(float(ci_high), 4)


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> tuple[int, int, float, float]:
    """Test de McNemar avec correction de continuité pour comparer deux modèles (F5).
    b : Modèle A correct et Modèle B incorrect
    c : Modèle A incorrect et Modèle B correct"""
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return b, c, 0.0, 1.0

    # Statistique de McNemar avec correction d'Edward
    stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
    p_value = 1.0 - chi2.cdf(stat, df=1)
    return b, c, round(float(stat), 4), float(p_value)


def save_official_predictions(
    model_slug: str,
    seed: int,
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    out_dir: str = "predictions",
):
    """Enregistre les prédictions officielles au format exigé :
    predictions/<modèle>_seed<k>.csv (id, y_true, y_pred, probabilités)"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_slug}_seed{seed}.csv")
    df_out = pd.DataFrame({
        "id": df_test["id"].values if "id" in df_test else range(len(df_test)),
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilite_sarcasme": np.round(probabilities, 4),
    })
    df_out.to_csv(path, index=False)
    print(f"Prédictions officielles enregistrées dans : {path}")


def classifier_30_erreurs(
    df_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    df_train_brut: pd.DataFrame = None,
    save_path: str = "results/analyse_30_erreurs.csv",
) -> pd.DataFrame:
    """Extrait 15 FN et 15 FP et les classe dans les 5 catégories imposées (F6 / Q12) :
    1. Connaissance du monde / contexte requis
    2. Ironie sans marqueur
    3. Marqueur mal interprété
    4. Dialecte
    5. Bruit d'annotation (croisé avec doublons contradictoires)
    """
    df = df_test.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["prob"] = np.round(y_prob, 4)

    # 1. Identifier les doublons contradictoires du train (4.1)
    contra_keys = set()
    if df_train_brut is not None:
        from src.preprocess import cle_dedup
        keys = df_train_brut["tweet"].map(cle_dedup)
        nuniq = df_train_brut.groupby(keys)["sarcasm"].nunique()
        contra_keys = set(nuniq[nuniq > 1].index)

    fn = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].head(15).copy()
    fp = df[(df["y_true"] == 0) & (df["y_pred"] == 1)].head(15).copy()

    fn["type_erreur"] = "Faux Négatif (Sarcasme manqué)"
    fp["type_erreur"] = "Faux Positif (Fausse alerte)"

    def categoriser(row):
        t = str(row["tweet"])
        t_clean = ablate_sarcasm_markers(t)
        has_markers = (t != t_clean)
        dialect = str(row.get("dialect", "")).lower()

        # A. Bruit d'annotation (croisement doublons)
        from src.preprocess import cle_dedup
        if cle_dedup(t) in contra_keys:
            return "Bruit d'annotation", "Tweet identique ou similaire annoté de façon contradictoire dans le train."

        # B. Marqueur mal interprété (FP avec rire/emoji, ou FN trompé par un faux marqueur)
        if row["type_erreur"].startswith("Faux Positif") and has_markers:
            return "Marqueur mal interprété", "Présence d'emojis ou rires expressifs ayant induit une fausse alerte sarcastique."

        # C. Ironie sans marqueur (FN sobre, antiphrase pure)
        if row["type_erreur"].startswith("Faux Négatif") and not has_markers:
            return "Ironie sans marqueur", "Sarcasme formulé sobrement par antiphrase, sans aucun emoji ni rire explicite."

        # D. Dialecte (idiomes régionaux)
        if dialect in ["egypt", "gulf", "levant", "magreb"] and any(w in t for w in ["مش", "كده", "عشان", "ايه", "ليه", "شو", "هيك", "واش", "برشا", "شكو", "حق"]):
            return "Dialecte", f"Tournure idiomatique ou argot dialectal ({dialect}) masquant l'intention réelle."

        # E. Connaissance du monde / contexte requis
        return "Connaissance du monde / contexte requis", "Nécessite la connaissance d'un contexte politique, culturel ou sportif externe."

    rows = []
    for _, r in pd.concat([fn, fp]).iterrows():
        cat, expl = categoriser(r)
        r_dict = r.to_dict()
        r_dict["categorie"] = cat
        r_dict["explication"] = expl
        rows.append(r_dict)

    res_df = pd.DataFrame(rows)
    cols = ["id", "type_erreur", "categorie", "dialect", "sentiment", "prob", "explication", "tweet"]
    cols_exist = [c for c in cols if c in res_df.columns]
    res_df = res_df[cols_exist]

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        res_df.to_csv(save_path, index=False)
        print(f"30 erreurs classées et sauvegardées avec succès dans : {save_path}")

    return res_df

