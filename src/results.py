import os
import pandas as pd

CHEMIN = "results/results.csv"
COLONNES = [
    "modele", "preprocessing", "graine", "seuil", "pr_auc_val", "f1_sarc_val",
    "f1_sarc_val_seuil_0_5", "accuracy_val_seuil", "n_train", "n_val",
    "temps_entrainement_s", "params_entraines", "versions", "note",
]

def ajouter_resultat(ligne: dict, chemin: str = CHEMIN) -> None:
    """Ajoute une ligne en respectant l'ordre des COLONNES. Champs absents = vides."""
    inconnues = set(ligne) - set(COLONNES)
    if inconnues:
        raise ValueError(f"colonnes inconnues : {inconnues}")
    df = pd.DataFrame([ligne]).reindex(columns=COLONNES)
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    ecrire_entete = not os.path.exists(chemin) or os.path.getsize(chemin) == 0
    df.to_csv(chemin, mode="a", header=ecrire_entete, index=False)