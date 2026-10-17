import os
import pandas as pd

CHEMIN = "results/results.csv"
COLONNES = [
    "modele", "preprocessing", "graine", "seuil", "pr_auc_val", "f1_sarc_val",
    "f1_sarc_val_seuil_0_5", "accuracy_val_seuil", "n_train", "n_val",
    "temps_entrainement_s", "params_entraines", "versions", "note",
]

def ajouter_resultat(ligne: dict, chemin: str = CHEMIN) -> None:
    """Ajoute ou met à jour une ligne en respectant l'ordre des COLONNES. Champs absents = vides."""
    inconnues = set(ligne) - set(COLONNES)
    if inconnues:
        raise ValueError(f"colonnes inconnues : {inconnues}")
    nouvelle_ligne = pd.DataFrame([ligne]).reindex(columns=COLONNES)
    os.makedirs(os.path.dirname(chemin), exist_ok=True)
    if os.path.exists(chemin) and os.path.getsize(chemin) > 0:
        existant = pd.read_csv(chemin)
        masque = (
            (existant["modele"] == ligne.get("modele")) &
            (existant["preprocessing"] == ligne.get("preprocessing")) &
            (existant["graine"] == ligne.get("graine"))
        )
        if masque.any():
            existant = existant[~masque]
        df_final = pd.concat([existant, nouvelle_ligne], ignore_index=True)
    else:
        df_final = nouvelle_ligne
    df_final.to_csv(chemin, index=False)