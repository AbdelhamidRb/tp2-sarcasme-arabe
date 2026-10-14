import re
import pandas as pd

def normaliser_bords(s: str) -> str:
    """Artefact d'export : espace insécable et espaces de bord/multiples."""
    s = s.replace("\u00a0", " ")
    return re.sub(r"\s+", " ", s).strip()

def cle_dedup(s: str) -> str:
    """Clé de comparaison : sans URL ni mention, minuscules, espaces normalisés."""
    s = re.sub(r"https?://\S+|@\w+", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()

def dedupliquer_train(train: pd.DataFrame, test: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Applique la politique de déduplication. Ne touche qu'à train.
    Le test n'est lu que pour détecter les fuites (il n'est jamais modifié)."""
    train = train.copy()
    train["tweet"] = train["tweet"].map(normaliser_bords)
    train["cle"] = train["tweet"].map(cle_dedup)
    n0 = len(train)

    # 1. Fuite : textes de train présents dans test
    fuite = train["cle"].isin(set(test["tweet"].map(cle_dedup)))
    train = train[~fuite]

    # 2. Groupes aux labels contradictoires : supprimés entièrement
    nuniq = train.groupby("cle")["sarcasm"].nunique()
    contra = nuniq[nuniq > 1].index
    train = train[~train["cle"].isin(contra)]

    # 3. Doublons cohérents : une seule occurrence
    train = train.drop_duplicates(subset="cle", keep="first").reset_index(drop=True)

    if verbose:
        print(f"train avant : {n0}")
        print(f"  fuite retirée (présents dans test) : {int(fuite.sum())}")
        print(f"  groupes contradictoires retirés    : {len(contra)}")
        print(f"train après : {len(train)}")
    return train