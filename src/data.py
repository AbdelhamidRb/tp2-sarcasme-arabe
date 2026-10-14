import pandas as pd
from sklearn.model_selection import train_test_split

BASE = "https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/"

def load_splits(seed: int = 42):
    full = pd.read_csv(BASE + "training_data.csv")
    test = pd.read_csv(BASE + "testing_data.csv")  # scellé jusqu'à la Partie F
    full["id"], test["id"] = range(len(full)), range(len(test))
    train, val = train_test_split(
        full, test_size=0.10, random_state=seed, stratify=full["sarcasm"]
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test