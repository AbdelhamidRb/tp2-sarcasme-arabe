# TP 2 — Détection du sarcasme en arabe : déséquilibre, multi-tâche et généralisation

**Module :** Transformers & LLMs  
**Enseignant :** Prof. Yassir MATRANE  
**Étudiant :** Abdelhamid Rherib ([@AbdelhamidRb](https://github.com/AbdelhamidRb))  
**Corpus :** ArSarcasm-v2 (WANLP 2021)

---

## 1. Description du projet

Ce projet a pour objectif de concevoir, comparer et évaluer des systèmes de détection du sarcasme sur des tweets en langue arabe (corpus **ArSarcasm-v2**). Le problème se caractérise par :
- Un **déséquilibre de classes** (~17 % de sarcastique en train/val, ~27 % en test).
- La nécessité de préserver les indices subtils du sarcasme (emojis, ponctuation répétée, allongements, rires arabes comme `هههه`).
- L'exploration de 5 familles d'approches :
  1. Baseline TF-IDF + Régression Logistique.
  2. PLM zero-shot sans entraînement (NLI multilingue et transfert par heuristique de sentiment).
  3. Embeddings statiques (AraVec) + réseaux profonds (BiLSTM, BiLSTM avec attention custom, mini-Transformer).
  4. PLM gelé (AraBERT-twitter) comme extracteur de représentations.
  5. Fine-tuning complet mono-tâche et multi-tâche (AraBERT-twitter vs MARBERTv2).
- Une évaluation en conditions réalistes de production : décalage de prior, dialecte non vu (retrait de l'égyptien), robustesse aux raccourcis d'apprentissage, significativité (bootstrap, McNemar) et analyse d'erreurs.

---

## 2. Structure du dépôt

```text
tp2-sarcasme-arabe/
├── AGENT.md                 # Contexte d'exécution et journal de bord de l'agent
├── README.md                # Documentation et guide de reproduction
├── requirements.txt         # Dépendances Python
├── src/
│   ├── __init__.py
│   ├── seed.py              # Fixation des graines aléatoires (torch, numpy, random)
│   ├── data.py              # Téléchargement et split stratifié 90/10 (seed 42)
│   ├── preprocess.py        # Normalisation et déduplication rigoureuse de train
│   └── results.py           # Journalisation standardisée dans results/results.csv
├── notebooks/
│   ├── 01_audit_texte.ipynb          # Audit du texte brut et identification des artefacts
│   ├── 02_partie_A_baseline.ipynb    # Exploration, classifieur trivial, baseline TF-IDF
│   ├── 03_partie_B_zero_shot.ipynb   # mDeBERTa NLI zero-shot et CAMeL-Lab sentiment
│   ├── 04_partie_C_embeddings.ipynb  # (À venir) AraVec, BiLSTM, Attention pooling, Transformer
│   ├── 05_partie_D_plm_gele.ipynb    # (À venir) AraBERT gelé + tête attention
│   ├── 06_partie_E_finetuning.ipynb  # (À venir) Fine-tuning mono/multi-tâche (MARBERT, AraBERT)
│   └── 07_partie_F_generalisation.ipynb # (À venir) Tests de généralisation et test officiel
├── results/
│   └── results.csv          # Tableau de bord des métriques sur validation et test
└── predictions/             # Fichiers de prédictions finaux formatés
```

---

## 3. Installation et Environnement

Le projet requiert **Python 3.10+** et un environnement avec GPU (ex. Google Colab T4).

```bash
git clone https://github.com/AbdelhamidRb/tp2-sarcasme-arabe.git
cd tp2-sarcasme-arabe
pip install -r requirements.txt
```

Dans Google Colab, chaque notebook clone automatiquement le dépôt et positionne le `sys.path`.

---

## 4. Résultats obtenus (Validation)

Métrique principale : **F1 de la classe sarcastique** ($F1_{\text{sarc}}$). L'accuracy n'est pas retenue (un classifieur trivial « jamais sarcastique » donne 82,7 % d'accuracy pour un F1 de 0).

| Modèle | Prétraitement | Seuil val | PR-AUC val | F1 sarc (val) | Note |
|---|---|:---:|:---:|:---:|---|
| **TF-IDF + LogisticRegression** | dedup train + espaces norm | 0.42 | **0.5217** | **0.5153** | Baseline officielle (`class_weight='balanced'`) |
| **mDeBERTa NLI zero-shot (FR)** | dedup train | 0.75 | 0.3495 | 0.4119 | Formulation NLI française retenue |
| **CAMeL-Lab sentiment** | dedup train | - | - | 0.4656 | Règle négatif $\Rightarrow$ sarcastique (contamination possible) |

---

## 5. Règles méthodologiques strictes
- **Test scellé :** `testing_data.csv` n'est jamais consulté pour le choix d'hyperparamètres, de seuils ou d'architectures.
- **Ajustement sur train seul :** Toute transformation statistique (TF-IDF, vocabulaires, pondérations) est ajustée exclusivement sur `train`.
- **Reproductibilité :** Graine fixée à 42 pour l'ensemble des modules.
