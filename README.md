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

| Modèle | Prétraitement | Seuil val | PR-AUC val | F1 sarc (val) | F1@0.5 | Temps (s) | Params | Note |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **TF-IDF + LogisticRegression** | dedup train + espaces norm | 0.42 | **0.5217** | 0.5153 | 0.5041 | 4.81s | 186k | Baseline officielle (`class_weight='balanced'`) |
| **mDeBERTa NLI zero-shot (FR)** | dedup train | 0.75 | 0.3495 | 0.4119 | - | - | - | Formulation NLI française retenue |
| **CAMeL-Lab sentiment** | dedup train | - | - | 0.4656 | - | - | - | Règle négatif $\Rightarrow$ sarcastique (contamination possible) |
| **C1. BiLSTM (Emb aléatoire)** | vocab min_freq=2 | 0.26 | 0.4230 | 0.4453 | 0.3563 | 22.6s | 5.35M | Initialisation aléatoire |
| **C1. BiLSTM (AraVec figé)** | vocab min_freq=2 + clean_str | 0.12 | 0.4699 | 0.4969 | 0.3028 | 13.7s | **441k** | AraVec feature extraction |
| **C1. BiLSTM (AraVec dégelé)** | vocab min_freq=2 + clean_str | 0.18 | 0.5026 | 0.5074 | 0.3000 | 17.6s | 5.35M | Fine-tuning AraVec |
| **C2. BiLSTM + Attention custom** | vocab min_freq=2 + clean_str | 0.26 | 0.4910 | 0.5099 | 0.4084 | 12.1s | 5.37M | Pooling attentionnel du cours |
| **C3. Mini-Transformer (avec pos)** | vocab min_freq=2 + clean_str | 0.28 | 0.4818 | 0.4963 | 0.3942 | 12.6s | 5.22M | Transformer 2 couches, 4 têtes |
| **C3. Mini-Transformer (SANS pos)** | vocab min_freq=2 + clean_str | 0.26 | 0.4876 | **0.5170** | 0.4022 | 12.3s | 5.22M | **Ablation encodage positionnel** |
| **C2. AttentionBiLSTM (Perte pondérée)** | vocab min_freq=2 + clean_str | 0.60 | 0.4680 | 0.5106 | **0.4764** | 18.5s | 5.37M | Poids classe = 4.81 |
| **C2. AttentionBiLSTM (Sur-échantillon)**| vocab min_freq=2 + clean_str | 0.22 | 0.4489 | 0.4906 | 0.4576 | 23.9s | 5.37M | Train équilibré 50/50 |

---

## 5. Règles méthodologiques strictes
- **Test scellé :** `testing_data.csv` n'est jamais consulté pour le choix d'hyperparamètres, de seuils ou d'architectures.
- **Ajustement sur train seul :** Toute transformation statistique (TF-IDF, vocabulaires, pondérations) est ajustée exclusivement sur `train`.
- **Reproductibilité :** Graine fixée à 42 pour l'ensemble des modules.
