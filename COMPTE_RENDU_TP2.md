# COMPTE-RENDU — TP 2
### Détection du sarcasme en arabe : déséquilibre, multi-tâche et généralisation
**Cours :** Transformers & LLMs — **Enseignant :** Prof. Yassir MATRANE  
**Dépôt GitHub officiel :** [https://github.com/AbdelhamidRb/tp2-sarcasme-arabe](https://github.com/AbdelhamidRb/tp2-sarcasme-arabe)

---

## Identité & Configuration

* **Binôme ou individuel :** Abdelhamid (`AbdelhamidRb`)
* **Filière / groupe :** Master 2 — Module Transformers & LLMs
* **Date :** Octobre 2026
* **Lien du dépôt Git :** [https://github.com/AbdelhamidRb/tp2-sarcasme-arabe](https://github.com/AbdelhamidRb/tp2-sarcasme-arabe)
* **PLM fine-tunés :** `aubmindlab/bert-base-arabertv02-twitter` & `UBC-NLP/MARBERTv2`
* **Prétraitement PLM utilisé :** Tokeniseurs WordPiece natifs sans racinisation ni lemmatisation agressive ; normalisation minimale des espaces (`\u00a0` $\rightarrow$ espace) et **préservation intégrale** des marqueurs expressifs (emojis, rires `هههه`, ponctuation répétée, allongements *tatweel*).

---

## Figures obligatoires (Page 5 du sujet)

### 1. Courbe $F1 = f(\text{seuil})$ sur le test officiel (Épreuve F1)
> *Figure générée sur les 3 000 tweets du test scellé illustrant l'impact du décalage de prior (+10 points de sarcasme entre validation et test).*

![Courbe F1 en fonction du seuil sur le test](results/prior_shift_f1_curve.png)

* **Seuil retenu sur validation (sans tricher) :** $s_{\text{val}} = 0{,}72 \implies \mathbf{F1_{\text{test}} = 0{,}6266}$ (Rappel = $65{,}41\%$, Précision = $60{,}13\%$).
* **Seuil optimal *a posteriori* sur le test :** $s_{\text{test}} = 0{,}57 \implies \mathbf{F1_{\text{max}} = 0{,}6428}$.

---

### 2. Attention personnalisée sur 5 tweets sarcastiques (Modèle D2)
> *Carte thermique des poids d'attention $\alpha_i$ de la tête BiLSTM + Attention custom au-dessus d'AraBERT-twitter gelé.*

![Visualisation de l'attention personnalisée D2](results/attention_d2_5_tweets.png)

* **Interprétation :** Les poids d'attention se focalisent de manière éclatante sur les marqueurs typiques de l'ironie : pic d'attention quasi-exclusif ($100\%$) sur la séquence de rire `ه ه ه ه ه ه ه` (Tweet 2), et concentration sur les prédicats de contradiction sémantique (`كوش`, `بيعت`, `للجزار`).

---

## Réponses aux Questions du Sujet

### Q1. Taux de sarcasme, classifieur trivial et choix de la métrique

#### A. Distribution du sarcasme dans le corpus
| Ensemble | Total tweets | Non sarcastiques (0) | Sarcastiques (1) | Taux de sarcasme (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Train (après dédup.)** | 11 217 | 9 286 | 1 931 | **17,21 %** |
| **Validation** | 1 255 | 1 038 | 217 | **17,29 %** |
| **Test officiel** | 3 000 | 2 179 | 821 | **27,37 %** *(+10,08 points)* |

#### B. Distribution par dialecte
* **Train :** `egypt` : 34,64 % (833 / 2 405) · `gulf` : 24,87 % (143 / 575) · `levant` : 21,20 % (120 / 566) · `magreb` : 39,47 % (15 / 38) · `msa` : 10,74 % (820 / 7 633).
* **Test :** `egypt` : 50,65 % (155 / 306) · `gulf` : 17,70 % (57 / 322) · `levant` : 29,79 % (14 / 47) · `magreb` : 0,00 % (0 / 2) · `msa` : 25,61 % (595 / 2 323).

#### C. Classifieur trivial « jamais sarcastique » (prédire 0 pour tout le monde)
* **Accuracy sur validation :** **82,71 %** (1 038 / 1 255).
* **F1-score de la classe sarcastique :** **0,0000** (Précision indéfinie, Rappel = 0).
* **Conclusion sur l'Accuracy :** L'accuracy est totalement trompeuse en présence de classes fortement déséquilibrées (paradoxe de l'accuracy). Un modèle prédisant systématiquement la classe majoritaire obtiendrait plus de 82 % de précision globale tout en étant rigoureusement aveugle et inutile en production.
* **Métrique retenue :** Le **F1-score de la classe sarcastique ($F1_{\text{sarc}}$)** est la métrique reine officielle (moyenne harmonique pénalisant sévèrement à la fois les faux positifs et les faux négatifs). Elle est complétée par la **PR-AUC (Precision-Recall Area Under Curve)**, qui mesure la qualité de calibration probabiliste du modèle indépendamment du choix d'un seuil binaire particulier.

---

### Q2. Raccourcis superficiels (*Shortcuts*) et validité des scores

#### A. Corrélation des indices superficiels avec le label (sur validation)
| Indice superficiel | Présence globale | Taux si présent | Taux si absent | Règle « Présent $\Rightarrow$ Sarc » | Règle « Absent $\Rightarrow$ Sarc » |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **URL (`http...`)** | 39,8 % | 8,22 % | 23,28 % | $F1 = 0{,}1145$ | $F1 = \mathbf{0{,}3618}$ |
| **Mention (`@user`)** | 24,0 % | 20,27 % | 16,35 % | $F1 = 0{,}2355$ | $F1 = 0{,}2664$ |
| **Espace insécable / bord** | 5,5 % | 40,58 % | 15,94 % | $F1 = 0{,}1958$ | $F1 = 0{,}2694$ |
| **Baseline TF-IDF (référence)** | — | — | — | — | $\mathbf{F1 = 0{,}5153}$ |

#### B. Conclusion sur la validité de nos scores
Aucune règle heuristique superficielle n'atteint les performances de notre baseline ($F1 = 0{,}5153$). L'absence d'URL présente une corrélation statistique notable (les tweets avec URL étant principalement des partages d'articles d'actualité au premier degré), mais ne constitue pas un raccourci suffisant. Nos modèles apprennent une véritable structure textuelle au-delà des biais de surface.

---

### Q3. Choix de prétraitement par famille de modèles et étude d'ablation

1. **TF-IDF (Baseline) :** Normalisation des espaces (`\u00a0`), déduplication stricte sans toucher au test. Extraction conjointe de n-grammes de mots (1–2) et de sous-mots/caractères (2–5, `char_wb`) avec `min_df=2` pour capter les racines et les variations morphologiques.
2. **Embeddings statiques (AraVec) :** Application de la fonction `clean_str` d'AraVec (unification des alifs, yaa et taa marbuta) strictement requise car les vecteurs Word2Vec Twitter d'AraVec ont été indexés sous cette forme canonique.
3. **Modèles pré-entraînés (AraBERT & MARBERTv2) :** Utilisation stricte des tokeniseurs WordPiece officiels. **Refus délibéré de toute lemmatisation ou racinisation (stemming)** :
   * **Chiffres à l'appui (Ablation F4) :** La suppression des emojis, rires et ponctuations répétées détruit immédiatement **6,33 %** des vrais positifs détectés (34 tweets sarcastiques irrémédiablement perdus). La racinisation détruirait en outre la morphologie des rires (`هههه`) et les allongements expressifs (*tatweel*), annihilant les indices intonatifs de l'ironie.

---

### Q4. Zero-shot par NLI (B1) et rôle de l'attention croisée

#### A. Scores obtenus selon la formulation de l'hypothèse
* **Arabe :** $\text{Template} = \text{"التغريدة هذه \{sarcastique/sérieuse\}"} \implies \text{PR-AUC} = 0{,}3375 \;; s = 0{,}82 \;; \mathbf{F1 = 0{,}3871}$.
* **Français :** $\text{Template} = \text{"Ce tweet est \{sarcastique/sérieux\}"} \implies \text{PR-AUC} = 0{,}3495 \;; s = 0{,}75 \;; \mathbf{F1 = 0{,}4119}$ *(Meilleur score)*.
* **Anglais :** $\text{Template} = \text{"This tweet is \{sarcastic/serious\}"} \implies \text{PR-AUC} = 0{,}3407 \;; s = 0{,}79 \;; \mathbf{F1 = 0{,}4040}$.

#### B. Pourquoi le NLI fonctionne sans exemple d'entraînement ?
Le modèle `mDeBERTa-v3` a été pré-entraîné massivement sur la tâche d'inférence textuelle multilingue (MNLI/XNLI) : déterminer si une prémisse implique (*entailment*), contredit ou est neutre vis-à-vis d'une hypothèse. En formulant l'hypothèse « Ce tweet est sarcastique », le problème de classification se transpose en un problème logique de compatibilité sémantique.

#### C. Rôle de l'attention prémisse-hypothèse
Dans l'architecture de DeBERTa, la prémisse et l'hypothèse sont concaténées au sein de la même séquence d'entrée : `[CLS] Tweet [SEP] Hypothèse [SEP]`. Dès les premières couches d'auto-attention (*cross-segment self-attention*), chaque token du tweet peut assister directement aux tokens de l'hypothèse (notamment le mot clé "sarcastique"). Cela permet au réseau de mesurer l'alignement discursif et affectif sans couche de classification supervisée préalable.

---

### Q5. Heuristique « Négatif $\Rightarrow$ Sarcastique » (B2) et contamination

* **Métriques obtenues :** Précision = **33,20 %** · Rappel = **77,88 %** · **F1-score = 0,4656** (part prédite positive : 40,56 %).
* **Relation sarcasme / sentiment :** Le sarcasme est un amplificateur de négativité : 41,91 % des tweets négatifs sont sarcastiques dans ce corpus, contre moins de 3 % pour les tweets positifs ou neutres. Cependant, la précision est faible (33,2 %) car la majorité des tweets négatifs expriment une critique directe sans ironie.
* **Contamination avérée :** Le modèle `CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment` a été fine-tuné sur les corpus ASTD, ArSAS et SemEval 2017. Or, ArSarcasm (v1) a précisément été construit en ré-annotant les tweets d'ASTD et SemEval 2017. **Le modèle a donc déjà rencontré une partie substantielle des tweets de test lors de son pré-entraînement**. Ce score de 0,4656 est donc contaminé et ne peut prétendre à une validité en situation réelle.

---

### Q6. Gestion du déséquilibre des classes (Partie C2)

#### A. Tableau comparatif des stratégies (Architecture C2 BiLSTM + Attention custom)
| Stratégie de déséquilibre | Seuil retenu sur val | $F1_{\text{val}}$ (au seuil retenu) | $F1_{\text{val}}$ (@ seuil 0,50) | PR-AUC val | Exactitude (Acc) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Perte pondérée ($w=[1{,}0 ; 4{,}81]$)** | $0{,}60$ | $\mathbf{0{,}5106}$ | $\mathbf{0{,}4764}$ | $0{,}4680$ | $0{,}7968$ |
| **Perte standard + seuil optimisé** | $0{,}26$ | $0{,}5099$ | $0{,}4084$ | $\mathbf{0{,}4910}$ | $0{,}7825$ |
| **Sur-échantillonnage de `train` (18 572)** | $0{,}22$ | $0{,}4906$ | $0{,}4576$ | $0{,}4489$ | $0{,}7618$ |

* **Observation clé :** La perte pondérée est la stratégie la plus élégante et la plus robuste : au seuil par défaut de $0{,}50$, elle atteint **$F1 = 0{,}4764$** (contre $0{,}4084$ pour la perte standard), car elle force le gradient à calibrer les logits de manière centrée. Le sur-échantillonnage dégrade le score ($0{,}4906$) par surapprentissage des répliques.
* **Pourquoi sur-échantillonner APRÈS le split train/val ?**  
  Si le sur-échantillonnage était effectué avant la séparation, des répliques identiques d'un même tweet sarcastique se trouveraient simultanément dans le jeu d'entraînement et dans le jeu de validation. Le modèle évaluerait alors sa capacité à restituer par cœur des exemples mémorisés (*data leakage*), rendant la validation frauduleusement optimiste et invalidant toute sélection de modèle.

---

### Q7. Comparaison des architectures profondes PyTorch (C1 vs C2 vs C3)

#### A. Tableau synthétique des résultats
| Modèle | Embedding / Configuration | Seuil val | PR-AUC val | $F1_{\text{val}}$ optimal | $F1_{\text{val}}$ (@ 0,5) | Paramètres |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **C1. BiLSTM (max-pool)** | Aléatoire (scratch) | $0{,}26$ | $0{,}4230$ | $0{,}4453$ | $0{,}3563$ | 5,35 M |
| **C1. BiLSTM (max-pool)** | AraVec 300d figé | $0{,}12$ | $0{,}4699$ | $0{,}4969$ | $0{,}3028$ | **441 k** |
| **C1. BiLSTM (max-pool)** | AraVec 300d dégelé (fine-tuné) | $0{,}18$ | $0{,}5026$ | $0{,}5074$ | $0{,}3000$ | 5,35 M |
| **C2. BiLSTM + Attention custom** | AraVec 300d dégelé | $0{,}26$ | $0{,}4910$ | $\mathbf{0{,}5099}$ | $\mathbf{0{,}4084}$ | 5,37 M |
| **C3. Mini-Transformer (2 couches)** | Avec encodage positionnel | $0{,}28$ | $0{,}4818$ | $0{,}4963$ | $0{,}3942$ | 5,22 M |
| **C3. Mini-Transformer (Ablation)** | **SANS encodage positionnel** | $0{,}26$ | $0{,}4876$ | $\mathbf{0{,}5170}$ | $0{,}4022$ | 5,22 M |

#### B. Concentration de l'attention et Effet de l'encodage positionnel
1. **Poids d'attention :** L'attention personnalisée (C2) apporte un bond qualitatif majeur au seuil opérationnel ($F1@0{,}5$ passe de $0{,}3000$ à $0{,}4084$). Les poids se concentrent nettement sur les rires répétés et les ponctuations expressives.
2. **Ablation de la position (C3) :** Retirer l'encodage positionnel sinusoïdal **améliore les performances (+0,0207 de F1, atteignant 0,5170)**. Dans un tweet court de 15 mots, le sarcasme fonctionne par accumulation d'indices lexicaux isolés (*bag-of-cues*) plutôt que par dépendance syntaxique rigide. L'encodage positionnel introduit une contrainte superflue qui favorise le surapprentissage sur de petits corpus.

---

### Q8. PLM gelé (Partie D) vs Fine-tuné (Partie E1) : Analyse gain / coût

| Approche | Architecture | Seuil val | $F1_{\text{val}}$ | Temps d'entraînement | Paramètres entraînés |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **D1 (PLM gelé)** | AraBERT `[CLS]` + LogisticRegression | $0{,}64$ | $0{,}6166$ | **9,44 s** | **769** |
| **D1 (PLM gelé)** | AraBERT `Mean-Pooling` + LogReg | $0{,}66$ | $0{,}6138$ | **4,25 s** | **769** |
| **D2 (PLM gelé)** | AraBERT gelé + BiLSTM + Attention custom | $0{,}32$ | $0{,}6652$ | 190,29 s | 936 k |
| **E1 (Fine-tuning)** | **AraBERT-twitter mono-tâche** | $0{,}50$ | $\mathbf{0{,}6797}$ | 159,36 s | 135 M |
| **E1 (Fine-tuning)** | **MARBERTv2 mono-tâche** | $0{,}64$ | $0{,}6694$ | 239,62 s | 162 M |

* **Gain et Coût :** Le fine-tuning complet d'AraBERT (E1) surpasse le modèle gelé D2 de **+1,45 point de F1** et offre une calibration parfaite (seuil naturel à $0{,}50$). Cependant, ce gain s'accompagne d'un coût calculatoire multiplié par 144 en nombre de paramètres rétropropagés (135 M contre 936 k).
* **AraBERT-twitter vs MARBERTv2 en mono-tâche :** AraBERT-twitter domine légèrement MARBERTv2 en mono-tâche sur validation ($0{,}6797$ vs $0{,}6694$) car son pré-entraînement intègre un pré-traitement textuel très spécifique aux abréviations et hashtags Twitter du corpus ArSarcasm.

---

### Q9. Apprentissage Multi-tâche (E2) et Dynamique du paramètre $\lambda$

#### A. Résultats de l'exploration de $\lambda$ (Encodeur partagé MARBERTv2)
* **Mono-tâche MARBERTv2 (Référence E1) :** $F1_{\text{val}} = 0{,}6694$ (seuil $0{,}64$).
* **Multi-tâche E2 ($\lambda = 0{,}3$) :** $\mathbf{F1_{\text{val}} = 0{,}6812}$ (seuil $0{,}72$) $\implies$ **Nouveau record absolu (+1,18 point)**.
* **Multi-tâche E2 ($\lambda = 0{,}5$) :** $F1_{\text{val}} = 0{,}6708$ (seuil $0{,}54$).
* **Multi-tâche E2 ($\lambda = 1{,}0$) :** $F1_{\text{val}} = 0{,}6667$ (seuil $0{,}34$) $\implies$ **Dégradation sous le niveau mono-tâche !**

#### B. Pourquoi une tâche auxiliaire peut aider... ou nuire ?
1. **Effet bénéfique ($\lambda = 0{,}3$) :** Le sarcasme est une discordance pragmatique entre le sentiment apparent et le sens profond. En contraignant l'encodeur à modéliser conjointement le sentiment (3 classes), la tâche auxiliaire injecte un signal affectif riche qui régularise les représentations de `[CLS]`.
2. **Effet néfaste ($\lambda = 1{,}0$) :** Lorsque le poids de la perte auxiliaire est trop élevé, on observe un phénomène d'**interférence de tâche (*task conflict / gradient competition*)**. Les gradients de la classification du sentiment dominent l'optimiseur AdamW et tirent les couches de l'encodeur dans une direction sous-optimale pour la détection spécifique du sarcasme.

---

### Q10. Décalage de prior (*Prior Shift*, Épreuve F1) et stratégie de production

* **Écart mesuré :** Le seuil choisi sur validation ($s_{\text{val}} = 0{,}72$) donne **$F1 = 0{,}6266$** sur le test, alors que le seuil optimal *a posteriori* est de **$s_{\text{test}} = 0{,}57$** ($F1 = 0{,}6428$).
* **Explication :** La proportion de tweets sarcastiques est de **17,29 %** en validation contre **27,37 %** dans le test (+58 % d'augmentation relative). Le modèle, calibré sur un environnement plus rare en sarcasme, s'est montré trop conservateur au seuil de $0{,}72$.
* **Dispositif préconisé en production :**
  1. *Monitoring de dérive (Covariate / Prior Shift)* : Suivi de la distribution des probabilités prédites via le test statistique de Kolmogorov-Smirnov sur des fenêtres glissantes de 1 000 tweets.
  2. *Recalibration non supervisée du seuil* : Estimation du prior réel par l'algorithme Espérance-Maximisation (Saerens et al., 2002) et ajustement bayésien immédiat du seuil de décision :
     $$s_{\text{prod}} = \frac{p_{\text{prod}}}{p_{\text{val}}} \cdot \frac{1 - p_{\text{val}}}{1 - p_{\text{prod}}} \cdot s_{\text{val}}$$

---

### Q11. Généralisation sur dialecte non vu (F2) et cas du Maghreb

* **Chute mesurée sur l'égyptien non vu :**
  - Modèle avec égyptien dans train : $F1_{\text{egypt}} = 0{,}7616$.
  - Modèle **sans aucun tweet égyptien** (-2 405 tweets) : $F1_{\text{egypt}} = \mathbf{0{,}7715}$ ($\Delta = -0{,}0099$).
  - **Analyse :** MARBERTv2 a été pré-entraîné sur 1 milliard de tweets arabes variés ; il possède intrinsèquement les marqueurs dialectaux égyptiens. Le retrait des données de train évite un surapprentissage thématique sans dégrader la compréhension dialectale.
* **Représentation du dialecte maghrébin (`magreb`) :**
  - **Train :** **38 tweets** (0,34 % du corpus).
  - **Test :** **2 tweets** (0,07 % du corpus).
* **Évaluation honnête avant déploiement au Maroc :**
  Le corpus ArSarcasm-v2 est virtuellement aveugle au dialecte maghrébin/marocain (darija). Déployer ce modèle au Maroc sur la seule base de ces scores serait une faute d'ingénierie majeure. Avant tout déploiement :
  1. Constituer un corpus d'évaluation dédié de 2 000 tweets en darija marocaine labellisés par des locuteurs natifs.
  2. Adopter un modèle intégrant un pré-entraînement maghrébin ciblé (ex: `DarijaBERT` ou `DziriBERT`).

---

### Q12. Analyse qualitative de 30 erreurs (F6) et Décision de production

#### A. Répartition des 30 erreurs analysées ([`results/analyse_30_erreurs.csv`](results/analyse_30_erreurs.csv))
* **1. Marqueur mal interprété (15/30, 50 %) :** Concerne **100 % des faux positifs**. Des tweets humoristiques ou bienveillants contenant des rires répétés (`هههه`) ou des points d'exclamation ont déclenché une fausse alerte sarcastique.
* **2. Connaissance du monde / contexte requis (9/30, 30 %) :** Faux négatifs portant sur l'actualité politique (procès de juges, ministres) ou sportive (arbitrage, entraîneurs Sami Al-Jaber, Pitorca) indétectables sans contexte historique externe.
* **3. Ironie sans marqueur (5/30, 16,7 %) :** Faux négatifs où le sarcasme est un euphémisme ou une antiphrase sobre formulée au premier degré.
* **4. Dialecte (1/30, 3,3 %) :** Expression argotique de football non comprise.

---

## Décision finale & Matrice de confusion (Partie G)

### Modèle retenu pour la production : **MARBERTv2 Multi-tâche ($\lambda = 0{,}3$, seuil $s = 0{,}72$)**

#### A. Matrice de Confusion sur le Test officiel (3 000 tweets)
![Matrice de confusion officielle](results/matrice_confusion_test.png)

| Vérité terrain \ Prédiction | Prédit Normal (0) | Prédit Sarcasme (1) | Total Réel |
| :--- | :---: | :---: | :---: |
| **Vrai Normal (0)** | **1 823** (VN) | **356** (FP) | 2 179 ($72{,}63\%$) |
| **Vrai Sarcasme (1)** | **284** (FN) | **537** (VP) | 821 ($27{,}37\%$) |
| **Total Prédit** | 2 107 | 893 | **3 000** |

* **Exactitude globale (Accuracy) :** **78,67 %**
* **Précision sarcasme :** **60,13 %** ($537 / 893$)
* **Rappel sarcasme :** **65,41 %** ($537 / 821$)
* **F1-score officiel retenu :** $\mathbf{0{,}6266}$ (Intervalle de confiance à 95 % : $[0{,}5991 \;;\; 0{,}6528]$).

#### B. Justification multicritère
1. **Qualité & Record :** Meilleur score absolu sur validation ($0{,}6812$) et excellente tenue sur le test ($0{,}6266$).
2. **Robustesse prouvée (F4) :** Résistance de **93,67 %** lors de l'ablation totale des emojis, rires et ponctuations.
3. **Invariance cross-dialectale (F2) :** Maintien d'un F1 de $0{,}7715$ sur l'égyptien même lorsque celui-ci est complètement retiré de l'entraînement.
4. **Alternative à faible coût :** En environnement contraint (CPU pur, latence < 2 ms), le modèle **D1 (AraBERT gelé `[CLS]` + Régression Logistique)** offre une alternative remarquable ($F1 = 0{,}6166$ sur val, 769 paramètres, inférence instantanée).

---

## Limites & Pistes d'amélioration

1. **Dépendance au contexte extra-textuel :** 30 % des erreurs proviennent du manque de connaissances factuelles du monde réel. Une piste prometteuse consisterait à enrichir les tweets via une architecture RAG (*Retrieval-Augmented Generation*) connectée à l'actualité ou Wikipédia arabe.
2. **Représentation des dialectes minoritaires :** Le modèle reste inadapté au Maghreb en raison de la quasi-absence de données d'entraînement (38 tweets). Un sur-échantillonnage dialectal ciblé et l'emploi d'adaptateurs légers (*LoRA / Dialect Adapters*) constitueraient les prochaines étapes de recherche.
