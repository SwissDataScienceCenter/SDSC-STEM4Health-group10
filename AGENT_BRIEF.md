# Brief agent : Decode the Bioreactor (équipe 10), améliorer le score du leaderboard

Tu reprends un projet de hackathon déjà très avancé. Lis ce brief en entier avant de toucher à quoi que ce soit.
Réponds à l'utilisateur **en français** ; le code et les commentaires restent **en anglais**.

## 1. Mission
- Hackathon SDSC STEM4Health, projet 1 « Decode the Bioreactor » (J&J). Culture cellulaire en fed-batch.
- On prédit `vcd`, `tcd`, `substrate`, `byproduct` pour **12 lots de test** × 13 échantillons (156 lignes du template),
  **uniquement** à partir de l'état à t=0, du profil de feeds (tous les bolus) et du volume du réacteur (donné à chaque
  échantillon, y compris pour le test).
- Métrique du leaderboard : **MAPE** moyenne sur les 4 variables (erreur relative : les petites valeurs pèsent lourd).
- **Objectif : passer dans le top 3** (barre ≈ 7.343 au dernier relevé, Team 7 à 7.018, Team 04 à 7.142).
  **Meilleur score actuel : 7.464** (`submissions/blend_base.csv`).

## 2. Règles à respecter (non négociables)
- Validation honnête **par lot** : jamais les points d'un même lot à la fois en entraînement et en validation. Un lot
  de validation/test est prédit seulement à partir de t=0, des feeds et du volume, jamais de ses mesures intermédiaires.
- `input-data/Project 1 - DTB/` est en **lecture seule** : ne jamais copier, modifier ou commiter ces données.
- Tout le code va dans le repo `/home/renku/work/SDSC-STEM4Health-group10` (le scratchpad /tmp est effacé aux
  redémarrages). Gros fichiers : `team-workspace/`.
- **Demander avant d'installer un paquet** et **avant tout git commit / push**.
- **Pas de sondage du leaderboard** : ne pas ajuster des prédictions individuelles d'après les retours du
  leaderboard. On choisit tout en validation croisée ; le leaderboard sert seulement à confirmer.
- C'est l'utilisateur qui soumet (notebook `Leaderboard_template.ipynb` à la racine du repo : mettre
  `SUBMISSION_PATH = "submissions/<fichier>.csv"` dans la cellule 2 puis Run All, `user="Team 10"`). Ne modifie pas
  ce notebook. Ton livrable est un CSV dans `submissions/` + le code qui le produit + la preuve de validation croisée.
- Ne parle pas de la présentation tant qu'on n'est pas top 3 : l'utilisateur veut du score.

## 3. Environnement
- Python : `/home/renku/work/.venv/bin/python` (pandas, numpy, scipy, scikit-learn, lightgbm, xgboost, torch,
  matplotlib installés). Lancer depuis la racine du repo, avec `PYTHONDONTWRITEBYTECODE=1`.
- Données : `src/data.py` (`load_raw()`, `measurements(raw, include_to_predict=...)`, `feeds_long(raw)`,
  `train_batches(raw)` = 49 lots, `test_batches(raw)` = 12 lots : Batch5, 6, 12, 24, 32, 35, 40, 48, 54, 56, 57, 60,
  `load_template()`).
- 16 cœurs : les validations croisées se lancent en parallèle en arrière-plan (`nohup ... &`, `OMP_NUM_THREADS=1`).
- Lis aussi `CLAUDE.md` (description des données) et `results/families/experiments_log.md`.

## 4. Protocole de validation (à reprendre tel quel)
- Plis : `evaluation.make_folds(train_ids, 7, seed)` (7 plis par lot), **6 graines**.
  **On choisit sur les graines 0-2, on confirme sur les graines 3-5** (jamais vues pendant le choix).
- Les modèles sont entraînés sur log(cible) ; les mélanges sont des **moyennes géométriques pondérées** ;
  `tcd = max(tcd, vcd)`.
- Les prédictions hors-pli (OOF) de chaque modèle sur ces plis sont déjà calculées dans
  `results/families/<modèle>_oof_s{0..5}.csv` (colonnes batch_id, j, elapsed_time, cibles) :
  `ours`, `their` (modèle du coéquipier), `ours_base`, `theirs_base`, `byphuber`, `families` (colonne `model`),
  `wave2` (colonne `model`). Pour comparer un nouveau modèle, produis ses OOF sur les **mêmes plis** et mélange-le
  au meilleur actuel.
- Piège déjà rencontré : corriger les prédictions à partir des erreurs OOF des autres lots (« cross-fitting » sur les
  mêmes plis) est **biaisé contre la correction**, car les erreurs OOF d'entraînement viennent de modèles qui ont vu
  les lots de validation. Pour une telle correction, il faut des OOF imbriqués.

## 5. Ce qui marche : historique du leaderboard
| Fichier (submissions/) | LB moyen | vcd | tcd | substrate | byproduct | Ce qui a changé |
|---|---|---|---|---|---|---|
| hybrid_ensemble | 7.938 | 7.279 | 7.245 | 8.454 | 8.772 | notre modèle seul |
| (coéquipier, src/submission.csv) | 7.743 | 7.233 | 7.214 | 8.284 | 8.241 | src/v2.py |
| blend_team | 7.725 | 7.183 | 7.178 | 8.297 | 8.241 | mélange des deux |
| blend_nonlinear | 7.686 | 7.164 | 7.044 | 8.297 | 8.241 | + SVR et arbres « pooled-time » pour vcd/tcd |
| blend_nonlinear_dose | 7.613 | 7.164 | 7.044 | 8.003 | 8.241 | + règle des doses tardives de substrat |
| blend_nonlinear_rules | 7.627 | | | 8.061 | | règle mi-culture réappliquée au mélange final : **pire, abandonnée** |
| **blend_base** | **7.464** | 7.164 | 7.044 | 8.003 | **7.644** | + byproduct avec le volume caché de base |
| blend_base2 | (en attente) | | | | | + volume caché aussi pour le substrat (CV −0.23) |

Chaîne pour reproduire le meilleur fichier :
1. `python src/run_regression.py` → `submissions/hybrid_ensemble.csv` (ridge par (cible, jour) + baselines +
   ODE hybride + règle du substrat ; `src/regression.py`, `src/control_rules.py`, `src/ode_paramnet.py`).
2. `src/v2.py` (coéquipier, ne pas modifier) → `src/submission.csv`.
3. `python src/blend_team.py` → `blend_team.csv` (poids sur le coéquipier : vcd 0.6, tcd 0.5, substrate 0.5, byproduct 1.0).
4. `python src/model_families.py` → `blend_nonlinear.csv` (vcd/tcd = 0.25 équipe + 0.55 SVR + 0.20 arbres).
5. `python src/final_submission.py --late-only` → `blend_nonlinear_dose.csv` (`control_rules.LateDoseRule`, q=0.2).
6. `python src/final_submission.py --base` → `blend_base.csv` (byproduct = 0.9 `src/byproduct_base.py` + 0.1 coéquipier).
7. `python src/final_submission.py --base2` → `blend_base2.csv` (substrate = 0.75 ours_base + 0.25 coéquipier puis
   LateDoseRule ; byproduct = 0.7 modèle de la note + 0.2 ours_base + 0.1 coéquipier).

Scores de validation croisée du meilleur actuel (moyenne 6 graines) : vcd ≈ 6.50, tcd ≈ 6.44,
substrate ≈ 6.48-6.58 (version base2), byproduct ≈ 7.80-7.87. Le leaderboard est en général plus haut
(test plus difficile, 12 lots seulement, bruit d'environ ±0.3-0.4 par variable).

## 6. Mécanismes du procédé découverts (le levier principal)
Le procédé est piloté par des **régulateurs qui réagissent aux mesures**. Leurs actions sont visibles dans les entrées
du test (feeds, volume) : c'est ce qui a donné les plus gros gains au leaderboard.
- **Volume caché = base du régulateur de pH** (`src/base_volume.py`) : par intervalle,
  `volume(j) − volume(j−1) − somme des feeds enregistrés`. Le lactate (byproduct) est un acide, la base ajoutée
  n'apparaît pas dans les feeds mais se voit dans le volume. `base_production` = somme sur les intervalles finissant
  aux échantillons 3, 4, 5 : corrélation 0.72 avec le pic de lactate. Gains : byproduct CV 8.67 → 7.88, LB
  8.241 → 7.644 ; substrate CV −0.17 (ridge) / −0.23 (mélange). Aucun gain sur vcd/tcd (légèrement pire).
  Le cumul `hidden_cum_t` n'aide pas.
- **Doses tardives de substrat** (après les échantillons 11 et 12, dans tous les lots) : seulement 2 tailles,
  dose/V = 0.006 si le S mesuré était < ~0.96, sinon 0.004 → `LateDoseRule` borne S(11), S(12) du bon côté du seuil.
  CV −0.14, **LB −0.29** sur le substrat.
- **Bolus de substrat en milieu de culture** (échantillons 5-10) quand S < ~1.9 : 7/49 lots d'entraînement,
  3/12 au test (Batch6, Batch35, Batch54). S(j0) ≈ 1.85, S(j0+1) ≈ 3.1, plancher 1.91 sinon
  (`SubstrateControlRule`, appliquée dans notre modèle seulement ; la réappliquer au mélange final a dégradé le LB).
- **Feed B** remplace le substrat consommé la veille (corr −0.85 entre log(dose_k/dose_k−1) et log(S_k/S_k−1)).
  Il renseigne sur des *différences* de S, pas sur les niveaux : en variables, il dégrade les modèles.
- **Feed C** : les grosses doses (0.0018) sont programmées, 1 par jour à partir du jour 5 (aucune info) ; les petites
  (≤ 0.0013) arrivent à la demande et à n'importe quelle heure (pH ?) ; info faible, `nCs_tot` déjà utilisé par le coéquipier.
- **Feed D** : dosé selon le volume, aucune information sur l'état.

## 7. Ce qui a échoué : ne pas refaire
Familles non linéaires pour substrate/byproduct (bien pires : sauts de feeds à heures fixes) ; SVR pondérée en temps,
ridge à coefficients variables, SVR sur les résidus, SVR par jour (vague 2 : 0) ; variables de doses Feed B ;
calibration exp(−σ²) pour la MAPE ; correction des biais par régime de contrôle (post-traitement ou variables :
sur-apprentissage, seulement 7 lots pilotés) ; importance weighting (pas de décalage train/test détectable,
AUC 0.43) ; réseau de neurones (sur-apprend) ; anciennes variables Feed C (`nC_early`, `nC_late_log`) : meilleures
en CV mais **pires au LB deux fois**.
Plafonds mesurés : l'ODE v3, même calée lot par lot sur les vraies mesures, fait 17 % sur le substrat et 15 % sur
le byproduct (elle a mis la concentration du feed de substrat à 1 g/L au lieu d'environ 300) ; le bilan matière de
fin de culture, même avec le vrai S précédent, fait 7-10 % sur S12/S13.

## 8. Où chercher maintenant (pistes, par ordre de promesse)
1. **Exploiter davantage le volume caché.** C'est la mine qui vient de payer. Par exemple : volumes cachés d'autres
   phases (base ajoutée en fin de culture → lactate tardif, échantillons 11-13 = la plus grosse part de l'erreur du
   byproduct) ; une variable par intervalle au lieu d'une somme ; la même variable dans les modèles SVR/arbres ;
   d'autres « signaux cachés » dans le bilan de volume (prélèvements, évaporation, anti-mousse...).
   Choisir les fenêtres sur les graines 0-2 seulement, puis confirmer sur 3-5.
2. **Erreur restante du substrat en fin de culture** (S11 ≈ 16 %, S12/S13 ≈ 11 %).
3. **vcd/tcd** (LB 7.16 / 7.04) : chercher un signal de procédé qui révèle la croissance (consommation, base, volume).
4. Tout nouveau mélange : poids choisis sur les graines 0-2, confirmés sur 3-5, puis un seul fichier soumis à la fois.

## 9. Livrables attendus
- Un nouveau fichier `submissions/<nom>.csv` (156 lignes, ordre du template, aucun NaN, tout > 0), le code qui le
  reproduit dans `src/` et les chiffres de validation croisée (graines 0-2 et 3-5, par variable) comparés au meilleur actuel.
- Ajoute tes essais (réussis et ratés) à `results/families/experiments_log.md`.
- Ne commite pas, ne pousse pas, ne soumets pas : dis à l'utilisateur quel fichier soumettre et ce que tu en attends.
