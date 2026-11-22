# Rapport final : Decode the Bioreactor, équipe 10

## Classement
- Meilleure soumission : `submissions/blend_base_kalman3_full.csv`, **MAPE moyenne 7.300** (vcd 7.164, tcd 7.044,
  substrat 7.349, byproduct 7.644), **7e place** au relevé du 22 nov. vers 11 h 45 UTC.
- Top 3 à ce moment : 6.709 / 6.725 / 6.752. Écart restant : **0.55 point**, sur les quatre variables.
- Progression depuis le début de la journée : 8.119 → 7.300 (−0.82).

## Le modèle, simplement
La prédiction de chaque variable combine plusieurs modèles statistiques et une compréhension du **pilotage du procédé**.

1. **Cellules (vcd, tcd).** Courbe moyenne en log, ajustée lot par lot à partir de l'état initial et des feeds :
   régressions ridge par jour (notre modèle et celui du coéquipier, avec ~20 % d'ODE mécanistique), plus une SVR
   et des forêts d'arbres apprises sur tous les jours à la fois. Mélange géométrique : 25 % modèles d'équipe,
   55 % SVR, 20 % arbres.
2. **Byproduct (lactate).** Régression robuste (Huber) par jour sur log(lactate) avec le **volume de base caché** :
   le lactate est un acide, le régulateur de pH ajoute de la base, elle n'est pas dans les feeds mais se voit dans le
   volume du réacteur (variation de volume − feeds enregistrés, intervalles 3 à 5). Corrélation 0.72 avec le pic
   de lactate. 90 % ce modèle, 10 % celui du coéquipier.
3. **Substrat (glucose).** Prédiction statistique (a priori) **corrigée par assimilation de données** (mise à jour
   gaussienne de type Kalman) :
   - le **Feed B** est dosé par un régulateur qui remplace le substrat consommé la veille ; ses doses relient deux
     jours consécutifs : S_k ≈ a + b·S_{k-1} + c·dose_k/V + e·dose_{k-1}/V (S_0 mesuré sert d'ancrage) ;
   - les **règles du contrôleur** deviennent des observations : bolus de substrat en milieu de culture quand S < 1.9
     (S juste avant ≈ 1.85, juste après ≈ 3.1, S ≥ 1.91 sinon entre les jours 5 et 10) ; taille des doses tardives
     (0.006·V si S < 0.96, 0.004·V sinon) ;
   - mathématiquement : S_post = S_prior + Σ Hᵀ (H Σ Hᵀ + R)⁻¹ (z − H S_prior), où Σ contient les incertitudes et
     les corrélations entre jours de l'a priori ; une correction sur un jour se propage aux jours voisins.

## Ce qui a amélioré le score, et pourquoi
| Étape | Gain au leaderboard | Pourquoi |
|---|---|---|
| Mélange avec le modèle du coéquipier | 7.938 → 7.725 | erreurs en partie indépendantes |
| SVR et arbres pour les cellules | → 7.686 | la croissance est une fonction lisse : apprendre sur tous les jours à la fois |
| Règle des doses tardives | → 7.613 (substrat −0.29) | la taille de la dose révèle le niveau de substrat mesuré |
| Volume de base caché | → 7.464 (byproduct −0.60) | la base ajoutée pour le pH mesure le lactate produit |
| Feed B comme capteur (Kalman) | → 7.438 puis 7.410 | le régulateur relie les jours consécutifs |
| Règles du contrôleur dans la mise à jour | → 7.389 puis **7.300** (substrat 7.35) | les contraintes se propagent aux jours voisins |

Ce qui n'a **pas** tenu au leaderboard malgré la validation croisée : les réglages fins (kalman4), les variables de
volume caché pour le substrat (base3), la règle du milieu de culture en post-traitement. Avec 12 lots de test, le
leaderboard bouge de ±0.3 à ±1 point par variable : seules les idées physiques ont transféré.

## Ce que les données disent du procédé
- **Croissance :** courbes parallèles en log ; ~0.7-0.8 /jour les jours 1-2, ralentissement dès le jour 3, plateau
  vers le jour 7-8 (pic ~31 millions/mL), puis lent déclin. La VCD initiale fixe le niveau ; le **lactate initial**
  apporte une information indépendante sur l'inoculum (fort effet, y compris pour les lots extrêmes). Ni le glucose
  ni le lactate ne limitent la croissance avant le jour 11 : le ralentissement commun vient probablement d'un
  nutriment non mesuré. Bruit de mesure de la VCD ≈ 6.5 %, d'où un plancher d'erreur vers 5-5.5 %.
- **Substrat :** variable **régulée** ; le Feed B (~20-60 g/L de substrat selon l'estimation) remplace la
  consommation de la veille ; un écart initial disparaît en ~6 jours. La consommation par cellule varie d'un lot à
  l'autre et dépend peu de la VCD : le bilan matière n'explique que 25-50 % des écarts entre lots.
- **Feed substrat (Feed A, ~300 g/L) :** bolus quand S < 1.9 entre les jours 5 et 10 (7/49 lots, 3/12 au test) et
  systématiquement après les échantillons 11 et 12, à deux tailles selon S < 0.96.
- **Feed C :** grosses doses programmées (1/jour dès le jour 5) ; petites doses à la demande, à toute heure, peu
  informatives. **Feed D :** dosé selon le volume (3 doses), sans information sur l'état.
- **Lactate :** produit par overflow les jours 1-4 (base ajoutée surtout dans l'intervalle 3), puis consommé ; les
  lots qui reçoivent un bolus de substrat en milieu de culture en produisent davantage.
- **q_S et q_P :** consommation de substrat et production de lactate ne sont pas proportionnelles à l'IVCD de façon
  utile ; la seule mesure indirecte fiable est celle des régulateurs (Feed B pour S, base pour P).

## Limites et pistes
- Notre score au leaderboard est ~0.5 au-dessus de notre validation croisée sur les cellules, alors que les équipes
  de tête sont à 6.5-6.8. Composition du test (4-5 lots hors de la plage d'entraînement) n'explique que ~0.2 ; aucune
  des hypothèses testées (extrapolation, horaires, dérive des lots, capteurs) n'explique le reste.
- Pistes non abouties : un capteur indirect de la VCD, un modèle du lactate tardif (jours 11-13, 10-18 % d'erreur).

Détails chiffrés : `results/top3_log.md`, `results/families/experiments_log.md`.
Reproduction : `python src/final_submission.py --kalman3 --mid` (après `run_regression.py`, `blend_team.py`,
`model_families.py`, `final_submission.py --late-only` et `--base`).
