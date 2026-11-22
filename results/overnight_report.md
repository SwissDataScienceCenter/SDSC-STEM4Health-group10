# Rapport de nuit : Decode the Bioreactor (21-22 nov. 2026, 23:17 → 23:35 UTC)

## En bref
- **Fichier à soumettre : `submissions/overnight_v2.csv`** (non soumis). Validé : 156 lignes, mêmes clés et même ordre
  que le template, aucun NaN, aucune valeur ≤ 0.
- **Score honnête en leave-one-batch-out (LOBO, 49 lots) : 7,33 % de MAPE moyenne** (estimation imbriquée prudente),
  contre **7,47 %** pour `ridge_blend.csv` dans les mêmes conditions. Soit **−0,15 point**, entièrement gagné sur vcd et tcd.
- ⚠️ Le « 7,52 % » de départ venait d'une CV groupée à 7 plis (graines 0-2), pas d'un LOBO. Dans le harnais LOBO commun
  (`src/experiments/common.py`, mêmes plis pour toutes les expériences), `ridge_blend` fait 7,474 %. Le harnais reproduit
  exactement `submissions/ridge_blend.csv` (écart relatif max = 0).

## Meilleur modèle : `overnight_v2` (`src/experiments/final.py`)
| cible | modèle |
|---|---|
| vcd, tcd | moyenne géométrique **50/50** de (a) la ridge par (cible, temps) avec pénalité ×0,25 et projection de la trajectoire prédite (log) sur la courbe moyenne + 1 mode FPCA, et (b) la moyenne géométrique des 8 lots d'entraînement les plus proches (état initial, volume, totaux de feeds, date du 1er feed substrat) |
| substrate, byproduct | inchangés : la référence `ridge_blend` (ridge + 0,2 de baseline) |

**Pourquoi ça marche.** La ridge et les plus proches voisins font des erreurs peu corrélées sur les densités cellulaires.
La projection sur 1 mode lisse les 13 prédictions indépendantes (une courbe de croissance a essentiellement un degré de
liberté : son amplitude). Sur substrat et byproduct, la projection et les voisins dégradent : les bolus de feed créent
des sauts que le lissage efface.

**Pourquoi 50/50 (le plus simple).** Toutes les CV imbriquées (poids choisis pour chaque lot tenu à l'écart sur les
48 autres) convergent vers 0,5 / 0,5 pour vcd et tcd. La baseline et les autres valeurs de k reçoivent ≈ 0.

| LOBO MAPE (%) | vcd | tcd | substrate | byproduct | moyenne |
|---|---|---|---|---|---|
| référence `ridge_blend` | 6,892 | 6,858 | 7,042 | 9,104 | 7,474 |
| **overnight_v2, estimation imbriquée prudente** (ensemble à 4 membres) | **6,636** | **6,526** | 7,042 | 9,104 | **7,327** |
| overnight_v2, poids fixes 50/50 évalués directement (légèrement optimiste) | 6,564 | 6,493 | 7,042 | 9,104 | 7,301 |

Robustesse : v2 bat la référence sur 34/49 lots (vcd) et 35/49 (tcd), et fait mieux ou pareil à chacun des 13 temps.
Les lots test ne sont pas plus extrêmes que les lots d'entraînement vis-à-vis de leurs 8 voisins (distance moyenne
2,51 contre 2,21 ; max 4,27 contre 5,13). Écart médian avec `ridge_blend.csv` sur le test : 1,7 % sur vcd/tcd, 0 ailleurs.
`submissions/overnight_v1.csv` (tcd à 0,5 / 0,1 baseline / 0,4 voisins) a le même score : préférer v2, plus simple.

## Classement des expériences (moyenne LOBO, honnête sauf mention)
| rang | expérience | moyenne | verdict |
|---|---|---|---|
| 1 | **Ensemble imbriqué ridge (FPCA1, ×0,25) + plus proches lots, vcd/tcd** → overnight_v2 | **7,327** | ✅ retenu |
| 2 | + ODE hybride de l'équipe comme 3e membre (vcd 6,540, tcd 6,499) | 7,276* | ❌ gain < 0,1 sur vcd/tcd, fuite légère (paramètres partagés ajustés sur les 49 lots), 28 s/pli |
| 3 | Lissage FPCA seul (K choisi en imbriqué : 1 pour vcd/tcd, aucun pour S/P) | 7,395 | intégré dans v2 |
| 4 | Ensemble ref + FPCA1 + PLS | 7,394 | ❌ la PLS n'apporte rien |
| 5 | Pénalité ×0,25 sur substrat/byproduct (6,980 / 9,051) | (−0,03) | ⏸ gain < 0,1 par cible, option |
| 6 | Features (dilution, substrat ajouté, timing, logs, ratios des CI) | 7,53 – 7,67 | ❌ tous pires |
| 7 | Paramètres de courbe (scores PCA prédits directement, K = 2-4) | 7,60 – 8,18 | ❌ trop lisse pour S/P |
| 8 | GP (ridge + GP sur les résidus ; GP pure) | 7,84 / 8,43 | ❌ ne gagne aucune cible, ARD sur-apprend |
| 9 | Modèle série hybride (taux spécifiques + bilans matière, roulé depuis t = 0) | 11,9 – 16,5 | ❌ erreurs cumulées sur 13 pas |

\* l'ODE n'améliore que vcd/tcd ; la moyenne inclut des poids S/P légèrement meilleurs, non retenus non plus.

Notes : la GP « meilleure sur le byproduct » n'est pas confirmée en LOBO dans les conditions du modèle final
(9,31 contre 9,10). Aucune feature Feed C/D n'a été utilisée : elles avaient échoué deux fois au leaderboard.

## Variables et feeds qui comptent (coefficients standardisés des ridges sur les 49 lots, `results/overnight/feature_importance_ridge.csv`)
- **vcd / tcd** : le byproduct initial (log P₀, effet +), la densité cellulaire initiale (log VCD₀, log TCD₀, +), le
  **total de Feed B** (+) et le volume initial (−). Les coefficients sont petits parce que la pénalité est forte : la
  ridge seule reste proche d'une courbe moyenne mise à l'échelle, d'où l'apport des voisins.
- **substrate** : le **Feed B cumulé jusqu'à t** domine (+0,52 en fin de culture : Feed B apporte du substrat), puis le
  total de Feed B (−, il signe une forte croissance donc une forte consommation), le volume courant, le feed substrat
  cumulé (+) et la date du 1er feed substrat.
- **byproduct** : le **volume courant Vt** (+0,75 en fin de culture), V₀ (−), le total de Feed B (−0,53 en fin : plus de
  feed, moins de byproduct accumulé) et les densités initiales.
- Modèle série (ridge sur les taux) : μ baisse avec log Xv (croissance dépendante de la densité) et avec le Feed B
  cumulé. Le taux de mort est quasi imprévisible à partir de l'état.

## Prochaines étapes suggérées
1. Soumettre `submissions/overnight_v2.csv` et comparer au score leaderboard de `ridge_blend.csv` : seuls vcd/tcd
   changent, ce qui teste directement l'apport voisins + lissage.
2. Si le gain se confirme, tester l'option ×0,25 sur substrat/byproduct (−0,03 de moyenne attendu).
3. Byproduct (9,1 %) reste la cible la plus faible : variables d'état non mesurées (pH, nutriment caché) ;
   l'ODE v3 n'arrive pas non plus à le reproduire (~15-19 %).
4. Refaire l'ODE avec les paramètres partagés réajustés dans chaque pli pour un score sans fuite, et la ré-ajouter
   seulement si le gain dépasse 0,1.

## Fichiers
- Code : `src/experiments/` (`common.py` harnais LOBO, `smooth_pls.py`, `members.py`, `nearest_k.py`, `ensemble.py`,
  `final.py`, `features.py`, `curve_params.py`, `gp.py`, `serial_hybrid.py`, `ode_member.py`).
- Prédictions hors pli + scores : `results/overnight/<nom>_oof.csv`, `_score.csv`. Journal détaillé : `results/overnight_log.md`.
- Reproduire : `python src/experiments/final.py overnight_v2` (après `common.py`, `members.py`, `smooth_pls.py`,
  `ensemble.py ens_ridge_s+ridge_s_k025+base_scaled+nearest ridge_s ridge_s_k025 base_scaled nearest`).
- Git : le commit 54d020c a conclu la fusion avec origin/main laissée en cours (conflits déjà résolus).
  `Leaderboard_template.ipynb` (modifié localement) et les fichiers `core.*` n'ont pas été commités.
