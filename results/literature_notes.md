# Notes bibliographiques : culture fed-batch de cellules CHO

Résumé court des mécanismes utiles pour lire nos données (projet DTB). Pour chaque point : ce que dit la
littérature, puis l'analyse du notebook `notebooks/04_data_deep_dive.ipynb` qui le teste.

## 1. Croissance : Monod, inhibition, logistique
- Les modèles cinétiques de cultures de cellules de mammifères sont surtout **non structurés et non ségrégés** :
  μ = μ_max · Π [S/(K_S+S)] · Π [K_I/(K_I+P)], avec glucose/glutamine comme substrats limitants et lactate/ammonium
  comme inhibiteurs (revue : Kyriakopoulos et al. 2018).
- Article de référence du challenge (Chiu & Du 2025) : modèle mécaniste avec dX_v/dt = (μ − μ_d)X_v,
  μ = μ_max·GLC/(K_glc+GLC)·K_Ilac/(K_Ilac+LAC) et μ_d = k_d·LAC/(K_Dlac+LAC)·K_DIglc/(K_DIglc+GLC).
- Fed-batch CHO à 5000 L : 6 EDO et 18 paramètres, le lactate et l'ammonium sont les principaux inhibiteurs
  (Xing et al. 2010).
- Les courbes de croissance CHO (batch et fed-batch) sont bien décrites par des **équations logistiques** avant la
  phase de mort (Goudar et al. 2005).
- L'arrêt de prolifération en fed-batch s'accompagne d'un blocage en **G0/G1** et d'une augmentation de la taille des
  cellules (Pan et al. 2019) : la croissance peut s'arrêter sans limitation par un substrat mesuré.
- *Testé en sections 4 et 7* : Monod en S, inhibition par P, capacité porteuse logistique, pilotage par le temps.

## 2. Mort, viabilité, lyse
- Mort du premier ordre dX_d/dt = k_d·X_v. k_d est constant ou augmente avec un métabolite toxique (lactate, ammonium),
  la carence en glucose ou l'âge de la culture (Xing et al. 2010 ; Chiu & Du 2025).
- Une partie des cellules mortes se **lyse** et disparaît des comptages, ce qui biaise TCD et k_d si on l'ignore
  (Kroll et al. 2017). Ordre de grandeur en fed-batch CHO : k_lys ≈ 4.6·10⁻⁴ h⁻¹ rapporté aux cellules totales
  (Sissolak et al. 2019).
- *Testé en section 5* : k_d par fenêtre d'âge, liens avec S, P et l'IVCD, test de lyse sur la quantité totale de cellules.

## 3. IVCD, taux spécifiques, rendements, Luedeking-Piret
- Les taux spécifiques s'estiment comme la **pente de la quantité cumulée** (consommée ou produite) en fonction de
  l'**IVCD** (intégrale trapézoïdale de la VCD) (pratique standard, par ex. études de scale-down de fed-batch CHO).
- **Pirt** : q_S = μ/Y_X/S + m_S (croissance + maintenance).
- **Luedeking-Piret** : dP/dt = α·dX/dt + β·X (part associée à la croissance + part non associée). Appliqué en CHO
  surtout au titre d'anticorps (Cytotechnology 2015, MIT ; Nature Comm. Eng. 2025).
- *Testé en section 6* : linéarité des cumuls en fonction de l'IVC, constance de q_S/q_P par phase et par lot, fits Pirt
  et Luedeking-Piret.

## 4. Lactate : overflow, lactate shift, run-away
- Les CHO convertissent une grande partie du glucose en lactate pendant la croissance rapide (**overflow / effet
  Warburg**). Le maximum théorique est de 2 mol de lactate par mol de glucose, soit 1 g/g.
- **Lactate shift** : passage à une consommation nette de lactate quand le **flux glycolytique devient faible**
  (consommation de glucose basse) et que le lactate est élevé, même si du glucose reste disponible (Mulukutla et al.
  2012, Metab. Eng.). Rôle de l'activité oxydative mitochondriale (Zagari et al. 2013). Le déclencheur exact reste
  discuté (revue Oxford, « Mechanisms driving the lactate switch »).
- Si le shift n'a pas lieu, on observe un **lactate run-away**. Le pH peut servir de capteur indirect précoce
  (vers J5) pour prédire le shift (Qin et al. 2026, Biotechnol. Bioeng.).
- Ordres de grandeur de l'inhibition (hybridome) : μ divisé par 2 à **55 mM de lactate** (≈ 5 g/L) et à 4 mM
  d'ammonium (Ozturk et al. 1992). Seuil du lactate observé en CHO industriel : ≈ 58 mM (Xing et al. 2008).
- *Testé en sections 3 (base cachée), 6 (overflow, shift) et 7 (inhibition)*.

## 5. Stratégies de feed par bolus, nutriments
- Les feeds concentrés (acides aminés, vitamines, glucose) sont ajoutés par bolus quotidiens, à volume fixe
  (% du volume) ou proportionnel à la VCD (**cell-specific feed rate**, « VCD-based feeding », Sci. Rep. 2025).
  Le glucose est souvent complété à part par une solution concentrée pour revenir à une consigne.
- L'épuisement de certains acides aminés (**asparagine, cystéine, glutamine**) arrête la croissance (Duarte et al.
  2014 ; Ghaffari et al. 2020), et plusieurs peuvent s'épuiser presque simultanément (Frontiers Bioeng. 2025).
- *Testé en section 2* (lois de dosage : calendrier, dose par cellule ou rétroaction) *et en section 7* (H3 : déplétion
  d'un nutriment apporté par B, C ou D).

## 6. Régulation du pH, base, antimousse
- Le pH est régulé par du CO₂ (acide) et une **base liquide** (Na₂CO₃ ou NaHCO₃ 0.45-1 M) ajoutée quand le pH passe
  sous la consigne. La base neutralise l'acide lactique, et son volume suit donc la production de lactate. Elle
  augmente aussi l'osmolalité (revue « PID controls: the forgotten bioprocess parameters », 2022).
- L'**antimousse** est ajouté en petites doses fixes, à la demande (capteur de mousse ou inspection) et/ou de façon
  routinière (notes d'application Eppendorf/Allegro).
- *Testé en section 3* : volume non enregistré (« caché ») et production de lactate. *Section 2* : nature de Feed C.

## 7. Shift de température
- Un passage de 37 °C à 31-33 °C en milieu de culture (souvent J3-J5) ralentit la croissance (arrêt en G1), prolonge
  la viabilité et augmente la productivité spécifique. C'est un événement **programmé dans le temps**, absent des
  données mesurées.
- *Testé indirectement en section 4.1/7* : un ralentissement identique en temps pour tous les lots est compatible
  avec un tel événement programmé.

## Références
- Chiu K-C, Du D (2025). A neural ODE model for predicting the growth of CHO cells in a bioreactor system.
  *Biotechnol. Bioprocess Eng.* 30:100-115. doi:10.1007/s12257-024-00141-2 (fourni : `paper_CHO.pdf`).
- Kyriakopoulos S, Ang KS, Lakshmanan M, et al. (2018). Kinetic modeling of mammalian cell culture bioprocessing: the
  quest to advance biomanufacturing. *Biotechnol. J.* 13:1700229. https://doi.org/10.1002/biot.201700229
- Xing Z, Bishop N, Leister K, Li ZJ (2010). Modeling kinetics of a large-scale fed-batch CHO cell culture by Markov
  chain Monte Carlo method. *Biotechnol. Prog.* 26:208-219. https://doi.org/10.1002/btpr.284
- Xing Z, et al. (2008). Identifying inhibitory threshold values of repressing metabolites in CHO cell culture using
  multivariate analysis methods. *Biotechnol. Prog.* https://doi.org/10.1021/bp070466m
- Goudar CT, Joeris K, Konstantinov KB, Piret JM (2005). Logistic equations effectively model mammalian cell batch
  and fed-batch kinetics. *Biotechnol. Prog.* 21:1109-1118.
- Ozturk SS, Riley MR, Palsson BO (1992). Effects of ammonia and lactate on hybridoma growth, metabolism, and antibody
  production. *Biotechnol. Bioeng.* 39:418-431. https://doi.org/10.1002/bit.260390408
- Mulukutla BC, Gramer M, Hu W-S (2012). On metabolic shift to lactate consumption in fed-batch culture of mammalian
  cells. *Metab. Eng.* 14:138-149.
- Zagari F, et al. (2013). Lactate metabolism shift in CHO cell culture: the role of mitochondrial oxidative activity.
  *New Biotechnol.* 30:238-245.
- Mechanisms driving the lactate switch in CHO cells (revue, Univ. Oxford ORA).
  https://ora.ox.ac.uk/objects/uuid:f713ef24-d559-4601-afb2-5177dca20e9b
- Qin L, Savery J, Cheeks M, Farid S, Goldrick S (2026). A hybrid mechanistic model capturing lactate run-away and
  back-metabolism in mammalian cell culture. *Biotechnol. Bioeng.* e70417. https://doi.org/10.1002/bit.70417
- Kroll P, Eilers K, Fricke J, Herwig C (2017). Impact of cell lysis on the description of cell growth and death in
  cell culture. *Eng. Life Sci.* https://pubmed.ncbi.nlm.nih.gov/32624789/
- Sissolak B, et al. (2019). Application of the Bradford assay for cell lysis quantification. *Biotechnol. J.*
  https://doi.org/10.1002/biot.201800714
- Pan X, et al. (2019). Transcriptome analysis of CHO cell size increase during a fed-batch process. *Biotechnol. J.*
  https://doi.org/10.1002/biot.201800156
- Duarte TM, et al. (2014). Metabolic responses of CHO cells to limitation of key amino acids. *Biotechnol. Bioeng.*
  https://pubmed.ncbi.nlm.nih.gov/24771076/
- Ghaffari N, et al. (2020). Effects of cysteine, asparagine, or glutamine limitations in CHO cell batch and fed-batch
  cultures. *Biotechnol. Prog.*
- Viable cell density as an indicator for dynamic feeding strategy in fed-batch and perfusion CHO cell culture (2025).
  *Sci. Rep.* https://www.nature.com/articles/s41598-025-28316-8
- PID controls: the forgotten bioprocess parameters (2022). *Discover Chem. Eng.*
  https://link.springer.com/article/10.1007/s43938-022-00008-z
- Using simple models to describe the kinetics of growth, ... in recombinant CHO cells (2015). *Cytotechnology*.
  https://dspace.mit.edu/handle/1721.1/105504
- Luedeking-Piret regression for multi-step-ahead forecasting and clone selection in mAb biomanufacturing (2025).
  *Commun. Eng.* https://www.nature.com/articles/s44172-025-00547-7
