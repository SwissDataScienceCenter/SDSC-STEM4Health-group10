-- V2__seed_organisms.sql
-- Seed data: model organisms, substrates, and parameters

-- ============================================================
-- Substrates
-- ============================================================
INSERT INTO substrates (id, name, molecular_weight, formula, category) VALUES
('glucose',   'D-Glucose',   180.16, 'C6H12O6', 'sugar'),
('glycerol',  'Glycerol',     92.09, 'C3H8O3',  'alcohol'),
('sucrose',   'Sucrose',     342.30, 'C12H22O11','sugar'),
('lactose',   'Lactose',     342.30, 'C12H22O11','sugar'),
('glutamine', 'L-Glutamine', 146.14, 'C5H10N2O3','amino_acid'),
('methanol',  'Methanol',     32.04, 'CH3OH',   'alcohol'),
('ethanol',   'Ethanol',      46.07, 'C2H5OH',  'alcohol'),
('acetate',   'Acetate',      59.04, 'C2H3O2',  'organic_acid'),
('citrate',   'Citrate',     192.12, 'C6H5O7',  'organic_acid'),
('xylose',    'D-Xylose',    150.13, 'C5H10O5', 'sugar');

-- ============================================================
-- Organisms
-- ============================================================
INSERT INTO organisms (id, name, organism_type, gram_stain, description, genome_model) VALUES
('ecoli_k12',   'Escherichia coli K-12 MG1655',   'bacteria', 'negative',
 'Model Gram-negative bacterium. Facultative anaerobe. Produces acetate under aerobic overflow, mixed acids anaerobically.',
 'iJO1366'),
('ecoli_bl21',  'Escherichia coli BL21(DE3)',     'bacteria', 'negative',
 'Protein production strain. Lacks lon and ompT proteases. Widely used for recombinant protein expression.',
 'iML1515'),
('scerevisiae', 'Saccharomyces cerevisiae S288C', 'yeast', NULL,
 'Budding yeast. Crabtree-positive: produces ethanol under aerobic excess glucose. Model eukaryote.',
 'Yeast8'),
('bsubtilis',   'Bacillus subtilis 168',          'bacteria', 'positive',
 'Gram-positive model. Spore-former. Natural competence. Produces enzymes and 2,3-butanediol.',
 'iBsu1103'),
('ppastoris',   'Komagataella phaffii (Pichia pastoris) GS115', 'yeast', NULL,
 'Methylotrophic yeast. Strong AOX1 promoter for recombinant protein. Grows on methanol.',
 'iPpa1257'),
('cglutamicum', 'Corynebacterium glutamicum ATCC 13032', 'bacteria', 'positive',
 'Industrial amino acid producer (L-glutamate, L-lysine). GRAS organism.',
 'iCGB21FR'),
('aniger',      'Aspergillus niger',               'fungi', NULL,
 'Filamentous fungus. Industrial producer of citric acid and enzymes (glucoamylase). GRAS status.',
 NULL),
('lactobacillus','Lactobacillus plantarum',        'bacteria', 'positive',
 'Lactic acid bacterium. Homofermentative. Probiotic. Used in food fermentation and silage.',
 NULL),
('zmobilis',    'Zymomonas mobilis ZM4',           'bacteria', 'negative',
 'Ethanologenic bacterium. Entner-Doudoroff pathway. Higher ethanol yield than yeast.',
 NULL);

-- ============================================================
-- Growth Parameters
-- ============================================================
INSERT INTO organism_growth_params (organism_id, model, mu_max, K_s, K_i, Y_XS, m_s, X_max, n) VALUES
('ecoli_k12',     'andrews',  0.70, 0.02, 50.0,  0.50, 0.025, NULL, 1.0),
('ecoli_bl21',    'monod',    0.60, 0.02, NULL,  0.48, 0.030, NULL, 1.0),
('scerevisiae',   'monod',    0.40, 0.05, NULL,  0.50, 0.015, NULL, 1.0),
('bsubtilis',     'monod',    0.55, 0.04, NULL,  0.40, 0.020, NULL, 1.0),
('ppastoris',     'monod',    0.18, 0.10, NULL,  0.45, 0.012, NULL, 1.0),
('cglutamicum',   'monod',    0.45, 0.03, NULL,  0.50, 0.018, NULL, 1.0),
('aniger',        'moser',    0.20, 0.08, NULL,  0.48, 0.015, NULL, 1.5),
('lactobacillus', 'monod',    0.35, 0.03, NULL,  0.20, 0.035, NULL, 1.0),
('zmobilis',      'monod',    0.30, 0.05, NULL,  0.10, 0.050, 15.0, 1.0);

-- ============================================================
-- Death Parameters
-- ============================================================
INSERT INTO organism_death_params (organism_id, k_d_max, k_lysis) VALUES
('ecoli_k12',     0.010, 0.005),
('ecoli_bl21',    0.012, 0.006),
('scerevisiae',   0.008, 0.004),
('bsubtilis',     0.015, 0.008),
('ppastoris',     0.006, 0.003),
('cglutamicum',   0.009, 0.004),
('aniger',        0.005, 0.002),
('lactobacillus', 0.008, 0.004),
('zmobilis',      0.020, 0.010);

-- ============================================================
-- Substrate Uptake Parameters
-- ============================================================
INSERT INTO organism_substrates (organism_id, substrate_id, q_S_max, K_m, priority, Y_XS) VALUES
-- E. coli K-12
('ecoli_k12',     'glucose',   1.50, 0.01, 1, 0.50),
('ecoli_k12',     'glycerol',  0.80, 0.05, 2, 0.48),
('ecoli_k12',     'acetate',   0.60, 0.10, 3, 0.35),
-- E. coli BL21
('ecoli_bl21',    'glucose',   1.20, 0.01, 1, 0.48),
-- S. cerevisiae
('scerevisiae',   'glucose',   1.80, 0.05, 1, 0.50),
('scerevisiae',   'ethanol',   0.40, 0.10, 2, 0.30),
('scerevisiae',   'glycerol',  0.30, 0.08, 3, 0.42),
-- B. subtilis
('bsubtilis',     'glucose',   1.00, 0.04, 1, 0.40),
('bsubtilis',     'glycerol',  0.50, 0.06, 2, 0.38),
-- P. pastoris
('ppastoris',     'glycerol',  0.40, 0.10, 1, 0.45),
('ppastoris',     'methanol',  0.15, 0.20, 2, 0.35),
('ppastoris',     'glucose',   0.30, 0.05, 3, 0.42),
-- C. glutamicum
('cglutamicum',   'glucose',   0.90, 0.03, 1, 0.50),
('cglutamicum',   'sucrose',   0.70, 0.05, 2, 0.48),
('cglutamicum',   'acetate',   0.40, 0.08, 3, 0.35),
-- A. niger
('aniger',        'glucose',   0.42, 0.08, 1, 0.48),
('aniger',        'sucrose',   0.38, 0.10, 2, 0.45),
-- Lactobacillus
('lactobacillus', 'glucose',   1.75, 0.03, 1, 0.20),
('lactobacillus', 'lactose',   1.20, 0.05, 2, 0.18),
-- Z. mobilis
('zmobilis',      'glucose',   2.50, 0.05, 1, 0.10),
('zmobilis',      'sucrose',   1.80, 0.08, 2, 0.09);

-- ============================================================
-- Products & Byproducts
-- ============================================================
INSERT INTO organism_products (organism_id, product_name, product_type, model, alpha, beta, Y_PS, q_S_crit, inhibition_type, K_p) VALUES
-- E. coli: acetate overflow
('ecoli_k12',     'acetate',  'byproduct', 'luedeking-piret', 0.0,  0.02, 0.15, 0.80, 'linear',  15.0),
-- E. coli: recombinant protein
('ecoli_bl21',    'protein',  'product',   'luedeking-piret', 0.05, 0.01, 0.10, NULL, 'none',    NULL),
('ecoli_bl21',    'acetate',  'byproduct', 'luedeking-piret', 0.0,  0.03, 0.12, 0.70, 'linear',  10.0),
-- S. cerevisiae: ethanol
('scerevisiae',   'ethanol',  'product',   'luedeking-piret', 0.45, 0.01, 0.48, 0.60, 'aiba',    50.0),
('scerevisiae',   'glycerol', 'byproduct', 'luedeking-piret', 0.02, 0.005, 0.05, NULL, 'none', NULL),
-- B. subtilis
('bsubtilis',     'acetoin',  'product',   'luedeking-piret', 0.0,  0.03, 0.30, NULL, 'none', NULL),
('bsubtilis',     'lactate',  'byproduct', 'luedeking-piret', 0.10, 0.01, 0.08, 0.85, 'linear', 20.0),
-- P. pastoris
('ppastoris',     'protein',  'product',   'luedeking-piret', 0.0,  0.02, 0.15, NULL, 'none', NULL),
-- C. glutamicum
('cglutamicum',   'glutamate','product',   'luedeking-piret', 0.0,  0.06, 0.40, NULL, 'linear', 120.0),
('cglutamicum',   'lactate',  'byproduct', 'luedeking-piret', 0.05, 0.01, 0.05, 0.80, 'linear', 30.0),
-- A. niger
('aniger',        'citrate',  'product',   'luedeking-piret', 0.0,  0.04, 0.70, NULL, 'linear', 100.0),
-- Lactobacillus
('lactobacillus', 'lactate',  'product',   'luedeking-piret', 0.90, 0.05, 0.95, NULL, 'aiba',    60.0),
-- Z. mobilis
('zmobilis',      'ethanol',  'product',   'luedeking-piret', 0.48, 0.02, 0.49, NULL, 'aiba',    70.0);

-- ============================================================
-- Environmental Parameters
-- ============================================================
INSERT INTO organism_environment (organism_id, temp_model, T_min, T_opt, T_max, pH_min, pH_opt, pH_max, pH_sigma) VALUES
('ecoli_k12',     'cardinal',  8.0, 37.0, 45.0,  5.0, 7.0, 9.0, 0.8),
('ecoli_bl21',    'cardinal', 10.0, 37.0, 42.0,  5.5, 7.0, 8.5, 0.7),
('scerevisiae',   'cardinal',  5.0, 30.0, 40.0,  3.5, 5.5, 7.0, 0.9),
('bsubtilis',     'cardinal', 15.0, 37.0, 55.0,  5.5, 7.0, 9.0, 0.8),
('ppastoris',     'cardinal', 15.0, 30.0, 35.0,  3.0, 5.5, 7.0, 0.9),
('cglutamicum',   'cardinal', 15.0, 30.0, 40.0,  6.0, 7.5, 9.0, 0.7),
('aniger',        'cardinal', 20.0, 33.0, 45.0,  2.5, 5.5, 8.0, 0.8),
('lactobacillus', 'cardinal', 15.0, 40.0, 50.0,  4.0, 6.0, 8.0, 0.7),
('zmobilis',      'cardinal', 15.0, 30.0, 40.0,  3.5, 5.5, 7.5, 0.8);

-- ============================================================
-- Oxygen Parameters
-- ============================================================
INSERT INTO organism_oxygen_params (organism_id, q_O2_max, Y_XO, m_O) VALUES
('ecoli_k12',     0.40, 0.95, 0.01),
('ecoli_bl21',    0.35, 0.90, 0.012),
('scerevisiae',   0.30, 0.85, 0.008),
('bsubtilis',     0.35, 0.80, 0.015),
('ppastoris',     0.25, 0.75, 0.006),
('cglutamicum',   0.28, 0.82, 0.010),
('aniger',        0.20, 0.70, 0.008),
('lactobacillus', 0.0,  0.0,  0.0),   -- facultative anaerobe
('zmobilis',      0.0,  0.0,  0.0);   -- anaerobic

-- ============================================================
-- Heat Parameters
-- ============================================================
INSERT INTO organism_heat_params (organism_id, Y_QX, Y_QS) VALUES
('ecoli_k12',     19.0, 14.0),
('ecoli_bl21',    20.0, 15.0),
('scerevisiae',   16.0, 12.0),
('bsubtilis',     18.0, 13.0),
('ppastoris',     15.0, 11.0),
('cglutamicum',   17.0, 12.0),
('aniger',        14.0, 10.0),
('lactobacillus', 22.0, 18.0),
('zmobilis',      24.0, 20.0);

-- Verify
SELECT 'Organisms: ' || COUNT(*)::text FROM organisms;
SELECT 'Products: ' || COUNT(*)::text FROM organism_products;
SELECT 'Substrate configs: ' || COUNT(*)::text FROM organism_substrates;
