-- V7: Seed DTB model version records for traceability.
-- These correspond to the Java model classes and their fitted parameters.

INSERT INTO model (code, kind, description, created_by)
VALUES
    ('DTB-v3', 'MECHANISTIC', 'V3 mechanistic model (22 parameters, 2 hidden states). Ported from Python mechanistic.py rates_v3.', 'migration'),
    ('DTB-v3-constrained', 'MECHANISTIC', 'V3 constrained model (12 free params, 10 locked to Python fit values).', 'migration')
ON CONFLICT (code) DO NOTHING;

-- V3 full model version (22 params from Python full-fit)
INSERT INTO model_version (model_id, version_label, kind, git_commit, parameters,
                           training_run_ids, validation_results, validation_scheme,
                           intended_use, limitations, status, created_by)
SELECT
    m.id, 'v1.0-python-fit', 'MECHANISTIC', 'affc404',
    '[{"code":"mu_max","value":11.163,"unit":"1/d","lower":0.1,"upper":20.0},
      {"code":"Ks","value":1.178,"unit":"g/L","lower":0.001,"upper":50.0},
      {"code":"K_N","value":10.0,"unit":"N0","lower":0.001,"upper":10.0},
      {"code":"Yxn","value":45.638,"unit":"cells/N0","lower":0.1,"upper":1000.0},
      {"code":"C_Nb","value":0.0001,"unit":"N0","lower":0.0001,"upper":1000.0},
      {"code":"kd0","value":0.000001,"unit":"1/d","lower":0.000001,"upper":0.5},
      {"code":"kd_N","value":2.0,"unit":"1/d","lower":0.000001,"upper":2.0},
      {"code":"K_dN","value":0.000277,"unit":"N0","lower":0.0001,"upper":10.0},
      {"code":"kd_D","value":0.04581,"unit":"1/(d·g/L·d)","lower":0.0000001,"upper":1.0},
      {"code":"k_r","value":9.983,"unit":"1/d","lower":0.0001,"upper":10.0},
      {"code":"k_lys","value":0.3083,"unit":"1/d","lower":0.00001,"upper":5.0},
      {"code":"Yxs","value":10000.0,"unit":"cells/g","lower":0.1,"upper":10000.0},
      {"code":"m","value":0.06193,"unit":"g/(cells·d)","lower":0.0001,"upper":2.0},
      {"code":"q_ovf","value":5.0,"unit":"g/(cells·d)","lower":0.001,"upper":5.0},
      {"code":"K_ovf","value":2.706,"unit":"g/L","lower":0.3,"upper":10.0},
      {"code":"n_ovf","value":20.0,"unit":"-","lower":1.0,"upper":20.0},
      {"code":"K_iP","value":0.08963,"unit":"g/L","lower":0.001,"upper":50.0},
      {"code":"Yps","value":1.0862,"unit":"g/g","lower":0.05,"upper":5.0},
      {"code":"qPc","value":2.0,"unit":"g/(cells·d)","lower":0.00001,"upper":2.0},
      {"code":"K_Pc","value":27.336,"unit":"g/L","lower":0.001,"upper":100.0},
      {"code":"C_sub","value":1.0,"unit":"g/L","lower":1.0,"upper":3000.0},
      {"code":"C_b","value":90.156,"unit":"g/L","lower":0.1,"upper":1000.0}]'::jsonb,
    ARRAY[]::uuid[],
    '{"mape_vcd":8.4,"mape_tcd":8.6,"mape_substrate":21.9,"mape_byproduct":19.7,"mape_mean":14.6}'::jsonb,
    '5-fold CV on 49 training batches (original Python fit)',
    'Predict VCD, TCD, substrate, and byproduct for DTB fed-batch CHO cell culture.',
    '8 of 22 parameters at bounds; byproduct MAPE higher than reference; model near structural limit.',
    'APPROVED', 'migration'
FROM model m
WHERE m.code = 'DTB-v3'
ON CONFLICT (model_id, version_label) DO NOTHING;

-- V3 constrained model version (12 free params + 10 locked)
INSERT INTO model_version (model_id, version_label, kind, git_commit, parameters,
                           training_run_ids, validation_results, validation_scheme,
                           intended_use, limitations, status, supersedes_id, created_by)
SELECT
    m.id, 'v1.0-constrained', 'MECHANISTIC', 'affc404',
    '[{"code":"mu_max","value":11.163,"unit":"1/d","lower":0.5,"upper":20.0},
      {"code":"Ks","value":1.178,"unit":"g/L","lower":0.1,"upper":50.0},
      {"code":"Yxs","value":10000.0,"unit":"cells/g","lower":50.0,"upper":10000.0},
      {"code":"m","value":0.062,"unit":"g/(cells·d)","lower":0.001,"upper":2.0},
      {"code":"q_ovf","value":5.0,"unit":"g/(cells·d)","lower":0.01,"upper":5.0},
      {"code":"K_ovf","value":2.706,"unit":"g/L","lower":0.1,"upper":10.0},
      {"code":"n_ovf","value":20.0,"unit":"-","lower":1.0,"upper":20.0},
      {"code":"K_iP","value":0.0896,"unit":"g/L","lower":0.01,"upper":10.0},
      {"code":"Yps","value":1.086,"unit":"g/g","lower":0.1,"upper":3.0},
      {"code":"qPc","value":2.0,"unit":"g/(cells·d)","lower":0.001,"upper":2.0},
      {"code":"C_sub","value":1.0,"unit":"g/L","lower":0.5,"upper":3000.0},
      {"code":"C_b","value":90.156,"unit":"g/L","lower":5.0,"upper":500.0}]'::jsonb,
    ARRAY[]::uuid[],
    '{"mape_vcd":8.5,"mape_tcd":8.8,"mape_substrate":21.4,"mape_byproduct":25.6,"mape_mean":16.1}'::jsonb,
    '5-fold CV on 44 training batches (Java fit)',
    'Same intended use as V3 full model, but 10 parameters locked to reduce overfitting risk.',
    'Byproduct MAPE higher (25.6%) due to fewer degrees of freedom.',
    'APPROVED', 'migration'
FROM model m
WHERE m.code = 'DTB-v3-constrained'
ON CONFLICT (model_id, version_label) DO NOTHING;
