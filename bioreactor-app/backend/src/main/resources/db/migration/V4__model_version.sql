-- V4: Model version record
-- Tracks model definitions, fitting runs and predictions for traceability.

CREATE TABLE IF NOT EXISTS model (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    code            text NOT NULL UNIQUE,
    kind            text NOT NULL CHECK (kind IN ('MECHANISTIC','HYBRID','ML')),
    description     text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    created_by      text,
    row_version     bigint NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS model_version (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id        uuid NOT NULL REFERENCES model(id),
    version_label   text NOT NULL,
    kind            text NOT NULL CHECK (kind IN ('MECHANISTIC','HYBRID','ML')),
    git_commit      text,
    parameters      jsonb NOT NULL,          -- [{"code":"mu_max","value":0.73,"unit":"1/d","lower":0.01,"upper":5.0}]
    training_run_ids uuid[],                 -- run.id values used for fitting
    training_data_sha256 text,
    validation_results jsonb,               -- per-variable MARE/RMSE from CV
    validation_scheme   text,
    intended_use    text,
    limitations     text,
    status          text NOT NULL DEFAULT 'DRAFT' CHECK (status IN ('DRAFT','APPROVED','RETIRED')),
    approved_by     text,
    approved_at     timestamptz,
    supersedes_id   uuid REFERENCES model_version(id),
    created_at      timestamptz NOT NULL DEFAULT now(),
    created_by      text,
    row_version     bigint NOT NULL DEFAULT 1,
    UNIQUE (model_id, version_label)
);

CREATE TABLE IF NOT EXISTS prediction_run (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id uuid NOT NULL REFERENCES model_version(id),
    run_id          uuid REFERENCES run(id),
    input_digest    text NOT NULL,           -- sha256 of the inputs (initial state + events + params)
    status          text NOT NULL DEFAULT 'COMPLETED',
    warnings        jsonb,
    created_at      timestamptz NOT NULL DEFAULT now(),
    created_by      text,
    row_version     bigint NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS prediction_value (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    prediction_run_id uuid NOT NULL REFERENCES prediction_run(id),
    parameter_code   text NOT NULL,
    elapsed_time     numeric(24,9) NOT NULL,
    value            numeric(24,9) NOT NULL,
    lower_band       numeric(24,9),
    upper_band       numeric(24,9),
    created_at       timestamptz NOT NULL DEFAULT now(),
    created_by       text,
    row_version      bigint NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_prediction_value_run_param_time
    ON prediction_value (prediction_run_id, parameter_code, elapsed_time);
