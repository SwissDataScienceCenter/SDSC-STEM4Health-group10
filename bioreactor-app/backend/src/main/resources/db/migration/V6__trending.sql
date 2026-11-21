-- V6: Trending tables

CREATE TABLE IF NOT EXISTS run_metric (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          uuid NOT NULL REFERENCES run(id),
    metric_code     text NOT NULL,
    value           numeric(24,9) NOT NULL,
    unit            text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    created_by      text,
    row_version     bigint NOT NULL DEFAULT 1,
    UNIQUE (run_id, metric_code)
);

CREATE TABLE IF NOT EXISTS trend_assessment (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    process_id      uuid NOT NULL REFERENCES process_definition(id),
    metric_code     text NOT NULL,
    period_start     timestamptz,
    period_end       timestamptz,
    method          text NOT NULL,
    rule_set_version text,
    result          jsonb NOT NULL,
    conclusion      text,
    created_by      text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    row_version     bigint NOT NULL DEFAULT 1
);
