-- V5: Monitoring and early warning tables

CREATE TABLE IF NOT EXISTS monitoring_finding (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          uuid NOT NULL REFERENCES run(id),
    parameter_code  text NOT NULL,
    elapsed_time    numeric(24,9) NOT NULL,
    kind            text NOT NULL CHECK (kind IN (
                        'OUTSIDE_REFERENCE_BAND',
                        'PREDICTED_LIMIT_BREACH',
                        'MODEL_MISMATCH'
                    )),
    severity        text NOT NULL DEFAULT 'WARNING' CHECK (severity IN ('INFO','WARNING','ALARM')),
    message         text NOT NULL,
    status          text NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN','ACKNOWLEDGED','DISMISSED')),
    decided_by      text,
    decided_at      timestamptz,
    reason          text,
    created_at      timestamptz NOT NULL DEFAULT now(),
    created_by      text,
    row_version     bigint NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_monitoring_finding_run
    ON monitoring_finding (run_id, parameter_code, elapsed_time);
