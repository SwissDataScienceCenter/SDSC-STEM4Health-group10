-- V3: Long-format run data store
-- Replaces the wide experiment_data table with a schema that works for any data set.

CREATE TABLE IF NOT EXISTS process_definition (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    code        text NOT NULL UNIQUE,
    name        text NOT NULL,
    version_label text,
    time_unit   text NOT NULL DEFAULT 'day',
    created_at  timestamptz NOT NULL DEFAULT now(),
    created_by  text,
    row_version bigint NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS process_step (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    process_id    uuid NOT NULL REFERENCES process_definition(id),
    sequence      int NOT NULL,
    name          text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now(),
    created_by    text,
    row_version   bigint NOT NULL DEFAULT 1,
    UNIQUE (process_id, sequence)
);

CREATE TABLE IF NOT EXISTS process_parameter (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    step_id       uuid REFERENCES process_step(id),
    code          text NOT NULL,
    name          text NOT NULL,
    unit          text,
    kind          text NOT NULL CHECK (kind IN ('STATE','INPUT','DERIVED','QUALITY')),
    critical      boolean NOT NULL DEFAULT false,
    lower_limit   numeric(24,9),
    upper_limit   numeric(24,9),
    justification text,
    created_at    timestamptz NOT NULL DEFAULT now(),
    created_by    text,
    row_version   bigint NOT NULL DEFAULT 1,
    UNIQUE (step_id, code)
);

CREATE TABLE IF NOT EXISTS equipment (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    code           text NOT NULL UNIQUE,
    name           text NOT NULL,
    kind           text,
    working_volume numeric(24,9),
    created_at     timestamptz NOT NULL DEFAULT now(),
    created_by     text,
    row_version    bigint NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS run (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    process_id     uuid NOT NULL REFERENCES process_definition(id),
    equipment_id   uuid REFERENCES equipment(id),
    code           text NOT NULL,
    kind           text NOT NULL CHECK (kind IN ('MANUFACTURING','DEVELOPMENT','SIMULATION')),
    dataset_code   text,
    role           text NOT NULL DEFAULT 'NONE' CHECK (role IN ('TRAIN','TEST','NONE')),
    created_at     timestamptz NOT NULL DEFAULT now(),
    created_by     text,
    row_version    bigint NOT NULL DEFAULT 1,
    UNIQUE (process_id, code)
);

CREATE TABLE IF NOT EXISTS reading (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id           uuid NOT NULL REFERENCES run(id),
    parameter_code   text NOT NULL,
    elapsed_time     numeric(24,9) NOT NULL,
    clock_time       timestamptz,
    numeric_value    numeric(24,9),
    unit             text,
    source           text NOT NULL DEFAULT 'OFFLINE_SAMPLE' CHECK (source IN ('SENSOR','OFFLINE_SAMPLE','IMPORT')),
    supersedes_id    uuid REFERENCES reading(id),
    created_at       timestamptz NOT NULL DEFAULT now(),
    created_by       text,
    row_version      bigint NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_reading_run_param_time
    ON reading (run_id, parameter_code, elapsed_time);

CREATE TABLE IF NOT EXISTS run_event (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id         uuid NOT NULL REFERENCES run(id),
    elapsed_time   numeric(24,9) NOT NULL,
    kind           text NOT NULL CHECK (kind IN ('FEED_BOLUS','SAMPLE','SETPOINT_CHANGE')),
    material_code  text,
    amount         numeric(24,9),
    unit           text,
    created_at     timestamptz NOT NULL DEFAULT now(),
    created_by     text,
    row_version    bigint NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_run_event_run_time
    ON run_event (run_id, elapsed_time);

CREATE TABLE IF NOT EXISTS import_file (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name      text NOT NULL,
    sha256         text,
    column_mapping jsonb NOT NULL,
    row_count      int,
    imported_at    timestamptz NOT NULL DEFAULT now(),
    created_at     timestamptz NOT NULL DEFAULT now(),
    created_by     text,
    row_version    bigint NOT NULL DEFAULT 1
);

-- Insert the DTB process definition and parameters
INSERT INTO process_definition (code, name, time_unit, created_by)
VALUES ('DTB-v1', 'Decode the Bioreactor challenge v1', 'day', 'migration');

-- DTB process steps
INSERT INTO process_step (process_id, sequence, name)
SELECT id, 1, 'fermentation'
FROM process_definition WHERE code = 'DTB-v1';

-- DTB parameters (one per measured column in experiment_data)
INSERT INTO process_parameter (step_id, code, name, unit, kind, critical)
SELECT ps.id, p.code, p.name, p.unit, p.kind::text, p.critical
FROM process_step ps
JOIN process_definition pd ON ps.process_id = pd.id,
(VALUES
    ('vcd',       'Viable cell density', '10^6 cells/mL', 'STATE', true),
    ('tcd',       'Total cell density',  '10^6 cells/mL', 'STATE', true),
    ('substrate', 'Substrate concentration', 'g/L', 'STATE', true),
    ('byproduct', 'Byproduct concentration', 'g/L', 'STATE', true),
    ('volume',    'Working volume',         'L',    'STATE', true),
    ('feed_b',    'Feed B volume',          'L',    'INPUT', false),
    ('feed_c',    'Feed C volume',          'L',    'INPUT', false),
    ('feed_d',    'Feed D volume',          'L',    'INPUT', false),
    ('temperature_1', 'Temperature sensor 1', 'C', 'DERIVED', false),
    ('temperature_2', 'Temperature sensor 2', 'C', 'DERIVED', false),
    ('pressure_1',    'Pressure sensor 1', 'bar', 'DERIVED', false)
    -- Source columns for the DTB import mapping; add others as needed
) AS p(code, name, unit, kind, critical)
WHERE pd.code = 'DTB-v1';
