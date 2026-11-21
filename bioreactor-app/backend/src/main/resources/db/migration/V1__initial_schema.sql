-- V1__initial_schema.sql
-- Bioreactor App - PostgreSQL Schema

-- ============================================================
-- 1. Organisms & Metabolic Configuration
-- ============================================================

CREATE TABLE organisms (
    id              VARCHAR(50) PRIMARY KEY,       -- e.g., 'ecoli_k12'
    name            VARCHAR(200) NOT NULL,          -- 'Escherichia coli K-12 MG1655'
    organism_type   VARCHAR(20) NOT NULL,           -- 'bacteria', 'yeast', 'fungi'
    gram_stain      VARCHAR(10),                    -- 'positive', 'negative', NULL
    description     TEXT,
    genome_model    VARCHAR(100),                   -- e.g., 'iJO1366'
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE organism_growth_params (
    id              SERIAL PRIMARY KEY,
    organism_id     VARCHAR(50) REFERENCES organisms(id) ON DELETE CASCADE,
    model           VARCHAR(30) NOT NULL DEFAULT 'monod',  -- monod, moser, andrews, logistic, etc.
    mu_max          DOUBLE PRECISION NOT NULL,      -- maximum specific growth rate (h⁻¹)
    K_s             DOUBLE PRECISION,               -- substrate affinity constant (g/L)
    K_i             DOUBLE PRECISION,               -- inhibition constant for Andrews/Haldane
    Y_XS            DOUBLE PRECISION NOT NULL,      -- biomass yield on substrate (g/g)
    m_s             DOUBLE PRECISION NOT NULL,      -- maintenance coefficient (g/g/h)
    X_max           DOUBLE PRECISION,               -- max biomass for logistic model
    n               DOUBLE PRECISION DEFAULT 1.0,   -- cooperativity for Moser
    UNIQUE(organism_id)
);

CREATE TABLE organism_death_params (
    id              SERIAL PRIMARY KEY,
    organism_id     VARCHAR(50) REFERENCES organisms(id) ON DELETE CASCADE,
    k_d_max         DOUBLE PRECISION NOT NULL DEFAULT 0.01,  -- max death rate (h⁻¹)
    k_lysis         DOUBLE PRECISION NOT NULL DEFAULT 0.005, -- lysis rate (h⁻¹)
    UNIQUE(organism_id)
);

-- ============================================================
-- 2. Substrates (Nutrients)
-- ============================================================

CREATE TABLE substrates (
    id              VARCHAR(20) PRIMARY KEY,        -- 'glucose', 'glycerol', 'glutamine', etc.
    name            VARCHAR(100) NOT NULL,
    molecular_weight DOUBLE PRECISION,              -- g/mol
    formula         VARCHAR(50),
    category        VARCHAR(30)                     -- 'sugar', 'alcohol', 'amino_acid', 'lipid', 'organic_acid'
);

CREATE TABLE organism_substrates (
    id              SERIAL PRIMARY KEY,
    organism_id     VARCHAR(50) REFERENCES organisms(id) ON DELETE CASCADE,
    substrate_id    VARCHAR(20) REFERENCES substrates(id) ON DELETE CASCADE,
    q_S_max         DOUBLE PRECISION NOT NULL,      -- max specific uptake rate (g/g/h)
    K_m             DOUBLE PRECISION NOT NULL,      -- half-saturation constant (g/L)
    priority        INTEGER NOT NULL DEFAULT 1,      -- consumption order (1 = first)
    Y_XS            DOUBLE PRECISION,               -- override global Y_XS for this substrate
    UNIQUE(organism_id, substrate_id)
);

-- ============================================================
-- 3. Products & Byproducts
-- ============================================================

CREATE TABLE organism_products (
    id              SERIAL PRIMARY KEY,
    organism_id     VARCHAR(50) REFERENCES organisms(id) ON DELETE CASCADE,
    product_name    VARCHAR(100) NOT NULL,          -- 'acetate', 'ethanol', 'lactate', etc.
    product_type    VARCHAR(20) NOT NULL DEFAULT 'byproduct',  -- 'product', 'byproduct'
    model           VARCHAR(30) NOT NULL DEFAULT 'luedeking-piret',
    alpha           DOUBLE PRECISION DEFAULT 0,     -- growth-associated coefficient
    beta            DOUBLE PRECISION DEFAULT 0,     -- non-growth-associated coefficient (h⁻¹)
    Y_PS            DOUBLE PRECISION,               -- product yield on substrate (g/g)
    q_S_crit        DOUBLE PRECISION,               -- overflow metabolism threshold (g/g/h)
    inhibition_type VARCHAR(30),                    -- 'aiba', 'linear', 'none'
    K_p             DOUBLE PRECISION,               -- product inhibition constant (g/L)
    UNIQUE(organism_id, product_name)
);

-- ============================================================
-- 4. Environmental Parameters
-- ============================================================

CREATE TABLE organism_environment (
    id              SERIAL PRIMARY KEY,
    organism_id     VARCHAR(50) REFERENCES organisms(id) ON DELETE CASCADE,
    temp_model      VARCHAR(30) DEFAULT 'cardinal',
    T_min           DOUBLE PRECISION,               -- °C
    T_opt           DOUBLE PRECISION NOT NULL,      -- °C
    T_max           DOUBLE PRECISION,               -- °C
    E_a             DOUBLE PRECISION,               -- activation energy (kJ/mol) for Arrhenius
    pH_min          DOUBLE PRECISION,
    pH_opt          DOUBLE PRECISION NOT NULL,
    pH_max          DOUBLE PRECISION,
    pH_sigma        DOUBLE PRECISION DEFAULT 0.8,   -- tolerance width for Gaussian
    UNIQUE(organism_id)
);

-- ============================================================
-- 5. Oxygen & Heat Parameters
-- ============================================================

CREATE TABLE organism_oxygen_params (
    id              SERIAL PRIMARY KEY,
    organism_id     VARCHAR(50) REFERENCES organisms(id) ON DELETE CASCADE,
    q_O2_max        DOUBLE PRECISION NOT NULL,      -- max specific O2 uptake (g/g/h)
    Y_XO            DOUBLE PRECISION NOT NULL,      -- biomass yield on oxygen (g/g)
    m_O             DOUBLE PRECISION DEFAULT 0,     -- maintenance oxygen (g/g/h)
    UNIQUE(organism_id)
);

CREATE TABLE organism_heat_params (
    id              SERIAL PRIMARY KEY,
    organism_id     VARCHAR(50) REFERENCES organisms(id) ON DELETE CASCADE,
    Y_QX            DOUBLE PRECISION,               -- heat yield from growth (kJ/g)
    Y_QS            DOUBLE PRECISION,               -- heat yield from substrate (kJ/g)
    UNIQUE(organism_id)
);

-- ============================================================
-- 6. Bioreactor Configurations
-- ============================================================

CREATE TABLE bioreactor_configs (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200) NOT NULL,
    organism_id     VARCHAR(50) REFERENCES organisms(id),
    mode            VARCHAR(20) NOT NULL DEFAULT 'batch', -- batch, fed_batch, chemostat, perfusion
    V_initial       DOUBLE PRECISION NOT NULL,      -- initial volume (L)
    V_max           DOUBLE PRECISION,               -- max volume for fed-batch
    kLa             DOUBLE PRECISION DEFAULT 150,   -- mass transfer coefficient (h⁻¹)
    O2_star         DOUBLE PRECISION DEFAULT 0.007, -- oxygen saturation (g/L)
    T_setpoint      DOUBLE PRECISION DEFAULT 37,    -- temperature setpoint (°C)
    pH_setpoint     DOUBLE PRECISION DEFAULT 7.0,
    heat_transfer   DOUBLE PRECISION DEFAULT 0.5,   -- kJ/h/K
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

-- Feed configuration for fed-batch / chemostat
CREATE TABLE bioreactor_feeds (
    id              SERIAL PRIMARY KEY,
    config_id       INTEGER REFERENCES bioreactor_configs(id) ON DELETE CASCADE,
    substrate_id    VARCHAR(20) REFERENCES substrates(id),
    feed_name       VARCHAR(100),
    concentration   DOUBLE PRECISION NOT NULL,      -- g/L in feed
    UNIQUE(config_id, substrate_id)
);

-- ============================================================
-- 7. Simulation Runs
-- ============================================================

CREATE TABLE simulation_runs (
    id              SERIAL PRIMARY KEY,
    config_id       INTEGER REFERENCES bioreactor_configs(id),
    name            VARCHAR(200),
    description     TEXT,
    t_start         DOUBLE PRECISION DEFAULT 0,     -- start time (h)
    t_end           DOUBLE PRECISION NOT NULL,      -- end time (h)
    dt              DOUBLE PRECISION DEFAULT 0.1,   -- time step (h)
    solver          VARCHAR(30) DEFAULT 'LSODA',    -- ODE solver
    status          VARCHAR(20) DEFAULT 'pending',   -- pending, running, completed, failed
    created_at      TIMESTAMP DEFAULT NOW(),
    started_at      TIMESTAMP,
    completed_at    TIMESTAMP
);

-- Initial conditions for a simulation
CREATE TABLE simulation_initial_conditions (
    id              SERIAL PRIMARY KEY,
    simulation_id   INTEGER REFERENCES simulation_runs(id) ON DELETE CASCADE,
    X_v             DOUBLE PRECISION NOT NULL,      -- viable biomass (g/L)
    X_d             DOUBLE PRECISION DEFAULT 0,     -- dead biomass (g/L)
    S               DOUBLE PRECISION NOT NULL,      -- substrate concentration (g/L)
    P               DOUBLE PRECISION DEFAULT 0,     -- product concentration (g/L)
    B               DOUBLE PRECISION DEFAULT 0,     -- byproduct concentration (g/L)
    dissolved_O2    DOUBLE PRECISION DEFAULT 0.007, -- dissolved oxygen (g/L)
    V               DOUBLE PRECISION NOT NULL,      -- volume (L)
    temperature     DOUBLE PRECISION DEFAULT 37,    -- temperature (°C)
    pH              DOUBLE PRECISION DEFAULT 7.0
);

-- ============================================================
-- 8. Simulation Results (Time Series)
-- ============================================================

CREATE TABLE simulation_results (
    id              SERIAL PRIMARY KEY,
    simulation_id   INTEGER REFERENCES simulation_runs(id) ON DELETE CASCADE,
    t               DOUBLE PRECISION NOT NULL,      -- time (h)
    X_v             DOUBLE PRECISION,               -- viable biomass
    X_d             DOUBLE PRECISION,               -- dead biomass
    S               DOUBLE PRECISION,               -- substrate
    P               DOUBLE PRECISION,               -- product
    B               DOUBLE PRECISION,               -- byproduct
    dissolved_O2    DOUBLE PRECISION,               -- dissolved oxygen
    V               DOUBLE PRECISION,               -- volume
    temperature     DOUBLE PRECISION,               -- temperature
    pH              DOUBLE PRECISION,
    mu              DOUBLE PRECISION,               -- specific growth rate
    q_S             DOUBLE PRECISION,               -- specific substrate uptake rate
    q_P             DOUBLE PRECISION,               -- specific product formation rate
    q_B             DOUBLE PRECISION,               -- specific byproduct formation rate
    q_O2            DOUBLE PRECISION,               -- specific oxygen uptake
    viability       DOUBLE PRECISION,               -- X_v / (X_v + X_d)
    
    UNIQUE(simulation_id, t)
);

CREATE INDEX idx_results_sim_time ON simulation_results(simulation_id, t);
CREATE INDEX idx_results_sim ON simulation_results(simulation_id);

-- ============================================================
-- 9. Experimental Data (imported from CSVs)
-- ============================================================

CREATE TABLE experiments (
    id              SERIAL PRIMARY KEY,
    project         VARCHAR(50) NOT NULL,            -- 'DTB' or 'BRPMM'
    batch_id        VARCHAR(50) NOT NULL,
    organism_id     VARCHAR(50) REFERENCES organisms(id),
    description     TEXT,
    created_at      TIMESTAMP DEFAULT NOW(),
    UNIQUE(project, batch_id)
);

CREATE TABLE experiment_data (
    id              SERIAL PRIMARY KEY,
    experiment_id   INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
    t               DOUBLE PRECISION NOT NULL,      -- elapsed time (h)
    vcd             DOUBLE PRECISION,               -- viable cell density
    tcd             DOUBLE PRECISION,               -- total cell density
    substrate       DOUBLE PRECISION,
    byproduct       DOUBLE PRECISION,
    volume          DOUBLE PRECISION,
    feed_substrate  DOUBLE PRECISION,
    feed_b          DOUBLE PRECISION,
    feed_c          DOUBLE PRECISION,
    feed_d          DOUBLE PRECISION,
    -- BRPMM fields
    flow_rate_1     DOUBLE PRECISION,
    density         DOUBLE PRECISION,
    flow_rate_2     DOUBLE PRECISION,
    pressure_1      DOUBLE PRECISION,
    temperature_1   DOUBLE PRECISION,
    temperature_2   DOUBLE PRECISION,
    temperature_3   DOUBLE PRECISION,
    temperature_4   DOUBLE PRECISION,
    pressure_2      DOUBLE PRECISION,
    pressure_3      DOUBLE PRECISION,
    pressure_4      DOUBLE PRECISION,
    
    to_predict      BOOLEAN DEFAULT FALSE,
    label           VARCHAR(20),                     -- 'accepted' or 'rejected' for BRPMM
    
    UNIQUE(experiment_id, t)
);

CREATE INDEX idx_exp_data ON experiment_data(experiment_id, t);

-- ============================================================
-- 10. Parameter Estimation Runs
-- ============================================================

CREATE TABLE parameter_estimation (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(200),
    organism_id     VARCHAR(50) REFERENCES organisms(id),
    experiment_id   INTEGER REFERENCES experiments(id),
    simulation_id   INTEGER REFERENCES simulation_runs(id),
    method          VARCHAR(50) DEFAULT 'least_squares',
    params_fitted   JSONB,                         -- which params were fitted
    params_initial  JSONB,                         -- initial guess
    params_final    JSONB,                         -- fitted values
    rmse            DOUBLE PRECISION,               -- root mean squared error
    r_squared       DOUBLE PRECISION,
    status          VARCHAR(20) DEFAULT 'pending',
    created_at      TIMESTAMP DEFAULT NOW()
);
