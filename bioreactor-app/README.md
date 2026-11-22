# Bioreactor Process Application

Full-stack application for monitoring, simulating, and analyzing bioreactor processes.

## Architecture

```
bioreactor-app/
├── backend/           # Java 21 + Spring Boot REST API
│   └── src/main/java/com/bioreactor/
│       ├── engine/    # Pure Java ODE engine (Simulator, Fitter, RK4, models)
│       ├── dtb/       # DTB challenge: V3 model, CSV reader, submission, CLI
│       ├── prediction/# PredictionEngine with uncertainty bands
│       ├── monitoring/# MonitoringEngine: reference bands, findings
│       ├── trending/  # TrendingEngine: I-MR charts, Cpk, run metrics
│       ├── model/     # JPA entities
│       ├── repository/# Data access (JPA)
│       ├── service/   # Business logic, CSV import, process analysis
│       ├── controller/# REST endpoints (v1 + v2 + DTB)
│       ├── vaadin/    # Vaadin Flow dashboard (3 tabs)
│       └── config/    # App configuration
├── frontend/          # Streamlit/Python web UI (team shared)
├── database/          # PostgreSQL schema & migrations
│   └── migrations/    # Flyway SQL migrations (V1–V7)
└── docs/              # Documentation
```

## Tech Stack

- **Backend:** Java 21, Spring Boot 3.4.1, JPA, Vaadin 24.5
- **Database:** PostgreSQL 18 (Flyway migrations)
- **Frontend:** Vaadin Flow (built-in) + Streamlit (team shared)
- **Data:** Project 1 (DTB) and Project 2 (BRPMM) CSVs
- **API Docs:** OpenAPI 3 at /v3/api-docs and /swagger-ui.html (springdoc 2.7.0)

## Quick Status

| Metric | Value |
|--------|-------|
| Tests | 40 (32 fast, 8 slow @Tag("slow")) |
| `mvn verify` | <2 min (fast), ~12 min (full) |
| DTB V3 holdout MAPE | 15.3% (VCD 8.4%, TCD 8.6%, Sub 22.2%, Byp 22.1%) |
| DTB V3 7-fold CV MAPE | 13.74% — matches Python 13.8% |
| Python → Java port error | < 1e-6 (Monod quantified) |

## Getting Started

### Database
```bash
# PostgreSQL already running on localhost:5432
# Database: bioreactor | User: appuser | Password: bioreactor123
```

### Backend
```bash
cd backend
./mvnw spring-boot:run
```

### Frontend
```bash
source ../../.venv/bin/activate
cd frontend
streamlit run app.py
```

## Git Strategy

- **This branch:** `bioreactor-app` — our work
- **`main` branch:** team's shared work — we learn from it, don't disturb it
