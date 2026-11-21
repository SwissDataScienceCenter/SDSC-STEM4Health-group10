# Bioreactor Process Application

Full-stack application for monitoring, simulating, and analyzing bioreactor processes.

## Architecture

```
bioreactor-app/
├── backend/           # Java 21 + Spring Boot REST API
│   └── src/main/java/com/bioreactor/
│       ├── model/     # Domain entities (Batch, Measurement, etc.)
│       ├── repository/# Data access (JPA)
│       ├── service/   # Business logic & calculations
│       ├── controller/# REST endpoints
│       └── config/    # App configuration
├── frontend/          # Streamlit/Python web UI
├── database/          # PostgreSQL schema & migrations
│   └── migrations/    # Flyway SQL migrations
└── docs/              # Documentation
```

## Tech Stack

- **Backend:** Java 21, Spring Boot, JPA
- **Database:** PostgreSQL 18
- **Frontend:** Streamlit (Python) or Gradio
- **Data:** Project 1 (DTB) and Project 2 (BRPMM) CSVs

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
