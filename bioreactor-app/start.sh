#!/bin/bash
# Bioreactor App — Startup Script
# Starts PostgreSQL, Java API, and Streamlit UI in sequence.
# Usage: ./start.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
JAVA_HOME="/home/renku/work/.jdks/jdk-21.0.13.1+2"
PYTHON="/home/renku/work/.venv/bin/python3"
PGBIN="/home/renku/work/.miniforge/bin"
MAVEN="/home/renku/work/.vscode/extensions/oracle.oracle-java-27.0.1-universal/nbcode/java/maven/bin/mvn.sh"

export PATH="$JAVA_HOME/bin:$PGBIN:$PATH"

echo "=== Bioreactor Application ==="
echo ""

# 1. PostgreSQL
echo "[1/3] Starting PostgreSQL..."
if pg_isready -q 2>/dev/null; then
    echo "  Already running."
else
    pg_ctl -D /home/renku/work/pgdata -l /tmp/pg.log start 2>/dev/null || {
        initdb -D /home/renku/work/pgdata 2>/dev/null
        pg_ctl -D /home/renku/work/pgdata -l /tmp/pg.log start
        sleep 2
        psql -U postgres -c "CREATE DATABASE bioreactor;" postgres 2>/dev/null || true
        psql -U postgres -c "CREATE USER appuser WITH PASSWORD 'bioreactor123';" postgres 2>/dev/null || true
        psql -U postgres -c "GRANT ALL PRIVILEGES ON DATABASE bioreactor TO appuser;" postgres 2>/dev/null || true
    }
    echo "  Started."
fi

# 2. Java API
echo "[2/3] Building Java API..."
cd "$PROJECT_DIR/backend"
bash "$MAVEN" clean package -DskipTests -q
echo "  Build complete."

echo "[2/3] Starting Java API on :8080..."
pkill -f "bioreactor-app" 2>/dev/null || true
nohup java -jar target/bioreactor-app-0.1.0.jar > /tmp/api.log 2>&1 &
sleep 5
echo "  Started. (log: /tmp/api.log)"

# 3. Streamlit
echo "[3/3] Starting Streamlit UI on :8501..."
cd "$PROJECT_DIR"
pkill -f streamlit 2>/dev/null || true
nohup streamlit run frontend/streamlit_app.py --server.port 8501 --server.headless true > /tmp/streamlit.log 2>&1 &
sleep 3

echo ""
echo "=== All services running ==="
echo "  Streamlit UI:  http://localhost:8501"
echo "  Java API:      http://localhost:8080/api/health"
echo "  PostgreSQL:    postgresql://appuser:bioreactor123@localhost:5432/bioreactor"
echo ""
echo "To stop: ./stop.sh"
