FROM python:3.11-slim

RUN pip install streamlit plotly requests

WORKDIR /app

COPY streamlit_app.py /app/
COPY requirements.txt /app/ 2>/dev/null || true

EXPOSE 8501

ENV API_URL=http://java-api:8080/api

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
