# Serving image for the API and the dashboard (no model or training code inside).
# Forecast files are produced on the host with `python -m src.forecast` and mounted at runtime.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVING_DIR=/app/data/serving

WORKDIR /app
COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt

COPY api/ api/
COPY dashboard/ dashboard/
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
