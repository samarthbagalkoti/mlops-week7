# Multi-stage for clean, small image
FROM python:3.11-slim AS base

WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONUNBUFFERED=1
# The service reads MODEL_STAGE and MLFLOW_TRACKING_URI at runtime
EXPOSE 8082
CMD ["uvicorn", "src.serve_app:app", "--host", "0.0.0.0", "--port", "8082"]

