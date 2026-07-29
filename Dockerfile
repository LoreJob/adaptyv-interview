# --- stage 1: build the React frontend ---
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- stage 2: python backend that also serves the built frontend ---
FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# app code + data the model needs at runtime (processed parquet + reference CSVs)
COPY src/ ./src/
COPY data/ ./data/
COPY reference/ ./reference/

# built frontend from stage 1
COPY --from=frontend /fe/dist ./frontend/dist

# precompute heavy endpoints (metrics/backtest) so the server never trains at
# request time -> fast, low memory on small hosts.
RUN python -m src.build_cache

ENV PORT=8000
EXPOSE 8000
# Render provides $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
