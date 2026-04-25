# ── NyayaRL Environment Server ───────────────────────────────────────────────
# Single-stage, minimal image. No ML libraries — Track 1 is logic-only.

FROM python:3.11-slim

WORKDIR /app

# ── Dependencies first (layer cache optimisation) ────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ──────────────────────────────────────────────────────
COPY . .

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
