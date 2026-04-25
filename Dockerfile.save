# ── NyayaRL — HF Spaces Deployment ───────────────────────────────────────────
# Full system: FastAPI server + Gradio interface + trained checkpoint.

FROM python:3.11-slim

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (layer-cached) ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ──────────────────────────────────────────────────────────────
COPY . .

# ── Trained checkpoint (if available) ────────────────────────────────────────
# Keep build working even if only an empty checkpoints/ directory exists.
RUN mkdir -p /app/checkpoints
COPY checkpoints/ /app/checkpoints/

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
