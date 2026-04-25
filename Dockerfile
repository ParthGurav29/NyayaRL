# ── NyayaRL Environment Server ───────────────────────────────────────────────
# Single-stage, minimal image. No ML libraries — Track 1 is logic-only.

FROM python:3.11-slim

WORKDIR /app

# ── Dependencies first (layer cache optimisation) ────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ──────────────────────────────────────────────────────
COPY . .

EXPOSE 7860

CMD ["python", "gradio_app.py"]
