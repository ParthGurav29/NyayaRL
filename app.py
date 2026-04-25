"""
app.py — Hugging Face Spaces entrypoint.

Runs a single ASGI app on port 7860:
- Gradio UI mounted at `/`
- FastAPI environment API mounted at `/api`
"""

from __future__ import annotations

from fastapi import FastAPI
import gradio as gr

from gradio_app import build_app
from server.app import app as api_app


app = FastAPI(title="NyayaRL Space")
app.mount("/api", api_app)

gradio_blocks = build_app()
app = gr.mount_gradio_app(app, gradio_blocks, path="/")

