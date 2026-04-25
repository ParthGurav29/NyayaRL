"""
client.py — Minimal HTTP client for the NyayaRL Environment API.

This is used when the environment is running remotely (e.g., HF Spaces) and
training/evaluation needs to interact via HTTP rather than in-process calls.

Endpoints expected (mounted under `/api` in the Space entrypoint):
- GET  /health
- POST /reset
- POST /step
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any

from nyayarl.models import Action, JudgmentLabel, StepType


def _jsonable(x: Any) -> Any:
    if is_dataclass(x):
        return {k: _jsonable(v) for k, v in asdict(x).items()}
    if isinstance(x, Enum):
        return x.value
    if isinstance(x, list):
        return [_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    return x


class NyayaRLEnvClient:
    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")

    def health(self) -> dict[str, Any]:
        return self._get("/health")

    def reset(self, *, session_id: str, curriculum_level: int = 1) -> dict[str, Any]:
        return self._post(
            "/reset",
            {"session_id": session_id, "curriculum_level": int(curriculum_level)},
        )

    def step(self, *, session_id: str, action: Action) -> dict[str, Any]:
        payload = {
            "session_id": session_id,
            "step_type": action.step_type.value,
            "cited_ipc_sections": list(action.cited_ipc_sections),
            "anchored_evidence_ids": list(action.anchored_evidence_ids),
            "anchored_witness_ids": list(action.anchored_witness_ids),
            "judgment": action.judgment.value if action.judgment else None,
        }
        return self._post("/step", payload)

    # --- internals ---
    def _get(self, path: str) -> dict[str, Any]:
        req = urllib.request.Request(self._base + path, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')}") from e

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(_jsonable(body)).encode("utf-8")
        req = urllib.request.Request(
            self._base + path,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'ignore')}") from e

