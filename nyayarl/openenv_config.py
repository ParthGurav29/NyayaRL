"""
openenv_config.py — Loader for `nyayarl/openenv.yaml`.

This is a lightweight way to configure whether NyayaRL uses a local in-process
environment or a remote HTTP environment (HF Spaces / OpenEnv-style).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class OpenEnvRemoteConfig:
    base_url: str
    api_path: str = "/api"
    timeout_seconds: int = 30

    @property
    def api_base_url(self) -> str:
        return self.base_url.rstrip("/") + self.api_path


@dataclass(frozen=True)
class OpenEnvSessionConfig:
    session_id: str = ""


@dataclass(frozen=True)
class OpenEnvConfig:
    mode: str = "local"  # local | remote
    remote: OpenEnvRemoteConfig | None = None
    session: OpenEnvSessionConfig = OpenEnvSessionConfig()


def load_openenv_config(path: str | Path | None = None) -> OpenEnvConfig:
    p = Path(path) if path is not None else Path(__file__).parent / "openenv.yaml"
    if not p.exists():
        return OpenEnvConfig()

    data = yaml.safe_load(p.read_text()) or {}
    mode = str(data.get("mode", "local")).strip().lower()

    remote_data = data.get("remote") or {}
    remote = None
    if mode == "remote":
        base_url = str(remote_data.get("base_url", "")).strip()
        if not base_url:
            raise ValueError("openenv.yaml: remote.base_url is required when mode=remote")
        remote = OpenEnvRemoteConfig(
            base_url=base_url,
            api_path=str(remote_data.get("api_path", "/api")),
            timeout_seconds=int(remote_data.get("timeout_seconds", 30)),
        )

    sess_data = data.get("session") or {}
    session = OpenEnvSessionConfig(session_id=str(sess_data.get("session_id", "")).strip())

    return OpenEnvConfig(mode=mode, remote=remote, session=session)

