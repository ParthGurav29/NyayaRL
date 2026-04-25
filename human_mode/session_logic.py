from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from nyayarl.agents import JudgeAgent, ProsecutionAgent
from nyayarl.case_generator import Track2CaseGenerator
from nyayarl.environment import NyayaRLEnvironment
from nyayarl.precedents import PrecedentsDB
from nyayarl.track2_adapters import Track2JudgeAdapter, Track2ProsecutionAdapter

from nyayarl.models import Action, JudgmentLabel, Observation, StepResult, StepType
from nyayarl.openenv_config import load_openenv_config
from nyayarl.client import NyayaRLEnvClient


def build_environment() -> NyayaRLEnvironment:
    cfg = load_openenv_config()
    if cfg.mode == "remote":
        # Remote mode uses HTTP client; environment stays local in Gradio/CLI.
        # Callers should use `NyayaRLEnvClient` directly for remote stepping.
        # We still return a local env so the UI remains functional.
        pass

    precedents = PrecedentsDB()
    judge = Track2JudgeAdapter(JudgeAgent(), precedents)
    prosecution = Track2ProsecutionAdapter(ProsecutionAgent())
    case_generator = Track2CaseGenerator()
    return NyayaRLEnvironment(case_generator=case_generator, judge=judge, prosecution=prosecution)


def build_remote_client() -> NyayaRLEnvClient:
    """
    Build an HTTP client from `nyayarl/openenv.yaml`.
    """
    cfg = load_openenv_config()
    if cfg.mode != "remote" or cfg.remote is None:
        raise RuntimeError("openenv.yaml mode is not 'remote'")
    return NyayaRLEnvClient(
        cfg.remote.api_base_url,
        timeout_seconds=cfg.remote.timeout_seconds,
    )


def next_step_type(observation: Observation) -> StepType:
    idx = len(observation.submitted_steps)
    order = [
        StepType.ACTUS_REUS,
        StepType.MENS_REA,
        StepType.LINKAGE,
        StepType.COUNTER_ARGUMENT,
        StepType.IPC_APPLICATION,
        StepType.PRECEDENT_CITATION,
    ]
    return order[min(idx, 5)]


def format_case_file(observation: Observation) -> str:
    cf = observation.case_file
    parts: list[str] = []
    parts.append(f"Case ID: {cf.case_id}")
    if cf.template_id:
        parts.append(f"Template: {cf.template_id}")
    parts.append(f"Curriculum level: {cf.curriculum_level}")
    parts.append("")
    parts.append("FIR:")
    parts.append(cf.fir)
    parts.append("")
    parts.append(f"Accused count: {cf.accused_count}")
    parts.append("")
    parts.append("Applicable IPC sections: " + ", ".join(cf.applicable_ipc_sections))
    parts.append("")
    parts.append("Evidence:")
    for e in cf.evidence_items:
        parts.append(f"- {e.id} [{e.type.value}] present={e.is_present}: {e.description}")
    parts.append("")
    parts.append("Witnesses:")
    for w in cf.witness_statements:
        parts.append(
            f"- {w.id} rel={w.reliability:.2f} contradict={w.is_contradicting}: {w.content}"
        )
    return "\n".join(parts)


def format_step_result(step_type: StepType, result: StepResult, done: bool, obs: Observation) -> str:
    lines = [
        f"Step: {step_type.value}",
        f"Valid: {result.is_valid}",
        f"Reward: {result.reward}",
    ]
    if result.failure_reason:
        lines.append(f"Failure reason: {result.failure_reason}")
    if result.prosecution_challenge:
        lines.append(f"Prosecution challenge: {result.prosecution_challenge}")
    if obs.prosecution_challenges:
        lines.append("")
        lines.append("All prosecution challenges so far:")
        for c in obs.prosecution_challenges:
            lines.append(f"- {c}")
    lines.append("")
    lines.append(f"Done: {done}")
    lines.append(f"Chain score: {obs.chain_score:.2f}")
    return "\n".join(lines)


def ensure_sessions_dir() -> Path:
    d = Path("human_mode") / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_session_log(payload: dict[str, Any], *, filename: str) -> Path:
    d = ensure_sessions_dir()
    path = d / filename
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path

