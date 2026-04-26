 
"""
gradio_app.py — Gradio adapter for NyayaRL.

Courtroom demo: case state lives on the FastAPI server (`/reset`, `/step`);
the defence agent (trained checkpoint or stub) runs locally to choose actions.
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import json
import os
import textwrap
import random
import time
import uuid
import socket
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import gradio as gr
import torch
import torch.nn as nn

from nyayarl.checkpointing import load_defence_agent_checkpoint
from nyayarl.agents import DefenceAgent as TrainedDefenceAgent
from nyayarl.client import NyayaRLEnvClient
from nyayarl.models import (
    Action,
    CaseFile,
    EvidenceItem,
    EvidenceType,
    JudgmentLabel,
    Observation,
    StepResult,
    StepType,
    WitnessStatement,
)
from nyayarl.precedents import PrecedentsDB


def _set_seed(seed: int) -> None:
    """
    Best-effort deterministic mode for local verification.

    Note: full determinism is not guaranteed across all PyTorch ops / backends,
    but this removes the common sources of run-to-run drift (Python RNG, Torch RNG).
    """
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass


# ── Constants ───────────────────────────────────────────────────────────────

_STEP_SEQUENCE = [
    StepType.ACTUS_REUS,
    StepType.MENS_REA,
    StepType.LINKAGE,
    StepType.COUNTER_ARGUMENT,
    StepType.IPC_APPLICATION,
    StepType.PRECEDENT_CITATION,
]

_STEP_LABELS = [
    "Step 1 — Actus Reus",
    "Step 2 — Mens Rea",
    "Step 3 — Linkage",
    "Step 4 — Counter Argument",
    "Step 5 — IPC Application",
    "Step 6 — Precedent Citation",
]

BASE_DIR = Path(__file__).resolve().parent

_SESSIONS_DIR = Path(
    os.getenv("NYAYARL_SESSIONS_DIR", str(BASE_DIR / "human_mode" / "sessions"))
)
_CHECKPOINT_DIR = Path(
    os.getenv("NYAYARL_CHECKPOINT_DIR", str(BASE_DIR / "checkpoints"))
)
_SEED_ENV = os.getenv("NYAYARL_SEED")
_SEED: int | None = int(_SEED_ENV) if _SEED_ENV is not None and str(_SEED_ENV).strip() else None

_API_BASE = os.getenv("NYAYARL_API_BASE", "http://127.0.0.1:8000").rstrip("/")
_STEP_DELAY_SEC = float(os.getenv("NYAYARL_STEP_UI_DELAY", "0.8"))
_CLAUDE_MODEL = os.getenv("NYAYARL_CLAUDE_MODEL", "claude-sonnet-4-20250514")

_PRECEDENTS_DB: PrecedentsDB | None = None


def _precedents_db() -> PrecedentsDB:
    global _PRECEDENTS_DB
    if _PRECEDENTS_DB is None:
        _PRECEDENTS_DB = PrecedentsDB()
    return _PRECEDENTS_DB


# ── Serialisation helper ────────────────────────────────────────────────────


def _serialize(obj: Any) -> Any:
    """Recursively convert dataclasses / enums to JSON-safe dicts."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


class StubDefenceAgent(nn.Module):
    """Deterministic stub agent for demo mode."""

    def __init__(self) -> None:
        super().__init__()
        self._param = nn.Parameter(torch.tensor(0.0))

    def select_action(self, observation: Observation) -> Action:
        step_idx = min(len(observation.submitted_steps), 5)
        step_type = _STEP_SEQUENCE[step_idx]
        cf = observation.case_file

        if step_type == StepType.ACTUS_REUS:
            eids = [e.id for e in cf.evidence_items if e.is_present and e.type in (EvidenceType.PHYSICAL, EvidenceType.FORENSIC)]
            return Action(step_type=step_type, anchored_evidence_ids=eids[:1] or ["E1"])
        if step_type == StepType.MENS_REA:
            wids = [w.id for w in cf.witness_statements if w.reliability > 0.5]
            return Action(step_type=step_type, anchored_witness_ids=wids[:1] or ["W1"])
        if step_type == StepType.LINKAGE:
            eids = [e.id for e in cf.evidence_items if e.is_present and e.type in (EvidenceType.DOCUMENTARY, EvidenceType.FORENSIC)]
            return Action(
                step_type=step_type,
                anchored_evidence_ids=eids[:1] or ["E3"],
                cited_ipc_sections=cf.applicable_ipc_sections[:1],
            )
        if step_type == StepType.COUNTER_ARGUMENT:
            wids = [w.id for w in cf.witness_statements if w.reliability > 0.5]
            return Action(step_type=step_type, anchored_witness_ids=wids[:1] or ["W1"])
        if step_type == StepType.IPC_APPLICATION:
            return Action(step_type=step_type, cited_ipc_sections=cf.applicable_ipc_sections[:1])
        return Action(step_type=step_type, judgment=JudgmentLabel.CONVICT, anchored_evidence_ids=[cf.precedent_id])

    def get_log_prob(self, observation: Observation, action: Action) -> torch.Tensor:
        return torch.tensor(-1.0)

    def get_entropy(self, observation: Observation) -> torch.Tensor:
        return torch.tensor(1.0)


# ── Checkpoint loading ──────────────────────────────────────────────────────


def _load_agent() -> nn.Module:

    """
    Attempt to load a trained checkpoint. Falls back to stub agent.
    """
    global _DEMO_MODE, _MODEL_LOAD_STATUS
    require_ckpt = os.getenv("NYAYARL_REQUIRE_CHECKPOINT") == "1"
    allow_legacy = os.getenv("NYAYARL_ALLOW_LEGACY_CHECKPOINT") == "1"
    agent: nn.Module = TrainedDefenceAgent(rng_seed=_SEED)

    if _CHECKPOINT_DIR.exists():
        # latest = _CHECKPOINT_DIR / "latest.pt"
        latest = _CHECKPOINT_DIR / "step_004999.pt"
        if latest.exists():
            ckpt_path = latest
        else:
            preferred = sorted(_CHECKPOINT_DIR.glob("step_*.pt"))
            ckpt_path = preferred[-1] if preferred else None

        if ckpt_path is not None:
            try:
                res = load_defence_agent_checkpoint(
                    checkpoint_path=ckpt_path,
                    agent=agent,
                    map_location="cpu",
                    require_full_match=not allow_legacy,
                    allow_legacy_partial=allow_legacy,
                    do_value_check=True,
                )
                if res.was_partial_load:
                    print(
                        "⚠ PARTIAL CHECKPOINT LOAD: model parameters were not fully restored. "
                        f"missing={len(res.diff.missing_in_checkpoint)} "
                        f"unexpected={len(res.diff.unexpected_in_checkpoint)} "
                        f"shape_mismatches={len(res.diff.shape_mismatches)} "
                        f"legacy={res.was_legacy_remap}"
                    )
                    if not allow_legacy:
                        raise RuntimeError("Partial load occurred while legacy loads are disabled.")
                else:
                    print("✅ Loaded checkpoint (strict) and verified values.")
                print(
                    f"✓ Loaded checkpoint: {os.path.realpath(str(ckpt_path))} "
                    f"(selected={ckpt_path} dir={_CHECKPOINT_DIR})"
                )
                _DEMO_MODE = bool(res.was_partial_load)
                _MODEL_LOAD_STATUS = "partial" if _DEMO_MODE else "strict"
                agent.eval()
                return agent
            except Exception as e:
                # If strict load fails, retry once with a partial load (drops mismatched tensors).
                # This keeps the UI using *mostly* trained weights instead of falling back to demo mode.
                try:
                    res = load_defence_agent_checkpoint(
                        checkpoint_path=ckpt_path,
                        agent=agent,
                        map_location="cpu",
                        require_full_match=False,
                        allow_legacy_partial=True,
                        do_value_check=False,
                    )
                    print(
                        "⚠ Loaded checkpoint with PARTIAL compatibility mode. "
                        f"missing={len(res.diff.missing_in_checkpoint)} "
                        f"unexpected={len(res.diff.unexpected_in_checkpoint)} "
                        f"shape_mismatches={len(res.diff.shape_mismatches)} "
                        f"legacy={res.was_legacy_remap}"
                    )
                    print(
                        f"✓ Loaded checkpoint: {os.path.realpath(str(ckpt_path))} "
                        f"(selected={ckpt_path} dir={_CHECKPOINT_DIR})"
                    )
                    _DEMO_MODE = False
                    _MODEL_LOAD_STATUS = "partial_compat"
                    agent.eval()
                    return agent
                except Exception:
                    print(f"⚠ Failed to load checkpoint from {ckpt_path}: {e}")
                if require_ckpt:
                    raise RuntimeError(
                        f"CRITICAL failure: Checkpoint load failed from {ckpt_path} "
                        f"(dir={_CHECKPOINT_DIR}). Deployment aborted."
                    ) from e
        else:
            print(
                f"⚠ No step_*.pt found in checkpoint dir: {_CHECKPOINT_DIR} "
                f"(set NYAYARL_CHECKPOINT_DIR to override)"
            )
            if require_ckpt:
                raise RuntimeError(
                    f"CRITICAL failure: No checkpoint found in {_CHECKPOINT_DIR}. Deployment aborted."
                )
    else:
        print(
            f"⚠ Checkpoint dir does not exist: {_CHECKPOINT_DIR} "
            f"(set NYAYARL_CHECKPOINT_DIR to override)"
        )
        if require_ckpt:
            raise RuntimeError(
                f"CRITICAL failure: Checkpoint dir does not exist: {_CHECKPOINT_DIR}. Deployment aborted."
            )

    _DEMO_MODE = True
    _MODEL_LOAD_STATUS = "stub"
    print("⚠ No trained checkpoint — running in demo mode")
    agent = StubDefenceAgent()
    agent.eval()
    return agent


_agent: nn.Module | None = None
_DEMO_MODE: bool = False
_MODEL_LOAD_STATUS: str = "unknown"


def _get_agent() -> nn.Module:
    global _agent
    if _agent is None:
        _agent = _load_agent()
    return _agent


# ── Legacy text formatters (unused by courtroom UI; kept for tooling) ───────


def _format_case_file(cf: CaseFile) -> str:
    """Render a case file as readable text."""
    lines = [
        f"━━━ CASE FILE — {cf.case_id} ━━━",
        "",
        f"FIR: {cf.fir}",
        "",
        "EVIDENCE ITEMS:",
    ]
    for e in cf.evidence_items:
        status = "✓ Present" if e.is_present else "✗ Missing"
        lines.append(f"  [{e.id}] {e.type.value.upper():12s} — {e.description}  {status}")

    lines.append("")
    lines.append("WITNESS STATEMENTS:")
    for w in cf.witness_statements:
        flag = " ⚠ CONTRADICTING" if w.is_contradicting else ""
        lines.append(f"  [{w.id}] Reliability: {w.reliability:.1f} — {w.content}{flag}")

    lines.append("")
    lines.append(f"APPLICABLE IPC SECTIONS: {', '.join(cf.applicable_ipc_sections)}")
    lines.append(f"CURRICULUM LEVEL: {cf.curriculum_level}")

    if _DEMO_MODE:
        lines.insert(0, "⚠ DEMO MODE — No trained model loaded. Agent uses heuristic actions.\n")

    return "\n".join(lines)


def _format_step_result(step_num: int, sr: StepResult, chain_score: float) -> str:
    """Render a step result as readable text."""
    lines = [
        f"━━━ STEP {step_num} RESULT ━━━",
        f"Valid:  {'✅ YES' if sr.is_valid else '❌ NO'}",
        f"Reward: {sr.reward:+.2f}",
    ]
    if sr.failure_reason:
        lines.append(f"Reason: {sr.failure_reason}")
    lines.append("")
    if sr.prosecution_challenge:
        lines.append(f"🏛 PROSECUTION: {sr.prosecution_challenge}")
    else:
        lines.append("🏛 PROSECUTION: No challenge raised")
    lines.append(f"\nChain Score: {chain_score:.2f}")
    return "\n".join(lines)


def _format_verdict(obs: Observation, total_reward: float, steps: int) -> str:
    """Render the final verdict."""
    valid = len(obs.submitted_steps)
    challenges = len(obs.prosecution_challenges)
    pwr = (challenges / steps * 100) if steps > 0 else 0

    if valid == 6:
        outcome = "🏆 WIN — All 6 steps validated!"
    elif obs.chain_score >= 0.5:
        outcome = "🤝 DRAW — Partial chain completed."
    else:
        outcome = "💀 LOSS — Argument chain failed."

    j_text = "N/A"
    if obs.submitted_steps and obs.submitted_steps[-1].judgment:
        j_text = obs.submitted_steps[-1].judgment.value.upper()

    return "\n".join([
        "━━━ FINAL VERDICT ━━━",
        f"Judgment:         {j_text}",
        f"Total reward:     {total_reward:.2f}",
        f"Steps taken:      {steps}",
        f"Valid steps:      {valid}",
        f"Prosecution wins: {pwr:.0f}%",
        "",
        outcome,
    ])


# ── Session log helpers ─────────────────────────────────────────────────────


def _save_session_log(state: dict) -> None:
    """Save session to JSON atomically."""
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = _SESSIONS_DIR / f"session_{ts}.json"
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp.rename(path)


def _load_session_logs() -> list[list[str]]:
    """Load all sessions for the dataframe display."""
    rows = []
    if not _SESSIONS_DIR.exists():
        return rows
    for p in sorted(_SESSIONS_DIR.glob("session_*.json"), reverse=True):
        try:
            with open(p) as f:
                d = json.load(f)
            rows.append([
                d.get("timestamp", "?")[:19],
                d.get("mode", "?"),
                str(d.get("curriculum_level", "?")),
                str(d.get("human_won", "?")),
                f"{d.get('verdict', {}).get('total_reward', 0):.2f}",
                str(d.get("verdict", {}).get("valid_steps", 0)),
            ])
        except Exception:
            continue
    return rows


# ── HTTP API + courtroom UI ─────────────────────────────────────────────────


def _api_client() -> NyayaRLEnvClient:
    return NyayaRLEnvClient(_API_BASE, timeout_seconds=60)


def _dict_to_evidence_item(d: dict[str, Any]) -> EvidenceItem:
    return EvidenceItem(
        id=str(d["id"]),
        description=str(d.get("description", "")),
        type=EvidenceType(str(d["type"])),
        is_present=bool(d.get("is_present", False)),
    )


def _dict_to_witness(d: dict[str, Any]) -> WitnessStatement:
    return WitnessStatement(
        id=str(d["id"]),
        content=str(d.get("content", "")),
        reliability=float(d.get("reliability", 0.0)),
        is_contradicting=bool(d.get("is_contradicting", False)),
    )


def _dict_to_case_file(d: dict[str, Any]) -> CaseFile:
    ev = [_dict_to_evidence_item(x) for x in d.get("evidence_items", [])]
    ws = [_dict_to_witness(x) for x in d.get("witness_statements", [])]
    return CaseFile(
        case_id=str(d["case_id"]),
        fir=str(d.get("fir", "")),
        accused_count=int(d.get("accused_count", 0)),
        template_id=str(d.get("template_id", "")),
        evidence_items=ev,
        witness_statements=ws,
        applicable_ipc_sections=[str(x) for x in d.get("applicable_ipc_sections", [])],
        curriculum_level=int(d.get("curriculum_level", 1)),
        precedent_id=str(d.get("precedent_id", "")),
    )


def _dict_to_action(d: dict[str, Any]) -> Action:
    j = d.get("judgment")
    return Action(
        step_type=StepType(str(d["step_type"])),
        cited_ipc_sections=[str(x) for x in d.get("cited_ipc_sections", [])],
        anchored_evidence_ids=[str(x) for x in d.get("anchored_evidence_ids", [])],
        anchored_witness_ids=[str(x) for x in d.get("anchored_witness_ids", [])],
        judgment=JudgmentLabel(str(j)) if j else None,
    )


def _dict_to_observation(d: dict[str, Any]) -> Observation:
    return Observation(
        case_file=_dict_to_case_file(d["case_file"]),
        submitted_steps=[_dict_to_action(x) for x in d.get("submitted_steps", [])],
        prosecution_challenges=[str(x) for x in d.get("prosecution_challenges", [])],
        chain_score=float(d.get("chain_score", 0.0)),
        curriculum_level=int(d.get("curriculum_level", 1)),
        is_done=bool(d.get("is_done", False)),
    )


def _template_matches_category(template_id: str, category: str) -> bool:
    t = (template_id or "").lower()
    if category == "All":
        return True
    if category == "Homicide":
        return "homicide" in t or t.startswith("302")
    if category == "Robbery":
        return "robbery" in t or "392" in t
    if category == "Assault":
        return "assault" in t or "323" in t
    if category == "Fraud":
        return "fraud" in t or "420" in t
    if category == "Domestic":
        return "domestic" in t or "498a" in t
    return True


def _lookup_evidence(cf: dict[str, Any], eid: str) -> tuple[str, str]:
    for e in cf.get("evidence_items", []):
        if str(e.get("id")) == eid:
            return str(e.get("description", eid)), str(e.get("type", "?"))
    return eid, "?"


def _lookup_witness_line(cf: dict[str, Any], wid: str) -> str:
    for w in cf.get("witness_statements", []):
        if str(w.get("id")) == wid:
            c = str(w.get("content", ""))[:100]
            return f"{wid} — {c}"
    return wid


def _precedent_alignment(case_file: dict[str, Any], judgment_value: str | None) -> tuple[bool, str]:
    pid = str(case_file.get("precedent_id", ""))
    if not judgment_value:
        return False, ""
    prec = _precedents_db().get_precedent(pid) or {}
    gt = str(prec.get("judgment", "partial"))
    return judgment_value == gt, gt


def _confidence_bar(pct: float, blocks: int = 12) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(pct / 100.0 * blocks))
    filled = max(0, min(blocks, filled))
    return "█" * filled + "░" * (blocks - filled)


def generate_explanation(
    *,
    case_file: dict[str, Any],
    submitted_steps: list[dict[str, Any]],
    prosecution_challenges: list[str],
    chain_score: float,
    verdict: str,
) -> str:
    """
    One-shot explainability call (Claude). Returns plain text.
    Called only after the episode is complete (done=True).
    """
    if not submitted_steps:
        return "No steps completed, so no explanation is available."

    api_key = os.getenv("GEMINI_API_KEY", "AIzaSyB2Z1hUptT35AOgSHWU7O9a1sso7VemKC0").strip()
    if not api_key:
        return "Set GEMINI_API_KEY to enable explanations."

    try:
        import google.generativeai as genai
    except Exception:
        return "Install `google-generativeai` to enable explanations."

    evidence_lookup = {
        str(e.get("id")): str(e.get("description", e.get("id", "")))
        for e in case_file.get("evidence_items", [])
    }
    witness_lookup = {
        str(w.get("id")): str(w.get("content", w.get("id", "")))
        for w in case_file.get("witness_statements", [])
    }

    step_summary: list[dict[str, Any]] = []
    for step in submitted_steps:
        ev = [
            evidence_lookup.get(str(i), str(i))
            for i in (step.get("anchored_evidence_ids") or [])
        ]
        wit = [
            witness_lookup.get(str(i), str(i))
            for i in (step.get("anchored_witness_ids") or [])
        ]
        ipc = [str(x) for x in (step.get("cited_ipc_sections") or [])]
        step_summary.append(
            {
                "step": str(step.get("step_type", "")),
                "evidence_used": ev,
                "witnesses_used": wit,
                "ipc_cited": ipc,
                "judgment": step.get("judgment"),
            }
        )

    prompt = f"""You are a legal reasoning explainer for an AI defence agent in the Indian legal system.

The agent just completed a case. Your job is to explain WHY it made the decisions it did, in plain English that a non-lawyer can understand. Write 3-5 sentences maximum. Be specific — mention the actual evidence, witnesses, and IPC sections used. Do not be generic.

CASE TYPE: {case_file.get("template_id", "")}
APPLICABLE IPC SECTIONS: {case_file.get("applicable_ipc_sections", [])}
PRECEDENT: {case_file.get("precedent_id", "")}
VERDICT: {str(verdict).upper()}
CHAIN SCORE: {round(float(chain_score) * 100, 1)}%

STEPS THE AGENT TOOK:
{json.dumps(step_summary, indent=2, ensure_ascii=False)}

PROSECUTION CHALLENGES RAISED:
{prosecution_challenges}

Write the explanation now. Do not use bullet points. Do not start with "The agent". Write as if explaining a courtroom decision to a student."""
    prompt = textwrap.dedent(prompt).strip()

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        if not response.text:
            return "Explanation unavailable (empty response)."
        return str(response.text).strip()
    except Exception as e:
        return f"Error communicating with Gemini: {e}"


_DEMO_CSS = """
<style>
.ny-panel { font-family: system-ui, sans-serif; font-size: 14px; }
.ny-case, .ny-reason { border: 1px solid #ccc; border-radius: 8px; padding: 12px; background: #fafafa; min-height: 200px; }
.ny-case h3, .ny-reason h3 { margin-top: 0; }
.ny-list { margin: 0; padding-left: 1.2rem; }
.ny-rel { color: #555; font-size: 0.9em; }
.ny-fir { white-space: pre-wrap; font-size: 0.85em; max-height: 220px; overflow: auto; }
.ny-card { border-radius: 6px; padding: 8px 10px; margin-bottom: 8px; border: 1px solid #ddd; }
.ny-card-pending { color: #888; background: #f5f5f5; }
.ny-card-running { background: #fff8e6; border-color: #e6c200; }
.ny-spin { display: inline-block; animation: nyspin 0.8s linear infinite; }
@keyframes nyspin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.ny-card-ok { background: #e8f8e8; border-color: #2a7; }
.ny-card-bad { background: #fdeaea; border-color: #c33; }
.ny-card-warn { background: #fff6e0; border-color: #d90; }
.ny-card-head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: space-between; }
.ny-reward { font-size: 0.9em; color: #333; }
.ny-sub { font-size: 0.88em; margin-top: 4px; margin-left: 1.2rem; color: #333; }
.ny-chain { margin-top: 12px; }
.ny-bar { font-family: monospace; font-size: 1.05em; }
.ny-pct { margin-left: 8px; }
.ny-chal { margin-top: 12px; }
.ny-muted { color: #777; }
.ny-verdict { margin-top: 14px; padding: 12px; border-radius: 8px; border: 2px solid #888; }
.ny-verdict-pending { background: #f0f0f0; }
.ny-verdict-convict { background: #e6f0ff; border-color: #246; }
.ny-verdict-acquit { background: #e8fff4; border-color: #262; }
.ny-verdict-partial { background: #f5f0ff; border-color: #626; }
.ny-vbig { font-size: 1.12em; font-weight: bold; margin-bottom: 6px; }
</style>
"""


def _render_case_panel(obs_dict: dict[str, Any] | None) -> str:
    if not obs_dict or "case_file" not in obs_dict:
        return "<div class='ny-case'><p class='ny-muted'>Click <b>New Case</b> to load a FIR.</p></div>"
    cf = obs_dict["case_file"]
    cid = html.escape(str(cf.get("case_id", "")))
    tid = html.escape(str(cf.get("template_id", "")))
    ipc = html.escape(", ".join(str(x) for x in cf.get("applicable_ipc_sections", [])))
    fir = html.escape(str(cf.get("fir", ""))[:2500])

    ev_lines = []
    for e in cf.get("evidence_items", []):
        mark = "✅" if e.get("is_present") else "❌"
        desc = html.escape(str(e.get("description", e.get("id", ""))))
        ev_lines.append(f"<li>{mark} {desc}</li>")

    wit_lines = []
    for w in cf.get("witness_statements", []):
        rel = float(w.get("reliability", 0.0))
        wid = html.escape(str(w.get("id", "")))
        c = html.escape(str(w.get("content", ""))[:100])
        wit_lines.append(f"<li>👤 <code>{wid}</code> {c} <span class='ny-rel'>{rel:.2f}</span></li>")

    return f"""<div class='ny-case'>
<h3>Case file</h3>
<p><b>Case ID:</b> {cid}</p>
<p><b>Template:</b> {tid}</p>
<p><b>IPC sections:</b> {ipc}</p>
<details><summary>FIR</summary><pre class='ny-fir'>{fir}</pre></details>
<h4>Evidence</h4>
<ul class='ny-list'>{"".join(ev_lines) or "<li class='ny-muted'>(none)</li>"}</ul>
<h4>Witnesses</h4>
<ul class='ny-list'>{"".join(wit_lines) or "<li class='ny-muted'>(none)</li>"}</ul>
</div>"""


def _step_card_html(
    label: str,
    cf: dict[str, Any],
    card: dict[str, Any] | None,
    running: bool,
) -> str:
    if running:
        return f"""<div class='ny-card ny-card-running'><span class='ny-spin'>⏳</span>
<b>{html.escape(label)}</b></div>"""
    if not card:
        return f"""<div class='ny-card ny-card-pending'><span>⬜</span> {html.escape(label)}</div>"""

    act = card.get("action") or {}
    sr = card.get("step_result") or {}
    valid = bool(sr.get("is_valid"))
    reward = float(sr.get("reward", 0.0))
    rw = f"{reward:+.1f}" if reward != 0.0 else "0.0"

    stype = str(act.get("step_type", ""))
    if valid and reward == 0.0 and stype == "counter_argument":
        icon, cls = "⚠️", "ny-card-warn"
    elif valid:
        icon, cls = "✅", "ny-card-ok"
    else:
        icon, cls = "❌", "ny-card-bad"

    eids = list(act.get("anchored_evidence_ids") or [])
    wids = list(act.get("anchored_witness_ids") or [])
    ipcs = list(act.get("cited_ipc_sections") or [])

    ev_disp = []
    for eid in eids:
        desc, et = _lookup_evidence(cf, str(eid))
        ev_disp.append(f"{html.escape(desc)} ({html.escape(et)})")
    ev_str = html.escape(", ".join(ev_disp) if ev_disp else "—")

    w_disp = [_lookup_witness_line(cf, str(w)) for w in wids]
    w_str = html.escape(", ".join(w_disp) if w_disp else "—")
    ipc_str = html.escape(", ".join(ipcs) if ipcs else "—")

    note_block = ""
    fail = sr.get("failure_reason")
    ch = sr.get("prosecution_challenge")
    if stype == "counter_argument" and reward == 0.0:
        note = "Did not address prosecution challenge" if valid else (str(fail) if fail else "Invalid step")
        note_block += f"<div class='ny-sub'><b>Note:</b> {html.escape(note)}</div>"
    elif fail:
        note_block += f"<div class='ny-sub'><b>Reason:</b> {html.escape(str(fail))}</div>"
    if ch:
        note_block += f"<div class='ny-sub'><b>Challenge:</b> {html.escape(str(ch))}</div>"

    return f"""<div class='ny-card {cls}'><div class='ny-card-head'><span>{icon}</span>
<b>{html.escape(label)}</b><span class='ny-reward'>reward: {rw}</span></div>
<div class='ny-sub'><b>Evidence used:</b> {ev_str}</div>
<div class='ny-sub'><b>Witness used:</b> {w_str}</div>
<div class='ny-sub'><b>IPC cited:</b> {ipc_str}</div>
{note_block}</div>"""


def _chain_bar(pct: float) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(pct / 5.0))
    bar = "█" * filled + "░" * (20 - filled)
    return f"""<div class='ny-chain'><b>Chain score</b>
<div class='ny-bar'><span>{html.escape(bar)}</span>
<span class='ny-pct'>{pct:.1f}%</span></div></div>"""


def _challenges_html(challenges: list[str]) -> str:
    if not challenges:
        return "<div class='ny-chal'><h4>Prosecution challenges</h4><p class='ny-muted'>(none yet)</p></div>"
    items = "".join(f"<li>⚔️ {html.escape(c)}</li>" for c in challenges)
    return f"<div class='ny-chal'><h4>Prosecution challenges raised</h4><ul class='ny-list'>{items}</ul></div>"


def _verdict_html(obs_dict: dict[str, Any] | None, done: bool) -> str:
    if not obs_dict:
        return "<div class='ny-verdict ny-verdict-pending'><b>VERDICT:</b> (pending)</div>"
    steps = obs_dict.get("submitted_steps") or []
    cf = obs_dict.get("case_file") or {}
    if not done:
        return "<div class='ny-verdict ny-verdict-pending'><b>VERDICT:</b> (pending)</div>"
    if len(steps) < 6:
        return (
            "<div class='ny-verdict ny-verdict-partial'><b>Episode ended</b> before all six "
            "steps completed.</div>"
        )

    last = steps[-1]
    j = last.get("judgment")
    if not j:
        return "<div class='ny-verdict ny-verdict-pending'><b>VERDICT:</b> (pending)</div>"

    jv = str(j).lower()
    j_up = str(j).upper()
    pid = html.escape(str(cf.get("precedent_id", "")))
    chain_pct = float(obs_dict.get("chain_score", 0.0)) * 100.0
    matched, _gt = _precedent_alignment(cf, jv)
    align = "✅ Precedent matched" if matched else "⚠️ Precedent not matched"
    if jv == "acquit":
        css = "ny-verdict-acquit"
    elif jv == "partial":
        css = "ny-verdict-partial"
    else:
        css = "ny-verdict-convict"

    return f"""<div class='ny-verdict {css}'>
<div class='ny-vbig'>⚖️ VERDICT: {html.escape(j_up)}</div>
<div><b>Precedent:</b> {pid}</div>
<div><b>Final chain score:</b> {chain_pct:.1f}%</div>
<div><b>Alignment:</b> {html.escape(align)}</div>
</div>"""


def _reasoning_panel_html(
    obs_dict: dict[str, Any] | None,
    cards: list[dict[str, Any] | None],
    running_slot: int | None,
    done: bool,
) -> str:
    cf = (obs_dict or {}).get("case_file") or {}
    parts = ["<div class='ny-reason'><h3>Reasoning chain</h3>"]
    for i, lab in enumerate(_STEP_LABELS):
        c = cards[i] if i < len(cards) else None
        run = running_slot == i
        parts.append(_step_card_html(lab, cf, c, run))
    chain = 0.0
    chals: list[str] = []
    if obs_dict:
        chain = float(obs_dict.get("chain_score", 0.0)) * 100.0
        chals = list(obs_dict.get("prosecution_challenges") or [])
    parts.append(_chain_bar(chain))
    parts.append(_challenges_html(chals))
    parts.append(_verdict_html(obs_dict, done))
    parts.append("</div>")
    return "\n".join(parts)


def _empty_demo_state() -> dict[str, Any]:
    return {
        "session_id": "",
        "obs_dict": None,
        "cards": [None] * 6,
        "done": True,
        "total_reward": 0.0,
        "turns": [],
        "category": "All",
        "curriculum_level": 1,
        "explanation_text": "",
    }


def _demo_pack_outputs(
    state: dict[str, Any],
    running_slot: int | None,
) -> tuple[str, str, dict[str, Any], Any, Any, Any, Any, Any, Any]:
    obs = state.get("obs_dict")
    cards = list(state.get("cards") or [None] * 6)
    while len(cards) < 6:
        cards.append(None)
    done = bool(state.get("done", False))
    case_h = _DEMO_CSS + "<div class='ny-panel'>" + _render_case_panel(obs) + "</div>"
    reason_h = _DEMO_CSS + "<div class='ny-panel'>" + _reasoning_panel_html(obs, cards, running_slot, done) + "</div>"
    busy = running_slot is not None
    api_note = f"API: `{_API_BASE}`"
    explanation_visible = False
    explanation_text = str(state.get("explanation_text", "") or "")
    explanation_summary = ""
    if obs:
        steps = list(obs.get("submitted_steps") or [])
        if done and len(steps) >= 6:
            explanation_visible = True
            cf = obs.get("case_file") or {}
            last_step = steps[-1] or {}
            last_j = str(last_step.get("judgment") or "").upper()
            chain_pct = float(obs.get("chain_score", 0.0)) * 100.0
            matched, _gt = _precedent_alignment(cf, str(last_step.get("judgment") or "").lower() or None)
            ptxt = "✅ Matched" if matched else "⚠️ Not matched"
            bar = _confidence_bar(chain_pct, blocks=12)
            explanation_summary = (
                f"Confidence: {bar}  {chain_pct:.0f}%   |   Precedent: {ptxt}   |   Judgment: {last_j or 'N/A'}"
            )
    return (
        case_h,
        reason_h,
        state,
        gr.update(interactive=not busy),
        gr.update(interactive=not busy),
        gr.update(value=api_note),
        gr.update(visible=explanation_visible),
        gr.update(value=explanation_text),
        gr.update(value=explanation_summary),
    )


def demo_new_case(category: str, curriculum_level: int, state: dict[str, Any]) -> tuple:
    sid = str(uuid.uuid4())
    client = _api_client()
    obs_dict: dict[str, Any] | None = None
    err = ""
    try:
        for _attempt in range(10):
            raw = client.reset(session_id=sid, curriculum_level=int(curriculum_level))
            tid = str(raw.get("case_file", {}).get("template_id", ""))
            if _template_matches_category(tid, category):
                obs_dict = raw
                break
        if obs_dict is None:
            err = f"No case matched category {category!r} after 10 resets."
    except Exception as e:
        err = f"API error (is the server running?): {e}"

    if err:
        st = _empty_demo_state()
        msg = _DEMO_CSS + f"<div class='ny-panel'><p style='color:#a33'>{html.escape(err)}</p></div>"
        return (
            msg,
            msg,
            st,
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(value=f"`{_API_BASE}` — error"),
            gr.update(visible=False),
            gr.update(value=""),
            gr.update(value=""),
        )

    st = {
        "session_id": sid,
        "obs_dict": obs_dict,
        "cards": [None] * 6,
        "done": False,
        "total_reward": 0.0,
        "turns": [],
        "category": category,
        "curriculum_level": int(curriculum_level),
        "case_file_serial": obs_dict.get("case_file"),
        "explanation_text": "",
    }
    return _demo_pack_outputs(st, None)


def _demo_apply_step_response(
    state: dict[str, Any],
    slot: int,
    action: Action,
    sr_dict: dict[str, Any],
    obs_dict: dict[str, Any],
    done: bool,
) -> None:
    state["obs_dict"] = obs_dict
    state["done"] = bool(done)
    state["total_reward"] = float(state.get("total_reward", 0.0)) + float(sr_dict.get("reward", 0.0))
    cards: list[Any] = list(state.get("cards") or [None] * 6)
    while len(cards) < 6:
        cards.append(None)
    cards[slot] = {"action": _serialize(action), "step_result": sr_dict}
    state["cards"] = cards
    state.setdefault("turns", []).append({"slot": slot, "action": _serialize(action), "step_result": sr_dict})
    if done:
        steps = obs_dict.get("submitted_steps") or []
        _save_session_log(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "mode": "courtroom_api",
                "curriculum_level": state.get("curriculum_level", 1),
                "case_file": state.get("case_file_serial"),
                "turns": state["turns"],
                "verdict": {
                    "total_reward": state["total_reward"],
                    "valid_steps": len(steps),
                    "chain_score": obs_dict.get("chain_score"),
                },
                "human_won": False,
            }
        )

    # Show loading text immediately if we just completed step 6.
    steps_now = obs_dict.get("submitted_steps") or []
    if bool(done) and len(steps_now) >= 6:
        last = steps_now[-1] or {}
        if last.get("judgment"):
            state["explanation_text"] = "⏳ Generating legal reasoning explanation..."


def demo_run_agent(state: dict[str, Any]):
    if not state or not state.get("session_id"):
        yield _demo_pack_outputs(_empty_demo_state(), None)
        return

    max_actions = 24
    for attempt in range(max_actions):
        if state.get("done") or not state.get("obs_dict"):
            break
        obs = _dict_to_observation(state["obs_dict"])
        if len(obs.submitted_steps) >= 6:
            state["done"] = True
            break
        if attempt > 0:
            time.sleep(_STEP_DELAY_SEC)

        slot = len(obs.submitted_steps)
        yield _demo_pack_outputs(state, slot)

        client = _api_client()
        try:
            with torch.no_grad():
                action = _get_agent().select_action(obs)
            resp = client.step(session_id=state["session_id"], action=action)
        except Exception:
            state["done"] = True
            yield _demo_pack_outputs(state, None)
            return

        new_obs = resp["observation"]
        sr = resp["step_result"]
        done = bool(resp.get("done", False))
        _demo_apply_step_response(state, slot, action, sr, new_obs, done)
        yield _demo_pack_outputs(state, None)
        if done:
            steps_now = list((state.get("obs_dict") or {}).get("submitted_steps") or [])
            if len(steps_now) >= 6:
                obs_now = state.get("obs_dict") or {}
                cf_now = obs_now.get("case_file") or {}
                verdict = str((steps_now[-1] or {}).get("judgment") or "")
                if verdict:
                    state["explanation_text"] = generate_explanation(
                        case_file=cf_now,
                        submitted_steps=steps_now,
                        prosecution_challenges=list(obs_now.get("prosecution_challenges") or []),
                        chain_score=float(obs_now.get("chain_score", 0.0)),
                        verdict=verdict,
                    )
                    yield _demo_pack_outputs(state, None)
            break

    yield _demo_pack_outputs(state, None)


def demo_next_step(state: dict[str, Any]):
    if not state or not state.get("session_id") or state.get("done"):
        yield _demo_pack_outputs(state or _empty_demo_state(), None)
        return
    obs_dict = state.get("obs_dict")
    if not obs_dict:
        yield _demo_pack_outputs(state, None)
        return
    obs = _dict_to_observation(obs_dict)
    if len(obs.submitted_steps) >= 6:
        state["done"] = True
        yield _demo_pack_outputs(state, None)
        return

    slot = len(obs.submitted_steps)
    yield _demo_pack_outputs(state, slot)

    client = _api_client()
    try:
        with torch.no_grad():
            action = _get_agent().select_action(obs)
        resp = client.step(session_id=state["session_id"], action=action)
    except Exception:
        state["done"] = True
        yield _demo_pack_outputs(state, None)
        return

    new_obs = resp["observation"]
    sr = resp["step_result"]
    done = bool(resp.get("done", False))
    _demo_apply_step_response(state, slot, action, sr, new_obs, done)
    yield _demo_pack_outputs(state, None)
    if done:
        steps_now = list((state.get("obs_dict") or {}).get("submitted_steps") or [])
        if len(steps_now) >= 6:
            obs_now = state.get("obs_dict") or {}
            cf_now = obs_now.get("case_file") or {}
            verdict = str((steps_now[-1] or {}).get("judgment") or "")
            if verdict:
                state["explanation_text"] = generate_explanation(
                    case_file=cf_now,
                    submitted_steps=steps_now,
                    prosecution_challenges=list(obs_now.get("prosecution_challenges") or []),
                    chain_score=float(obs_now.get("chain_score", 0.0)),
                    verdict=verdict,
                )
                yield _demo_pack_outputs(state, None)


def refresh_sessions() -> list[list[str]]:
    """Reload session log dataframe."""
    return _load_session_logs()


# ── Gradio interface ────────────────────────────────────────────────────────


def build_app() -> gr.Blocks:
    """Construct the full Gradio interface."""
    _get_agent()
    if _MODEL_LOAD_STATUS == "stub":
        demo_banner = "⚠️ **Demo mode** — no trained model loaded. Agent uses heuristic actions."
    elif _MODEL_LOAD_STATUS in ("partial", "partial_compat"):
        demo_banner = "⚠️ **Checkpoint loaded (partial)** — running trained weights with 1+ mismatched tensor(s) skipped."
    else:
        demo_banner = ""

    with gr.Blocks(title="NyayaRL ⚖️") as app:
        gr.Markdown("# NyayaRL — Legal RL agent (courtroom view)")
        if demo_banner:
            gr.Markdown(demo_banner)
        gr.Markdown(
            "Case state is on the FastAPI server (`POST /reset`, `POST /step`). "
            "Start the API with `uvicorn server.app:app --port 8000` or set `NYAYARL_API_BASE`."
        )

        demo_state = gr.State(_empty_demo_state())

        with gr.Row():
            with gr.Column(scale=1):
                case_html = gr.HTML(value=_DEMO_CSS + "<div class='ny-panel'></div>")
            with gr.Column(scale=1):
                reasoning_html = gr.HTML(value=_DEMO_CSS + "<div class='ny-panel'></div>")

        with gr.Row(visible=False) as explanation_row:
            with gr.Column():
                gr.Markdown("### 🧠 Why did the agent decide this?")
                explanation_box = gr.Textbox(
                    label="",
                    lines=5,
                    interactive=False,
                    elem_id="explanation_panel",
                )
                explanation_summary = gr.Markdown(value="")

        api_status = gr.Markdown(value=f"API: `{_API_BASE}`")

        with gr.Row():
            new_case_btn = gr.Button("New Case", variant="primary")
            run_agent_btn = gr.Button("▶ Run Agent", variant="secondary")
            next_step_btn = gr.Button("Next step (manual)", variant="secondary")
            category_dd = gr.Dropdown(
                choices=["All", "Homicide", "Robbery", "Assault", "Fraud", "Domestic"],
                value="All",
                label="Category",
                scale=1,
            )
            level_dd = gr.Dropdown(choices=[1, 2, 3, 4], value=1, label="Curriculum level", scale=1)

        with gr.Tab("Session log"):
            gr.Markdown(
                "> Session logs are stored locally and **do not persist** across Space restarts."
            )
            refresh_btn = gr.Button("Refresh")
            session_table = gr.Dataframe(
                headers=["Timestamp", "Mode", "Level", "Human Won", "Reward", "Valid Steps"],
                value=_load_session_logs(),
                interactive=False,
            )
            refresh_btn.click(fn=refresh_sessions, inputs=[], outputs=[session_table])

        outs = [
            case_html,
            reasoning_html,
            demo_state,
            run_agent_btn,
            next_step_btn,
            api_status,
            explanation_row,
            explanation_box,
            explanation_summary,
        ]

        new_case_btn.click(
            fn=demo_new_case,
            inputs=[category_dd, level_dd, demo_state],
            outputs=outs,
        )

        run_agent_btn.click(fn=demo_run_agent, inputs=[demo_state], outputs=outs)
        next_step_btn.click(fn=demo_next_step, inputs=[demo_state], outputs=outs)

    return app


# ── Launch ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None, help="Deterministic mode seed")
    args = p.parse_args()

    if args.seed is not None:
        _set_seed(args.seed)
        os.environ["NYAYARL_SEED"] = str(args.seed)
        _SEED = int(args.seed)

    app = build_app()

    def _is_port_free(port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", int(port)))
            return True
        except OSError:
            return False

    # If the user explicitly set GRADIO_SERVER_PORT, honor it strictly.
    # Otherwise, try a small range so re-running doesn't crash on an occupied port.
    env_port = os.getenv("GRADIO_SERVER_PORT")
    if env_port is not None and str(env_port).strip():
        port = int(env_port)
        app.launch(server_name="127.0.0.1", server_port=port, share=False)
    else:
        base = 7860
        chosen = None
        for p0 in range(base, base + 50):
            if _is_port_free(p0):
                chosen = p0
                break
        if chosen is None:
            # Last resort: let Gradio raise a helpful error.
            chosen = base
        app.launch(server_name="127.0.0.1", server_port=int(chosen), share=False)
