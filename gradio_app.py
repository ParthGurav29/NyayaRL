
"""
gradio_app.py — Gradio adapter for NyayaRL.

Wraps the existing environment, validator, and agent in a browser-based
interface for Hugging Face Spaces. Contains zero game logic — every
meaningful operation delegates to the existing classes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import gradio as gr
import torch
import torch.nn as nn

from nyayarl.checkpointing import load_defence_agent_checkpoint
from nyayarl.environment import NyayaRLEnvironment
from nyayarl.agents import DefenceAgent as TrainedDefenceAgent
from human_mode.session_logic import build_environment
from nyayarl.models import (
    Action,
    CaseFile,
    EvidenceItem,
    EvidenceType,
    JudgmentLabel,
    Observation,
    StepResult,
    StepType,
    Verdict,
    WitnessStatement,
)


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
    # Optional NumPy support (only if installed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    # CuDNN determinism (no-op on Mac/MPS)
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

# This automatically finds the folder where gradio_app.py lives
BASE_DIR = Path(__file__).resolve().parent

_SESSIONS_DIR = Path(
    os.getenv("NYAYARL_SESSIONS_DIR", str(BASE_DIR / "human_mode" / "sessions"))
)
_CHECKPOINT_DIR = Path(
    os.getenv("NYAYARL_CHECKPOINT_DIR", str(BASE_DIR / "checkpoints"))
)
_SEED_ENV = os.getenv("NYAYARL_SEED")
_SEED: int | None = int(_SEED_ENV) if _SEED_ENV is not None and str(_SEED_ENV).strip() else None

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


# Environment wiring uses Track2 components via `human_mode.session_logic.build_environment`.


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
        elif step_type == StepType.MENS_REA:
            wids = [w.id for w in cf.witness_statements if w.reliability > 0.5]
            return Action(step_type=step_type, anchored_witness_ids=wids[:1] or ["W1"])
        elif step_type == StepType.LINKAGE:
            eids = [e.id for e in cf.evidence_items if e.is_present and e.type in (EvidenceType.DOCUMENTARY, EvidenceType.FORENSIC)]
            return Action(step_type=step_type, anchored_evidence_ids=eids[:1] or ["E3"], cited_ipc_sections=cf.applicable_ipc_sections[:1])
        elif step_type == StepType.COUNTER_ARGUMENT:
            wids = [w.id for w in cf.witness_statements if w.reliability > 0.5]
            return Action(step_type=step_type, anchored_witness_ids=wids[:1] or ["W1"])
        elif step_type == StepType.IPC_APPLICATION:
            return Action(step_type=step_type, cited_ipc_sections=cf.applicable_ipc_sections[:1])
        else:
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
    global _DEMO_MODE
    require_ckpt = os.getenv("NYAYARL_REQUIRE_CHECKPOINT") == "1"
    allow_legacy = os.getenv("NYAYARL_ALLOW_LEGACY_CHECKPOINT") == "1"
    agent: nn.Module = TrainedDefenceAgent(rng_seed=_SEED)

    if _CHECKPOINT_DIR.exists():
        # Prefer an explicit "latest.pt" if present (supports symlinks).
        # Otherwise fall back to the newest step_*.pt lexicographically.
        latest = _CHECKPOINT_DIR / "latest.pt"
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
                # realpath resolves latest.pt symlink to the actual step_*.pt file.
                print(
                    f"✓ Loaded checkpoint: {os.path.realpath(str(ckpt_path))} "
                    f"(selected={ckpt_path} dir={_CHECKPOINT_DIR})"
                )
                _DEMO_MODE = bool(res.was_partial_load)
                agent.eval()
                return agent
            except Exception as e:
                print(f"⚠ Failed to load checkpoint from {ckpt_path}: {e}")
                if require_ckpt:
                    raise RuntimeError(
                        f"CRITICAL failure: Checkpoint load failed from {ckpt_path} "
                        f"(dir={_CHECKPOINT_DIR}). Deployment aborted."
                    ) from e
                # If we failed to load a trained checkpoint, we must clearly enter demo mode.
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
    print("⚠ No trained checkpoint — running in demo mode")
    agent = StubDefenceAgent()
    agent.eval()
    return agent


_agent: nn.Module | None = None


def _get_agent() -> nn.Module:
    global _agent
    if _agent is None:
        _agent = _load_agent()
    return _agent


# ── Case file formatting ───────────────────────────────────────────────────


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
                str(d.get('verdict', {}).get('valid_steps', 0)),
            ])
        except Exception:
            continue
    return rows


# ── Gradio callbacks ───────────────────────────────────────────────────────


def start_case(mode: str, level: int, state: dict) -> tuple:
    """Reset environment and display the case file."""
    env = build_environment()

    observation = env.reset(level)
    cf = observation.case_file

    # Build checkbox choices
    evidence_choices = [f"[{e.id}] {e.type.value.upper()} — {e.description} ({'✓' if e.is_present else '✗'})" for e in cf.evidence_items]
    witness_choices = [f"[{w.id}] Rel:{w.reliability:.1f} — {w.content}{' ⚠' if w.is_contradicting else ''}" for w in cf.witness_statements]
    ipc_choices = cf.applicable_ipc_sections

    # Store state
    state = {
        "env": env,
        "observation": observation,
        "mode": mode.lower(),
        "level": level,
        "step_num": 0,
        "total_reward": 0.0,
        "done": False,
        "turns": [],
        "case_file_serial": _serialize(cf),
    }

    step_idx = len(observation.submitted_steps)
    current_step = _STEP_LABELS[step_idx] if step_idx < 6 else "Complete"
    is_step_6 = step_idx == 5

    return (
        _format_case_file(cf),        # case_display
        current_step,                   # step_label
        gr.update(choices=evidence_choices, value=[]),   # evidence_checklist
        gr.update(choices=witness_choices, value=[]),     # witness_checklist
        gr.update(choices=ipc_choices, value=[]),         # ipc_checklist
        gr.update(visible=is_step_6),                    # judgment_radio visibility
        "",                             # step_result_display
        0.0,                            # chain_score
        "",                             # verdict_display
        gr.update(visible=True),        # submit button visible
        state,                          # gr.State
    )


def submit_step(
    evidence_sel: list[str],
    witness_sel: list[str],
    ipc_sel: list[str],
    judgment_sel: str | None,
    state: dict,
) -> tuple:
    """Process one step submission."""
    if not state or state.get("done", True):
        return (
            "Episode is done. Click 'Start Case' for a new episode.",
            "", 0.0, "", gr.update(), state
        )

    env: NyayaRLEnvironment = state["env"]
    obs: Observation = state["observation"]
    mode = state["mode"]
    cf = obs.case_file
    step_idx = len(obs.submitted_steps)

    if step_idx >= 6:
        state["done"] = True
        return ("All 6 steps already submitted.", "", 0.0, "", gr.update(), state)

    step_type = _STEP_SEQUENCE[step_idx]
    state["step_num"] += 1

    if mode == "defence":
        # Parse human selections back to IDs
        evidence_ids = []
        for sel in evidence_sel:
            eid = sel.split("]")[0].replace("[", "").strip()
            evidence_ids.append(eid)

        witness_ids = []
        for sel in witness_sel:
            wid = sel.split("]")[0].replace("[", "").strip()
            witness_ids.append(wid)

        judgment = None
        if step_type == StepType.PRECEDENT_CITATION:
            if judgment_sel == "ACQUIT":
                judgment = JudgmentLabel.ACQUIT
            elif judgment_sel == "CONVICT":
                judgment = JudgmentLabel.CONVICT
            elif judgment_sel == "PARTIAL":
                judgment = JudgmentLabel.PARTIAL
            # Add precedent anchor
            evidence_ids.append(cf.precedent_id)

        action = Action(
            step_type=step_type,
            anchored_evidence_ids=evidence_ids,
            anchored_witness_ids=witness_ids,
            cited_ipc_sections=ipc_sel,
            judgment=judgment,
        )
    else:
        # Prosecution mode — agent builds the action
        with torch.no_grad():
            action = _get_agent().select_action(obs)

    # Execute step
    try:
        new_obs, step_result, done = env.step(action)
    except (ValueError, RuntimeError) as exc:
        return (
            f"⚠ Error: {exc}",
            _STEP_LABELS[step_idx],
            obs.chain_score,
            "",
            gr.update(),
            state,
        )

    state["observation"] = new_obs
    state["total_reward"] += step_result.reward
    state["done"] = done
    state["turns"].append({
        "step_type": step_type.value,
        "action": _serialize(action),
        "step_result": _serialize(step_result),
        "actor": "human" if mode == "defence" else "agent",
    })

    result_text = _format_step_result(state["step_num"], step_result, new_obs.chain_score)

    if mode == "prosecution" and step_result.is_valid:
        result_text += "\n\n📋 Agent's action:"
        result_text += f"\n  Evidence: {action.anchored_evidence_ids}"
        result_text += f"\n  Witnesses: {action.anchored_witness_ids}"
        result_text += f"\n  IPC: {action.cited_ipc_sections}"
        if action.judgment:
            result_text += f"\n  Judgment: {action.judgment.value.upper()}"

    # Verdict
    verdict_text = ""
    if done:
        verdict_text = _format_verdict(new_obs, state["total_reward"], state["step_num"])
        human_won = (mode == "defence" and len(new_obs.submitted_steps) == 6) or \
                    (mode == "prosecution" and len(new_obs.submitted_steps) < 6)
        _save_session_log({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "curriculum_level": state["level"],
            "case_file": state["case_file_serial"],
            "turns": state["turns"],
            "verdict": {
                "total_reward": state["total_reward"],
                "valid_steps": len(new_obs.submitted_steps),
                "chain_score": new_obs.chain_score,
            },
            "human_won": human_won,
        })

    # Next step info
    next_step_idx = len(new_obs.submitted_steps)
    if done or next_step_idx >= 6:
        current_step = "Complete"
        is_step_6 = False
    else:
        current_step = _STEP_LABELS[next_step_idx]
        is_step_6 = next_step_idx == 5

    return (
        result_text,                               # step_result_display
        current_step,                              # step_label
        new_obs.chain_score,                       # chain_score
        verdict_text,                              # verdict_display
        gr.update(visible=is_step_6),              # judgment_radio visibility
        state,                                     # gr.State
    )


def refresh_sessions() -> list[list[str]]:
    """Reload session log dataframe."""
    return _load_session_logs()


# ── Gradio interface ────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    """Construct the full Gradio interface."""

    # Ensure agent is initialised so the banner reflects reality.
    _get_agent()
    demo_banner = "⚠️ **Demo mode** — no trained model loaded. Agent uses heuristic actions." if _DEMO_MODE else ""

    with gr.Blocks(
        title="NyayaRL ⚖️",
    ) as app:

        gr.Markdown("# ⚖️ NyayaRL — Indian Legal Reasoning Arena")
        if demo_banner:
            gr.Markdown(demo_banner)

        state = gr.State({})

        with gr.Tab("🎮 Play a Case"):
            with gr.Row():
                mode_dropdown = gr.Dropdown(
                    choices=["Defence", "Prosecution"],
                    value="Defence",
                    label="Play as",
                    scale=1,
                )
                level_dropdown = gr.Dropdown(
                    choices=[1, 2, 3, 4],
                    value=1,
                    label="Curriculum Level",
                    scale=1,
                )
                start_btn = gr.Button("🚀 Start Case", variant="primary", scale=1)

            case_display = gr.Textbox(
                label="Case File",
                lines=15,
                interactive=False,
            )

            step_label = gr.Textbox(
                label="Current Step",
                value="Click 'Start Case' to begin",
                interactive=False,
                max_lines=1,
            )

            with gr.Row():
                evidence_check = gr.CheckboxGroup(
                    choices=[],
                    label="📦 Evidence to Anchor",
                    scale=1,
                )
                witness_check = gr.CheckboxGroup(
                    choices=[],
                    label="🧑‍⚖️ Witnesses to Anchor",
                    scale=1,
                )

            with gr.Row():
                ipc_check = gr.CheckboxGroup(
                    choices=[],
                    label="📜 IPC Sections to Cite",
                    scale=1,
                )
                judgment_radio = gr.Radio(
                    choices=["ACQUIT", "CONVICT", "PARTIAL"],
                    label="⚖️ Judgment (Step 6 only)",
                    visible=False,
                    scale=1,
                )

            submit_btn = gr.Button("✅ Submit Step", variant="primary", visible=True)

            step_result_display = gr.Textbox(
                label="Step Result",
                lines=8,
                interactive=False,
            )

            chain_score = gr.Number(label="Chain Score", value=0.0, interactive=False)

            verdict_display = gr.Textbox(
                label="Final Verdict",
                lines=10,
                interactive=False,
            )

            # ── Wire events ──────────────────────────────────────────
            start_btn.click(
                fn=start_case,
                inputs=[mode_dropdown, level_dropdown, state],
                outputs=[
                    case_display,
                    step_label,
                    evidence_check,
                    witness_check,
                    ipc_check,
                    judgment_radio,
                    step_result_display,
                    chain_score,
                    verdict_display,
                    submit_btn,
                    state,
                ],
            )

            submit_btn.click(
                fn=submit_step,
                inputs=[evidence_check, witness_check, ipc_check, judgment_radio, state],
                outputs=[
                    step_result_display,
                    step_label,
                    chain_score,
                    verdict_display,
                    judgment_radio,
                    state,
                ],
            )

        with gr.Tab("📋 Session Log"):
            gr.Markdown(
                "> ⚠️ Session logs are stored locally and **do not persist** "
                "across Space restarts."
            )
            refresh_btn = gr.Button("🔄 Refresh")
            session_table = gr.Dataframe(
                headers=["Timestamp", "Mode", "Level", "Human Won", "Reward", "Valid Steps"],
                value=_load_session_logs(),
                interactive=False,
            )
            refresh_btn.click(fn=refresh_sessions, inputs=[], outputs=[session_table])

    return app


# ── Launch ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None, help="Deterministic mode seed")
    args = p.parse_args()

    if args.seed is not None:
        _set_seed(args.seed)
        # Make the seed visible to the rest of the module (agent init uses this).
        os.environ["NYAYARL_SEED"] = str(args.seed)
        _SEED = int(args.seed)

    app = build_app()
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    app.launch(server_name="127.0.0.1", server_port=port, share=False)
