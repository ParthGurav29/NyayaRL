"""
cli_interface.py — Interactive human-vs-agent CLI session.

A human plays one side of a case, the trained defence agent plays the other.
The deterministic judge scores both at every step and issues a final verdict.
Logs every session for the imitation learning dataset.

No web UI, no frontend — pure terminal.
"""

from __future__ import annotations

import dataclasses
import json
import random
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from nyayarl.environment import NyayaRLEnvironment
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

# ── Canonical step sequence ─────────────────────────────────────────────────

_STEP_SEQUENCE = [
    StepType.ACTUS_REUS,
    StepType.MENS_REA,
    StepType.LINKAGE,
    StepType.COUNTER_ARGUMENT,
    StepType.IPC_APPLICATION,
    StepType.PRECEDENT_CITATION,
]

_STEP_LABELS = {
    StepType.ACTUS_REUS: "Step 1 — ACTUS REUS",
    StepType.MENS_REA: "Step 2 — MENS REA",
    StepType.LINKAGE: "Step 3 — LINKAGE",
    StepType.COUNTER_ARGUMENT: "Step 4 — COUNTER ARGUMENT",
    StepType.IPC_APPLICATION: "Step 5 — IPC APPLICATION",
    StepType.PRECEDENT_CITATION: "Step 6 — PRECEDENT CITATION",
}

_SESSIONS_DIR = Path("human_mode/sessions")

# ── Display helpers ─────────────────────────────────────────────────────────

_BAR = "━" * 44


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


def display_case_file(cf: CaseFile) -> None:
    """Print the case file in a readable labelled format."""
    print(f"\n{_BAR}")
    print(f"CASE FILE — {cf.case_id}")
    print(_BAR)

    print(f"\nFIR:\n{cf.fir}\n")

    print("EVIDENCE ITEMS:")
    for e in cf.evidence_items:
        status = "✓ Present" if e.is_present else "✗ Missing"
        print(f"  [{e.id}] {e.type.value.upper():12s} — {e.description}  {status}")

    print("\nWITNESS STATEMENTS:")
    for w in cf.witness_statements:
        flag = " ⚠ Contradicting" if w.is_contradicting else ""
        print(f"  [{w.id}] Reliability: {w.reliability:.1f} — {w.content}{flag}")

    print("\nAPPLICABLE IPC SECTIONS:")
    print("  " + ", ".join(cf.applicable_ipc_sections))

    print(f"\nCURRICULUM LEVEL: {cf.curriculum_level}")
    print(_BAR)


def display_step_result(step_num: int, sr: StepResult, chain_score: float) -> None:
    """Print the result of one step."""
    print(f"\n{_BAR}")
    print(f"STEP {step_num} RESULT")
    print(_BAR)
    print(f"  Valid:   {'YES' if sr.is_valid else 'NO'}")
    print(f"  Reward:  {sr.reward:+.2f}")
    if sr.failure_reason:
        print(f"  Reason:  {sr.failure_reason}")

    print(f"\n  PROSECUTION CHALLENGE:")
    if sr.prosecution_challenge:
        print(f"  {sr.prosecution_challenge}")
    else:
        print("  None raised")

    print(f"\n  CHAIN SCORE: {chain_score:.2f}")
    print(_BAR)


def display_verdict(obs: Observation, total_reward: float, steps_taken: int) -> None:
    """Print the final episode verdict."""
    valid_steps = len(obs.submitted_steps)
    challenges = len(obs.prosecution_challenges)
    pwr = (challenges / steps_taken * 100) if steps_taken > 0 else 0.0

    # Determine outcome
    if valid_steps == 6:
        outcome = "WIN — All 6 steps validated, argument chain complete!"
    elif obs.chain_score >= 0.5:
        outcome = "DRAW — Partial chain completed."
    else:
        outcome = "LOSS — Argument chain failed."

    judgment_text = "N/A"
    if obs.submitted_steps:
        last_j = obs.submitted_steps[-1].judgment
        if last_j is not None:
            judgment_text = last_j.value.upper()

    print(f"\n{_BAR}")
    print("FINAL VERDICT")
    print(_BAR)
    print(f"  Judgment:          {judgment_text}")
    print(f"  Total reward:      {total_reward:.2f}")
    print(f"  Steps taken:       {steps_taken}")
    print(f"  Valid steps:       {valid_steps}")
    print(f"  Prosecution wins:  {pwr:.0f}%")
    print()
    print(f"  {outcome}")
    print(_BAR)


# ── Input helpers (menu-based only, never crashes) ──────────────────────────


def _ask_int(prompt: str, valid_range: range) -> int:
    """Prompt until the user enters a valid integer in range."""
    while True:
        raw = input(prompt).strip()
        try:
            val = int(raw)
            if val in valid_range:
                return val
        except ValueError:
            pass
        print("  Invalid selection, try again.")


def _ask_multi_int(prompt: str, valid_range: range) -> list[int]:
    """Prompt for comma-separated integers; all must be in range, at least 1."""
    while True:
        raw = input(prompt).strip()
        try:
            selections = [int(x.strip()) for x in raw.split(",")]
            if all(s in valid_range for s in selections) and len(selections) >= 1:
                return selections
        except ValueError:
            pass
        print("  Invalid selection, try again.")


# ── Stub agents ─────────────────────────────────────────────────────────────


class StubCaseGenerator:
    """Generates a structurally valid case file."""

    def generate(self, curriculum_level: int) -> CaseFile:
        return CaseFile(
            case_id=f"CLI_{curriculum_level:03d}_{random.randint(100, 999)}",
            fir="[Stub FIR] The accused was apprehended at the scene with physical evidence linking them to the alleged offence under the Indian Penal Code.",
            accused_count=1,
            evidence_items=[
                EvidenceItem(id="E1", description="Bloodstained knife recovered from scene", type=EvidenceType.PHYSICAL, is_present=True),
                EvidenceItem(id="E2", description="DNA analysis report matching accused", type=EvidenceType.FORENSIC, is_present=True),
                EvidenceItem(id="E3", description="CCTV footage from adjacent premises", type=EvidenceType.DOCUMENTARY, is_present=True),
                EvidenceItem(id="E4", description="Mobile phone call records", type=EvidenceType.TESTIMONIAL, is_present=False),
            ],
            witness_statements=[
                WitnessStatement(id="W1", content="Saw accused fleeing the scene at approx. 11:30 PM", reliability=0.85, is_contradicting=False),
                WitnessStatement(id="W2", content="Heard loud argument from the victim's house", reliability=0.70, is_contradicting=False),
                WitnessStatement(id="W3", content="Claims accused was elsewhere (alibi)", reliability=0.40, is_contradicting=True),
            ],
            applicable_ipc_sections=["302", "307", "34"],
            curriculum_level=curriculum_level,
            precedent_id="ILDC_2023_001",
        )


class StubJudge:
    """Returns a placeholder verdict."""

    def evaluate(self, case_file: CaseFile, submitted_steps: list[Action]) -> Verdict:
        judgment = JudgmentLabel.PARTIAL
        if submitted_steps and submitted_steps[-1].judgment is not None:
            judgment = submitted_steps[-1].judgment
        return Verdict(
            judgment=judgment,
            matched_precedent_id=case_file.precedent_id,
            precedent_matched=True,
            total_reward=0.0,
            prosecution_win_rate=0.2,
        )


class StubDefenceAgent(nn.Module):
    """Deterministic stub agent for CLI sessions."""

    def __init__(self) -> None:
        super().__init__()
        self._param = nn.Parameter(torch.tensor(0.0))

    def select_action(self, observation: Observation) -> Action:
        step_idx = min(len(observation.submitted_steps), 5)
        step_type = _STEP_SEQUENCE[step_idx]
        cf = observation.case_file

        if step_type == StepType.ACTUS_REUS:
            # Pick first present PHYSICAL/FORENSIC evidence
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

        else:  # PRECEDENT_CITATION
            return Action(step_type=step_type, judgment=JudgmentLabel.CONVICT, anchored_evidence_ids=[cf.precedent_id])

    def get_log_prob(self, observation: Observation, action: Action) -> torch.Tensor:
        return torch.tensor(-1.0)

    def get_entropy(self, observation: Observation) -> torch.Tensor:
        return torch.tensor(1.0)


# ── Action construction from human menu input ──────────────────────────────


def _human_build_action(step_type: StepType, case_file: CaseFile) -> Action:
    """Guide the human through numbered menus to construct an Action."""
    evidence_ids: list[str] = []
    witness_ids: list[str] = []
    ipc_sections: list[str] = []
    judgment: JudgmentLabel | None = None

    if step_type == StepType.ACTUS_REUS:
        # Select evidence (PHYSICAL / FORENSIC preferred)
        print("\n  Select evidence to anchor (enter numbers separated by comma):")
        for i, e in enumerate(case_file.evidence_items, 1):
            status = "✓" if e.is_present else "✗"
            print(f"    {i}. [{e.id}] {e.type.value.upper()} — {e.description}  {status}")
        choices = _ask_multi_int("  > ", range(1, len(case_file.evidence_items) + 1))
        evidence_ids = [case_file.evidence_items[c - 1].id for c in choices]

    elif step_type == StepType.MENS_REA:
        # Select witnesses
        print("\n  Select witnesses to anchor (enter numbers separated by comma):")
        for i, w in enumerate(case_file.witness_statements, 1):
            flag = " ⚠ Contradicting" if w.is_contradicting else ""
            print(f"    {i}. [{w.id}] Reliability: {w.reliability:.1f} — {w.content}{flag}")
        choices = _ask_multi_int("  > ", range(1, len(case_file.witness_statements) + 1))
        witness_ids = [case_file.witness_statements[c - 1].id for c in choices]

    elif step_type == StepType.LINKAGE:
        # Select evidence (DOCUMENTARY / FORENSIC preferred)
        print("\n  Select evidence for chain-of-custody (enter numbers separated by comma):")
        for i, e in enumerate(case_file.evidence_items, 1):
            status = "✓" if e.is_present else "✗"
            print(f"    {i}. [{e.id}] {e.type.value.upper()} — {e.description}  {status}")
        choices = _ask_multi_int("  > ", range(1, len(case_file.evidence_items) + 1))
        evidence_ids = [case_file.evidence_items[c - 1].id for c in choices]

        # Also select IPC sections
        print("\n  Select IPC sections to cite (enter numbers separated by comma):")
        for i, s in enumerate(case_file.applicable_ipc_sections, 1):
            print(f"    {i}. {s}")
        choices = _ask_multi_int("  > ", range(1, len(case_file.applicable_ipc_sections) + 1))
        ipc_sections = [case_file.applicable_ipc_sections[c - 1] for c in choices]

    elif step_type == StepType.COUNTER_ARGUMENT:
        # Select evidence and/or witnesses
        print("\n  Select evidence to anchor (enter numbers, or 0 to skip):")
        for i, e in enumerate(case_file.evidence_items, 1):
            print(f"    {i}. [{e.id}] {e.type.value.upper()} — {e.description}")
        raw = input("  > ").strip()
        if raw and raw != "0":
            try:
                for x in raw.split(","):
                    idx = int(x.strip())
                    if 1 <= idx <= len(case_file.evidence_items):
                        evidence_ids.append(case_file.evidence_items[idx - 1].id)
            except ValueError:
                pass

        print("\n  Select witnesses to anchor (enter numbers, or 0 to skip):")
        for i, w in enumerate(case_file.witness_statements, 1):
            print(f"    {i}. [{w.id}] Reliability: {w.reliability:.1f} — {w.content}")
        raw = input("  > ").strip()
        if raw and raw != "0":
            try:
                for x in raw.split(","):
                    idx = int(x.strip())
                    if 1 <= idx <= len(case_file.witness_statements):
                        witness_ids.append(case_file.witness_statements[idx - 1].id)
            except ValueError:
                pass

        # Must have at least one
        if not evidence_ids and not witness_ids:
            print("  ⚠ Must anchor at least one evidence or witness. Defaulting to first witness.")
            witness_ids = [case_file.witness_statements[0].id]

    elif step_type == StepType.IPC_APPLICATION:
        print("\n  Select IPC sections to cite (enter numbers separated by comma):")
        for i, s in enumerate(case_file.applicable_ipc_sections, 1):
            print(f"    {i}. {s}")
        choices = _ask_multi_int("  > ", range(1, len(case_file.applicable_ipc_sections) + 1))
        ipc_sections = [case_file.applicable_ipc_sections[c - 1] for c in choices]

    elif step_type == StepType.PRECEDENT_CITATION:
        # Judgment selection
        print("\n  Select judgment:")
        print("    1. ACQUIT")
        print("    2. CONVICT")
        print("    3. PARTIAL")
        j_choice = _ask_int("  > ", range(1, 4))
        judgment = [JudgmentLabel.ACQUIT, JudgmentLabel.CONVICT, JudgmentLabel.PARTIAL][j_choice - 1]

        # Precedent anchor
        evidence_ids = [case_file.precedent_id]
        print(f"\n  Anchoring precedent: {case_file.precedent_id}")

    return Action(
        step_type=step_type,
        cited_ipc_sections=ipc_sections,
        anchored_evidence_ids=evidence_ids,
        anchored_witness_ids=witness_ids,
        judgment=judgment,
    )


# ── Session logging ─────────────────────────────────────────────────────────


def _save_session(
    mode: str,
    curriculum_level: int,
    case_file: CaseFile,
    turns: list[dict],
    observation: Observation,
    total_reward: float,
    human_won: bool,
) -> Path:
    """Write the full session atomically to a JSON file."""
    _SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_id = f"session_{ts}"
    path = _SESSIONS_DIR / f"{session_id}.json"

    data = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "curriculum_level": curriculum_level,
        "case_file": _serialize(case_file),
        "turns": turns,
        "verdict": {
            "total_reward": total_reward,
            "valid_steps": len(observation.submitted_steps),
            "chain_score": observation.chain_score,
            "is_done": observation.is_done,
        },
        "human_won": human_won,
    }

    # Atomic write: write to temp then rename
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    tmp_path.rename(path)

    return path


# ── Main session loop ───────────────────────────────────────────────────────


def run_session() -> None:
    """Run one interactive CLI session."""

    # ── Mode selection ───────────────────────────────────────────────────
    print(f"\n{_BAR}")
    print("  NyayaRL — Interactive Legal Reasoning")
    print(_BAR)
    print("\n  Play as:")
    print("    1. Defence (you build the argument chain)")
    print("    2. Prosecution (agent builds, you challenge)")
    mode_choice = _ask_int("  > ", range(1, 3))
    mode = "defence" if mode_choice == 1 else "prosecution"

    # ── Curriculum level ─────────────────────────────────────────────────
    print("\n  Select curriculum level:")
    print("    1. Level 1 (Easiest)")
    print("    2. Level 2")
    print("    3. Level 3")
    print("    4. Level 4 (Hardest)")
    print("    5. Random")
    level_choice = _ask_int("  > ", range(1, 6))
    curriculum_level = level_choice if level_choice <= 4 else random.randint(1, 4)

    # ── Setup ────────────────────────────────────────────────────────────
    env = NyayaRLEnvironment(
        case_generator=StubCaseGenerator(),
        judge=StubJudge(),
    )
    agent = StubDefenceAgent()
    agent.eval()

    observation = env.reset(curriculum_level)
    case_file = observation.case_file

    display_case_file(case_file)

    turns: list[dict] = []
    total_reward = 0.0
    step_num = 0
    done = False

    # ── Turn loop ────────────────────────────────────────────────────────
    while not done:
        step_idx = len(observation.submitted_steps)
        if step_idx >= 6:
            break
        step_type = _STEP_SEQUENCE[step_idx]
        step_num += 1

        print(f"\n{'─' * 44}")
        print(f"  {_STEP_LABELS[step_type]}")
        print(f"{'─' * 44}")

        if mode == "defence":
            # ── Human as defence ─────────────────────────────────────
            t_start = time.time()
            action = _human_build_action(step_type, case_file)
            t_end = time.time()
            time_taken = t_end - t_start

            try:
                observation, step_result, done = env.step(action)
            except (ValueError, RuntimeError) as exc:
                print(f"\n  ⚠ Environment error: {exc}")
                print("  Skipping this step.")
                turns.append({
                    "step_type": step_type.value,
                    "action": _serialize(action),
                    "step_result": {"is_valid": False, "reward": 0.0, "failure_reason": str(exc)},
                    "time_taken_seconds": time_taken,
                    "actor": "human",
                })
                continue

            total_reward += step_result.reward
            display_step_result(step_num, step_result, observation.chain_score)

            # Show agent prosecution response
            if step_result.prosecution_challenge:
                print(f"\n  🏛  Prosecution agent responds:")
                print(f"     \"{step_result.prosecution_challenge}\"")

            turns.append({
                "step_type": step_type.value,
                "action": _serialize(action),
                "step_result": _serialize(step_result),
                "time_taken_seconds": round(time_taken, 2),
                "actor": "human",
            })

        else:
            # ── Agent as defence, human as prosecution ───────────────
            print("\n  🤖 Agent is constructing argument...")
            t_start = time.time()

            with torch.no_grad():
                action = agent.select_action(observation)

            try:
                observation, step_result, done = env.step(action)
            except (ValueError, RuntimeError) as exc:
                print(f"\n  ⚠ Agent triggered error: {exc}")
                t_end = time.time()
                turns.append({
                    "step_type": step_type.value,
                    "action": _serialize(action),
                    "step_result": {"is_valid": False, "reward": 0.0, "failure_reason": str(exc)},
                    "time_taken_seconds": round(t_end - t_start, 2),
                    "actor": "agent",
                })
                continue

            t_end = time.time()
            total_reward += step_result.reward

            # Show what the agent did
            print(f"\n  Agent anchored evidence: {action.anchored_evidence_ids}")
            print(f"  Agent anchored witnesses: {action.anchored_witness_ids}")
            print(f"  Agent cited IPC sections: {action.cited_ipc_sections}")
            if action.judgment:
                print(f"  Agent judgment: {action.judgment.value.upper()}")

            display_step_result(step_num, step_result, observation.chain_score)

            # Human prosecution challenge (menu-based)
            if step_result.is_valid:
                print("\n  Your turn as prosecution — select a challenge:")
                print("    1. Challenge witness credibility")
                print("    2. Challenge evidence admissibility")
                print("    3. Challenge IPC section applicability")
                print("    4. No challenge (accept step)")
                challenge_choice = _ask_int("  > ", range(1, 5))
                challenges = [
                    "Prosecution challenge: witness credibility is disputed",
                    "Prosecution challenge: evidence admissibility questioned",
                    "Prosecution challenge: IPC section applicability disputed",
                    None,
                ]
                human_challenge = challenges[challenge_choice - 1]
                if human_challenge:
                    print(f"\n  🏛  You challenge: \"{human_challenge}\"")

            turns.append({
                "step_type": step_type.value,
                "action": _serialize(action),
                "step_result": _serialize(step_result),
                "time_taken_seconds": round(t_end - t_start, 2),
                "actor": "agent",
            })

    # ── Verdict ──────────────────────────────────────────────────────────
    valid_steps = len(observation.submitted_steps)
    human_won = (mode == "defence" and valid_steps == 6) or (mode == "prosecution" and valid_steps < 6)
    display_verdict(observation, total_reward, step_num)

    # ── Save session ─────────────────────────────────────────────────────
    session_path = _save_session(
        mode=mode,
        curriculum_level=curriculum_level,
        case_file=case_file,
        turns=turns,
        observation=observation,
        total_reward=total_reward,
        human_won=human_won,
    )
    print(f"\n  Session saved: {session_path}")


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    """Main loop: play sessions until the human quits."""
    print("\n" + "═" * 44)
    print("   NyayaRL — Legal Argument Chain Trainer")
    print("═" * 44)

    while True:
        try:
            run_session()
        except KeyboardInterrupt:
            print("\n\n  Session interrupted.")

        print()
        again = input("  Play again? (y/n): ").strip().lower()
        if again != "y":
            print("\n  Thanks for playing. Sessions logged for imitation learning.")
            print("  Goodbye!\n")
            break


if __name__ == "__main__":
    main()
