from __future__ import annotations

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nyayarl.agents import DefenceAgent
from nyayarl.models import Action, JudgmentLabel

from human_mode.session_logic import (
    build_environment,
    format_case_file,
    format_step_result,
    next_step_type,
    write_session_log,
)


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    cands = sorted(checkpoint_dir.glob("checkpoint_step_*.json"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _load_defence_from_checkpoint() -> tuple[DefenceAgent | None, str | None]:
    ckpt_dir = Path("checkpoints")
    latest = _latest_checkpoint(ckpt_dir)
    if latest is None:
        return None, None
    with open(latest, "r") as f:
        state = json.load(f)
    defence_state = state.get("defence_agent", {})
    return DefenceAgent.from_state_dict(defence_state), str(latest)


def main() -> int:
    print("NyayaRL Human Mode")
    print("==================")
    print("")

    print("Mode selection:")
    print("1) human (you pick actions)")
    print("2) demo (policy proposes actions)")
    mode = input("Select mode [1/2]: ").strip() or "1"

    level_str = input("Select curriculum level [1-4]: ").strip() or "1"
    curriculum_level = max(1, min(4, int(level_str)))

    defence, ckpt_path = _load_defence_from_checkpoint()
    if ckpt_path:
        print(f"Loaded trained checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found. Running in demo mode (untrained policy).")

    env = build_environment()
    obs = env.reset(curriculum_level)
    print("")
    print("CASE FILE")
    print("---------")
    print(format_case_file(obs))

    session_steps: list[dict] = []
    started_at = time.time()

    try:
        while not obs.is_done:
            st = next_step_type(obs)
            print("")
            print(f"Step menu — next step: {st.value}")
            print("Provide comma-separated IDs; leave blank for none.")

            if mode == "2":
                if defence is None:
                    defence = DefenceAgent()
                action = defence.select_action(obs)
                print(f"[demo] proposed action: {action}")
            else:
                evidence_ids = input("Evidence IDs: ").strip()
                witness_ids = input("Witness IDs: ").strip()
                ipc_sections = input("IPC sections: ").strip()
                judgment_str = ""
                if st.value == "precedent_citation":
                    judgment_str = input("Judgment (acquit/convict/partial): ").strip()

                def _split(x: str) -> list[str]:
                    return [s.strip() for s in x.split(",") if s.strip()]

                judgment = None
                if judgment_str:
                    judgment = JudgmentLabel(judgment_str)

                action = Action(
                    step_type=st,
                    anchored_evidence_ids=_split(evidence_ids),
                    anchored_witness_ids=_split(witness_ids),
                    cited_ipc_sections=_split(ipc_sections),
                    judgment=judgment,
                )

            obs, step_result, done = env.step(action)
            print("")
            print(format_step_result(st, step_result, done, obs))
            session_steps.append(
                {
                    "action": {
                        "step_type": action.step_type.value,
                        "anchored_evidence_ids": action.anchored_evidence_ids,
                        "anchored_witness_ids": action.anchored_witness_ids,
                        "cited_ipc_sections": action.cited_ipc_sections,
                        "judgment": action.judgment.value if action.judgment else None,
                    },
                    "step_result": {
                        "is_valid": step_result.is_valid,
                        "reward": step_result.reward,
                        "failure_reason": step_result.failure_reason,
                        "prosecution_challenge": step_result.prosecution_challenge,
                    },
                    "done": done,
                }
            )

    finally:
        ended_at = time.time()
        path = write_session_log(
            {
                "started_at": started_at,
                "ended_at": ended_at,
                "duration_sec": ended_at - started_at,
                "checkpoint": ckpt_path,
                "curriculum_level": curriculum_level,
                "steps": session_steps,
            },
            filename=f"session_{int(started_at)}.json",
        )
        print("")
        print(f"Session log written to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

