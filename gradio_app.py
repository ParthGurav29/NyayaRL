from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr
from fastapi import FastAPI

sys.path.insert(0, str(Path(__file__).parent))

from nyayarl.models import Action, JudgmentLabel

from human_mode.session_logic import (
    build_environment,
    format_case_file,
    format_step_result,
    next_step_type,
)


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    cands = sorted(checkpoint_dir.glob("checkpoint_step_*.json"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="NyayaRL Human Mode") as demo:
        env_state = gr.State(value=None)  # NyayaRLEnvironment
        obs_state = gr.State(value=None)  # Observation
        log_state = gr.State(value=[])  # list[dict]

        checkpoint_path = _latest_checkpoint(Path("checkpoints"))
        banner = (
            "Checkpoint loaded: " + checkpoint_path.name
            if checkpoint_path
            else "Demo mode: no checkpoint found in ./checkpoints"
        )
        gr.Markdown(f"**{banner}**")

        with gr.Row():
            level = gr.Dropdown(choices=["1", "2", "3", "4"], value="1", label="Curriculum level")
            start_btn = gr.Button("Start Case", variant="primary")

        case_display = gr.Textbox(label="Case file", lines=18)

        with gr.Row():
            evidence_ids = gr.Textbox(label="Evidence IDs (comma-separated)")
            witness_ids = gr.Textbox(label="Witness IDs (comma-separated)")
        ipc_sections = gr.Textbox(label="IPC sections (comma-separated)")
        judgment = gr.Dropdown(
            choices=["", "acquit", "convict", "partial"],
            value="",
            label="Judgment (only step 6)",
        )
        submit_btn = gr.Button("Submit Step")

        result_display = gr.Textbox(label="Step result", lines=10)

        with gr.Tab("Session log"):
            log_json = gr.JSON(label="Log")

        def start_case(level_str: str):
            env = build_environment()
            obs = env.reset(int(level_str))
            return env, obs, [], format_case_file(obs), "", []

        def submit_step(env, obs, log: list[dict], ev: str, wi: str, ipc: str, judg: str):
            if env is None or obs is None:
                return env, obs, log, "Start a case first.", log

            st = next_step_type(obs)

            def _split(x: str) -> list[str]:
                return [s.strip() for s in (x or "").split(",") if s.strip()]

            j = None
            if st.value == "precedent_citation" and judg:
                j = JudgmentLabel(judg)

            action = Action(
                step_type=st,
                anchored_evidence_ids=_split(ev),
                anchored_witness_ids=_split(wi),
                cited_ipc_sections=_split(ipc),
                judgment=j,
            )

            obs2, step_result, done = env.step(action)
            out = format_step_result(st, step_result, done, obs2)

            log = list(log) + [
                {
                    "step_type": st.value,
                    "action": {
                        "anchored_evidence_ids": action.anchored_evidence_ids,
                        "anchored_witness_ids": action.anchored_witness_ids,
                        "cited_ipc_sections": action.cited_ipc_sections,
                        "judgment": action.judgment.value if action.judgment else None,
                    },
                    "result": {
                        "is_valid": step_result.is_valid,
                        "reward": step_result.reward,
                        "failure_reason": step_result.failure_reason,
                        "prosecution_challenge": step_result.prosecution_challenge,
                    },
                    "done": done,
                }
            ]

            return env, obs2, log, out, log

        start_btn.click(
            fn=start_case,
            inputs=[level],
            outputs=[env_state, obs_state, log_state, case_display, result_display, log_json],
        )
        submit_btn.click(
            fn=submit_step,
            inputs=[env_state, obs_state, log_state, evidence_ids, witness_ids, ipc_sections, judgment],
            outputs=[env_state, obs_state, log_state, result_display, log_json],
        )

    return demo


fastapi_app = FastAPI()


@fastapi_app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


ui = build_ui()
app = gr.mount_gradio_app(fastapi_app, ui, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)

