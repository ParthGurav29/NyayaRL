"""
app.py — FastAPI layer for NyayaRL.

Runs from the project root via: uvicorn server.app:app --reload
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Imports from nyayarl core logic
from nyayarl.environment import NyayaRLEnvironment
from nyayarl.models import (
    Action,
    CaseFile,
    EvidenceItem,
    EvidenceType,
    JudgmentLabel,
    StepType,
    Verdict,
    WitnessStatement,
)

# ── Global State ─────────────────────────────────────────────────────────────

# Dictionary to track environment sessions
environments: Dict[str, NyayaRLEnvironment] = {}

# ── Pydantic request models ──────────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Request body for ``POST /reset``."""
    session_id: str
    curriculum_level: int = Field(..., ge=1, le=4)


class ActionRequest(BaseModel):
    """
    Request body for ``POST /step``.
    Mirrors ``models.Action`` fields for JSON deserialization.
    """
    session_id: str
    step_type: str
    cited_ipc_sections: list[str] = []
    anchored_evidence_ids: list[str] = []
    anchored_witness_ids: list[str] = []
    judgment: str | None = None


# ── Serialisation helpers ────────────────────────────────────────────────────

def _serialize(obj: Any) -> Any:
    """
    Recursively convert a dataclass (with nested dataclasses and enums)
    into a JSON-serializable dict.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def _parse_action(body: ActionRequest) -> Action:
    """Convert a Pydantic ``ActionRequest`` into a ``models.Action``."""
    try:
        step_type = StepType(body.step_type)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid step_type: '{body.step_type}'. "
            f"Must be one of: {[s.value for s in StepType]}",
        )

    judgment = None
    if body.judgment is not None:
        try:
            judgment = JudgmentLabel(body.judgment)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid judgment: '{body.judgment}'. "
                f"Must be one of: {[j.value for j in JudgmentLabel]}",
            )

    return Action(
        step_type=step_type,
        cited_ipc_sections=body.cited_ipc_sections,
        anchored_evidence_ids=body.anchored_evidence_ids,
        anchored_witness_ids=body.anchored_witness_ids,
        judgment=judgment,
    )


# ── Stub dependencies ────────────────────────────────────────────────────────

class StubCaseGenerator:
    """Generates a minimal but structurally valid case file."""
    def generate(self, curriculum_level: int) -> CaseFile:
        return CaseFile(
            case_id=f"STUB_{curriculum_level:03d}",
            fir="[Stub FIR] Placeholder first information report.",
            accused_count=1,
            evidence_items=[
                EvidenceItem(
                    id="E1",
                    description="Physical evidence (stub)",
                    type=EvidenceType.PHYSICAL,
                    is_present=True,
                ),
                EvidenceItem(
                    id="E2",
                    description="Forensic evidence (stub)",
                    type=EvidenceType.FORENSIC,
                    is_present=True,
                ),
                EvidenceItem(
                    id="E3",
                    description="Documentary evidence (stub)",
                    type=EvidenceType.DOCUMENTARY,
                    is_present=True,
                ),
            ],
            witness_statements=[
                WitnessStatement(
                    id="W1",
                    content="Witness statement (stub)",
                    reliability=0.85,
                    is_contradicting=False,
                ),
            ],
            applicable_ipc_sections=["302", "307", "34"],
            curriculum_level=curriculum_level,
            precedent_id="ILDC_STUB_001",
        )


class StubJudge:
    """Returns a placeholder verdict."""
    def evaluate(
        self,
        case_file: CaseFile,
        submitted_steps: list[Action],
    ) -> Verdict:
        return Verdict(
            judgment=submitted_steps[-1].judgment or JudgmentLabel.PARTIAL,
            matched_precedent_id=case_file.precedent_id,
            precedent_matched=True,
            total_reward=0.0,
            prosecution_win_rate=0.0,
        )


# ── App instance ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="NyayaRL Environment",
    description="Stateful RL environment for Indian legal reasoning.",
    version="0.1.0",
)

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check for Docker and deployment."""
    return {"status": "ok"}


@app.post("/reset")
async def reset(body: ResetRequest) -> JSONResponse:
    """Start a new episode at the given curriculum level for the session."""
    session_id = body.session_id
    
    # Initialize a new environment if this session doesn't exist
    if session_id not in environments:
        environments[session_id] = NyayaRLEnvironment(
            case_generator=StubCaseGenerator(),
            judge=StubJudge(),
        )
    
    env = environments[session_id]
    
    try:
        observation = env.reset(body.curriculum_level)
        return JSONResponse(content=_serialize(observation))
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )


@app.post("/step")
async def step(body: ActionRequest) -> JSONResponse:
    """Submit one action in the argument chain for a specific session."""
    session_id = body.session_id
    
    if session_id not in environments:
        raise HTTPException(
            status_code=400,
            detail=f"Session '{session_id}' not found. Call /reset first to initialize."
        )
        
    env = environments[session_id]
    
    # Convert request to domain Action
    action = _parse_action(body)

    try:
        observation, step_result, done = env.step(action)
    except RuntimeError as exc:
        msg = str(exc)
        if "already done" in msg:
            raise HTTPException(
                status_code=400,
                detail="Episode is done. Call /reset to start a new episode.",
            )
        raise HTTPException(
            status_code=400,
            detail="Episode not initialised. Call /reset first.",
        )
    except ValueError as exc:
        # Wrong step ordering
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc)},
        )

    return JSONResponse(
        content={
            "observation": _serialize(observation),
            "step_result": _serialize(step_result),
            "done": done,
        }
    )
