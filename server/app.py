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
from nyayarl.models import Action, JudgmentLabel, StepType


# Track 2 components
from nyayarl.case_generator import Track2CaseGenerator
from nyayarl.track2_adapters import Track2JudgeAdapter, Track2ProsecutionAdapter
from nyayarl.agents import JudgeAgent, ProsecutionAgent
from nyayarl.precedents import PrecedentsDB
from nyayarl.agents import JudgeAgent, ProsecutionAgent
from nyayarl.case_generator import Track2CaseGenerator
from nyayarl.precedents import PrecedentsDB
from nyayarl.track2_adapters import Track2JudgeAdapter, Track2ProsecutionAdapter


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


# ── Track 2 component factories ──────────────────────────────────────────────


def _create_track2_components():
    """Create Track 2 components for environment initialization."""
    case_generator = Track2CaseGenerator()
    precedents_db = PrecedentsDB()
    judge_agent = JudgeAgent()
    prosecution_agent = ProsecutionAgent()

    judge = Track2JudgeAdapter(judge_agent, precedents_db)
    prosecution = Track2ProsecutionAdapter(prosecution_agent)

    return case_generator, judge, prosecution

# ── Shared Track2-backed dependencies ───────────────────────────────────────

_precedents = PrecedentsDB()
_judge = Track2JudgeAdapter(JudgeAgent(), _precedents)
_prosecution = Track2ProsecutionAdapter(ProsecutionAgent())


def _new_environment() -> NyayaRLEnvironment:
    # Per-session generator to avoid shared RNG state between users.
    case_generator = Track2CaseGenerator()
    return NyayaRLEnvironment(
        case_generator=case_generator,
        judge=_judge,
        prosecution=_prosecution,
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
    return {
        "status": "ok",
        "precedents_loaded": str(len(_precedents.get_all_case_ids())),
        "sessions": str(len(environments)),
    }


@app.post("/reset")
async def reset(body: ResetRequest) -> JSONResponse:
    """Start a new episode at the given curriculum level for the session."""
    session_id = body.session_id

    # Initialize a new environment if this session doesn't exist
    if session_id not in environments:

        case_gen, judge, prosecution = _create_track2_components()
        environments[session_id] = NyayaRLEnvironment(
            case_generator=case_gen,
            judge=judge,
            prosecution=prosecution,
        )

        environments[session_id] = _new_environment()
    
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
