"""
models.py — Single source of truth for every data shape in NyayaRL.

No logic, no functions, no imports from this project.
Only data shape definitions using dataclasses and Enum.
Every other file in the system imports from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enums ────────────────────────────────────────────────────────────────────


class StepType(Enum):
    """The six sequential steps in a legal argument chain."""

    ACTUS_REUS = "actus_reus"
    MENS_REA = "mens_rea"
    LINKAGE = "linkage"
    COUNTER_ARGUMENT = "counter_argument"
    IPC_APPLICATION = "ipc_application"
    PRECEDENT_CITATION = "precedent_citation"


class JudgmentLabel(Enum):
    """Possible final judgment recommendations."""

    ACQUIT = "acquit"
    CONVICT = "convict"
    PARTIAL = "partial"


class EvidenceType(Enum):
    """Categories of evidence that can appear in a case file."""

    PHYSICAL = "physical"
    FORENSIC = "forensic"
    DOCUMENTARY = "documentary"
    TESTIMONIAL = "testimonial"


# ── Case-level schemas ──────────────────────────────────────────────────────


@dataclass
class EvidenceItem:
    """A single piece of evidence in the case file."""

    id: str
    description: str
    type: EvidenceType
    is_present: bool


@dataclass
class WitnessStatement:
    """A witness statement with a reliability score."""

    id: str
    content: str
    reliability: float  # must be in [0.0, 1.0]
    is_contradicting: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError(
                f"WitnessStatement reliability must be between 0.0 and 1.0, "
                f"got {self.reliability}"
            )


@dataclass
class CaseFile:
    """
    Full case file — the input to the environment at episode start.

    Contains the FIR, evidence, witness statements, accused details,
    applicable IPC sections, and curriculum metadata.
    """

    case_id: str
    fir: str
    accused_count: int
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    witness_statements: list[WitnessStatement] = field(default_factory=list)
    applicable_ipc_sections: list[str] = field(default_factory=list)
    curriculum_level: int = 1  # must be 1–4
    precedent_id: str = ""

    def __post_init__(self) -> None:
        if self.curriculum_level not in (1, 2, 3, 4):
            raise ValueError(
                f"CaseFile curriculum_level must be 1–4, "
                f"got {self.curriculum_level}"
            )


# ── Action / Step schemas ───────────────────────────────────────────────────


@dataclass
class Action:
    """
    What an agent sends to the environment at each step.

    `judgment` must be None for steps 1–5 and set for step 6
    (PRECEDENT_CITATION).
    """

    step_type: StepType
    cited_ipc_sections: list[str] = field(default_factory=list)
    anchored_evidence_ids: list[str] = field(default_factory=list)
    anchored_witness_ids: list[str] = field(default_factory=list)
    judgment: Optional[JudgmentLabel] = None


@dataclass
class StepResult:
    """What the environment returns after processing one action."""

    is_valid: bool
    reward: float
    failure_reason: Optional[str] = None
    prosecution_challenge: Optional[str] = None


# ── Observation / Verdict schemas ───────────────────────────────────────────


@dataclass
class Observation:
    """
    Full state snapshot returned to the agent after each step.

    Contains the case context, history of submitted actions and
    prosecution challenges, running score, and termination flag.
    """

    case_file: CaseFile
    submitted_steps: list[Action] = field(default_factory=list)
    prosecution_challenges: list[str] = field(default_factory=list)
    chain_score: float = 0.0
    curriculum_level: int = 1
    is_done: bool = False


@dataclass
class Verdict:
    """Terminal output at episode end."""

    judgment: JudgmentLabel
    matched_precedent_id: str
    precedent_matched: bool
    total_reward: float
    prosecution_win_rate: float
