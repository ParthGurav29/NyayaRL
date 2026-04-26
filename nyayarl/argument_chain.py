"""
argument_chain.py — Deterministic, rule-based validator for the 6-step
legal argument chain.

No LLM, no randomness. Takes an Action + episode history, checks whether
the action satisfies the logical dependency rules for its step type,
returns a StepResult. This is the core of the reward signal.
"""

from __future__ import annotations

import zlib

from nyayarl.models import (
    Action,
    CaseFile,
    EvidenceType,
    JudgmentLabel,
    StepResult,
    StepType,
)

from rewards.scoring_rubric import PENALTY_SKIPPED_STEP
from rewards.reward_calculator import RewardCalculator

# ── Canonical step ordering ──────────────────────────────────────────────────

_STEP_ORDER: list[StepType] = [
    StepType.ACTUS_REUS,
    StepType.MENS_REA,
    StepType.LINKAGE,
    StepType.COUNTER_ARGUMENT,
    StepType.IPC_APPLICATION,
    StepType.PRECEDENT_CITATION,
]

# Reward values come from rewards/scoring_rubric.py


class ArgumentChainValidator:
    """
    Stateless validator for a single step in the argument chain.

    Usage::

        validator = ArgumentChainValidator()
        result = validator.validate(action, case_file, submitted_steps)
    """

    # ── Public API ───────────────────────────────────────────────────────

    def validate(
        self,
        action: Action,
        case_file: CaseFile,
        submitted_steps: list[Action],
        prosecution_challenges_so_far: list[str] | None = None,
    ) -> StepResult:
        """
        Validate *action* against *case_file* and *submitted_steps*.

        Returns a fully-populated ``StepResult``.

        Raises ``ValueError`` if ``action.step_type`` is not the expected
        next step given the current history.
        """
        # --- ordering enforcement ---
        expected_index = len(submitted_steps)
        if expected_index >= len(_STEP_ORDER):
            return StepResult(
                is_valid=False,
                reward=PENALTY_SKIPPED_STEP,
                failure_reason="All 6 steps already submitted",
            )

        expected_type = _STEP_ORDER[expected_index]
        if action.step_type != expected_type:
            raise ValueError(
                f"Expected step type {expected_type.value} "
                f"(step {expected_index + 1}), "
                f"got {action.step_type.value}"
            )

        # --- dispatch to per-step validator ---
        dispatch = {
            StepType.ACTUS_REUS: self._validate_actus_reus,
            StepType.MENS_REA: self._validate_mens_rea,
            StepType.LINKAGE: self._validate_linkage,
            StepType.COUNTER_ARGUMENT: lambda a, c, s: self._validate_counter_argument(
                a,
                c,
                s,
                prosecution_challenges_so_far=prosecution_challenges_so_far or [],
            ),
            StepType.IPC_APPLICATION: self._validate_ipc_application,
            StepType.PRECEDENT_CITATION: self._validate_precedent_citation,
        }
        return dispatch[action.step_type](action, case_file, submitted_steps)

    # ── Private — per-step validators ────────────────────────────────────

    def __init__(self) -> None:
        self._rewards = RewardCalculator()

    def _validate_actus_reus(
        self,
        action: Action,
        case_file: CaseFile,
        submitted_steps: list[Action],
    ) -> StepResult:
        """Step 1 — Actus Reus."""
        out = self._rewards.actus_reus(action=action, case_file=case_file)
        return StepResult(
            is_valid=out.is_valid,
            reward=out.reward,
            failure_reason=out.failure_reason,
            prosecution_challenge=out.prosecution_challenge,
        )

    def _validate_mens_rea(
        self,
        action: Action,
        case_file: CaseFile,
        submitted_steps: list[Action],
    ) -> StepResult:
        """Step 2 — Mens Rea."""
        # Step 1 must already exist
        result = self._require_prior_steps(submitted_steps, required_count=1)
        if result is not None:
            return result

        out = self._rewards.mens_rea(action=action, case_file=case_file)
        return StepResult(
            is_valid=out.is_valid,
            reward=out.reward,
            failure_reason=out.failure_reason,
            prosecution_challenge=out.prosecution_challenge,
        )

    def _validate_linkage(
        self,
        action: Action,
        case_file: CaseFile,
        submitted_steps: list[Action],
    ) -> StepResult:
        """Step 3 — Linkage (chain-of-custody)."""
        # Steps 1 and 2 must exist
        result = self._require_prior_steps(submitted_steps, required_count=2)
        if result is not None:
            return result

        out = self._rewards.linkage(action=action, case_file=case_file)
        return StepResult(
            is_valid=out.is_valid,
            reward=out.reward,
            failure_reason=out.failure_reason,
            prosecution_challenge=out.prosecution_challenge,
        )

    def _validate_counter_argument(
        self,
        action: Action,
        case_file: CaseFile,
        submitted_steps: list[Action],
        *,
        prosecution_challenges_so_far: list[str],
    ) -> StepResult:
        """Step 4 — Counter Argument."""
        # Steps 1, 2, 3 must exist
        result = self._require_prior_steps(submitted_steps, required_count=3)
        if result is not None:
            return result

        # No challenges raised yet:
        # award proactive bonus ONLY if the counter matches the *next* challenge target step
        # implied by the challenge bank (deterministic proxy keyed by template_id + progress).
        #
        # (A stricter version would call the prosecution agent; this keeps validator self-contained.)
        template_id = case_file.template_id or ""
        # IMPORTANT: built-in hash() is salted per process; using it here makes reward shaping
        # non-reproducible across runs. Use stable hash to keep training/eval consistent.
        key = f"{template_id}:{len(submitted_steps)}"
        next_target_step = (int(zlib.crc32(key.encode('utf-8','ignore'))) % 3) + 2  # {2,3,4}

        proactive_ok = False
        if next_target_step == 2:
            proactive_ok = bool(action.anchored_witness_ids)
        elif next_target_step in (1, 3, 4, 6):
            proactive_ok = bool(action.anchored_evidence_ids)
        elif next_target_step == 5:
            proactive_ok = bool(action.cited_ipc_sections)

        out = self._rewards.counter_argument(
            action=action,
            case_file=case_file,
            prosecution_challenges_so_far=prosecution_challenges_so_far,
            proactive_ok=proactive_ok,
        )
        return StepResult(
            is_valid=out.is_valid,
            reward=out.reward,
            failure_reason=out.failure_reason,
            prosecution_challenge=out.prosecution_challenge,
        )

    def _validate_ipc_application(
        self,
        action: Action,
        case_file: CaseFile,
        submitted_steps: list[Action],
    ) -> StepResult:
        """Step 5 — IPC Application."""
        # All prior 4 steps must exist
        result = self._require_prior_steps(submitted_steps, required_count=4)
        if result is not None:
            return result

        grounded_sections = self._collect_grounded_ipc_sections(submitted_steps)
        out = self._rewards.ipc_application(
            action=action,
            case_file=case_file,
            grounded_sections=grounded_sections,
        )
        return StepResult(
            is_valid=out.is_valid,
            reward=out.reward,
            failure_reason=out.failure_reason,
            prosecution_challenge=out.prosecution_challenge,
        )

    def _validate_precedent_citation(
        self,
        action: Action,
        case_file: CaseFile,
        submitted_steps: list[Action],
    ) -> StepResult:
        """Step 6 — Precedent Citation."""
        # All prior 5 steps must exist
        result = self._require_prior_steps(submitted_steps, required_count=5)
        if result is not None:
            return result

        # Judgment must be a valid JudgmentLabel (type-safety, but belt-and-suspenders)
        if not isinstance(action.judgment, JudgmentLabel):
            return StepResult(
                is_valid=False,
                reward=self._rewards.precedent_citation(action=action).reward,
                failure_reason=(
                    f"judgment must be a JudgmentLabel, got {type(action.judgment).__name__}"
                ),
            )

        # Track 1's Action schema does not have a dedicated precedent-id field.
        # The environment + judge evaluate precedent match at episode end; here we
        # only require that the agent finalises a judgment to terminate the chain.

        out = self._rewards.precedent_citation(action=action)
        return StepResult(
            is_valid=out.is_valid,
            reward=out.reward,
            failure_reason=out.failure_reason,
            prosecution_challenge=out.prosecution_challenge,
        )

    # ── Private — shared helpers ─────────────────────────────────────────

    @staticmethod
    def _require_prior_steps(
        submitted_steps: list[Action],
        required_count: int,
    ) -> StepResult | None:
        """
        Return a failing ``StepResult`` if fewer than *required_count*
        steps have been submitted.  Returns ``None`` when the requirement
        is satisfied (caller proceeds with its own validation).
        """
        if len(submitted_steps) < required_count:
            missing_step = _STEP_ORDER[len(submitted_steps)]
            return StepResult(
                is_valid=False,
                reward=PENALTY_SKIPPED_STEP,
                failure_reason=(
                    f"Step {len(submitted_steps) + 1} "
                    f"({missing_step.value}) required before this step"
                ),
            )
        return None

    @staticmethod
    def _collect_grounded_ipc_sections(
        submitted_steps: list[Action],
    ) -> set[str]:
        """
        Collect all IPC sections cited in steps 1–3 (indices 0–2).

        Used by Step 5 to determine which sections were properly grounded.
        """
        grounded: set[str] = set()
        for step in submitted_steps[:3]:
            grounded.update(step.cited_ipc_sections)
        return grounded
