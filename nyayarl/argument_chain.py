"""
argument_chain.py — Deterministic, rule-based validator for the 6-step
legal argument chain.

No LLM, no randomness. Takes an Action + episode history, checks whether
the action satisfies the logical dependency rules for its step type,
returns a StepResult. This is the core of the reward signal.
"""

from __future__ import annotations

from nyayarl.models import (
    Action,
    CaseFile,
    EvidenceType,
    JudgmentLabel,
    StepResult,
    StepType,
)

# ── Canonical step ordering ──────────────────────────────────────────────────

_STEP_ORDER: list[StepType] = [
    StepType.ACTUS_REUS,
    StepType.MENS_REA,
    StepType.LINKAGE,
    StepType.COUNTER_ARGUMENT,
    StepType.IPC_APPLICATION,
    StepType.PRECEDENT_CITATION,
]

# ── Hardcoded reward values (will be imported from scoring_rubric.py later) ─

_REWARD_VALID_STEP = 1.0
_REWARD_PROACTIVE_COUNTER = 1.5
_PENALTY_MISSING_FOUNDATION = -1.0
_PENALTY_EVIDENCE_NOT_PRESENT = -0.5
_PENALTY_IPC_NOT_APPLICABLE = -0.75
_PENALTY_SKIPPED_STEP = -1.0
_PENALTY_CONTRADICTING_WITNESS = -0.25
_PENALTY_UNNECESSARY_IPC = -0.1


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
                reward=_PENALTY_SKIPPED_STEP,
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
            StepType.COUNTER_ARGUMENT: self._validate_counter_argument,
            StepType.IPC_APPLICATION: self._validate_ipc_application,
            StepType.PRECEDENT_CITATION: self._validate_precedent_citation,
        }
        return dispatch[action.step_type](action, case_file, submitted_steps)

    # ── Private — per-step validators ────────────────────────────────────

    def _validate_actus_reus(
        self,
        action: Action,
        case_file: CaseFile,
        submitted_steps: list[Action],
    ) -> StepResult:
        """Step 1 — Actus Reus."""
        # Must anchor at least one evidence ID
        if not action.anchored_evidence_ids:
            return StepResult(
                is_valid=False,
                reward=_PENALTY_EVIDENCE_NOT_PRESENT,
                failure_reason="Actus Reus requires at least one anchored evidence ID",
            )

        evidence_map = {e.id: e for e in case_file.evidence_items}

        for eid in action.anchored_evidence_ids:
            # Evidence must exist in the case file
            if eid not in evidence_map:
                return StepResult(
                    is_valid=False,
                    reward=_PENALTY_EVIDENCE_NOT_PRESENT,
                    failure_reason=f"Evidence ID '{eid}' not found in case file",
                )
            item = evidence_map[eid]
            # Evidence must be present
            if not item.is_present:
                return StepResult(
                    is_valid=False,
                    reward=_PENALTY_EVIDENCE_NOT_PRESENT,
                    failure_reason=f"Evidence '{eid}' is not present (is_present=False)",
                )
            # Type must be PHYSICAL or FORENSIC
            if item.type not in (EvidenceType.PHYSICAL, EvidenceType.FORENSIC):
                return StepResult(
                    is_valid=False,
                    reward=_PENALTY_EVIDENCE_NOT_PRESENT,
                    failure_reason=(
                        f"Evidence '{eid}' must be PHYSICAL or FORENSIC for "
                        f"Actus Reus, got {item.type.value}"
                    ),
                )

        return StepResult(is_valid=True, reward=_REWARD_VALID_STEP)

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

        # Must anchor at least one witness ID
        if not action.anchored_witness_ids:
            return StepResult(
                is_valid=False,
                reward=_PENALTY_MISSING_FOUNDATION,
                failure_reason="Mens Rea requires at least one anchored witness ID",
            )

        witness_map = {w.id: w for w in case_file.witness_statements}

        has_contradicting = False
        for wid in action.anchored_witness_ids:
            # Witness must exist
            if wid not in witness_map:
                return StepResult(
                    is_valid=False,
                    reward=_PENALTY_MISSING_FOUNDATION,
                    failure_reason=f"Witness ID '{wid}' not found in case file",
                )
            witness = witness_map[wid]
            # Reliability must be > 0.5
            if witness.reliability <= 0.5:
                return StepResult(
                    is_valid=False,
                    reward=_PENALTY_MISSING_FOUNDATION,
                    failure_reason=(
                        f"Witness '{wid}' reliability is {witness.reliability}, "
                        f"must be above 0.5"
                    ),
                )
            if witness.is_contradicting:
                has_contradicting = True

        reward = _REWARD_VALID_STEP
        prosecution_challenge = None
        if has_contradicting:
            reward += _PENALTY_CONTRADICTING_WITNESS
            prosecution_challenge = (
                "Prosecution challenge: contradicting witness cited — "
                "credibility of testimony is disputed"
            )

        return StepResult(
            is_valid=True,
            reward=reward,
            prosecution_challenge=prosecution_challenge,
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

        # Must anchor at least one evidence ID of type DOCUMENTARY or FORENSIC
        if not action.anchored_evidence_ids:
            return StepResult(
                is_valid=False,
                reward=_PENALTY_EVIDENCE_NOT_PRESENT,
                failure_reason=(
                    "Linkage requires at least one anchored evidence ID "
                    "(DOCUMENTARY or FORENSIC)"
                ),
            )

        evidence_map = {e.id: e for e in case_file.evidence_items}

        for eid in action.anchored_evidence_ids:
            if eid not in evidence_map:
                return StepResult(
                    is_valid=False,
                    reward=_PENALTY_EVIDENCE_NOT_PRESENT,
                    failure_reason=f"Evidence ID '{eid}' not found in case file",
                )
            item = evidence_map[eid]
            if item.type not in (EvidenceType.DOCUMENTARY, EvidenceType.FORENSIC):
                return StepResult(
                    is_valid=False,
                    reward=_PENALTY_EVIDENCE_NOT_PRESENT,
                    failure_reason=(
                        f"Evidence '{eid}' must be DOCUMENTARY or FORENSIC for "
                        f"Linkage, got {item.type.value}"
                    ),
                )

        # Must cite at least one IPC section
        if not action.cited_ipc_sections:
            return StepResult(
                is_valid=False,
                reward=_PENALTY_IPC_NOT_APPLICABLE,
                failure_reason="Linkage requires at least one cited IPC section",
            )

        # Every cited section must be in applicable list
        for section in action.cited_ipc_sections:
            if section not in case_file.applicable_ipc_sections:
                return StepResult(
                    is_valid=False,
                    reward=_PENALTY_IPC_NOT_APPLICABLE,
                    failure_reason=(
                        f"IPC section '{section}' is not in the case file's "
                        f"applicable sections"
                    ),
                )

        return StepResult(is_valid=True, reward=_REWARD_VALID_STEP)

    def _validate_counter_argument(
        self,
        action: Action,
        case_file: CaseFile,
        submitted_steps: list[Action],
    ) -> StepResult:
        """Step 4 — Counter Argument."""
        # Steps 1, 2, 3 must exist
        result = self._require_prior_steps(submitted_steps, required_count=3)
        if result is not None:
            return result

        # Must anchor at least one witness ID or evidence ID
        if not action.anchored_witness_ids and not action.anchored_evidence_ids:
            return StepResult(
                is_valid=False,
                reward=_PENALTY_MISSING_FOUNDATION,
                failure_reason=(
                    "Counter Argument requires at least one anchored witness ID "
                    "or evidence ID"
                ),
            )

        # Validate referenced evidence IDs exist
        evidence_ids = {e.id for e in case_file.evidence_items}
        for eid in action.anchored_evidence_ids:
            if eid not in evidence_ids:
                return StepResult(
                    is_valid=False,
                    reward=_PENALTY_EVIDENCE_NOT_PRESENT,
                    failure_reason=f"Evidence ID '{eid}' not found in case file",
                )

        # Validate referenced witness IDs exist
        witness_ids = {w.id for w in case_file.witness_statements}
        for wid in action.anchored_witness_ids:
            if wid not in witness_ids:
                return StepResult(
                    is_valid=False,
                    reward=_PENALTY_MISSING_FOUNDATION,
                    failure_reason=f"Witness ID '{wid}' not found in case file",
                )

        # Proactive counter-argument bonus: if the agent gets here before
        # any prosecution challenge has been raised in prior steps, reward
        # the proactive defence.  We detect this by checking whether any
        # prior StepResult attached a prosecution_challenge — but since we
        # only have submitted Actions (not StepResults) here, the bonus is
        # awarded unconditionally.  (The environment can refine this later.)
        return StepResult(is_valid=True, reward=_REWARD_PROACTIVE_COUNTER)

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

        # Must cite at least one IPC section
        if not action.cited_ipc_sections:
            return StepResult(
                is_valid=False,
                reward=_PENALTY_IPC_NOT_APPLICABLE,
                failure_reason="IPC Application requires at least one cited IPC section",
            )

        # Every cited section must be in applicable list
        for section in action.cited_ipc_sections:
            if section not in case_file.applicable_ipc_sections:
                return StepResult(
                    is_valid=False,
                    reward=_PENALTY_IPC_NOT_APPLICABLE,
                    failure_reason=(
                        f"IPC section '{section}' is not in the case file's "
                        f"applicable sections"
                    ),
                )

        # Penalty for sections that were never grounded in steps 1–3
        grounded_sections = self._collect_grounded_ipc_sections(submitted_steps)
        unnecessary = [
            s for s in action.cited_ipc_sections if s not in grounded_sections
        ]
        penalty = len(unnecessary) * _PENALTY_UNNECESSARY_IPC
        reward = _REWARD_VALID_STEP + penalty

        challenge = None
        if unnecessary:
            challenge = (
                f"Prosecution challenge: IPC section(s) "
                f"{', '.join(unnecessary)} cited without prior grounding "
                f"in steps 1–3"
            )

        return StepResult(
            is_valid=True,
            reward=reward,
            prosecution_challenge=challenge,
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

        # Judgment must not be None
        if action.judgment is None:
            return StepResult(
                is_valid=False,
                reward=_PENALTY_MISSING_FOUNDATION,
                failure_reason="Precedent Citation requires a judgment (acquit/convict/partial)",
            )

        # Judgment must be a valid JudgmentLabel (type-safety, but belt-and-suspenders)
        if not isinstance(action.judgment, JudgmentLabel):
            return StepResult(
                is_valid=False,
                reward=_PENALTY_MISSING_FOUNDATION,
                failure_reason=(
                    f"judgment must be a JudgmentLabel, got {type(action.judgment).__name__}"
                ),
            )

        # Anchored evidence IDs should reference the precedent
        if case_file.precedent_id and case_file.precedent_id not in action.anchored_evidence_ids:
            return StepResult(
                is_valid=False,
                reward=_PENALTY_EVIDENCE_NOT_PRESENT,
                failure_reason=(
                    f"Precedent Citation must anchor the case precedent ID "
                    f"'{case_file.precedent_id}'"
                ),
            )

        return StepResult(is_valid=True, reward=_REWARD_VALID_STEP)

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
                reward=_PENALTY_SKIPPED_STEP,
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
