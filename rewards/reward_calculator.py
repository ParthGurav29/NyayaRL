"""
reward_calculator.py — Centralised reward computation for NyayaRL.

The project historically embedded reward numbers inside:
- `nyayarl/argument_chain.py` (per-step shaping + some challenge logic)
- `nyayarl/environment.py` (terminal rewards)

The performance audit expects a stable module at `rewards/reward_calculator.py`
that actually implements the reward conditions. This file is that source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from nyayarl.models import Action, CaseFile, EvidenceType, StepType
from rewards.scoring_rubric import (
    PENALTY_CONTRADICTING_WITNESS,
    PENALTY_EVIDENCE_NOT_PRESENT,
    PENALTY_IPC_NOT_APPLICABLE,
    PENALTY_MISSING_FOUNDATION,
    PENALTY_SKIPPED_STEP,
    PENALTY_UNNECESSARY_IPC,
    REWARD_PROACTIVE_COUNTER,
    REWARD_VALID_STEP,
    TERMINAL_EFFICIENCY_BONUS,
    TERMINAL_PRECEDENT_MATCHED,
    TERMINAL_PRECEDENT_UNMATCHED,
    TERMINAL_PROSECUTION_PENALTY,
    TERMINAL_STUCK_PENALTY,
    TERMINAL_TIMEOUT_PENALTY,
)


@dataclass(frozen=True)
class RewardOutcome:
    is_valid: bool
    reward: float
    failure_reason: str | None = None
    prosecution_challenge: str | None = None


class RewardCalculator:
    """
    Computes rewards/penalties for individual steps and episode termination.

    NOTE: Validity decisions live in `ArgumentChainValidator`. This class assumes
    those checks have already been applied (or uses the same predicates) and
    focuses on reward accounting.
    """

    # ── per-step rewards ──────────────────────────────────────────────

    def actus_reus(self, *, action: Action, case_file: CaseFile) -> RewardOutcome:
        # Step 1 condition: present PHYSICAL/FORENSIC evidence anchored.
        if not action.anchored_evidence_ids:
            return RewardOutcome(
                is_valid=False,
                reward=PENALTY_EVIDENCE_NOT_PRESENT,
                failure_reason="Actus Reus requires at least one anchored evidence ID",
            )

        evidence_map = {e.id: e for e in case_file.evidence_items}
        for eid in action.anchored_evidence_ids:
            if eid not in evidence_map:
                return RewardOutcome(
                    is_valid=False,
                    reward=PENALTY_EVIDENCE_NOT_PRESENT,
                    failure_reason=f"Evidence ID '{eid}' not found in case file",
                )
            item = evidence_map[eid]
            if not item.is_present:
                return RewardOutcome(
                    is_valid=False,
                    reward=PENALTY_EVIDENCE_NOT_PRESENT,
                    failure_reason=f"Evidence '{eid}' is not present (is_present=False)",
                )
            if item.type not in (EvidenceType.PHYSICAL, EvidenceType.FORENSIC):
                return RewardOutcome(
                    is_valid=False,
                    reward=PENALTY_EVIDENCE_NOT_PRESENT,
                    failure_reason=(
                        f"Evidence '{eid}' must be PHYSICAL or FORENSIC for Actus Reus, "
                        f"got {item.type.value}"
                    ),
                )
        return RewardOutcome(is_valid=True, reward=REWARD_VALID_STEP)

    def mens_rea(self, *, action: Action, case_file: CaseFile) -> RewardOutcome:
        # Step 2 condition: witness anchored with reliability > 0.5
        if not action.anchored_witness_ids:
            return RewardOutcome(
                is_valid=False,
                reward=PENALTY_MISSING_FOUNDATION,
                failure_reason="Mens Rea requires at least one anchored witness ID",
            )

        witness_map = {w.id: w for w in case_file.witness_statements}
        has_contradicting = False
        for wid in action.anchored_witness_ids:
            if wid not in witness_map:
                return RewardOutcome(
                    is_valid=False,
                    reward=PENALTY_MISSING_FOUNDATION,
                    failure_reason=f"Witness ID '{wid}' not found in case file",
                )
            w = witness_map[wid]
            if w.reliability <= 0.5:
                return RewardOutcome(
                    is_valid=False,
                    reward=PENALTY_MISSING_FOUNDATION,
                    failure_reason=f"Witness '{wid}' reliability is {w.reliability}, must be above 0.5",
                )
            if w.is_contradicting:
                has_contradicting = True

        reward = REWARD_VALID_STEP
        challenge = None
        if has_contradicting:
            reward += PENALTY_CONTRADICTING_WITNESS
            contradicting_ids = [wid for wid in action.anchored_witness_ids if witness_map.get(wid) and witness_map[wid].is_contradicting]
            challenge = (
                "Prosecution challenge: contradicting witness cited "
                f"(witness_id={contradicting_ids[0] if contradicting_ids else 'unknown'}) — "
                "credibility of testimony is disputed"
            )
        return RewardOutcome(is_valid=True, reward=reward, prosecution_challenge=challenge)

    def linkage(self, *, action: Action, case_file: CaseFile) -> RewardOutcome:
        # Step 3 condition: present documentary/digital/forensic evidence anchored + IPC cited (applicable)
        if not action.anchored_evidence_ids:
            return RewardOutcome(
                is_valid=False,
                reward=PENALTY_EVIDENCE_NOT_PRESENT,
                failure_reason="Linkage requires at least one anchored evidence ID (DOCUMENTARY, DIGITAL or FORENSIC)",
            )

        evidence_map = {e.id: e for e in case_file.evidence_items}
        for eid in action.anchored_evidence_ids:
            if eid not in evidence_map:
                return RewardOutcome(
                    is_valid=False,
                    reward=PENALTY_EVIDENCE_NOT_PRESENT,
                    failure_reason=f"Evidence ID '{eid}' not found in case file",
                )
            item = evidence_map[eid]
            if not item.is_present:
                return RewardOutcome(
                    is_valid=False,
                    reward=PENALTY_EVIDENCE_NOT_PRESENT,
                    failure_reason=f"Evidence '{eid}' is not present (is_present=False)",
                )
            if item.type not in (EvidenceType.DOCUMENTARY, EvidenceType.DIGITAL, EvidenceType.FORENSIC):
                return RewardOutcome(
                    is_valid=False,
                    reward=PENALTY_EVIDENCE_NOT_PRESENT,
                    failure_reason=f"Evidence '{eid}' must be DOCUMENTARY, DIGITAL or FORENSIC for Linkage, got {item.type.value}",
                )

        if not action.cited_ipc_sections:
            return RewardOutcome(
                is_valid=False,
                reward=PENALTY_IPC_NOT_APPLICABLE,
                failure_reason="Linkage requires at least one cited IPC section",
            )
        for section in action.cited_ipc_sections:
            if section not in case_file.applicable_ipc_sections:
                return RewardOutcome(
                    is_valid=False,
                    reward=PENALTY_IPC_NOT_APPLICABLE,
                    failure_reason="IPC section '{section}' is not in the case file's applicable sections".format(section=section),
                )
        return RewardOutcome(is_valid=True, reward=REWARD_VALID_STEP)

    def counter_argument(
        self,
        *,
        action: Action,
        case_file: CaseFile,
        prosecution_challenges_so_far: list[str],
        proactive_ok: bool,
    ) -> RewardOutcome:
        # Step 4 base validity: must anchor at least one witness/evidence; IDs must exist.
        if not action.anchored_witness_ids and not action.anchored_evidence_ids:
            return RewardOutcome(
                is_valid=False,
                reward=PENALTY_MISSING_FOUNDATION,
                failure_reason="Counter Argument requires at least one anchored witness ID or evidence ID",
            )
        evidence_ids = {e.id for e in case_file.evidence_items}
        for eid in action.anchored_evidence_ids:
            if eid not in evidence_ids:
                return RewardOutcome(
                    is_valid=False,
                    reward=PENALTY_EVIDENCE_NOT_PRESENT,
                    failure_reason=f"Evidence ID '{eid}' not found in case file",
                )
        witness_ids = {w.id for w in case_file.witness_statements}
        for wid in action.anchored_witness_ids:
            if wid not in witness_ids:
                return RewardOutcome(
                    is_valid=False,
                    reward=PENALTY_MISSING_FOUNDATION,
                    failure_reason=f"Witness ID '{wid}' not found in case file",
                )

        # Condition: prosecution challenge present → reward only if addressed
        if prosecution_challenges_so_far:
            text = " ".join(prosecution_challenges_so_far)
            anchored = set(action.anchored_evidence_ids) | set(action.anchored_witness_ids)
            addressed = any(a and (a in text) for a in anchored)
            return RewardOutcome(
                is_valid=True,
                reward=(REWARD_VALID_STEP if addressed else 0.0),
                failure_reason=None if addressed else "Counter Argument did not address any raised prosecution challenge",
            )

        # Condition: proactive counter-argument bonus (targeted)
        if proactive_ok:
            return RewardOutcome(is_valid=True, reward=REWARD_PROACTIVE_COUNTER)

        return RewardOutcome(
            is_valid=True,
            reward=0.0,
            failure_reason="No prosecution challenge raised yet; counter argument not targeted (no bonus)",
        )

    def ipc_application(self, *, action: Action, case_file: CaseFile, grounded_sections: set[str]) -> RewardOutcome:
        # Step 5 condition: cite applicable sections; penalise unnecessary sections (not grounded)
        if not action.cited_ipc_sections:
            return RewardOutcome(
                is_valid=False,
                reward=PENALTY_IPC_NOT_APPLICABLE,
                failure_reason="IPC Application requires at least one cited IPC section",
            )
        for section in action.cited_ipc_sections:
            if section not in case_file.applicable_ipc_sections:
                return RewardOutcome(
                    is_valid=False,
                    reward=PENALTY_IPC_NOT_APPLICABLE,
                    failure_reason=f"IPC section '{section}' is not in the case file's applicable sections",
                )

        unnecessary = [s for s in action.cited_ipc_sections if s not in grounded_sections]
        penalty = len(unnecessary) * PENALTY_UNNECESSARY_IPC
        reward = REWARD_VALID_STEP + penalty
        challenge = None
        if unnecessary:
            challenge = (
                "Prosecution challenge: IPC section(s) "
                f"{', '.join(unnecessary)} cited without prior grounding in steps 1–3"
            )
        return RewardOutcome(is_valid=True, reward=reward, prosecution_challenge=challenge)

    def precedent_citation(self, *, action: Action) -> RewardOutcome:
        # Step 6 condition: must provide judgment label (validator enforces type)
        if action.judgment is None:
            return RewardOutcome(
                is_valid=False,
                reward=PENALTY_MISSING_FOUNDATION,
                failure_reason="Precedent Citation requires a judgment (acquit/convict/partial)",
            )
        return RewardOutcome(is_valid=True, reward=REWARD_VALID_STEP)

    # ── terminal rewards ───────────────────────────────────────────────

    def terminal(self, *, termination: str, verdict_precedent_matched: bool | None, action_count: int) -> float:
        if termination == "stuck":
            return float(TERMINAL_STUCK_PENALTY)
        if termination == "timeout":
            return float(TERMINAL_TIMEOUT_PENALTY)
        if termination != "success":
            return 0.0

        reward = 0.0
        reward += TERMINAL_PRECEDENT_MATCHED if verdict_precedent_matched else TERMINAL_PRECEDENT_UNMATCHED

        # Efficiency: success in 6 or fewer actions (perfect chain without wasted actions)
        if action_count <= 6:
            reward += TERMINAL_EFFICIENCY_BONUS

        # Prosecution dominance penalty is applied in environment using verdict.prosecution_win_rate.
        # This calculator exposes the constant; the env provides the actual verdict-derived condition.
        return float(reward)


def expected_step_order() -> list[StepType]:
    return [
        StepType.ACTUS_REUS,
        StepType.MENS_REA,
        StepType.LINKAGE,
        StepType.COUNTER_ARGUMENT,
        StepType.IPC_APPLICATION,
        StepType.PRECEDENT_CITATION,
    ]

