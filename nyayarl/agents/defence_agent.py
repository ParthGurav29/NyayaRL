from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn as nn

from nyayarl.models import Action, EvidenceType, JudgmentLabel, Observation, StepType


class DefenceAgent(nn.Module):
    """
    Differentiable defence policy with trainable logits (PyTorch).

    Contract (Track 3):
    - select_action(observation) -> Action
    - get_log_prob(observation, action) -> torch.Tensor (scalar, requires_grad=True)
    """

    def __init__(self, *, rng_seed: int | None = None) -> None:
        super().__init__()
        self._rng = random.Random(rng_seed)

        # 0..3 for counts, 3 judgments
        self.evidence_count_logits = nn.Parameter(torch.tensor([0.0, 0.2, 0.0, -0.2], dtype=torch.float32))
        self.witness_count_logits = nn.Parameter(torch.tensor([0.0, 0.2, 0.0, -0.2], dtype=torch.float32))
        self.ipc_count_logits = nn.Parameter(torch.tensor([0.0, 0.3, 0.1, -0.3], dtype=torch.float32))
        self.judgment_logits = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32))

    def select_action(self, observation: Observation) -> Action:
        step_idx = len(observation.submitted_steps)
        step_type = [
            StepType.ACTUS_REUS,
            StepType.MENS_REA,
            StepType.LINKAGE,
            StepType.COUNTER_ARGUMENT,
            StepType.IPC_APPLICATION,
            StepType.PRECEDENT_CITATION,
        ][min(step_idx, 5)]

        case = observation.case_file

        # ── Step-aware action masking ────────────────────────────────────
        # Build per-step valid ID pools from the case file.
        # The mask filters BEFORE sampling — invalid actions are impossible.

        valid_evidence_ids: list[str] = []
        valid_witness_ids: list[str] = []
        valid_ipc: list[str] = []

        if step_type == StepType.ACTUS_REUS:
            # Step 1: only PHYSICAL and FORENSIC evidence that is present
            valid_evidence_ids = [
                e.id for e in case.evidence_items
                if e.is_present and e.type in (EvidenceType.PHYSICAL, EvidenceType.FORENSIC)
            ]
        elif step_type == StepType.MENS_REA:
            # Step 2: only witnesses with reliability > 0.5
            valid_witness_ids = [
                w.id for w in case.witness_statements
                if w.reliability > 0.5
            ]
        elif step_type == StepType.LINKAGE:
            # Step 3: only DOCUMENTARY / DIGITAL and FORENSIC evidence that is present
            valid_evidence_ids = [
                e.id for e in case.evidence_items
                if e.is_present and e.type in (EvidenceType.DOCUMENTARY, EvidenceType.DIGITAL, EvidenceType.FORENSIC)
            ]
            valid_ipc = list(case.applicable_ipc_sections)
        elif step_type == StepType.COUNTER_ARGUMENT:
            # Step 4: all present evidence + reliable witnesses
            valid_evidence_ids = [e.id for e in case.evidence_items if e.is_present]
            valid_witness_ids = [w.id for w in case.witness_statements if w.reliability > 0.5]
        elif step_type == StepType.IPC_APPLICATION:
            # Step 5: only applicable IPC sections
            valid_ipc = list(case.applicable_ipc_sections)
        elif step_type == StepType.PRECEDENT_CITATION:
            # Step 6: precedent ID goes in evidence slot
            valid_evidence_ids = [case.precedent_id] if case.precedent_id else []

        # ── Sample counts (clamped to available pool size) ───────────────
        evidence_k = self._sample_count(self.evidence_count_logits, max_k=min(3, len(valid_evidence_ids)))
        witness_k = self._sample_count(self.witness_count_logits, max_k=min(3, len(valid_witness_ids)))
        ipc_k = self._sample_count(self.ipc_count_logits, max_k=min(3, len(valid_ipc)))

        anchored_evidence_ids = self._rng.sample(valid_evidence_ids, k=evidence_k) if valid_evidence_ids and evidence_k > 0 else []
        anchored_witness_ids = self._rng.sample(valid_witness_ids, k=witness_k) if valid_witness_ids and witness_k > 0 else []
        cited_ipc_sections = self._rng.sample(valid_ipc, k=ipc_k) if valid_ipc and ipc_k > 0 else []

        judgment = None
        if step_type == StepType.PRECEDENT_CITATION:
            judgment = self._sample_judgment()

        return Action(
            step_type=step_type,
            cited_ipc_sections=cited_ipc_sections,
            anchored_evidence_ids=anchored_evidence_ids,
            anchored_witness_ids=anchored_witness_ids,
            judgment=judgment,
        )

    def get_log_prob(self, observation: Observation, action: Action) -> torch.Tensor:
        """
        Differentiable log-prob for the action under current policy parameters.

        Note: we model only the discrete "counts" + judgment choice; the specific IDs are treated
        as environment-conditioned sampling noise and are not part of the trainable distribution.
        """
        e_k = len(action.anchored_evidence_ids)
        w_k = len(action.anchored_witness_ids)
        i_k = len(action.cited_ipc_sections)

        logp = torch.tensor(0.0, dtype=torch.float32)
        logp = logp + self._logp_count(self.evidence_count_logits, e_k, max_k=3)
        logp = logp + self._logp_count(self.witness_count_logits, w_k, max_k=3)
        logp = logp + self._logp_count(self.ipc_count_logits, i_k, max_k=3)

        if action.step_type == StepType.PRECEDENT_CITATION and action.judgment is not None:
            order = [JudgmentLabel.ACQUIT, JudgmentLabel.CONVICT, JudgmentLabel.PARTIAL]
            idx = order.index(action.judgment) if action.judgment in order else 2
            dist = torch.distributions.Categorical(logits=self.judgment_logits)
            logp = logp + dist.log_prob(torch.tensor(idx, dtype=torch.int64))
        return logp

    def get_entropy(self, observation: Observation) -> torch.Tensor:
        """
        Differentiable entropy of the trainable parts of the policy.

        We model only the discrete count distributions + judgment distribution.
        """
        step_idx = len(observation.submitted_steps)
        step_type = [
            StepType.ACTUS_REUS,
            StepType.MENS_REA,
            StepType.LINKAGE,
            StepType.COUNTER_ARGUMENT,
            StepType.IPC_APPLICATION,
            StepType.PRECEDENT_CITATION,
        ][min(step_idx, 5)]

        # Count distributions are always present (but may be masked by max_k=0).
        ent = torch.tensor(0.0, dtype=torch.float32)
        ent = ent + self._entropy_count(self.evidence_count_logits, max_k=3)
        ent = ent + self._entropy_count(self.witness_count_logits, max_k=3)
        ent = ent + self._entropy_count(self.ipc_count_logits, max_k=3)

        if step_type == StepType.PRECEDENT_CITATION:
            ent = ent + torch.distributions.Categorical(logits=self.judgment_logits).entropy()

        return ent

    def state_dict(self) -> dict[str, Any]:
        return {
            "evidence_count_logits": self.evidence_count_logits.detach().cpu().tolist(),
            "witness_count_logits": self.witness_count_logits.detach().cpu().tolist(),
            "ipc_count_logits": self.ipc_count_logits.detach().cpu().tolist(),
            "judgment_logits": self.judgment_logits.detach().cpu().tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "DefenceAgent":
        agent = cls()
        with torch.no_grad():
            if "evidence_count_logits" in state:
                agent.evidence_count_logits.copy_(torch.tensor(state["evidence_count_logits"], dtype=torch.float32))
            if "witness_count_logits" in state:
                agent.witness_count_logits.copy_(torch.tensor(state["witness_count_logits"], dtype=torch.float32))
            if "ipc_count_logits" in state:
                agent.ipc_count_logits.copy_(torch.tensor(state["ipc_count_logits"], dtype=torch.float32))
            if "judgment_logits" in state:
                agent.judgment_logits.copy_(torch.tensor(state["judgment_logits"], dtype=torch.float32))
        return agent

    # ── helpers ──────────────────────────────────────────────────────────

    def _sample_count(self, logits: torch.Tensor, *, max_k: int) -> int:
        dist = torch.distributions.Categorical(logits=logits[: max_k + 1])
        k = int(dist.sample().item())
        return max(0, min(k, max_k))

    def _logp_count(self, logits: torch.Tensor, k: int, *, max_k: int) -> torch.Tensor:
        k = max(0, min(int(k), max_k))
        dist = torch.distributions.Categorical(logits=logits[: max_k + 1])
        return dist.log_prob(torch.tensor(k, dtype=torch.int64))

    def _sample_judgment(self) -> JudgmentLabel:
        order = [JudgmentLabel.ACQUIT, JudgmentLabel.CONVICT, JudgmentLabel.PARTIAL]
        dist = torch.distributions.Categorical(logits=self.judgment_logits)
        idx = int(dist.sample().item())
        return order[max(0, min(idx, 2))]

    def _entropy_count(self, logits: torch.Tensor, *, max_k: int) -> torch.Tensor:
        dist = torch.distributions.Categorical(logits=logits[: max_k + 1])
        return dist.entropy()

