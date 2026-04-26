from __future__ import annotations

import random
import torch
import torch.nn as nn

from nyayarl.models import Action, EvidenceType, JudgmentLabel, Observation, StepType


class DefenceAgent(nn.Module):
    """
    Differentiable defence policy (PyTorch).

    Contract (Track 3):
    - select_action(observation) -> Action
    - get_log_prob(observation, action) -> torch.Tensor (scalar, requires_grad=True)
    """

    def __init__(
        self,
        *,
        rng_seed: int | None = None,
        hidden_dim: int = 64,
        hash_buckets: int = 2048,
    ) -> None:
        super().__init__()
        self._rng = random.Random(rng_seed)
        # Inference-time control: during eval/demo we want a crisp final verdict.
        # Default: greedy (argmax) for Step 6 judgment when module is in eval mode.
        self._greedy_step6_judgment_in_eval: bool = True

        # Small observation encoder -> logits for discrete choices.
        #
        # We intentionally keep the action distribution compact:
        # - counts: k ∈ {0,1,2,3} for evidence/witness/ipc (masked by availability)
        # - judgment: {acquit, convict, partial} on step 6 only
        #
        # Specific IDs are sampled uniformly from the per-step valid pool.
        # (This keeps the policy differentiable without a huge variable-size ID softmax.)
        self._hidden_dim = int(hidden_dim)
        in_dim = 6 + 8  # step one-hot (6) + numeric features (8)
        self._encoder = nn.Sequential(
            nn.Linear(in_dim, self._hidden_dim),
            nn.Tanh(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.Tanh(),
        )
        self._head_evidence_k = nn.Linear(self._hidden_dim, 4)  # 0..3
        self._head_witness_k = nn.Linear(self._hidden_dim, 4)   # 0..3
        self._head_ipc_k = nn.Linear(self._hidden_dim, 4)       # 0..3
        self._head_judgment = nn.Linear(self._hidden_dim, 3)    # acquit/convict/partial

        # ── ID scorers (masked softmax) ──────────────────────────────────
        # We score variable-length candidate sets by hashing string IDs into buckets.
        # This enables a per-ID distribution without a fixed global vocabulary.
        self._hash_buckets = int(hash_buckets)
        emb_dim = max(16, self._hidden_dim // 4)
        self._id_embed = nn.Embedding(self._hash_buckets, emb_dim)
        self._evidence_type_embed = nn.Embedding(len(EvidenceType), emb_dim)
        self._witness_flag_embed = nn.Embedding(2, emb_dim)  # contradicting flag

        scorer_in = self._hidden_dim + emb_dim
        self._score_ipc = nn.Sequential(nn.Linear(scorer_in, self._hidden_dim), nn.Tanh(), nn.Linear(self._hidden_dim, 1))
        self._score_evidence = nn.Sequential(nn.Linear(scorer_in, self._hidden_dim), nn.Tanh(), nn.Linear(self._hidden_dim, 1))
        self._score_witness = nn.Sequential(nn.Linear(scorer_in, self._hidden_dim), nn.Tanh(), nn.Linear(self._hidden_dim, 1))

    def forward(self, observation: Observation) -> dict[str, torch.Tensor]:
        """
        Produce unnormalised logits for the policy's discrete choices.

        Returns:
          dict with keys: evidence_k_logits, witness_k_logits, ipc_k_logits, judgment_logits
        """
        step_idx = min(len(observation.submitted_steps), 5)
        step_oh = torch.zeros(6, dtype=torch.float32)
        step_oh[step_idx] = 1.0

        cf = observation.case_file

        # Numeric features (all small / bounded-ish):
        # - present evidence count
        # - reliable witness count (>0.5)
        # - applicable ipc count
        # - prosecution challenges so far
        # - chain_score
        # - curriculum_level
        # - submitted_steps_len
        # - is_done flag
        present_e = sum(1 for e in cf.evidence_items if e.is_present)
        reliable_w = sum(1 for w in cf.witness_statements if w.reliability > 0.5)
        ipc_n = len(cf.applicable_ipc_sections)
        ch_n = len(observation.prosecution_challenges)
        submitted_n = len(observation.submitted_steps)
        is_done = 1.0 if observation.is_done else 0.0

        num = torch.tensor(
            [
                float(present_e),
                float(reliable_w),
                float(ipc_n),
                float(ch_n),
                float(observation.chain_score),
                float(observation.curriculum_level),
                float(submitted_n),
                float(is_done),
            ],
            dtype=torch.float32,
        )

        x = torch.cat([step_oh, num], dim=0)
        # Keep everything on the module's device.
        x = x.to(next(self.parameters()).device)
        h = self._encoder(x)
        return {
            "h": h,
            "evidence_k_logits": self._head_evidence_k(h),
            "witness_k_logits": self._head_witness_k(h),
            "ipc_k_logits": self._head_ipc_k(h),
            "judgment_logits": self._head_judgment(h),
        }

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

        logits = self.forward(observation)
        h = logits["h"]

        # ── Sample counts (clamped to available pool size) ───────────────
        evidence_k = self._sample_count(
            logits["evidence_k_logits"], max_k=min(3, len(valid_evidence_ids))
        )
        witness_k = self._sample_count(
            logits["witness_k_logits"], max_k=min(3, len(valid_witness_ids))
        )
        ipc_k = self._sample_count(
            logits["ipc_k_logits"], max_k=min(3, len(valid_ipc))
        )

        # ── Hard procedural floor (deadline-mode safety) ─────────────────
        # Several steps are invalid if k==0 when candidates exist. If the policy
        # collapses to always picking 0 (common early in training), episodes
        # become impossible to complete and eval "successes" collapses to 0.
        if step_type == StepType.ACTUS_REUS and valid_evidence_ids:
            evidence_k = max(1, evidence_k)
        if step_type == StepType.MENS_REA and valid_witness_ids:
            witness_k = max(1, witness_k)
        if step_type == StepType.LINKAGE:
            if valid_evidence_ids:
                evidence_k = max(1, evidence_k)
            if valid_ipc:
                ipc_k = max(1, ipc_k)
        if step_type == StepType.IPC_APPLICATION and valid_ipc:
            ipc_k = max(1, ipc_k)

        # ── Sample IDs from model-scored masked distributions ─────────────
        anchored_evidence_ids = (
            self._sample_ids_evidence(h, case, valid_evidence_ids, k=evidence_k)
            if valid_evidence_ids and evidence_k > 0
            else []
        )
        anchored_witness_ids = (
            self._sample_ids_witness(h, case, valid_witness_ids, k=witness_k)
            if valid_witness_ids and witness_k > 0
            else []
        )
        cited_ipc_sections = (
            self._sample_ids_ipc(h, valid_ipc, k=ipc_k)
            if valid_ipc and ipc_k > 0
            else []
        )

        judgment = None
        if step_type == StepType.PRECEDENT_CITATION:
            greedy = (not self.training) and bool(self._greedy_step6_judgment_in_eval)
            judgment = self._sample_judgment(logits["judgment_logits"], greedy=greedy)

        return Action(
            step_type=step_type,
            cited_ipc_sections=cited_ipc_sections,
            anchored_evidence_ids=anchored_evidence_ids,
            anchored_witness_ids=anchored_witness_ids,
            judgment=judgment,
        )

    def set_greedy_step6_judgment_in_eval(self, enabled: bool) -> None:
        """
        Control whether Step 6 judgment is greedy (argmax) in eval mode.

        Training is never affected: when `self.training` is True we always sample.
        """
        self._greedy_step6_judgment_in_eval = bool(enabled)

    def get_log_prob(self, observation: Observation, action: Action) -> torch.Tensor:
        """
        Differentiable log-prob for the action under current policy parameters.

        Note: we model only the discrete "counts" + judgment choice; the specific IDs are treated
        as environment-conditioned sampling noise and are not part of the trainable distribution.
        """
        logits = self.forward(observation)
        h = logits["h"]
        e_k = len(action.anchored_evidence_ids)
        w_k = len(action.anchored_witness_ids)
        i_k = len(action.cited_ipc_sections)

        logp = torch.tensor(0.0, dtype=torch.float32, device=logits["evidence_k_logits"].device)
        logp = logp + self._logp_count(logits["evidence_k_logits"], e_k, max_k=3)
        logp = logp + self._logp_count(logits["witness_k_logits"], w_k, max_k=3)
        logp = logp + self._logp_count(logits["ipc_k_logits"], i_k, max_k=3)

        # ID-level logprob (prevents uniform-ID sampling from ignoring policy)
        cf = observation.case_file
        step_idx = min(len(observation.submitted_steps), 5)
        step_type = [
            StepType.ACTUS_REUS,
            StepType.MENS_REA,
            StepType.LINKAGE,
            StepType.COUNTER_ARGUMENT,
            StepType.IPC_APPLICATION,
            StepType.PRECEDENT_CITATION,
        ][step_idx]

        if step_type in (StepType.ACTUS_REUS, StepType.LINKAGE, StepType.COUNTER_ARGUMENT, StepType.PRECEDENT_CITATION):
            valid_evidence_ids = self._valid_evidence_ids(cf, step_type)
            logp = logp + self._logp_ids_evidence(h, cf, valid_evidence_ids, action.anchored_evidence_ids)
        if step_type in (StepType.MENS_REA, StepType.COUNTER_ARGUMENT):
            valid_witness_ids = self._valid_witness_ids(cf)
            logp = logp + self._logp_ids_witness(h, cf, valid_witness_ids, action.anchored_witness_ids)
        if step_type in (StepType.LINKAGE, StepType.IPC_APPLICATION):
            valid_ipc = list(cf.applicable_ipc_sections)
            logp = logp + self._logp_ids_ipc(h, valid_ipc, action.cited_ipc_sections)

        if action.step_type == StepType.PRECEDENT_CITATION and action.judgment is not None:
            order = [JudgmentLabel.ACQUIT, JudgmentLabel.CONVICT, JudgmentLabel.PARTIAL]
            idx = order.index(action.judgment) if action.judgment in order else 2
            dist = torch.distributions.Categorical(logits=logits["judgment_logits"])
            logp = logp + dist.log_prob(torch.tensor(idx, dtype=torch.int64, device=logits["judgment_logits"].device))
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
        logits = self.forward(observation)
        h = logits["h"]
        ent = torch.tensor(0.0, dtype=torch.float32, device=logits["evidence_k_logits"].device)
        ent = ent + self._entropy_count(logits["evidence_k_logits"], max_k=3)
        ent = ent + self._entropy_count(logits["witness_k_logits"], max_k=3)
        ent = ent + self._entropy_count(logits["ipc_k_logits"], max_k=3)

        if step_type == StepType.PRECEDENT_CITATION:
            ent = ent + torch.distributions.Categorical(logits=logits["judgment_logits"]).entropy()

        # Add ID-selection entropy where applicable (masked softmax entropies).
        cf = observation.case_file
        if step_type in (StepType.ACTUS_REUS, StepType.LINKAGE, StepType.COUNTER_ARGUMENT):
            vids = self._valid_evidence_ids(cf, step_type)
            ent = ent + self._entropy_ids(self._probs_evidence(h, cf, vids))
        if step_type in (StepType.MENS_REA, StepType.COUNTER_ARGUMENT):
            wids = self._valid_witness_ids(cf)
            ent = ent + self._entropy_ids(self._probs_witness(h, cf, wids))
        if step_type in (StepType.LINKAGE, StepType.IPC_APPLICATION):
            ent = ent + self._entropy_ids(self._probs_ipc(h, list(cf.applicable_ipc_sections)))

        return ent

    # NOTE: We intentionally rely on `nn.Module.state_dict()` / `load_state_dict()`
    # for checkpoint compatibility across training + Gradio.

    # ── helpers ──────────────────────────────────────────────────────────

    def _sample_count(self, logits: torch.Tensor, *, max_k: int) -> int:
        dist = torch.distributions.Categorical(logits=logits[: max_k + 1])
        k = int(dist.sample().item())
        return max(0, min(k, max_k))

    def _logp_count(self, logits: torch.Tensor, k: int, *, max_k: int) -> torch.Tensor:
        k = max(0, min(int(k), max_k))
        dist = torch.distributions.Categorical(logits=logits[: max_k + 1])
        return dist.log_prob(torch.tensor(k, dtype=torch.int64))

    def _sample_judgment(self, logits: torch.Tensor, *, greedy: bool) -> JudgmentLabel:
        order = [JudgmentLabel.ACQUIT, JudgmentLabel.CONVICT, JudgmentLabel.PARTIAL]
        if greedy:
            idx = int(torch.argmax(logits).item())
        else:
            dist = torch.distributions.Categorical(logits=logits)
            idx = int(dist.sample().item())
        return order[max(0, min(idx, 2))]

    def _entropy_count(self, logits: torch.Tensor, *, max_k: int) -> torch.Tensor:
        dist = torch.distributions.Categorical(logits=logits[: max_k + 1])
        return dist.entropy()

    # ── ID scoring helpers ───────────────────────────────────────────────

    def _hash_bucket(self, s: str) -> int:
        # Stable within-process hashing isn't guaranteed across runs, so use a simple
        # deterministic rolling hash.
        h = 2166136261
        for ch in s.encode("utf-8", "ignore"):
            h = (h ^ ch) * 16777619
            h &= 0xFFFFFFFF
        return int(h % self._hash_buckets)

    def _valid_evidence_ids(self, cf, step_type: StepType) -> list[str]:
        if step_type == StepType.ACTUS_REUS:
            return [e.id for e in cf.evidence_items if e.is_present and e.type in (EvidenceType.PHYSICAL, EvidenceType.FORENSIC)]
        if step_type == StepType.LINKAGE:
            return [e.id for e in cf.evidence_items if e.is_present and e.type in (EvidenceType.DOCUMENTARY, EvidenceType.DIGITAL, EvidenceType.FORENSIC)]
        if step_type == StepType.COUNTER_ARGUMENT:
            return [e.id for e in cf.evidence_items if e.is_present]
        if step_type == StepType.PRECEDENT_CITATION:
            return [cf.precedent_id] if cf.precedent_id else []
        return []

    def _valid_witness_ids(self, cf) -> list[str]:
        return [w.id for w in cf.witness_statements if w.reliability > 0.5]

    def _entropy_ids(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.numel() == 0:
            return torch.tensor(0.0, device=probs.device)
        # avoid log(0)
        p = torch.clamp(probs, min=1e-8)
        return -(p * torch.log(p)).sum()

    def _probs_ipc(self, h: torch.Tensor, ipc: list[str]) -> torch.Tensor:
        if not ipc:
            return torch.zeros(0, device=h.device)
        ids = torch.tensor([self._hash_bucket(s) for s in ipc], dtype=torch.long, device=h.device)
        emb = self._id_embed(ids)
        x = torch.cat([h.expand(len(ipc), -1), emb], dim=1)
        scores = self._score_ipc(x).squeeze(-1)
        return torch.softmax(scores, dim=0)

    def _probs_evidence(self, h: torch.Tensor, cf, evidence_ids: list[str]) -> torch.Tensor:
        if not evidence_ids:
            return torch.zeros(0, device=h.device)
        ev_map = {e.id: e for e in cf.evidence_items}
        ids = torch.tensor([self._hash_bucket(s) for s in evidence_ids], dtype=torch.long, device=h.device)
        emb_id = self._id_embed(ids)
        # Type embedding: unknown types map to 0 via fallback
        et_idx = []
        for eid in evidence_ids:
            et = ev_map.get(eid).type if eid in ev_map else EvidenceType.DOCUMENTARY
            et_idx.append(list(EvidenceType).index(et))
        emb_t = self._evidence_type_embed(torch.tensor(et_idx, dtype=torch.long, device=h.device))
        emb = emb_id + emb_t
        x = torch.cat([h.expand(len(evidence_ids), -1), emb], dim=1)
        scores = self._score_evidence(x).squeeze(-1)
        return torch.softmax(scores, dim=0)

    def _probs_witness(self, h: torch.Tensor, cf, witness_ids: list[str]) -> torch.Tensor:
        if not witness_ids:
            return torch.zeros(0, device=h.device)
        w_map = {w.id: w for w in cf.witness_statements}
        ids = torch.tensor([self._hash_bucket(s) for s in witness_ids], dtype=torch.long, device=h.device)
        emb_id = self._id_embed(ids)
        flags = []
        for wid in witness_ids:
            flags.append(1 if (wid in w_map and w_map[wid].is_contradicting) else 0)
        emb_f = self._witness_flag_embed(torch.tensor(flags, dtype=torch.long, device=h.device))
        emb = emb_id + emb_f
        x = torch.cat([h.expand(len(witness_ids), -1), emb], dim=1)
        scores = self._score_witness(x).squeeze(-1)
        return torch.softmax(scores, dim=0)

    def _sample_without_replacement(self, probs: torch.Tensor, k: int) -> list[int]:
        if k <= 0 or probs.numel() == 0:
            return []
        k = min(int(k), int(probs.numel()))
        # torch.multinomial handles non-replacement sampling from probs
        idx = torch.multinomial(probs, num_samples=k, replacement=False)
        return [int(i) for i in idx.tolist()]

    def _sample_ids_ipc(self, h: torch.Tensor, ipc: list[str], *, k: int) -> list[str]:
        probs = self._probs_ipc(h, ipc)
        idxs = self._sample_without_replacement(probs, k)
        return [ipc[i] for i in idxs]

    def _sample_ids_evidence(self, h: torch.Tensor, cf, evidence_ids: list[str], *, k: int) -> list[str]:
        probs = self._probs_evidence(h, cf, evidence_ids)
        idxs = self._sample_without_replacement(probs, k)
        return [evidence_ids[i] for i in idxs]

    def _sample_ids_witness(self, h: torch.Tensor, cf, witness_ids: list[str], *, k: int) -> list[str]:
        probs = self._probs_witness(h, cf, witness_ids)
        idxs = self._sample_without_replacement(probs, k)
        return [witness_ids[i] for i in idxs]

    def _logp_ids_from_probs(self, probs: torch.Tensor, chosen_indices: list[int]) -> torch.Tensor:
        if not chosen_indices:
            return torch.tensor(0.0, device=probs.device)
        # Approx log-prob for without-replacement sampling:
        # sum_t log( p(i_t) / sum_{j not yet chosen} p(j) )
        p = torch.clamp(probs, min=1e-12)
        logp = torch.tensor(0.0, device=probs.device)
        chosen_set: set[int] = set()
        for idx in chosen_indices:
            # Avoid in-place mutation of a boolean mask that participates in autograd.
            # Build a fresh mask each iteration.
            remaining = torch.ones_like(p, dtype=torch.bool)
            if chosen_set:
                remaining[list(chosen_set)] = False
            denom = torch.clamp(p[remaining].sum(), min=1e-12)
            logp = logp + torch.log(p[int(idx)] / denom)
            chosen_set.add(int(idx))
        return logp

    def _logp_ids_ipc(self, h: torch.Tensor, ipc: list[str], chosen: list[str]) -> torch.Tensor:
        if not chosen:
            return torch.tensor(0.0, device=h.device)
        probs = self._probs_ipc(h, ipc)
        idx_map = {s: i for i, s in enumerate(ipc)}
        idxs = [idx_map[s] for s in chosen if s in idx_map]
        return self._logp_ids_from_probs(probs, idxs)

    def _logp_ids_evidence(self, h: torch.Tensor, cf, evidence_ids: list[str], chosen: list[str]) -> torch.Tensor:
        if not chosen:
            return torch.tensor(0.0, device=h.device)
        probs = self._probs_evidence(h, cf, evidence_ids)
        idx_map = {s: i for i, s in enumerate(evidence_ids)}
        idxs = [idx_map[s] for s in chosen if s in idx_map]
        return self._logp_ids_from_probs(probs, idxs)

    def _logp_ids_witness(self, h: torch.Tensor, cf, witness_ids: list[str], chosen: list[str]) -> torch.Tensor:
        if not chosen:
            return torch.tensor(0.0, device=h.device)
        probs = self._probs_witness(h, cf, witness_ids)
        idx_map = {s: i for i, s in enumerate(witness_ids)}
        idxs = [idx_map[s] for s in chosen if s in idx_map]
        return self._logp_ids_from_probs(probs, idxs)

