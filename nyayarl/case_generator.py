from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

from nyayarl.models import CaseFile, EvidenceItem, EvidenceType, WitnessStatement


def _evidence_type_from_template(value: str) -> EvidenceType:
    """
    Map Track 2 template evidence types onto Track 1's EvidenceType enum.
    """
    v = (value or "").lower()
    if v in ("physical",):
        return EvidenceType.PHYSICAL
    if v in ("forensic",):
        return EvidenceType.FORENSIC
    if v in ("documentary",):
        return EvidenceType.DOCUMENTARY
    if v in ("digital",):
        return EvidenceType.DIGITAL
    if v in ("testimonial",):
        return EvidenceType.TESTIMONIAL
    return EvidenceType.DOCUMENTARY


def validate_template_schema(template: dict[str, Any]) -> None:
    """
    Best-effort schema validation for Track 2 case templates.

    This runs at startup to fail fast on data issues that would otherwise silently
    corrupt training rewards.
    """
    template_id = str(template.get("template_id", "")).strip() or "<unknown>"
    evidence_pool = template.get("evidence_pool") or []
    if not isinstance(evidence_pool, list):
        raise ValueError(f"Template {template_id}: evidence_pool must be a list")

    for ev in evidence_pool:
        if not isinstance(ev, dict):
            raise ValueError(f"Template {template_id}: evidence_pool entries must be objects")
        if "id" not in ev:
            raise ValueError(f"Template {template_id}: evidence item missing 'id'")
        et = str(ev.get("type", "")).lower().strip()
        _ = _evidence_type_from_template(et)  # raises never; validates mapping exists

    ipc_sections = template.get("ipc_sections", [])
    if not isinstance(ipc_sections, list):
        raise ValueError(f"Template {template_id}: ipc_sections must be a list")

    precedents = template.get("precedents", [])
    if precedents is not None and not isinstance(precedents, list):
        raise ValueError(f"Template {template_id}: precedents must be a list")


class Track2CaseGenerator:
    """
    Deterministic-ish case generator backed by Track 2 JSON templates.

    It produces Track 1 `CaseFile` objects that are consumed by `NyayaRLEnvironment`.
    Difficulty is controlled via `difficulty_params`, which comes from the curriculum.
    """

    def __init__(
        self,
        case_templates_dir: str | Path | None = None,
        *,
        rng_seed: int | None = None,
    ) -> None:
        self._templates_dir = (
            Path(case_templates_dir)
            if case_templates_dir is not None
            else Path(__file__).parent.parent / "data" / "case_templates"
        )
        self._rng = random.Random(rng_seed)
        self._templates: list[dict[str, Any]] = self._load_templates()
        self._difficulty_params: dict[str, Any] = {}

    def set_difficulty_params(self, params: dict[str, Any]) -> None:
        self._difficulty_params = dict(params or {})

    def generate(self, curriculum_level: int) -> CaseFile:
        if not self._templates:
            raise RuntimeError(f"No case templates found in {self._templates_dir}")

        template = self._choose_template_for_level(curriculum_level)

        evidence_items = self._sample_evidence(template)
        witnesses = self._sample_witnesses(template)

        ipc_sections = [str(x) for x in template.get("ipc_sections", [])]
        template_id = template.get("template_id", "")

        accused_count = self._sample_accused_count(curriculum_level)

        fir = (
            f"FIR (template {template_id}): {template.get('description', '').strip()}\n\n"
            f"Notes: {template.get('notes', '').strip()}"
        ).strip()

        precedent_id = self._choose_precedent_id(template)

        return CaseFile(
            case_id=f"{template_id}_{self._rng.randint(1000, 9999)}",
            template_id=template_id,
            fir=fir,
            accused_count=accused_count,
            evidence_items=evidence_items,
            witness_statements=witnesses,
            applicable_ipc_sections=ipc_sections,
            curriculum_level=curriculum_level,
            precedent_id=precedent_id,
        )

    # ── internals ──────────────────────────────────────────────────────────

    def _load_templates(self) -> list[dict[str, Any]]:
        if not self._templates_dir.exists():
            return []
        templates: list[dict[str, Any]] = []
        for p in sorted(self._templates_dir.glob("*.json")):
            with open(p, "r") as f:
                t = json.load(f)
                validate_template_schema(t)
                templates.append(t)
        return templates

    def _choose_template_for_level(self, curriculum_level: int) -> dict[str, Any]:
        # Prefer templates whose advertised range includes the requested level.
        eligible: list[dict[str, Any]] = []
        for t in self._templates:
            lo, hi = (t.get("difficulty_range") or [1, 4])[:2]
            if int(lo) <= curriculum_level <= int(hi):
                eligible.append(t)
        if not eligible:
            eligible = self._templates
        return self._rng.choice(eligible)

    def _sample_accused_count(self, curriculum_level: int) -> int:
        rng = self._difficulty_params.get("accused_count_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            lo, hi = int(rng[0]), int(rng[1])
            lo = max(1, lo)
            hi = max(lo, hi)
            return self._rng.randint(lo, hi)
        # Simple default scaling by curriculum.
        if curriculum_level <= 2:
            return 1
        return self._rng.choice([1, 2])

    def _choose_precedent_id(self, template: dict[str, Any]) -> str:
        precedents = template.get("precedents") or []
        if not precedents:
            return ""

        # Deterministic selection based on template_id to ensure
        # the same template always maps to the same precedent
        template_id = template.get("template_id", "")
        if not template_id:
            # Fallback to first precedent if no template_id
            return str(precedents[0])

        # Use hash of template_id to deterministically select a precedent
        # This ensures the same template always gets the same precedent
        hash_value = hash(template_id)
        index = abs(hash_value) % len(precedents)
        return str(precedents[index])

    def _sample_evidence(self, template: dict[str, Any]) -> list[EvidenceItem]:
        pool = template.get("evidence_pool") or []
        bias = float(self._difficulty_params.get("evidence_presence_bias", 0.0))
        items: list[EvidenceItem] = []
        for ev in pool:
            base_p = float(ev.get("presence_prob", 0.7))
            p = min(0.98, max(0.02, base_p + bias))
            present = self._rng.random() < p
            items.append(
                EvidenceItem(
                    id=str(ev.get("id", "")),
                    description=str(ev.get("description", "")),
                    type=_evidence_type_from_template(str(ev.get("type", ""))),
                    is_present=present,
                )
            )
        return items

    def _sample_witnesses(self, template: dict[str, Any]) -> list[WitnessStatement]:
        pool = template.get("witness_pool") or []
        reliability_bias = float(
            self._difficulty_params.get("witness_reliability_bias", 0.0)
        )
        contradiction_prob = float(
            self._difficulty_params.get("contradiction_prob", 0.1)
        )
        witnesses: list[WitnessStatement] = []
        for w in pool:
            a = float(w.get("reliability_alpha", 6))
            b = float(w.get("reliability_beta", 4))
            rel = self._rng.betavariate(a, b)
            rel = min(1.0, max(0.0, rel + reliability_bias))
            contradict = self._rng.random() < contradiction_prob
            witnesses.append(
                WitnessStatement(
                    id=str(w.get("id", "")),
                    content=str(w.get("description", "")),
                    reliability=rel,
                    is_contradicting=contradict,
                )
            )
        return witnesses
