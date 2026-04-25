


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
    if v in ("documentary", "digital"):
        # Track 2 uses "digital" in some templates; Track 1 treats these as documentary.
        return EvidenceType.DOCUMENTARY
    if v in ("testimonial",):
        return EvidenceType.TESTIMONIAL
    return EvidenceType.DOCUMENTARY


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
                templates.append(json.load(f))
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
        if isinstance(precedents, list) and precedents:
            return str(self._rng.choice(precedents))
        return ""

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
        reliability_bias = float(self._difficulty_params.get("witness_reliability_bias", 0.0))
        contradiction_prob = float(self._difficulty_params.get("contradiction_prob", 0.1))
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

