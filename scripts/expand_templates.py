"""
expand_templates.py — generate more case templates by cloning existing ones.

This is a stopgap to get from ~5 templates to >=20 for demo/training,
without hand-authoring JSON.

It produces *synthetic* variants by:
- copying a base template
- changing template_id + description + notes
- prefixing evidence/witness IDs to keep them unique per template
- slightly perturbing presence_prob and witness reliability priors
- rotating the precedents list deterministically

Usage:
  python scripts/expand_templates.py --src data/case_templates --dst data/case_templates --target 20
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _load_templates(src: Path) -> list[tuple[Path, dict[str, Any]]]:
    out: list[tuple[Path, dict[str, Any]]] = []
    for p in sorted(src.glob("*.json")):
        out.append((p, json.loads(p.read_text())))
    return out


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _make_variant(base: dict[str, Any], *, variant_idx: int, rng: random.Random) -> dict[str, Any]:
    t = json.loads(json.dumps(base))  # deep copy
    base_id = str(t.get("template_id", "template")).strip() or "template"
    new_id = f"{base_id}_v{variant_idx:02d}"
    t["template_id"] = new_id
    t["description"] = f"{t.get('description','').strip()} (variant {variant_idx})".strip()
    t["notes"] = f"{t.get('notes','').strip()} | auto-variant {variant_idx}".strip()

    # Ensure unique evidence IDs and perturb presence_prob a bit.
    for ev in t.get("evidence_pool", []) or []:
        ev_id = str(ev.get("id", "ev")).strip() or "ev"
        ev["id"] = f"{new_id}:{ev_id}"
        if "presence_prob" in ev:
            ev["presence_prob"] = _clamp(float(ev["presence_prob"]) + rng.uniform(-0.15, 0.15), 0.05, 0.98)

    # Ensure unique witness IDs and perturb beta priors a bit.
    for w in t.get("witness_pool", []) or []:
        w_id = str(w.get("id", "w")).strip() or "w"
        w["id"] = f"{new_id}:{w_id}"
        if "reliability_alpha" in w:
            w["reliability_alpha"] = int(_clamp(float(w["reliability_alpha"]) + rng.uniform(-2, 2), 2, 10))
        if "reliability_beta" in w:
            w["reliability_beta"] = int(_clamp(float(w["reliability_beta"]) + rng.uniform(-2, 2), 2, 10))

    # Rotate precedents deterministically.
    prec = list(t.get("precedents", []) or [])
    if prec:
        shift = variant_idx % len(prec)
        t["precedents"] = prec[shift:] + prec[:shift]

    return t


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=str, default="data/case_templates")
    p.add_argument("--dst", type=str, default="data/case_templates")
    p.add_argument("--target", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    existing = _load_templates(dst)
    existing_ids = {str(t.get("template_id", "")) for _, t in existing}

    base_templates = _load_templates(src)
    if not base_templates:
        raise SystemExit(f"No base templates found in {src}")

    rng = random.Random(int(args.seed))
    want = int(args.target)

    i = 1
    while len(existing_ids) < want:
        base_path, base = rng.choice(base_templates)
        variant = _make_variant(base, variant_idx=i, rng=rng)
        tid = variant["template_id"]
        if tid in existing_ids:
            i += 1
            continue

        out_path = dst / f"{tid}.json"
        out_path.write_text(json.dumps(variant, indent=2, sort_keys=False))
        existing_ids.add(tid)
        i += 1

    print(f"Templates now: {len(existing_ids)} in {dst}")


if __name__ == "__main__":
    main()

