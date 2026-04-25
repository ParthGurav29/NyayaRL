from __future__ import annotations

import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


def _load_json(p: Path) -> dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)


def _write_json(p: Path, obj: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _mutate_template(t: dict[str, Any], *, rng: random.Random, suffix: str) -> dict[str, Any]:
    """
    Produce a held-out variant that is structurally similar but different in:
    - evidence presence probabilities
    - witness reliability distributions / contradicting likelihood
    - IPC section list
    """
    out = deepcopy(t)

    base_id = str(out.get("template_id", "")).strip() or "unknown"
    out["template_id"] = f"{base_id}_heldout_{suffix}"

    # Evidence: perturb presence probabilities and shuffle pool order
    ev_pool = list(out.get("evidence_pool") or [])
    rng.shuffle(ev_pool)
    for ev in ev_pool:
        if not isinstance(ev, dict):
            continue
        base_p = float(ev.get("presence_prob", 0.7))
        # +/- up to 0.25, keep within [0.05, 0.95]
        delta = rng.uniform(-0.25, 0.25)
        ev["presence_prob"] = max(0.05, min(0.95, base_p + delta))
    out["evidence_pool"] = ev_pool

    # Witnesses: shift reliability alpha/beta and adjust contradiction propensity via "notes" if present.
    w_pool = list(out.get("witness_pool") or [])
    rng.shuffle(w_pool)
    for w in w_pool:
        if not isinstance(w, dict):
            continue
        a = float(w.get("reliability_alpha", 6))
        b = float(w.get("reliability_beta", 4))
        # increase variance and shift mean slightly by tweaking alpha/beta
        a = max(1.0, a + rng.uniform(-2.0, 2.0))
        b = max(1.0, b + rng.uniform(-2.0, 2.0))
        w["reliability_alpha"] = a
        w["reliability_beta"] = b
        # optionally add per-witness "contradiction_hint" for future use (ignored by current generator)
        if rng.random() < 0.4:
            w["contradiction_hint"] = rng.choice(["low", "medium", "high"])
    out["witness_pool"] = w_pool

    # IPC sections: keep a non-empty subset and optionally add one plausible extra from the original set.
    ipc = [str(x) for x in (out.get("ipc_sections") or [])]
    if ipc:
        # choose 1..len(ipc) subset
        k = rng.randint(1, max(1, len(ipc)))
        rng.shuffle(ipc)
        ipc = ipc[:k]
    out["ipc_sections"] = ipc

    # Notes: mark explicitly heldout variant
    notes = str(out.get("notes", "") or "")
    out["notes"] = (notes + "\n\n[HELDOUT_VARIANT] generated for eval harness").strip()
    return out


def main() -> None:
    train_dir = Path("data/case_templates")
    held_dir = Path("data/case_templates_heldout")

    pts = sorted(train_dir.glob("*.json"))
    if len(pts) < 20:
        raise SystemExit(f"Expected 20 training templates in {train_dir}, found {len(pts)}")

    # Use exactly the 20 training templates present.
    train_templates = [_load_json(p) for p in pts]

    rng = random.Random(20260426)

    # Create 10 heldout variants drawn from distinct base templates.
    chosen = rng.sample(list(range(len(train_templates))), k=10)

    # Clear existing heldout jsons (keep .gitkeep if present)
    for existing in held_dir.glob("*.json"):
        existing.unlink()

    for i, idx in enumerate(chosen, start=1):
        base = train_templates[idx]
        suffix = f"v{i:02d}"
        variant = _mutate_template(base, rng=rng, suffix=suffix)
        name = f"{variant['template_id']}.json"
        _write_json(held_dir / name, variant)

    print(f"Wrote 10 heldout templates to {held_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

