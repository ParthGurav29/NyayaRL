from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _load(p: Path) -> dict[str, Any]:
    with open(p, "r") as f:
        return json.load(f)


def main() -> None:
    train_dir = Path("data/case_templates")
    held_dir = Path("data/case_templates_heldout")

    train_files = sorted(train_dir.glob("*.json"))
    held_files = sorted(held_dir.glob("*.json"))

    if not held_files:
        raise SystemExit(f"Heldout templates dir is empty: {held_dir} (need *.json)")

    train_ids = set()
    for p in train_files:
        t = _load(p)
        tid = str(t.get("template_id", "")).strip()
        if tid:
            train_ids.add(tid)

    held_ids = set()
    for p in held_files:
        t = _load(p)
        tid = str(t.get("template_id", "")).strip()
        if tid:
            held_ids.add(tid)

    overlap = sorted(train_ids & held_ids)
    if overlap:
        raise SystemExit(
            "Template overlap detected between training and heldout. "
            f"overlap_count={len(overlap)} examples={overlap[:10]}"
        )

    print(
        f"OK: no template_id overlap. train={len(train_ids)} heldout={len(held_ids)}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

