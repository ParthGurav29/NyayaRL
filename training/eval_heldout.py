"""
eval_heldout.py — Held-out evaluation harness for NyayaRL.

Loads the latest checkpoint from a directory, then evaluates judgment alignment
on a deterministic, *held-out* set of case templates (i.e. templates excluded
from training).

This script intentionally reports ONE headline number:
  - judgment alignment (% precedent_matched on successful episodes)
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
from pathlib import Path

import torch

from nyayarl.checkpointing import load_defence_agent_checkpoint
from nyayarl.agents import DefenceAgent, JudgeAgent, ProsecutionAgent
from nyayarl.case_generator import Track2CaseGenerator
from nyayarl.environment import NyayaRLEnvironment
from nyayarl.precedents import PrecedentsDB
from nyayarl.track2_adapters import Track2JudgeAdapter, Track2ProsecutionAdapter
from scripts.check_template_overlap import main as _check_template_overlap_main


def _latest_checkpoint(checkpoint_dir: Path) -> Path:
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
    latest = checkpoint_dir / "latest.pt"
    if latest.exists():
        return latest
    pts = sorted(checkpoint_dir.glob("step_*.pt"))
    if not pts:
        raise FileNotFoundError(f"No step_*.pt checkpoints found in {checkpoint_dir}")
    return pts[-1]


@torch.no_grad()
def eval_alignment(
    *,
    checkpoint_path: Path,
    templates_dir: Path,
    episodes: int,
    seed: int,
) -> float:
    # Hard fail if heldout directory is empty or overlaps with training.
    _check_template_overlap_main()

    # Determinism: seed Python + Torch for identical runs.
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # NOTE on determinism:
    # Track2CaseGenerator uses an internal RNG that advances across episodes.
    # If any code change adds/removes RNG draws inside `generate()` (e.g., solvability fixes),
    # the *sequence of templates sampled across episodes* can shift even with the same seed.
    #
    # To make eval results comparable across code changes, we re-seed the generator per episode
    # using the episode seed (epi_seed) rather than relying on a stateful RNG across episodes.
    precedents = PrecedentsDB()
    judge = Track2JudgeAdapter(JudgeAgent(), precedents)

    agent = DefenceAgent(rng_seed=seed)
    # Evaluation must never silently run with random weights.
    # If the checkpoint does not fully match the current architecture, fail loudly.
    load_defence_agent_checkpoint(
        checkpoint_path=checkpoint_path,
        agent=agent,
        map_location="cpu",
        require_full_match=True,
        allow_legacy_partial=False,
        do_value_check=True,
    )
    agent.eval()

    successes = 0
    matched = 0
    total = int(episodes)

    # Evaluate across curriculum levels to avoid cherry-picking.
    levels = [1, 2, 3, 4]
    for i in range(episodes):
        # Re-seed per episode so the entire trajectory (sampling + env) is reproducible.
        epi_seed = seed + int(i)
        random.seed(epi_seed)
        torch.manual_seed(epi_seed)
        level = levels[i % len(levels)]
        case_gen = Track2CaseGenerator(case_templates_dir=templates_dir, rng_seed=epi_seed)
        env = NyayaRLEnvironment(case_generator=case_gen, judge=judge, prosecution=None)
        obs = env.reset(level)
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, sr, done = env.step(action)
        if len(obs.submitted_steps) == 6:
            successes += 1
            verdict = env.get_last_verdict()
            if verdict is not None and verdict.precedent_matched:
                matched += 1

    if successes == 0:
        return 0.0
    return matched / successes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None, help="Path to a specific checkpoint .pt file")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory containing step_*.pt (used if --checkpoint is not provided)")
    p.add_argument(
        "--heldout-templates-dir",
        type=str,
        default="data/case_templates_heldout",
        help="Directory of held-out templates excluded from training",
    )
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(Path(args.checkpoint_dir))
    alignment = eval_alignment(
        checkpoint_path=ckpt,
        templates_dir=Path(args.heldout_templates_dir),
        episodes=int(args.episodes),
        seed=int(args.seed),
    )
    def _category_key(template_id: str) -> str:
        # Normalize template_id into the 5 headline buckets.
        # Examples:
        # - 302_304_homicide_v01_heldout_v06 -> 302_304
        # - 498A_304B_domestic_heldout_v02  -> 498A_304B
        parts = (template_id or "").split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return template_id or "unknown"

    # Re-run eval to expose denominators + entropy + per-category breakdown (deterministic).
    seed = int(args.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    precedents = PrecedentsDB()
    judge = Track2JudgeAdapter(JudgeAgent(), precedents)
    agent = DefenceAgent(rng_seed=seed)
    load_defence_agent_checkpoint(
        checkpoint_path=ckpt,
        agent=agent,
        map_location="cpu",
        require_full_match=True,
        allow_legacy_partial=False,
        do_value_check=True,
    )
    agent.eval()
    successes = 0
    matched = 0
    total = int(args.episodes)
    entropies: list[float] = []
    per_cat: dict[str, dict[str, int]] = {}
    levels = [1, 2, 3, 4]
    for i in range(total):
        epi_seed = seed + int(i)
        random.seed(epi_seed)
        torch.manual_seed(epi_seed)
        level = levels[i % len(levels)]
        case_gen = Track2CaseGenerator(case_templates_dir=Path(args.heldout_templates_dir), rng_seed=epi_seed)
        env = NyayaRLEnvironment(case_generator=case_gen, judge=judge, prosecution=None)
        obs = env.reset(level)
        cat = _category_key(obs.case_file.template_id)
        per_cat.setdefault(cat, {"episodes": 0, "successes": 0, "matched": 0})
        per_cat[cat]["episodes"] += 1
        done = False
        while not done:
            # Entropy of the trainable policy (counts + IDs where applicable).
            try:
                entropies.append(float(agent.get_entropy(obs).item()))
            except Exception:
                pass
            action = agent.select_action(obs)
            obs, sr, done = env.step(action)
        if len(obs.submitted_steps) == 6:
            successes += 1
            per_cat[cat]["successes"] += 1
            verdict = env.get_last_verdict()
            if verdict is not None and verdict.precedent_matched:
                matched += 1
                per_cat[cat]["matched"] += 1
    ratio = (matched / successes) if successes else 0.0
    mean_entropy = (sum(entropies) / len(entropies)) if entropies else 0.0
    print(f"episodes={total}  successes={successes}  matched={matched}  alignment={ratio:.2%}")
    print(f"mean_policy_entropy={mean_entropy:.4f}")
    for cat in sorted(per_cat.keys()):
        e = per_cat[cat]["episodes"]
        s = per_cat[cat]["successes"]
        m = per_cat[cat]["matched"]
        a = (m / s) if s else 0.0
        print(f"category={cat}  episodes={e}  successes={s}  matched={m}  alignment={a:.2%}")
    print(f"heldout_alignment={alignment * 100:.2f}%  (ckpt={ckpt})")


if __name__ == "__main__":
    main()

