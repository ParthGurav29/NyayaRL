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
    allow_partial_checkpoint: bool = False,
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
    res = load_defence_agent_checkpoint(
        checkpoint_path=checkpoint_path,
        agent=agent,
        map_location="cpu",
        require_full_match=not bool(allow_partial_checkpoint),
        allow_legacy_partial=bool(allow_partial_checkpoint),
        do_value_check=not bool(allow_partial_checkpoint),
    )
    if res.was_partial_load:
        print(
            "⚠ partial checkpoint load enabled (eval_alignment): "
            f"missing={len(res.diff.missing_in_checkpoint)} "
            f"unexpected={len(res.diff.unexpected_in_checkpoint)} "
            f"shape_mismatches={len(res.diff.shape_mismatches)} legacy={res.was_legacy_remap}"
        )
    agent.eval()
    # Default behavior after this change: Step 6 judgment is greedy in eval mode.
    # (Can be overridden in `main()` for ablations.)

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
    p.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help="Allow loading checkpoints with minor shape mismatches (drops mismatched tensors).",
    )
    p.add_argument(
        "--category",
        type=str,
        default=None,
        help="Optional 2-part category filter like '323_325' (assault), '302_304', etc.",
    )
    p.add_argument(
        "--print-failures",
        type=int,
        default=0,
        help="If >0, print up to N successful-but-mismatched episodes with template_id and judgment (pred vs gt).",
    )
    p.add_argument(
        "--sample-step6",
        action="store_true",
        help="Ablation: sample Step 6 judgment even in eval mode (disables greedy Step 6).",
    )
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Alias: force greedy Step 6 judgment (equivalent to NOT passing --sample-step6).",
    )
    args = p.parse_args()

    ckpt = Path(args.checkpoint) if args.checkpoint else _latest_checkpoint(Path(args.checkpoint_dir))
    alignment = eval_alignment(
        checkpoint_path=ckpt,
        templates_dir=Path(args.heldout_templates_dir),
        episodes=int(args.episodes),
        seed=int(args.seed),
        allow_partial_checkpoint=bool(args.allow_partial_checkpoint),
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
    # Step 6 is greedy by default; --sample-step6 flips it.
    # --greedy is a convenience alias to match common CLI expectations.
    sample_step6 = bool(args.sample_step6)
    if bool(args.greedy):
        sample_step6 = False
    agent.set_greedy_step6_judgment_in_eval(not sample_step6)
    res = load_defence_agent_checkpoint(
        checkpoint_path=ckpt,
        agent=agent,
        map_location="cpu",
        require_full_match=not bool(args.allow_partial_checkpoint),
        allow_legacy_partial=bool(args.allow_partial_checkpoint),
        do_value_check=not bool(args.allow_partial_checkpoint),
    )
    if res.was_partial_load:
        print(
            "⚠ partial checkpoint load enabled: "
            f"missing={len(res.diff.missing_in_checkpoint)} "
            f"unexpected={len(res.diff.unexpected_in_checkpoint)} "
            f"shape_mismatches={len(res.diff.shape_mismatches)} legacy={res.was_legacy_remap}"
        )
    agent.eval()
    successes = 0
    matched = 0
    total = int(args.episodes)
    entropies: list[float] = []
    per_cat: dict[str, dict[str, int]] = {}
    printed = 0
    want_print = int(args.print_failures or 0)
    levels = [1, 2, 3, 4]
    target_cat = _category_key(str(args.category)) if args.category else None
    i = 0
    attempts = 0
    max_attempts = max(total * 200, total + 50)  # guard against infinite loops on tiny buckets
    while i < total and attempts < max_attempts:
        attempts += 1
        epi_seed = seed + int(attempts - 1)
        random.seed(epi_seed)
        torch.manual_seed(epi_seed)
        level = levels[(attempts - 1) % len(levels)]
        case_gen = Track2CaseGenerator(case_templates_dir=Path(args.heldout_templates_dir), rng_seed=epi_seed)
        env = NyayaRLEnvironment(case_generator=case_gen, judge=judge, prosecution=None)
        obs = env.reset(level)
        cat = _category_key(obs.case_file.template_id)
        if target_cat is not None and cat != target_cat:
            continue

        per_cat.setdefault(cat, {"episodes": 0, "successes": 0, "matched": 0})
        per_cat[cat]["episodes"] += 1
        i += 1

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
            elif verdict is not None and want_print > 0 and printed < want_print:
                # Ground truth comes from the precedent referenced by the case.
                gt = precedents.get_precedent(obs.case_file.precedent_id) if obs.case_file.precedent_id else None
                gt_j = str((gt or {}).get("judgment", "partial"))
                pred_j = str(verdict.judgment.value)
                cf = obs.case_file
                present_e = sum(1 for e in cf.evidence_items if e.is_present)
                reliable_w = sum(1 for w in cf.witness_statements if w.reliability > 0.5)
                contradicting_rel = sum(1 for w in cf.witness_statements if (w.reliability > 0.5) and bool(w.is_contradicting))
                ipc = ",".join(list(cf.applicable_ipc_sections)[:6])
                print(
                    "FAIL "
                    f"template_id={obs.case_file.template_id} "
                    f"category={cat} "
                    f"precedent_id={obs.case_file.precedent_id} "
                    f"pred={pred_j} gt={gt_j} "
                    f"present_e={present_e} reliable_w={reliable_w} contradicting_rel={contradicting_rel} "
                    f"ipc=[{ipc}]"
                )
                printed += 1
    if attempts >= max_attempts and i < total and target_cat is not None:
        print(f"Warning: category filter '{target_cat}' is sparse; collected {i}/{total} episodes after {attempts} attempts.")
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

