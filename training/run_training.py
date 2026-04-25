from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from nyayarl.agents import DefenceAgent, JudgeAgent, ProsecutionAgent
from nyayarl.case_generator import Track2CaseGenerator
from nyayarl.environment import NyayaRLEnvironment
from nyayarl.precedents import PrecedentsDB
from nyayarl.track2_adapters import Track2JudgeAdapter, Track2ProsecutionAdapter

from training.curriculum import CurriculumManager
from training.grpo_trainer import GRPOTrainer


REQUIRED_CONFIG_KEYS = [
    "total_train_steps",
    "eval_every",
    "eval_episodes",
    "checkpoint_every",
    "checkpoint_dir",
    "log_dir",
    "G",
    "lr",
    "clip_epsilon",
    "entropy_coef",
    "window_size",
    "promotion_threshold",
]


def _load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must parse to a mapping/dict")

    missing = [k for k in REQUIRED_CONFIG_KEYS if k not in cfg]
    if missing:
        raise ValueError(
            "Missing required config key(s): "
            + ", ".join(missing)
            + ". Please update config.yaml."
        )
    return cfg


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    cands = sorted(checkpoint_dir.glob("checkpoint_step_*.json"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _save_checkpoint(checkpoint_dir: Path, step: int, curriculum_level: int, defence: DefenceAgent) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_step_{step}.json"
    payload = {
        "step": int(step),
        "curriculum_level": int(curriculum_level),
        "defence_agent": defence.state_dict(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _load_checkpoint(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start from step 1 (ignore checkpoints) and reset logs/checkpoints.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)

    checkpoint_dir = Path(cfg["checkpoint_dir"])
    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    training_log_path = log_dir / "training.jsonl"

    # ── Build Track 2 dependencies ────────────────────────────────────────
    precedents = PrecedentsDB()
    judge_adapter = Track2JudgeAdapter(JudgeAgent(), precedents)
    prosecution_adapter = Track2ProsecutionAdapter(ProsecutionAgent())

    case_generator = Track2CaseGenerator()

    env = NyayaRLEnvironment(
        case_generator=case_generator,
        judge=judge_adapter,
        prosecution=prosecution_adapter,
    )

    curriculum = CurriculumManager(
        window_size=int(cfg["window_size"]),
        promotion_threshold=float(cfg["promotion_threshold"]),
        min_window_size=20,
    )

    defence = DefenceAgent()

    # ── Resume if checkpoint exists ───────────────────────────────────────
    start_step = 0
    if args.fresh:
        if checkpoint_dir.exists():
            for p in checkpoint_dir.glob("checkpoint_step_*.json"):
                p.unlink(missing_ok=True)
        if training_log_path.exists():
            training_log_path.unlink(missing_ok=True)
        print("[fresh] cleared checkpoints and training log; starting from step 1")
    else:
        latest = _latest_checkpoint(checkpoint_dir)
        if latest is not None:
            state = _load_checkpoint(latest)
            start_step = int(state.get("step", 0))
            curriculum.level = int(state.get("curriculum_level", 1))
            defence = DefenceAgent.from_state_dict(state.get("defence_agent", {}))
            print(
                f"[resume] loaded checkpoint {latest.name} (step={start_step}, level={curriculum.level})"
            )

    trainer = GRPOTrainer(
        environment=env,
        defence_agent=defence,
        G=int(cfg["G"]),
        lr=float(cfg["lr"]),
        clip_epsilon=float(cfg["clip_epsilon"]),
        entropy_coef=float(cfg["entropy_coef"]),
    )

    total_train_steps = int(cfg["total_train_steps"])

    # ── Train loop ────────────────────────────────────────────────────────
    for step in range(start_step, total_train_steps):
        curriculum_level = curriculum.level
        # Difficulty parameters for *this* episode batch.
        case_generator.set_difficulty_params(curriculum.get_difficulty_params(curriculum_level))

        metrics = trainer.train_step(curriculum_level=curriculum_level)

        # Curriculum signal: only count as matched if the environment produced a verdict AND it matched.
        # (If episode terminated by stuck/timeout, verdict is None.)
        verdict = env.get_last_verdict()
        judgment_matched = bool(verdict.precedent_matched) if verdict is not None else False
        curriculum.record_episode(judgment_matched, curriculum_level)

        # Contract 3 ordering: record -> check -> promote -> get params next loop.
        if curriculum.should_promote():
            new_level = curriculum.promote()
            print(f"[curriculum] promoted to level {new_level}")
        # Pre-load difficulty params for the next step (no-op for current loop).
        case_generator.set_difficulty_params(curriculum.get_difficulty_params(curriculum.level))

        # Logging
        record = {
            "step": step + 1,
            **metrics,
            "curriculum_status": curriculum.status(),
        }
        with open(training_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(
            f"[train] step={step+1}/{total_train_steps} "
            f"level={metrics['curriculum_level']} "
            f"mean_reward={metrics['mean_reward']:.3f} std={metrics['std_reward']:.3f} "
            f"policy_loss={metrics['policy_loss']:.3f}"
        )

        if (step + 1) % int(cfg["checkpoint_every"]) == 0 or (step + 1) == total_train_steps:
            ckpt = _save_checkpoint(checkpoint_dir, step + 1, curriculum.level, defence)
            print(f"[checkpoint] saved {ckpt}")

    print("training loop OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

