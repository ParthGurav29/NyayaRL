"""
run_training.py — Entry point for the NyayaRL training pipeline.

Wires every component together — environment, agents, trainer, curriculum —
and runs the training loop. Owns checkpointing, logging, and evaluation.
Nothing novel lives here — it just orchestrates.
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

from nyayarl.environment import NyayaRLEnvironment
from nyayarl.models import (
    Action,
    CaseFile,
    EvidenceItem,
    EvidenceType,
    JudgmentLabel,
    Observation,
    StepType,
    Verdict,
    WitnessStatement,
)
from nyayarl.agents import DefenceAgent
from training.curriculum import CurriculumManager
from training.grpo_trainer import GRPOTrainer

# ── Required config keys ────────────────────────────────────────────────────

_REQUIRED_CONFIG_KEYS = [
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


# ── Config loading + validation ─────────────────────────────────────────────


def load_config(path: str) -> dict[str, Any]:
    """Load and validate config from a YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    missing = [k for k in _REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise ValueError(
            f"Missing required config keys: {', '.join(missing)}. "
            f"Required: {_REQUIRED_CONFIG_KEYS}"
        )

    return config


# ── Stub dependencies (replaced when Track 2/3 deliver real ones) ───────────

_STEP_SEQUENCE = [
    StepType.ACTUS_REUS,
    StepType.MENS_REA,
    StepType.LINKAGE,
    StepType.COUNTER_ARGUMENT,
    StepType.IPC_APPLICATION,
    StepType.PRECEDENT_CITATION,
]


class StubCaseGenerator:
    """Generates a structurally valid case file for any curriculum level."""

    def generate(self, curriculum_level: int) -> CaseFile:
        return CaseFile(
            case_id=f"TRAIN_{curriculum_level:03d}",
            fir="[Stub FIR] Placeholder first information report.",
            accused_count=1,
            evidence_items=[
                EvidenceItem(
                    id="E1",
                    description="Physical evidence",
                    type=EvidenceType.PHYSICAL,
                    is_present=True,
                ),
                EvidenceItem(
                    id="E2",
                    description="Forensic evidence",
                    type=EvidenceType.FORENSIC,
                    is_present=True,
                ),
                EvidenceItem(
                    id="E3",
                    description="Documentary evidence",
                    type=EvidenceType.DOCUMENTARY,
                    is_present=True,
                ),
            ],
            witness_statements=[
                WitnessStatement(
                    id="W1",
                    content="Witness statement",
                    reliability=0.85,
                    is_contradicting=False,
                ),
            ],
            applicable_ipc_sections=["302", "307", "34"],
            curriculum_level=curriculum_level,
            precedent_id="ILDC_STUB_001",
        )


class StubJudge:
    """Returns a placeholder verdict."""

    def evaluate(self, case_file: CaseFile, submitted_steps: list[Action]) -> Verdict:
        judgment = JudgmentLabel.PARTIAL
        if submitted_steps and submitted_steps[-1].judgment is not None:
            judgment = submitted_steps[-1].judgment
        return Verdict(
            judgment=judgment,
            matched_precedent_id=case_file.precedent_id,
            precedent_matched=True,
            total_reward=0.0,
            prosecution_win_rate=0.2,
        )


class StubDefenceAgent(nn.Module):
    """
    Minimal trainable agent that follows the perfect 6-step path.

    Stands in until Track 3 delivers the real policy network.
    """

    def __init__(self) -> None:
        super().__init__()
        self._policy_head = nn.Linear(1, 1)  # learnable params for gradient flow

    def select_action(self, observation: Observation) -> Action:
        step_idx = min(len(observation.submitted_steps), 5)
        step_type = _STEP_SEQUENCE[step_idx]

        if step_type == StepType.ACTUS_REUS:
            return Action(step_type=step_type, anchored_evidence_ids=["E1"])
        elif step_type == StepType.MENS_REA:
            return Action(step_type=step_type, anchored_witness_ids=["W1"])
        elif step_type == StepType.LINKAGE:
            return Action(
                step_type=step_type,
                anchored_evidence_ids=["E3"],
                cited_ipc_sections=["302"],
            )
        elif step_type == StepType.COUNTER_ARGUMENT:
            return Action(step_type=step_type, anchored_witness_ids=["W1"])
        elif step_type == StepType.IPC_APPLICATION:
            return Action(step_type=step_type, cited_ipc_sections=["302"])
        else:  # PRECEDENT_CITATION
            return Action(
                step_type=step_type,
                judgment=JudgmentLabel.CONVICT,
                anchored_evidence_ids=["ILDC_STUB_001"],
            )

    def get_log_prob(self, observation: Observation, action: Action) -> torch.Tensor:
        x = self._policy_head(torch.tensor([1.0]))
        return -1.0 + x.squeeze() * 0.01

    def get_entropy(self, observation: Observation) -> torch.Tensor:
        x = self._policy_head(torch.tensor([1.0]))
        return torch.tensor(1.0) + x.squeeze() * 0.001


# ── Checkpointing ───────────────────────────────────────────────────────────


def save_checkpoint(
    path: Path,
    agent: nn.Module,
    optimiser: torch.optim.Optimizer,
    curriculum: CurriculumManager,
    step: int,
) -> None:
    """Save training state to a checkpoint file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialise curriculum window contents
    window_data = {}
    for level in range(1, 5):
        window_data[level] = list(curriculum._windows[level])

    torch.save(
        {
            "step": step,
            "agent_state_dict": agent.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "curriculum_level": curriculum.level,
            "curriculum_windows": window_data,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    agent: nn.Module,
    optimiser: torch.optim.Optimizer,
    curriculum: CurriculumManager,
) -> int:
    """
    Load training state from a checkpoint file. Returns the step to resume from.
    """
    checkpoint = torch.load(path, weights_only=False)

    agent.load_state_dict(checkpoint["agent_state_dict"])
    optimiser.load_state_dict(checkpoint["optimiser_state_dict"])

    # Restore curriculum level
    curriculum._level = checkpoint["curriculum_level"]

    # Restore window contents
    for level, entries in checkpoint["curriculum_windows"].items():
        level_int = int(level)
        curriculum._windows[level_int].clear()
        curriculum._windows[level_int].extend(entries)

    return checkpoint["step"]


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the most recent checkpoint in the directory, or None."""
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
    return checkpoints[-1] if checkpoints else None


# ── Evaluation loop ─────────────────────────────────────────────────────────


@torch.no_grad()
def run_eval(
    agent: nn.Module,
    env: NyayaRLEnvironment,
    curriculum_level: int,
    num_episodes: int,
) -> dict[str, float]:
    """
    Run evaluation episodes with the agent in deterministic mode.

    No policy updates — inference only.
    """
    agent.eval()

    total_rewards: list[float] = []
    steps_taken: list[int] = []
    judgments_matched: list[bool] = []
    prosecution_win_rates: list[float] = []

    for _ in range(num_episodes):
        observation = env.reset(curriculum_level)
        ep_reward = 0.0
        ep_steps = 0
        done = False

        while not done:
            action = agent.select_action(observation)
            observation, step_result, done = env.step(action)
            ep_reward += step_result.reward
            ep_steps += 1

        # Track judgment/precedent match when the episode succeeds.
        verdict = env.get_last_verdict()
        matched = bool(verdict.precedent_matched) if verdict is not None else False

        challenges = sum(1 for c in observation.prosecution_challenges)
        pwr = challenges / ep_steps if ep_steps > 0 else 0.0

        total_rewards.append(ep_reward)
        steps_taken.append(ep_steps)
        judgments_matched.append(matched)
        prosecution_win_rates.append(pwr)

    agent.train()

    n = len(total_rewards)
    mean_alignment = sum(judgments_matched) / n if n > 0 else 0.0

    return {
        "mean_alignment": mean_alignment,
        "mean_reward": sum(total_rewards) / n if n > 0 else 0.0,
        "mean_steps": sum(steps_taken) / n if n > 0 else 0.0,
        "mean_prosecution_win_rate": sum(prosecution_win_rates) / n if n > 0 else 0.0,
        "pass_threshold": mean_alignment >= 0.8,
        "num_episodes": n,
    }


# ── Logging ─────────────────────────────────────────────────────────────────


class TrainingLogger:
    """Writes JSON-line logs and human-readable stdout summaries."""

    def __init__(self, log_dir: str) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_dir / "training.jsonl"

    def log_step(
        self,
        step: int,
        metrics: dict[str, float],
        curriculum_status: dict,
        promoted: bool,
        eval_results: dict | None = None,
    ) -> None:
        """Write one log entry (JSON line + stdout)."""
        entry = {
            "step": step,
            "curriculum_level": int(metrics.get("curriculum_level", 0)),
            "mean_reward": metrics.get("mean_reward", 0.0),
            "std_reward": metrics.get("std_reward", 0.0),
            "policy_loss": metrics.get("policy_loss", 0.0),
            "entropy": metrics.get("entropy", 0.0),
            "alignment": curriculum_status.get("alignment", 0.0),
            "promoted": promoted,
            "eval": eval_results,
        }

        # JSON line
        with open(self._log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Human-readable stdout
        level = entry["curriculum_level"]
        reward = entry["mean_reward"]
        std = entry["std_reward"]
        alignment = entry["alignment"] * 100
        loss = entry["policy_loss"]
        promo = " ⬆ PROMOTED!" if promoted else ""
        print(
            f"Step {step:04d} | Level {level} | "
            f"Reward {reward:.2f} ± {std:.1f} | "
            f"Alignment {alignment:.1f}% | "
            f"Loss {loss:.4f}{promo}"
        )

        if eval_results:
            ea = eval_results["mean_alignment"] * 100
            er = eval_results["mean_reward"]
            ep = "PASS" if eval_results["pass_threshold"] else "FAIL"
            print(f"         └─ Eval: Alignment {ea:.1f}% [{ep}] | Reward {er:.2f}")


# ── Main ────────────────────────────────────────────────────────────────────


def main(config_path: str = "config.yaml") -> None:
    """Run the full training pipeline."""
    # 1. Parse config
    config = load_config(config_path)
    print(f"Loaded config from {config_path}")
    print(f"  total_train_steps: {config['total_train_steps']}")
    print(f"  G: {config['G']}, lr: {config['lr']}")
    print()

    # 2. Instantiate environment
    env = NyayaRLEnvironment(
        case_generator=StubCaseGenerator(),
        judge=StubJudge(),
    )

    # 3. Instantiate curriculum manager
    curriculum = CurriculumManager(
        starting_level=1,
        promotion_threshold=config["promotion_threshold"],
        window_size=config["window_size"],
    )

    # 4. Instantiate defence agent
    agent = DefenceAgent()

    # 5. Instantiate GRPO trainer
    trainer = GRPOTrainer(
        defence_agent=agent,
        environment=env,
        G=config["G"],
        lr=config["lr"],
        clip_epsilon=config["clip_epsilon"],
        entropy_coef=config["entropy_coef"],
    )

    # 6. Set up logger
    logger = TrainingLogger(config["log_dir"])

    # 7. Set up checkpoint directory
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 8. Resume from checkpoint if available
    start_step = 0
    latest_ckpt = find_latest_checkpoint(checkpoint_dir)
    if latest_ckpt is not None:
        start_step = load_checkpoint(latest_ckpt, agent, trainer.optimiser, curriculum)
        print(f"Resumed from checkpoint: {latest_ckpt} (step {start_step})")
        start_step += 1  # resume from the NEXT step
    else:
        print("No checkpoint found — starting fresh.")
    print()

    # ── Training loop ────────────────────────────────────────────────────
    total_steps = config["total_train_steps"]
    eval_every = config["eval_every"]
    eval_episodes = config["eval_episodes"]
    checkpoint_every = config["checkpoint_every"]

    for step in range(start_step, total_steps):
        # 1. Get current curriculum level
        level = curriculum.level

        # 2. Train step
        metrics = trainer.train_step(level, global_step=step)

        # 3. Record each episode from the last group rollout
        for rollout in trainer.last_rollouts:
            # Use reached_step_6 as proxy for judgment_matched
            # (refined when real judge from Track 2 is integrated)
            curriculum.record_episode(
                judgment_matched=rollout.reached_step_6,
                curriculum_level=level,
            )

        # 4. Check promotion
        promoted = False
        if curriculum.should_promote():
            new_level = curriculum.promote()
            promoted = True
            print(f"\n{'=' * 60}")
            print(f"  PROMOTED: Level {level} → Level {new_level}")
            print(f"{'=' * 60}\n")

        # 5. Evaluation
        eval_results = None
        if (step + 1) % eval_every == 0:
            eval_results = run_eval(agent, env, curriculum.level, eval_episodes)

        # 6. Log
        logger.log_step(
            step=step,
            metrics=metrics,
            curriculum_status=curriculum.status(),
            promoted=promoted,
            eval_results=eval_results,
        )

        # 7. Checkpoint
        if (step + 1) % checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"step_{step:06d}.pt"
            save_checkpoint(ckpt_path, agent, trainer.optimiser, curriculum, step)
            print(f"  💾 Checkpoint saved: {ckpt_path}")

    print()
    print("=" * 60)
    print(f"Training complete. {total_steps} steps finished.")
    print(f"Final curriculum level: {curriculum.level}")
    print(f"Final alignment: {curriculum.current_alignment():.2%}")
    print("=" * 60)


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(config_file)
