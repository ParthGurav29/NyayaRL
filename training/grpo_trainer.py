
"""
grpo_trainer.py — Group Relative Policy Optimisation for the defence agent.

GRPO replaces the critic (value network) with group-normalised advantages:
run G rollouts for the same input, compute reward mean/std across the group,
normalise each reward as (r - mean) / std. That normalised value IS the
advantage. No value network, no bootstrapping, no TD error.

This file has one job: collect rollouts, compute advantages, update defence
policy, return metrics. No checkpoint logic, no curriculum promotion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from nyayarl.models import Action, Observation, StepResult
from nyayarl.environment import NyayaRLEnvironment


# ── Protocol for the defence agent ──────────────────────────────────────────


class DefenceAgent(Protocol):
    """
    Interface that any trainable defence agent must satisfy.

    ``select_action`` samples from the current policy.
    ``get_log_prob`` computes log π(action | observation) for a given pair.
    ``parameters`` returns the iterable of learnable parameters.
    ``get_entropy`` returns the entropy of the action distribution.
    """

    def select_action(self, observation: Observation) -> Action: ...

    def get_log_prob(self, observation: Observation, action: Action) -> torch.Tensor: ...

    def get_entropy(self, observation: Observation) -> torch.Tensor: ...

    def parameters(self) -> Any: ...


# ── Data structures ─────────────────────────────────────────────────────────


@dataclass
class TransitionRecord:
    """One (observation, action, step_result, old_log_prob) in a trajectory."""

    observation: Observation
    action: Action
    step_result: StepResult
    old_log_prob: float

@dataclass
class Rollout:
    """A complete episode trajectory with its total reward."""

    trajectory: list[TransitionRecord] = field(default_factory=list)
    total_reward: float = 0.0
    reached_step_6: bool = False
    prosecution_win_rate: float = 0.0


# ── GRPO Trainer ────────────────────────────────────────────────────────────


class GRPOTrainer:
    """

    Group Relative Policy Optimisation trainer.

    For each training step:
    1. Collect G full-episode rollouts at the given curriculum level
    2. Compute group-normalised advantages (no critic needed)
    3. Update the defence policy via clipped surrogate objective + entropy bonus

    Usage::

        trainer = GRPOTrainer(agent, env, G=8)
        metrics = trainer.train_step(curriculum_level=1)
    """

    def __init__(
        self,
        defence_agent: DefenceAgent,
        environment: NyayaRLEnvironment,
        G: int = 8,
        lr: float = 1e-4,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
    ) -> None:
        self._agent = defence_agent
        self._env = environment
        self._G = G
        self._clip_epsilon = clip_epsilon
        self._entropy_coef = entropy_coef
        self._max_grad_norm = max_grad_norm

        self._optimiser = torch.optim.Adam(
            self._agent.parameters(), lr=lr
        )
        self._last_rollouts: list[Rollout] = []

    # ── Public API ───────────────────────────────────────────────────────

    def train_step(self, curriculum_level: int) -> dict[str, float]:
        """
        Single GRPO training iteration.

        Collects G rollouts → computes group advantages → updates policy.
        Returns a dict of training metrics.
        """
        rollouts = self.collect_group_rollouts(curriculum_level)
        self._last_rollouts = rollouts
        advantages = self.compute_advantages(rollouts)
        metrics = self.update_policy(rollouts, advantages)

        # Append extra metrics
        metrics["curriculum_level"] = float(curriculum_level)
        metrics["num_successful"] = float(
            sum(1 for r in rollouts if r.reached_step_6)
        )
        metrics["mean_prosecution_win_rate"] = (
            sum(r.prosecution_win_rate for r in rollouts) / len(rollouts)
            if rollouts
            else 0.0
        )

        return metrics

    @property
    def last_rollouts(self) -> list[Rollout]:
        """The rollouts from the most recent ``train_step`` call."""
        return self._last_rollouts

    @property
    def optimiser(self) -> torch.optim.Optimizer:
        """The optimiser instance — exposed for checkpointing."""
        return self._optimiser

    # ── Rollout collection ───────────────────────────────────────────────

    def collect_group_rollouts(
        self, curriculum_level: int
    ) -> list[Rollout]:
        """
        Run G full episodes and return the rollouts.

        ``old_log_prob`` is computed under ``torch.no_grad()`` so it is
        treated as a frozen constant during the policy update.
        """
        rollouts: list[Rollout] = []

        for _ in range(self._G):
            rollout = self._run_single_episode(curriculum_level)
            rollouts.append(rollout)

        return rollouts

    @torch.no_grad()
    def _run_single_episode(self, curriculum_level: int) -> Rollout:
        """Run one full episode, recording transitions with frozen log probs."""
        observation = self._env.reset(curriculum_level)
        trajectory: list[TransitionRecord] = []
        total_reward = 0.0
        done = False

        while not done:
            action = self._agent.select_action(observation)

            # Frozen log prob — detached from the computation graph
            log_prob = self._agent.get_log_prob(observation, action)
            old_log_prob = log_prob.item()

            next_observation, step_result, done = self._env.step(action)

            trajectory.append(
                TransitionRecord(
                    observation=observation,
                    action=action,
                    step_result=step_result,
                    old_log_prob=old_log_prob,
                )
            )

            total_reward += step_result.reward
            observation = next_observation

        # Determine if the episode completed successfully (all 6 valid steps)
        valid_steps = sum(
            1 for t in trajectory if t.step_result.is_valid
        )
        reached_step_6 = valid_steps == 6

        # Prosecution win rate: fraction of steps that had a prosecution challenge
        challenges = sum(
            1 for t in trajectory
            if t.step_result.prosecution_challenge is not None
        )
        prosecution_win_rate = (
            challenges / len(trajectory) if trajectory else 0.0
        )

        return Rollout(
            trajectory=trajectory,
            total_reward=total_reward,
            reached_step_6=reached_step_6,
            prosecution_win_rate=prosecution_win_rate,
        )

    # ── Advantage computation ────────────────────────────────────────────

    @staticmethod
    def compute_advantages(rollouts: list[Rollout]) -> list[float]:
        """
        Group-normalise rewards across the G rollouts.

        Returns ``(r - mean) / (std + 1e-8)`` for each rollout.
        """
        rewards = [r.total_reward for r in rollouts]
        n = len(rewards)

        if n == 0:
            return []

        mean_r = sum(rewards) / n
        var_r = sum((r - mean_r) ** 2 for r in rewards) / n
        std_r = math.sqrt(var_r)

        return [(r - mean_r) / (std_r + 1e-8) for r in rewards]

    # ── Policy update ────────────────────────────────────────────────────

    def update_policy(
        self,
        rollouts: list[Rollout],
        advantages: list[float],
    ) -> dict[str, float]:
        """
        Compute and apply the GRPO gradient step.

        For each rollout × advantage, recompute log probs under the
        *current* policy, form the clipped surrogate loss, add entropy
        bonus, backprop, clip gradients, step.

        Returns training metrics.
        """
        total_policy_loss = torch.tensor(0.0)
        total_entropy = torch.tensor(0.0)
        total_steps = 0

        for rollout, advantage in zip(rollouts, advantages):
            adv_tensor = torch.tensor(advantage, dtype=torch.float32)

            for record in rollout.trajectory:
                # Current log prob (in the computation graph)
                log_prob = self._agent.get_log_prob(
                    record.observation, record.action
                )
                old_log_prob = torch.tensor(
                    record.old_log_prob, dtype=torch.float32
                )

                # Policy ratio
                ratio = torch.exp(log_prob - old_log_prob)

                # Clipped surrogate
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self._clip_epsilon,
                    1.0 + self._clip_epsilon,
                )

                # Loss: negative because we maximise expected advantage
                surrogate = torch.min(
                    ratio * adv_tensor,
                    clipped_ratio * adv_tensor,
                )
                policy_loss = -surrogate

                # Entropy bonus
                entropy = self._agent.get_entropy(record.observation)
                entropy_loss = -self._entropy_coef * entropy

                total_policy_loss = total_policy_loss + policy_loss + entropy_loss
                total_entropy = total_entropy + entropy.detach()
                total_steps += 1

        # Average over all timesteps
        if total_steps > 0:
            total_policy_loss = total_policy_loss / total_steps
            mean_entropy = (total_entropy / total_steps).item()
        else:
            mean_entropy = 0.0

        # Gradient step
        self._optimiser.zero_grad()
        if total_steps > 0:
            total_policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._agent.parameters(), self._max_grad_norm
            )
        self._optimiser.step()

        # Compute metrics
        rewards = [r.total_reward for r in rollouts]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        std_reward = (
            math.sqrt(sum((r - mean_reward) ** 2 for r in rewards) / len(rewards))
            if rewards
            else 0.0
        )
        mean_advantage = (
            sum(advantages) / len(advantages) if advantages else 0.0
        )

        return {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_advantage": mean_advantage,
            "policy_loss": total_policy_loss.item() if total_steps > 0 else 0.0,
            "entropy": mean_entropy,
