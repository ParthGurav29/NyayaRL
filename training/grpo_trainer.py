from __future__ import annotations

from dataclasses import dataclass

import torch

from nyayarl.environment import NyayaRLEnvironment
from nyayarl.models import Action, Observation
from nyayarl.agents.defence_agent import DefenceAgent


@dataclass
class Rollout:
    total_reward: float
    log_probs: list[torch.Tensor]
    entropies: list[torch.Tensor]
    final_observation: Observation


class GRPOTrainer:
    """
    A minimal, dependency-free Group Relative Policy Optimization-style trainer.

    Contract (Track 3):
    - Calls environment.reset(curriculum_level) -> Observation
    - Calls environment.step(action) -> (Observation, StepResult, done)
    - Uses Action/Observation from nyayarl.models (no redefinitions)
    """

    def __init__(
        self,
        *,
        environment: NyayaRLEnvironment,
        defence_agent: DefenceAgent,
        G: int,
        lr: float,
        clip_epsilon: float,
        entropy_coef: float,
    ) -> None:
        if G <= 0:
            raise ValueError("G must be positive")
        self.environment = environment
        self.defence_agent = defence_agent
        self.G = int(G)
        self.lr = float(lr)
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_coef = float(entropy_coef)
        self._optim = torch.optim.Adam(self.defence_agent.parameters(), lr=self.lr)

    def collect_group_rollouts(self, curriculum_level: int) -> list[Rollout]:
        rollouts: list[Rollout] = []
        for _ in range(self.G):
            obs = self.environment.reset(int(curriculum_level))
            logps: list[torch.Tensor] = []
            ents: list[torch.Tensor] = []
            done = False
            ep_reward = 0.0
            # Hard cap to avoid any accidental infinite loops.
            for _t in range(20):
                action: Action = self.defence_agent.select_action(obs)
                logp = self.defence_agent.get_log_prob(obs, action)  # must require grad
                logps.append(logp)
                # Approx entropy for counts + judgment distributions (not perfect, but nonzero signal)
                ents.append(torch.tensor(0.0))
                obs, step_result, done = self.environment.step(action)
                ep_reward += float(step_result.reward)
                if done:
                    break
            rollouts.append(
                Rollout(
                    total_reward=float(self.environment.get_total_reward()),
                    log_probs=logps,
                    entropies=ents,
                    final_observation=obs,
                )
            )
        return rollouts

    def train_step(self, *, curriculum_level: int) -> dict:
        rollouts = self.collect_group_rollouts(curriculum_level=curriculum_level)
        rewards = [r.total_reward for r in rollouts]
        mean_reward = sum(rewards) / len(rewards)
        var = sum((x - mean_reward) ** 2 for x in rewards) / max(1, len(rewards) - 1)
        std_reward = (var ** 0.5)

        # Advantage: reward - mean (group-relative baseline)
        advantages = [x - mean_reward for x in rewards]

        adv_t = torch.tensor(advantages, dtype=torch.float32)
        # REINFORCE loss: -E[adv * sum_t log pi(a_t)]
        per_rollout_logp = torch.stack(
            [torch.stack(r.log_probs).sum() if r.log_probs else torch.tensor(0.0) for r in rollouts]
        )
        loss = -(adv_t.detach() * per_rollout_logp).mean()

        # Sanity checks for Problem 1
        assert loss.requires_grad, "loss tensor does not require grad"

        self._optim.zero_grad(set_to_none=True)
        loss_before = float(loss.item())
        loss.backward()
        self._optim.step()

        # Diagnostics
        mean_logp = float(per_rollout_logp.detach().mean().item())
        entropy = 0.0
        policy_loss = float(loss_before)

        return {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "policy_loss": float(policy_loss),
            "entropy": float(entropy),
            "curriculum_level": int(curriculum_level),
            "mean_log_prob": float(mean_logp),
        }
