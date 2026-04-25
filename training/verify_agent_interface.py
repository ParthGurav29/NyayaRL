"""
verify_agent_interface.py — Smoke test for DefenceAgent + GRPOTrainer wiring.

Runs a tiny training step using Track2 environment components to confirm:
- agent has select_action/get_log_prob/get_entropy/parameters
- rollouts collect successfully
- policy update runs without crashing
"""

from __future__ import annotations

from nyayarl.agents import DefenceAgent, JudgeAgent, ProsecutionAgent
from nyayarl.case_generator import Track2CaseGenerator
from nyayarl.environment import NyayaRLEnvironment
from nyayarl.precedents import PrecedentsDB
from nyayarl.track2_adapters import Track2JudgeAdapter, Track2ProsecutionAdapter
from training.grpo_trainer import GRPOTrainer


def main() -> None:
    precedents = PrecedentsDB()
    judge = Track2JudgeAdapter(JudgeAgent(), precedents)
    prosecution = Track2ProsecutionAdapter(ProsecutionAgent())
    case_generator = Track2CaseGenerator(rng_seed=123)
    env = NyayaRLEnvironment(case_generator=case_generator, judge=judge, prosecution=prosecution)

    agent = DefenceAgent(rng_seed=123)
    trainer = GRPOTrainer(defence_agent=agent, environment=env, G=2, lr=1e-3)

    metrics = trainer.train_step(curriculum_level=1, global_step=0)
    print("OK")
    for k in sorted(metrics.keys()):
        print(f"{k}: {metrics[k]}")


if __name__ == "__main__":
    main()

