"""
eval_harness.py — Fixed-seed evaluation harness for NyayaRL.

Runs the defence agent against the Track2-backed environment and reports:
- chain completion rate
- judgment accuracy vs precedent (via Track2JudgeAdapter)
- mean reward and prosecution pressure stats
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from nyayarl.agents import DefenceAgent, JudgeAgent, ProsecutionAgent
from nyayarl.case_generator import Track2CaseGenerator
from nyayarl.environment import NyayaRLEnvironment
from nyayarl.precedents import PrecedentsDB
from nyayarl.track2_adapters import Track2JudgeAdapter, Track2ProsecutionAdapter


@torch.no_grad()
def run_eval(*, episodes: int, level: int, seed: int) -> dict:
    # Seed the Track2CaseGenerator so eval is repeatable.
    case_gen = Track2CaseGenerator(rng_seed=seed)
    precedents = PrecedentsDB()
    judge = Track2JudgeAdapter(JudgeAgent(), precedents)
    prosecution = Track2ProsecutionAdapter(ProsecutionAgent())

    env = NyayaRLEnvironment(case_generator=case_gen, judge=judge, prosecution=prosecution)
    agent = DefenceAgent(rng_seed=seed)
    agent.eval()

    completed = 0
    precedent_matched = 0
    rewards: list[float] = []
    prosecution_win_rates: list[float] = []

    for _ in range(episodes):
        obs = env.reset(level)
        done = False
        steps = 0
        while not done:
            action = agent.select_action(obs)
            obs, sr, done = env.step(action)
            steps += 1

        rewards.append(env.get_total_reward())
        prosecution_win_rates.append(len(obs.prosecution_challenges) / steps if steps else 0.0)

        if len(obs.submitted_steps) == 6:
            completed += 1
            verdict = env.get_last_verdict()
            if verdict is not None and verdict.precedent_matched:
                precedent_matched += 1

    n = float(episodes) if episodes else 1.0
    return {
        "episodes": episodes,
        "level": level,
        "seed": seed,
        "chain_completion_rate": completed / n,
        "precedent_match_rate": (precedent_matched / n) if episodes else 0.0,
        "mean_total_reward": sum(rewards) / n,
        "mean_prosecution_win_rate": sum(prosecution_win_rates) / n,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    report = run_eval(episodes=args.episodes, level=args.level, seed=args.seed)
    print(json.dumps(report, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

