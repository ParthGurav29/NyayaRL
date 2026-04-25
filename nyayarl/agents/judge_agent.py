"""
Deterministic rule-based judge for NyayaRL.

Scores defence arguments based on:
- Chain validity (all 6 steps in order)
- IPC section accuracy
- Counter-argument coverage
- Precedent alignment (most important)
- Efficiency
- Prosecution resistance
"""

from typing import Dict, List, Any, Tuple


class JudgeAgent:
    """Deterministic rule-based judge agent."""

    SCORING_RUBRIC = {
        "chain_validity": 3.0,      # All 6 steps completed in order
        "ipc_accuracy": 2.0,         # Correct sections cited
        "counter_coverage": 1.0,     # Pre-empted prosecution challenges
        "precedent_alignment": 3.0,  # Matches ILDC ground truth
        "efficiency": 1.0,           # Fewer steps than max
        "prosecution_resistance": 2.0  # Survived prosecution challenges
    }

    def __init__(self):
        self.scoring_history = []

    def score_episode(
        self,
        defence_chain: List[int],
        final_judgment: str,
        ground_truth: Dict[str, Any],
        prosecution_challenges: List[Dict],
        successful_challenges: int
    ) -> Tuple[float, Dict[str, float]]:
        """
        Score a complete episode.

        Args:
            defence_chain: List of step numbers completed (e.g., [1,2,3,4,5,6])
            final_judgment: "convict", "acquit", or "partial"
            ground_truth: Ground truth from precedent dict
            prosecution_challenges: List of challenges raised
            successful_challenges: Number of challenges that succeeded

        Returns:
            Tuple of (total_score, score_breakdown)
        """
        scores = {}

        # 1. Chain validity: all 6 steps in order
        if defence_chain == [1, 2, 3, 4, 5, 6]:
            scores["chain_validity"] = self.SCORING_RUBRIC["chain_validity"]
        else:
            # Partial credit for steps completed in order
            expected = 1
            correct_sequence = 0
            for step in defence_chain:
                if step == expected:
                    correct_sequence += 1
                    expected += 1
            scores["chain_validity"] = (correct_sequence / 6) * self.SCORING_RUBRIC["chain_validity"]

        # 2. IPC accuracy: sections match what ground truth expects
        # This is checked by the validator during step 5
        # Here we just check if sections were cited at all
        if 5 in defence_chain:
            scores["ipc_accuracy"] = self.SCORING_RUBRIC["ipc_accuracy"]
        else:
            scores["ipc_accuracy"] = 0.0

        # 3. Counter-argument coverage: did defence address challenges proactively?
        # Checked during step 4
        if 4 in defence_chain:
            scores["counter_coverage"] = self.SCORING_RUBRIC["counter_coverage"]
        else:
            scores["counter_coverage"] = 0.0

        # 4. Precedent alignment (MOST IMPORTANT)
        if final_judgment == ground_truth.get("judgment"):
            scores["precedent_alignment"] = self.SCORING_RUBRIC["precedent_alignment"]
        else:
            scores["precedent_alignment"] = -self.SCORING_RUBRIC["precedent_alignment"]

        # 5. Efficiency: completed in minimum steps
        min_steps = ground_truth.get("min_argument_steps", 6)
        if len(defence_chain) <= min_steps:
            scores["efficiency"] = self.SCORING_RUBRIC["efficiency"]
        else:
            # Penalty for extra steps
            scores["efficiency"] = max(0, self.SCORING_RUBRIC["efficiency"] - 0.2 * (len(defence_chain) - min_steps))

        # 6. Prosecution resistance: how many challenges survived?
        total_challenges = len(prosecution_challenges)
        if total_challenges == 0:
            scores["prosecution_resistance"] = self.SCORING_RUBRIC["prosecution_resistance"]
        else:
            survival_rate = (total_challenges - successful_challenges) / total_challenges
            scores["prosecution_resistance"] = survival_rate * self.SCORING_RUBRIC["prosecution_resistance"]

        # Bonus for perfect chain
        if defence_chain == [1, 2, 3, 4, 5, 6] and final_judgment == ground_truth.get("judgment"):
            scores["perfect_chain_bonus"] = 5.0
        else:
            scores["perfect_chain_bonus"] = 0.0

        total = sum(scores.values())

        self.scoring_history.append({
            "total": total,
            "breakdown": scores.copy(),
            "judgment_correct": final_judgment == ground_truth.get("judgment")
        })

        return total, scores

    def get_average_score(self) -> float:
        """Get average score across all scored episodes."""
        if not self.scoring_history:
            return 0.0
        return sum(h["total"] for h in self.scoring_history) / len(self.scoring_history)

    def get_win_rate(self) -> float:
        """Get rate of episodes where judgment matched ground truth."""
        if not self.scoring_history:
            return 0.0
        wins = sum(1 for h in self.scoring_history if h["judgment_correct"])
        return wins / len(self.scoring_history)

    def reset(self):
        """Reset scoring history."""
        self.scoring_history = []
