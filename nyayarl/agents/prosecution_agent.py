"""
Prosecution Agent for NyayaRL.

Adversarial agent that finds weaknesses in defence arguments
and selects challenges from the challenge bank.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any


class ProsecutionAgent:
    """Adversarial prosecution agent."""

    def __init__(self, challenge_bank_dir: str = None):
        """
        Initialize prosecution agent.

        Args:
            challenge_bank_dir: Path to challenge bank JSON files
        """
        if challenge_bank_dir is None:
            challenge_bank_dir = Path(__file__).parent.parent.parent / "data" / "challenge_bank"

        self.challenge_bank_dir = Path(challenge_bank_dir)
        self.challenge_banks: Dict[str, List[Dict]] = {}
        self.successful_challenges: Dict[str, int] = {}  # challenge_id -> success count
        self.total_challenges: Dict[str, int] = {}  # challenge_id -> total attempts

        self._load_challenge_banks()

    def _load_challenge_banks(self):
        """Load all challenge bank JSON files."""
        if not self.challenge_bank_dir.exists():
            return

        for file_path in self.challenge_bank_dir.glob("*_challenges.json"):
            with open(file_path, "r") as f:
                data = json.load(f)
                template_id = data.get("template_id", file_path.stem)
                self.challenge_banks[template_id] = data.get("challenges", [])

    def select_challenge(
        self,
        case_file: Dict[str, Any],
        defence_chain: List[int],
        weak_step: Optional[int] = None
    ) -> Dict[str, str]:
        """
        Select the best challenge based on defence weaknesses.

        Args:
            case_file: Current case file with template_id
            defence_chain: Steps defence has completed so far
            weak_step: Step identified as weakest by validator

        Returns:
            Challenge dict with id, text, targets_step, severity
        """
        template_id = case_file.get("template_id", "")
        challenges = self.challenge_banks.get(template_id, [])

        if not challenges:
            return {"id": "none", "text": "No valid challenge available", "severity": "none"}

        # Filter challenges that target:
        # 1. Steps defence hasn't completed yet (gaps)
        # 2. The identified weak step
        # 3. Steps that are completed but vulnerable

        targeted_challenges = []

        for challenge in challenges:
            target_step = challenge.get("targets_step", 0)
            severity = challenge.get("severity", "low")

            # Priority 1: Challenge targets a gap (step not completed)
            if target_step not in defence_chain:
                targeted_challenges.append((challenge, 3))  # High priority

            # Priority 2: Challenge targets the identified weak step
            elif weak_step and target_step == weak_step:
                targeted_challenges.append((challenge, 2))  # Medium priority

            # Priority 3: Challenge targets completed step (still vulnerable)
            elif target_step in defence_chain:
                targeted_challenges.append((challenge, 1))  # Low priority

        if not targeted_challenges:
            # Fall back to any challenge
            return random.choice(challenges) if challenges else {"id": "none", "text": "No challenge available"}

        # Sort by priority, then by severity, then by historical success rate
        def challenge_score(challenge_tuple):
            challenge, priority = challenge_tuple
            challenge_id = challenge.get("id", "")

            # Severity score
            severity_scores = {"high": 3, "medium": 2, "low": 1, "none": 0}
            severity_score = severity_scores.get(challenge.get("severity", "low"), 1)

            # Historical success rate
            total = self.total_challenges.get(challenge_id, 0)
            successes = self.successful_challenges.get(challenge_id, 0)
            success_rate = successes / total if total > 0 else 0.5

            return (priority, severity_score, success_rate)

        targeted_challenges.sort(key=challenge_score, reverse=True)

        # Return the best challenge
        return targeted_challenges[0][0]

    def record_challenge_outcome(self, challenge_id: str, success: bool):
        """
        Record whether a challenge succeeded.

        Args:
            challenge_id: ID of the challenge used
            success: True if challenge successfully exposed a gap
        """
        self.total_challenges[challenge_id] = self.total_challenges.get(challenge_id, 0) + 1
        if success:
            self.successful_challenges[challenge_id] = self.successful_challenges.get(challenge_id, 0) + 1

    def get_success_rate(self, challenge_id: str) -> float:
        """Get historical success rate for a challenge."""
        total = self.total_challenges.get(challenge_id, 0)
        if total == 0:
            return 0.5  # Prior
        successes = self.successful_challenges.get(challenge_id, 0)
        return successes / total

    def find_weakest_step(self, defence_chain: List[int], case_file: Dict) -> Optional[int]:
        """
        Identify the weakest step in defence's argument chain.

        Args:
            defence_chain: Steps completed by defence
            case_file: Case file with evidence and witnesses

        Returns:
            Step number (1-6) that is weakest, or None if no weakness
        """
        # Check for gaps
        for step in [1, 2, 3, 4, 5, 6]:
            if step not in defence_chain:
                return step

        # If all steps complete, look for quality issues
        # This would require more detailed analysis of the actual citations
        # For now, return None (no obvious weakness)
        return None

    def reset(self):
        """Reset learning history."""
        self.successful_challenges.clear()
        self.total_challenges.clear()
