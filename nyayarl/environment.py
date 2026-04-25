"""
environment.py — Core logic layer for NyayaRL.

Owns episode state, orchestrates flow between case file, validator,
reward, and agents. Has no HTTP knowledge — that's ``app.py``'s job.

Track 2's judge and prosecution agents plug in via constructor injection.
"""

from __future__ import annotations

from typing import Protocol

from nyayarl.argument_chain import ArgumentChainValidator
from nyayarl.models import (
    Action,
    CaseFile,
    JudgmentLabel,
    Observation,
    StepResult,
    StepType,
    Verdict,
)

# ── Terminal reward placeholders (will be imported from scoring_rubric.py) ───

_TERMINAL_PRECEDENT_MATCHED = 5.0
_TERMINAL_PRECEDENT_UNMATCHED = -3.0
_TERMINAL_EFFICIENCY_BONUS = 1.0
_TERMINAL_PROSECUTION_PENALTY = -1.0
_TERMINAL_STUCK_PENALTY = -5.0
_TERMINAL_TIMEOUT_PENALTY = -5.0

# ── Constants ────────────────────────────────────────────────────────────────

_TOTAL_STEPS = 6
_MAX_ACTIONS = 10
_MAX_CONSECUTIVE_INVALID = 3
_MIN_VALID_PATH_LENGTH = 6  # one valid action per step type


# ── Protocols for Track 2 injected dependencies ─────────────────────────────


class CaseGenerator(Protocol):
    """Produces a fresh ``CaseFile`` for a given curriculum level."""

    def generate(self, curriculum_level: int) -> CaseFile: ...


class Judge(Protocol):
    """
    Computes a ``Verdict`` at episode end.

    Receives the case file and the full list of valid submitted actions.
    """

    def evaluate(
        self,
        case_file: CaseFile,
        submitted_steps: list[Action],
    ) -> Verdict: ...


# ── Environment ──────────────────────────────────────────────────────────────


class NyayaRLEnvironment:
    """
    Core reinforcement-learning environment for NyayaRL.

    Manages episode lifecycle through two public methods:

    - ``reset(curriculum_level)`` — start a new episode
    - ``step(action)`` — submit one action in the argument chain

    Judge and case generator are injected via the constructor so the
    environment is testable without real Track 2 agents.
    """

    def __init__(
        self,
        case_generator: CaseGenerator,
        judge: Judge,
    ) -> None:
        self._case_generator = case_generator
        self._judge = judge
        self._validator = ArgumentChainValidator()

        # ── Episode state (all reset in reset()) ─────────────────────────
        self._case_file: CaseFile | None = None
        self._submitted_steps: list[Action] = []
        self._prosecution_challenges: list[str] = []
        self._total_reward: float = 0.0
        self._prosecution_challenge_count: int = 0
        self._is_done: bool = True  # starts done — must call reset()
        self._curriculum_level: int = 1
        self._action_count: int = 0
        self._consecutive_invalid: int = 0

    # ── Public API ───────────────────────────────────────────────────────

    def reset(self, curriculum_level: int) -> Observation:
        """
        Clear all episode state and start a fresh episode.

        Calls the case generator to produce a new ``CaseFile`` for the
        given *curriculum_level*. Returns the initial ``Observation``.
        """
        self._case_file = self._case_generator.generate(curriculum_level)
        self._submitted_steps = []
        self._prosecution_challenges = []
        self._total_reward = 0.0
        self._prosecution_challenge_count = 0
        self._is_done = False
        self._curriculum_level = curriculum_level
        self._action_count = 0
        self._consecutive_invalid = 0

        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, StepResult, bool]:
        """
        Process one agent action through the argument chain.

        Returns ``(observation, step_result, done)``.

        Raises ``RuntimeError`` if the episode is already done.
        """
        # 1. Cannot step a finished episode
        if self._is_done:
            raise RuntimeError(
                "Episode is already done. Call reset() to start a new episode."
            )

        assert self._case_file is not None  # guaranteed after reset()

        self._action_count += 1

        # 2. Validate the action
        step_result = self._validator.validate(
            action, self._case_file, self._submitted_steps
        )

        # 3. If valid, append to submitted steps
        if step_result.is_valid:
            self._submitted_steps.append(action)
            self._consecutive_invalid = 0
        else:
            self._consecutive_invalid += 1

        # 4. Track prosecution challenges
        if step_result.prosecution_challenge is not None:
            self._prosecution_challenges.append(step_result.prosecution_challenge)
            self._prosecution_challenge_count += 1

        # 5. Accumulate reward
        self._total_reward += step_result.reward

        # 6. Check termination
        termination = self._check_termination(step_result)

        if termination is not None:
            self._is_done = True

            # 7. Apply terminal reward
            terminal_reward = self._compute_terminal_reward(termination)
            self._total_reward += terminal_reward

        # 8–9. Build observation and return
        observation = self._build_observation()
        return observation, step_result, self._is_done

    # ── Private — termination logic ──────────────────────────────────────

    def _check_termination(self, step_result: StepResult) -> str | None:
        """
        Determine if the episode should end.

        Returns the termination reason string, or ``None`` to continue.
        """
        # Success: all 6 steps submitted and last step was valid
        if (
            len(self._submitted_steps) == _TOTAL_STEPS
            and step_result.is_valid
        ):
            return "success"

        # Stuck: 3 consecutive invalid steps
        if self._consecutive_invalid >= _MAX_CONSECUTIVE_INVALID:
            return "stuck"

        # Timeout: 10 actions taken without completing the chain
        if self._action_count >= _MAX_ACTIONS:
            return "timeout"

        return None

    def _compute_terminal_reward(self, termination: str) -> float:
        """
        Compute the one-time terminal reward applied when the episode ends.
        """
        assert self._case_file is not None

        if termination in ("stuck", "timeout"):
            return _TERMINAL_STUCK_PENALTY if termination == "stuck" else _TERMINAL_TIMEOUT_PENALTY

        # ── Success path — call the judge ────────────────────────────────
        verdict = self._judge.evaluate(self._case_file, self._submitted_steps)

        reward = 0.0

        # Precedent match bonus/penalty
        if verdict.precedent_matched:
            reward += _TERMINAL_PRECEDENT_MATCHED
        else:
            reward += _TERMINAL_PRECEDENT_UNMATCHED

        # Efficiency bonus: completed the chain with no wasted actions
        if self._action_count <= _MIN_VALID_PATH_LENGTH:
            reward += _TERMINAL_EFFICIENCY_BONUS

        # Prosecution dominance penalty
        if verdict.prosecution_win_rate > 0.6:
            reward += _TERMINAL_PROSECUTION_PENALTY

        return reward

    # ── Private — observation builder ────────────────────────────────────

    def _build_observation(self) -> Observation:
        """
        Snapshot the current episode state into an ``Observation``.
        """
        assert self._case_file is not None

        # chain_score = valid steps / total actions taken (0.0 if no actions yet)
        if self._action_count > 0:
            chain_score = len(self._submitted_steps) / self._action_count
        else:
            chain_score = 0.0

        return Observation(
            case_file=self._case_file,
            submitted_steps=list(self._submitted_steps),  # defensive copy
            prosecution_challenges=list(self._prosecution_challenges),
            chain_score=chain_score,
            curriculum_level=self._curriculum_level,
            is_done=self._is_done,
        )
