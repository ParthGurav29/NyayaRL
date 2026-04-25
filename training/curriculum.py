"""
curriculum.py — Curriculum level manager for NyayaRL.

Tracks defence agent performance via rolling windows of judgment alignment
results and decides when to promote to the next difficulty level.

No environment calls, no agent calls, no training logic — pure bookkeeper.
"""

from __future__ import annotations

from collections import deque

# ── Difficulty parameters per level ──────────────────────────────────────────

_DIFFICULTY_PARAMS: dict[int, dict] = {
    1: {
        "accused_count": 1,
        "ipc_section_count": 1,
        "evidence_complete": True,
        "contradicting_witnesses": 0,
        "precedent_conflict": False,
    },
    2: {
        "accused_count": 1,
        "ipc_section_count": 3,
        "evidence_complete": True,
        "contradicting_witnesses": 1,
        "precedent_conflict": False,
    },
    3: {
        "accused_count": 3,
        "ipc_section_count": 3,
        "evidence_complete": False,
        "contradicting_witnesses": 2,
        "precedent_conflict": False,
    },
    4: {
        "accused_count": 3,
        "ipc_section_count": 4,
        "evidence_complete": False,
        "contradicting_witnesses": 2,
        "precedent_conflict": True,
    },
}

_MAX_LEVEL = 4


class CurriculumManager:
    """
    Tracks judgment alignment per curriculum level and decides promotions.

    Maintains a separate rolling window (``collections.deque``) for each
    level 1–4. Promotion requires the agent to sustain alignment above the
    threshold across a full window of episodes at the current level.

    Usage::

        cm = CurriculumManager()
        cm.record_episode(judgment_matched=True, curriculum_level=1)
        if cm.should_promote():
            new_level = cm.promote()
    """

    def __init__(
        self,
        starting_level: int = 1,
        promotion_threshold: float = 0.8,
        window_size: int = 100,
    ) -> None:
        self._level = starting_level
        self._promotion_threshold = promotion_threshold
        self._window_size = window_size

        # Per-level rolling windows — deque auto-evicts oldest entries
        self._windows: dict[int, deque[bool]] = {
            level: deque(maxlen=window_size) for level in range(1, _MAX_LEVEL + 1)
        }

    # ── Public API ───────────────────────────────────────────────────────

    def record_episode(
        self, judgment_matched: bool, curriculum_level: int
    ) -> None:
        """
        Record whether the agent's judgment matched the ILDC precedent.

        Appends to the rolling window for *curriculum_level*. Window is
        capped at ``window_size`` — oldest entries drop off automatically.
        """
        if curriculum_level not in self._windows:
            return  # silently ignore out-of-range levels
        self._windows[curriculum_level].append(judgment_matched)

    def current_alignment(self) -> float:
        """
        Judgment alignment rate over the rolling window for the active level.

        Returns 0.0 if the window is empty.
        """
        window = self._windows[self._level]
        if not window:
            return 0.0
        return sum(window) / len(window)

    def should_promote(self) -> bool:
        """
        True if the agent has earned promotion to the next level.

        Requires:
        - alignment >= threshold
        - window is full (at least ``window_size`` episodes recorded)
        - not already at max level (4)
        """
        if self._level >= _MAX_LEVEL:
            return False
        window = self._windows[self._level]
        if len(window) < self._window_size:
            return False
        return self.current_alignment() >= self._promotion_threshold

    def promote(self) -> int:
        """
        Increment the curriculum level by 1.

        Clears the window for the new level (fresh slate). Returns the
        new level. Defensively guards against promotion beyond max.
        """
        if self._level >= _MAX_LEVEL:
            return self._level

        self._level += 1
        # Fresh slate — past performance doesn't carry forward
        self._windows[self._level].clear()
        return self._level

    @staticmethod
    def get_difficulty_params(level: int) -> dict:
        """
        Return case-generation parameters for the given level.

        The returned dict is passed directly to the case generator (Track 2).
        """
        if level not in _DIFFICULTY_PARAMS:
            raise ValueError(
                f"Invalid curriculum level: {level}. Must be 1–{_MAX_LEVEL}."
            )
        return dict(_DIFFICULTY_PARAMS[level])  # defensive copy

    def status(self) -> dict:
        """Current state snapshot for logging."""
        window = self._windows[self._level]
        return {
            "level": self._level,
            "alignment": self.current_alignment(),
            "window_size": self._window_size,
            "episodes_in_window": len(window),
            "promotion_threshold": self._promotion_threshold,
        }

    @property
    def level(self) -> int:
        """Current curriculum level (read-only)."""
        return self._level
