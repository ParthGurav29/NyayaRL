from __future__ import annotations

from collections import deque


class CurriculumManager:
    """
    Sliding-window curriculum that promotes when recent win-rate exceeds a threshold.

    Contract (Track 3):
    - record_episode(judgment_matched, curriculum_level)
    - should_promote()
    - promote()
    - get_difficulty_params(level)
    - status()
    """

    def __init__(
        self,
        *,
        window_size: int = 20,
        promotion_threshold: float = 0.8,
        min_window_size: int = 20,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 <= promotion_threshold <= 1.0:
            raise ValueError("promotion_threshold must be in [0, 1]")
        if min_window_size <= 0:
            raise ValueError("min_window_size must be positive")

        self.window_size = int(window_size)
        self.promotion_threshold = float(promotion_threshold)
        self.min_window_size = int(min_window_size)
        self.level: int = 1

        # Track recent outcomes per level.
        self._windows: dict[int, deque[bool]] = {}

    def record_episode(self, judgment_matched: bool, curriculum_level: int) -> None:
        lvl = int(curriculum_level)
        if lvl not in self._windows:
            self._windows[lvl] = deque(maxlen=self.window_size)
        self._windows[lvl].append(bool(judgment_matched))

    def should_promote(self) -> bool:
        if self.level >= 4:
            return False
        w = self._windows.get(self.level)
        required = max(self.window_size, self.min_window_size)
        if w is None or len(w) < required:
            return False
        rate = sum(1 for x in w if x) / len(w)
        return rate >= self.promotion_threshold

    def promote(self) -> int:
        # Levels are 1..4 in Track 1.
        if self.level < 4:
            self.level += 1
            # Prevent instant re-promotion on stale history.
            self._windows.pop(self.level, None)
        return self.level

    def get_difficulty_params(self, level: int) -> dict:
        lvl = int(level)
        # Keys must match what `Track2CaseGenerator.set_difficulty_params` expects.
        if lvl <= 1:
            return {
                "evidence_presence_bias": +0.10,
                "witness_reliability_bias": +0.05,
                "contradiction_prob": 0.05,
                "accused_count_range": (1, 1),
            }
        if lvl == 2:
            return {
                "evidence_presence_bias": 0.00,
                "witness_reliability_bias": 0.00,
                "contradiction_prob": 0.10,
                "accused_count_range": (1, 1),
            }
        if lvl == 3:
            return {
                "evidence_presence_bias": -0.05,
                "witness_reliability_bias": -0.03,
                "contradiction_prob": 0.18,
                "accused_count_range": (1, 2),
            }
        return {
            "evidence_presence_bias": -0.10,
            "witness_reliability_bias": -0.06,
            "contradiction_prob": 0.25,
            "accused_count_range": (1, 2),
        }

    def status(self) -> dict:
        w = self._windows.get(self.level)
        recent = list(w) if w is not None else []
        rate = (sum(1 for x in recent if x) / len(recent)) if recent else 0.0
        return {
            "level": self.level,
            "window_size": self.window_size,
            "min_window_size": self.min_window_size,
            "promotion_threshold": self.promotion_threshold,
            "recent_episodes": len(recent),
            "recent_win_rate": rate,
        }

