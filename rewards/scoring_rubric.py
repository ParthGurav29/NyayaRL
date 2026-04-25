"""
Centralized reward constants for NyayaRL.

Track 1 originally inlined these constants in `argument_chain.py` and `environment.py`.
Track 3 expects a stable module location for rewards.
"""

# Per-step rewards/penalties (non-terminal)
REWARD_VALID_STEP = 1.0
REWARD_PROACTIVE_COUNTER = 1.5
PENALTY_MISSING_FOUNDATION = -1.0
PENALTY_EVIDENCE_NOT_PRESENT = -0.5
PENALTY_IPC_NOT_APPLICABLE = -0.75
PENALTY_SKIPPED_STEP = -1.0
PENALTY_CONTRADICTING_WITNESS = -0.25
PENALTY_UNNECESSARY_IPC = -0.1

# Terminal rewards
TERMINAL_PRECEDENT_MATCHED = 5.0
TERMINAL_PRECEDENT_UNMATCHED = -3.0
TERMINAL_EFFICIENCY_BONUS = 1.0
TERMINAL_PROSECUTION_PENALTY = -1.0
TERMINAL_STUCK_PENALTY = -2.0
TERMINAL_TIMEOUT_PENALTY = -5.0

