from __future__ import annotations

from dataclasses import replace
from typing import Any

from nyayarl.agents.judge_agent import JudgeAgent
from nyayarl.agents.prosecution_agent import ProsecutionAgent
from nyayarl.models import Action, CaseFile, JudgmentLabel, Verdict
from nyayarl.precedents import PrecedentsDB


class Track2JudgeAdapter:
    """
    Adapter: Track 2 `JudgeAgent` -> Track 1 `Judge` protocol (`evaluate`).
    """

    def __init__(self, judge_agent: JudgeAgent, precedents_db: PrecedentsDB) -> None:
        self._judge = judge_agent
        self._db = precedents_db

    def evaluate(self, case_file: CaseFile, submitted_steps: list[Action]) -> Verdict:
        # Track 2 judge expects a chain of step numbers and a final judgment string.
        defence_chain = list(range(1, len(submitted_steps) + 1))

        final_action = submitted_steps[-1] if submitted_steps else None
        final_judgment_str = (
            final_action.judgment.value if final_action and final_action.judgment else "partial"
        )

        precedent = self._db.get_precedent(case_file.precedent_id) if case_file.precedent_id else None
        ground_truth = precedent or {"judgment": "partial", "min_argument_steps": 6}

        total_score, breakdown = self._judge.score_episode(
            defence_chain=defence_chain,
            final_judgment=final_judgment_str,
            ground_truth=ground_truth,
            prosecution_challenges=[],  # Track 1 validator provides deterministic challenges; env tracks count.
            successful_challenges=0,
        )

        gt_judgment = str(ground_truth.get("judgment", "partial"))
        precedent_matched = final_judgment_str == gt_judgment

        label_map = {
            "convict": JudgmentLabel.CONVICT,
            "acquit": JudgmentLabel.ACQUIT,
            "partial": JudgmentLabel.PARTIAL,
        }
        judgment_label = label_map.get(final_judgment_str, JudgmentLabel.PARTIAL)

        return Verdict(
            judgment=judgment_label,
            matched_precedent_id=case_file.precedent_id,
            precedent_matched=precedent_matched,
            total_reward=float(total_score),
            prosecution_win_rate=0.0,
        )


class Track2ProsecutionAdapter:
    """
    Adapter: Track 2 `ProsecutionAgent` -> minimal interface used by Track 1 env.
    """

    def __init__(self, prosecution_agent: ProsecutionAgent) -> None:
        self._p = prosecution_agent

    def propose_challenge(self, case_file: CaseFile, defence_chain_len: int) -> str | None:
        # Track 2 expects dict-based case_file and a list of step numbers.
        template_id = case_file.template_id or ""
        chain = list(range(1, defence_chain_len + 1))
        challenge = self._p.select_challenge(
            case_file={"template_id": template_id},
            defence_chain=chain,
            weak_step=self._p.find_weakest_step(chain, {"template_id": template_id}),
        )
        text = challenge.get("text") if isinstance(challenge, dict) else None
        return str(text) if text else None

