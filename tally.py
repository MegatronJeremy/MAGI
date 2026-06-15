"""Tally strategies.

Pluggable so the aristocratic layer is a drop-in: `MajorityVote` is pure
MAGI (1 agent = 1 vote); `WeightedVote` reads each agent's `weight` so a
council with a rotating consul / weighted seniority just swaps the strategy.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol


class TallyStrategy(Protocol):
    def tally(self, ballots: list[dict]) -> dict: ...


def _result(scores: dict[str, float], ballots: list[dict]) -> dict:
    top_score = max(scores.values())
    leaders = [opt for opt, s in scores.items() if s == top_score]
    return {
        "winner": leaders[0] if len(leaders) == 1 else None,
        "tie_between": leaders if len(leaders) > 1 else None,
        "scores": dict(scores),
        "ballots": ballots,
    }


class MajorityVote(TallyStrategy):
    """One agent, one vote."""

    def tally(self, ballots: list[dict]) -> dict:
        scores: dict[str, float] = defaultdict(float)
        for b in ballots:
            scores[b["choice"]] += 1
        return _result(scores, ballots)


class WeightedVote(TallyStrategy):
    """Each ballot contributes its agent's weight."""

    def tally(self, ballots: list[dict]) -> dict:
        scores: dict[str, float] = defaultdict(float)
        for b in ballots:
            scores[b["choice"]] += b.get("weight", 1.0)
        return _result(scores, ballots)
