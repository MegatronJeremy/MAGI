"""Tally strategies.

Pluggable so the aristocratic layer is a drop-in: `MajorityVote` is pure
MAGI (1 agent = 1 vote); `WeightedVote` reads each agent's `weight` so a
council with a rotating consul / weighted seniority just swaps the strategy.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol


class TallyStrategy(Protocol):
    def tally(self, ballots: list[dict], options: list[str] | None = None) -> dict: ...


def _result(scores: dict[str, float], ballots: list[dict]) -> dict:
    if not scores:
        # All ballots were abstentions (choice=None after repair failure)
        return {
            "winner": None,
            "tie_between": None,
            "scores": {},
            "ballots": ballots,
            "deadlock": "all ballots were abstentions",
        }
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

    def tally(self, ballots: list[dict], options: list[str] | None = None) -> dict:
        scores: dict[str, float] = defaultdict(float)
        for b in ballots:
            if b.get("choice") is not None:
                scores[b["choice"]] += 1
        return _result(scores, ballots)


class WeightedVote(TallyStrategy):
    """Each ballot contributes its agent's weight."""

    def tally(self, ballots: list[dict], options: list[str] | None = None) -> dict:
        scores: dict[str, float] = defaultdict(float)
        for b in ballots:
            if b.get("choice") is not None:
                scores[b["choice"]] += b.get("weight", 1.0)
        return _result(scores, ballots)


class ConsulTieBreaker(TallyStrategy):
    """Wraps a base tally and resolves ties via a rotating 'consul'.

    The aristocratic layer: one agent holds the consul seat at a time. When the
    base tally deadlocks, the consul's own ballot decides — but only among the
    tied options (the consul can't drag in a losing choice). The seat rotates
    after every resolved decision, so power circulates rather than concentrating.

    Wrapping (rather than subclassing) means this composes with ANY base tally:
    ConsulTieBreaker(MajorityVote()) or ConsulTieBreaker(WeightedVote()).
    """

    def __init__(self, base: TallyStrategy, consul_order: list[str]):
        self.base = base
        self.consul_order = consul_order  # agent names, rotation sequence
        self._idx = 0

    @property
    def current_consul(self) -> str:
        return self.consul_order[self._idx % len(self.consul_order)]

    def _rotate(self) -> None:
        self._idx += 1

    def tally(self, ballots: list[dict], options: list[str] | None = None) -> dict:
        result = self.base.tally(ballots, options)
        if result["winner"] is not None:
            return result  # no tie, consul not needed, seat does NOT rotate

        tied = set(result["tie_between"])
        consul = self.current_consul
        consul_ballot = next((b for b in ballots if b.get("voter") == consul), None)

        decided = None
        if consul_ballot and consul_ballot.get("choice") in tied:
            decided = consul_ballot["choice"]
        else:
            # consul abstained or voted for a non-tied option: fall back to the
            # first tied option deterministically so we never hard-deadlock.
            decided = result["tie_between"][0]

        result["winner"] = decided
        result["tie_break"] = {"consul": consul, "among": result["tie_between"]}
        result["tie_between"] = None
        self._rotate()  # seat passes to the next agent for the next decision
        return result
