"""Council orchestrator.

Composes a debate loop with pluggable context + tally strategies:
  1. N rounds of debate (each agent sees everyone before it)
  2. all agents cast a structured vote
  3. strategy tallies -> decision

Sequential by design: with one GPU the agents share the accelerator and
speak in turn. Swap the inner loop to asyncio.gather only if you serve
multiple model instances or have multiple GPUs.
"""

from __future__ import annotations

from typing import Callable

from magi.agents import Agent

from .context import ContextStrategy, KeepHeadTail
from .tally import MajorityVote, TallyStrategy


class Council:
    def __init__(
        self,
        agents: list[Agent],
        rounds: int = 2,
        context: ContextStrategy | None = None,
        tally: TallyStrategy | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ):
        self.agents = agents
        self.rounds = rounds
        self.context = context or KeepHeadTail()
        self.tally_strategy = tally or MajorityVote()
        self.on_event = on_event or (lambda kind, data: None)

    async def deliberate(self, task: str) -> list[dict]:
        transcript: list[dict] = []
        for rnd in range(1, self.rounds + 1):
            for agent in self.agents:
                ctx = self.context.trim(transcript)
                reply = await agent.respond(ctx, task)
                turn = {"name": agent.name, "content": reply, "round": rnd}
                transcript.append(turn)
                self.on_event("turn", turn)
        return transcript

    async def vote(self, task: str, transcript: list[dict], options: list[str]) -> dict:
        ctx = self.context.trim(transcript)
        ballots = []
        for agent in self.agents:
            ballot = await agent.vote(ctx, task, options)
            ballots.append(ballot)
            self.on_event("ballot", ballot)
        result = self.tally_strategy.tally(ballots)
        self.on_event("result", result)
        return result
