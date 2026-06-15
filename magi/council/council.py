"""Council orchestrator.

Composes a debate loop with pluggable context + tally strategies:
  1. N phased rounds of debate
  2. all agents cast a structured vote
  3. strategy tallies -> decision
"""

from __future__ import annotations

import asyncio
from typing import Callable

from magi.agents import Agent
from magi.llm.pool import BackendPool
from .context import ContextStrategy, KeepHeadTail
from .synthesis import Synthesizer
from .tally import MajorityVote, TallyStrategy


class Council:
    def __init__(
            self,
            agents: list[Agent],
            rounds: int = 3,
            context: ContextStrategy | None = None,
            tally: TallyStrategy | None = None,
            synthesizer: Synthesizer | None = None,
            backend_pool: BackendPool | None = None,
            on_event: Callable[[str, dict], None] | None = None,
    ):
        self.agents = agents
        self.rounds = rounds
        self.context = context or KeepHeadTail()
        self.tally_strategy = tally or MajorityVote()
        self.synthesizer = synthesizer  # optional; if None, no synthesis step
        self.backend_pool = backend_pool
        self.on_event = on_event or (lambda kind, data: None)

    async def deliberate(self, task: str, context: str = "") -> list[dict]:
        transcript: list[dict] = []
        for rnd in range(1, self.rounds + 1):
            prior_transcript = list(transcript)
            proposal_context = self.context.trim(prior_transcript)
            self.on_event("phase", {"round": rnd, "phase": "propose"})
            proposal_replies = await asyncio.gather(
                *[
                    self._agent_respond(agent, index, proposal_context, task, context)
                    for index, agent in enumerate(self.agents)
                ]
            )
            proposals = []
            for agent, reply in zip(self.agents, proposal_replies):
                turn = {
                    "name": agent.name,
                    "content": reply,
                    "round": rnd,
                    "phase": "propose",
                }
                proposals.append(turn)
                transcript.append(turn)
                self.on_event("turn", turn)

            critique_context = self.context.trim(list(transcript))
            self.on_event("phase", {"round": rnd, "phase": "critique"})
            critique_replies = await asyncio.gather(
                *[
                    self._agent_critique(
                        agent,
                        index,
                        task,
                        context,
                        proposals,
                        critique_context,
                    )
                    for index, agent in enumerate(self.agents)
                ]
            )
            for agent, reply in zip(self.agents, critique_replies):
                turn = {
                    "name": agent.name,
                    "content": reply,
                    "round": rnd,
                    "phase": "critique",
                }
                transcript.append(turn)
                self.on_event("turn", turn)

            # A future "revise" phase can slot in here after critiques are frozen.
        return transcript

    async def _agent_respond(
            self,
            agent: Agent,
            index: int,
            transcript: list[dict],
            task: str,
            context: str,
    ) -> str:
        if self.backend_pool is None:
            return await agent.respond(transcript, task, context=context)
        return await self.backend_pool.run_agent_method(
            agent, index, "respond", transcript, task, context=context
        )

    async def _agent_critique(
            self,
            agent: Agent,
            index: int,
            task: str,
            context: str,
            proposals: list[dict],
            transcript: list[dict],
    ) -> str:
        if self.backend_pool is None:
            return await agent.critique(task, context, proposals, transcript)
        return await self.backend_pool.run_agent_method(
            agent, index, "critique", task, context, proposals, transcript
        )

    async def vote(
            self, task: str, transcript: list[dict], options: list[str], context: str = ""
    ) -> dict:
        ctx = self.context.trim(transcript)
        ballots = []
        for agent in self.agents:
            ballot = await agent.vote(ctx, task, options, context=context)
            ballots.append(ballot)
            self.on_event("ballot", ballot)
        result = self.tally_strategy.tally(ballots)
        self.on_event("result", result)
        return result

    async def synthesize(
            self, task: str, transcript: list[dict], context: str = ""
    ) -> str | None:
        """Neutral synthesis of the full debate. Returns None if no synthesizer
        is configured. Uses the Untrimmed transcript on purpose: the scribe
        should see everything, even if individual agents saw a trimmed view."""
        if self.synthesizer is None:
            return None
        text = await self.synthesizer.synthesize(task, transcript, context=context)
        self.on_event("synthesis", {"text": text})
        return text
