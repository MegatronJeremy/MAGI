"""Council orchestrator.

Composes a debate loop with pluggable context + tally strategies:
  1. N phased rounds of debate
  2. all agents cast a structured vote
  3. strategy tallies -> decision
"""

from __future__ import annotations

import asyncio
import math
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
            rounds: int = 2,
            min_rounds: int = 1,
            pass_threshold: int | None = None,
            context: ContextStrategy | None = None,
            tally: TallyStrategy | None = None,
            synthesizer: Synthesizer | None = None,
            backend_pool: BackendPool | None = None,
            on_event: Callable[[str, dict], None] | None = None,
    ):
        self.agents = agents
        self.rounds = max(1, rounds)
        self.min_rounds = max(1, min(min_rounds, self.rounds))
        self.pass_threshold = pass_threshold
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
                    self._agent_respond(
                        agent,
                        index,
                        self._with_agent_history(prior_transcript, proposal_context, agent.name),
                        task,
                        context,
                    )
                    for index, agent in enumerate(self.agents)
                ]
            )
            proposals = []
            for agent, reply in zip(self.agents, proposal_replies):
                turn = self._turn(agent.name, reply, rnd, "propose")
                proposals.append(turn)
                transcript.append(turn)
                self.on_event("turn", turn)

            full_critique_transcript = list(transcript)
            critique_context = self.context.trim(full_critique_transcript)
            self.on_event("phase", {"round": rnd, "phase": "critique"})
            critique_replies = await asyncio.gather(
                *[
                    self._agent_critique(
                        agent,
                        index,
                        task,
                        context,
                        proposals,
                        self._with_agent_history(
                            full_critique_transcript,
                            critique_context,
                            agent.name,
                        ),
                    )
                    for index, agent in enumerate(self.agents)
                ]
            )
            critique_turns = []
            for agent, reply in zip(self.agents, critique_replies):
                turn = self._turn(agent.name, reply, rnd, "critique")
                critique_turns.append(turn)
                transcript.append(turn)
                self.on_event("turn", turn)

            if rnd >= self.min_rounds and self._has_converged(critique_turns):
                pass_count = sum(1 for turn in critique_turns if turn.get("pass"))
                self.on_event(
                    "converged",
                    {
                        "round": rnd,
                        "passes": pass_count,
                        "agents": len(self.agents),
                        "threshold": self._pass_threshold(),
                    },
                )
                break

            # A future "revise" phase can slot in here after critiques are frozen.
        return transcript

    def _pass_threshold(self) -> int:
        if self.pass_threshold is not None:
            return max(1, min(self.pass_threshold, len(self.agents)))
        return math.ceil(len(self.agents) / 2)

    def _has_converged(self, critique_turns: list[dict]) -> bool:
        pass_count = sum(1 for turn in critique_turns if turn.get("pass"))
        return pass_count >= self._pass_threshold()

    def _turn(self, name: str, reply: str, round_number: int, phase: str) -> dict:
        content = str(reply or "").strip()
        passed = content.casefold().startswith("pass")
        if passed:
            content = content.splitlines()[0].strip()
        return {
            "name": name,
            "content": content,
            "round": round_number,
            "phase": phase,
            "pass": passed,
        }

    def _with_agent_history(
            self,
            full_transcript: list[dict],
            trimmed_transcript: list[dict],
            agent_name: str,
    ) -> list[dict]:
        trimmed_keys = {self._turn_key(turn) for turn in trimmed_transcript}
        return [
            turn
            for turn in full_transcript
            if turn.get("name") == agent_name or self._turn_key(turn) in trimmed_keys
        ]

    @staticmethod
    def _turn_key(turn: dict) -> tuple[object, object, object, object]:
        return (
            turn.get("round"),
            turn.get("phase"),
            turn.get("name"),
            turn.get("content"),
        )

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
            self,
            task: str,
            transcript: list[dict],
            options: list[str],
            context: str = "",
            synthesis: str = "",
    ) -> dict:
        ctx = self.context.trim(transcript)
        ballots = []
        for agent in self.agents:
            ballot = await agent.vote(
                ctx, task, options, context=context, synthesis=synthesis
            )
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
