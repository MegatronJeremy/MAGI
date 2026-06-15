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


class Council:
    def __init__(self, agents, rounds: int, tally, on_event=None):
        self.agents = agents
        self.rounds = rounds
        self.tally = tally
        self.on_event = on_event

    async def deliberate(self, task: str, context: str = "") -> list[dict]:
        transcript: list[dict] = []

        for round_no in range(1, self.rounds + 1):
            for agent in self.agents:
                content = await agent.respond(transcript, task, context=context)
                turn = {
                    "round": round_no,
                    "name": agent.name,
                    "content": content,
                }
                transcript.append(turn)

                if self.on_event:
                    self.on_event("turn", turn)

        return transcript

    async def vote(
            self,
            task: str,
            transcript: list[dict],
            options: list[str],
            context: str = "",
    ) -> dict:
        ballots = []

        for agent in self.agents:
            ballot = await agent.vote(
                transcript,
                task,
                options,
                context=context,
            )
            ballots.append(ballot)

            if self.on_event:
                self.on_event("ballot", ballot)

        result = self.tally.tally(ballots, options)

        if self.on_event:
            self.on_event("result", result)

        return result
