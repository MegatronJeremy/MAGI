"""The Agent: a persona bound to a backend.

Each agent is the same (or any) model with a distinct system prompt.
A `weight` field is included now so the aristocratic / weighted-vote
layer plugs in later without touching this class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from magi.llm import Backend

from .voting import parse_vote


@dataclass
class Agent:
    name: str
    persona: str
    backend: Backend
    model: str = "llama3.1:8b"
    temperature: float = 0.7
    weight: float = 1.0          # used by the weighted/aristocratic voting layer
    domains: list[str] = field(default_factory=list)  # for expertise routing later

    async def respond(self, transcript: list[dict], task: str) -> str:
        system = (
            f"You are {self.name}, one member of a council of AI agents debating a problem.\n"
            f"YOUR PERSONALITY AND BIAS:\n{self.persona}\n\n"
            "Rules:\n"
            "- Stay in character. Argue from YOUR perspective, even against the others.\n"
            "- Be concise: 2-4 sentences. No preamble.\n"
            "- Reference what others said if relevant; push back when you disagree.\n"
        )
        debate = "\n".join(f"[{m['name']}]: {m['content']}" for m in transcript)
        user = (
            f"TASK UNDER DISCUSSION:\n{task}\n\n"
            f"DEBATE SO FAR:\n{debate if debate else '(you speak first)'}\n\n"
            f"Give YOUR ({self.name}) contribution now."
        )
        return await self.backend.chat(
            self.model, system, user, temperature=self.temperature
        )

    async def vote(self, transcript: list[dict], task: str, options: list[str]) -> dict:
        system = (
            f"You are {self.name}. Persona:\n{self.persona}\n\n"
            "You must now VOTE. Respond ONLY with JSON, no markdown, no extra text:\n"
            '{"choice": "<one of the options verbatim>", "reason": "<one sentence>"}'
        )
        debate = "\n".join(f"[{m['name']}]: {m['content']}" for m in transcript)
        user = (
            f"TASK:\n{task}\n\nDEBATE:\n{debate}\n\n"
            f"OPTIONS (pick exactly one, verbatim):\n"
            + "\n".join(f"- {o}" for o in options)
        )
        raw = await self.backend.chat(self.model, system, user, temperature=0.2)
        ballot = parse_vote(raw, options)
        ballot["voter"] = self.name
        ballot["weight"] = self.weight
        return ballot
