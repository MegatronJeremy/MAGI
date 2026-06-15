"""The Agent: a persona bound to a backend.

Each agent is the same (or any) model with a distinct system prompt.
A `weight` field is included now so the aristocratic / weighted-vote
layer plugs in later without touching this class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from magi.llm import Backend
from .voting import UNPARSEABLE_VOTE_REASON, parse_vote


@dataclass
class Agent:
    name: str
    persona: str
    backend: Backend
    model: str = "llama3.1:8b"
    temperature: float = 0.7
    weight: float = 1.0  # used by the weighted/aristocratic voting layer
    domains: list[str] = field(default_factory=list)  # for expertise routing later

    async def respond(self, transcript: list[dict], task: str, context: str = "") -> str:
        system = (
            f"You are {self.name}, one member of a council of AI agents debating a problem.\n"
            f"YOUR PERSONALITY AND BIAS:\n{self.persona}\n\n"
            "Rules:\n"
            "- Stay in character. Argue from YOUR perspective, even against the others.\n"
            "- Be concise: 2-4 sentences. No preamble.\n"
            "- Reference what others said if relevant; push back when you disagree.\n"
        )
        ctx_block = f"BACKGROUND CONTEXT (about the person asking):\n{context}\n\n" if context else ""
        debate = "\n".join(f"[{m['name']}]: {m['content']}" for m in transcript)
        user = (
            f"{ctx_block}"
            f"TASK UNDER DISCUSSION:\n{task}\n\n"
            f"DEBATE SO FAR:\n{debate if debate else '(you speak first)'}\n\n"
            f"Give YOUR ({self.name}) contribution now."
        )
        return await self.backend.chat(
            self.model, system, user, temperature=self.temperature
        )

    async def vote(
            self, transcript: list[dict], task: str, options: list[str], context: str = ""
    ) -> dict:
        system = (
            f"You are {self.name}. Persona:\n{self.persona}\n\n"
            "You must now VOTE. Respond ONLY with JSON, no markdown, no extra text:\n"
            '{"choice": "<one of the options verbatim>", "reason": "<one sentence>"}'
        )
        ctx_block = f"BACKGROUND CONTEXT:\n{context}\n\n" if context else ""
        debate = "\n".join(f"[{m['name']}]: {m['content']}" for m in transcript)
        user = (
                f"{ctx_block}"
                f"TASK:\n{task}\n\nDEBATE:\n{debate}\n\n"
                f"OPTIONS (pick exactly one, verbatim):\n"
                + "\n".join(f"- {o}" for o in options)
        )
        raw = await self.backend.chat(self.model, system, user, temperature=0.2)
        ballot = parse_vote(raw, options)
        if ballot.get("reason") == UNPARSEABLE_VOTE_REASON:
            repair_system = (
                f"You are {self.name}. You must repair an invalid vote.\n"
                "Respond ONLY with valid JSON, no markdown, no prose:\n"
                '{"choice": "<one of the options verbatim>", "reason": "<one sentence>"}'
            )
            repair_user = (
                f"Your previous response was not parseable as a valid vote:\n{raw}\n\n"
                "Choose exactly one of these options, verbatim:\n"
                + "\n".join(f"- {option}" for option in options)
            )
            repaired_raw = await self.backend.chat(
                self.model, repair_system, repair_user, temperature=0.0
            )
            repaired = parse_vote(repaired_raw, options)
            if repaired.get("reason") != UNPARSEABLE_VOTE_REASON:
                ballot = repaired
        ballot["voter"] = self.name
        ballot["weight"] = self.weight
        return ballot
