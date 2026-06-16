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
    vote_temperature: float = 0.6  # keep variance at vote time; 0.2 funneled all agents to the same answer
    weight: float = 1.0  # used by the weighted/aristocratic voting layer
    domains: list[str] = field(default_factory=list)  # this agent's allowed lane
    forbidden: list[str] = field(default_factory=list)  # axes this agent must NOT argue from
    instance: str | None = None  # optional BackendPool pin for assignment="pinned"

    def _lane_block(self) -> str:
        """Lane-locking instructions injected into propose/critique prompts.

        Pins the agent to its domain and explicitly forbids borrowing another
        member's axis. Without this, an 8B model that runs out of in-lane points
        drifts into the others' territory and the council collapses to consensus.
        """
        if not self.domains and not self.forbidden:
            return ""
        parts = []
        if self.domains:
            parts.append("YOUR LANE — argue ONLY from: " + ", ".join(self.domains) + ".")
        if self.forbidden:
            parts.append(
                "FORBIDDEN — never make arguments about: "
                + ", ".join(self.forbidden)
                + ". If one of those feels relevant, note it is another member's "
                  "concern and do NOT make the argument yourself."
            )
        parts.append(
            "Adding 'something new' means a new point WITHIN your lane, never "
            "borrowing another member's angle. If you have no new in-lane point, "
            "reply PASS - <reason> rather than drifting out of lane."
        )
        return "\n".join(parts) + "\n"

    async def respond(self, transcript: list[dict], task: str, context: str = "") -> str:
        system = (
            f"You are {self.name}, one member of a council of AI agents debating a problem.\n"
            f"YOUR PERSONALITY AND BIAS:\n{self.persona}\n\n"
            f"{self._lane_block()}"
            "Rules:\n"
            "- Stay in character. Argue from YOUR perspective, even against the others.\n"
            "- Be concise: 2-4 sentences. No preamble.\n"
            "- Reference what others said if relevant; push back when you disagree.\n"
            "- Do not restate points you have already made in earlier turns. Each turn must add something NEW: "
            "a fresh angle, a concession, a counterexample, or a sharper formulation of the disagreement. "
            "If you genuinely have nothing new to add, reply with exactly the single line: "
            "PASS - <short reason>. Do not pad.\n"
            "- Vary your phrasing. Never begin two consecutive turns with the same words.\n"
        )
        ctx_block = f"BACKGROUND CONTEXT (about the person asking):\n{context}\n\n" if context else ""
        debate = "\n".join(
            f"[round {m.get('round', '?')} {m.get('phase', 'turn')} | {m['name']}]: {m['content']}"
            for m in transcript
        )
        user = (
            f"{ctx_block}"
            f"TASK UNDER DISCUSSION:\n{task}\n\n"
            f"DEBATE SO FAR:\n{debate if debate else '(you speak first)'}\n\n"
            f"Your own earlier turns, if any, are included above under {self.name}; do not repeat them.\n\n"
            f"Give YOUR ({self.name}) contribution now."
        )
        return await self.backend.chat(
            self.model, system, user, temperature=self.temperature
        )

    async def critique(
            self,
            task: str,
            context: str,
            proposals: list[dict],
            transcript: list[dict],
    ) -> str:
        system = (
            f"You are {self.name}, one member of a council of AI agents debating a problem.\n"
            f"YOUR PERSONALITY AND BIAS:\n{self.persona}\n\n"
            f"{self._lane_block()}"
            "Rules:\n"
            "- Stay in character. Argue from YOUR perspective, even against the others.\n"
            "- Challenge the weakest points in the other members' positions.\n"
            "- Be concise: 2-4 sentences. No preamble.\n"
            "- Do not restate points you have already made in earlier turns. Each turn must add something NEW: "
            "a fresh angle, a concession, a counterexample, or a sharper formulation of the disagreement. "
            "If you genuinely have nothing new to add, reply with exactly the single line: "
            "PASS - <short reason>. Do not pad.\n"
            "- Vary your phrasing. Never begin two consecutive turns with the same words.\n"
        )
        ctx_block = f"BACKGROUND CONTEXT (about the person asking):\n{context}\n\n" if context else ""
        debate = "\n".join(
            f"[round {m.get('round', '?')} {m.get('phase', 'turn')} | {m['name']}]: {m['content']}"
            for m in transcript
        )
        other_positions = "\n".join(
            f"[{m['name']}]: {m['content']}"
            for m in proposals
            if m["name"] != self.name
        )
        user = (
            f"{ctx_block}"
            f"TASK UNDER DISCUSSION:\n{task}\n\n"
            f"DEBATE SO FAR, INCLUDING THIS ROUND'S PROPOSALS:\n{debate}\n\n"
            f"Your own earlier turns, if any, are included above under {self.name}; do not repeat them.\n\n"
            f"HERE ARE THE OTHER MEMBERS' POSITIONS THIS ROUND:\n"
            f"{other_positions if other_positions else '(no other proposals)'}\n\n"
            "Challenge the weakest points from YOUR perspective, 2-4 sentences."
        )
        return await self.backend.chat(
            self.model, system, user, temperature=self.temperature
        )

    async def vote(
            self,
            transcript: list[dict],
            task: str,
            options: list[str],
            context: str = "",
            synthesis: str = "",
    ) -> dict:
        synthesis_clause = ""
        synthesis_block = ""
        if synthesis:
            synthesis_clause = (
                "Your reason MUST engage the UNRESOLVED/MISSING factor the synthesis "
                "identified: say how your choice handles it or what it is contingent on. "
                "A vote that ignores the central unknown is invalid.\n"
            )
            synthesis_block = f"NEUTRAL SYNTHESIS OF THE DEBATE:\n{synthesis}\n\n"

        lane_clause = ""
        if self._lane_block():
            lane_clause = (
                    "\n"
                    + self._lane_block()
                    + "Your vote and your reason MUST follow from YOUR lane above. Choose the "
                      "option that best serves your domain's priorities, even if you suspect the "
                      "other members will choose differently. Do NOT converge on what seems like "
                      "the agreeable or commonly-praised answer; vote your axis. If two options "
                      "serve your lane equally, break the tie toward the one the other members "
                      "are LEAST likely to pick.\n"
            )

        system = (
                f"You are {self.name}. Persona:\n{self.persona}\n"
                + lane_clause
                + "\nYou must now VOTE. Respond ONLY with JSON, no markdown, no extra text:\n"
                  '{"choice": "<one of the options verbatim>", "reason": "<one sentence>"}\n'
                + synthesis_clause
        )
        ctx_block = f"BACKGROUND CONTEXT:\n{context}\n\n" if context else ""
        debate = "\n".join(f"[{m['name']}]: {m['content']}" for m in transcript)
        user = (
                f"{ctx_block}"
                f"{synthesis_block}"
                f"TASK:\n{task}\n\nDEBATE:\n{debate}\n\n"
                f"OPTIONS (pick exactly one, verbatim):\n"
                + "\n".join(f"- {o}" for o in options)
        )
        raw = await self.backend.chat(self.model, system, user, temperature=self.vote_temperature)
        ballot = parse_vote(raw, options)
        if ballot.get("reason") == UNPARSEABLE_VOTE_REASON:
            repair_system = (
                    f"You are {self.name}. You must repair an invalid vote.\n"
                    "Respond ONLY with valid JSON, no markdown, no prose:\n"
                    '{"choice": "<one of the options verbatim>", "reason": "<one sentence>"}\n'
                    + synthesis_clause
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
