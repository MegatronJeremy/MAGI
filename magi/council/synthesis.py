"""Synthesis.

The debate produces rich, conflicting argument; the vote crushes it into one
word. The synthesizer recovers what the vote throws away: a neutral reader (NOT
one of the partisan council members) digests the whole transcript into the parts
that actually carry decision value — where the council agreed, where it split and
why, and what remains unresolved.

This is deliberately a separate role with its own persona. Reusing a debating
agent would bias the summary toward that agent's stance; the whole value here is
neutrality.
"""

from __future__ import annotations

from magi.llm import Backend


class Synthesizer:
    def __init__(self, backend: Backend, model: str, temperature: float = 0.3):
        self.backend = backend
        self.model = model
        self.temperature = temperature

    async def synthesize(
            self, task: str, transcript: list[dict], context: str = ""
    ) -> str:
        system = (
            "You are the SCRIBE of a council. You did NOT take part in the debate "
            "and you hold no position. Your job is to digest the council's debate "
            "into the few things that actually help a decision get made.\n\n"
            "Write a tight synthesis with exactly these four parts, each 1-3 "
            "sentences, plain prose, no fluff:\n"
            "1. POINTS OF AGREEMENT — what (if anything) all members converged on.\n"
            "2. THE REAL SPLIT — where they genuinely disagreed and the strongest "
            "reason on each side.\n"
            "3. UNRESOLVED / MISSING — the single most important fact or assumption "
            "left hanging that would change the answer if known.\n"
            "4. STRONGEST CONSIDERATION — the one argument that survived the most "
            "scrutiny, named plainly. Do NOT issue a verdict or tell the person "
            "what to do; surface the substance and let them decide."
        )
        ctx_block = f"BACKGROUND CONTEXT:\n{context}\n\n" if context else ""
        debate = "\n".join(f"[{m['name']}]: {m['content']}" for m in transcript)
        user = (
            f"{ctx_block}TASK THE COUNCIL DEBATED:\n{task}\n\n"
            f"FULL DEBATE:\n{debate}\n\n"
            "Write the four-part synthesis now."
        )
        return await self.backend.chat(
            self.model, system, user, temperature=self.temperature
        )
