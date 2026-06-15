"""Derive vote options from a debate when the user didn't supply any."""

from __future__ import annotations

from magi.agents import Agent


async def derive_options(agent: Agent, task: str, transcript: list[dict]) -> list[str]:
    debate = "\n".join(f"[{m['name']}]: {m['content']}" for m in transcript)
    system = (
        "You summarize a debate into 2-4 distinct, mutually exclusive options to "
        "vote on. Respond as a plain newline-separated list. No numbering, no commentary."
    )
    user = f"TASK:\n{task}\n\nDEBATE:\n{debate}\n\nDistinct options:"
    raw = await agent.backend.chat(agent.model, system, user, temperature=0.3)
    opts = [line.strip("-•* \t") for line in raw.splitlines() if line.strip()]
    return opts[:4] if opts else ["Proceed", "Do not proceed"]
