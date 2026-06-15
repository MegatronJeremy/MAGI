"""Derive vote options from a debate when the user didn't supply any."""

from __future__ import annotations


async def derive_options(
        agent,
        task: str,
        transcript: list[dict],
        context: str = "",
        max_options: int = 3,
) -> list[str]:
    max_options = max(2, max_options)

    system = (
        "You derive voting options from a council debate.\n"
        "Return ONLY a JSON array of short option strings. No markdown, no explanation."
    )

    ctx_block = f"BACKGROUND CONTEXT:\n{context}\n\n" if context else ""
    debate = "\n".join(f"[{m['name']}]: {m['content']}" for m in transcript)

    user = (
        f"{ctx_block}"
        f"TASK:\n{task}\n\n"
        f"DEBATE:\n{debate}\n\n"
        f"Derive 2 to {max_options} distinct voting options from the debate."
    )

    raw = await agent.backend.chat(
        agent.model,
        system,
        user,
        temperature=0.2,
    )

    import json

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        options = [
            line.strip("- ").strip()
            for line in raw.splitlines()
            if line.strip()
        ]
        return options[:max_options]

    if not isinstance(parsed, list):
        raise ValueError("derive_options expected the model to return a JSON list")

    options = [str(option).strip() for option in parsed if str(option).strip()]
    return options[:max_options]
