"""Derive vote options from a debate when the user didn't supply any."""

from __future__ import annotations

import json
import re


_JUNK_OPTIONS = {
    "",
    "`",
    "```",
    "```json",
    "json",
    "[",
    "]",
    "{",
    "}",
    ",",
}


def _without_code_fences(text: str) -> str:
    lines = [
        line
        for line in text.strip().splitlines()
        if not line.strip().startswith("```")
    ]
    return "\n".join(lines).strip()


def _json_candidates(raw: str) -> list[str]:
    stripped = _without_code_fences(raw)
    candidates = [raw.strip(), stripped]

    start = stripped.find("[")
    end = stripped.rfind("]")
    if 0 <= start < end:
        candidates.append(stripped[start:end + 1])

    return [candidate for candidate in candidates if candidate]


def _clean_option(value: object) -> str | None:
    option = str(value).strip()
    option = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", option)
    option = option.strip().strip(",;").strip().strip("\"'`").strip()
    option = option.strip(",;").strip()

    if option.casefold() in _JUNK_OPTIONS:
        return None
    if not any(char.isalnum() for char in option):
        return None
    return option


def _dedupe(options: list[str]) -> list[str]:
    seen = set()
    unique = []
    for option in options:
        key = option.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(option)
    return unique


def parse_options(raw: str, max_options: int) -> list[str]:
    """Parse model-derived options while tolerating common markdown wrapping."""

    for candidate in _json_candidates(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            parsed = parsed.get("options", [])
        if isinstance(parsed, list):
            options = [_clean_option(option) for option in parsed]
            clean = _dedupe([option for option in options if option])
            if clean:
                return clean[:max_options]

    options = []
    for line in _without_code_fences(raw).splitlines():
        cleaned = _clean_option(line)
        if cleaned:
            options.append(cleaned)

    return _dedupe(options)[:max_options]


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

    options = parse_options(raw, max_options)
    if not options:
        excerpt = raw.replace("\n", " ")[:120]
        raise ValueError(f"derive_options could not parse vote options from model output: {excerpt}")

    return options
