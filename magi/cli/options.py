"""Derive vote options from a debate when the user didn't supply any."""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

# Matches "OPTION A:", "Option B:", "A:", "B:" at the start of a line (with
# optional whitespace). Captures the label letter and the rest of the line.
_EXPLICIT_OPTION_RE = re.compile(
    r"^\s*(?:option\s+)?([A-Za-z])\s*[:.)-]\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_explicit_options(task: str) -> list[str] | None:
    """Return verbatim options when the task already enumerates them.

    Detects patterns like:
      OPTION A: ...
      OPTION B: ...
      Option C: ...
      A) ...
      B) ...

    Returns None if fewer than 2 are found (fall through to LLM derivation).
    """
    matches = _EXPLICIT_OPTION_RE.findall(task)
    if len(matches) < 2:
        return None

    # Keep only consecutive alpha labels starting from the first one found
    first_label = matches[0][0].upper()
    start_ord = ord(first_label)
    options: list[str] = []
    for i, (label, text) in enumerate(matches):
        if ord(label.upper()) == start_ord + i:
            options.append(text.strip())
        else:
            break  # gap in sequence — stop; remaining may be sub-items
    return options if len(options) >= 2 else None

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

_META_MARKERS = (
    "re-evaluate",
    "reevaluate",
    "reassess",
    "reconsider",
    "revisit",
    "gather more information",
    "more data",
    "wait and see",
    "it depends",
    "circle back",
    "table the decision",
)

_META_PREFIXES = (
    "let's",
    "lets",
    "let us",
)


def _whole_word_marker_pattern(marker: str) -> re.Pattern:
    escaped = re.escape(marker).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


_META_PATTERNS = tuple(_whole_word_marker_pattern(marker) for marker in _META_MARKERS)


def _extract_option_value(value: object) -> object:
    if not isinstance(value, dict):
        return value

    for key in ("option", "choice", "title", "label", "name", "text"):
        option = value.get(key)
        if option:
            return option

    for option in value.values():
        if isinstance(option, str) and option.strip():
            return option

    return ""


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
    value = _extract_option_value(value)
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


def _primary_text(option: str) -> str:
    text = option.strip()
    text = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", text)
    text = text.strip().strip("\"'`").strip()
    lowered = text.casefold()
    for prefix in _META_PREFIXES:
        if lowered.startswith(prefix):
            return text[len(prefix):].lstrip(" '").strip()
    return text


def _is_action(option: str) -> bool:
    """Reject options whose primary verb is deferral instead of an action."""

    primary = _primary_text(option)
    if not primary:
        return False

    first_clause = re.split(r"[,;:—–]|\b(?:after|once|when|if|unless)\b", primary, maxsplit=1)[0]
    first_clause = first_clause.strip()
    return not any(pattern.search(first_clause) for pattern in _META_PATTERNS)


def _filter_actions(options: list[str], max_options: int) -> list[str]:
    actions = [option for option in options if _is_action(option)]
    return actions[:max_options]


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

    # If the task already enumerates explicit options (OPTION A/B/C), use them
    # verbatim — no LLM call needed, and context cannot contaminate them.
    explicit = _extract_explicit_options(task)
    if explicit:
        log.debug("derive_options: using %d explicit options from task", len(explicit))
        return explicit[:max_options]

    system = (
        "You derive voting options from a council debate.\n"
        f"Return ONLY a JSON array of 2 to {max_options} short option strings. "
        "No markdown, no explanation.\n"
        "Each option must be a concrete, mutually exclusive ACTION the voter can choose now. "
        "Phrase each option as an imperative "
        "(e.g. \"Launch the product now\" or \"Delay launch until Q2\").\n"
        "CRITICAL: derive options ONLY from the TASK and DEBATE below. "
        "Never invent details from any background context. "
        "FORBIDDEN: meta-options or refusals to choose, including re-evaluate, reevaluate, "
        "reassess, reconsider, revisit later, gather more information, more data, wait and see, "
        "it depends, circle back, or table the decision."
    )

    # Deliberately omit `context` — personal background must not bleed into
    # option labels, which become canonical strings matched against agent votes.
    debate = "\n".join(f"[{m['name']}]: {m['content']}" for m in transcript)

    user = (
        f"TASK:\n{task}\n\n"
        f"DEBATE:\n{debate}\n\n"
        f"Derive 2 to {max_options} distinct, immediately-actionable voting options "
        "from the TASK and DEBATE only. "
        "Use concrete imperatives only. Do not include any option whose main action is to delay, "
        "re-evaluate, gather more information, or revisit later."
    )

    raw = await agent.backend.chat(
        agent.model,
        system,
        user,
        temperature=0.2,
    )

    options = _filter_actions(parse_options(raw, max_options), max_options)
    if not options:
        return ["Proceed", "Do not proceed"][:max_options]

    return options
