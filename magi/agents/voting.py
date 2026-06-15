"""Robust vote parsing.

Local models often wrap JSON in markdown, emit extra prose, or paraphrase the
option text. This snaps those responses back to a valid option, with graceful
degradation.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import json
import re


UNPARSEABLE_VOTE_REASON = "(unparseable vote, defaulted)"


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

    start = stripped.find("{")
    end = stripped.rfind("}")
    if 0 <= start < end:
        candidates.append(stripped[start:end + 1])

    return [candidate for candidate in candidates if candidate]


def _clean_choice(choice: object) -> str:
    text = str(choice).strip()
    text = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", text)
    return text.strip().strip(",;").strip().strip("\"'`").strip()


def _snap_choice(choice: str, options: list[str]) -> str | None:
    normalized = choice.casefold()
    for option in options:
        if normalized == option.casefold():
            return option
    for option in options:
        option_normalized = option.casefold()
        if option_normalized in normalized or normalized in option_normalized:
            return option

    best_option = None
    best_score = 0.0
    for option in options:
        score = SequenceMatcher(None, normalized, option.casefold()).ratio()
        if score > best_score:
            best_option = option
            best_score = score
    return best_option if best_score >= 0.72 else None


def _load_vote_json(raw: str) -> dict | None:
    decoder = json.JSONDecoder()
    for candidate in _json_candidates(raw):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_vote(raw: str, options: list[str]) -> dict:
    data = _load_vote_json(raw)
    if data is not None:
        choice = _clean_choice(data.get("choice", ""))
        snapped = _snap_choice(choice, options)
        if snapped:
            return {"choice": snapped, "reason": str(data.get("reason", ""))}

    raw_normalized = raw.casefold()
    for option in options:  # last resort: scan freeform text
        if option.casefold() in raw_normalized:
            return {"choice": option, "reason": "(parsed from freeform text)"}

    return {"choice": options[0], "reason": UNPARSEABLE_VOTE_REASON}
