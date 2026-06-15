"""Robust vote parsing.

Local models often wrap JSON in markdown, emit extra prose, or paraphrase the
option text. This snaps those responses back to a valid option, with graceful
degradation.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import json
import logging
import re


UNPARSEABLE_VOTE_REASON = "(unparseable vote, defaulted)"

log = logging.getLogger(__name__)


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
    if isinstance(choice, dict):
        for key in ("choice", "option", "title", "label", "name", "text"):
            value = choice.get(key)
            if value:
                choice = value
                break
        else:
            choice = next(
                (value for value in choice.values() if isinstance(value, str) and value.strip()),
                "",
            )

    text = str(choice).strip()
    text = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", text)
    return text.strip().strip(",;").strip().strip("\"'`").strip()


def _snap_choice(choice: str, options: list[str]) -> str | None:
    normalized = choice.casefold().strip()

    # Exact match
    for option in options:
        if normalized == option.casefold():
            return option

    # Positional letter/label: "A", "B", "option a", "option 1", etc.
    _label_map = {
        letter: i
        for i, letter in enumerate("abcdefghijklmnopqrstuvwxyz")
        if i < len(options)
    }
    _label_map.update({str(i + 1): i for i in range(len(options))})
    bare = re.sub(r"^option\s*", "", normalized).strip().rstrip(".:)")
    if bare in _label_map and _label_map[bare] < len(options):
        return options[_label_map[bare]]

    # Substring containment
    for option in options:
        option_normalized = option.casefold()
        if option_normalized in normalized or normalized in option_normalized:
            return option

    # Fuzzy similarity — lower threshold so paraphrases don't fall through
    best_option = None
    best_score = 0.0
    for option in options:
        score = SequenceMatcher(None, normalized, option.casefold()).ratio()
        if score > best_score:
            best_option = option
            best_score = score

    log.debug("snap_choice: %r → best=%r score=%.3f", choice, best_option, best_score)
    return best_option if best_score >= 0.35 else None


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
    reason_from_json = None
    if data is not None:
        raw_choice = _clean_choice(data.get("choice", ""))
        reason_from_json = str(data.get("reason", ""))
        snapped = _snap_choice(raw_choice, options)
        log.debug(
            "parse_vote JSON: raw_choice=%r → snapped=%r (options=%r)",
            raw_choice, snapped, options,
        )
        if snapped:
            return {"choice": snapped, "reason": reason_from_json}
        # JSON parsed but choice didn't snap — fall through to freeform so we
        # still use the reason field if freeform succeeds.

    raw_normalized = raw.casefold()
    for option in options:  # scan freeform text for verbatim option
        if option.casefold() in raw_normalized:
            log.debug("parse_vote freeform hit: %r", option)
            return {
                "choice": option,
                "reason": reason_from_json or "(parsed from freeform text)",
            }

    # Last resort: scan for "option A/B/C/1/2/3" patterns in the raw text.
    # Take the LAST match — the agent's conclusion typically appears at the end
    # ("I considered option A but ultimately choose option C").
    last_snapped = None
    for m in re.finditer(r"\boption\s*([a-zA-Z]|\d+)\b", raw, re.IGNORECASE):
        label = m.group(1).casefold()
        candidate = _snap_choice(label, options)
        if candidate:
            last_snapped = candidate
    if last_snapped:
        log.debug("parse_vote label scan hit (last): %r", last_snapped)
        return {
            "choice": last_snapped,
            "reason": reason_from_json or "(parsed from label scan)",
        }

    # Truly unparseable — signal the repair path; do NOT fabricate a vote for
    # options[0]. choice=None is recorded as an abstention if repair also fails.
    log.warning("parse_vote: unparseable — raw=%r", raw[:200])
    return {"choice": None, "reason": UNPARSEABLE_VOTE_REASON}
