"""Robust vote parsing.

Local models often wrap JSON in markdown or paraphrase the option text.
This snaps any of those back to a valid option, with graceful degradation.
"""

from __future__ import annotations

import json


def parse_vote(raw: str, options: list[str]) -> dict:
    text = raw.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(text)
        choice = data.get("choice", "")
        if choice in options:
            return {"choice": choice, "reason": data.get("reason", "")}
        for o in options:  # model paraphrased — snap to closest option
            if o.lower() in choice.lower() or choice.lower() in o.lower():
                return {"choice": o, "reason": data.get("reason", "")}
    except json.JSONDecodeError:
        pass
    for o in options:  # last resort: scan freeform text
        if o.lower() in raw.lower():
            return {"choice": o, "reason": "(parsed from freeform text)"}
    return {"choice": options[0], "reason": "(unparseable vote, defaulted)"}
