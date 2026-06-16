"""MCP server exposing the local MAGI council over stdio."""

from __future__ import annotations

import os
import traceback
from typing import Any

from mcp.server.fastmcp import FastMCP

from magi.agents import get_council
from magi.cli.runner import TALLY_CHOICES, default_args, run_council
from magi.llm import get_backend


mcp = FastMCP("magi-council")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.casefold() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _mcp_args(question: str, context: str, rounds: int, vote: bool):
    tally = os.environ.get("MAGI_TALLY", "majority")
    if tally not in TALLY_CHOICES:
        tally = "majority"

    return default_args(
        task=question,
        context=context,
        rounds=max(1, rounds),
        council=os.environ.get("MAGI_COUNCIL", "magi"),
        model=os.environ.get("MAGI_MODEL", "qwen3:14b"),
        model_secondary=os.environ.get("MAGI_MODEL_SECONDARY", "qwen3:8b"),
        small_gpu_vram_mib_threshold=_env_int("MAGI_SMALL_GPU_VRAM_MIB", 14_000),
        backend=os.environ.get("MAGI_BACKEND", "ollama"),
        host=os.environ.get("MAGI_HOST", "http://localhost:11434"),
        auto_instances=_env_bool("MAGI_AUTO_INSTANCES", True),
        auto_spawn_ollama=_env_bool("MAGI_AUTO_SPAWN_OLLAMA", True),
        ollama_command=os.environ.get("MAGI_OLLAMA_COMMAND") or None,
        ollama_startup_timeout=float(os.environ.get("MAGI_OLLAMA_STARTUP_TIMEOUT", "30.0")),
        scan_ports=_env_int("MAGI_SCAN_PORTS", 8),
        assignment=os.environ.get("MAGI_ASSIGNMENT", "pooled"),
        max_options=_env_int("MAGI_MAX_OPTIONS", 3),
        tally=tally,
        no_synthesis=False,
        no_vote=not vote,
        tui=False,
    )


def _excerpt(text: object, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _compact_transcript(
    transcript: list[dict],
    *,
    max_turn_chars: int = 700,
    max_turns: int = 72,
) -> dict[str, Any]:
    turns = transcript[:max_turns]
    omitted = max(0, len(transcript) - len(turns))
    rounds: dict[int, dict[str, list[dict]]] = {}

    for turn in turns:
        round_number = int(turn.get("round", 0))
        phase = str(turn.get("phase", "turn"))
        rounds.setdefault(round_number, {}).setdefault(phase, []).append(
            {
                "agent": turn.get("name"),
                "excerpt": _excerpt(turn.get("content"), max_turn_chars),
                "chars": len(str(turn.get("content") or "")),
            }
        )

    return {
        "rounds": [
            {"round": round_number, "phases": phases}
            for round_number, phases in sorted(rounds.items())
        ],
        "turn_count": len(transcript),
        "omitted_turns": omitted,
        "max_turn_chars": max_turn_chars,
    }


def _vote_payload(options: list[str] | None, vote_result: dict | None) -> dict | None:
    if vote_result is None:
        return None
    return {
        "options": options or [],
        "scores": vote_result.get("scores", {}),
        "winner": vote_result.get("winner"),
        "tie_between": vote_result.get("tie_between"),
        "tie_break": vote_result.get("tie_break"),
        "ballots": vote_result.get("ballots", []),
    }


@mcp.tool()
async def consult_council(
    question: str,
    context: str = "",
    rounds: int = 3,
    vote: bool = True,
) -> dict:
    """Ask the local MAGI council to deliberate, synthesize, and optionally vote."""

    if not question.strip():
        return {"ok": False, "error": "question is required"}

    events: list[dict] = []

    try:
        args = _mcp_args(question.strip(), context or "", rounds, vote)
        result = await run_council(
            args,
            on_event=lambda kind, data: events.append({"kind": kind, "data": data}),
        )
        return {
            "ok": True,
            "synthesis": result.synthesis,
            "vote": _vote_payload(result.options, result.vote_result),
            "transcript_summary": _compact_transcript(result.transcript),
            "agents": result.agents,
            "warnings": result.warnings,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "error_type": exc.__class__.__name__,
            "hint": "Check that Ollama is running, the configured model is pulled, and MAGI_* env vars match your CLI setup.",
            "events": events[-20:],
            "traceback": traceback.format_exc(limit=5),
        }


@mcp.tool()
async def list_council() -> dict:
    """Return the configured MAGI council members and persona descriptions."""

    try:
        backend = get_backend(
            os.environ.get("MAGI_BACKEND", "ollama"),
            host=os.environ.get("MAGI_HOST", "http://localhost:11434"),
        )
        council = get_council(
            os.environ.get("MAGI_COUNCIL", "magi"),
            backend,
            model=os.environ.get("MAGI_MODEL", "qwen3:14b"),
        )
        return {
            "ok": True,
            "council": os.environ.get("MAGI_COUNCIL", "magi"),
            "agents": [
                {
                    "name": agent.name,
                    "persona": agent.persona,
                    "domains": agent.domains,
                    "weight": agent.weight,
                    "model": agent.model,
                }
                for agent in council
            ],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "error_type": exc.__class__.__name__}


if __name__ == "__main__":
    mcp.run(transport="stdio")
