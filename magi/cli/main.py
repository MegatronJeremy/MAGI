"""MAGI council CLI.

    python -m magi.cli.main "Should we rewrite the renderer in Rust or stay in C++?"

    python -m magi.cli.main "Pick a database" \
        --rounds 3 --options "PostgreSQL" "SQLite" "DuckDB" \
        --tally weighted
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import logging

from .runner import TALLY_CHOICES, run_council
from .pool_config import (
    DEFAULT_MANAGED_PORT_BASE,
    DEFAULT_OLLAMA_STARTUP_TIMEOUT,
    DEFAULT_SCAN_PORTS,
)


def _display_text(value: object) -> str:
    if isinstance(value, dict):
        for key in ("choice", "option", "title", "label", "name", "text"):
            option = value.get(key)
            if option:
                return str(option)
        for option in value.values():
            if isinstance(option, str) and option.strip():
                return option
    if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
        if isinstance(parsed, dict):
            return _display_text(parsed)
    return str(value)


def _printer(kind: str, data: dict) -> None:
    if kind == "turn":
        phase = data.get("phase", "turn").upper()
        print(f"\n--- Round {data['round']} | {phase} | {data['name']} ---\n{data['content']}")
    elif kind == "phase":
        print(f"\n>>> Round {data['round']} | {data['phase'].upper()}")
    elif kind == "error":
        print(f"\nWARNING: {data['message']}")
    elif kind == "warning":
        print(f"\n{data['message']}")
    elif kind == "context":
        print(f"(using {data['chars']} chars of background context)")
    elif kind == "options" and data.get("status") == "deriving":
        print("\nDeriving options from the debate...")
    elif kind == "options" and data.get("status") == "ready":
        print("\nVOTING OPTIONS:")
        for index, option in enumerate(data["options"], start=1):
            print(f"  {index}. {_display_text(option)}")
    elif kind == "ballot":
        print(f"\nBALLOT: {data['voter']}")
        print(f"  Choice: {_display_text(data['choice'])}")
        print(f"  Reason: {data['reason']}")
    elif kind == "synthesis":
        print(f"\n{'=' * 60}\nSYNTHESIS (neutral scribe)\n{'=' * 60}\n{data['text']}")
    elif kind == "result":
        print(f"\n{'=' * 60}\nRESULT")
        print("Scores:")
        for option, score in data["scores"].items():
            marker = " (winner)" if option == data.get("winner") else ""
            print(f"  - {_display_text(option)}: {score:g}{marker}")
        if data.get("tie_break"):
            tb = data["tie_break"]
            print(f"Tie break: consul {tb['consul']} chose among {', '.join(tb['among'])}")
        if data["winner"]:
            print(f"DECISION: {data['winner']}")
        else:
            print("Deadlock between:")
            for option in data["tie_between"]:
                print(f"  - {_display_text(option)}")
            print("(no consul configured; escalate to a human operator here.)")
        print("=" * 60)


async def run(args):
    print(f"\n{'=' * 60}\nTASK: {args.task}\n{'=' * 60}")
    await run_council(args, on_event=_printer, warn=print)


def main():
    p = argparse.ArgumentParser(description="Local multi-agent MAGI council")
    p.add_argument("task", help="The question/problem for the council")
    p.add_argument("--rounds", type=int, default=3, help="Debate rounds (default 3)")
    p.add_argument("--options", nargs="+", default=None,
                   help="Explicit vote options; derived from debate if omitted")
    p.add_argument("--council", default="magi", help="Council preset (default: magi)")
    p.add_argument("--model", default="llama3.1:8b", help="Ollama model tag")
    p.add_argument("--backend", default="ollama", help="LLM backend (default: ollama)")
    p.add_argument("--host", default="http://localhost:11434", help="Backend host URL")
    p.add_argument("--auto-instances", dest="auto_instances", action="store_true", default=True,
                   help="Auto-discover local Ollama instances from --host upward (default)")
    p.add_argument("--no-auto-instances", dest="auto_instances", action="store_false",
                   help="Disable Ollama port discovery and use only --host")
    p.add_argument("--auto-spawn-ollama", dest="auto_spawn_ollama", action="store_true", default=True,
                   help="Auto-spawn missing local Ollama servers for detected GPUs (default)")
    p.add_argument("--no-auto-spawn-ollama", dest="auto_spawn_ollama", action="store_false",
                   help="Disable automatic Ollama server spawning")
    p.add_argument("--ollama-command", default=None,
                   help="Path to the ollama executable for auto-spawn")
    p.add_argument("--ollama-startup-timeout", type=float, default=DEFAULT_OLLAMA_STARTUP_TIMEOUT,
                   help=f"Seconds to wait for spawned Ollama servers (default {DEFAULT_OLLAMA_STARTUP_TIMEOUT:g})")
    p.add_argument("--scan-ports", type=int, default=DEFAULT_SCAN_PORTS,
                   help=f"Number of managed ports to scan/spawn (default {DEFAULT_SCAN_PORTS})")
    p.add_argument("--managed-port-base", type=int, default=DEFAULT_MANAGED_PORT_BASE,
                   dest="managed_port_base",
                   help=f"Base port for MAGI-managed Ollama servers (default {DEFAULT_MANAGED_PORT_BASE})")
    p.add_argument("--backends", nargs="+", default=None, metavar="HOST",
                   help="Explicit backend URLs (e.g. localhost:11434 localhost:11435); "
                        "skips auto-discovery and auto-spawn entirely")
    p.add_argument("--assignment", choices=["round_robin", "pinned", "pooled"], default="pooled",
                   help="Backend assignment policy (default: pooled)")
    p.add_argument("--max-options", type=int, default=3,
                   help="Cap on derived vote options (default 3; keep <= agent count)")
    p.add_argument("--context", default=None,
                   help="Background context about the asker, injected into prompts")
    p.add_argument("--context-file", default=None,
                   help="Path to a text file with background context (overrides --context)")
    p.add_argument("--tally", choices=TALLY_CHOICES, default="consul",
                   help="Vote tally: consul (default, rotating tie-breaker) | majority | weighted")
    p.add_argument("--no-synthesis", action="store_true",
                   help="Skip the neutral synthesis step")
    p.add_argument("--no-vote", action="store_true",
                   help="Skip voting; produce only the synthesis (recommended for nuanced questions)")
    p.add_argument("--debug", action="store_true",
                   help="Enable DEBUG logging (shows vote snap decisions)")
    p.add_argument("--tui", dest="tui", action="store_true", default=True,
                   help="Launch the live MAGI terminal UI (default)")
    p.add_argument("--no-tui", dest="tui", action="store_false",
                   help="Use plain-text terminal output instead of the TUI")
    args = p.parse_args()
    if getattr(args, "debug", False):
        logging.basicConfig(level=logging.DEBUG, format="[%(name)s] %(message)s")
    if args.tui:
        from magi.tui.app import run_tui

        asyncio.run(run_tui(args))
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
