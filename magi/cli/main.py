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

from magi.agents import get_council
from magi.council import (
    Council,
    ConsulTieBreaker,
    MajorityVote,
    Synthesizer,
    WeightedVote,
)
from magi.llm import get_backend
from .options import derive_options

# Base tallies selectable directly; "consul" is built after agents exist.
BASE_TALLIES = {"majority": MajorityVote, "weighted": WeightedVote}
TALLY_CHOICES = list(BASE_TALLIES) + ["consul"]


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
        print(f"\n--- Round {data['round']} | {data['name']} ---\n{data['content']}")
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
    backend = get_backend(args.backend, host=args.host)
    agents = get_council(args.council, backend, model=args.model)

    if args.tally == "consul":
        order = [a.name for a in agents]  # rotation order = council order
        tally = ConsulTieBreaker(MajorityVote(), consul_order=order)
    else:
        tally = BASE_TALLIES[args.tally]()

    synthesizer = None
    if not args.no_synthesis:
        synthesizer = Synthesizer(backend, model=args.model)

    council = Council(
        agents,
        rounds=args.rounds,
        tally=tally,
        synthesizer=synthesizer,
        on_event=_printer,
    )

    print(f"\n{'=' * 60}\nTASK: {args.task}\n{'=' * 60}")

    context = args.context or ""
    if args.context_file:
        from pathlib import Path
        context = Path(args.context_file).read_text(encoding="utf-8")
    if context:
        print(f"(using {len(context)} chars of background context)")

    transcript = await council.deliberate(args.task, context=context)

    # Synthesis first; it's the substantive output. The vote is an optional,
    # lossy summary on top of it.
    if not args.no_synthesis:
        await council.synthesize(args.task, transcript, context=context)

    if args.no_vote:
        return

    options = args.options
    if not options:
        print("\nDeriving options from the debate...")
        options = await derive_options(
            agents[0], args.task, transcript, context=context, max_options=args.max_options
        )
    print("\nVOTING OPTIONS:")
    for index, option in enumerate(options, start=1):
        print(f"  {index}. {_display_text(option)}")

    await council.vote(args.task, transcript, options, context=context)


def main():
    p = argparse.ArgumentParser(description="Local multi-agent MAGI council")
    p.add_argument("task", help="The question/problem for the council")
    p.add_argument("--rounds", type=int, default=2, help="Debate rounds (default 2)")
    p.add_argument("--options", nargs="+", default=None,
                   help="Explicit vote options; derived from debate if omitted")
    p.add_argument("--council", default="magi", help="Council preset (default: magi)")
    p.add_argument("--model", default="llama3.1:8b", help="Ollama model tag")
    p.add_argument("--backend", default="ollama", help="LLM backend (default: ollama)")
    p.add_argument("--host", default="http://localhost:11434", help="Backend host URL")
    p.add_argument("--max-options", type=int, default=3,
                   help="Cap on derived vote options (default 3; keep <= agent count)")
    p.add_argument("--context", default=None,
                   help="Background context about the asker, injected into prompts")
    p.add_argument("--context-file", default=None,
                   help="Path to a text file with background context (overrides --context)")
    p.add_argument("--tally", choices=TALLY_CHOICES, default="majority",
                   help="Vote tally: majority | weighted | consul (rotating tie-breaker)")
    p.add_argument("--no-synthesis", action="store_true",
                   help="Skip the neutral synthesis step")
    p.add_argument("--no-vote", action="store_true",
                   help="Skip voting; produce only the synthesis (recommended for nuanced questions)")
    p.add_argument("--tui", dest="tui", action="store_true", default=True,
                   help="Launch the live MAGI terminal UI (default)")
    p.add_argument("--no-tui", dest="tui", action="store_false",
                   help="Use plain-text terminal output instead of the TUI")
    args = p.parse_args()
    if args.tui:
        from magi.tui.app import run_tui

        asyncio.run(run_tui(args))
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
