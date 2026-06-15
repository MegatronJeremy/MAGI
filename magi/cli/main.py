"""MAGI council CLI.

    python -m magi.cli.main "Should we rewrite the renderer in Rust or stay in C++?"

    python -m magi.cli.main "Pick a database" \
        --rounds 3 --options "PostgreSQL" "SQLite" "DuckDB" \
        --tally weighted
"""

from __future__ import annotations

import argparse
import asyncio

from magi.agents import get_council
from magi.council import Council, ConsulTieBreaker, MajorityVote, WeightedVote
from magi.llm import get_backend
from .options import derive_options

# Base tallies selectable directly; "consul" is built after agents exist.
BASE_TALLIES = {"majority": MajorityVote, "weighted": WeightedVote}
TALLY_CHOICES = list(BASE_TALLIES) + ["consul"]


def _printer(kind: str, data: dict) -> None:
    if kind == "turn":
        print(f"\n── Round {data['round']} · {data['name']} ──\n{data['content']}")
    elif kind == "ballot":
        print(f"\n🗳  {data['voter']} → {data['choice']}  ({data['reason']})")
    elif kind == "result":
        print(f"\n{'=' * 60}\nSCORES: {data['scores']}")
        if data.get("tie_break"):
            tb = data["tie_break"]
            print(f"TIE among {len(tb['among'])} options → broken by consul {tb['consul']}")
        if data["winner"]:
            print(f"DECISION: {data['winner']}")
        else:
            print(f"DEADLOCK between: {data['tie_between']}")
            print("(no consul configured — escalate to a human operator here.)")
        print("=" * 60)


async def run(args):
    backend = get_backend(args.backend, host=args.host)
    agents = get_council(args.council, backend, model=args.model)

    if args.tally == "consul":
        order = [a.name for a in agents]  # rotation order = council order
        tally = ConsulTieBreaker(MajorityVote(), consul_order=order)
    else:
        tally = BASE_TALLIES[args.tally]()

    council = Council(
        agents,
        rounds=args.rounds,
        tally=tally,
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

    options = args.options
    if not options:
        print("\n— deriving options from the debate —")
        options = await derive_options(agents[0], args.task, transcript, context=context)
    print(f"\nVOTING OPTIONS: {options}")

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
    p.add_argument("--context", default=None,
                   help="Background context about the asker, injected into prompts")
    p.add_argument("--context-file", default=None,
                   help="Path to a text file with background context (overrides --context)")
    p.add_argument("--tally", choices=TALLY_CHOICES, default="consul",
                   help="Vote tally: majority | weighted | consul (rotating tie-breaker)")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
