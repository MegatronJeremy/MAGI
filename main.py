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
from magi.council import Council, MajorityVote, WeightedVote
from magi.llm import get_backend

from .options import derive_options

TALLIES = {"majority": MajorityVote, "weighted": WeightedVote}


def _printer(kind: str, data: dict) -> None:
    if kind == "turn":
        print(f"\n── Round {data['round']} · {data['name']} ──\n{data['content']}")
    elif kind == "ballot":
        print(f"\n🗳  {data['voter']} → {data['choice']}  ({data['reason']})")
    elif kind == "result":
        print(f"\n{'='*60}\nSCORES: {data['scores']}")
        if data["winner"]:
            print(f"DECISION: {data['winner']}")
        else:
            print(f"DEADLOCK between: {data['tie_between']}")
            print("(MAGI would escalate to a human operator here.)")
        print("=" * 60)


async def run(args):
    backend = get_backend(args.backend, host=args.host)
    agents = get_council(args.council, backend, model=args.model)
    council = Council(
        agents,
        rounds=args.rounds,
        tally=TALLIES[args.tally](),
        on_event=_printer,
    )

    print(f"\n{'='*60}\nTASK: {args.task}\n{'='*60}")
    transcript = await council.deliberate(args.task)

    options = args.options
    if not options:
        print("\n— deriving options from the debate —")
        options = await derive_options(agents[0], args.task, transcript)
    print(f"\nVOTING OPTIONS: {options}")

    await council.vote(args.task, transcript, options)


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
    p.add_argument("--tally", choices=list(TALLIES), default="majority",
                   help="Vote tally strategy")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
