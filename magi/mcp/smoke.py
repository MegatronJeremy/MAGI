"""Smoke-test the MAGI MCP tool handler without Claude."""

from __future__ import annotations

import argparse
import asyncio
import json

from .server import consult_council, list_council


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test MAGI MCP tools")
    parser.add_argument("question", nargs="?", default="Should we use SQLite or PostgreSQL?")
    parser.add_argument("--context", default="")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--no-vote", action="store_true")
    parser.add_argument("--list", action="store_true", help="Call list_council instead")
    args = parser.parse_args()

    if args.list:
        result = await list_council()
    else:
        result = await consult_council(
            args.question,
            context=args.context,
            rounds=args.rounds,
            vote=not args.no_vote,
        )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(_main())
