"""CLI helpers for backend pool configuration."""

from __future__ import annotations

import argparse
from urllib.parse import urlparse
from typing import Callable

from magi.llm import BackendPool


DEFAULT_SCAN_PORTS = 8


def _auto_ollama_entries(host: str, model: str, count: int) -> list[dict]:
    if "://" not in host:
        host = f"http://{host}"
    parsed = urlparse(host)
    scheme = parsed.scheme or "http"
    hostname = parsed.hostname or "localhost"
    start_port = parsed.port or 11434
    return [
        {
            "name": f"ollama-{port}",
            "host": f"{scheme}://{hostname}:{port}",
            "model": model,
        }
        for port in range(start_port, start_port + max(1, count))
    ]


async def build_backend_pool(
    args: argparse.Namespace,
    warn: Callable[[str], None] | None = None,
) -> BackendPool:
    if args.backend == "ollama" and args.auto_instances:
        pool = BackendPool.from_entries(
            _auto_ollama_entries(args.host, args.model, args.scan_ports),
            assignment=args.assignment,
            default_backend=args.backend,
            default_model=args.model,
            warn=warn,
        )
        live_count = await pool.health_check(warn_dead=False, warn_empty=False)
        if live_count:
            return pool
        warn = warn or print
        warn(f"WARNING no Ollama servers found while scanning from {args.host}; using --host only")

    pool = BackendPool.default_local(
        backend_name=args.backend,
        host=args.host,
        model=args.model,
        assignment=args.assignment,
        warn=warn,
    )
    await pool.health_check()
    return pool


def route_agents_for_downstream(agents: list, pool: BackendPool) -> None:
    """Assign stable backends/models for synthesis helpers, option derivation, and voting."""

    for index, agent in enumerate(agents):
        instance = pool.backend_for(agent, index)
        agent.backend = instance.backend
        if instance.model:
            agent.model = instance.model
