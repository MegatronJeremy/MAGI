"""Shared council execution path for CLI, TUI, and MCP."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from magi.agents import Agent, get_council
from magi.council import (
    ConsulTieBreaker,
    Council,
    MajorityVote,
    Synthesizer,
    WeightedVote,
)
from .options import derive_options
from .pool_config import (
    DEFAULT_OLLAMA_STARTUP_TIMEOUT,
    DEFAULT_SCAN_PORTS,
    build_backend_pool,
    route_agents_for_downstream,
)

BASE_TALLIES = {"majority": MajorityVote, "weighted": WeightedVote}
TALLY_CHOICES = list(BASE_TALLIES) + ["consul"]


@dataclass(frozen=True)
class CouncilRunResult:
    task: str
    context: str
    rounds: int
    council: str
    model: str
    backend: str
    agents: list[dict]
    transcript: list[dict]
    synthesis: str | None
    options: list[str] | None
    vote_result: dict | None
    events: list[dict]
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def default_args(**overrides) -> argparse.Namespace:
    """Build the same configuration shape argparse gives the CLI."""

    values = {
        "task": "",
        "rounds": 3,
        "options": None,
        "council": "magi",
        "model": "qwen3:14b",
        "model_secondary": "qwen3:8b",
        "small_gpu_vram_mib_threshold": 14_000,
        "backend": "ollama",
        "host": "http://localhost:11434",
        "auto_instances": True,
        "auto_spawn_ollama": True,
        "ollama_command": None,
        "ollama_startup_timeout": 30.0,
        "scan_ports": DEFAULT_SCAN_PORTS,
        "assignment": "pooled",
        "max_options": 3,
        "context": None,
        "context_file": None,
        "tally": "consul",
        "no_synthesis": False,
        "no_vote": False,
        "tui": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def load_context(args: argparse.Namespace) -> str:
    context = args.context or ""
    if args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8")
    return context


def agent_metadata(agents: list[Agent]) -> list[dict]:
    return [
        {
            "name": agent.name,
            "persona": agent.persona,
            "domains": list(agent.domains),
            "weight": agent.weight,
            "model": agent.model,
            "instance": agent.instance,
        }
        for agent in agents
    ]


async def run_council(
        args: argparse.Namespace,
        *,
        on_event: Callable[[str, dict], None] | None = None,
        warn: Callable[[str], None] | None = None,
) -> CouncilRunResult:
    """Run the complete MAGI pipeline using the same assembly everywhere."""

    events: list[dict] = []
    warnings: list[str] = []

    def emit(kind: str, data: dict) -> None:
        events.append({"kind": kind, "data": data})
        if on_event:
            on_event(kind, data)

    def capture_warning(message: str) -> None:
        warnings.append(message)
        emit("warning", {"message": message})
        if warn and on_event is None:
            warn(message)

    pool = await build_backend_pool(args, warn=capture_warning)
    primary = pool.primary_instance()
    backend = primary.backend
    agents = get_council(args.council, backend, model=args.model)
    route_agents_for_downstream(agents, pool)
    agents_info = agent_metadata(agents)
    emit("agents", {"names": [agent["name"] for agent in agents_info], "agents": agents_info})

    if args.tally == "consul":
        order = [agent.name for agent in agents]
        tally = ConsulTieBreaker(MajorityVote(), consul_order=order)
    else:
        tally = BASE_TALLIES[args.tally]()

    synthesizer = None
    if not args.no_synthesis:
        synthesizer = Synthesizer(backend, model=primary.model or args.model)

    council = Council(
        agents,
        rounds=args.rounds,
        tally=tally,
        synthesizer=synthesizer,
        backend_pool=pool,
        on_event=emit,
    )

    context = load_context(args)
    if context:
        emit("context", {"chars": len(context)})

    transcript = await council.deliberate(args.task, context=context)

    synthesis = None
    if not args.no_synthesis:
        synthesis = await council.synthesize(args.task, transcript, context=context)

    options = None
    vote_result = None
    if not args.no_vote:
        options = args.options
        if not options:
            emit("options", {"status": "deriving"})
            options = await derive_options(
                agents[0],
                args.task,
                transcript,
                context=context,
                max_options=args.max_options,
            )
        emit("options", {"status": "ready", "options": options})
        vote_result = await council.vote(args.task, transcript, options, context=context, synthesis=synthesis)

    return CouncilRunResult(
        task=args.task,
        context=context,
        rounds=args.rounds,
        council=args.council,
        model=args.model,
        backend=args.backend,
        agents=agents_info,
        transcript=transcript,
        synthesis=synthesis,
        options=options,
        vote_result=vote_result,
        events=events,
        warnings=warnings,
    )
