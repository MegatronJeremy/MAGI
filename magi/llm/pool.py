"""Backend pools for routing council agents across model instances."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

import httpx

from .base import Backend


ASSIGNMENT_POLICIES = {"round_robin", "pinned", "pooled"}


@dataclass
class BackendInstance:
    name: str
    backend: Backend
    model: str | None = None
    host: str | None = None
    backend_name: str = "ollama"
    tags: list[str] | None = None


class BackendPool:
    """Ordered pool of named backend instances.

    The pool is backend-agnostic for actual chat calls. Ollama-specific health
    checks are best-effort and only run when an instance exposes a host.
    """

    def __init__(
        self,
        instances: list[BackendInstance],
        assignment: str = "pooled",
        warn: Callable[[str], None] | None = None,
    ):
        if assignment not in ASSIGNMENT_POLICIES:
            raise ValueError(
                f"Unknown assignment '{assignment}'. Available: {sorted(ASSIGNMENT_POLICIES)}"
            )
        if not instances:
            raise ValueError("BackendPool requires at least one instance")
        self.instances = instances
        self.assignment = assignment
        self.warn = warn or print
        self._queue: asyncio.Queue[BackendInstance] = asyncio.Queue()
        self._reset_queue()

    @classmethod
    def from_entries(
        cls,
        entries: list[dict],
        *,
        assignment: str = "pooled",
        default_backend: str = "ollama",
        default_model: str = "llama3.1:8b",
        warn: Callable[[str], None] | None = None,
    ) -> "BackendPool":
        from . import get_backend

        instances = []
        for index, entry in enumerate(entries):
            backend_name = entry.get("backend", default_backend)
            host = entry.get("host")
            model = entry.get("model", default_model)
            kwargs = dict(entry.get("kwargs", {}))
            if host:
                kwargs["host"] = host
            instances.append(
                BackendInstance(
                    name=entry.get("name") or f"{backend_name}-{index}",
                    backend=get_backend(backend_name, **kwargs),
                    model=model,
                    host=host,
                    backend_name=backend_name,
                )
            )
        return cls(instances, assignment=assignment, warn=warn)

    @classmethod
    def default_local(
        cls,
        *,
        backend_name: str = "ollama",
        host: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        assignment: str = "pooled",
        warn: Callable[[str], None] | None = None,
    ) -> "BackendPool":
        return cls.from_entries(
            [{"name": "local", "backend": backend_name, "host": host, "model": model}],
            assignment=assignment,
            default_backend=backend_name,
            default_model=model,
            warn=warn,
        )

    async def health_check(
        self,
        *,
        warn_dead: bool = True,
        warn_empty: bool = True,
        warn_live: bool = True,
    ) -> int:
        checks = await asyncio.gather(
            *(self._check_instance(instance, warn_live=warn_live) for instance in self.instances),
            return_exceptions=True,
        )
        live = []
        for instance, result in zip(self.instances, checks):
            if isinstance(result, Exception):
                if warn_dead:
                    self.warn(
                        f"WARNING backend instance {instance.name} failed health check: {result}"
                    )
                continue
            if result:
                live.append(instance)

        if live:
            self.instances = live
            self._reset_queue()
            return len(live)
        elif warn_empty:
            self.warn("WARNING no backend instances passed health check; continuing anyway")
        return 0

    def backend_for(self, agent, index: int) -> BackendInstance:
        if self.assignment == "pinned":
            pinned = getattr(agent, "instance", None)
            if pinned:
                for instance in self.instances:
                    if instance.name == pinned:
                        return instance
                self.warn(
                    f"WARNING agent {agent.name} pinned to unknown instance {pinned}; "
                    "falling back to round_robin"
                )
        return self.instances[index % len(self.instances)]

    async def acquire(self) -> BackendInstance:
        return await self._queue.get()

    def release(self, instance: BackendInstance) -> None:
        self._queue.put_nowait(instance)

    async def run_agent_method(
        self,
        agent,
        index: int,
        method_name: str,
        *args,
        **kwargs,
    ) -> str:
        if self.assignment == "pooled":
            return await self._run_pooled(agent, method_name, *args, **kwargs)

        instance = self.backend_for(agent, index)
        try:
            return await self._call_on_instance(instance, agent, method_name, *args, **kwargs)
        except Exception as exc:
            self.warn(
                f"WARNING {agent.name} failed on backend instance {instance.name}: {exc}"
            )
            return f"(backend error on {instance.name}: {exc})"

    def primary_instance(self) -> BackendInstance:
        return self.instances[0]

    def _reset_queue(self) -> None:
        self._queue = asyncio.Queue()
        for instance in self.instances:
            self._queue.put_nowait(instance)

    async def _check_instance(self, instance: BackendInstance, *, warn_live: bool = True) -> bool:
        if instance.backend_name != "ollama" or not instance.host:
            if warn_live:
                self.warn(
                    f"backend instance {instance.name}: no Ollama health check available; assuming live"
                )
            return True

        url = f"{instance.host.rstrip('/')}/api/tags"
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
        data = response.json()
        tags = [
            model.get("name", "")
            for model in data.get("models", [])
            if isinstance(model, dict) and model.get("name")
        ]
        instance.tags = tags
        model_note = (
            f"serves {instance.model}"
            if instance.model in tags
            else f"model {instance.model} not listed; available: {', '.join(tags) or '(none)'}"
        )
        if warn_live:
            self.warn(f"backend instance {instance.name} live at {instance.host}: {model_note}")
        return True

    async def _run_pooled(self, agent, method_name: str, *args, **kwargs) -> str:
        attempts = max(1, len(self.instances))
        last_error: Exception | None = None
        tried: set[str] = set()
        held: list[BackendInstance] = []
        try:
            for _ in range(attempts):
                instance = await self.acquire()
                if instance.name in tried:
                    held.append(instance)
                    continue
                tried.add(instance.name)
                try:
                    result = await self._call_on_instance(
                        instance, agent, method_name, *args, **kwargs
                    )
                    held.append(instance)
                    return result
                except Exception as exc:
                    last_error = exc
                    self.warn(
                        f"WARNING {agent.name} failed on backend instance {instance.name}: {exc}"
                    )
                    held.append(instance)
            instance_names = ", ".join(sorted(tried)) or "(none)"
            return f"(backend error after trying {instance_names}: {last_error})"
        finally:
            for instance in held:
                self.release(instance)

    async def _call_on_instance(
        self,
        instance: BackendInstance,
        agent,
        method_name: str,
        *args,
        **kwargs,
    ) -> str:
        from dataclasses import replace

        routed_agent = replace(
            agent,
            backend=instance.backend,
            model=instance.model or agent.model,
        )
        method = getattr(routed_agent, method_name)
        return await method(*args, **kwargs)
