"""Backend abstraction.

A Backend is anything that can turn (system, user, params) into text.
This indirection means the council never hardcodes Ollama: swap in vLLM,
llama.cpp, or a remote endpoint by implementing this one protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    async def chat(
        self,
        model: str,
        system: str,
        user: str,
        *,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Return the model's text completion for a single system+user turn."""
        ...
