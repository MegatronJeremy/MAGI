from __future__ import annotations

from .base import Backend
from .ollama import OllamaBackend


def get_backend(name: str = "ollama", **kwargs) -> Backend:
    if name == "ollama":
        return OllamaBackend(**kwargs)
    raise ValueError("Unknown backend. Available: ['ollama']")


__all__ = [
    "Backend",
    "OllamaBackend",
    "get_backend",
]
