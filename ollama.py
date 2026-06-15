"""Ollama backend — local, GPU-accelerated inference.

Talks to the Ollama HTTP server (default localhost:11434). On AMD hardware
Ollama uses its ROCm build; see README for the env vars that matter.
"""

from __future__ import annotations

import httpx

from .base import Backend


class OllamaBackend(Backend):
    def __init__(self, host: str = "http://localhost:11434", timeout: float = 120.0):
        self.url = f"{host.rstrip('/')}/api/chat"
        self.timeout = timeout

    async def chat(
        self,
        model: str,
        system: str,
        user: str,
        *,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, **kwargs.get("options", {})},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(self.url, json=payload)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
