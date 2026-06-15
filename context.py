"""Context management.

Transcripts grow fast with more agents x more rounds. These strategies
keep the prompt bounded. `KeepHeadTail` is the cheap default; swap in a
summarizing strategy for production (summarize the elided middle with the
model instead of dropping it).
"""

from __future__ import annotations

from typing import Protocol


class ContextStrategy(Protocol):
    def trim(self, transcript: list[dict]) -> list[dict]: ...


class KeepHeadTail:
    """Keep the first turn (anchors framing) plus the most recent N."""

    def __init__(self, cap: int = 16):
        self.cap = cap

    def trim(self, transcript: list[dict]) -> list[dict]:
        if len(transcript) <= self.cap:
            return transcript
        head = transcript[:1]
        tail = transcript[-(self.cap - 2):]
        marker = {"name": "system", "content": "(...earlier turns elided...)"}
        return head + [marker] + tail


class NoTrim:
    """Pass everything through — fine for short debates."""

    def trim(self, transcript: list[dict]) -> list[dict]:
        return transcript
