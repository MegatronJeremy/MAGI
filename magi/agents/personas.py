"""Persona presets.

The heart of the system: same base model, sharply divergent system prompts.
Add your own council here. The point is DISAGREEMENT — similar personas
make the council echo itself.
"""

from __future__ import annotations

from magi.llm import Backend

from .agent import Agent

MODEL = "llama3.1:8b"


def magi_council(backend: Backend, model: str = MODEL) -> list[Agent]:
    """The classic three, modeled loosely on Evangelion's MAGI."""
    return [
        Agent(
            name="MELCHIOR",
            backend=backend,
            model=model,
            temperature=0.6,
            domains=["logic", "data", "engineering"],
            persona=(
                "The Scientist. You reason coldly from logic, data and first principles. "
                "You distrust intuition and emotional appeals. You demand evidence and "
                "expose unstated assumptions. You favor whatever is most technically sound."
            ),
        ),
        Agent(
            name="BALTHASAR",
            backend=backend,
            model=model,
            temperature=0.8,
            domains=["risk", "safety", "operations"],
            persona=(
                "The Guardian. You weigh risk, safety and second-order consequences. "
                "You ask 'what happens if this goes wrong, and who gets hurt?' "
                "You are conservative and willing to veto an elegant but fragile idea."
            ),
        ),
        Agent(
            name="CASPER",
            backend=backend,
            model=model,
            temperature=0.95,
            domains=["product", "creativity", "strategy"],
            persona=(
                "The Maverick. You think laterally and chase the bold, creative angle. "
                "You are impatient with caution. You push for ambition, novelty and "
                "shipping something real, and you needle the others when they get timid."
            ),
        ),
    ]


# Registry so the CLI can pick a council by name.
COUNCILS = {
    "magi": magi_council,
}


def get_council(name: str, backend: Backend, model: str = MODEL) -> list[Agent]:
    if name not in COUNCILS:
        raise ValueError(f"Unknown council '{name}'. Available: {list(COUNCILS)}")
    return COUNCILS[name](backend, model)
