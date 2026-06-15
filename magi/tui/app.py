"""Textual terminal UI for live MAGI council sessions."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass

from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, LoadingIndicator, MarkdownViewer, RichLog, Static

from magi.cli.runner import run_council


AGENT_STYLES = {
    "MELCHIOR": {"class": "melchior", "accent": "#7df9ff"},
    "BALTHASAR": {"class": "balthasar", "accent": "#ffbf3f"},
    "CASPER": {"class": "casper", "accent": "#ff5c8a"},
}


def _display_text(value: object) -> str:
    if isinstance(value, dict):
        for key in ("choice", "option", "title", "label", "name", "text"):
            option = value.get(key)
            if option:
                return str(option)
        for option in value.values():
            if isinstance(option, str) and option.strip():
                return option
    if isinstance(value, str) and value.strip().startswith("{") and value.strip().endswith("}"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value
        if isinstance(parsed, dict):
            return _display_text(parsed)
    return str(value)


@dataclass(frozen=True)
class CouncilPhase:
    label: str
    detail: str = ""


@dataclass(frozen=True)
class LogEntry:
    content: object
    width: int | None
    expand: bool
    shrink: bool
    scroll_end: bool | None
    animate: bool


class VerticalOnlyRichLog(RichLog):
    """RichLog variant that only scrolls vertically."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._entries: list[LogEntry] = []
        self._replaying_entries = False
        self._processing_deferred_entries = False
        self._last_reflow_width = 0

    def write(
        self,
        content,
        width: int | None = None,
        expand: bool = False,
        shrink: bool = True,
        scroll_end: bool | None = None,
        animate: bool = False,
    ):
        if not self._replaying_entries and not self._processing_deferred_entries:
            stored_content = content.copy() if isinstance(content, Text) else content
            self._entries.append(
                LogEntry(stored_content, width, expand, shrink, scroll_end, animate)
            )
        return super().write(content, width, expand, shrink, scroll_end, animate)

    def validate_scroll_x(self, value: float) -> float:
        return 0.0

    def action_scroll_left(self) -> None:
        self.scroll_to(x=0, y=self.scroll_y, animate=False)

    def action_scroll_right(self) -> None:
        self.scroll_to(x=0, y=self.scroll_y, animate=False)

    def _on_mouse_scroll_left(self, event) -> None:
        event.stop()
        self.scroll_to(x=0, y=self.scroll_y, animate=False)

    def _on_mouse_scroll_right(self, event) -> None:
        event.stop()
        self.scroll_to(x=0, y=self.scroll_y, animate=False)

    def on_resize(self, event) -> None:
        was_size_known = self._size_known
        self._processing_deferred_entries = True
        try:
            super().on_resize(event)
        finally:
            self._processing_deferred_entries = False

        width = self.scrollable_content_region.width
        if width <= 0 or not self._size_known:
            return

        if not was_size_known:
            self._last_reflow_width = width
            return

        if width != self._last_reflow_width:
            self._last_reflow_width = width
            self._reflow_entries()

    def _reflow_entries(self) -> None:
        if not self._entries:
            return

        was_at_end = self.is_vertical_scroll_end
        scroll_y = self.scroll_y

        self._replaying_entries = True
        try:
            super().clear()
            for entry in self._entries:
                content = entry.content.copy() if isinstance(entry.content, Text) else entry.content
                super().write(
                    content,
                    entry.width,
                    entry.expand,
                    entry.shrink,
                    False,
                    False,
                )
        finally:
            self._replaying_entries = False

        self.scroll_to(x=0, y=0, animate=False)
        if was_at_end:
            self.scroll_end(animate=False, x_axis=False, immediate=True)
        else:
            self.scroll_to(x=0, y=scroll_y, animate=False)


class AgentPanel(Vertical):
    """One live transcript panel for a MAGI agent."""

    def __init__(self, agent_name: str) -> None:
        super().__init__(classes=f"agent-panel {AGENT_STYLES[agent_name]['class']}")
        self.agent_name = agent_name
        self.header = Static(agent_name, classes="agent-title")
        self.turn_log = VerticalOnlyRichLog(
            markup=True,
            wrap=True,
            min_width=1,
            highlight=False,
            classes="agent-log",
        )
        self.thinking = LoadingIndicator(classes="thinking")

    def compose(self) -> ComposeResult:
        yield self.header
        yield self.turn_log
        yield self.thinking

    def on_mount(self) -> None:
        self.set_thinking(False)

    def add_turn(self, round_number: int, phase: str, content: str) -> None:
        escaped = escape(content.strip())
        self.turn_log.write(
            f"[b]ROUND {round_number:02d}[/b] [dim]{escape(phase.upper())}[/dim]\n{escaped}\n"
        )

    def set_thinking(self, active: bool) -> None:
        self.set_class(active, "active")
        self.thinking.display = active


class SynthesisViewer(MarkdownViewer):
    """Markdown viewer that only scrolls vertically."""

    def validate_scroll_x(self, value: float) -> float:
        return 0.0

    def action_scroll_left(self) -> None:
        self.scroll_to(x=0, y=self.scroll_y, animate=False)

    def action_scroll_right(self) -> None:
        self.scroll_to(x=0, y=self.scroll_y, animate=False)

    def _on_mouse_scroll_left(self, event) -> None:
        event.stop()
        self.scroll_to(x=0, y=self.scroll_y, animate=False)

    def _on_mouse_scroll_right(self, event) -> None:
        event.stop()
        self.scroll_to(x=0, y=self.scroll_y, animate=False)


class MagiTuiApp(App[None]):
    """Full-screen MAGI console that renders council events as they arrive."""

    TITLE = "MAGI"
    SUB_TITLE = "Local Council"
    BINDINGS = [("q", "quit", "Quit"), ("ctrl+c", "quit", "Quit")]

    CSS = """
    Screen {
        background: #030706;
        color: #b8f7c4;
        layout: vertical;
    }

    Header {
        background: #07100d;
        color: #9dffb0;
        text-style: bold;
    }

    Footer {
        background: #07100d;
        color: #6fdc82;
    }

    #agent-row {
        height: 52%;
    }

    .agent-panel {
        width: 1fr;
        height: 100%;
        margin: 0 1;
        border: tall #1b3d2a;
        background: #050908;
    }

    .agent-panel.active {
        border: heavy #8cff9c;
    }

    .agent-title {
        height: 3;
        content-align: center middle;
        text-style: bold;
        background: #07100d;
    }

    .melchior {
        border: tall #218c92;
    }

    .melchior .agent-title {
        color: #7df9ff;
    }

    .balthasar {
        border: tall #9b6e16;
    }

    .balthasar .agent-title {
        color: #ffbf3f;
    }

    .casper {
        border: tall #8e2443;
    }

    .casper .agent-title {
        color: #ff5c8a;
    }

    .agent-log {
        height: 1fr;
        padding: 0 1;
        background: #030706;
        overflow-x: hidden;
    }

    .thinking {
        height: 3;
        content-align: center middle;
        color: #8cff9c;
    }

    #synthesis-pane {
        height: 28%;
        margin: 0 1;
        border: tall #2c6a3e;
        background: #030706;
    }

    #synthesis-title,
    #vote-title {
        height: 1;
        color: #9dffb0;
        text-style: bold;
        background: #07100d;
        padding: 0 1;
    }

    #synthesis {
        height: 1fr;
        padding: 0 1;
        color: #d9ffe1;
        overflow-x: hidden;
    }

    #synthesis Markdown {
        width: 100%;
        overflow-x: hidden;
    }

    #vote-pane {
        height: 1fr;
        min-height: 8;
        margin: 0 1 1 1;
        border: tall #315c24;
        background: #030706;
    }

    #vote-log {
        height: 1fr;
        padding: 0 1;
        background: #030706;
        overflow-x: hidden;
    }

    .phase {
        color: #ffbf3f;
        text-style: bold;
    }
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args
        self.agent_names: list[str] = []
        self.agent_panels: dict[str, AgentPanel] = {}
        self.ballots: list[dict] = []
        self.phase = CouncilPhase("BOOT")
        self._ballot_header_shown = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="agent-row"):
            for name in ("MELCHIOR", "BALTHASAR", "CASPER"):
                panel = AgentPanel(name)
                self.agent_panels[name] = panel
                yield panel
        with Vertical(id="synthesis-pane"):
            yield Static("SYNTHESIS", id="synthesis-title")
            yield SynthesisViewer(
                "_Awaiting neutral scribe output._",
                show_table_of_contents=False,
                id="synthesis",
            )
        with Vertical(id="vote-pane"):
            yield Static("VOTE / RESULT", id="vote-title")
            yield VerticalOnlyRichLog(
                markup=True,
                wrap=True,
                min_width=1,
                highlight=False,
                id="vote-log",
            )
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"MAGI :: {self.args.task}"
        self.query_one("#vote-log", RichLog).write(
            f"[b green]TASK[/b green] {escape(self.args.task)}"
        )
        self.run_worker(self._run_council(), exclusive=True)

    async def _run_council(self) -> None:
        try:
            self._set_phase("DELIBERATION", "round 1")
            result = await run_council(self.args, on_event=self._on_council_event)
            self._set_thinking(None)
            detail = "vote skipped" if result.vote_result is None else "session finished"
            self._set_phase("COMPLETE", detail)
        except Exception as exc:  # pragma: no cover - visible runtime failure path
            self._set_thinking(None)
            self._set_phase("ERROR", str(exc))
            self.query_one("#vote-log", RichLog).write(f"[b red]ERROR[/b red] {escape(str(exc))}")

    def _on_council_event(self, kind: str, data: dict) -> None:
        if kind == "warning":
            self._log_pool_message(data["message"])
        elif kind == "context":
            self._set_phase("CONTEXT", f"{data['chars']} chars loaded")
        elif kind == "turn":
            name = data["name"]
            self.agent_panels[name].add_turn(
                data["round"],
                data.get("phase", "turn"),
                data["content"],
            )
        elif kind == "synthesis":
            self.run_worker(self._update_synthesis(data["text"]))
        elif kind == "phase":
            self._set_phase("DELIBERATION", f"round {data['round']} {data['phase']}")
            self._set_thinking_names(self.agent_names)
        elif kind == "agents":
            self.agent_names = data["names"]
        elif kind == "options" and data.get("status") == "deriving":
            self._set_phase("OPTIONS", "deriving vote choices")
            self._set_thinking(self.agent_names[0] if self.agent_names else None)
        elif kind == "options" and data.get("status") == "ready":
            self._set_thinking(None)
            self._show_options(data["options"])
            self._set_phase("VOTE", "collecting ballots")
            self._set_thinking(self.agent_names[0] if self.agent_names else None)
        elif kind == "ballot":
            self.ballots.append(data)
            self._render_ballot(data)
            self._set_next_thinker_after_ballot(data["voter"])
        elif kind == "result":
            self._set_thinking(None)
            self._render_result(data)

    def _set_next_thinker_after_turn(self, name: str, round_number: int) -> None:
        if name not in self.agent_names:
            self._set_thinking(None)
            return
        index = self.agent_names.index(name)
        if index + 1 < len(self.agent_names):
            self._set_thinking(self.agent_names[index + 1])
        elif round_number < self.args.rounds:
            self._set_phase("DELIBERATION", f"round {round_number + 1}")
            self._set_thinking(self.agent_names[0])
        else:
            self._set_thinking(None)

    def _set_next_thinker_after_ballot(self, voter: str) -> None:
        if voter not in self.agent_names:
            self._set_thinking(None)
            return
        index = self.agent_names.index(voter)
        next_name = self.agent_names[index + 1] if index + 1 < len(self.agent_names) else None
        self._set_thinking(next_name)

    def _set_thinking(self, active_name: str | None) -> None:
        for name, panel in self.agent_panels.items():
            panel.set_thinking(name == active_name)

    def _set_thinking_names(self, active_names: list[str]) -> None:
        active = set(active_names)
        for name, panel in self.agent_panels.items():
            panel.set_thinking(name in active)

    def _set_phase(self, label: str, detail: str = "") -> None:
        self.phase = CouncilPhase(label, detail)
        suffix = f" :: {detail}" if detail else ""
        self.sub_title = f"{label}{suffix}"

    def _log_pool_message(self, message: str) -> None:
        self.query_one("#vote-log", RichLog).write(f"[dim]{escape(message)}[/dim]")

    async def _update_synthesis(self, text: str) -> None:
        viewer = self.query_one("#synthesis", SynthesisViewer)
        await viewer.document.update(text)
        viewer.scroll_home(animate=False)
        viewer.scroll_to(x=0, y=0, animate=False)

    def _show_options(self, options: list[str]) -> None:
        vote_log = self.query_one("#vote-log", RichLog)
        vote_log.write("[b green]OPTIONS[/b green]")
        for index, option in enumerate(options, start=1):
            vote_log.write(f"  [green]{index}.[/green] {escape(_display_text(option))}")

    def _render_ballot(self, ballot: dict) -> None:
        voter = ballot["voter"]
        accent = AGENT_STYLES.get(voter, {}).get("accent", "#9dffb0")
        choice = escape(_display_text(ballot["choice"]))
        reason = escape(str(ballot["reason"]))
        vote_log = self.query_one("#vote-log", RichLog)
        if not self._ballot_header_shown:
            vote_log.write("[b green]BALLOTS[/b green]")
            self._ballot_header_shown = True
        vote_log.write(f"[{accent}]{voter}[/]")
        vote_log.write(f"  Choice: [b]{choice}[/b]")
        if reason:
            vote_log.write(f"  Reason: {reason}")

    def _render_result(self, result: dict) -> None:
        vote_log = self.query_one("#vote-log", RichLog)
        scores = result.get("scores", {})
        winner = result.get("winner")
        max_score = max(scores.values(), default=1.0)

        vote_log.write("[b green]RESULT[/b green]")
        for option, score in scores.items():
            width = int((score / max_score) * 24) if max_score else 0
            bar = "#" * width
            marker = "  WINNER" if option == winner else ""
            style = "bold #9dffb0" if option == winner else "#6fdc82"
            label = escape(_display_text(option))
            vote_log.write(
                f"[{style}]{label}[/]\n  [{style}]{bar:<24} {score:g}{marker}[/]"
            )

        if result.get("tie_break"):
            tie_break = result["tie_break"]
            consul = escape(str(tie_break["consul"]))
            among = ", ".join(escape(str(option)) for option in tie_break["among"])
            vote_log.write(
                f"[reverse bold #ffbf3f] TIE BREAK [/] "
                f"consul {consul} resolved deadlock among {among}"
            )

        if winner:
            vote_log.write(Text.assemble(("DECISION: ", "bold green"), (str(winner), "bold white")))
        else:
            tied = ", ".join(str(option) for option in result.get("tie_between") or [])
            vote_log.write(f"[b red]DEADLOCK[/b red] {escape(tied)}")


async def run_tui(args: argparse.Namespace) -> None:
    """Run the MAGI Textual app for parsed CLI arguments."""

    app = MagiTuiApp(args)
    await app.run_async()
