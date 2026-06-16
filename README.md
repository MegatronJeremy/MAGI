# MAGI — local multi-agent council

Three (or more) AI agents with distinct personalities debate a problem, then
vote on a decision. Everything runs locally and GPU-accelerated through Ollama.
Modeled on the MAGI system from Evangelion.

Same base model, sharply divergent system prompts. The whole point is to make
them **disagree** — if personas are too similar, the council just echoes itself.

## The three personalities

The default council is an Evangelion reference: MELCHIOR, BALTHASAR, and CASPER
are named after the three MAGI supercomputers created by Dr. Naoko Akagi. In the
show, the MAGI are organic computers implanted with three aspects of Naoko's
personality through the Personality Transplant OS: Melchior as the scientist,
Balthasar as the mother, and Casper as the woman.

MAGI borrows that idea rather than copying it literally. Here, the three names
become separate agent personas that force a problem through conflicting lenses
before any synthesis or vote happens.

The structure is also loosely inspired by Jungian psychology: useful judgment is
not treated as one flat voice, but as a dialogue between competing psychic
functions and archetypal pressures. MAGI makes that split explicit. Each agent is
a deliberately biased part of the council, not a neutral assistant.

- **MELCHIOR, Reason**: the scientist aspect. Logic, evidence, mechanisms,
  first principles, and causal rigor. MELCHIOR attacks assumptions and distrusts
  emotional or intuitive reasoning when it outruns evidence.
- **BALTHASAR, Care**: the mother aspect. Protection, duty, continuity,
  relationships, and long-term wellbeing. BALTHASAR asks who is helped, who is
  harmed, and what obligations are being honored or abandoned.
- **CASPER, Selfhood**: the woman aspect. Desire, autonomy, dignity, identity,
  and lived experience. CASPER asks what the asker actually wants and what kind
  of person the choice makes them.

The intended effect is not consensus theater. It is structured disagreement:
proposal, critique, synthesis, and optionally a vote.

## Project structure

```
magi/
├── llm/                  # backend abstraction (swap inference engines)
│   ├── base.py           #   Backend protocol
│   ├── ollama.py         #   Ollama implementation (ROCm / CUDA)
│   ├── pool.py           #   BackendPool: multi-instance routing
│   └── __init__.py       #   registry: get_backend("ollama")
├── agents/
│   ├── agent.py          #   Agent dataclass: persona + backend + weight
│   ├── personas.py       #   council presets (MELCHIOR / BALTHASAR / CASPER)
│   ├── voting.py         #   robust vote parsing (handles messy local-model JSON)
│   └── __init__.py       #   registry: get_council("magi")
├── council/
│   ├── council.py        #   orchestrator: debate loop + vote
│   ├── context.py        #   pluggable context trimming strategies
│   ├── tally.py          #   pluggable vote tally (majority / weighted)
│   └── __init__.py
└── cli/
    ├── main.py           #   CLI entry point, wires everything
    ├── pool_config.py    #   GPU detection, Ollama auto-spawn, pool assembly
    └── options.py        #   derive vote options from the debate
```

The layering is deliberate: the council never hardcodes Ollama or a tally rule.
Backends, personas, context strategies, and tally strategies are all swappable
through small registries / protocols, so the aristocratic layer (below) drops in
without touching the orchestrator.

## Setup

```bash
pip install -e .          # or: pip install -r requirements.txt
```

### Ollama

Install from [ollama.com](https://ollama.com). Ollama ships a unified binary that
selects the right backend at runtime:

- **AMD (ROCm)**: detected automatically from `gfx`-architecture; ROCm 7.x is
  bundled for RDNA3/RDNA4 (gfx1100–gfx1201).
- **NVIDIA (CUDA)**: uses the system CUDA driver; CUDA 12+ recommended.

```bash
ollama serve
```

Useful env vars for AMD if your card is not auto-detected:

```bash
# Force a compatible gfx version — set to match your chip
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export OLLAMA_NUM_GPU=999   # push as many layers as possible onto the GPU
```

### Pull models

MAGI uses two model tiers by default:

| Tier | Default | GPU target | VRAM budget |
|------|---------|-----------|-------------|
| Primary | `qwen3:14b` | Large-VRAM GPU (≥ 14 GB) | ~9.3 GB weights + KV cache |
| Secondary | `qwen3:8b` | Small-VRAM GPU (< 14 GB) | ~5.2 GB weights + KV cache |

```bash
ollama pull qwen3:14b   # primary — ~9.3 GB
ollama pull qwen3:8b    # secondary — ~5.2 GB
```

Both models support Qwen3's optional **thinking mode** (chain-of-thought
reasoning before each answer). It is enabled by default (`MAGI_THINK=true`).
Set `MAGI_THINK=false` to disable it for faster, lower-latency responses.

## Running

```bash
# console script (after pip install -e .)
magi "Should I render reflections with SSR or ray tracing?"

# or as a module
python -m magi.cli.main "Pick a database for the project" \
    --rounds 3 \
    --options "PostgreSQL" "SQLite" "DuckDB" \
    --tally weighted
```

If you omit `--options`, the council derives the distinct positions from the
debate before voting, so the vote is grounded in what was actually argued.

### Automatic multi-GPU Ollama

MAGI auto-discovers and auto-spawns local Ollama servers. On startup it scans
ports starting at `--host` (`11434` through `11441` by default), keeps any live
servers, detects local GPUs, then starts missing `ollama serve` processes on the
next free ports — one server per detected GPU:

```bash
magi "Should we rewrite the renderer in Rust?" --no-tui
```

**Per-GPU model selection**: spawned servers on GPUs with less than
`--small-gpu-vram-mib` (default 14 000 MiB) receive `--model-secondary`
(`qwen3:8b`) instead of the primary `--model` (`qwen3:14b`). This prevents OOM
on 12 GB cards while keeping the sharpest model on the 16 GB card. The threshold
and models are fully configurable:

```bash
# Mixed AMD 16 GB (primary) + NVIDIA 12 GB (secondary) — default behaviour:
magi "Compare these architecture options" --no-tui
# → port 11434: qwen3:14b on ROCm  (16 GB AMD, VRAM ≥ 14 000 MiB)
# → port 11435: qwen3:8b  on CUDA  (12 GB NVIDIA, VRAM < 14 000 MiB)

# Override models explicitly:
magi "..." --model qwen3:14b --model-secondary qwen3:8b --no-tui

# Raise the threshold (e.g. treat 16 GB cards as secondary too):
magi "..." --small-gpu-vram-mib 17000 --no-tui

# Disable secondary model (use same model everywhere):
magi "..." --model-secondary qwen3:14b --no-tui
```

> **Important:** MAGI is **not** splitting one model across vendor backends.
> Each Ollama server is a separate process bound to one GPU and one runtime
> (ROCm *or* CUDA). Mixing ROCm and CUDA in a single `llama.cpp` process is not
> supported. MAGI routes whole chat requests to independent servers — one
> request, one GPU, one backend family.

Other useful controls:

```bash
magi "..." --scan-ports 12 --no-tui
magi "..." --no-auto-spawn-ollama --no-tui   # use only already-running servers
magi "..." --ollama-command /path/to/ollama --no-tui
```

Auto-spawn is best-effort. GPU detection uses `nvidia-smi` for NVIDIA
(including VRAM size) and `rocm-smi`/Windows video-controller detection for AMD.
Spawned processes receive `OLLAMA_HOST` plus vendor visibility masks
(`CUDA_VISIBLE_DEVICES` for NVIDIA; `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`,
and `GPU_DEVICE_ORDINAL` for AMD) so each server is pinned to one GPU.

You can still start servers manually if you want full control:

```bash
# AMD (ROCm) on port 11434
HIP_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11434 ollama serve

# NVIDIA (CUDA) on port 11435
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11435 ollama serve
```

Assignment policies:

- `pooled` (default): each model call acquires the next free instance, so no
  instance handles two calls at once and each propose/critique phase can run up
  to N agents in parallel.
- `round_robin`: agent index `i` uses instance `i % N`.
- `pinned`: set an agent's optional `instance` field to an instance name; agents
  without a pin fall back to round-robin.

Debate rounds are phased. In each round, all agents first produce blind
proposals concurrently from the same frozen prior transcript. After those
proposals are appended in stable council order, all agents critique concurrently
with visibility into the proposal set.

## MCP server for Claude Desktop / Claude Code

MAGI can expose the same local council pipeline as an MCP server over stdio.
Claude launches the server process, and the server still talks only to local
Ollama instances.

Install the project dependencies first:

```bash
pip install -e .
```

Run the server directly to confirm it imports and waits for MCP stdio traffic:

```bash
python -m magi.mcp.server
```

Claude Desktop registration example for `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "magi-council": {
      "command": "C:\\Dev\\MAGI\\.venv\\Scripts\\python.exe",
      "args": ["-m", "magi.mcp.server"],
      "cwd": "C:\\Dev\\MAGI",
      "env": {
        "MAGI_MODEL": "qwen3:14b",
        "MAGI_MODEL_SECONDARY": "qwen3:8b",
        "MAGI_SMALL_GPU_VRAM_MIB": "14000",
        "MAGI_THINK": "true",
        "MAGI_BACKEND": "ollama",
        "MAGI_HOST": "http://localhost:11434",
        "MAGI_ASSIGNMENT": "pooled",
        "MAGI_AUTO_INSTANCES": "true",
        "MAGI_AUTO_SPAWN_OLLAMA": "true"
      }
    }
  }
}
```

Set `"MAGI_THINK": "false"` if you find thinking-mode latency too high for
interactive use (council rounds will be significantly faster at the cost of
less explicit chain-of-thought reasoning).

The primary MCP tool is:

```text
consult_council(question: str, context: str = "", rounds: int = 3, vote: bool = true)
```

It returns JSON with `synthesis`, optional `vote`, `transcript_summary`,
`agents`, and `warnings`. Errors such as Ollama being down are returned as
`{"ok": false, "error": "...", ...}` instead of crashing the MCP server.

There is also:

```text
list_council()
```

Use the smoke helper before wiring Claude:

```bash
python -m magi.mcp.smoke --list
python -m magi.mcp.smoke "Should we use SQLite or PostgreSQL?" --rounds 1 --no-vote
```

You can also use the MCP Inspector if you have Node available:

```bash
npx @modelcontextprotocol/inspector .\.venv\Scripts\python.exe -m magi.mcp.server
```

Example `consult_council` result shape:

```json
{
  "ok": true,
  "synthesis": "1. POINTS OF AGREEMENT ...",
  "vote": {
    "options": ["Use SQLite", "Use PostgreSQL"],
    "scores": {"Use SQLite": 2, "Use PostgreSQL": 1},
    "winner": "Use SQLite",
    "tie_between": null,
    "tie_break": null,
    "ballots": []
  },
  "transcript_summary": {
    "turn_count": 6,
    "omitted_turns": 0,
    "max_turn_chars": 700,
    "rounds": [
      {
        "round": 1,
        "phases": {
          "propose": [{"agent": "MELCHIOR", "excerpt": "...", "chars": 312}],
          "critique": [{"agent": "MELCHIOR", "excerpt": "...", "chars": 280}]
        }
      }
    ]
  },
  "agents": [{"name": "MELCHIOR", "persona": "..."}],
  "warnings": []
}
```

## Extending it

**New backend (vLLM, llama.cpp, remote):** implement the `Backend` protocol in
`magi/llm/base.py`, register it in `magi/llm/__init__.py`. Nothing else changes.

**New council / personas:** add a factory in `magi/agents/personas.py` and
register it in `COUNCILS`. Select with `--council yourname`.

**Aristocratic layer (weighted votes + rotating consul):** most of the wiring is
already here:
- `Agent.weight` and `Agent.domains` fields exist.
- `WeightedVote` tally reads `weight` — use `--tally weighted`.
- For a rotating consul/mandate, hold state outside the loop, rotate after each
  decision, and give the current consul a tie-breaking weight or veto. This is
  logic *around* the model — the engine underneath is unchanged.

**Expertise routing:** use `Agent.domains` to weight or pre-select agents based
on the task topic before the debate starts.

## Known challenges (already partly handled)

- **Context growth** with more agents × rounds → `context.py` keeps the first
  turn plus the most recent N. For production, summarize the elided middle with
  the model instead of dropping it.
- **Convergence** — agents either over-agree (shared base model) or loop forever.
  Mitigated by sharp personas and a hard round cap. If they agree too fast, raise
  temperatures and make the prompts more opinionated.
- **Speed** — propose and critique phases dispatch concurrently. With a single
  local instance, Ollama still serializes the real work; with multiple local
  Ollama servers, automatic discovery and `--assignment pooled` keep them busy.
  On mixed VRAM hardware (e.g. 16 GB AMD + 12 GB NVIDIA), the secondary model
  on the smaller card still participates fully and meaningfully; the primary
  model carries heavier reasoning turns through pooled scheduling.
