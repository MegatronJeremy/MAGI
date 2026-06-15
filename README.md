# MAGI — local multi-agent council

Three (or more) AI agents with distinct personalities debate a problem, then
vote on a decision. Everything runs locally and GPU-accelerated through Ollama.
Modeled on the MAGI system from Evangelion.

Same base model, sharply divergent system prompts. The whole point is to make
them **disagree** — if personas are too similar, the council just echoes itself.

## Project structure

```
magi/
├── llm/                  # backend abstraction (swap inference engines)
│   ├── base.py           #   Backend protocol
│   ├── ollama.py         #   Ollama implementation (ROCm on AMD)
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

### Ollama on AMD (ROCm)

Your AMD GPU matters here. Ollama ships an official ROCm build:

```bash
# Linux: the installer pulls the ROCm build automatically if it detects an AMD GPU
curl -fsSL https://ollama.com/install.sh | sh

ollama serve            # start the server
rocminfo                # should list your GPU agent
```

Useful env vars for AMD:

```bash
# if your card isn't on the official support list (common for some RDNA cards),
# force a compatible gfx version — set to match your chip
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export OLLAMA_NUM_GPU=999   # push as many layers as possible onto the GPU
```

Pull a model:

```bash
ollama pull llama3.1:8b     # default; needs ~8GB+ VRAM
# smaller alternatives if VRAM is tight:
ollama pull qwen2.5:7b
ollama pull phi3:mini
```

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

MAGI auto-discovers local Ollama servers by scanning ports starting at
`--host`. The default scan checks `11434` through `11441`, keeps the live
servers, and routes each propose/critique phase across that pool:

```bash
ollama serve

OLLAMA_HOST=127.0.0.1:11435 ollama serve
```

Then run MAGI normally:

```bash
magi "Should we rewrite the renderer in Rust?" --no-tui
```

Use a wider scan if you run more ports:

```bash
magi "Compare these architecture options" --scan-ports 12 --no-tui
```

Pull the model on each server environment you intend to use:

```bash
ollama pull llama3.1:8b
```

Assignment policies:

- `pooled` (default): each model call acquires the next free instance, so no
  instance handles two calls at once and each propose/critique phase can run up
  to N agents in parallel.
- `round_robin`: agent index `i` uses instance `i % N`.
- `pinned`: set an agent's optional `instance` field to an instance name; agents
  without a pin fall back to round-robin.

Mixed NVIDIA + AMD works because MAGI is not splitting one model across vendor
backends. Each Ollama server is a separate process bound to one local GPU/vendor
runtime, and MAGI only routes whole chat requests to those independent servers.
Keep one process per GPU/backend family; do not expect one Ollama model load to
span NVIDIA CUDA and AMD ROCm at the same time.

Debate rounds are phased. In each round, all agents first produce blind
proposals concurrently from the same frozen prior transcript. After those
proposals are appended in stable council order, all agents critique concurrently
with visibility into the proposal set.

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
