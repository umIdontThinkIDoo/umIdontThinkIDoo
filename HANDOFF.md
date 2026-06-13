# Odysseus — Handoff to the next Claude Code instance

> ## ⏩ RESUME HERE (2026-06-13) — fix iteration shipped; roadmap quick-wins underway
> **State: clean and green.** The whole bug-fix/stabilization iteration is **done
> and pushed** (branch `odysseus`, tip at/after `26cf28c`, synced to
> `origin/odysseus`). The earlier disk-full crisis is **resolved** — root FS is
> healthy and the "Above-Average" drive is mounted via an fstab `nofail` entry.
> Run `python` only inside the venv: `source /home/dari/odysseus/venv/bin/activate`.
>
> **DONE this session — Ollama Start button (roadmap F), the first quick-win:**
> - `src/ollama_control.py` — start/stop/status for a **loopback-only** Ollama
>   (`OLLAMA_HOST=127.0.0.1:11434`, hardcoded — can't be made to bind 0.0.0.0).
>   Resolves `OLLAMA_MODELS` from setting→env→systemd-unit (no hardcoded path).
>   `stop()` refuses to kill a server Odysseus didn't start (PID-tracked).
> - `routes/model_routes.py` — `GET /api/ollama/status`, `POST /api/ollama/start`
>   (503 on failure), `POST /api/ollama/stop` (409 if external). Admin-guarded.
> - `static/index.html` + `static/js/admin.js` — "Local Ollama Server" card atop
>   Settings → Add Models (services panel), status dot + Start/Stop, refreshed on
>   tab open and via `refreshAll()`.
> - `tests/test_ollama_control.py` — 17 tests. **Verified end-to-end against real
>   ollama**: starts bound to 127.0.0.1 only (confirmed via `ss`), models visible,
>   stop clean; full HTTP path exercised via TestClient.
>
> **NEXT (in progress): JSONL quick-win (new-wants #3).** Expose JSONL training-
> data in the Forge UI. Backend already accepts local JSONL via `--dataset` and
> `TrainingRequest.dataset_path` passes it for SFT/DPO/CPT (NOT qlora/rl_loop).
> Gaps to close: a dataset-path field in `static/js/forge.js` SFT/DPO/CPT forms,
> decide the qlora/rl_loop JSONL story, add a test. Then: web scraper→RAG (B),
> model-driven interview (C). See memory `feasibility-assessment-2026-06`.


You are taking over an in-progress effort on **Odysseus**, a self-hosted AI
workspace (FastAPI backend + vanilla-JS PWA frontend, ~154k LOC, ~600 test
files). You are most likely running in a **terminal on the user's own
machine** — which is the whole point of this handoff (see "Why you're here").

> **Operating rules — read first**
> 1. **Orient before you touch anything.** Read the code paths involved, run the
>    test suite, and reproduce the current state. Do NOT start editing from this
>    doc alone.
> 2. **If you are unsure about a fact, search it.** Don't guess at library
>    behavior (TRL/GRPO, Unsloth, Ollama API, gpt-oss format). Guessing cost the
>    previous instance hours (see "Mistakes").
> 3. **Confirm understanding with the user before large/architectural changes.**
> 4. **Never leak services.** Anything that launches a local model server binds
>    `127.0.0.1` only — never `0.0.0.0`, never a tunnel. (Hard rule, see Ollama.)
> 5. **Keep secrets out of git** — no tunnel URLs, tokens, or model endpoints in
>    commits.

---

## Why you're here (and why a terminal session is the right move)

The previous instance ran in Anthropic's **web sandbox**, which created three
artificial problems that **do not exist for you** on the user's machine:

- **Ollama is just `localhost:11434`.** The web sandbox could not reach the
  user's Ollama, so the previous instance spent ~15 turns fighting ngrok /
  cloudflared tunnels — all of which was unnecessary. You talk to Ollama
  directly. **Do not build or use any tunnel.**
- **You can run the test suite locally** and iterate fast.
- **You can `git push`.** The web sandbox's git proxy died mid-session, so the
  last two fixes never reached the remote (see "Job #1").

## Repo / branch state

- Remote: `github.com/umIdontThinkIDoo/umIdontThinkIDoo`, branch **`odysseus`**
  (a working branch carrying the full upstream Odysseus history + our fixes).
  Upstream origin was a fork of `the-east-star/odysseus`.
- Remote tip currently: **`6f62209`**.
- **Two commits are NOT on the remote** (web git proxy died). They exist in the
  user's `~/odysseus-fixed` working tree (applied as file changes), and were:
  - `ea9c83a` fix(forge): Personalizer now actually saves memories and mines chats
  - `c13ef1e` feat(forge): always-visible Forge section in the sidebar

> **VERIFIED 2026-06-12 (terminal session, Opus 4.8) — Job #1 still NOT done.**
> `git log` tip is `6f62209` and the local `odysseus` branch is clean-tracking
> `origin/odysseus` at the same SHA. The "two commits" above are **not commits
> at all yet** — they survive only as **uncommitted working-tree edits**:
> ```
>  M static/index.html      (+29 lines — Forge sidebar section)
>  M static/js/forge.js     (+30/-28 — Personalizer: text/ field + per-session extract)
>  ?? HANDOFF.md            (this file, untracked)
> ```
> Confirmed the `forge.js` diff matches the described fixes (`content`→`text` on
> `/api/memory/add`; mine chats via per-session `POST /api/memory/extract` reading
> `{suggestions:[...]}` instead of the old free-text JSON that 422'd). **Next
> instance: this is the first thing to land** — commit these two files (split into
> the two logical commits or one is fine), then push, then run the suite.

### Job #1 — land the outstanding work
In the user's clone (`~/odysseus-fixed`), commit the working-tree changes
(Personalizer `forge.js` + the Forge-sidebar `index.html`/`forge.js`) and push to
`origin/odysseus`. Verify `git log` shows both before continuing. Then run the
full suite (below) so you start from green.

## Setup & test commands (on the user's machine)

```bash
cd ~/odysseus-fixed
python3 -m venv venv && source venv/bin/activate   # or reuse ~/odysseus/venv
pip install -r requirements.txt
python -m pytest -q                                 # expect ~3091 passing, 0 failing
python -m uvicorn app:app --host 127.0.0.1 --port 7000
```
Login user/pass were set during `setup.py` (the user chose `dari` / their own
password). ChromaDB runs locally on `:8100` on this machine (vector RAG/memory
come up live). If port 7000 is busy, an **old odysseus instance is squatting it**
— `sudo fuser -k 7000/tcp`.

## The user's environment (important specifics)

- OS: Pop!_OS (Ubuntu-based), user `dari`.
- Ollama runs as a **systemd service** as user `ollama`, with
  `OLLAMA_MODELS=/media/dari/Above-Average/ollama_models` (a **removable drive**).
  This causes a real gotcha: the `ollama` service user **cannot traverse**
  `/media/dari/...` (it's `dari`-owned), so a service restart can crash-loop on
  `mkdir ... permission denied`, and a manual `ollama serve` run as `dari` uses a
  *different* (empty) model store unless `OLLAMA_MODELS` is set. Any
  "Start Ollama" feature MUST account for this (run as the right user / set
  `OLLAMA_MODELS`).
- Models present: `huihui_ai/gemma-4-abliterated:26b` (uncensored chat).
  Considering `gpt-oss-20b` for the agent/RL role (see roadmap).
- Also installed on the box: Open WebUI, Docker/containerd, LM Studio — so
  multiple things may contend for port 11434 / model dirs. Check `ss -ltnp`.

## What's already fixed (verify, don't redo)

All have tests and passed the suite in the previous environment:
- **training_routes.py** — launch via `sys.executable` (not bare `python`);
  removed dead `find_bash()` gate; don't pass `--dataset` to qlora; validate
  qlora needs `base_adapter_path` and rl_loop needs `book_filename` (clean 400s).
  Tests: `tests/test_training_routes.py`.
- **ingest_routes.py** — owner via `require_user` (was reading a never-set
  `request.state.user`, filing everything under "admin"); feed
  `add_documents_batch` 2-tuples `(text, meta)` not 3-tuples (was crashing every
  file with "too many values to unpack"). Tests: `tests/test_ingest_routes.py`.
- **forge.js** — RAG-ingest upload used wrong field (`file`→`files`), batched in
  one request, dropped the nonexistent `/api/ingest/start`; Personalizer now
  posts `text` (not `content`) to `/api/memory/add`, and mines chats via
  per-session `/api/memory/extract`.
- **research_routes.py** — user-facing grammar fix.
- **mcp_manager.py** — own each MCP connection's lifecycle in one task to fix the
  anyio "cancel scope in a different task" teardown error (verified clean on the
  user's machine).
- **session restore/rename** — repaired calls to dead routes.
- **index.html + forge.js** — added an always-visible **Forge** sidebar section
  (Train / QLoRA Merge / RL Loop / RAG Ingest / Personalizer / Self-Train),
  because the icon-rail (the only prior entry point) is hidden whenever the
  sidebar is open. (This is `c13ef1e`, part of Job #1.)

## Mistakes the previous instance made (don't repeat)

1. **Burned ~15 turns on a tunnel `403 host_not_allowed`** that was the **web
   sandbox's own outbound egress policy** blocking `*.trycloudflare.com` — not
   ngrok, cloudflared, or Ollama. Lesson: **isolate the failing layer first**
   (one `curl https://example.com` would have shown the egress block immediately).
   You won't hit this, but the lesson generalizes.
2. **Patched the hidden-Forge symptom** before recognizing the real pattern: the
   UI is rigid/modal and hides features by mode. See UI roadmap.
3. **Cannot certify "zero bugs"** on 154k LOC — don't claim it. Claim what's
   verified: suite green + specific fixes tested.

---

## Roadmap — requested features (confirm scope, then build)

The user wants this to feel like **a seamless app update**: new capabilities are
discoverable, consistently surfaced in **Settings**, and don't feel bolted on.

### A. UI: make it "free-flowing" (not rigid/modal)
The user finds the UI too strict. Principle: **every feature discoverable from a
consistent place; modeless where possible; panels reflow/resize.** The
Forge-hidden-in-the-rail bug was a symptom.
> **DECIDED (user, 2026-06): all three dimensions.**
> (a) **Layout** — drag/resize/dock panels, multiple open at once;
> (b) **Discoverability** — nothing hidden behind modes (rail-vs-sidebar
>     exclusivity must die);
> (c) **Chat/compose feel** — smoother streaming, less boxed-in.
> Suggested order: (b) first (cheapest, highest payoff), then (a), then (c).
> This is an architectural pass, not spot fixes — design it, show the user a
> plan, then implement.

### B. Website scraper → RAG, with progress bar
New ingest source type. Crawl (seed URL + depth/same-domain limits) → extract
(`beautifulsoup4` + `nh3`, already deps) → reuse the **existing** ingest
chunk/embed/progress path in `routes/ingest_routes.py` (same `_state` progress
mechanism the file upload uses). Surface in the RAG Ingest tab + Settings.

### C. Interview mode driven by a real model
Personalizer currently uses 10 hardcoded questions. Replace with: feed the
running model the conversation-so-far and have it ask the next most useful
question (dynamic follow-ups), stream the reply, save answers to memory (the
fix already in place). Requires a configured model endpoint.

### D. ETAs for RL Loop, QLoRA, and RAG ingest
Don't fake a timer — emit **structured progress**. Training scripts and the
ingest worker should log machine-readable lines (`{"step":N,"total":M,"ts":...}`)
instead of prose; one backend helper computes
`ETA = steps_remaining × rolling_avg_step_time`. RL Loop has a known shape
(levels × steps); ingest already tracks files/chunks.

### E. Job supervisor: log / verify / report / RETRY
The reliability layer the user explicitly asked for:
- Structured logging + heartbeats; a single source-of-truth status endpoint/SSE.
- Explicit success markers (training has `[TRAINING_OK]`; add one for ingest).
- **Idempotent, resumable units with retry.** Ingest: per-file/per-chunk retry
  with backoff (today a failure silently marks the file errored). Training:
  **checkpoint-resume** — RL Loop already saves a LoRA adapter per level, so a
  crash at level 3 resumes at level 3, not from scratch.

### F. "Start Ollama Instance" button in Settings — LOOPBACK ONLY
Settings panel: probe `127.0.0.1:11434`, show Running/Stopped, Start/Stop.
- **HARD SECURITY RULE: bind `127.0.0.1` only. Never `OLLAMA_HOST=0.0.0.0`,
  never a tunnel, never external exposure.** ("Don't make it leak.")
- Handle the model-store gotcha above (run as `dari` / set `OLLAMA_MODELS` to the
  drive path, or detect the existing service).

### G. RL Loop Sandbox (new sub-tab of RL Loop) — agentic tool-use training
**Goal:** train the model to *use its tools to build something*, NOT to memorize
— disciplined tool-calls, efficient reasoning/tokens, behaving well under context
compaction, grounding in RAG. This is **agentic RL**.

Design:
- **Sandbox task with a verifiable reward.** "Build something" must become
  checkable (e.g. "write code that passes these tests" / "produce an artifact
  matching a spec"). The sandbox runs the model's output and scores it.
  Verifiable reward is essential.
- **Reward shaping:** verified task success **+** valid tool-calls **−** token
  penalty (this is how you train efficiency/terseness) **+** RAG-grounding bonus.
- **Context compression is mostly a HARNESS feature** — Odysseus already has a
  context compactor (`src/context_compactor.py`, see tests). RL teaches the model
  to behave well under compaction; don't try to RL the mechanism itself.
- **Pragmatic ladder — do NOT start with raw RL:**
  1. **Expert-iteration SFT:** run the agent on tasks, keep trajectories that
     succeeded + were efficient, SFT on those. Cheap, stable, teaches tool-call
     format fast.
  2. **GRPO** on top to squeeze efficiency/reasoning.
- **Honest expectation:** RL won't make a ~3B-active model dramatically smarter.
  It reliably buys *behavior*: tool-call discipline, terser reasoning, better
  RAG-grounding, fewer wasted tokens. Set this expectation with the user.

### Model & training-stack decisions — DECIDED (user, 2026-06)
1. **RL base = `gpt-oss-20b` + Unsloth.** Rebuild `scripts/training/rl_loop.py`
   on Unsloth (currently raw TRL GRPO). Rationale (verified 2026-06):
   21B total / **3.6B active**, **128K context** (YaRN), **native tool use**,
   adjustable reasoning effort (low/med/high — directly the "efficient
   reasoning" knob), ~16GB inference, **GRPO trainable in ~15GB VRAM via
   Unsloth** (FP8 RL, longer context, ready-made reward fns, agentic/ReTool
   multi-turn support).
2. **Upgrade the whole training stack to Unsloth, not just RL:**
   - **QLoRA** (`qlora_merge.py` + SFT paths) → Unsloth (faster, less VRAM).
   - **CPT** (`cpt_train.py`) → Unsloth, and treat CPT as the **raw-knowledge
     injection** path (books/corpora → continued pretraining), distinct from
     the RL/behavior path. The user explicitly wants CPT in the lineup.
3. **One serving model only (gpt-oss-20b), but cross-model distillation:**
   the user wants **every other model to learn from what gpt-oss does**.
   Architecture: the RL Sandbox's **verified successful agent trajectories**
   (tool-calls, reasoning, outcomes) are persisted as a dataset; that dataset
   is then used to **SFT-distill other models** (e.g. the abliterated gemma)
   via the same Unsloth pipeline. So: gpt-oss = the "teacher" that explores
   and earns rewards; everything else = students of its trajectory corpus.
   Build trajectory persistence into the sandbox from day one (cheap if done
   early, painful to retrofit).
> Sources: huggingface.co/openai/gpt-oss-20b ·
> unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune (+ /gpt-oss-reinforcement-learning)

---

## Architecture map (verified 2026-06-12 by walking the tree)

Read this to orient fast; then read the actual files before editing. Counts:
**53 route modules, 91 `src/` modules, ~530 test files.**

### Entry point — `app.py` (~49 KB, the wiring hub)
- Builds the `FastAPI` app, middleware stack (CORS → GZip →
  `SecurityHeadersMiddleware` → `_RequestTimeoutMiddleware` → `AuthMiddleware`),
  mounts `/static` via a `_RevalidatingStatic`, and **wires ~50 routers** through
  `app.include_router(setup_*_routes(...))` calls (lines ~534–752). Lifecycle is a
  modern `_lifespan` context manager (line ~871), NOT `@app.on_event`.
- Page routes (`/`, `/notes`, `/calendar`, `/cookbook`, `/email`, `/memory`,
  `/gallery`, `/tasks`, `/library`, `/backgrounds`, `/command-panel`, `/login`)
  serve the static HTML shells. Health/introspection: `/api/version`,
  `/api/health`, `/api/ready`, `/api/runtime`.
- **Dependency-injection pattern:** most routers are factory functions
  (`setup_xxx_routes(deps)`), so shared singletons (`session_manager`,
  `rag_manager`, `memory_vector`, `task_scheduler`, `webhook_manager`, etc.) are
  constructed once in `app.py` / `src/app_initializer.py` and passed in. To add a
  route, follow that pattern — don't reach for globals.

### Layers
- **`core/`** — foundation: `auth.py`, `session_manager.py`, `database.py`,
  `models.py`, `middleware.py`, `atomic_io.py`, `platform_compat.py`,
  `constants.py`, `exceptions.py`. The stable substrate everything else builds on.
- **`routes/`** (53) — thin HTTP/SSE layer; one module per feature area (chat,
  memory, ingest, training, mcp, email, calendar, research, gallery, tasks,
  session, document, …). **Active fix area:** `training_routes.py`,
  `ingest_routes.py`, `session_routes.py`, `research_routes.py`, `mcp_routes.py`.
- **`src/`** (91) — the brains. Notable for the roadmap:
  - Chat/agent: `chat_handler.py`, `chat_processor.py`, `agent_loop.py`,
    `agent_runs.py`, `ai_interaction.py`, `llm_core.py`, `model_context.py`.
  - Tools: `tool_*` family (`tool_schemas`, `tool_parsing`, `tool_execution`,
    `tool_policy`, `tool_security`, `tool_implementations`, `tool_index`) — the
    tool-use substrate the **RL Sandbox (Roadmap G)** will train against.
  - RAG/memory: `rag_manager.py`, `rag_vector.py`, `rag_singleton.py`,
    `memory_vector.py`, `memory.py`, `memory_provider.py`, `embeddings.py`,
    `embedding_lanes.py`, `chroma_client.py` (talks to ChromaDB on `:8100`).
  - **`context_compactor.py`** — the existing context-compaction harness the
    handoff (Roadmap G) says to teach the model to behave under, NOT to re-RL.
  - Jobs/scheduling: `bg_jobs.py`, `bg_monitor.py`, `task_scheduler.py`,
    `event_bus.py`, `service_health.py`, `readiness.py` — the substrate for the
    **job supervisor (Roadmap E)** and **ETAs (Roadmap D)**.
  - Research/web: `deep_research.py`, `research_handler.py`, `research_utils.py`,
    `url_security.py`, `url_safety.py` — relevant to the **website scraper
    (Roadmap B)**; reuse `url_security` for crawl-target validation.
  - MCP: `mcp_manager.py` (the cancel-scope fix lives here), `mcp_oauth.py`,
    `builtin_mcp.py`.
  - Model discovery / endpoints: `model_discovery.py`, `endpoint_resolver.py`,
    `config.py`, `settings.py` — where the **Ollama "Start" button (Roadmap F)**
    and model-endpoint config belong.
- **`services/`** — heavier subsystem workers, each its own package:
  `tts/`, `stt/`, `search/`, `research/`, `memory/`, `hwfit/`, `docs/`, `shell/`,
  `youtube/`, `faces/`.
- **`scripts/training/`** — the training pipeline (all **currently raw
  TRL/transformers**, the Unsloth migration is NOT started):
  `sft_train.py`, `dpo_train.py`, `cpt_train.py`, `qlora_merge.py`,
  `rl_loop.py` (GRPO via `trl.GRPOTrainer`, confirmed at `rl_loop.py:188,298,312`).
  Launched by `routes/training_routes.py` via `sys.executable`.
- **`scripts/odysseus*`** — a large CLI surface (`odysseus`, `odysseus-mail`,
  `odysseus-notes`, `odysseus-research`, …) plus shell-completion under
  `scripts/_completion/` and shared helpers in `scripts/_lib/`.
- **`static/`** — vanilla-JS PWA (no build step). `index.html` is the main shell;
  `app.js` + `static/js/*.js` (~60 modules) are feature scripts loaded directly.
  `forge.js` = the training/RAG UI ("Forge"). `sw.js`/`manifest.json` = PWA.
  `static/js/MODULE_SUMMARY.md` documents the JS modules — read it before frontend
  work. The roadmap's **UI free-flow pass (A)** centers on `index.html`,
  `sidebar-layout.js`, `section-management.js`, `windowResize.js`.
- **`companion/`** — device-pairing companion (separate router).
- **`integrations/`** — `claude/` and `codex/` plugin + skill definitions
  (`SKILL.md`, `mcp-settings.json`) that let external agents drive Odysseus.
- **`data/`** — local state (sqlite `app.db`, `sessions.json`, `memory.json`,
  `training_state.json`, `settings.json`, `auth.json`, …). **Do not commit.**

### Stack / deps (`requirements.txt`)
FastAPI + uvicorn + pydantic v2 + SQLAlchemy. RAG is **core**, not optional:
`chromadb-client` (HTTP client → standalone ChromaDB on `:8100`) + `fastembed`
(local ONNX embeddings), degrading to keyword fallback if absent. `mcp` for MCP.
Optional (`requirements-optional.txt`): `faster-whisper` (local STT),
`duckduckgo-search`, `PyMuPDF` (AGPL — PDF forms only), `markitdown` (office/epub).
**License posture:** MIT core is guarded — PyMuPDF (AGPL) is quarantined to
optional/form-filling. Keep new heavy/cloud deps out of `requirements.txt`.

### How the roadmap maps onto this tree (quick index for the next instance)
- **B (web scraper→RAG):** new source type in `routes/ingest_routes.py`, reuse its
  `_state` progress mechanism; crawl with `beautifulsoup4`+`nh3` (already deps),
  validate targets via `src/url_security.py`.
- **C (model-driven interview):** `static/js/forge.js` `_buildPersonalizer` +
  `src/memory*`; drive questions through the configured model endpoint.
- **D (ETAs) / E (supervisor):** `scripts/training/*` emit machine-readable
  progress; aggregate in `src/bg_jobs.py`/`bg_monitor.py`/`event_bus.py`; surface
  via one status SSE.
- **F (Ollama button):** `routes/model_routes.py` + `src/model_discovery.py`/
  `endpoint_resolver.py` + Settings UI. **127.0.0.1 only.**
- **G (RL Sandbox):** `scripts/training/rl_loop.py` (→ Unsloth) + the `src/tool_*`
  stack for verifiable tool-use reward; persist trajectories for distillation.

## Suggested order of work
1. **Job #1**: commit + push the two pending fixes; confirm suite green.
2. Ship the low-risk wins first (B scraper, C model interview, F Ollama button,
   then the new Settings surfaces) — these are the visible "app update."
3. Then D (ETAs) + E (job supervisor/retry) as the reliability layer.
4. UI architectural pass (A) — design first, confirm with user, then implement.
5. Training-stack migration to Unsloth (QLoRA, CPT, then RL) + pull
   `gpt-oss-20b` via Ollama/HF.
6. Then G (RL Sandbox) — biggest effort; SFT (expert iteration) before GRPO;
   persist verified trajectories from day one for the distillation pipeline.

All decisions are made — there are no open questions blocking you. Confirm
understanding with the user, then execute.

Keep the test suite green at each step. Add tests for every new route/worker
(the previous instance added route tests for the training/ingest fixes —
match that bar).
