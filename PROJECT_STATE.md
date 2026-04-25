# PROJECT_STATE — NyayaRL

## Build status (track-wise)
NyayaRL is organized as three development tracks.

| Track | Scope | Status |
|---|---|---:|
| Track 1 | Core infrastructure: schemas, deterministic validator, state-machine environment, FastAPI API | **DONE** |
| Track 2 | Data + agents: ILDC case generation, realistic judges, prosecution challenge banks | **TODO** |
| Track 3 | Training + UI: GRPO training hardening, model packaging, human-in-the-loop web demo | **TODO (partial demo exists)** |

## Implemented feature set (what works today)
### API capabilities (FastAPI)
Implemented in `server/app.py` and backed by `nyayarl/environment.py`.

- **Session handling**: `session_id` maps to an in-memory `NyayaRLEnvironment` instance.
- **Episode lifecycle**:
  - `POST /reset`: starts a new episode at a curriculum level (1–4).
  - `POST /step`: applies an `Action` and returns `{observation, step_result, done}`.
- **Deterministic validation**: strict step ordering + per-step constraints.
- **Reward signaling**:
  - step-level rewards/penalties via `ArgumentChainValidator`
  - terminal adjustments via environment termination logic
- **Episode termination**:
  - success: 6 valid ordered steps
  - stuck: too many consecutive invalid submissions
  - timeout: max actions reached
- **Error semantics**:
  - invalid step type / judgment enum values rejected at request parsing
  - stepping a missing session requires `/reset` first

### Environment capabilities (state machine)
Implemented in `nyayarl/environment.py`.

- Maintains episode state: case file, valid submitted steps, prosecution challenges, reward totals.
- Produces `Observation` snapshots after every `reset` and `step`.
- Embeds the validator; environment is runnable without HTTP (training-ready).

### Deterministic rules (“rules of the game”)
Implemented in `nyayarl/argument_chain.py`.

- Hard step ordering enforced (`StepType` sequence).
- Concrete constraints per step (evidence types, witness reliability, IPC applicability, precedent anchoring, judgment required at step 6).
- Produces explicit `failure_reason` for invalid submissions.

## Current frontier (most recent technical win)
- **HF Spaces-ready deployment composition**: a single ASGI app serves **Gradio at `/`** and mounts the environment API under **`/api`** (port **7860**).
- **Checkpoint-aware demo mode**: Gradio attempts to load latest `step_*.pt` from `/app/checkpoints`; if missing/unloadable, it runs in demo mode so the Space still functions.

## Roadmap
### Track 2 — Data + agents (TODO)
**Goal**: Replace stub case/judge logic with realistic ILDC-backed components and adversarial prosecution behavior.

- **ILDC dataset integration**
  - Case generator that yields `CaseFile` instances with real evidence/witness structures.
  - Precedent IDs grounded in ILDC metadata.
- **Judge implementation**
  - Deterministic scoring rubric aligned with ILDC labels + IPC constraints.
  - Produce `Verdict` with precedent match correctness and meaningful totals.
- **Prosecution agent**
  - “Challenge bank” conditioned on step type, evidence gaps, contradictions.
  - Make `prosecution_challenge` informative and adversarial but deterministic.

### Track 3 — Training + UI (TODO; partial demo exists)
**Goal**: Train a policy that reliably completes the chain and package it for robust inference in Spaces.

- **GRPO trainer hardening**
  - Ensure stable checkpoint formats and versioning.
  - Add evaluation harnesses and regression checks for rule changes.
- **Model packaging**
  - Support pulling checkpoints from HF Hub (optional) and caching at startup.
  - Keep startup < 30s: background load if needed.
- **Human-in-the-loop web demo**
  - Maintain `gr.State` per user session; no global episode state.
  - Session logs as ephemeral artifacts; UI explains non-persistence clearly.

## AI context hook (for future LLMs and coding agents)
### Where to find the “rules”
- **Validator**: `nyayarl/argument_chain.py` is the authoritative spec for step validity + immediate rewards.
- **State machine**: `nyayarl/environment.py` is authoritative for episode bookkeeping, termination, and observation construction.
- **Schemas**: `nyayarl/models.py` is the single source of truth for data shapes; do not invent fields.

### How to interpret logs and failures
- **`failure_reason`** in `StepResult` is the canonical explanation for invalid submissions.
- A `ValueError` originating from the validator usually indicates **wrong step ordering** (submitted `step_type` not equal to expected next step).
- A `RuntimeError` from the environment indicates invalid episode lifecycle usage (e.g., stepping after done; missing reset).

### Safe modification policy
- Rules can change independently of transport/UI. Prefer modifying:
  1) `nyayarl/argument_chain.py` (rules)  
  2) `nyayarl/environment.py` (episode behavior)  
  3) `server/app.py` (API boundary)  
  4) `gradio_app.py` (presentation)  
  5) `Dockerfile`/`app.py` (deployment composition)  

## Known constraints / limitations
- **Ephemeral storage**: container/Space filesystem is not a database; `human_mode/sessions/` logs do not persist across restarts.
- **No persistent DB**: session state is in-process memory only; scaling horizontally would require external state (Redis/DB) or sticky sessions.
- **Stub components still present**:
  - Case generation and judge logic are stubs in multiple entrypoints until Track 2 lands.
- **Checkpoint compatibility**:
  - Checkpoints are `torch.save(...)` artifacts with `agent_state_dict`; loading requires an agent class with matching parameter names/shapes.

