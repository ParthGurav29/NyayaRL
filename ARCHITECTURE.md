# ARCHITECTURE — NyayaRL

## Project mission
NyayaRL is an **API-first reinforcement learning (RL) environment** for **Indian legal reasoning**. Agents learn to construct a **deterministic 6-step legal argument chain** (a fixed ordered DAG) from a case file to a final judgment recommendation. The environment is designed for:

- **Deterministic evaluation** (no LLM judges; reproducible rewards).
- **Stateful episodes** behind a **stateless HTTP API**.
- **Schema-first interoperability** via dataclasses/Enums (Pydantic at the API boundary).

## Core reasoning chain (ordered DAG)
Each episode requires completing the following steps in order:

1. `actus_reus` — physical act (evidence anchoring)
2. `mens_rea` — intent (witness anchoring)
3. `linkage` — connect accused to act (chain-of-custody + IPC grounding)
4. `counter_argument` — address opposition (prosecution challenge handling)
5. `ipc_application` — cite IPC sections (grounded in prior steps)
6. `precedent_citation` — align with precedent + final judgment label

The “DAG” aspect is the **dependency structure**: later steps require foundations established earlier (e.g., IPC citations must be applicable/grounded).

## Module map (single source of truth)
This repo is intentionally layered to keep **rules**, **state**, and **transport** separate.

| Module | Role | Stability | Notes |
|---|---|---:|---|
| `nyayarl/models.py` | **Schemas** (dataclasses + Enums) for *all* domain I/O: `Action`, `Observation`, `Verdict`, `StepType`, etc. | Highest | No project imports; shapes only. |
| `nyayarl/argument_chain.py` | **Deterministic validator**: validates one `Action` against `CaseFile` + history; outputs `StepResult` and (optional) prosecution challenge text. | Highest | This is the “rules of the game.” |
| `nyayarl/environment.py` | **State machine**: owns episode lifecycle (`reset`, `step`), stores history, accumulates reward, termination checks, builds `Observation`. | High | Calls the validator; no HTTP knowledge. |
| `server/app.py` | **FastAPI environment API**: session map + endpoints (`/reset`, `/step`, `/health`). | Medium | Pure transport + (currently) stub generator/judge wiring. |
| `app.py` | **HF Spaces entrypoint**: mounts **Gradio at `/`** and FastAPI API at **`/api`** under a single ASGI app. | Medium | Deployment composition; no domain logic. |
| `gradio_app.py` | **Human-mode web UI adapter**: browser UI; keeps per-user episode state in `gr.State`; logs sessions to `human_mode/sessions/`. | Medium | Must not duplicate rules; should delegate to environment methods. |

## End-to-end logic flow (API request → reward → response)
The runtime flow is designed to keep the HTTP layer stateless while the environment is stateful.

```mermaid
flowchart TD
  client[Client] --> reset["POST /reset (session_id, curriculum_level)"]
  reset --> fastapi[FastAPI (server/app.py)]
  fastapi --> envLookup{"Lookup env by session_id"}
  envLookup -->|missing| envCreate["Create NyayaRLEnvironment\n(case_generator, judge)"]
  envLookup -->|present| envGet["Use existing NyayaRLEnvironment"]
  envCreate --> envReset["env.reset(level)"]
  envGet --> envReset
  envReset --> obs0["Observation (case_file, chain_score, is_done=false, ...)"]
  obs0 --> client

  client --> step["POST /step (session_id, Action fields)"]
  step --> fastapi2[FastAPI (server/app.py)]
  fastapi2 --> envLookup2{"Lookup env by session_id"}
  envLookup2 --> envStep["env.step(action)"]
  envStep --> validator["ArgumentChainValidator.validate(...)"]
  validator --> stepResult["StepResult (is_valid, reward, challenge, reason)"]
  stepResult --> rewardAcc["Accumulate reward + termination checks"]
  rewardAcc --> obs1["Observation snapshot"]
  obs1 --> response["Response {observation, step_result, done}"]
  response --> client
```

### What is stateful vs stateless?
- **Stateless (HTTP)**: Each request contains `session_id` and the action payload. FastAPI validates shape and routes the request.
- **Stateful (environment)**: Episode state (case file, submitted steps, reward totals, done flag) lives in a per-session `NyayaRLEnvironment` instance.

## Design philosophies
### Deterministic validation (rule-based, not LLM-evaluated)
NyayaRL uses `ArgumentChainValidator` rules rather than LLM scoring because:

- **Reproducibility**: Same input → same output reward, essential for RL training stability.
- **No hallucinations**: LLM evaluators can invent evidence/legal citations or misread constraints.
- **Clear failure modes**: Invalid steps return specific `failure_reason`, enabling debugging and curriculum design.
- **Unit-testable**: Each step validator is deterministic and can be exhaustively tested.

### Session persistence (multi-user mapping)
The environment API maintains an in-memory mapping:

- Key: `session_id: str`
- Value: `NyayaRLEnvironment` instance

This enables multiple concurrent users (or agents) to run independent episodes. **Important constraint**: the mapping is in-memory only; process restart clears sessions.

### Stateless API vs stateful environment
NyayaRL enforces a strict separation:

- **Transport** (`server/app.py`): parsing, error codes, serialization.
- **State machine** (`nyayarl/environment.py`): lifecycle, history, termination.
- **Rules** (`nyayarl/argument_chain.py`): correctness and reward for each step.

This keeps the environment usable outside HTTP (e.g., training loop) and keeps HTTP predictable.

## Schema summary (developer mental model)
All domain shapes are defined in `nyayarl/models.py`.

### `Action`
Sent by client/agent to progress the chain.

- `step_type: StepType` (must match the next expected step)
- `anchored_evidence_ids: list[str]`
- `anchored_witness_ids: list[str]`
- `cited_ipc_sections: list[str]`
- `judgment: JudgmentLabel | None` (required at step 6)

### `StepResult`
Returned for each submitted action.

- `is_valid: bool`
- `reward: float`
- `failure_reason: str | None`
- `prosecution_challenge: str | None`

### `Observation`
State snapshot returned after every step/reset.

- `case_file: CaseFile`
- `submitted_steps: list[Action]` (valid steps only)
- `prosecution_challenges: list[str]`
- `chain_score: float` (currently: valid_steps / actions_taken)
- `curriculum_level: int`
- `is_done: bool`

### `Verdict` (terminal judge output)
Produced by the environment’s injected `Judge` at episode end (success path).

- `judgment: JudgmentLabel`
- `matched_precedent_id: str`
- `precedent_matched: bool`
- `total_reward: float`
- `prosecution_win_rate: float`

## Where to change what (maintenance guide)
- **Add/modify rules**: `nyayarl/argument_chain.py`
- **Change episode termination / reward aggregation**: `nyayarl/environment.py`
- **Change API shapes**: `nyayarl/models.py` (then update `server/app.py` Pydantic mapping)
- **Change HTTP routes/serialization**: `server/app.py`
- **Change Space deployment composition**: `app.py`, `Dockerfile`
- **Change UI without touching rules**: `gradio_app.py`

