---
title: "NyayaRL ⚖️ — Teaching an AI to Argue Indian Law (and Survive Cross-Examination)"
thumbnail: /blog/assets/nyayarl/thumbnail.png
authors:
  - user: ParthGurav29
tags:
  - reinforcement-learning
  - multi-agent
  - legal-ai
  - grpo
  - openenv
  - indian-law
---

# NyayaRL ⚖️ — Teaching an AI to Argue Indian Law (and Survive Cross-Examination)

*What happens when you put a defence lawyer, a prosecutor, and a judge — all as AI agents — in the same room, and make them argue real Indian court cases?*

That's essentially what NyayaRL is. And honestly, it's been one of the most fun and unexpectedly difficult problems we've worked on.

---

## The Problem We Kept Thinking About

Legal reasoning in India is genuinely hard. You've got the Indian Penal Code (IPC), thousands of High Court and Supreme Court precedents documented in datasets like [ILDC](https://arxiv.org/abs/2105.04537), contradicting witnesses, missing evidence, and a very strict procedural structure that courts expect arguments to follow.

Most AI systems treat legal tasks as reading comprehension — give the model a fact pattern, get a verdict. That works up to a point, but it misses something crucial: **the adversarial, procedural nature of a real trial**. A defence argument isn't just correct or incorrect in isolation. It has to survive challenges from the prosecution. It has to cite the right IPC sections *after* grounding them in prior steps. It has to align with precedent.

We wanted to build a system that actually *learns* to navigate that process — not just pattern-match to a label.

---

## The Core Idea: Legal Reasoning as a Sequential Decision Problem

We modelled a court argument as a **6-step directed acyclic graph (DAG)** that every case must traverse, in order:

| Step | Legal Concept | What the Agent Must Do |
|------|--------------|------------------------|
| 1 | **Actus Reus** | Anchor the physical act to present physical/forensic evidence |
| 2 | **Mens Rea** | Establish intent using reliable witness testimony |
| 3 | **Linkage** | Connect the accused to the act via documentary/digital chain-of-custody + IPC grounding |
| 4 | **Counter-Argument** | Pre-empt or respond to prosecution challenges |
| 5 | **IPC Application** | Cite applicable IPC sections, grounded in steps 1–3 |
| 6 | **Precedent Citation** | Align with an ILDC precedent and deliver a final judgment (Acquit / Convict / Partial) |

Each step has hard dependency rules — you literally cannot submit step 5 before steps 1–4 are valid. This isn't an arbitrary constraint; it mirrors the logical structure courts actually expect.

The reward signal is entirely **deterministic and rule-based**. No LLM judge, no subjective scoring. Same input → same reward, every time. This was a deliberate choice for reproducibility and training stability.

---

## Three Agents, One Courtroom

This is where the multi-agent design comes in, and it's where things get interesting.

### 🧑‍💼 The Defence Agent

The `DefenceAgent` is a differentiable PyTorch policy that we train with GRPO. At each step, it:

1. Encodes the current observation (step index, available evidence counts, witness reliability stats, chain score, curriculum level) into a hidden state via a two-layer Tanh network.
2. Samples *how many* evidence items, witnesses, and IPC sections to cite (discrete count distributions, masked by what's actually available in the case file).
3. Scores each candidate ID using a per-type masked softmax scorer, then samples without replacement.
4. At step 6, picks a final judgment — greedy at eval time, sampled during training.

The key design insight here is **action masking before sampling**. The policy never sees invalid choices. If step 1 only allows physical/forensic evidence that's actually present in the case file, those are the only candidates in the pool. This prevents the agent from wasting training signal on structurally impossible actions.

### ⚔️ The Prosecution Agent

The `ProsecutionAgent` is adversarial and deterministic (for now). It maintains a **challenge bank** — a set of pre-defined challenges indexed by case template and step type. When the defence submits a step, the prosecution:

1. Identifies gaps (steps not yet completed) and any weak spots in the defence chain.
2. Selects a challenge based on priority (gap > weak step > completed but vulnerable step), severity score, and historical success rate.
3. Records whether the challenge succeeded, updating its own win-rate tracker.

The prosecution isn't trained with gradients — it's more like a structured adversary. But it creates genuine pressure: the defence *has* to anticipate challenges at step 4 (counter-argument), and a high prosecution win rate directly reduces the terminal reward.

### 🧑‍⚖️ The Judge Agent

The `JudgeAgent` scores a completed episode across six dimensions:

- **Chain validity** (3.0 pts) — did all 6 steps complete in order?
- **IPC accuracy** (2.0 pts) — were the right sections cited?
- **Counter-argument coverage** (1.0 pts) — did the defence pre-empt the prosecution?
- **Precedent alignment** (3.0 pts, most important) — does the final judgment match the ILDC ground truth?
- **Efficiency** (1.0 pts) — did the defence complete the chain in minimum steps?
- **Prosecution resistance** (2.0 pts) — how many challenges did the defence survive?

There's also a **perfect chain bonus** (+5.0) when all 6 steps are valid *and* the judgment matches ground truth. That bonus creates a natural incentive for the agent to go for completeness rather than bailing out early.

---

## OpenEnv: Local or Remote, Your Choice

One thing we cared about from the start was making the environment **accessible and composable** — not locked to a specific training script or runtime.

We built the NyayaRL environment as an **OpenEnv-style HTTP API** (served by FastAPI) alongside a **local in-process mode** for training. The mode is controlled by a single YAML file:

```yaml
# nyayarl/openenv.yaml
version: 1
mode: local   # flip to "remote" to point at your HF Space

remote:
  base_url: "https://your-space.hf.space"
  api_path: "/api"
  timeout_seconds: 30
```

When `mode: remote`, any agent that can send HTTP requests — Python, a Jupyter notebook, a different codebase entirely — can interact with the environment through the `/api/reset` and `/api/step` endpoints. Sessions are keyed by a `session_id` UUID, so multiple agents (or multiple humans) can run independent episodes concurrently against the same deployed Space.

When `mode: local`, the environment runs in-process with zero network overhead, which is what the GRPO training loop uses.

This separation — **transport vs. state machine vs. rules** — was intentional. The `ArgumentChainValidator` that enforces the legal rules doesn't know or care whether it's being called by a FastAPI route handler or directly by the training loop. The `NyayaRLEnvironment` state machine doesn't know what HTTP is. Each layer does one thing.

---

## Training with GRPO (No Critic Needed)

We use **Group Relative Policy Optimisation (GRPO)** to train the defence agent. The reason we picked GRPO over standard PPO is simple: we don't want to maintain a separate value network.

The idea is elegant — for each training step, collect G rollouts (we use G=8) from the same curriculum level. Then, for each timestep t across those rollouts, normalise the return-to-go:

```
A_t = (R_t - mean(R_t across group)) / max(std(R_t), ε)
```

That normalised value is the advantage. No bootstrapping. No TD error. No value function. The group *is* the baseline.

Then we apply a clipped surrogate objective (PPO-style) with an entropy bonus to discourage premature policy collapse:

```
loss = -min(ratio * A_t, clip(ratio, 1-ε, 1+ε) * A_t) - β * entropy
```

The entropy coefficient β is annealed over training — high early on to encourage exploration, lower later to let the policy sharpen.

---

## Curriculum: 4 Levels of Difficulty

Not all cases are equal. We designed a 4-level curriculum that automatically promotes the agent when it's ready:

| Level | What Gets Harder |
|-------|-----------------|
| 1 | Simple single-accused case, clear evidence, reliable witnesses |
| 2 | More accused, some evidence missing |
| 3 | Contradicting witnesses, multiple IPC sections required |
| 4 | Full complexity — missing evidence, contradicting witnesses, strict precedent matching |

The curriculum is automatic — the training loop tracks success rate per level and promotes the agent when it's consistently completing chains at the current difficulty. Starting at level 4 would just confuse the agent; starting at level 1 gets it to understand the structure first.

---

## Try It: Human Mode

The Gradio app running on the Space lets you play either role:

- **Play as Defence**: You receive a case file with evidence items, witness statements, and applicable IPC sections. You construct each step by selecting from the available items. The judge scores your step immediately after submission, tells you whether it was valid, what the reward was, and whether the prosecution raised a challenge.

- **Play as Prosecution**: The trained defence agent builds the argument chain, and you choose which challenges to raise at each step.

It's actually a surprisingly good way to develop intuition for what the model has and hasn't learned. When you play against it, you quickly see which steps it handles confidently and where it's still shaky.

---

## What's Next

We're honest about where NyayaRL is today: the core infrastructure (environment, validator, API, agents, training loop) is solid, but some components are still stubs. Specifically:

- **Track 2 — Real Data**: The case generator currently produces synthetic cases. We're working on integrating the ILDC dataset so that cases come from real Indian High Court proceedings. That also means a more faithful prosecution challenge bank conditioned on actual case facts.
- **Track 3 — Training Scale**: We need to harden the GRPO training at scale, add proper evaluation harnesses, and support checkpoint loading from HF Hub for faster cold starts on Spaces.

If you're working on legal NLP, adversarial multi-agent systems, or structured reasoning environments, we'd love to hear from you. The repo is at [github.com/ParthGurav29/NyayaRL](https://github.com/ParthGurav29/NyayaRL).

---

## A Quick Note on Design Choices

A few things we got asked about when we shared early drafts:

**"Why not use an LLM as the judge?"** Reproducibility. An LLM judge on the same input can return different scores across runs, which makes RL training unstable and comparisons between checkpoints unreliable. Deterministic rules give us unit-testable, auditable reward functions.

**"Why not let the agent generate free-form legal text?"** Because we wanted to isolate *structural reasoning* — the ability to understand legal dependencies and procedural ordering — from language generation. The action space (which evidence, witnesses, IPC sections to cite) is already rich enough to require genuine legal reasoning without the confound of fluency.

**"Why GRPO over PPO?"** One less hyperparameter to tune (the value function learning rate), cleaner implementation, and empirically it works well for episode-level tasks where the total episode reward is what matters.

---

*NyayaRL is part of a broader research direction on applying RL to structured expert reasoning — domains where the "rules of the game" are known, adversarial pressure is real, and the quality of reasoning matters beyond surface fluency.*

*— The NyayaRL Team*

---

> **Links**
> - 🤗 [Space / Demo](https://huggingface.co/spaces/ParthGurav29/NyayaRL)
> - 💻 [GitHub Repository](https://github.com/ParthGurav29/NyayaRL)
> - 📄 [Paper / Write-up](https://arxiv.org/)
