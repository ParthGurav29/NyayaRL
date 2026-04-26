# NyayaRL — Google Colab notebooks

This folder contains Colab-ready notebooks for the only model in NyayaRL that is actually trained with gradients: the PyTorch `DefenceAgent` policy.

> **Heads up:** `JudgeAgent` and `ProsecutionAgent` are deterministic / rule-based — they have no parameters and nothing to "train", so there is no notebook for them.

## Notebooks

| File | Purpose |
| --- | --- |
| `train_defence_agent.ipynb` | Train the `DefenceAgent` from scratch with GRPO. Writes `checkpoints/step_*.pt` and `logs/training.jsonl`. |
| `eval_defence_agent.ipynb`  | Evaluate a saved `step_*.pt` against Track 2 cases (in-distribution and, if available, held-out). |

## What you must upload to Colab

The notebooks expect the project source at `/content/nyayarl-openenv/`. The cleanest way is to zip the repo and upload via Colab's Files panel.

### Option A — upload `nyayarl_openenv.zip`

Build the zip locally with **only the files needed for training**:

```bash
cd /Users/parthmangeshgurav/Desktop/nyayarl-openenv
zip -r nyayarl_openenv.zip \
  nyayarl/ \
  training/ \
  rewards/ \
  data/case_templates/ \
  data/precedents/ \
  data/challenge_bank/ \
  scripts/check_template_overlap.py \
  config.yaml \
  pyproject.toml \
  -x '**/__pycache__/*' '**/.DS_Store'
```

Then in Colab:
1. Open `train_defence_agent.ipynb`.
2. Click the **Files** icon in the left sidebar → upload `nyayarl_openenv.zip` to `/content/`.
3. Run cells in order.

### Option B — clone from GitHub (public repos only)

Skip cell 3a, run cell 3b after editing `REPO_URL`. Make sure the repo includes the data folders listed above.

## Minimum required folders / files inside the project

```
nyayarl/                    # core library (env, agents, models, etc.)
training/                   # run_training.py, grpo_trainer.py, curriculum.py, eval_*.py
rewards/                    # reward_calculator.py, scoring_rubric.py
data/case_templates/*.json  # 26 Track 2 case templates
data/precedents/*.json      # ILDC ground-truth precedents
data/challenge_bank/*.json  # prosecution challenge banks
scripts/check_template_overlap.py   # used by held-out eval
config.yaml                 # default training hyperparameters (notebook overrides paths)
pyproject.toml              # optional, only if you want `pip install -e .`
```

You do **not** need to upload:
- `gradio_app.py`, `lextrust_css.py`, `human_mode/` (UI only)
- `server/` (FastAPI is not needed; training calls the env in-process)
- `tests/`, `.venv/`, `.pytest_cache/`, any existing `checkpoints/`

## Persistent checkpoints

Both notebooks default to `USE_DRIVE = True`. When mounted, all artifacts go to:

```
/content/drive/MyDrive/nyayarl/
├── checkpoints/   # step_*.pt
└── logs/          # training.jsonl
```

Set `USE_DRIVE = False` if you don't want Drive — artifacts then live in the ephemeral `/content/` and disappear when the runtime is recycled.

## Recommended Colab runtime

- Runtime → Change runtime type → **T4 GPU** (or any GPU). The model is small, but GRPO collects `G=16` rollouts per step, so a GPU helps.
- Disk: default Colab disk is fine (project + checkpoints fit comfortably under 1 GB).

## Hyperparameter notes

The notebook writes a fresh `config_colab.yaml`. Edit cell 5 to tune:
- `total_train_steps` (default 2000) — increase for stronger policies.
- `G` (default 16) — group size for GRPO. Larger = more stable, slower.
- `eval_every`, `eval_episodes`, `checkpoint_every` — speed/visibility trade-offs.

The structural hyperparameters (`window_size`, `promotion_threshold`) follow `config.yaml`.
