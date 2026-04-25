# NyayaRL ⚖️ — Indian Legal Reasoning Arena

NyayaRL is a multi-agent reinforcement learning environment for Indian legal reasoning. It models the adversarial process of constructing a legally sound argument chain — from establishing *actus reus* through precedent citation — where a defence agent builds a 6-step argument and a prosecution agent challenges each step. A deterministic judge scores every move against the Indian Penal Code and ILDC precedent database.

## 🎮 How to Play

### Tab 1 — "Play a Case"

1. **Select your role**: Defence (you build the argument chain) or Prosecution (the trained agent builds, you challenge)
2. **Select difficulty**: Levels 1–4 control case complexity — more accused, more IPC sections, missing evidence, contradicting witnesses
3. **Click "Start Case"** to receive a full case file with evidence items, witness statements, and applicable IPC sections
4. **Construct each step** by selecting evidence, witnesses, and IPC sections from the checklists
5. After each step, the judge scores it immediately — showing validity, reward, and any prosecution challenge raised
6. On step 6, select your final judgment (Acquit / Convict / Partial) before submitting

### What the Judge Scores

- **Evidence anchoring**: Is the cited evidence present? Is it the right type for this step?
- **Witness reliability**: Does the cited witness have reliability > 0.5?
- **IPC grounding**: Were cited sections properly established in earlier steps?
- **Chain completeness**: Did all 6 steps validate in sequence?
- **Precedent alignment**: Does the final judgment match the ILDC ground truth?

### Tab 2 — "Session Log"

View all completed sessions with timestamps, modes, rewards, and judgment alignment.

> ⚠️ **Note**: HF Spaces filesystem is ephemeral — session logs do not persist across Space restarts.

## 📦 Project

Built as part of the NyayaRL research project on applying RL to Indian legal reasoning.

- **Environment**: Deterministic 6-step argument chain validator
- **Training**: Group Relative Policy Optimisation (GRPO) — critic-free policy gradient
- **Curriculum**: 4-level progressive difficulty with automatic promotion

### Links

- **Project repository**: https://github.com/ParthGurav29/NyayaRL
- **Paper / write-up**: https://arxiv.org/
