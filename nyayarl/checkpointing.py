from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class StateDictDiff:
    missing_in_checkpoint: list[str]
    unexpected_in_checkpoint: list[str]
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]]


def _shape(x: Any) -> tuple[int, ...] | None:
    try:
        return tuple(x.shape)  # type: ignore[attr-defined]
    except Exception:
        return None


def inventory_state_dict(sd: dict[str, Any]) -> dict[str, tuple[int, ...] | None]:
    return {k: _shape(v) for k, v in sd.items()}


def diff_checkpoint_vs_model(*, checkpoint_sd: dict[str, Any], model: nn.Module) -> StateDictDiff:
    model_sd = model.state_dict()
    ckpt_keys = set(checkpoint_sd.keys())
    model_keys = set(model_sd.keys())

    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)

    mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for k in sorted(model_keys & ckpt_keys):
        ms = _shape(model_sd[k])
        cs = _shape(checkpoint_sd[k])
        if ms is not None and cs is not None and tuple(ms) != tuple(cs):
            mismatches.append((k, tuple(cs), tuple(ms)))

    return StateDictDiff(
        missing_in_checkpoint=missing,
        unexpected_in_checkpoint=unexpected,
        shape_mismatches=mismatches,
    )


def _is_legacy_bias_only_checkpoint(agent_sd: dict[str, Any]) -> bool:
    # The catastrophic failure mode we’ve seen in this repo: a “checkpoint”
    # that contains only 4 flat bias-like vectors from an older non-module agent.
    legacy_keys = {
        "evidence_count_logits",
        "witness_count_logits",
        "ipc_count_logits",
        "judgment_logits",
        # variants with suffixes sometimes appear
        "evidence_count_logits.bias",
        "witness_count_logits.bias",
        "ipc_count_logits.bias",
        "judgment_logits.bias",
        "evidence_count_logits.weight",
        "witness_count_logits.weight",
        "ipc_count_logits.weight",
        "judgment_logits.weight",
    }
    keys = set(agent_sd.keys())
    return bool(keys) and keys.issubset(legacy_keys)


def _remap_legacy_bias_keys(agent_sd: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "evidence_count_logits": "_head_evidence_k.bias",
        "witness_count_logits": "_head_witness_k.bias",
        "ipc_count_logits": "_head_ipc_k.bias",
        "judgment_logits": "_head_judgment.bias",
        "evidence_count_logits.bias": "_head_evidence_k.bias",
        "witness_count_logits.bias": "_head_witness_k.bias",
        "ipc_count_logits.bias": "_head_ipc_k.bias",
        "judgment_logits.bias": "_head_judgment.bias",
        "evidence_count_logits.weight": "_head_evidence_k.weight",
        "witness_count_logits.weight": "_head_witness_k.weight",
        "ipc_count_logits.weight": "_head_ipc_k.weight",
        "judgment_logits.weight": "_head_judgment.weight",
    }
    out: dict[str, Any] = {}
    for k, v in agent_sd.items():
        out[mapping.get(k, k)] = v
    return out


def _sample_keys_for_value_check(sd: dict[str, Any], *, k: int = 3) -> list[str]:
    # Deterministic: first k tensor keys sorted.
    keys: list[str] = []
    for name in sorted(sd.keys()):
        v = sd[name]
        if torch.is_tensor(v):
            keys.append(name)
        if len(keys) >= k:
            break
    return keys


def _assert_values_match(
    *,
    checkpoint_sd: dict[str, Any],
    model: nn.Module,
    keys: Iterable[str],
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    model_sd = model.state_dict()
    for name in keys:
        if name not in checkpoint_sd:
            raise RuntimeError(f"Value-check key not found in checkpoint state_dict: {name}")
        if name not in model_sd:
            raise RuntimeError(f"Value-check key not found in model state_dict after load: {name}")
        a = checkpoint_sd[name]
        b = model_sd[name]
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            raise RuntimeError(f"Value-check requires tensors, got: {name} ({type(a)} vs {type(b)})")
        if a.shape != b.shape:
            raise RuntimeError(f"Value-check shape mismatch for {name}: ckpt={tuple(a.shape)} model={tuple(b.shape)}")
        if not torch.allclose(a, b, atol=atol, rtol=rtol):
            # Pull a small summary to help debugging without dumping huge tensors.
            diff = (a - b).abs().max().item() if a.numel() else 0.0
            raise RuntimeError(f"Checkpoint load value mismatch for {name}: max_abs_diff={diff}")


@dataclass(frozen=True)
class LoadResult:
    checkpoint_path: Path
    was_legacy_remap: bool
    was_partial_load: bool
    diff: StateDictDiff


def load_defence_agent_checkpoint(
    *,
    checkpoint_path: Path,
    agent: nn.Module,
    map_location: str | torch.device = "cpu",
    require_full_match: bool = True,
    allow_legacy_partial: bool = False,
    do_value_check: bool = True,
) -> LoadResult:
    """
    Fail-loud checkpoint loader for DefenceAgent.

    - By default, requires a full strict load (no missing/unexpected/shape mismatch).
    - Legacy bias-only checkpoints are rejected unless allow_legacy_partial=True.
    - When legacy partial load is allowed, we still surface the full diff and mark partial.
    - Optional numeric value check compares a few tensors post-load.
    """
    checkpoint_path = Path(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict) or "agent_state_dict" not in ckpt:
        raise RuntimeError(f"Invalid checkpoint format (missing agent_state_dict): {checkpoint_path}")

    raw_agent_sd = ckpt["agent_state_dict"]
    if not isinstance(raw_agent_sd, dict):
        raise RuntimeError(f"Invalid agent_state_dict type in checkpoint: {type(raw_agent_sd)}")

    was_legacy = _is_legacy_bias_only_checkpoint(raw_agent_sd)
    agent_sd_to_load = raw_agent_sd
    strict = True

    if was_legacy:
        if not allow_legacy_partial:
            raise RuntimeError(
                "Refusing to load legacy bias-only checkpoint (would random-init most weights). "
                f"checkpoint={checkpoint_path}"
            )
        agent_sd_to_load = _remap_legacy_bias_keys(raw_agent_sd)
        strict = False

    diff = diff_checkpoint_vs_model(checkpoint_sd=agent_sd_to_load, model=agent)
    has_any_mismatch = bool(diff.missing_in_checkpoint or diff.unexpected_in_checkpoint or diff.shape_mismatches)

    if require_full_match and (was_legacy or has_any_mismatch):
        raise RuntimeError(
            "Checkpoint/model mismatch (refusing partial load). "
            f"checkpoint={checkpoint_path} "
            f"missing={len(diff.missing_in_checkpoint)} unexpected={len(diff.unexpected_in_checkpoint)} "
            f"shape_mismatches={len(diff.shape_mismatches)} legacy={was_legacy}"
        )

    msg = agent.load_state_dict(agent_sd_to_load, strict=strict)

    # Value sanity check: ensure the model actually contains the checkpoint values.
    if do_value_check:
        # If strict load, we can compare using the same key names.
        # If legacy remap, compare remapped keys (post-remap) against model.
        keys_for_check = _sample_keys_for_value_check(agent_sd_to_load, k=3)
        _assert_values_match(checkpoint_sd=agent_sd_to_load, model=agent, keys=keys_for_check)

    # Additional paranoia: surface non-empty missing/unexpected from the actual load call.
    missing_keys = list(getattr(msg, "missing_keys", []) or [])
    unexpected_keys = list(getattr(msg, "unexpected_keys", []) or [])
    was_partial = (not strict) or bool(missing_keys or unexpected_keys) or bool(diff.missing_in_checkpoint or diff.unexpected_in_checkpoint or diff.shape_mismatches)

    return LoadResult(
        checkpoint_path=checkpoint_path,
        was_legacy_remap=was_legacy,
        was_partial_load=was_partial,
        diff=diff,
    )

