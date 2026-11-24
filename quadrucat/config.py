"""YAML experiment configs with `base:` inheritance and `--set a.b.c=value` overrides.

Only depends on PyYAML so it can be imported without JAX.
"""
from __future__ import annotations

import copy
import pathlib
from typing import Any, Iterable, Mapping

import yaml


def deep_merge(base: Mapping, update: Mapping) -> dict:
    """Recursively merge `update` into a copy of `base`."""
    out = copy.deepcopy(dict(base))
    for k, v in update.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    """cfg['a']['b']['c'] = value for dotted_key 'a.b.c' (creates dicts on the way)."""
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if node.get(k) is None:
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def parse_overrides(overrides: Iterable[str]) -> list[tuple[str, Any]]:
    """['a.b=1', 'c=[1,2]'] -> [('a.b', 1), ('c', [1, 2])]. Values are parsed as YAML."""
    parsed = []
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.path=value")
        key, raw = item.split("=", 1)
        parsed.append((key.strip(), yaml.safe_load(raw)))
    return parsed


def _load_with_bases(path: pathlib.Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = deep_merge(_load_with_bases((path.parent / base).resolve()), cfg)
    return cfg


def load_config(path: str | pathlib.Path, overrides: Iterable[str] = ()) -> dict:
    cfg = _load_with_bases(pathlib.Path(path).resolve())
    for key, value in parse_overrides(overrides):
        set_dotted(cfg, key, value)
    return cfg


def save_config(cfg: Mapping, path: str | pathlib.Path) -> None:
    pathlib.Path(path).write_text(yaml.safe_dump(dict(cfg), sort_keys=False), encoding="utf-8")


def apply_overrides(target, overrides: Mapping, strict: bool = True, prefix: str = "") -> None:
    """Apply a nested dict of overrides onto an ml_collections.ConfigDict (or dict).

    With strict=True, unknown keys raise a KeyError listing the valid keys, which
    catches typos such as `reward_config.scales.feet_slipp`.
    """
    for k, v in (overrides or {}).items():
        full = f"{prefix}{k}"
        if k not in target:
            if strict:
                raise KeyError(f"Unknown config key '{full}'. Valid keys here: {sorted(target.keys())}")
            print(f"[config] adding new key '{full}' = {v!r}")
            target[k] = v
            continue
        current = target[k]
        if isinstance(v, Mapping) and hasattr(current, "keys"):
            apply_overrides(current, v, strict=strict, prefix=full + ".")
        else:
            # ConfigDict is type-checked: keep floats as floats when YAML gives ints.
            if isinstance(current, float) and isinstance(v, int) and not isinstance(v, bool):
                v = float(v)
            target[k] = v


def find_params(run_dir, checkpoint: str | None = None) -> pathlib.Path:
    """Explicit checkpoint, else runs/<run>/params.pkl, else the latest checkpoint."""
    run_dir = pathlib.Path(run_dir)
    if checkpoint:
        return pathlib.Path(checkpoint)
    if (run_dir / "params.pkl").exists():
        return run_dir / "params.pkl"
    ckpts = sorted((run_dir / "checkpoints").glob("params_*.pkl"))
    if not ckpts:
        raise FileNotFoundError(f"No params.pkl or checkpoints in {run_dir}")
    return ckpts[-1]
