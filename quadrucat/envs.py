"""Environment construction on top of MuJoCo Playground (Go1 joystick tasks)."""
from __future__ import annotations

from typing import Any, Mapping

import jax

from quadrucat.config import apply_overrides


def make_env(cfg: Mapping[str, Any], extra_overrides: Mapping | None = None):
    """Build a Playground env from the experiment config.

    Returns (env, env_cfg). `cfg["env"]["overrides"]` is a nested dict mirroring the
    Playground ConfigDict, e.g. {"reward_config": {"scales": {"feet_slip": 0.0}}}.
    """
    from mujoco_playground import registry

    from quadrucat import cat_envs

    cat_envs.register()  # adds Go1CatWalk / Go1CatFall to the registry
    name = cfg["env"]["name"]
    env_cfg = registry.get_default_config(name)
    apply_overrides(env_cfg, cfg["env"].get("overrides") or {}, strict=True)
    if extra_overrides:
        apply_overrides(env_cfg, extra_overrides, strict=True)
    env = registry.load(name, config=env_cfg)
    return env, env_cfg


def get_randomizer(cfg: Mapping[str, Any]):
    """Domain-randomization function for brax training, or None when disabled."""
    if not cfg["env"].get("domain_randomization", True):
        return None
    from mujoco_playground import registry

    from quadrucat import cat_envs

    cat_envs.register()

    return registry.get_domain_randomizer(cfg["env"]["name"])


class StateOnlyObs:
    """Expose only obs[key] (e.g. 'state') to the learner.

    The Go1 envs return a dict observation {'state', 'privileged_state'} for
    asymmetric actor-critic PPO. Brax SAC expects a flat array, so we hide the
    dict while keeping the full obs in `info` in case the env needs it on step.
    NOTE: with this wrapper SAC's critic does NOT see privileged state, unlike PPO.
    Mention this when comparing PPO vs SAC.
    """

    _FULL = "_full_obs"

    def __init__(self, env, key: str = "state"):
        self._env = env
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)

    # Playground's domain-randomization wrapper calls `env.unwrapped` and swaps
    # `_mjx_model` on it. Return self so steps still go through this wrapper, and
    # forward the model swap to the real env.
    @property
    def unwrapped(self):
        return self

    @property
    def _mjx_model(self):
        return self._env.unwrapped._mjx_model

    @_mjx_model.setter
    def _mjx_model(self, value):
        self._env.unwrapped._mjx_model = value

    @property
    def observation_size(self):
        size = self._env.observation_size
        if isinstance(size, Mapping):
            size = size[self._key]
        if isinstance(size, (tuple, list)):
            size = size[0] if len(size) == 1 else size
        return size

    def reset(self, rng: jax.Array):
        state = self._env.reset(rng)
        info = dict(state.info)
        info[self._FULL] = state.obs
        return state.replace(obs=state.obs[self._key], info=info)

    def step(self, state, action):
        full = state.info[self._FULL]
        nstate = self._env.step(state.replace(obs=full), action)
        info = dict(nstate.info)
        info[self._FULL] = nstate.obs
        return nstate.replace(obs=nstate.obs[self._key], info=info)
