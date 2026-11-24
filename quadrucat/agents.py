"""PPO / SAC hyperparameters and network builders (Brax), shared by train and eval."""
from __future__ import annotations

import functools
from typing import Any, Mapping

from quadrucat.config import apply_overrides

# Fallback SAC values if a config has no `sac:` section (see configs/experiments/sac.yaml).
# Playground ships no SAC config for Go1, so these are a starting point to tune.
SAC_DEFAULTS = dict(
    num_timesteps=20_000_000,
    num_evals=10,
    reward_scaling=1.0,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    discounting=0.97,
    learning_rate=3e-4,
    num_envs=256,
    batch_size=512,
    grad_updates_per_step=8,
    max_replay_size=4_000_000,
    min_replay_size=8192,
    tau=0.005,
    network_factory=dict(hidden_layer_sizes=(256, 256, 256)),
)


def algo_params(cfg: Mapping[str, Any]) -> dict:
    """Resolved hyperparameters as a plain dict (network_factory kwargs included)."""
    algo = cfg.get("algo", "ppo")
    if algo == "ppo":
        from mujoco_playground.config import locomotion_params

        from quadrucat import cat_envs

        cat_envs.register()
        # New cat envs start from the PPO recipe of the Playground env they extend.
        name = cat_envs.PPO_DEFAULTS_FROM.get(cfg["env"]["name"], cfg["env"]["name"])
        params = locomotion_params.brax_ppo_config(name)
        apply_overrides(params, cfg.get("ppo") or {}, strict=False)
        return params.to_dict()
    if algo == "sac":
        params = dict(SAC_DEFAULTS)
        apply_overrides(params, cfg.get("sac") or {}, strict=False)
        return params
    raise ValueError(f"Unknown algo '{algo}' (expected 'ppo' or 'sac')")


def network_factory(cfg: Mapping[str, Any], params: dict):
    """Return (network_factory, network_kwargs) for brax train()."""
    net_kwargs = dict(params.get("network_factory") or {})
    for k in ("policy_hidden_layer_sizes", "value_hidden_layer_sizes", "hidden_layer_sizes"):
        if k in net_kwargs:
            net_kwargs[k] = tuple(net_kwargs[k])
    if cfg.get("algo", "ppo") == "ppo":
        from brax.training.agents.ppo import networks as ppo_networks

        return functools.partial(ppo_networks.make_ppo_networks, **net_kwargs), net_kwargs
    from brax.training.agents.sac import networks as sac_networks

    return functools.partial(sac_networks.make_sac_networks, **net_kwargs), net_kwargs


def make_inference_fn(cfg: Mapping[str, Any], env):
    """Rebuild the policy constructor used at training time, to load saved params.

    `env` must be the same (possibly wrapped) env used for training so that
    observation/action sizes match.
    """
    from brax.training.acme import running_statistics

    params = algo_params(cfg)
    factory, _ = network_factory(cfg, params)
    preprocess = (
        running_statistics.normalize
        if params.get("normalize_observations", True)
        else (lambda x, y: x)
    )
    network = factory(env.observation_size, env.action_size, preprocess_observations_fn=preprocess)
    if cfg.get("algo", "ppo") == "ppo":
        from brax.training.agents.ppo import networks as ppo_networks

        return ppo_networks.make_inference_fn(network)
    from brax.training.agents.sac import networks as sac_networks

    return sac_networks.make_inference_fn(network)
