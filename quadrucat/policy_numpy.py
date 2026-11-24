"""Pure-NumPy inference for an exported policy (no JAX needed).

Used by the ROS2 node. The .npz is produced by scripts/export_policy.py and holds:
  obs_mean, obs_std        observation normalizer
  W0, b0, W1, b1, ...      MLP layers (activation on all but the last)
  metadata                 JSON string: activation, action_size, default_pose,
                           action_scale, ctrl_dt, Kp, Kd, joint_names, obs_layout...
"""
from __future__ import annotations

import json

import numpy as np

_ACTIVATIONS = {
    "swish": lambda x: x / (1.0 + np.exp(-x)),
    "silu": lambda x: x / (1.0 + np.exp(-x)),
    "relu": lambda x: np.maximum(x, 0.0),
    "tanh": np.tanh,
    "elu": lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0.0))),
}


class NumpyPolicy:
    def __init__(self, path: str):
        data = np.load(path, allow_pickle=False)
        self.metadata = json.loads(str(data["metadata"]))
        self.obs_mean = data["obs_mean"].astype(np.float64)
        self.obs_std = data["obs_std"].astype(np.float64)
        n_layers = sum(1 for k in data.files if k.startswith("W"))
        self.layers = [(data[f"W{i}"].astype(np.float64), data[f"b{i}"].astype(np.float64)) for i in range(n_layers)]
        self.act = _ACTIVATIONS[self.metadata.get("activation", "swish")]
        self.action_size = int(self.metadata["action_size"])
        self.obs_size = self.layers[0][0].shape[0]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic action in [-1, 1] for one observation (or a batch)."""
        x = (np.asarray(obs, dtype=np.float64) - self.obs_mean) / self.obs_std
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = self.act(x)
        loc = x[..., : self.action_size]
        return np.tanh(loc)  # mode of brax's tanh-Normal policy
