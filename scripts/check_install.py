"""Sanity check: JAX sees the GPU, Playground loads Go1, and a jitted step runs.

  python scripts/check_install.py
"""
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402

print("jax", jax.__version__, "devices:", jax.devices())
if jax.default_backend() != "gpu":
    print("WARNING: JAX is not using a GPU. Training will be extremely slow. "
          "On Windows use WSL2 + `pip install \"jax[cuda12]==0.9.2\"`.")

from mujoco_playground import registry  # noqa: E402

name = "Go1JoystickFlatTerrain"
env_cfg = registry.get_default_config(name)
if jax.default_backend() != "gpu":
    env_cfg.impl = "jax"  # the default 'warp' backend needs CUDA
env = registry.load(name, config=env_cfg)  # first call downloads MuJoCo Menagerie (needs git + internet)
print(name, "obs:", env.observation_size, "act:", env.action_size, "ctrl_dt:", env.dt)

reset, step = jax.jit(env.reset), jax.jit(env.step)
state = reset(jax.random.PRNGKey(0))
t = time.time()
state = step(state, jp.zeros(env.action_size))
jax.block_until_ready(state.obs)
print(f"first step (incl. compile): {time.time() - t:.1f}s")
t = time.time()
for _ in range(100):
    state = step(state, jp.zeros(env.action_size))
jax.block_until_ready(state.obs)
print(f"100 single-env steps: {time.time() - t:.2f}s  -> OK")
