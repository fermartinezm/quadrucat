# quadrucat

Reinforcement Learning final project (UPF).

## Goal

Teach a simulated **Unitree Go1** quadruped to walk and follow velocity commands
`(vx, vy, yaw_rate)`, then run the trained policy in a **different simulator (Gazebo, via ROS2)**
and measure how much performance drops when the physics engine changes (sim-to-sim transfer).

## Techniques

- **PPO** (Proximal Policy Optimization) as the main algorithm, with **SAC** as an off-policy baseline (Brax implementations).
- **Massively parallel simulation** on GPU: thousands of environments in MuJoCo Playground (JAX / MJX).
- **Reward shaping** for locomotion (velocity tracking plus penalties on energy, foot slip, posture, etc.).
- **Domain randomization** of physics (friction, masses, joint offsets) for robustness and transfer.
- **Asymmetric actor-critic**: the critic sees privileged simulator state, the actor only sees what a real robot could sense.
- **Sim-to-sim deployment**: the policy is exported to plain NumPy and run in Gazebo through a ROS2 node with the same PD control as in training.

## Plan

1. Train a baseline PPO walking policy in MuJoCo Playground.
2. **Reward ablations**: remove one reward term at a time and see what breaks (reward hacking).
3. **Domain randomization on vs off**.
4. **PPO vs SAC**.
5. **Robustness**: random pushes and extra sensor noise at evaluation time.
6. **Gazebo transfer**: run the same policy in Gazebo and compare tracking error and falls with MuJoCo.

If the Gazebo transfer fails, the MuJoCo experiments are still a complete project and the failure is reported and analysed.

## Status

Base code ready, no real policy trained yet.

- MuJoCo side (train PPO/SAC, evaluate, video, export, sim2sim in plain MuJoCo): tested end to end with tiny CPU runs.
- Gazebo side (generated Go1 URDF, world, controllers, launch, policy node, log analysis): written,
  with the robot model, observation, PD control and metrics checked by tests, but **not yet run in Gazebo**
  (see [`ros2/README.md`](ros2/README.md)).

## Cat mode

Two extra tasks (see [EXPERIMENTS.md](EXPERIMENTS.md) §5-6):
- **`configs/cat_walk.yaml`**: cat-like walk with *direct register* (each hind foot lands where the
  same-side front foot stepped), 3-foot support and a narrow track. Same observation as the base
  policy, so it deploys the same way.
- **`configs/cat_fall.yaml`**: dropped from a height with a random orientation, the robot must land
  on its feet and stand up. Limited by the hardware: no spine and no tail.

`scripts/evaluate_cat.py` measures register error, gait and landing success.

## Running your own experiments

See **[EXPERIMENTS.md](EXPERIMENTS.md)**: workflow, how to read the training diagnostics,
the planned experiments and an experiment log to fill in. All RL knobs (rewards, PPO/SAC
hyperparameters, domain randomization, noise, pushes, network sizes) are listed with their
defaults and comments in [`configs/base.yaml`](configs/base.yaml) and
[`configs/experiments/sac.yaml`](configs/experiments/sac.yaml).

## Repository layout

```
configs/          base.yaml, smoke.yaml, cat_walk.yaml, cat_fall.yaml, smoke_cat.yaml and experiments/
quadrucat/        configs, env construction (+ cat_envs.py: Go1CatWalk, Go1CatFall), Brax networks, and the deployment code shared with ROS2:
                  observation.py (policy input), policy_numpy.py (inference), metrics.py
scripts/          train.py, evaluate.py, evaluate_cat.py, export_policy.py, sim2sim_mujoco.py, mjcf_to_urdf.py,
                  analyze_gazebo_log.py, plot_results.py, run_ablations.sh, check_install.py
ros2/             quadrucat_deploy ROS2 package (Go1 URDF, Gazebo world, controllers, launch, policy node)
tests/            unit tests + consistency tests against the training model (need mujoco_playground)
```

## Setup

JAX on GPU requires **Linux or WSL2** (not native Windows).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
python scripts/check_install.py      # should list a CUDA device
python -m pytest -q
```

**Versions:** MuJoCo Playground 1.x (Nov 2026) was rewritten on MuJoCo Warp + PyTorch and no longer ships
the JAX Go1 tasks, so `requirements.txt` pins `playground==0.2.0` and its matching stack.
On a CPU-only machine (debugging only) add `--set env.overrides.impl=jax` to any script.
For videos on a headless machine set `MUJOCO_GL=egl` (or `osmesa`) and install `ffmpeg`.

## Usage

```bash
python scripts/train.py --config configs/smoke.yaml                     # quick pipeline check
python scripts/train.py --config configs/base.yaml --seed 0             # full training
python scripts/evaluate.py runs/<run> --video                           # tracking error, falls, video
python scripts/evaluate.py runs/<run> --tag pushes --set env.overrides.pert_config.enable=true
python scripts/export_policy.py runs/<run>                              # -> runs/<run>/policy.npz for ROS2
python scripts/sim2sim_mujoco.py runs/<run>                             # same deployment code, plain MuJoCo
python scripts/analyze_gazebo_log.py gz_eval.csv --run-dir runs/<run>   # Gazebo metrics (see ros2/README.md)
bash scripts/run_ablations.sh                                           # reward-term ablations
python scripts/plot_results.py runs/* --out-dir figures                 # diagnostics + reward terms + table
```

Any config value can be overridden with `--set dotted.key=value` (e.g. `--set ppo.learning_rate=1e-4`). Each run saves its resolved config for reproducibility.
All evaluations write `runs/<run>/eval/<tag>.json` with the same metrics (velocity RMSE, yaw-rate RMSE,
fall rate, survival time), so `plot_results.py runs/<run>` prints MJX vs MuJoCo C vs Gazebo side by side.
After changing the robot model, regenerate the Gazebo description with `python scripts/mjcf_to_urdf.py`.

### Training on a Kaggle GPU (automatic)

No local GPU needed: one command packages the code, uploads it as a new version of a private Kaggle
dataset, runs `notebooks/quadrucat_kaggle.ipynb` on a T4 x2, waits and unpacks the results into `runs/`.

```bash
pip install kaggle        # plus a Kaggle API token in ~/.kaggle (kaggle.com/settings > API)
python scripts/kaggle_run.py launch --name check --config configs/cat_walk_v2.yaml --quick   # smoke test
python scripts/kaggle_run.py launch --name catwalk-v2 --config configs/cat_walk_v2.yaml
python scripts/kaggle_run.py status --name catwalk-v2     # also: wait, fetch
```

`--set key=value`, `--seed`, `--notes` and several `--config` files work as in `train.py`. With
`--no-wait` the command returns after the push; different `--name`s run in parallel (max. 2 GPU sessions).
The notebook checks that the mounted dataset holds exactly the code that was uploaded.

## References

- Rudin et al., 2021. *Learning to Walk in Minutes Using Massively Parallel Deep RL.*
- Zakka et al., 2025. *MuJoCo Playground.*
- Tobin et al., 2017. *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.*
- Margolis & Agrawal, 2022. *Walk These Ways.*
- Lee et al., 2020. *Learning Quadrupedal Locomotion over Challenging Terrain.*
- Schulman et al., 2017. *Proximal Policy Optimization Algorithms.*
