# Versions

One folder per policy version: `mujoco/` and `gazebo/` hold the evaluation results per seed, `report.md` compares the simulators. The code is shared (quadrucat/, ros2/, scripts/); a version is a config in configs/.

| version | what | seeds | falls MJX + pushes | register error MJX [cm] | falls Gazebo | terrain crossings MJX | terrain crossings Gazebo |
|---|---|---|---|---|---|---|---|
| [base](base/report.md) | Go1 velocity tracking (configs/base.yaml) | 1 | 0% | 20.7 | 0% | – | – |
| [v1](v1/report.md) | cat walk v1 (configs/cat_walk.yaml) | 1 | 36% | 19.6 | – | – | – |
| [v2](v2/report.md) | cat walk v2 (configs/cat_walk_v2.yaml) | 1 | 0% | 0.7 | – | – | – |
| [v3](v3/report.md) | cat walk v3, the cat walk baseline (configs/cat_walk_v3.yaml) | 3 | 0 ± 0% | 0.9 ± 0.1 | 0 ± 0% | 45% | 42% |
| [terrain](terrain/report.md) | cat walk on rough ground, stairs, drops and beams (configs/cat_terrain.yaml) | 1 | 0% | 0.9 | – | 62% | 60% |
| [terrain_blind](terrain_blind/report.md) | terrain ablation: no height scan (experiments/terrain_blind.yaml) | 1 | 0% | 0.9 | – | 61% | 60% |
