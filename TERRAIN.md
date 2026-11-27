# Cat walk on rough terrain, stairs, drops and beams (and the cat fall)

Starting point: **cat walk v3** (`configs/cat_walk_v3.yaml`) walks on flat ground with 0% falls and 94%
direct-register footfalls. Goal: the same cat walk on uneven ground, steps/drops and narrow beams,
plus landing on the feet after a fall.

**The workflow is: edit one file, launch, read one report.**

| what | file / command |
|---|---|
| tune the terrain task | [`configs/cat_terrain.yaml`](configs/cat_terrain.yaml) (every knob commented) |
| tune the fall task | [`configs/cat_fall.yaml`](configs/cat_fall.yaml) |
| train + evaluate on Kaggle | `python scripts/kaggle_run.py launch --name terrain --config configs/cat_terrain.yaml` |
| one change without editing | `... --set env.overrides.terrain.beam_width=[0.6,0.2]` |
| quick pipeline check (~15 min) | `... launch --name check --config configs/cat_terrain.yaml --quick` |
| results | `versions/terrain/report.md` (written automatically when the run is fetched) |

`launch` waits for the run, downloads it into `runs/` and calls `scripts/compare_sims.py collect`,
which copies the results into `versions/<version>/` and rewrites the report. Use a different `--name`
per run: Kaggle runs two at a time.

## 1. Method: what changes compared with the flat cat walk, and why

All of it is in `quadrucat/terrain.py` (the map) and `quadrucat/terrain_env.py` (the env `Go1Terrain`,
a subclass of the cat walk env, so all cat terms still work).

1. **Terrain = one heightfield with a grid of tiles.** One terrain type per row
   (`rough`, `stairs_up`, `stairs_down`, `boxes` = random drops/steps, `beam` = narrow path with a pit on
   both sides), difficulty grows along x (level 0 easy, level 9 hard). Every value is an `[easy, hard]`
   range, e.g. `beam_width: [0.60, 0.15]`. Playground's own rough terrain is a single 5 cm bumpy floor:
   no stairs, no beams, no difficulty scale.
2. **Curriculum** (Rudin et al. 2022). Each env keeps its terrain type; after every episode it goes up a
   level if it walked ≥ `promote_frac` of the commanded distance without falling, down if it fell or
   walked < `demote_frac`. Envs that pass the last level restart at a random level so the easy terrain is
   not forgotten. *Why:* on random levels most early episodes are on terrain where the policy only
   collects falls and learns nothing. The evaluation during training runs on all levels (no
   curriculum, `env.eval_overrides`), so `eval/episode_reward` in `figures/<run>/diagnostics.png` is
   progress on the whole map. (Brax can also log the training envs' level, but its callback fails on
   the Kaggle GPU stack, so it is off.)
3. **Height scan** (`height_scan.in_actor`). 13 × 7 ground heights around the trunk, in the heading
   frame, with noise. *Why:* a blind policy finds a step or the edge of a beam only with its feet; with
   the scan it can lift a foot before a step and put its feet on the beam. The critic always gets the
   clean scan plus foot heights, floor friction and the level (asymmetric actor-critic: privileged
   information only where it costs nothing at deployment).
4. **Terrain-relative foot rewards.** Playground measures swing height as world z, so on a 14 cm step
   the "lift the foot 10 cm" terms would ask for z = 0.10 m. Here `feet_clearance` and `feet_height`
   use the height above the ground under the foot.
5. **`leg_collision` penalty.** The Go1 model collides only with its feet: shins and trunk pass through
   a stair edge. This term charges how deep knees, shins and the trunk bottom go below the ground, so
   the policy cannot use a step a real robot would hit.
6. **Termination**: trunk too close to the ground under it, off a beam (`beam_fall_height`), or off the map.
7. **Heading commands**: the yaw command is a P-controller towards a target heading, so robots walk
   straight lines (on beams: along the beam) and "distance walked" is meaningful for the curriculum.
8. **Domain randomization** is kept from v3 (floor friction 0.4-1.0, masses, CoM, joint offsets, pushes)
   and the scan gets its own noise. The reward changes from v3 are only: `leg_collision -2`,
   `orientation -5 → -2` (stairs need pitch), `lin_vel_z -0.5 → -0.2` (climbing needs vertical speed),
   `max_foot_height 0.10 → 0.12`.

Not done (on purpose): a teacher-student policy (privileged teacher, then a student with observation
history). The asymmetric critic is the cheap version of the same idea; if the blind ablation is far
below the scan policy and the Gazebo result needs a blind policy, distillation is the next step.

## 2. What to tune (in `configs/cat_terrain.yaml`), and what to look for

| knob | default | raise it when / effect |
|---|---|---|
| `terrain.beam_width` | `[0.60, 0.15]` | max beam level stuck low: make the hard end wider (0.2) |
| `terrain.step_height`, `box_height` | `[0.03, 0.14]` | the Go1 standing height is 0.28 m: above ~0.15 it needs a different gait |
| `terrain.type_weights` | beam 1.5, rest 1 | more envs on the terrain type that lags |
| `terrain.promote_frac` / `demote_frac` | 0.5 / 0.25 | level climbs too fast and success drops: raise promote_frac |
| `terrain.init_level_max` | 2 | first levels used before the curriculum moves envs |
| `height_scan.noise` | 0.02 m | robustness to a bad depth sensor; 0 = perfect perception |
| `height_scan.in_actor` | true | false = blind (deploys unchanged on a robot without a height map) |
| `reward_config.scales.leg_collision` | -2 | `reward/leg_collision` stays large: legs going through steps |
| `reward_config.max_foot_height` | 0.12 m | tripping on steps: higher; slow, high-stepping gait: lower |
| `reward_config.scales.orientation` | -2 | body wobbles on flat ground: more negative |
| `reward_config.scales.direct_register` etc. | as v3 | the cat gait; see the `terrain_no_cat` ablation |
| `ppo.num_timesteps` | 400M | `eval/episode_reward` still rising at the end: train longer |
| `naccdmax` | 65536 | "CCD overflow" lines in the log = dropped foot contacts: raise it or lower `ppo.num_envs` |

Ablations (same command, one change each, compare in `versions/summary.md`):

| config | question |
|---|---|
| `experiments/terrain_blind.yaml` | how much of the skill needs the height scan? |
| `experiments/terrain_no_curriculum.yaml` | does the curriculum matter? |
| `experiments/terrain_no_cat.yaml` | does the cat gait (register, narrow track) help on beams? |
| `experiments/terrain_no_leg_collision.yaml` | does the policy cheat through steps without the penalty? |
| `experiments/terrain_no_dr.yaml` | robustness without domain randomization (check it in Gazebo) |

Run several in one Kaggle session: `--config configs/cat_terrain.yaml configs/experiments/terrain_blind.yaml`
(trained one after the other). Use `--seed 1`, `--seed 2` for the seeds of the final numbers.

## 3. How it is evaluated

- **Terrain crossing** (`scripts/evaluate_terrain.py`): for each terrain type and level 0, 3, 6, 9, the
  robot starts at the edge of a tile and must cross it at 0.4 m/s. Report: success rate (falls) per type
  and level, plus `max_level_<type>` (highest level with ≥ 80% success). The v3 flat policy gets the same
  test (blind) as the baseline: `python scripts/evaluate_terrain.py runs/<v3 run> --tag flat_policy`.
- **Flat ground** with the usual `evaluate.py` / `evaluate_cat.py` / footfalls, on the level-0 rough
  tile, so you can check the cat walk did not get worse on flat ground (compare with `versions/v3`).
- **Gazebo**: the same tile as a mesh (identical triangles to MuJoCo's heightfield), the policy node
  computes the height scan from the robot pose on the same map. See `ros2/README.md` ("Terrain"):
  `ros2/terrain_eval.sh runs/<run>` crosses every tile, then
  `python scripts/analyze_gazebo_terrain.py <logs> --run-dir runs/<run>` and `compare_sims.py collect`.

## 4. The fall ("cat fall")

`configs/cat_fall.yaml` (env `Go1CatFall`): dropped from 0.6-1.2 m with a random orientation, land on the
feet, then stand up. Same workflow:
`python scripts/kaggle_run.py launch --name fall --config configs/cat_fall.yaml`; results in
`versions/fall/report.md` (lands on its feet, feet first, trunk hits, peak acceleration, standing at the
end; for random, upside-down and high drops). Knobs: `drop_height`, `max_tilt` (start with 1.57 via
`experiments/cat_fall_easy.yaml`), `observe_height`, the reward scales `land_on_feet`, `air_orientation`,
`trunk_contact`, `impact`. A Go1 has no spine and no tail, so mid-air turning is limited to swinging the
legs: expect good results up to ~90° and partial ones upside down.

## 5. Known limits

- Feet-only collision model (as all Playground Go1 joystick tasks); `leg_collision` is a penalty, not
  physics. Gazebo has full-body collisions, so Gazebo is the honest check for stairs.
- A stair edge is a ramp one grid cell wide (4 cm) in both MuJoCo and Gazebo.
- The height scan in Gazebo comes from the known map and the true pose (no camera): it tests the
  policy, not a perception pipeline.
