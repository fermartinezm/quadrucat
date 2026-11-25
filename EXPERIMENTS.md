# Experiments guide

How to run the project's experiments yourselves, what to expect, and how to explain the results.
Nothing here is pre-tuned: `configs/base.yaml` reproduces the reference Playground recipe, and
every change is yours to make and justify.

## 1. Workflow for every experiment

1. **Hypothesis first.** Write down what you change and what you expect before running.
2. **Change one thing.** New YAML in `configs/experiments/` (inherit with `base: ../base.yaml`)
   or `--set key=value` on the command line.
3. **Train with 3 seeds** (`--seed 0/1/2`). One seed is not evidence; PPO results vary a lot.
   ```bash
   for s in 0 1 2; do python scripts/train.py --config configs/experiments/no_domain_rand.yaml --seed $s \
       --notes "H: without DR the policy overfits nominal physics -> worse under pushes and in Gazebo"; done
   ```
4. **Evaluate the same way for every run** (fixed commands, fixed seeds):
   ```bash
   python scripts/evaluate.py runs/<run>                       # nominal
   python scripts/evaluate.py runs/<run> --tag pushes  --set env.overrides.pert_config.enable=true
   python scripts/evaluate.py runs/<run> --tag noise2x --set env.overrides.noise_config.level=2.0
   python scripts/evaluate.py runs/<run> --video               # look at the gait, always
   ```
5. **Compare** (groups seeds automatically, mean ± std):
   ```bash
   python scripts/plot_results.py runs/go1_ppo_base_* runs/go1_ppo_no_dr_* --out-dir figures/dr
   ```
   `diagnostics.png` (learning curves + PPO health), `reward_terms.png` (what the agent
   is actually optimizing) and a table of eval metrics.
6. **Log it** in the table at the end of this file.

A full base run is 200M steps; on one modern GPU expect roughly tens of minutes to an hour.
For quick iterations use `--set ppo.num_timesteps=50_000_000` and say so in the report.

## 2. Reading the diagnostics (why does it work or not?)

| Symptom | Likely cause | What to check / try |
|---|---|---|
| Episode length stays short | robot falls; termination dominates | `termination`, `orientation` terms; video |
| Long episodes, return flat, robot stands still | "lazy" local optimum: standing avoids penalties, tracking reward too weak | `tracking_lin_vel` curve; penalties vs tracking in `reward_terms.png`; lower penalty scales or raise `tracking_sigma` |
| High tracking reward but ugly gait (shuffling, hopping, knees) | reward hacking: missing a shaping term | ablation results, video, `feet_*` terms |
| Return goes up then collapses | policy updates too large | `training/kl_mean` spikes; lower `learning_rate`, fewer `num_updates_per_batch` |
| Action std collapses early, learning stalls | premature convergence | `policy_dist_mean_std`; raise `entropy_cost` |
| Action std stays high, jittery actions | too much exploration / weak action penalty | lower `entropy_cost`; `action_rate` term |
| Value loss keeps growing | critic can't fit returns | reward scale, `discounting`, critic size |
| Good in MJX, bad under pushes / noise / Gazebo | overfitting to nominal simulator | DR on/off, `noise_config`, `pert_config` |

Notes on the numbers:
* Per-step reward = `max(0, dt * Σ scale_k * term_k)`. **The clip at 0 matters**: while
  penalties outweigh rewards, the agent gets no signal about *how* bad things are.
* `eval/episode_reward/<term>` are the per-term sums (scaled, before `dt` and before the clip),
  so they show each term's share even when the total is clipped.
* Use `scripts/evaluate.py` metrics (velocity RMSE, falls), not return, to compare runs with
  *different reward functions*: returns of different rewards are not comparable.

## 3. Planned experiments

### A. Reward ablations (reward hacking)
`SEEDS="0 1 2" bash scripts/run_ablations.sh` removes one term at a time (16 runs × seeds;
pick a subset with `TERMS="feet_slip pose orientation feet_air_time action_rate"`).
For each: did tracking get better/worse, did falls change, **what does the gait look like**?
Typical things to look for (verify, don't assume): without `orientation` the body tilts;
without `pose` odd postures; without `feet_air_time`/`feet_clearance` shuffling or dragging feet;
without `action_rate` jittery motion; without `termination`/`stand_still` behaviour at zero command changes.
Also try *reweighting* instead of removing (e.g. `tracking_sigma`, `feet_slip` ×10).

### B. Domain randomization on vs off
`configs/base.yaml` vs `configs/experiments/no_domain_rand.yaml`. Look at nominal
performance, under pushes/noise, and in sim2sim/Gazebo (different physics). Note: `--set` on the
eval env only changes env config values, not the randomized physics parameters.

### C. PPO vs SAC
`configs/experiments/sac.yaml`. Compare at equal env steps **and** equal wall-clock time
(both are logged: `step`, `wall_time` in `metrics.jsonl`). Caveat: SAC's critic does not see
privileged state. SAC is not tuned: report what you tried.

### D. Robustness
Evaluate every policy with `--tag pushes` and `--tag noise2x` (section 1). Train with pushes
(`configs/experiments/push_training.yaml`) or without noise (`no_obs_noise.yaml`) and compare.

### E. Transfer: MJX → MuJoCo C → Gazebo
```bash
python scripts/export_policy.py runs/<run>
python scripts/sim2sim_mujoco.py runs/<run>                  # same code as the ROS2 node
# Gazebo: see ros2/README.md, then
python scripts/analyze_gazebo_log.py gz_eval.csv --run-dir runs/<run> --tag gazebo
python scripts/plot_results.py runs/<run>                     # table: default / sim2sim / gazebo
```
The middle step separates "our deployment code is wrong" from "the simulators differ".
Compare DR-on vs DR-off policies here: this is the core result of the transfer part.

### Other knobs worth trying (one at a time)
| Knob | Where | Question |
|---|---|---|
| `num_envs` 1024 / 4096 / 8192 | `ppo` | how much does massive parallelism matter (Rudin et al.)? |
| `entropy_cost` 0 / 0.01 / 0.05 | `ppo` | exploration vs convergence |
| `discounting` 0.95 / 0.97 / 0.99 | `ppo` | effective horizon |
| network sizes | `ppo.network_factory` | capacity |
| `value_obs_key: state` | `ppo.network_factory` | value of the privileged critic |
| `action_scale`, `Kp`/`Kd` | `env.overrides` | action space / actuation (affects transfer) |
| `command_config` ranges | `env.overrides` | curriculum / task difficulty |

## 5. Cat-like walk: direct register (`configs/cat_walk.yaml`)

**What a cat does.** In a walk, a cat puts each hind paw almost exactly in the print the
front paw on the same side has just left ("direct register"). It walks slowly, mostly with
three paws on the ground (a lateral-sequence walk: LH, LF, RH, RF) and with a narrow track.

**How it is implemented** (`quadrucat/cat_envs.py`, env `Go1CatWalk` = the joystick task + 3 terms):

| term | when | value | meaning |
|---|---|---|---|
| `direct_register` | each hind touchdown, if `cmd_vx > register_min_vx` | `exp(-(d/register_sigma)^2)` | `d` = horizontal distance between the hind foot and the **front print** of the same side |
| `three_foot_support` | every step, if moving | 1 if exactly 3 feet are down | walk instead of trot/pace |
| `foot_track_width` | every step, if moving | `Σ_stance | |y_foot| - foot_track_halfwidth |` | narrow, cat-like track |

The *front print* follows the front foot while it is on the ground and freezes when it lifts,
so at a hind touchdown it is the spot the front foot vacated. If the front foot is still on
the ground, `d` is about one body length (0.37 m) and the reward is ~0: the term implicitly
asks for the lateral sequence "front lifts, then the hind foot lands in its place".
Everything else (the 16 base terms, PPO, DR, noise) is unchanged from `base.yaml`, and with the
three scales at 0 `Go1CatWalk` is *exactly* the base task (checked in `tests/test_cat_envs.py`).
The actor observation is unchanged (48 values), so export, sim2sim and Gazebo work as before.

**Things to reason about (and write about):**
* *Memory.* The reward depends on where the front foot was, which is not in the actor's
  observation. The policy can still learn it (the gait is periodic and it knows its joint
  angles), and the critic gets the hind-to-print vectors (`register_in_privileged: true`) so the
  value function is not confused by a reward it cannot explain. Ablate that flag.
* *Geometry.* With hips 0.376 m apart, register needs a stride of roughly one body length.
  At 0.3 m/s that means slow, long steps: it competes with `feet_air_time`, `feet_clearance`
  and `max_foot_height`, tuned for the base trot. Watch `hind_steps_per_s` and `duty_factor`.
* *Sparse vs dense.* `direct_register` only fires ~2-3 times/s, so its scale must be larger
  than per-step terms (see the comment in `cat_walk.yaml`). Try 1 / 5 / 20 and `register_sigma`
  0.03 / 0.05 / 0.10 (a wide kernel gives gradient from far away, a narrow one is precise).
* *Reward hacking to look for:* hind foot "stamping" (tiny touchdowns to collect the event
  reward), shuffling with very short steps, standing still at low commands.

**Measure it** (works on any walking run, so the baseline gives the "before" number):
```bash
python scripts/train.py --config configs/cat_walk.yaml --seed 0
python scripts/evaluate_cat.py runs/go1_ppo_base_s0_*            # baseline: no register reward
python scripts/evaluate_cat.py runs/go1_catwalk_s0_* --video
python scripts/plot_results.py runs/go1_ppo_base_* runs/go1_catwalk_* --out-dir figures/catwalk
python scripts/plot_results.py runs/go1_catwalk_* --metric eval/episode_register_hits
```
`evaluate_cat.py` reports per command: `register_err_median` [m], `register_hit_rate`
(touchdowns closer than `register_hit_tol`), `three_foot_frac`, `duty_factor` (walk > 0.5),
`hind_steps_per_s`, plus the usual `lin_vel_rmse` and falls. Training curves log
`eval/episode_hind_touchdowns` and `eval/episode_register_hits` (their ratio = hit rate).

**Reference ("before") numbers**, your current baseline `go1_ppo_base_s0_20261124-163812`
evaluated with `evaluate_cat.py` (CPU, 4 episodes × 10 s per command, so only indicative):
the hind foot lands **0.16-0.23 m** from the front print (median), 0 % hits within 5 cm,
duty factor ≈ 0.5 and only 8-17 % of the time on 3 feet: it trots, as expected.
`cat_walk` should bring `register_err_median` towards a few cm and `three_foot_frac` up.

**Suggested experiments:** base vs `cat_walk` vs `experiments/cat_walk_no_register.yaml`
(does register emerge from slow commands + 3-foot support alone?); ablations of the cat terms:
`CONFIG=configs/cat_walk.yaml TERMS="direct_register three_foot_support foot_track_width" bash scripts/run_ablations.sh`;
sweep `register_sigma`, `direct_register`; how much tracking error does a cat gait cost?

### Cat walk v2 (`configs/cat_walk_v2.yaml`)

Seed 0 of `cat_walk.yaml` walks with 3-foot support (87%) but never registers: median error 24 cm,
0 hits, and 36% falls in the `pushes` evaluation (base policy: 0%). Four changes, all off by default:

| Symptom | Change |
|---|---|
| `exp(-(0.24/0.05)^2)` = 1e-10: no slope for PPO | `register_sigma: 0.06` + `register_sigma_wide: 0.25`, scale 10 |
| Stride 28 cm (speed / step rate per hind foot), needs >= 37.6 cm; paying per touchdown favours short steps | `register_stride_ref: 0.376`, `pose: 0.3` |
| Half of the moving commands go backwards, where register is not rewarded | `forward_prob: 0.6` |
| Falls under kicks | `pert_config.enable: true` while training, 300M steps |

Ablations: `experiments/cat_walk_v2_no_push.yaml` (gait terms only) and
`experiments/cat_walk_sigma02.yaml` (only a wider kernel, sigma 0.2, and scale 10).
The Kaggle notebook evaluates `default` without pushes for every run, so runs are comparable.

## 6. Landing on the feet after a drop (`configs/cat_fall.yaml`)

**Task.** `Go1CatFall` extends Playground's `Go1Getup`: at reset the robot is released at a
random height (`drop_height`, trunk 0.6-1.2 m) with a random orientation (any tilt up to
`max_tilt`, including upside down), random joint angles and a random spin; full-body
collisions are simulated. The policy must land on its feet and then stand (the Getup rewards).

| term | value |
|---|---|
| `land_on_feet` | **one-shot** at the first ground contact: 1 if only feet touch, times `max(0, up_z)` |
| `air_orientation` | per step before the first contact: upright trunk |
| `trunk_contact` | per step the trunk touches the floor (back/side landing) |
| `impact` | per step: trunk acceleration above `impact_threshold` (soft landing) |
| Getup terms | `orientation`, `torso_height`, `posture`, `stand_still`, regularizers |

Observation: Getup's 42 values plus, by default, trunk height (`observe_height`, a downward
range sensor on a real robot) and the accelerometer (`observe_accelerometer`, ~0 in free fall).
Without them the policy cannot know *when* the ground arrives: ablate with
`experiments/cat_fall_no_height_obs.yaml`. This policy is a separate controller; it is not
deployed to Gazebo (a real system would switch to it when the IMU detects free fall).

**Be honest about what is physically possible.** A cat rights itself in the air by twisting
its flexible spine (front and rear halves rotate in opposite directions) and using its tail,
with zero total angular momentum. The Go1 has a rigid trunk and no tail: the only way to rotate
the trunk in the air is to swing the legs (angular-momentum exchange; legs are ~half the mass),
and a drop from 1 m lasts only ~0.3-0.45 s. Realistic expectations:
* small/moderate tilts (< ~45-60°): the policy can correct them in the air and land on its feet;
* near upside down: a full 180° flip in 0.4 s is not realistic; expect it to land on its
  side/back, protect itself (low impact), and then **right itself on the ground** (getup).
  That is still "landing like a cat" in the ways the hardware allows.
* higher drops give more time to rotate but bigger impacts. Initial spin can help or hurt.
So report `land_on_feet_rate` **per initial-tilt bin** (the eval does this) instead of one number,
and compare with the do-nothing baseline. Related work: Kurtz et al., 2022 (*Mini Cheetah, the
Falling Cat*: leg-only mid-air reorientation of a quadruped); Rudin et al., 2022 (*Cat-like
jumping and landing of legged robots in low gravity using deep RL*).

**Run and measure:**
```bash
python scripts/evaluate_cat.py --config configs/cat_fall.yaml --zero-policy   # passive baseline
python scripts/train.py --config configs/cat_fall.yaml --seed 0
python scripts/evaluate_cat.py runs/go1_catfall_s0_* --video
python scripts/evaluate_cat.py runs/go1_catfall_s0_* --tag upside_down --set env.overrides.min_tilt=2.36
python scripts/evaluate_cat.py runs/go1_catfall_s0_* --tag high --set env.overrides.drop_height=[1.5,2.0]
python scripts/plot_results.py runs/go1_catfall_* --metric eval/episode_landed_on_feet
```
Metrics: `land_on_feet_rate` (feet-only first contact with the trunk tilted < 60°; + per tilt bin
0-45/45-90/90-135/135-180°), `feet_first_rate` (feet first in any orientation), `touch_up_z`
(uprightness at first contact), `trunk_hit_rate`, `peak_acc_g`, `upright_end_rate`,
`time_to_stand_s`. Training logs `eval/episode_landed_on_feet` (= success rate, one landing
per episode) and `eval/episode_trunk_contact_steps`.
**Passive baseline (measured, 256 drops, `configs/cat_fall.yaml`, no training):**

| zero action | land on feet | 0-45° | 45-90° | 90-135° | 135-180° | trunk hits |
|---|---|---|---|---|---|---|
| `action_mode: default` (holds home pose) | 30 % | 95 % | 42 % | 1 % | 0 % | 77 % |
| `action_mode: delta` (limp legs) | 14 % | 67 % | 10 % | 0 % | 0 % | 70 % |

So a stiff robot already lands on its feet when it starts nearly upright; the learning
problem is the 45-180° range. That is the number a trained policy has to beat, per bin.

Note: the zero-action baseline depends on `action_mode`: in `delta` mode zero action keeps the
joint *targets* at the current angles, i.e. limp legs; in `default` mode it holds the home pose.

**Suggested experiments:** curriculum (`experiments/cat_fall_easy.yaml`, tilt ≤ 90°, then
fine-tune or compare); landing terms on/off (`experiments/cat_fall_getup_only.yaml`: does
landing on the feet emerge from the Getup reward alone?); observation ablation; `land_on_feet`
scale (one-shot rewards need large scales: ×dt = 0.02); `discounting` 0.95/0.97/0.99 (the
landing is ~20 steps after release); `action_mode` delta vs default; DR on/off.

## 4. Experiment log (fill in)

| Date | Run(s) | Change | Hypothesis | Result (lin RMSE / falls / video) | Explanation |
|---|---|---|---|---|---|
| | `go1_ppo_base_s{0,1,2}` | none (baseline) | reference | | |
