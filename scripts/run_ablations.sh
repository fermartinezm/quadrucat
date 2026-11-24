#!/usr/bin/env bash
# Reward-term ablations: retrain with one reward scale set to 0 at a time.
#   bash scripts/run_ablations.sh                 # all terms, seed 0
#   SEEDS="0 1 2" TERMS="feet_slip pose" bash scripts/run_ablations.sh
#   CONFIG=configs/smoke.yaml bash scripts/run_ablations.sh   # quick dry run
#   CONFIG=configs/cat_walk.yaml TERMS="direct_register three_foot_support foot_track_width" bash scripts/run_ablations.sh
#   CONFIG=configs/cat_fall.yaml TERMS="land_on_feet air_orientation trunk_contact impact" bash scripts/run_ablations.sh
#
# Term names must match reward_config.scales in the Playground Go1 joystick env
# (an unknown name fails immediately with the list of valid keys).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/base.yaml}"
SEEDS="${SEEDS:-0}"            # use "0 1 2" for the report: single seeds are noisy
TERMS="${TERMS:-tracking_ang_vel lin_vel_z ang_vel_xy orientation pose stand_still termination torques action_rate energy feet_clearance feet_height feet_slip feet_air_time dof_pos_limits}"

for seed in $SEEDS; do
  for term in $TERMS; do
    echo "=== ablate ${term} (seed ${seed}) ==="
    python scripts/train.py --config "$CONFIG" --seed "$seed" --name "ablate_${term}" \
      --set "env.overrides.reward_config.scales.${term}=0.0" \
      --notes "ablation: reward term ${term} removed"
  done
done
