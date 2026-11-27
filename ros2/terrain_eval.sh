#!/usr/bin/env bash
# Terrain crossings in Gazebo for one run: every terrain type x level, REPS repetitions, then the
# analysis and the versions/ report. Same test as scripts/evaluate_terrain.py in MuJoCo.
#
#   (ROS 2 env sourced, from the repo root)
#   ros2/terrain_eval.sh runs/<run>                 # levels 0 3 6 9, 2 repetitions
#   ros2/terrain_eval.sh runs/<run> "0 5 9" 3
#   TYPES="beam" RATE=250 ros2/terrain_eval.sh runs/<run>   # only beams, physics capped at 1/4 speed
#
# Works for terrain policies (height scan computed by the node from the same map) and for flat
# policies such as cat walk v3 (blind). Logs: /tmp/gz_terrain/<run>/<type>_l<level>_r<k>.csv
set -euo pipefail
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"   # the policy node imports quadrucat from the repo
RUN=$(realpath "$1"); LEVELS=${2:-"0 3 6 9"}; REPS=${3:-2}
TYPES=${TYPES:-"rough stairs_up stairs_down boxes beam"}
SPEED=${SPEED:-0.4}; SEG=${SEG:-14}; PD=${PD:-gazebo}; RATE=${RATE:-}
OUT=${OUT:-/tmp/gz_terrain/$(basename "$RUN")}
[ -f "$RUN/policy.npz" ] || python3 scripts/export_policy.py "$RUN"
mkdir -p "$OUT"
pkill -f "gz sim" 2>/dev/null || true   # another Gazebo on the same topics would mix the odometry
for t in $TYPES; do for l in $LEVELS; do for r in $(seq 0 $((REPS - 1))); do
  csv="$OUT/${t}_l${l}_r${r}.csv"
  [ "$(stat -c%s "$csv" 2>/dev/null || echo 0)" -gt 20000 ] && continue   # done already (resume a batch)
  echo "[terrain_eval] $t level $l rep $r"
  timeout 300 ros2 launch quadrucat_deploy gazebo.launch.py policy_path:="$RUN/policy.npz" headless:=true \
    pd:="$PD" terrain_type:="$t" terrain_level:="$l" eval_commands:="[$SPEED,0,0]" eval_segment_s:="$SEG" \
    log_csv:="$csv" ${RATE:+sim_rate_hz:=$RATE} > "$OUT/${t}_l${l}_r${r}.log" 2>&1 || true
  pkill -f "gz sim" 2>/dev/null || true   # a gz server killed by SIGTERM can outlive the launch
done; done; done
python3 scripts/analyze_gazebo_terrain.py "$OUT"/*.csv --run-dir "$RUN"
python3 scripts/compare_sims.py collect "$RUN"
