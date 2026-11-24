import csv
import pathlib
import sys

import numpy as np

from quadrucat.metrics import summarize, tracking_metrics

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import analyze_gazebo_log  # noqa: E402


def test_perfect_tracking():
    v = np.tile([0.5, 0.0], (2, 100, 1))
    m = tracking_metrics([0.5, 0.0, 0.2], v, np.full((2, 100), 0.2), np.ones((2, 100)), dt=0.02, warmup=10)
    assert m["lin_vel_rmse"] == 0 and m["yaw_rate_rmse"] == 0 and m["fall_rate"] == 0
    assert m["survival_s"] == 2.0


def test_fall_masks_error_after_fall():
    alive = np.ones((1, 100))
    alive[0, 50:] = 0
    v = np.zeros((1, 100, 2))
    v[0, 50:] = 100.0  # garbage after falling must be ignored
    m = tracking_metrics([0.0, 0.0, 0.0], v, np.zeros((1, 100)), alive, dt=0.02, warmup=0)
    assert m["lin_vel_rmse"] == 0 and m["fall_rate"] == 1.0 and m["survival_s"] == 1.0
    assert summarize([m, m])["fall_rate"] == 1.0


def test_analyze_gazebo_csv(tmp_path):
    path = tmp_path / "log.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "segment", "cmd_vx", "cmd_vy", "cmd_wz", "vx", "vy", "wz", "gravity_z", "fallen"])
        for i in range(200):
            seg = i // 100
            cmd = (0.5, 0.0, 0.0) if seg == 0 else (0.0, 0.0, 1.0)
            meas = (0.4, 0.0, 0.0) if seg == 0 else (0.0, 0.0, 0.8)
            w.writerow([i * 0.02, seg, *cmd, *meas, -1.0, 0])
    res = analyze_gazebo_log.analyze(analyze_gazebo_log.load_log(path), warmup_s=0.2)
    assert len(res["per_command"]) == 2
    assert abs(res["per_command"][0]["lin_vel_rmse"] - 0.1) < 1e-9
    assert abs(res["per_command"][1]["yaw_rate_rmse"] - 0.2) < 1e-9
