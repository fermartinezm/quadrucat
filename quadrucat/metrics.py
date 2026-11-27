"""Velocity-tracking metrics shared by MuJoCo (MJX), MuJoCo C (sim2sim) and Gazebo.

All inputs are NumPy arrays with a leading episode axis: [episodes, steps, ...].
Errors are pooled over all alive steps after `warmup` steps.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

KEYS = ("lin_vel_rmse", "yaw_rate_rmse", "fall_rate", "survival_s")


def tracking_metrics(command: Sequence[float], lin_vel_xy: np.ndarray, yaw_rate: np.ndarray,
                     alive: np.ndarray, dt: float, warmup: int = 50) -> dict:
    """command=(vx, vy, wz); lin_vel_xy [E,T,2] body frame; yaw_rate [E,T]; alive [E,T] in {0,1}."""
    lin_vel_xy, yaw_rate, alive = (np.asarray(x, dtype=np.float64) for x in (lin_vel_xy, yaw_rate, alive))
    if lin_vel_xy.ndim == 2:  # single episode
        lin_vel_xy, yaw_rate, alive = lin_vel_xy[None], yaw_rate[None], alive[None]
    mask = alive[:, warmup:]
    denom = max(mask.sum(), 1.0)
    lin_err = ((lin_vel_xy[:, warmup:] - np.asarray(command[:2])) ** 2).sum(-1)
    yaw_err = (yaw_rate[:, warmup:] - command[2]) ** 2
    return {
        "command": [float(c) for c in command],
        "lin_vel_rmse": float(np.sqrt((lin_err * mask).sum() / denom)),
        "yaw_rate_rmse": float(np.sqrt((yaw_err * mask).sum() / denom)),
        "fall_rate": float(1.0 - alive[:, -1].mean()),
        "survival_s": float(alive.sum(1).mean() * dt),
    }


def summarize(per_command: list[dict]) -> dict:
    return {k: float(np.mean([r[k] for r in per_command])) for k in KEYS}


def format_row(row: dict) -> str:
    return (f"cmd={row['command']}  lin_rmse={row['lin_vel_rmse']:.3f}  yaw_rmse={row['yaw_rate_rmse']:.3f}  "
            f"falls={row['fall_rate']:.2f}  survival={row['survival_s']:.1f}s")


# ---------------------------------------------------------------------------
# Cat tasks (scripts/evaluate_cat.py). Pure NumPy so they can be unit-tested.
# ---------------------------------------------------------------------------

def gait_metrics(contact: np.ndarray, alive: np.ndarray, dt: float, warmup: int = 50) -> dict:
    """contact [E,T,4] (FR, FL, RR, RL) in {0,1}; alive [E,T].

    duty_factor       mean fraction of time a foot is on the ground (walk > 0.5 > trot/run)
    three_foot_frac   fraction of time with exactly 3 feet down (lateral-sequence walk)
    two_foot_frac     fraction with exactly 2 feet down (trot / pace / walk transitions)
    flight_frac       fraction with 0 or 1 foot down
    """
    contact, alive = np.asarray(contact, np.float64), np.asarray(alive, np.float64)
    if contact.ndim == 2:
        contact, alive = contact[None], alive[None]
    c, m = contact[:, warmup:], alive[:, warmup:]
    denom = max(m.sum(), 1.0)
    n = c.sum(-1)
    return {
        "duty_factor": float((c.mean(-1) * m).sum() / denom),
        "three_foot_frac": float(((n == 3) * m).sum() / denom),
        "two_foot_frac": float(((n == 2) * m).sum() / denom),
        "flight_frac": float(((n <= 1) * m).sum() / denom),
    }


def register_metrics(dist: np.ndarray, touchdown: np.ndarray, alive: np.ndarray, dt: float,
                     tol: float = 0.05, warmup: int = 50) -> dict:
    """dist [E,T,2] hind-to-front-print distance [m] at each step; touchdown [E,T,2] in {0,1}.

    register_err_median   median distance at hind touchdowns [m] (direct register -> ~0)
    register_hit_rate     fraction of hind touchdowns closer than `tol`
    hind_steps_per_s      hind touchdowns per second (both hind feet)
    """
    dist, touchdown, alive = (np.asarray(x, np.float64) for x in (dist, touchdown, alive))
    if dist.ndim == 2:
        dist, touchdown, alive = dist[None], touchdown[None], alive[None]
    sel = (touchdown[:, warmup:] * alive[:, warmup:, None]) > 0
    d = dist[:, warmup:][sel]
    t_alive = max(alive[:, warmup:].sum() * dt, 1e-9)
    return {
        "register_err_median": float(np.median(d)) if d.size else float("nan"),
        "register_hit_rate": float((d < tol).mean()) if d.size else float("nan"),
        "hind_steps_per_s": float(d.size / t_alive),
    }


def footprint_signals(feet_xy: np.ndarray, contact: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Offline copy of the Go1CatWalk bookkeeping: (dist [T,2], hind touchdown [T,2]) for register_metrics.

    feet_xy [T,4,2] world, contact [T,4] bool, both FR, FL, RR, RL at the control rate.
    """
    contact = np.asarray(contact).astype(bool)
    T = len(contact)
    dist, td = np.zeros((T, 2)), np.zeros((T, 2))
    air, last = np.zeros(4), np.zeros(4, bool)
    prints, valid = np.zeros((2, 2)), np.zeros(2, bool)
    for t in range(T):
        c = contact[t]
        first = (air > 0) & (c | last)      # Playground's first_contact
        air = (air + 1.0) * ~c
        last = c
        td[t] = first[2:] & valid
        dist[t] = np.linalg.norm(feet_xy[t, 2:] - prints, axis=-1)
        prints = np.where(c[:2, None], feet_xy[t, :2], prints)
        valid = valid | c[:2]
    return dist, td


TILT_BINS_DEG = (0, 45, 90, 135, 180)


def landing_metrics(*, init_tilt: np.ndarray, feet_only_touch: np.ndarray, touch_up_z: np.ndarray,
                    trunk_touch: np.ndarray, acc_norm: np.ndarray, up_z: np.ndarray, height: np.ndarray,
                    landed: np.ndarray, dt: float, touch_up_z_min: float = 0.5, upright_up_z: float = 0.9,
                    stand_height: float = 0.22) -> dict:
    """Per-episode arrays for the drop task. init_tilt [E] rad; touch_up_z [E]; others [E,T].

    land_on_feet_rate   first ground contact was feet only AND the trunk was upright enough
                        (up_z > touch_up_z_min, 0.5 = tilted less than 60°)
    feet_first_rate     first contact feet only, whatever the orientation (legs can reach the
                        floor even upside down, so this alone is not "on its feet")
    touch_up_z          mean trunk up-axis z at first contact (1 upright, -1 upside down)
    trunk_hit_rate      episodes where the trunk touched the floor at any time
    peak_acc_g          mean over episodes of the peak trunk acceleration [g]
    upright_end_rate    standing upright at the end of the episode
    time_to_stand_s     mean time until first upright+standing (episodes that made it)
    land_on_feet_rate_<a>-<b>deg   success rate binned by the initial tilt
    """
    E = len(init_tilt)
    feet_first = np.asarray(feet_only_touch).sum(1) > 0
    feet_ok = feet_first & (np.asarray(touch_up_z) > touch_up_z_min)
    # standing only counts after the first ground contact (at release it is high and may be upright)
    standing = (np.asarray(up_z) > upright_up_z) & (np.asarray(height) > stand_height) & (np.asarray(landed) > 0)
    first = np.where(standing.any(1), standing.argmax(1), -1)
    out = {
        "land_on_feet_rate": float(feet_ok.mean()),
        "feet_first_rate": float(feet_first.mean()),
        "touch_up_z": float(np.mean(touch_up_z)),
        "trunk_hit_rate": float((np.asarray(trunk_touch).sum(1) > 0).mean()),
        "peak_acc_g": float(np.asarray(acc_norm).max(1).mean() / 9.81),
        "upright_end_rate": float(standing[:, -1].mean()),
        "time_to_stand_s": float(first[first >= 0].mean() * dt) if (first >= 0).any() else float("nan"),
    }
    tilt_deg = np.degrees(np.asarray(init_tilt))
    for lo, hi in zip(TILT_BINS_DEG[:-1], TILT_BINS_DEG[1:]):
        sel = (tilt_deg >= lo) & (tilt_deg < hi + (1e-6 if hi == 180 else 0))
        out[f"land_on_feet_rate_{lo}-{hi}deg"] = float(feet_ok[sel].mean()) if sel.any() else float("nan")
    out["episodes"] = E
    return out
