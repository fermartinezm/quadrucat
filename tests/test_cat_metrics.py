"""Cat-task metrics (pure NumPy, no JAX needed)."""
import numpy as np

from quadrucat.metrics import gait_metrics, landing_metrics, register_metrics


def test_gait_metrics_walk_vs_trot():
    # lateral-sequence walk: one foot in the air at a time -> always 3 feet down
    walk = np.ones((1, 100, 4))
    for t in range(100):
        walk[0, t, t % 4] = 0
    m = gait_metrics(walk, np.ones((1, 100)), dt=0.02, warmup=0)
    assert m["three_foot_frac"] == 1.0 and m["duty_factor"] == 0.75
    # trot: diagonal pairs alternate -> always 2 feet down
    trot = np.zeros((1, 100, 4))
    trot[0, ::2, [0, 3]] = 1
    trot[0, 1::2, [1, 2]] = 1
    m = gait_metrics(trot, np.ones((1, 100)), dt=0.02, warmup=0)
    assert m["two_foot_frac"] == 1.0 and m["three_foot_frac"] == 0.0


def test_register_metrics_only_counts_touchdowns_while_alive():
    dist = np.full((1, 10, 2), 0.3)
    td = np.zeros((1, 10, 2))
    td[0, 2, 0], td[0, 5, 1], td[0, 8, 0] = 1, 1, 1
    dist[0, 2, 0], dist[0, 5, 1], dist[0, 8, 0] = 0.01, 0.02, 0.5
    alive = np.ones((1, 10))
    alive[0, 7:] = 0                     # fell before the 3rd touchdown
    m = register_metrics(dist, td, alive, dt=0.1, tol=0.05, warmup=0)
    assert m["register_hit_rate"] == 1.0
    assert np.isclose(m["register_err_median"], 0.015)
    assert np.isclose(m["hind_steps_per_s"], 2 / 0.7)


def test_landing_metrics_bins_and_rates():
    E, T = 4, 5
    tilt = np.radians([10, 60, 120, 170])
    feet_only = np.zeros((E, T))
    feet_only[0, 2] = feet_only[1, 3] = 1                    # the two small-tilt drops land on feet
    trunk = np.zeros((E, T))
    trunk[3, 3] = 1                                          # upside-down drop hits the back
    up_z = np.full((E, T), 0.95)
    up_z[3] = -0.5
    height = np.full((E, T), 0.27)
    m = landing_metrics(init_tilt=tilt, feet_only_touch=feet_only, touch_up_z=np.array([1, .9, .2, -1.]),
                        trunk_touch=trunk, acc_norm=np.full((E, T), 9.81), up_z=up_z, height=height,
                        landed=np.ones((E, T)), dt=0.02)
    assert m["land_on_feet_rate"] == 0.5 and m["trunk_hit_rate"] == 0.25
    assert m["land_on_feet_rate_0-45deg"] == 1.0 and m["land_on_feet_rate_135-180deg"] == 0.0
    assert m["upright_end_rate"] == 0.75 and np.isclose(m["peak_acc_g"], 1.0)
    # feet first but upside down does not count as landing on the feet
    feet_only[2, 1] = 1
    m2 = landing_metrics(init_tilt=tilt, feet_only_touch=feet_only, touch_up_z=np.array([1, .9, .2, -1.]),
                         trunk_touch=trunk, acc_norm=np.full((E, T), 9.81), up_z=up_z, height=height,
                         landed=np.ones((E, T)), dt=0.02)
    assert m2["land_on_feet_rate"] == 0.5 and m2["feet_first_rate"] == 0.75
    # standing before the first contact (still in the air) is not "standing up"
    landed = np.zeros((E, T))
    landed[:, 3:] = 1
    m3 = landing_metrics(init_tilt=tilt, feet_only_touch=feet_only, touch_up_z=np.array([1, .9, .2, -1.]),
                         trunk_touch=trunk, acc_norm=np.full((E, T), 9.81), up_z=up_z, height=height,
                         landed=landed, dt=0.02)
    assert np.isclose(m3["time_to_stand_s"], 3 * 0.02)
