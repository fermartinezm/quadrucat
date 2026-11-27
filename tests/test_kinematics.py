import numpy as np

from quadrucat.kinematics import FootKinematics
from quadrucat.metrics import footprint_signals


def test_home_feet_are_symmetric_and_below_trunk():
    q = np.array([0.1, 0.9, -1.8, -0.1, 0.9, -1.8] * 2)
    feet = FootKinematics().feet_body(q)
    assert np.allclose(feet[0] * [1, -1, 1], feet[1])          # FR / FL mirror
    assert np.allclose(feet[:, 2], feet[0, 2]) and -0.30 < feet[0, 2] < -0.22
    assert np.allclose(feet[0, 0] - feet[2, 0], 2 * 0.1881)    # wheelbase


def test_feet_world_batch():
    q = np.zeros((5, 12)); pos = np.tile([1.0, 2, 3], (5, 1)); rot = np.tile(np.eye(3), (5, 1, 1))
    fk = FootKinematics()
    assert np.allclose(fk.feet_world(q, pos, rot), fk.feet_body(q) + [1, 2, 3])


def test_footprint_signals_perfect_register():
    T = 12
    contact = np.ones((T, 4), bool); feet = np.zeros((T, 4, 2))
    feet[:, 0] = [0.3, 0]; contact[6:, 0] = False; feet[6:, 0] = [0.6, 0]   # front lifts, print stays at 0.3
    contact[2:8, 2] = False; feet[8:, 2] = [0.3, 0]                           # hind lands on the print
    dist, td = footprint_signals(feet, contact)
    assert td[:, 0].sum() == 1 and td[8, 0] == 1 and np.isclose(dist[8, 0], 0)
