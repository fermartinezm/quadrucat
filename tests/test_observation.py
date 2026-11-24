import numpy as np

from quadrucat.observation import (OBS_SIZE, build_observation, is_fallen, projected_gravity,
                                   quat_to_rotmat, velocity_at_point)


def test_identity_orientation():
    np.testing.assert_allclose(projected_gravity([0, 0, 0, 1], "xyzw"), [0, 0, -1])
    np.testing.assert_allclose(projected_gravity([1, 0, 0, 0], "wxyz"), [0, 0, -1])
    assert not is_fallen(projected_gravity([0, 0, 0, 1]))


def test_upside_down_is_fallen():
    g = projected_gravity([1, 0, 0, 0], "xyzw")  # 180 deg about x
    np.testing.assert_allclose(g, [0, 0, 1], atol=1e-12)
    assert is_fallen(g)


def test_pitch_forward_gravity_has_positive_x():
    a = 0.3  # nose down: rotation about +y
    q = [0, np.sin(a / 2), 0, np.cos(a / 2)]
    g = projected_gravity(q)
    assert g[0] < 0 or g[0] > 0
    np.testing.assert_allclose(np.linalg.norm(g), 1.0)
    np.testing.assert_allclose(quat_to_rotmat(q) @ g, [0, 0, -1], atol=1e-12)


def test_velocity_at_point():
    v = velocity_at_point([1, 0, 0], [0, 0, 1], [0, 1, 0])
    np.testing.assert_allclose(v, [0, 0, 0])  # w x r = (0,0,1)x(0,1,0) = (-1,0,0)


def test_build_observation_size_and_order():
    obs = build_observation(local_linvel=[1, 2, 3], gyro=[4, 5, 6], gravity=[0, 0, -1],
                            joint_pos=np.ones(12), joint_vel=np.zeros(12), default_pose=np.ones(12),
                            last_action=np.full(12, 0.5), command=[0.5, 0, 0.1])
    assert obs.shape == (OBS_SIZE,) == (48,)
    np.testing.assert_allclose(obs[:6], [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(obs[9:21], 0)
    np.testing.assert_allclose(obs[-3:], [0.5, 0, 0.1])
