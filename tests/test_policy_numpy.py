import json

import numpy as np

from quadrucat.policy_numpy import NumpyPolicy


def test_numpy_policy_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    sizes = [48, 32, 32, 24]  # obs -> hidden -> hidden -> 2 * action
    arrays = {"obs_mean": rng.normal(size=48), "obs_std": rng.uniform(0.5, 2, size=48)}
    for i in range(3):
        arrays[f"W{i}"] = rng.normal(size=(sizes[i], sizes[i + 1])) * 0.3
        arrays[f"b{i}"] = rng.normal(size=sizes[i + 1]) * 0.1
    meta = {"activation": "swish", "action_size": 12}
    path = tmp_path / "policy.npz"
    np.savez(path, metadata=np.array(json.dumps(meta)), **arrays)

    pol = NumpyPolicy(str(path))
    obs = rng.normal(size=(5, 48))
    x = (obs - arrays["obs_mean"]) / arrays["obs_std"]
    for i in range(3):
        x = x @ arrays[f"W{i}"] + arrays[f"b{i}"]
        if i < 2:
            x = x / (1 + np.exp(-x))
    expected = np.tanh(x[:, :12])
    np.testing.assert_allclose(pol(obs), expected, rtol=1e-6, atol=1e-6)
    assert pol(obs[0]).shape == (12,)
    assert np.all(np.abs(pol(obs)) <= 1)
