"""Forward kinematics of the Go1 feet from the Gazebo URDF. Pure NumPy + stdlib.

Used to compute foot positions (gait / footprint metrics) offline from logged joint angles
and trunk pose, the same way for MuJoCo and Gazebo logs.
"""
from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

import numpy as np

FEET = ("FR", "FL", "RR", "RL")          # same order as Playground's FEET_SITES
FOOT_IN_CALF = np.array([0.0, 0.0, -0.213])
DEFAULT_URDF = pathlib.Path(__file__).resolve().parents[1] / "ros2" / "quadrucat_deploy" / "urdf" / "go1.urdf"


def _axis_rot(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rodrigues, batched over angle [...] -> [..., 3, 3]."""
    x, y, z = axis / np.linalg.norm(axis)
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    a = np.asarray(angle)[..., None, None]
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)


class FootKinematics:
    def __init__(self, urdf: str | pathlib.Path = DEFAULT_URDF, joint_names: list[str] | None = None):
        joints = {j.get("name"): j for j in ET.parse(urdf).getroot().findall("joint")}
        self.joint_names = joint_names or [f"{f}_{p}_joint" for f in FEET for p in ("hip", "thigh", "calf")]
        self.chains = []
        for foot in FEET:
            chain = []
            for part in ("hip", "thigh", "calf"):
                j = joints[f"{foot}_{part}_joint"]
                o = j.find("origin")
                if np.any(np.abs(np.array((o.get("rpy") or "0 0 0").split(), float)) > 1e-9):
                    raise ValueError("joint frames with rpy are not supported")
                chain.append((self.joint_names.index(j.get("name")), np.array(o.get("xyz").split(), float),
                              np.array(j.find("axis").get("xyz").split(), float)))
            self.chains.append(chain)

    def feet_body(self, q: np.ndarray) -> np.ndarray:
        """q [..., 12] -> foot centres in the trunk frame [..., 4, 3]."""
        q = np.asarray(q, np.float64)
        out = []
        for chain in self.chains:
            rot = np.broadcast_to(np.eye(3), q.shape[:-1] + (3, 3))
            pos = np.zeros(q.shape[:-1] + (3,))
            for idx, xyz, axis in chain:
                pos = pos + rot @ xyz
                rot = rot @ _axis_rot(axis, q[..., idx])
            out.append(pos + rot @ FOOT_IN_CALF)
        return np.stack(out, -2)

    def feet_world(self, q: np.ndarray, trunk_pos: np.ndarray, rot: np.ndarray) -> np.ndarray:
        """rot [..., 3, 3] body->world, trunk_pos [..., 3] -> [..., 4, 3]."""
        return np.asarray(trunk_pos)[..., None, :] + self.feet_body(q) @ np.swapaxes(rot, -1, -2)
