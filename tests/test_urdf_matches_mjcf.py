"""The generated Gazebo URDF must reproduce the training model's kinematics and inertia.

Kinematics: loads ros2/quadrucat_deploy/urdf/go1.urdf back into MuJoCo and compares every
body's pose relative to the trunk at random joint configurations.
Inertia: parsed straight from the URDF XML (MuJoCo's URDF importer ignores the rpy of
<inertial><origin>), and compared as full 3x3 tensors in each link frame.
Skipped without mujoco_playground.
"""
import pathlib
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mujoco_playground")

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "ros2" / "quadrucat_deploy"


def _load_urdf_in_mujoco():
    text = (PKG / "urdf" / "go1.urdf").read_text()
    text = text.replace("package://quadrucat_deploy/meshes/", "")
    # MuJoCo's URDF importer ignores ros2_control/gazebo, but strip them to be safe.
    text = re.sub(r"<ros2_control.*?</ros2_control>", "", text, flags=re.S)
    text = re.sub(r"<gazebo.*?</gazebo>", "", text, flags=re.S)
    text = text.replace("<robot name=\"go1\">",
                        "<robot name=\"go1\"><mujoco><compiler meshdir=\"" + str(PKG / "meshes") +
                        "\" discardvisual=\"false\" fusestatic=\"false\" balanceinertia=\"false\"/></mujoco>")
    return mujoco.MjModel.from_xml_string(text)


def _rpy2mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def test_urdf_inertia_matches_training_model():
    from mujoco_playground import registry

    cfg = registry.get_default_config("Go1JoystickFlatTerrain")
    cfg.impl = "jax"
    ref = registry.load("Go1JoystickFlatTerrain", config=cfg).mj_model
    root = ET.parse(PKG / "urdf" / "go1.urdf").getroot()
    checked = 0
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        b = ref.body(link.get("name")).id
        o = inertial.find("origin")
        xyz = np.array(o.get("xyz").split(), float)
        rot = _rpy2mat(*np.array(o.get("rpy").split(), float))
        i = inertial.find("inertia")
        tensor = rot @ np.diag([float(i.get(k)) for k in ("ixx", "iyy", "izz")]) @ rot.T
        r_ref = _quat2mat(ref.body_iquat[b])
        np.testing.assert_allclose(xyz, ref.body_ipos[b], atol=1e-7)
        np.testing.assert_allclose(tensor, r_ref @ np.diag(ref.body_inertia[b]) @ r_ref.T, atol=1e-8)
        assert float(inertial.find("mass").get("value")) == pytest.approx(ref.body_mass[b], rel=1e-6)
        checked += 1
    assert checked == 13  # trunk + 12 leg links


def _quat2mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def test_urdf_matches_training_model():
    from mujoco_playground import registry

    cfg = registry.get_default_config("Go1JoystickFlatTerrain")
    cfg.impl = "jax"
    ref = registry.load("Go1JoystickFlatTerrain", config=cfg).mj_model
    urdf = _load_urdf_in_mujoco()
    d_ref, d_urdf = mujoco.MjData(ref), mujoco.MjData(urdf)

    names = [ref.joint(int(ref.actuator_trnid[a, 0])).name for a in range(ref.nu)]
    urdf_qadr = [urdf.jnt_qposadr[urdf.joint(n).id] for n in names]
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(ref.jnt_range[1:, 0], ref.jnt_range[1:, 1])
        d_ref.qpos[:] = ref.qpos0
        d_ref.qpos[7:] = q
        d_urdf.qpos[:] = urdf.qpos0
        d_urdf.qpos[urdf_qadr] = q
        mujoco.mj_kinematics(ref, d_ref)
        mujoco.mj_kinematics(urdf, d_urdf)
        trunk_r = ref.body("trunk").id
        p0, r0 = d_ref.xpos[trunk_r], d_ref.xmat[trunk_r].reshape(3, 3)
        trunk_u = urdf.body("trunk").id
        p1, r1 = d_urdf.xpos[trunk_u], d_urdf.xmat[trunk_u].reshape(3, 3)
        for b in range(1, ref.nbody):
            name = ref.body(b).name
            if ref.body_rootid[b] != trunk_r:
                continue
            u = urdf.body(name).id
            np.testing.assert_allclose(r0.T @ (d_ref.xpos[b] - p0), r1.T @ (d_urdf.xpos[u] - p1), atol=1e-6, err_msg=name)
            np.testing.assert_allclose(r0.T @ d_ref.xmat[b].reshape(3, 3), r1.T @ d_urdf.xmat[u].reshape(3, 3), atol=1e-6)

    for n in names:
        jr, ju = ref.joint(n), urdf.joint(n)
        np.testing.assert_allclose(jr.range, ju.range, atol=1e-6)
        assert urdf.dof_frictionloss[ju.dofadr[0]] == pytest.approx(ref.dof_frictionloss[jr.dofadr[0]])
