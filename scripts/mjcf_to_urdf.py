"""Generate the Gazebo Go1 description (URDF + meshes) from the exact MuJoCo model used in training.

Copying masses, inertias, joint frames/axes/limits, joint friction and foot geometry from
the compiled MjModel keeps the two simulators as close as the formats allow. Writes:
  ros2/quadrucat_deploy/urdf/go1.urdf     (with ros2_control, IMU and odometry plugins)
  ros2/quadrucat_deploy/meshes/*.stl      (visual meshes from MuJoCo Menagerie, BSD-3)

  python scripts/mjcf_to_urdf.py            # needs mujoco_playground (any machine, no GPU)

Not representable in URDF/Gazebo, so it is a known (reportable) mismatch:
  * joint armature (rotor inertia, 0.005 kg m^2 in MuJoCo)
  * MuJoCo's soft contact model / solver settings
Kd is NOT put in the URDF as joint damping: the ROS2 node applies it in its PD loop
(MuJoCo implements Kd as joint damping, so this is equivalent).

Placeholders replaced by the launch file at runtime:
  package://quadrucat_deploy/  -> file://<installed share dir>/
  @CONTROLLERS_YAML@           -> path of config/controllers.yaml
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "ros2" / "quadrucat_deploy"
PKG_URI = "package://quadrucat_deploy"
JOINT_VELOCITY_LIMIT = 30.1  # rad/s, Go1 motor spec (not in the MJCF)


# ------------------------------------------------------------------ math helpers
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    return quat_mul(quat_mul(q, np.r_[0.0, v]), quat_conj(q))[1:]


def quat_to_rpy(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def fmt(v):
    vals = [0.0 if abs(float(x)) < 1e-10 else float(x) for x in np.atleast_1d(v)]
    return " ".join(f"{x:.8g}" for x in vals)


def origin(parent, pos, quat):
    ET.SubElement(parent, "origin", xyz=fmt(pos), rpy=fmt(quat_to_rpy(quat)))


# ------------------------------------------------------------------ conversion
def build_urdf(model, mesh_files: dict[int, str], foot_mu: float) -> ET.Element:
    import mujoco

    robot = ET.Element("robot", name="go1")
    root_body = model.body("trunk").id

    def body_name(b):
        return model.body(b).name

    for b in range(1, model.nbody):
        if b != root_body and model.body_rootid[b] != root_body:
            continue  # world geoms etc.
        link = ET.SubElement(robot, "link", name=body_name(b))

        inertial = ET.SubElement(link, "inertial")
        origin(inertial, model.body_ipos[b], model.body_iquat[b])
        ET.SubElement(inertial, "mass", value=f"{model.body_mass[b]:.8g}")
        ixx, iyy, izz = model.body_inertia[b]
        ET.SubElement(inertial, "inertia", ixx=f"{ixx:.8g}", iyy=f"{iyy:.8g}", izz=f"{izz:.8g}",
                      ixy="0", ixz="0", iyz="0")

        for g in np.where(model.geom_bodyid == b)[0]:
            gtype = model.geom_type[g]
            pos, quat = model.geom_pos[g].copy(), model.geom_quat[g].copy()
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                # Compiled mesh geoms are re-centred on the mesh's inertial frame; undo
                # that offset so the original STL file lands in the right place.
                m = model.geom_dataid[g]
                inv = quat_conj(model.mesh_quat[m])
                quat = quat_mul(quat, inv)
                pos = pos - quat_rotate(quat, model.mesh_pos[m])
                vis = ET.SubElement(link, "visual")
                origin(vis, pos, quat)
                geo = ET.SubElement(vis, "geometry")
                ET.SubElement(geo, "mesh", filename=f"{PKG_URI}/meshes/{mesh_files[m]}")
                mat = ET.SubElement(vis, "material", name="dark")
                ET.SubElement(mat, "color", rgba="0.2 0.2 0.2 1")
                continue
            # Collisions: only those that collide in training (the feet), plus the trunk box
            # so that a fallen robot rests on its body instead of sinking through the floor.
            collides = model.geom_contype[g] or model.geom_conaffinity[g]
            is_trunk_box = b == root_body and gtype == mujoco.mjtGeom.mjGEOM_BOX
            if not (collides or is_trunk_box):
                continue
            col = ET.SubElement(link, "collision", name=f"{body_name(b)}_collision_{g}")
            origin(col, pos, quat)
            geo = ET.SubElement(col, "geometry")
            size = model.geom_size[g]
            if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                ET.SubElement(geo, "sphere", radius=f"{size[0]:.8g}")
            elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
                ET.SubElement(geo, "box", size=fmt(2 * size))
            else:
                raise NotImplementedError(f"collision geom type {gtype} on {body_name(b)}")

        if b == root_body:
            continue
        # Joint connecting this body to its parent.
        jids = [j for j in range(model.njnt) if model.jnt_bodyid[j] == b]
        parent = body_name(model.body_parentid[b])
        if len(jids) != 1 or model.jnt_type[jids[0]] != mujoco.mjtJoint.mjJNT_HINGE:
            raise NotImplementedError(f"body {body_name(b)} needs exactly one hinge joint")
        j = jids[0]
        if np.any(np.abs(model.jnt_pos[j]) > 1e-9):
            raise NotImplementedError("joint anchor not at body origin")
        dof = model.jnt_dofadr[j]
        act = [a for a in range(model.nu) if model.actuator_trnid[a, 0] == j]
        effort = float(model.actuator_forcerange[act[0], 1]) if act else 0.0
        joint = ET.SubElement(robot, "joint", name=model.joint(j).name, type="revolute")
        origin(joint, model.body_pos[b], model.body_quat[b])
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=body_name(b))
        ET.SubElement(joint, "axis", xyz=fmt(model.jnt_axis[j]))
        lo, hi = model.jnt_range[j]
        ET.SubElement(joint, "limit", lower=f"{lo:.8g}", upper=f"{hi:.8g}", effort=f"{effort:.8g}",
                      velocity=f"{JOINT_VELOCITY_LIMIT}")
        ET.SubElement(joint, "dynamics", damping="0", friction=f"{model.dof_frictionloss[dof]:.8g}")

    # IMU: fixed link at the MuJoCo 'imu' site (body-aligned).
    imu = model.site("imu")
    ET.SubElement(robot, "link", name="imu_link")
    j = ET.SubElement(robot, "joint", name="imu_joint", type="fixed")
    origin(j, imu.pos, imu.quat)
    ET.SubElement(j, "parent", link=body_name(imu.bodyid[0]))
    ET.SubElement(j, "child", link="imu_link")
    return robot


def add_gazebo(robot: ET.Element, model, default_pose, foot_mu: float) -> None:
    joints = [model.joint(int(model.actuator_trnid[a, 0])).name for a in range(model.nu)]
    rc = ET.SubElement(robot, "ros2_control", name="GazeboSimSystem", type="system")
    hw = ET.SubElement(rc, "hardware")
    ET.SubElement(hw, "plugin").text = "gz_ros2_control/GazeboSimSystem"
    for a, name in enumerate(joints):
        je = ET.SubElement(rc, "joint", name=name)
        lim = float(model.actuator_forcerange[a, 1])
        ci = ET.SubElement(je, "command_interface", name="effort")
        ET.SubElement(ci, "param", name="min").text = f"{-lim:.8g}"
        ET.SubElement(ci, "param", name="max").text = f"{lim:.8g}"
        si = ET.SubElement(je, "state_interface", name="position")
        ET.SubElement(si, "param", name="initial_value").text = f"{default_pose[a]:.8g}"
        ET.SubElement(je, "state_interface", name="velocity")
        ET.SubElement(je, "state_interface", name="effort")

    gz = ET.SubElement(robot, "gazebo")
    p = ET.SubElement(gz, "plugin", filename="gz_ros2_control-system", name="gz_ros2_control::GazeboSimROS2ControlPlugin")
    ET.SubElement(p, "parameters").text = "@CONTROLLERS_YAML@"

    gz = ET.SubElement(robot, "gazebo")
    p = ET.SubElement(gz, "plugin", filename="gz-sim-odometry-publisher-system",
                      name="gz::sim::systems::OdometryPublisher")
    for k, v in dict(odom_frame="odom", robot_base_frame="trunk", odom_publish_frequency="200",
                     odom_topic="odom", dimensions="3").items():
        ET.SubElement(p, k).text = v

    gz = ET.SubElement(robot, "gazebo", reference="imu_link")
    s = ET.SubElement(gz, "sensor", name="imu_sensor", type="imu")
    for k, v in dict(always_on="1", update_rate="500", visualize="false", topic="imu").items():
        ET.SubElement(s, k).text = v
    # imu_link has no mass, so urdf2sdf lumps it into the trunk and moves the sensor there with the
    # imu_joint pose (preserving it as a frame would make urdf2sdf drop the link AND its sensor).

    for foot in ("FR_calf", "FL_calf", "RR_calf", "RL_calf"):
        gz = ET.SubElement(robot, "gazebo", reference=foot)
        ET.SubElement(gz, "mu1").text = f"{foot_mu}"
        ET.SubElement(gz, "mu2").text = f"{foot_mu}"
        # Foot contact sensor (gait metrics). urdf2sdf names the collision <name>_collision.
        coll = robot.find(f"link[@name='{foot}']/collision").get("name")
        s = ET.SubElement(gz, "sensor", name=f"{foot[:2]}_foot_contact", type="contact")
        for k, v in dict(always_on="1", update_rate="1000").items():
            ET.SubElement(s, k).text = v
        c = ET.SubElement(s, "contact")   # gz-sim reads the contact topic from <contact><topic>
        ET.SubElement(c, "collision").text = f"{coll}_collision"
        ET.SubElement(c, "topic").text = f"contact/{foot[:2]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="Go1JoystickFlatTerrain")
    ap.add_argument("--out-dir", default=str(PKG))
    args = ap.parse_args()

    import mujoco  # noqa: F401
    from mujoco_playground import registry
    from mujoco_playground._src import mjx_env

    cfg = registry.get_default_config(args.env)
    cfg.impl = "jax"
    env = registry.load(args.env, config=cfg)
    model = env.mj_model
    floor = model.geom("floor").id
    foot_mu = float(model.geom_friction[floor, 0])  # floor has contact priority in MuJoCo

    out = pathlib.Path(args.out_dir)
    (out / "urdf").mkdir(parents=True, exist_ok=True)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    src = pathlib.Path(mjx_env.MENAGERIE_PATH) / "unitree_go1"
    mesh_files = {}
    for m in range(model.nmesh):
        fname = f"{model.mesh(m).name}.stl"
        shutil.copy(src / "assets" / fname, out / "meshes" / fname)
        mesh_files[m] = fname
    shutil.copy(src / "LICENSE", out / "meshes" / "LICENSE_mujoco_menagerie_go1")

    robot = build_urdf(model, mesh_files, foot_mu)
    add_gazebo(robot, model, np.array(env._default_pose), foot_mu)
    ET.indent(robot)
    header = ("<?xml version=\"1.0\"?>\n<!-- GENERATED by scripts/mjcf_to_urdf.py from the MuJoCo Playground "
              f"'{args.env}' model. Do not edit by hand; re-run the script. -->\n")
    path = out / "urdf" / "go1.urdf"
    path.write_text(header + ET.tostring(robot, encoding="unicode") + "\n", encoding="utf-8")
    print(f"[urdf] wrote {path} (total mass {model.body_subtreemass[model.body('trunk').id]:.3f} kg, "
          f"foot friction {foot_mu})")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
