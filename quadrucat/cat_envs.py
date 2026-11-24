"""Cat-inspired Go1 tasks built on top of the MuJoCo Playground 0.2.0 Go1 envs.

Two new environments are registered in the Playground registry (call `register()`):

Go1CatWalk  (subclass of the Go1 joystick task, same 48-value actor observation)
    Velocity tracking plus optional "cat gait" reward terms:
      direct_register     hind foot lands where the same-side front foot last stood
      three_foot_support  3 feet on the ground (a walk, not a trot / gallop)
      foot_track_width    feet close to a narrow lateral track (cats walk "on a line")
    All new scales default to 0, so Go1CatWalk with default config == Go1JoystickFlatTerrain.
    The actor observation is unchanged, so export / sim2sim / ROS2 deployment still work.

Go1CatFall  (subclass of the Go1 getup task, full-body collisions)
    The robot is dropped from a random height with a random orientation, joint angles and
    angular velocity, and must land on its feet (then stand up). New reward terms:
      land_on_feet     one-shot reward at the first ground contact: only feet touch, scaled
                       by how upright the trunk is at that instant
      air_orientation  upright trunk while still in the air
      trunk_contact    trunk touching the floor (landing on the back / side)
      impact           trunk acceleration above a threshold (hard landing)
    All new scales default to 0 and `drop_from_height_prob`/`settle_time` reproduce Go1Getup
    when left at the Getup values; configs/cat_fall.yaml sets the cat values.

Every value is a normal Playground config key, so it can be changed from YAML or with
`--set env.overrides.reward_config.scales.direct_register=2.0` (unknown keys still raise).
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict

# Which Playground env provides the PPO defaults (network sizes, etc.) for each new env.
PPO_DEFAULTS_FROM = {
    "Go1CatWalk": "Go1JoystickFlatTerrain",
    "Go1CatFall": "Go1Getup",
}

# Feet order in Playground (consts.FEET_SITES): FR, FL, RR, RL.
# Hind feet [RR, RL] are compared with the same-side front feet [FR, FL].
FRONT, HIND = slice(0, 2), slice(2, 4)

# Trunk collision geometry of go1_mjx_fullcollisions.xml: two cylinders along body x.
_TRUNK_CYL_Y = (-0.04, 0.04)
_TRUNK_CYL_RADIUS = 0.058
_TRUNK_CYL_HALF_LEN = 0.125


# =============================================================================
# Pure helpers (also used by the tests)
# =============================================================================

def register_reward(hind_xy, front_print_xy, touchdown, sigma):
    """exp(-(d / sigma)^2) for each hind touchdown, d = |hind - same-side front print|.

    hind_xy, front_print_xy: [2, 2] world xy; touchdown: [2] {0,1}. Returns (reward, d).
    """
    d = jp.linalg.norm(hind_xy - front_print_xy, axis=-1)
    return jp.sum(jp.exp(-jp.square(d / sigma)) * touchdown), d


def update_front_print(front_print_xy, valid, front_xy, front_contact):
    """Front footprint = position of the front foot while it is on the ground.

    While the front foot is in stance the print follows it; when it lifts, the print stays
    at the spot it left. So at a hind touchdown the target is the spot the front foot vacated.
    """
    c = front_contact[:, None]
    return jp.where(c, front_xy, front_print_xy), valid | front_contact


def trunk_lowest_z(trunk_pos, trunk_rot):
    """Lowest world z of the two trunk collision cylinders (axis = body x)."""
    axis = trunk_rot[:, 0]
    az = jp.abs(axis[2])
    lows = []
    for y in _TRUNK_CYL_Y:
        center = trunk_pos + trunk_rot @ jp.array([0.0, y, 0.0])
        lows.append(center[2] - _TRUNK_CYL_HALF_LEN * az
                    - _TRUNK_CYL_RADIUS * jp.sqrt(jp.clip(1.0 - az * az, 0.0, 1.0)))
    return jp.minimum(lows[0], lows[1])


def random_tilted_quat(rng, max_tilt, min_tilt=0.0):
    """Random yaw, and a tilt of the body up-axis away from world up, uniformly distributed
    on the sphere between `min_tilt` and `max_tilt` (max_tilt = pi: any orientation incl.
    upside down). Returns (quat wxyz, tilt angle)."""
    k_yaw, k_cos, k_dir = jax.random.split(rng, 3)
    yaw = jax.random.uniform(k_yaw, minval=-jp.pi, maxval=jp.pi)
    cos_t = jax.random.uniform(k_cos, minval=jp.cos(max_tilt), maxval=jp.cos(min_tilt))
    tilt = jp.arccos(jp.clip(cos_t, -1.0, 1.0))
    phi = jax.random.uniform(k_dir, minval=-jp.pi, maxval=jp.pi)
    q_yaw = jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])
    # tilt about a horizontal axis (cos phi, sin phi, 0)
    s = jp.sin(tilt / 2)
    q_tilt = jp.array([jp.cos(tilt / 2), s * jp.cos(phi), s * jp.sin(phi), 0.0])
    return _quat_mul(q_tilt, q_yaw), tilt


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return jp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


# =============================================================================
# Go1CatWalk
# =============================================================================

def catwalk_default_config() -> config_dict.ConfigDict:
    from mujoco_playground._src.locomotion.go1 import joystick

    cfg = joystick.default_config()
    s = cfg.reward_config.scales
    s.direct_register = 0.0
    s.three_foot_support = 0.0
    s.foot_track_width = 0.0
    r = cfg.reward_config
    r.register_sigma = 0.05        # [m] width of the register kernel exp(-(d/sigma)^2)
    r.register_min_vx = 0.1        # [m/s] only reward register when walking forward faster than this
    r.register_hit_tol = 0.05      # [m] a touchdown closer than this counts as a "hit" (metric only)
    r.foot_track_halfwidth = 0.13  # [m] target |y| of stance feet in the body frame (Go1 default ~0.13)
    cfg.register_in_privileged = True   # critic sees hind-to-print vectors (actor never does)
    return cfg


def _catwalk_class():
    from mujoco_playground._src.locomotion.go1 import joystick

    class CatWalk(joystick.Joystick):
        """Go1 joystick + cat gait terms. See module docstring."""

        def __init__(self, config=None, config_overrides=None):
            super().__init__(task="flat_terrain", config=config or catwalk_default_config(),
                             config_overrides=config_overrides)

        def reset(self, rng):
            state = super().reset(rng)
            info = dict(state.info)
            info["front_print"] = jp.zeros((2, 2))
            info["print_valid"] = jp.zeros(2, dtype=bool)
            info["register_dist"] = jp.zeros(2)
            info["hind_touchdown"] = jp.zeros(2)
            metrics = dict(state.metrics)
            metrics["hind_touchdowns"] = jp.zeros(())
            metrics["register_hits"] = jp.zeros(())
            obs = self._add_register_obs(state.obs, state.data, info)
            return state.replace(obs=obs, info=info, metrics=metrics)

        def step(self, state, action):
            state = super().step(state, action)
            state.metrics["hind_touchdowns"] = jp.sum(state.info["hind_touchdown"])
            tol = self._config.reward_config.register_hit_tol
            state.metrics["register_hits"] = jp.sum(
                state.info["hind_touchdown"] * (state.info["register_dist"] < tol))
            return state

        def _get_obs(self, data, info):
            obs = super()._get_obs(data, info)
            if "front_print" not in info:   # during super().reset(); added in reset()
                return obs
            return self._add_register_obs(obs, data, info)

        def _add_register_obs(self, obs, data, info):
            if not self._config.register_in_privileged:
                return obs
            rot = data.xmat[self._torso_body_id]
            hind = data.site_xpos[self._feet_site_id][HIND]
            delta = jp.concatenate([hind[:, :2] - info["front_print"], jp.zeros((2, 1))], -1)
            delta_body = (delta @ rot)[:, :2]   # rot.T @ v for each row
            extra = jp.concatenate([delta_body.ravel(), info["print_valid"].astype(jp.float32)])
            return {**obs, "privileged_state": jp.concatenate([obs["privileged_state"], extra])}

        def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
            rewards = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
            rc = self._config.reward_config
            cmd = info["command"]
            moving = jp.linalg.norm(cmd) > 0.01
            feet = data.site_xpos[self._feet_site_id]

            # direct register
            touchdown = (first_contact[HIND] & info["print_valid"]).astype(jp.float32)
            gate = cmd[0] > rc.register_min_vx
            r_reg, d = register_reward(feet[HIND, :2], info["front_print"], touchdown, rc.register_sigma)
            rewards["direct_register"] = r_reg * gate
            info["register_dist"] = d
            info["hind_touchdown"] = touchdown * gate
            info["front_print"], info["print_valid"] = update_front_print(
                info["front_print"], info["print_valid"], feet[FRONT, :2], contact[FRONT])

            # walk: exactly three feet on the ground
            rewards["three_foot_support"] = (jp.sum(contact) == 3) * moving

            # narrow track: lateral foot position (body frame) of stance feet
            trunk_pos = data.xpos[self._torso_body_id]
            rot = data.xmat[self._torso_body_id]
            feet_body = (feet - trunk_pos) @ rot
            err = jp.abs(jp.abs(feet_body[:, 1]) - rc.foot_track_halfwidth)
            rewards["foot_track_width"] = jp.sum(err * contact) * moving
            return rewards

    return CatWalk


# =============================================================================
# Go1CatFall
# =============================================================================

def catfall_default_config() -> config_dict.ConfigDict:
    from mujoco_playground._src.locomotion.go1 import getup

    cfg = getup.default_config()
    cfg.drop_from_height_prob = 1.0   # always dropped (Getup: 0.6, else starts standing)
    cfg.settle_time = 0.0             # no passive settling: control starts in the air (Getup: 0.5 s)
    cfg.drop_height = [0.6, 1.2]      # trunk height at release [m], uniform
    cfg.max_tilt = float(np.pi)       # max angle between body up and world up at release [rad]
    cfg.min_tilt = 0.0                # min angle (e.g. 2.4 to evaluate only near upside-down drops)
    cfg.init_angvel = 0.5             # initial trunk angular velocity U(-a, a) per axis [rad/s]
    cfg.init_linvel = 0.5             # initial trunk linear velocity U(-a, a) per axis [m/s]
    cfg.joint_init = "uniform"        # "uniform" over the joint range (Getup) | "default" pose + noise
    cfg.action_mode = "delta"         # "delta": target = q + scale*a (Getup) | "default": q_default + scale*a
    cfg.observe_height = False        # add trunk height (a downward range sensor) to the actor obs
    cfg.observe_accelerometer = False  # add the IMU accelerometer (reads ~0 in free fall) to the actor obs
    cfg.terminate_on_trunk_contact = False
    # Warp (GPU) buffer for convex-convex (CCD) contacts, summed over all parallel envs.
    # The full-collision Go1 needs CCD for leg-leg / leg-floor pairs; Warp's default is far too
    # small for thousands of envs and floods the output with "CCD overflow" (dropped contacts).
    cfg.naccdmax = 4 * 8192
    cfg.noise_config.scales.height = 0.02
    cfg.noise_config.scales.accelerometer = 0.5
    s = cfg.reward_config.scales
    s.land_on_feet = 0.0
    s.air_orientation = 0.0
    s.trunk_contact = 0.0
    s.impact = 0.0
    cfg.reward_config.impact_threshold = 30.0   # [m/s^2] accelerometer norm above this is penalized (~3 g)
    cfg.reward_config.trunk_contact_margin = 0.005  # [m] trunk closer than this to the floor = contact
    cfg.reward_config.land_upright_up_z = 0.5  # metric only: "on its feet" needs up_z > this at touchdown
    return cfg


def _catfall_class():
    from mujoco import mjx
    from mujoco_playground._src import mjx_env
    from mujoco_playground._src.locomotion.go1 import getup
    from mujoco_playground._src.locomotion.go1 import go1_constants as consts

    class CatFall(getup.Getup):
        """Dropped with random orientation; land on the feet, then stand. See module docstring."""

        def __init__(self, config=None, config_overrides=None):
            super().__init__(config=config or catfall_default_config(), config_overrides=config_overrides)

        def _post_init(self):
            super()._post_init()
            m = self._mj_model
            self._torso_body_id = m.body(consts.ROOT_BODY).id
            self._feet_sensor_adr = np.array([m.sensor_adr[i] for i in self._feet_floor_found_sensor])
            # contact sensors for non-foot geoms (front legs only in the Playground XML)
            names = [m.sensor(i).name for i in range(m.nsensor)]
            body_sensors = [n for n in names if n.endswith("_floor_found") and n.split("_floor")[0] not in consts.FEET_GEOMS]
            self._body_sensor_adr = np.array([m.sensor_adr[m.sensor(n).id] for n in body_sensors])

        # ---------------- reset ----------------
        def _drop_qpos(self, rng):
            c = self._config
            k_h, k_q, k_j = jax.random.split(rng, 3)
            qpos = jp.zeros(self.mjx_model.nq)
            qpos = qpos.at[2].set(jax.random.uniform(k_h, minval=c.drop_height[0], maxval=c.drop_height[1]))
            quat, tilt = random_tilted_quat(k_q, c.max_tilt, c.min_tilt)
            qpos = qpos.at[3:7].set(quat)
            if c.joint_init == "uniform":
                q = jax.random.uniform(k_j, (12,), minval=self._lowers, maxval=self._uppers)
            elif c.joint_init == "default":
                q = jp.clip(self._default_pose + jax.random.uniform(k_j, (12,), minval=-0.2, maxval=0.2),
                            self._lowers, self._uppers)
            else:
                raise ValueError(f"joint_init must be 'uniform' or 'default', got {c.joint_init!r}")
            return qpos.at[7:].set(q), tilt

        def reset(self, rng):
            c = self._config
            rng, k1, k2, k3 = jax.random.split(rng, 4)
            drop_qpos, tilt = self._drop_qpos(k2)
            dropped = jax.random.bernoulli(k1, c.drop_from_height_prob)
            qpos = jp.where(dropped, drop_qpos, self._init_q)
            tilt = jp.where(dropped, tilt, 0.0)
            qvel = jp.zeros(self.mjx_model.nv)
            ka, kb = jax.random.split(k3)
            qvel = qvel.at[0:3].set(jax.random.uniform(ka, (3,), minval=-c.init_linvel, maxval=c.init_linvel))
            qvel = qvel.at[3:6].set(jax.random.uniform(kb, (3,), minval=-c.init_angvel, maxval=c.init_angvel))
            data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:],
                                     impl=self.mjx_model.impl.value,
                                     naconmax=c.naconmax, naccdmax=min(c.naccdmax, c.naconmax),
                                     njmax=c.njmax)
            data = mjx.forward(self.mjx_model, data)
            if self._settle_steps > 0:
                data = mjx_env.step(self.mjx_model, data, qpos[7:], self._settle_steps)
                data = data.replace(time=0.0)
            info = {
                "rng": rng,
                "last_act": jp.zeros(self.mjx_model.nu),
                "last_last_act": jp.zeros(self.mjx_model.nu),
                "landed": jp.zeros((), dtype=bool),
                "init_tilt": tilt,
                "init_height": qpos[2],
                "first_touch": jp.zeros(()),
                "feet_only_touch": jp.zeros(()),
                "touch_up_z": jp.zeros(()),
                "trunk_touch": jp.zeros(()),
                "acc_norm": jp.zeros(()),
            }
            metrics = {f"reward/{k}": jp.zeros(()) for k in c.reward_config.scales.keys()}
            for k in ("landed_on_feet", "first_touch", "trunk_contact_steps"):
                metrics[k] = jp.zeros(())
            obs = self._get_obs(data, info)
            reward, done = jp.zeros(2)
            return mjx_env.State(data, obs, reward, done, metrics, info)

        # ---------------- step ----------------
        def step(self, state, action):
            c = self._config
            base = self._default_pose if c.action_mode == "default" else state.data.qpos[7:]
            motor_targets = base + action * c.action_scale
            data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

            info = state.info
            feet = data.sensordata[self._feet_sensor_adr] > 0
            body = (data.sensordata[self._body_sensor_adr] > 0) if len(self._body_sensor_adr) else jp.zeros(0, bool)
            trunk = trunk_lowest_z(data.xpos[self._torso_body_id], data.xmat[self._torso_body_id]) \
                < c.reward_config.trunk_contact_margin
            touching = jp.any(feet) | jp.any(body) | trunk
            first_touch = touching & ~info["landed"]
            feet_only = jp.any(feet) & ~jp.any(body) & ~trunk
            up_z = -self.get_gravity(data)[2]       # 1 upright, -1 upside down
            acc_norm = jp.linalg.norm(self.get_accelerometer(data))
            info["first_touch"] = first_touch.astype(jp.float32)
            info["feet_only_touch"] = (first_touch & feet_only).astype(jp.float32)
            info["touch_up_z"] = jp.where(first_touch, up_z, info["touch_up_z"])
            info["trunk_touch"] = trunk.astype(jp.float32)
            info["acc_norm"] = acc_norm

            obs = self._get_obs(data, info)
            done = self._get_termination(data)
            if c.terminate_on_trunk_contact:
                done = done | trunk

            rewards = self._get_reward(data, action, info, state.metrics, done)
            rc = c.reward_config
            rewards["land_on_feet"] = info["feet_only_touch"] * jp.clip(up_z, 0.0, 1.0)
            rewards["air_orientation"] = (~info["landed"]) * self._reward_orientation(self.get_gravity(data))
            rewards["trunk_contact"] = trunk.astype(jp.float32)
            rewards["impact"] = jp.maximum(acc_norm - rc.impact_threshold, 0.0) / 9.81
            rewards = {k: v * rc.scales[k] for k, v in rewards.items()}
            reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

            info["landed"] = info["landed"] | touching
            info["last_last_act"] = info["last_act"]
            info["last_act"] = action
            for k, v in rewards.items():
                state.metrics[f"reward/{k}"] = v
            state.metrics["landed_on_feet"] = info["feet_only_touch"] * (up_z > rc.land_upright_up_z)
            state.metrics["first_touch"] = info["first_touch"]
            state.metrics["trunk_contact_steps"] = info["trunk_touch"]
            return state.replace(data=data, obs=obs, reward=reward, done=jp.float32(done))

        # ---------------- observation ----------------
        def _get_obs(self, data, info):
            obs = super()._get_obs(data, info)
            c = self._config
            extra = []
            if c.observe_height:
                h = data.xpos[self._torso_body_id][2:3]
                info["rng"], k = jax.random.split(info["rng"])
                extra.append(h + (2 * jax.random.uniform(k, (1,)) - 1) * c.noise_config.level
                             * c.noise_config.scales.height)
            if c.observe_accelerometer:
                a = self.get_accelerometer(data)
                info["rng"], k = jax.random.split(info["rng"])
                extra.append(a + (2 * jax.random.uniform(k, (3,)) - 1) * c.noise_config.level
                             * c.noise_config.scales.accelerometer)
            if not extra:
                return obs
            state = jp.concatenate([obs["state"], *extra])
            return {"state": state, "privileged_state": jp.concatenate([obs["privileged_state"], *extra])}

    return CatFall


# =============================================================================
# Registration
# =============================================================================
_REGISTERED = False


def register() -> None:
    """Register Go1CatWalk / Go1CatFall in mujoco_playground's registry (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    from mujoco_playground._src import locomotion
    from mujoco_playground._src.locomotion.go1 import randomize

    locomotion.register_environment("Go1CatWalk", _catwalk_class(), catwalk_default_config)
    locomotion.register_environment("Go1CatFall", _catfall_class(), catfall_default_config)
    locomotion._randomizer["Go1CatWalk"] = randomize.domain_randomize
    locomotion._randomizer["Go1CatFall"] = randomize.domain_randomize
    _REGISTERED = True


def task_kind(env_name: str) -> str:
    """'fall' for drop/landing tasks, 'walk' for velocity-tracking tasks."""
    return "fall" if env_name in ("Go1CatFall", "Go1Getup") else "walk"


def describe(env_name: str) -> dict[str, Any]:
    return {"name": env_name, "kind": task_kind(env_name), "ppo_defaults_from": PPO_DEFAULTS_FROM.get(env_name, env_name)}
