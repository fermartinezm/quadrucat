"""Go1Terrain: the cat walk (Go1CatWalk) on rough ground, stairs, drops and narrow beams.

What changes compared with the flat cat walk, and why (details in TERRAIN.md):

1. Terrain: one heightfield with a grid of tiles (quadrucat/terrain.py). Each env keeps one terrain
   type (row) and moves between difficulty levels (columns).
2. Curriculum: at the end of every episode the env moves up one level if the robot walked at
   least `promote_frac` of the commanded distance without falling, down one level if it fell or
   walked less than `demote_frac`. Envs that pass the last level restart at a random level, so the
   easy terrain is not forgotten. Without it most envs would start on terrain they cannot learn from.
3. Height scan (actor observation, optional): ground height at a grid of points around the trunk,
   in the yaw-aligned body frame, with noise. Without it the policy is "blind" and can only react
   after a foot hits an edge; with it it can place feet on a beam or lift them before a step.
   The critic always gets the clean scan plus foot heights, floor friction and the level
   (asymmetric actor-critic: privileged information only where it is free).
4. Terrain-relative feet rewards: Playground measures foot height in world z, so on a step the
   swing-height terms would push feet towards z = 0.1 m instead of 0.1 m above the ground.
   Here feet_clearance and feet_height use the height above the ground under the foot.
5. leg_collision: the feet-only Go1 model lets shins and trunk pass through the ground; this term
   penalizes how deep knees, shins and the trunk bottom go below the terrain, so the policy cannot
   "cheat" through a step edge that a real robot would hit.
6. Termination also when the trunk gets closer than `termination_height` to the highest ground
   just below it (fell off a beam, collapsed on a step) and when the robot leaves the map.
7. Heading commands (optional): the yaw-rate command is a P-controller towards a target heading, so
   robots walk straight lines (on beams: along the beam) and "distance walked" means something
   for the curriculum.

Training uses Playground's default auto-reset (back to the first state without calling reset),
which keeps `info` between episodes. A new episode is recognised by data.time == 0, and the robot is
then moved to a tile of its (updated) level. The policy acts once on the cached first observation.
"""
from __future__ import annotations

import copy
import pathlib
import tempfile

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict

from quadrucat import terrain

LEGS = ("FR", "FL", "RR", "RL")
HOME_Z = 0.278   # trunk height of the Go1 'home' keyframe on flat ground [m]


def terrain_default_config() -> config_dict.ConfigDict:
    from quadrucat.cat_envs import catwalk_default_config

    cfg = catwalk_default_config()
    cfg.naconmax = 8 * 8192      # as Playground's rough terrain: a foot on a heightfield can make several contacts
    cfg.njmax = 12 + 48
    cfg.terrain = config_dict.create(
        **copy.deepcopy(terrain.DEFAULT_PARAMS),          # map: types, levels, sizes [m] (see quadrucat/terrain.py)
        curriculum=True,
        init_level_max=2,
        min_cmd_dist=1.0,
        promote_frac=0.5,
        demote_frac=0.25,
        spawn_jitter=1.0,
        eval_type="",                        # "" = random by type_weights; a type name pins it
        eval_level=-1,                       # -1 = random in [0, init_level_max]
        eval_start="center",                 # "center" | "edge" (start at -x edge, face +x: crossing test)
    )
    cfg.height_scan = config_dict.create(
        in_actor=True,
        x=[-0.5, 0.7, 13],                   # [min, max, count] in the yaw-aligned body frame [m]
        y=[-0.3, 0.3, 7],
        offset=0.3,                          # value = trunk_z - ground - offset, clipped to [-1, 1]
        noise=0.02,                          # [m] uniform noise per point (times noise_config.level)
    )
    cfg.heading_command = config_dict.create(
        enable=True,
        kp=1.0,
        max_yaw_rate=1.0,
        beam_vx=[0.2, 0.6],                  # forward speed range on beam rows [m/s]
    )
    cfg.termination_height = 0.05
    cfg.beam_fall_height = 0.12     # on a beam tile, trunk below this (beam top = 0) = fell off the beam
    cfg.reward_config.scales.leg_collision = 0.0
    cfg.reward_config.leg_collision_radius = 0.02
    return cfg


def _scene_xml(asset: str, geom: str) -> str:
    return f"""<mujoco model="go1 terrain">
  <include file="go1_mjx_feetonly.xml"/>
  <statistic center="0 0 0.1" extent="0.8" meansize="0.04"/>
  <visual>
    <headlight diffuse=".8 .8 .8" ambient=".3 .3 .3" specular="0 0 0"/>
    <global azimuth="120" elevation="-20"/>
    <quality shadowsize="8192"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.75 0.75 0.72"
      rgb2="0.65 0.65 0.62" markrgb="0.3 0.3 0.3" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="8 8" reflectance="0"/>
    {asset}
  </asset>
  <worldbody>
    {geom}
  </worldbody>
  <include file="sensor_feet.xml"/>
  <keyframe>
    <key name="home" qpos="0 0 {HOME_Z} 1 0 0 0 0.1 0.9 -1.8 -0.1 0.9 -1.8 0.1 0.9 -1.8 -0.1 0.9 -1.8"
      ctrl="0.1 0.9 -1.8 -0.1 0.9 -1.8 0.1 0.9 -1.8 -0.1 0.9 -1.8"/>
  </keyframe>
</mujoco>
"""


def _yaw_quat(yaw):
    return jp.array([jp.cos(yaw / 2), 0.0, 0.0, jp.sin(yaw / 2)])


def _wrap(a):
    return jp.mod(a + jp.pi, 2 * jp.pi) - jp.pi


def curriculum_update(level, fell, dist, cmd_dist, rng, n_levels, min_cmd_dist, promote_frac, demote_frac):
    """New level after an episode (pure function, tested in tests/test_terrain.py)."""
    moved = cmd_dist >= min_cmd_dist
    up = ~fell & moved & (dist >= promote_frac * cmd_dist)
    down = fell | (moved & (dist < demote_frac * cmd_dist))
    new = level + up.astype(jp.int32) - down.astype(jp.int32)
    restart = jax.random.randint(rng, (), 0, n_levels)
    return jp.where(new > n_levels - 1, restart, jp.clip(new, 0, n_levels - 1))


def _terrain_class():
    from mujoco import mjx
    from mujoco_playground._src.locomotion.go1 import base as go1_base

    from quadrucat.cat_envs import _catwalk_class

    CatWalk = _catwalk_class()

    class Terrain(CatWalk):
        """Go1CatWalk on the terrain map. See the module docstring."""

        def __init__(self, config=None, config_overrides=None):
            config = config or terrain_default_config()
            self._tmap = terrain.build_map(config.terrain)
            asset, geom, data = terrain.hfield_xml(self._tmap)
            path = pathlib.Path(tempfile.mkdtemp(prefix="go1_terrain_")) / "scene_terrain.xml"
            path.write_text(_scene_xml(asset, geom), encoding="utf-8")
            go1_base.Go1Env.__init__(self, xml_path=path.as_posix(), config=config,
                                     config_overrides=config_overrides)
            self._mj_model.hfield_data[:] = data.ravel()
            self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
            self._post_init()

        # ---------------- setup ----------------
        def _post_init(self):
            super()._post_init()
            c = self._config
            tm = self._tmap
            m = self._mj_model
            self._H = jp.asarray(tm.heights)
            self._calf_body_id = np.array([m.body(f"{leg}_calf").id for leg in LEGS])
            self._scan_pts = jp.asarray(terrain.scan_grid(c.height_scan.x, c.height_scan.y))
            types = list(c.terrain.types)
            self._n_rows, self._n_levels = len(types), int(c.terrain.n_levels)
            self._row_is_beam = jp.array([t == "beam" for t in types])
            w = np.asarray(c.terrain.type_weights, dtype=np.float32)
            if w.shape != (len(types),):
                raise ValueError("terrain.type_weights needs one weight per terrain.types entry")
            self._row_logits = jp.log(jp.asarray(w / w.sum()) + 1e-12)
            T, plat = tm.tile_size, float(c.terrain.platform_size)
            jx, jy = [], []
            for t in types:   # where in a tile a robot may start
                if t == "beam":
                    jx.append(T / 2 - 0.5); jy.append(0.0)
                elif t in ("stairs_up", "stairs_down", "boxes"):
                    jx.append(plat / 2 - 0.2); jy.append(plat / 2 - 0.2)
                else:
                    jx.append(min(c.terrain.spawn_jitter, T / 2 - 0.5)); jy.append(min(c.terrain.spawn_jitter, T / 2 - 0.5))
            self._jitter = jp.array(np.stack([jx, jy], -1), dtype=jp.float32)
            self._eval_row = types.index(c.terrain.eval_type) if c.terrain.eval_type else -1
            hx, hy = tm.half_extent
            self._bounds = jp.array([hx - 0.5, hy - 0.5])
            self._map_center = jp.array([tm.x0 + hx, tm.y0 + hy])
            # footprint used for the spawn height and the termination check (world-aligned) [m]
            self._foot_window = jp.array([[0, 0], [0.25, 0.15], [0.25, -0.15], [-0.25, 0.15], [-0.25, -0.15],
                                          [0.15, 0], [-0.15, 0], [0, 0.1], [0, -0.1]], dtype=jp.float32)

        def ground(self, xy):
            tm = self._tmap
            return terrain.sample_heights(self._H, tm.x0, tm.y0, tm.resolution, xy)

        def _yaw(self, data):
            r = data.xmat[self._torso_body_id]
            return jp.arctan2(r[1, 0], r[0, 0])

        # ---------------- spawn / curriculum ----------------
        def _spawn(self, rng, row, level):
            c = self._config.terrain
            cx, cy = terrain.tile_center(row, level, self._n_rows, self._n_levels, self._tmap.tile_size)
            center = jp.array([cx, cy], dtype=jp.float32)
            k1, k2, k3 = jax.random.split(rng, 3)
            beam = self._row_is_beam[row]
            if c.eval_start == "edge":
                xy = center + jp.array([-self._tmap.tile_size / 2 + 0.4, 0.0])
                yaw = jp.float32(0.0)
            else:
                xy = center + jax.random.uniform(k1, (2,), minval=-1.0, maxval=1.0) * self._jitter[row]
                flip = jp.where(jax.random.bernoulli(k3), jp.pi, 0.0)
                yaw = jp.where(beam, flip + jax.random.uniform(k2, minval=-0.1, maxval=0.1),
                               jax.random.uniform(k2, minval=-jp.pi, maxval=jp.pi))
            z = HOME_Z + 0.02 + jp.max(self.ground(xy + self._foot_window))
            return xy, yaw, z

        def _place(self, data, xy, yaw, z):
            qpos = data.qpos.at[0:2].set(xy).at[2].set(z).at[3:7].set(_yaw_quat(yaw))
            return data.replace(qpos=qpos)

        def _fix_command(self, cmd, row):
            """Beam rows walk forward along the beam: vx in beam_vx, no sideways command."""
            lo, hi = self._config.heading_command.beam_vx
            beam_cmd = jp.array([jp.clip(jp.abs(cmd[0]), lo, hi), 0.0, cmd[2]])
            return jp.where(self._row_is_beam[row], beam_cmd, cmd)

        def _heading_yaw_rate(self, data, target):
            hc = self._config.heading_command
            return jp.clip(hc.kp * _wrap(target - self._yaw(data)), -hc.max_yaw_rate, hc.max_yaw_rate)

        # ---------------- reset ----------------
        def reset(self, rng):
            c = self._config.terrain
            kt, kl = jax.random.split(jax.random.fold_in(rng, 13))
            row = (jp.int32(self._eval_row) if self._eval_row >= 0
                   else jax.random.categorical(kt, self._row_logits).astype(jp.int32))
            lmax = min(int(c.init_level_max), self._n_levels - 1)
            level = (jp.int32(min(int(c.eval_level), self._n_levels - 1)) if c.eval_level >= 0
                     else jax.random.randint(kl, (), 0, lmax + 1))
            return self.reset_at(rng, row, level)

        def reset_at(self, rng, row, level):
            """Reset on a given tile (row and level may be traced: one compile for all tiles)."""
            state = super().reset(rng)
            row, level = jp.asarray(row, jp.int32), jp.asarray(level, jp.int32)
            ks = jax.random.fold_in(rng, 11)
            xy, yaw, z = self._spawn(ks, row, level)
            data = mjx.forward(self.mjx_model, self._place(state.data, xy, yaw, z))
            info = dict(state.info)
            info.update(
                terrain_row=row, terrain_level=level, spawn_xy=xy, heading_target=yaw,
                ep_dist=jp.float32(0.0), ep_cmd_dist=jp.float32(0.0), ep_fell=jp.bool_(False),
                placed=jp.bool_(True), swing_peak_rel=jp.zeros(4),
            )
            cmd = self._fix_command(info["command"], row)
            if self._config.heading_command.enable:
                cmd = cmd.at[2].set(self._heading_yaw_rate(data, yaw))
            info["command"] = cmd
            metrics = dict(state.metrics)
            metrics["terrain_level"] = jp.float32(0.0)
            metrics["out_of_bounds"] = jp.float32(0.0)
            obs = self._get_obs(data, info)
            return state.replace(data=data, obs=obs, info=info, metrics=metrics)

        # ---------------- step ----------------
        def _new_episode(self, state):
            """Training auto-reset put the cached first state back: curriculum + move to a new tile."""
            c = self._config.terrain
            info = dict(state.info)
            new_ep = (state.data.time <= 0.0) & ~info["placed"]
            info["rng"], k_lvl, k_spawn = jax.random.split(info["rng"], 3)
            level = info["terrain_level"]
            if c.curriculum:
                updated = curriculum_update(level, info["ep_fell"], info["ep_dist"], info["ep_cmd_dist"], k_lvl,
                                            self._n_levels, c.min_cmd_dist, c.promote_frac, c.demote_frac)
                level = jp.where(new_ep, updated, level)
            row = info["terrain_row"]
            xy, yaw, z = self._spawn(k_spawn, row, level)
            moved = self._place(state.data, xy, yaw, z)
            data = state.data.replace(qpos=jp.where(new_ep, moved.qpos, state.data.qpos))
            fresh = {
                "terrain_level": level, "spawn_xy": xy, "heading_target": yaw,
                "ep_dist": jp.float32(0.0), "ep_cmd_dist": jp.float32(0.0), "ep_fell": jp.bool_(False),
                "swing_peak_rel": jp.zeros(4), "swing_peak": jp.zeros(4), "feet_air_time": jp.zeros(4),
                "last_contact": jp.zeros(4, dtype=bool), "last_act": jp.zeros_like(info["last_act"]),
                "last_last_act": jp.zeros_like(info["last_last_act"]),
                "print_valid": jp.zeros(2, dtype=bool), "hind_valid": jp.zeros(2, dtype=bool),
            }
            for k, v in fresh.items():
                info[k] = jp.where(new_ep, v, info[k]) if k != "terrain_level" else v
            return state.replace(data=data, info=info)

        def step(self, state, action):
            first = state.info["placed"] | (state.data.time <= 0.0)   # first step of an episode
            state = self._new_episode(state)
            info = state.info
            row = info["terrain_row"]
            cmd = self._fix_command(info["command"], row)
            heading = self._config.heading_command.enable
            if heading:
                cmd = cmd.at[2].set(self._heading_yaw_rate(state.data, info["heading_target"]))
            info["command"] = cmd

            state = super().step(state, action)   # CatWalk -> Joystick: physics, obs, reward, command resampling
            info = state.info
            data = state.data

            # heading target follows a resampled command (not in the crossing evaluation)
            if heading and self._config.terrain.eval_start != "edge":
                changed = jp.any(info["command"] != cmd)
                info["rng"], k = jax.random.split(info["rng"])
                moving = jp.linalg.norm(info["command"][:2]) > 0.1
                new_target = jp.where(moving, self._yaw(data) + jax.random.uniform(k, minval=-jp.pi, maxval=jp.pi),
                                      self._yaw(data))
                beam = self._row_is_beam[row]
                info["heading_target"] = jp.where(changed & ~beam, _wrap(new_target), info["heading_target"])
            info["command"] = self._fix_command(info["command"], row)

            # episode statistics for the curriculum
            xy = data.xpos[self._torso_body_id, :2]
            oob = self._out_of_bounds(data)
            info["ep_fell"] = (state.done > 0.5) & ~oob
            info["ep_dist"] = jp.linalg.norm(xy - info["spawn_xy"])
            info["ep_cmd_dist"] = info["ep_cmd_dist"] + jp.linalg.norm(cmd[:2]) * self.dt
            info["placed"] = jp.bool_(False)
            # only on the first step, so the per-episode sum that Brax logs is the episode's level
            state.metrics["terrain_level"] = info["terrain_level"].astype(jp.float32) * first
            state.metrics["out_of_bounds"] = oob.astype(jp.float32)
            return state.replace(info=info)

        # ---------------- termination ----------------
        def _out_of_bounds(self, data):
            xy = data.xpos[self._torso_body_id, :2] - self._map_center
            return jp.any(jp.abs(xy) > self._bounds)

        def _get_termination(self, data):
            flipped = super()._get_termination(data)
            pos = data.xpos[self._torso_body_id]
            highest = jp.max(self.ground(pos[:2] + self._foot_window))
            too_low = pos[2] - highest < self._config.termination_height
            # in a beam pit the ground is 0.4 m down, so "too low" alone would let a robot that fell
            # off keep walking on the pit floor
            T = self._tmap.tile_size
            row = jp.clip(jp.round(pos[1] / T + (self._n_rows - 1) / 2), 0, self._n_rows - 1).astype(jp.int32)
            on_tiles = jp.abs(pos[0]) < self._n_levels * T / 2
            off_beam = self._row_is_beam[row] & on_tiles & (pos[2] < self._config.beam_fall_height)
            return flipped | too_low | off_beam | self._out_of_bounds(data)

        # ---------------- observation ----------------
        def height_scan(self, data):
            """Clean scan: trunk_z - ground - offset at the scan points, clipped to [-1, 1]."""
            pos = data.xpos[self._torso_body_id]
            yaw = self._yaw(data)
            cy, sy = jp.cos(yaw), jp.sin(yaw)
            rot = jp.array([[cy, -sy], [sy, cy]])
            pts = pos[:2] + self._scan_pts @ rot.T
            return jp.clip(pos[2] - self.ground(pts) - self._config.height_scan.offset, -1.0, 1.0)

        def feet_height(self, data):
            feet = data.site_xpos[self._feet_site_id]
            return feet[:, 2] - self.ground(feet[:, :2])

        def _get_obs(self, data, info):
            obs = super()._get_obs(data, info)
            if "terrain_row" not in info:   # inside the parent reset; reset() rebuilds the obs
                return obs
            c = self._config
            scan = self.height_scan(data)
            info["rng"], k = jax.random.split(info["rng"])
            noisy = scan + (2 * jax.random.uniform(k, scan.shape) - 1) * c.noise_config.level * c.height_scan.noise
            state = jp.concatenate([obs["state"], noisy]) if c.height_scan.in_actor else obs["state"]
            level = info["terrain_level"].astype(jp.float32) / max(self._n_levels - 1, 1)
            priv = jp.concatenate([obs["privileged_state"], scan, self.feet_height(data),
                                   jp.stack([self.mjx_model.geom_friction[0, 0], level])])
            return {"state": state, "privileged_state": priv}

        # ---------------- reward ----------------
        def _cost_feet_clearance(self, data):
            feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
            vel_norm = jp.sqrt(jp.linalg.norm(feet_vel[..., :2], axis=-1))
            delta = jp.abs(self.feet_height(data) - self._config.reward_config.max_foot_height)
            return jp.sum(delta * vel_norm)

        def _cost_feet_height(self, swing_peak, first_contact, info):
            peak = info.get("swing_peak_rel", swing_peak)
            return super()._cost_feet_height(peak, first_contact, info)

        def _leg_collision(self, data):
            r = self._config.reward_config.leg_collision_radius
            knees = data.xpos[self._calf_body_id]
            feet = data.site_xpos[self._feet_site_id]
            rot = data.xmat[self._torso_body_id]
            corners = jp.array([[0.15, 0.06, -0.06], [0.15, -0.06, -0.06], [-0.15, 0.06, -0.06],
                                [-0.15, -0.06, -0.06], [0.0, 0.0, -0.06]])
            trunk = data.xpos[self._torso_body_id] + corners @ rot.T
            pts = jp.concatenate([knees, 0.5 * (knees + feet), trunk])
            radius = jp.concatenate([jp.full(8, r), jp.zeros(5)])
            return jp.sum(jp.maximum(self.ground(pts[:, :2]) + radius - pts[:, 2], 0.0))

        def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
            if "swing_peak_rel" in info:
                info["swing_peak_rel"] = jp.maximum(info["swing_peak_rel"], self.feet_height(data))
            rewards = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
            rewards["leg_collision"] = self._leg_collision(data)
            if "swing_peak_rel" in info:
                info["swing_peak_rel"] = info["swing_peak_rel"] * ~contact
            return rewards

    return Terrain
