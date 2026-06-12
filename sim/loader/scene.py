"""YAML/JSON scene loading for SAP Warp.

Source note: the SAP modifications in this module are based on Newton's
loader/runtime code and adapted so SAP Warp can stay compatible with
Newton-owned Warp arrays and imported USD/URDF/MJCF assets.
"""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import dataclass, fields
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Sequence
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import warp as wp
import yaml

from sim.collision.flags import SapShapeFlags
from sim.collision.heightfield import SapHeightfieldData
from sim.collision.sdf_texture import SapTextureSDFData
from sim.collision.types import SapGeoType
from sim.resources.collision_model import SapCollisionModel, SapCollisionState
from sim.resources.mesh import SapMesh
from sim.sap_kinematics import sap_eval_fk
from sim.sap_runtime import SapControl, SapModel, SapState
from sim.sap_runtime import SAP_BODY_FLAG_DYNAMIC, SAP_BODY_FLAG_KINEMATIC


SCENE_SCHEMA_VERSION = 1
DEFAULT_REPLICATE_SPACING = (2.0, 2.0, 0.0)

_SAP_JOINT_PRISMATIC = 0
_SAP_JOINT_REVOLUTE = 1
_SAP_JOINT_BALL = 2
_SAP_JOINT_FIXED = 3
_SAP_JOINT_FREE = 4
_SAP_JOINT_DISTANCE = 5
_SAP_JOINT_D6 = 6


class SapSceneLoaderError(ValueError):
    """Base exception raised when a SAP scene file cannot be loaded or validated."""
    pass


class SapUnsupportedSceneFeature(SapSceneLoaderError):
    """Exception that reports scene features recognized by the loader but not yet supported by the SAP
    runtime.
    """
    def __init__(self, features: list[str] | tuple[str, ...]):
        self.features = tuple(features)
        details = ", ".join(self.features)
        super().__init__(f"SAP scene loader does not yet support: {details}")


@dataclass(frozen=True)
class SapLoadedScene:
    """Loaded scene bundle returned by load_sap_scene. It keeps collision data, SAP runtime arrays,
    labels, unsupported-feature notes, and the source path together for the caller.
    """
    collision_model: SapCollisionModel
    collision_state: SapCollisionState
    sap_model: SapModel
    sap_state: SapState
    sap_control: SapControl
    body_labels: tuple[str, ...]
    shape_labels: tuple[str, ...]
    unsupported_features: tuple[str, ...] = ()
    source_path: Path | None = None
    shape_colors: tuple[tuple[float, float, float] | None, ...] = ()


@dataclass
class _SapShapeConfig:
    density: float = 1000.0
    ke: float = 2.5e3
    tau: float | None = 0.01
    mu: float = 1.0
    restitution: float = 0.0
    mu_torsional: float = 0.005
    mu_rolling: float = 0.0001
    margin: float = 0.0
    gap: float | None = None
    is_solid: bool = True
    collision_group: int = 1
    collision_filter_parent: bool = True
    has_shape_collision: bool = True
    has_particle_collision: bool = True
    is_visible: bool = True
    is_site: bool = False
    is_hydroelastic: bool = False
    kh: float = 1.0e10
    sdf_narrow_band_range: tuple[float, float] = (-0.1, 0.1)
    sdf_target_voxel_size: float | None = None
    sdf_max_resolution: int | None = None
    sdf_texture_format: str = "uint16"

    def copy(self) -> "_SapShapeConfig":
        copied = _SapShapeConfig(
            **{field_info.name: getattr(self, field_info.name) for field_info in fields(_SapShapeConfig)}
        )
        for name, value in self.__dict__.items():
            if not hasattr(copied, name):
                setattr(copied, name, value)
        return copied

    def flags(self) -> int:
        shape_flags = int(SapShapeFlags.VISIBLE) if self.is_visible else 0
        shape_flags |= int(SapShapeFlags.COLLIDE_SHAPES) if self.has_shape_collision else 0
        shape_flags |= int(SapShapeFlags.COLLIDE_PARTICLES) if self.has_particle_collision else 0
        shape_flags |= int(SapShapeFlags.SITE) if self.is_site else 0
        shape_flags |= int(SapShapeFlags.HYDROELASTIC) if self.is_hydroelastic else 0
        return shape_flags

    def apply(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            if name in {
                "kd",
                "kf",
                "ka",
                "shape_kd",
                "shape_kf",
                "shape_ka",
                "contact_ka",
                "shape_material_ka",
            }:
                continue
            coerced = _coerce_config_value(value)
            if name == "flags":
                self._set_flags(int(coerced))
            else:
                setattr(self, name, coerced)
        if self.is_site:
            self.has_shape_collision = False
            self.has_particle_collision = False
            self.density = 0.0
            self.collision_group = 0

    def _set_flags(self, value: int) -> None:
        self.is_visible = bool(value & int(SapShapeFlags.VISIBLE))
        self.is_hydroelastic = bool(value & int(SapShapeFlags.HYDROELASTIC))
        if value & int(SapShapeFlags.SITE):
            self.is_site = True
            self.has_shape_collision = False
            self.has_particle_collision = False
            self.density = 0.0
            self.collision_group = 0
        else:
            defaults = _SapShapeConfig()
            self.is_site = False
            self.density = defaults.density
            self.collision_group = defaults.collision_group
            self.has_shape_collision = bool(value & int(SapShapeFlags.COLLIDE_SHAPES))
            self.has_particle_collision = bool(value & int(SapShapeFlags.COLLIDE_PARTICLES))


@dataclass
class _SapJointDofConfig:
    axis: wp.vec3
    target_pos: float = 0.0
    target_vel: float = 0.0
    target_ke: float = 0.0
    target_kd: float = 0.0
    limit_lower: float = -1.0e10
    limit_upper: float = 1.0e10
    limit_ke: float = 1.0e4
    limit_kd: float = 10.0
    armature: float = 0.0
    effort_limit: float = 1.0e6
    velocity_limit: float = 1.0e6
    friction: float = 0.0
    actuator_mode: int = 0

    def apply(self, values: dict[str, Any] | None) -> None:
        if not isinstance(values, dict):
            return
        for name, value in values.items():
            if value is None or not hasattr(self, str(name)):
                continue
            if str(name) == "axis":
                self.axis = _vec3_from_any(value)
            elif str(name) == "actuator_mode":
                self.actuator_mode = int(value)
            else:
                setattr(self, str(name), _coerce_config_value(value))


class _SapSceneBuilder:
    def __init__(self, *, rigid_gap: float = 0.1) -> None:
        self.rigid_gap = float(rigid_gap)
        self.default_shape_cfg = _SapShapeConfig()
        self.default_joint_cfg = _SapJointDofConfig(axis=wp.vec3(1.0, 0.0, 0.0))
        self.default_body_armature = 0.0
        self.current_world = -1
        self.world_count = 1
        self.gravity = -9.81
        self.up_vector = (0.0, 0.0, 1.0)

        self.body_q: list[wp.transform] = []
        self.body_qd: list[wp.spatial_vector] = []
        self.body_mass: list[float] = []
        self.body_inertia: list[wp.mat33] = []
        self.body_inv_mass: list[float] = []
        self.body_inv_inertia: list[wp.mat33] = []
        self.body_com: list[wp.vec3] = []
        self.body_flags: list[int] = []
        self.body_lock_inertia: list[bool] = []
        self.body_labels: list[str] = []
        self.body_shapes: dict[int, list[int]] = {-1: []}
        self.body_world: list[int] = []

        self.joint_type: list[int] = []
        self.joint_parent: list[int] = []
        self.joint_parents: dict[int, list[int]] = {}
        self.joint_children: dict[int, list[int]] = {}
        self.joint_child: list[int] = []
        self.joint_axis: list[wp.vec3] = []
        self.joint_X_p: list[wp.transform] = []
        self.joint_X_c: list[wp.transform] = []
        self.joint_q: list[float] = []
        self.joint_qd: list[float] = []
        self.joint_q_start: list[int] = []
        self.joint_qd_start: list[int] = []
        self.joint_dof_dim: list[tuple[int, int]] = []
        self.joint_labels: list[str] = []
        self.joint_articulation: list[int] = []
        self.joint_target_pos: list[float] = []
        self.joint_target_vel: list[float] = []
        self.joint_target_ke: list[float] = []
        self.joint_target_kd: list[float] = []
        self.joint_target_mode: list[int] = []
        self.joint_limit_lower: list[float] = []
        self.joint_limit_upper: list[float] = []
        self.joint_limit_ke: list[float] = []
        self.joint_limit_kd: list[float] = []
        self.joint_effort_limit: list[float] = []
        self.joint_armature: list[float] = []
        self.joint_f: list[float] = []
        self.joint_act: list[float] = []
        self.articulation_start: list[int] = []
        self.articulation_labels: list[str] = []

        self.shape_labels: list[str] = []
        self.shape_colors: list[tuple[float, float, float] | None] = []
        self.shape_transform: list[wp.transform] = []
        self.shape_body: list[int] = []
        self.shape_flags: list[int] = []
        self.shape_type: list[int] = []
        self.shape_scale: list[tuple[float, float, float]] = []
        self.shape_is_solid: list[bool] = []
        self.shape_source_refs: list[Any] = []
        self.shape_margin: list[float] = []
        self.shape_gap: list[float] = []
        self.shape_collision_group: list[int] = []
        self.shape_collision_radius: list[float] = []
        self.shape_world: list[int] = []
        self.shape_collision_filter_pairs: list[tuple[int, int]] = []
        self.shape_material_ke: list[float] = []
        self.shape_material_tau: list[float] = []
        self.shape_material_mu: list[float] = []
        self.shape_material_restitution: list[float] = []
        self.shape_material_mu_torsional: list[float] = []
        self.shape_material_mu_rolling: list[float] = []
        self.shape_material_kh: list[float] = []

    @property
    def shape_count(self) -> int:
        return len(self.shape_type)

    def add_body(
        self,
        *,
        xform: Any,
        label: str | None = None,
        mass: float = 0.0,
        com: Any = None,
        inertia: Any = None,
        armature: float | None = None,
        lock_inertia: bool = False,
        is_kinematic: bool = False,
    ) -> int:
        body_id = len(self.body_q)
        body_mass = float(mass)
        body_com = _vec3(com) if com is not None else wp.vec3(0.0, 0.0, 0.0)
        if inertia is None:
            body_inertia = wp.mat33(0.0)
        else:
            body_inertia = wp.mat33(np.asarray(inertia, dtype=np.float32).reshape(3, 3))
        body_armature = self.default_body_armature if armature is None else float(armature)
        if body_armature != 0.0:
            body_inertia = body_inertia + wp.mat33(np.eye(3, dtype=np.float32)) * float(body_armature)

        self.body_q.append(_as_transform(xform))
        self.body_qd.append(wp.spatial_vector())
        self.body_mass.append(body_mass)
        self.body_inertia.append(body_inertia)
        self.body_inv_mass.append(1.0 / body_mass if body_mass > 0.0 else 0.0)
        self.body_inv_inertia.append(wp.inverse(body_inertia) if any(x for x in body_inertia) else body_inertia)
        self.body_com.append(body_com)
        self.body_flags.append(int(SAP_BODY_FLAG_KINEMATIC) if bool(is_kinematic) else int(SAP_BODY_FLAG_DYNAMIC))
        self.body_lock_inertia.append(bool(lock_inertia))
        self.body_labels.append(label or f"body_{body_id}")
        self.body_shapes[body_id] = []
        self.body_world.append(self.current_world)
        return body_id

    def add_shape(
        self,
        *,
        body: int,
        shape_type: int,
        xform: Any,
        cfg: _SapShapeConfig,
        scale: tuple[float, float, float],
        src: Any = None,
        label: str | None = None,
        color: Any = None,
    ) -> int:
        shape = self.shape_count
        self.body_shapes.setdefault(body, [])
        if cfg.has_shape_collision:
            for same_body_shape in self.body_shapes[body]:
                self.add_shape_collision_filter_pair(same_body_shape, shape)

        self.shape_body.append(body)
        self.body_shapes[body].append(shape)
        self.shape_labels.append(label or f"shape_{shape}")
        self.shape_colors.append(_normalize_color(color))
        self.shape_transform.append(_as_transform(xform))

        flags = cfg.flags()
        if (flags & int(SapShapeFlags.HYDROELASTIC)) and shape_type in (int(SapGeoType.PLANE), int(SapGeoType.HFIELD)):
            flags &= ~int(SapShapeFlags.HYDROELASTIC)
        self.shape_flags.append(flags)
        self.shape_type.append(shape_type)
        self.shape_scale.append(scale)
        self.shape_is_solid.append(bool(cfg.is_solid))
        self.shape_source_refs.append(src)
        self.shape_margin.append(float(cfg.margin))
        self.shape_material_ke.append(float(cfg.ke))
        self.shape_material_tau.append(float(cfg.tau) if cfg.tau is not None else math.nan)
        self.shape_material_mu.append(float(cfg.mu))
        self.shape_material_restitution.append(float(cfg.restitution))
        self.shape_material_mu_torsional.append(float(cfg.mu_torsional))
        self.shape_material_mu_rolling.append(float(cfg.mu_rolling))
        self.shape_material_kh.append(float(cfg.kh))
        self.shape_gap.append(float(cfg.gap if cfg.gap is not None else self.rigid_gap))
        self.shape_collision_group.append(int(cfg.collision_group))
        self.shape_collision_radius.append(_compute_shape_radius(shape_type, scale, src))
        self.shape_world.append(self.current_world)

        if cfg.has_shape_collision and cfg.collision_filter_parent and body > -1 and body in self.joint_parents:
            for parent_body in self.joint_parents[body]:
                if parent_body > -1:
                    for parent_shape in self.body_shapes[parent_body]:
                        self.add_shape_collision_filter_pair(parent_shape, shape)

        if cfg.has_shape_collision and cfg.collision_filter_parent and body > -1 and body in self.joint_children:
            for child_body in self.joint_children[body]:
                for child_shape in self.body_shapes[child_body]:
                    self.add_shape_collision_filter_pair(shape, child_shape)

        if cfg.density > 0.0 and body >= 0 and not self.body_lock_inertia[body]:
            m, c, inertia = _sap_compute_inertia_shape(
                shape_type,
                scale,
                src,
                cfg.density,
                cfg.is_solid,
                cfg.margin,
            )
            com_body = wp.transform_point(self.shape_transform[shape], c)
            self._update_body_mass(body, m, inertia, com_body, self.shape_transform[shape].q)

        return shape

    def _update_body_mass(self, body: int, mass: float, inertia: wp.mat33, p: wp.vec3, q: wp.quat) -> None:
        if body == -1:
            return
        new_mass = self.body_mass[body] + float(mass)
        if new_mass == 0.0:
            return

        old_mass = self.body_mass[body]
        new_com = (self.body_com[body] * old_mass + p * float(mass)) / new_mass
        com_offset = new_com - self.body_com[body]
        shape_offset = new_com - p
        new_inertia = _sap_transform_inertia(
            old_mass,
            self.body_inertia[body],
            com_offset,
            wp.quat_identity(),
        ) + _sap_transform_inertia(float(mass), inertia, shape_offset, q)

        self.body_mass[body] = new_mass
        self.body_inertia[body] = new_inertia
        self.body_com[body] = new_com
        self.body_inv_mass[body] = 1.0 / new_mass if new_mass > 0.0 else 0.0
        self.body_inv_inertia[body] = wp.inverse(new_inertia) if any(x for x in new_inertia) else new_inertia

    def add_shape_collision_filter_pair(self, shape_a: int, shape_b: int) -> None:
        self.shape_collision_filter_pairs.append((min(int(shape_a), int(shape_b)), max(int(shape_a), int(shape_b))))

    def add_joint(
        self,
        *,
        parent: int,
        child: int,
        joint_type: int = _SAP_JOINT_FIXED,
        linear_axes: Sequence[_SapJointDofConfig | Any] | None = None,
        angular_axes: Sequence[_SapJointDofConfig | Any] | None = None,
        parent_xform: Any = None,
        child_xform: Any = None,
        label: str | None = None,
        collision_filter_parent: bool = True,
    ) -> int:
        if linear_axes is None:
            linear_axes = ()
        if angular_axes is None:
            angular_axes = ()
        linear_axis_configs = [_joint_dof_config(axis) for axis in linear_axes]
        angular_axis_configs = [_joint_dof_config(axis) for axis in angular_axes]

        joint_id = len(self.joint_type)
        self.joint_type.append(int(joint_type))
        self.joint_parent.append(int(parent))
        self.joint_child.append(int(child))
        self.joint_X_p.append(_as_transform(parent_xform))
        self.joint_X_c.append(_as_transform(child_xform))
        self.joint_q_start.append(len(self.joint_q))
        self.joint_qd_start.append(len(self.joint_qd))
        self.joint_dof_dim.append((len(linear_axis_configs), len(angular_axis_configs)))
        self.joint_labels.append(label or f"joint_{joint_id}")
        self.joint_articulation.append(-1)

        for axis_cfg in itertools.chain(linear_axis_configs, angular_axis_configs):
            self.joint_axis.append(axis_cfg.axis)
            self.joint_target_pos.append(float(axis_cfg.target_pos))
            self.joint_target_vel.append(float(axis_cfg.target_vel))
            self.joint_target_ke.append(float(axis_cfg.target_ke))
            self.joint_target_kd.append(float(axis_cfg.target_kd))
            self.joint_target_mode.append(int(axis_cfg.actuator_mode))
            self.joint_limit_lower.append(float(axis_cfg.limit_lower))
            self.joint_limit_upper.append(float(axis_cfg.limit_upper))
            self.joint_limit_ke.append(float(axis_cfg.limit_ke))
            self.joint_limit_kd.append(float(axis_cfg.limit_kd))
            self.joint_effort_limit.append(float(axis_cfg.effort_limit))
            self.joint_armature.append(float(axis_cfg.armature))
            self.joint_f.append(0.0)
            self.joint_act.append(0.0)

        dof_count, coord_count = _joint_dof_count(int(joint_type), len(linear_axis_configs) + len(angular_axis_configs))
        self.joint_q.extend([0.0] * coord_count)
        self.joint_qd.extend([0.0] * dof_count)
        if joint_type in {_SAP_JOINT_FREE, _SAP_JOINT_DISTANCE, _SAP_JOINT_BALL} and coord_count > 0:
            self.joint_q[-1] = 1.0
        if joint_type == _SAP_JOINT_FREE:
            self.joint_q[self.joint_q_start[joint_id] : self.joint_q_start[joint_id] + 7] = list(self.body_q[child])

        self.joint_parents.setdefault(child, []).append(parent)
        children = self.joint_children.setdefault(parent, [])
        if child not in children:
            children.append(child)

        if collision_filter_parent and parent > -1:
            for child_shape in self.body_shapes.get(child, []):
                if not self.shape_flags[child_shape] & int(SapShapeFlags.COLLIDE_SHAPES):
                    continue
                for parent_shape in self.body_shapes.get(parent, []):
                    if not self.shape_flags[parent_shape] & int(SapShapeFlags.COLLIDE_SHAPES):
                        continue
                    self.add_shape_collision_filter_pair(parent_shape, child_shape)
        return joint_id

    def add_articulation(self, joints: Sequence[int], label: str | None = None) -> None:
        if not joints:
            return
        sorted_joints = [int(joint) for joint in joints]
        art_id = len(self.articulation_start)
        self.articulation_start.append(sorted_joints[0])
        self.articulation_labels.append(label or f"articulation_{art_id}")
        for joint in sorted_joints:
            self.joint_articulation[joint] = art_id

    def add_ground_plane(
        self,
        *,
        height: float = 0.0,
        cfg: _SapShapeConfig | None = None,
        label: str | None = None,
    ) -> int:
        return self.add_shape(
            body=-1,
            shape_type=int(SapGeoType.PLANE),
            xform=wp.transform(wp.vec3(0.0, 0.0, float(height)), wp.quat_identity()),
            cfg=cfg if cfg is not None else self.default_shape_cfg.copy(),
            scale=(0.0, 0.0, 0.0),
            label=label or "ground_plane",
        )

    def replicate(self, world_count: int, *, spacing: tuple[float, float, float] = DEFAULT_REPLICATE_SPACING) -> None:
        world_count = int(world_count)
        if world_count <= 1:
            self.world_count = 1
            return

        base = {
            "body_q": list(self.body_q),
            "body_qd": list(self.body_qd),
            "body_mass": list(self.body_mass),
            "body_inertia": list(self.body_inertia),
            "body_inv_mass": list(self.body_inv_mass),
            "body_inv_inertia": list(self.body_inv_inertia),
            "body_com": list(self.body_com),
            "body_flags": list(self.body_flags),
            "body_lock_inertia": list(self.body_lock_inertia),
            "body_labels": list(self.body_labels),
            "joint_type": list(self.joint_type),
            "joint_parent": list(self.joint_parent),
            "joint_child": list(self.joint_child),
            "joint_axis": list(self.joint_axis),
            "joint_X_p": list(self.joint_X_p),
            "joint_X_c": list(self.joint_X_c),
            "joint_q": list(self.joint_q),
            "joint_qd": list(self.joint_qd),
            "joint_q_start": list(self.joint_q_start),
            "joint_qd_start": list(self.joint_qd_start),
            "joint_dof_dim": list(self.joint_dof_dim),
            "joint_labels": list(self.joint_labels),
            "joint_articulation": list(self.joint_articulation),
            "joint_target_pos": list(self.joint_target_pos),
            "joint_target_vel": list(self.joint_target_vel),
            "joint_target_ke": list(self.joint_target_ke),
            "joint_target_kd": list(self.joint_target_kd),
            "joint_target_mode": list(self.joint_target_mode),
            "joint_limit_lower": list(self.joint_limit_lower),
            "joint_limit_upper": list(self.joint_limit_upper),
            "joint_limit_ke": list(self.joint_limit_ke),
            "joint_limit_kd": list(self.joint_limit_kd),
            "joint_effort_limit": list(self.joint_effort_limit),
            "joint_armature": list(self.joint_armature),
            "joint_f": list(self.joint_f),
            "joint_act": list(self.joint_act),
            "articulation_start": list(self.articulation_start),
            "articulation_labels": list(self.articulation_labels),
            "shape_labels": list(self.shape_labels),
            "shape_colors": list(self.shape_colors),
            "shape_transform": list(self.shape_transform),
            "shape_body": list(self.shape_body),
            "shape_flags": list(self.shape_flags),
            "shape_type": list(self.shape_type),
            "shape_scale": list(self.shape_scale),
            "shape_is_solid": list(self.shape_is_solid),
            "shape_source_refs": list(self.shape_source_refs),
            "shape_margin": list(self.shape_margin),
            "shape_gap": list(self.shape_gap),
            "shape_collision_group": list(self.shape_collision_group),
            "shape_collision_radius": list(self.shape_collision_radius),
            "shape_collision_filter_pairs": list(self.shape_collision_filter_pairs),
            "shape_material_ke": list(self.shape_material_ke),
            "shape_material_tau": list(self.shape_material_tau),
            "shape_material_mu": list(self.shape_material_mu),
            "shape_material_restitution": list(self.shape_material_restitution),
            "shape_material_mu_torsional": list(self.shape_material_mu_torsional),
            "shape_material_mu_rolling": list(self.shape_material_mu_rolling),
            "shape_material_kh": list(self.shape_material_kh),
        }
        base_body_count = len(base["body_q"])
        base_joint_count = len(base["joint_type"])
        base_shape_count = len(base["shape_type"])
        base_coord_count = len(base["joint_q"])
        base_dof_count = len(base["joint_qd"])
        base_articulation_count = len(base["articulation_start"])

        self.body_q = []
        self.body_qd = []
        self.body_mass = []
        self.body_inertia = []
        self.body_inv_mass = []
        self.body_inv_inertia = []
        self.body_com = []
        self.body_flags = []
        self.body_lock_inertia = []
        self.body_labels = []
        self.body_shapes = {-1: []}
        self.body_world = []

        self.joint_type = []
        self.joint_parent = []
        self.joint_parents = {}
        self.joint_children = {}
        self.joint_child = []
        self.joint_axis = []
        self.joint_X_p = []
        self.joint_X_c = []
        self.joint_q = []
        self.joint_qd = []
        self.joint_q_start = []
        self.joint_qd_start = []
        self.joint_dof_dim = []
        self.joint_labels = []
        self.joint_articulation = []
        self.joint_target_pos = []
        self.joint_target_vel = []
        self.joint_target_ke = []
        self.joint_target_kd = []
        self.joint_target_mode = []
        self.joint_limit_lower = []
        self.joint_limit_upper = []
        self.joint_limit_ke = []
        self.joint_limit_kd = []
        self.joint_effort_limit = []
        self.joint_armature = []
        self.joint_f = []
        self.joint_act = []
        self.articulation_start = []
        self.articulation_labels = []

        self.shape_labels = []
        self.shape_colors = []
        self.shape_transform = []
        self.shape_body = []
        self.shape_flags = []
        self.shape_type = []
        self.shape_scale = []
        self.shape_is_solid = []
        self.shape_source_refs = []
        self.shape_margin = []
        self.shape_gap = []
        self.shape_collision_group = []
        self.shape_collision_radius = []
        self.shape_world = []
        self.shape_collision_filter_pairs = []
        self.shape_material_ke = []
        self.shape_material_tau = []
        self.shape_material_mu = []
        self.shape_material_restitution = []
        self.shape_material_mu_torsional = []
        self.shape_material_mu_rolling = []
        self.shape_material_kh = []

        world_offsets = _compute_sap_world_offsets(world_count, spacing)
        for world_index, offset in enumerate(world_offsets):
            body_offset = world_index * base_body_count
            joint_offset = world_index * base_joint_count
            shape_offset = world_index * base_shape_count
            coord_offset = world_index * base_coord_count
            dof_offset = world_index * base_dof_count
            articulation_offset = world_index * base_articulation_count

            for body_q in base["body_q"]:
                self.body_q.append(_offset_transform(body_q, offset))
            self.body_qd.extend(base["body_qd"])
            self.body_mass.extend(base["body_mass"])
            self.body_inertia.extend(base["body_inertia"])
            self.body_inv_mass.extend(base["body_inv_mass"])
            self.body_inv_inertia.extend(base["body_inv_inertia"])
            self.body_com.extend(base["body_com"])
            self.body_flags.extend(base["body_flags"])
            self.body_lock_inertia.extend(base["body_lock_inertia"])
            self.body_labels.extend(base["body_labels"])
            self.body_world.extend([world_index] * base_body_count)

            self.joint_type.extend(base["joint_type"])
            self.joint_parent.extend(parent + body_offset if parent >= 0 else -1 for parent in base["joint_parent"])
            self.joint_child.extend(child + body_offset for child in base["joint_child"])
            self.joint_axis.extend(base["joint_axis"])
            self.joint_X_c.extend(base["joint_X_c"])
            self.joint_q_start.extend(start + coord_offset for start in base["joint_q_start"])
            self.joint_qd_start.extend(start + dof_offset for start in base["joint_qd_start"])
            self.joint_dof_dim.extend(base["joint_dof_dim"])
            self.joint_labels.extend(base["joint_labels"])
            self.joint_articulation.extend(
                articulation + articulation_offset if articulation >= 0 else -1
                for articulation in base["joint_articulation"]
            )
            self.joint_target_pos.extend(base["joint_target_pos"])
            self.joint_target_vel.extend(base["joint_target_vel"])
            self.joint_target_ke.extend(base["joint_target_ke"])
            self.joint_target_kd.extend(base["joint_target_kd"])
            self.joint_target_mode.extend(base["joint_target_mode"])
            self.joint_limit_lower.extend(base["joint_limit_lower"])
            self.joint_limit_upper.extend(base["joint_limit_upper"])
            self.joint_limit_ke.extend(base["joint_limit_ke"])
            self.joint_limit_kd.extend(base["joint_limit_kd"])
            self.joint_effort_limit.extend(base["joint_effort_limit"])
            self.joint_armature.extend(base["joint_armature"])
            self.joint_f.extend(base["joint_f"])
            self.joint_act.extend(base["joint_act"])
            self.articulation_start.extend(start + joint_offset for start in base["articulation_start"])
            self.articulation_labels.extend(base["articulation_labels"])

            world_joint_q = list(base["joint_q"])
            for joint_index, joint_type in enumerate(base["joint_type"]):
                parent = base["joint_parent"][joint_index]
                q_start = base["joint_q_start"][joint_index]
                if parent == -1 and joint_type in {_SAP_JOINT_FREE, _SAP_JOINT_DISTANCE}:
                    world_joint_q[q_start + 0] += float(offset[0])
                    world_joint_q[q_start + 1] += float(offset[1])
                    world_joint_q[q_start + 2] += float(offset[2])
            self.joint_q.extend(world_joint_q)
            self.joint_qd.extend(base["joint_qd"])

            for joint_index, xform in enumerate(base["joint_X_p"]):
                parent = base["joint_parent"][joint_index]
                joint_type = base["joint_type"][joint_index]
                if parent == -1 and joint_type not in {_SAP_JOINT_FREE, _SAP_JOINT_DISTANCE}:
                    self.joint_X_p.append(_offset_transform(xform, offset))
                else:
                    self.joint_X_p.append(xform)

            self.shape_labels.extend(base["shape_labels"])
            self.shape_colors.extend(base["shape_colors"])
            self.shape_flags.extend(base["shape_flags"])
            self.shape_type.extend(base["shape_type"])
            self.shape_scale.extend(base["shape_scale"])
            self.shape_is_solid.extend(base["shape_is_solid"])
            self.shape_source_refs.extend(base["shape_source_refs"])
            self.shape_margin.extend(base["shape_margin"])
            self.shape_gap.extend(base["shape_gap"])
            self.shape_collision_group.extend(base["shape_collision_group"])
            self.shape_collision_radius.extend(base["shape_collision_radius"])
            self.shape_material_ke.extend(base["shape_material_ke"])
            self.shape_material_tau.extend(base["shape_material_tau"])
            self.shape_material_mu.extend(base["shape_material_mu"])
            self.shape_material_restitution.extend(base["shape_material_restitution"])
            self.shape_material_mu_torsional.extend(base["shape_material_mu_torsional"])
            self.shape_material_mu_rolling.extend(base["shape_material_mu_rolling"])
            self.shape_material_kh.extend(base["shape_material_kh"])

            for shape_index, body in enumerate(base["shape_body"]):
                replicated_body = body + body_offset if body >= 0 else -1
                replicated_shape = shape_index + shape_offset
                self.shape_body.append(replicated_body)
                self.shape_world.append(world_index)
                if replicated_body >= 0:
                    self.shape_transform.append(base["shape_transform"][shape_index])
                else:
                    self.shape_transform.append(_offset_transform(base["shape_transform"][shape_index], offset))
                self.body_shapes.setdefault(replicated_body, []).append(replicated_shape)

            for shape_a, shape_b in base["shape_collision_filter_pairs"]:
                self.add_shape_collision_filter_pair(shape_a + shape_offset, shape_b + shape_offset)

            for joint_index, (parent, child) in enumerate(
                zip(
                    self.joint_parent[joint_offset:],
                    self.joint_child[joint_offset:],
                    strict=True,
                )
            ):
                self.joint_parents.setdefault(child, []).append(parent)
                children = self.joint_children.setdefault(parent, [])
                if child not in children:
                    children.append(child)

        self.world_count = world_count

    def finalize(
        self,
        *,
        device: Any = None,
        requires_grad: bool = False,
        rigid_contact_max: int = 0,
        sap_model: SapModel | None = None,
    ) -> SapCollisionModel:
        shape_count = self.shape_count
        if shape_count == 0:
            raise SapSceneLoaderError("SAP scene loader requires at least one collision shape")

        with wp.ScopedDevice(device):
            shape_source_ptr = self._finalize_shape_sources(device=device, requires_grad=requires_grad)
            aabb_lower, aabb_upper, voxel_resolution = self._build_local_aabbs()
            shape_contact_pairs = self._find_shape_contact_pairs(device=device)
            shape_transform = (
                sap_model.shape_transform
                if sap_model is not None and sap_model.shape_transform is not None
                else wp.array(self.shape_transform, dtype=wp.transform, device=device, requires_grad=requires_grad)
            )
            shape_body = (
                sap_model.shape_body
                if sap_model is not None
                else wp.array(self.shape_body, dtype=wp.int32, device=device)
            )
            shape_type = (
                sap_model.shape_type
                if sap_model is not None and sap_model.shape_type is not None
                else wp.array(self.shape_type, dtype=wp.int32, device=device)
            )
            shape_scale = (
                sap_model.shape_scale
                if sap_model is not None and sap_model.shape_scale is not None
                else wp.array(self.shape_scale, dtype=wp.vec3, device=device, requires_grad=requires_grad)
            )
            shape_collision_radius = (
                sap_model.shape_collision_radius
                if sap_model is not None and sap_model.shape_collision_radius is not None
                else wp.array(self.shape_collision_radius, dtype=wp.float32, device=device, requires_grad=requires_grad)
            )
            shape_margin = (
                sap_model.shape_margin
                if sap_model is not None and sap_model.shape_margin is not None
                else wp.array(self.shape_margin, dtype=wp.float32, device=device, requires_grad=requires_grad)
            )
            shape_gap = (
                sap_model.shape_gap
                if sap_model is not None and sap_model.shape_gap is not None
                else wp.array(self.shape_gap, dtype=wp.float32, device=device, requires_grad=requires_grad)
            )
            shape_flags = (
                sap_model.shape_flags
                if sap_model is not None and sap_model.shape_flags is not None
                else wp.array(self.shape_flags, dtype=wp.int32, device=device)
            )
            shape_world = (
                sap_model.shape_world
                if sap_model is not None and sap_model.shape_world is not None
                else wp.array(self.shape_world, dtype=wp.int32, device=device)
            )
            shape_sdf_index = (
                sap_model.shape_sdf_index
                if sap_model is not None and sap_model.shape_sdf_index is not None
                else wp.full(shape_count, -1, dtype=wp.int32, device=device)
            )
            texture_sdf_data = (
                sap_model.texture_sdf_data
                if sap_model is not None and sap_model.texture_sdf_data is not None
                else wp.zeros(0, dtype=SapTextureSDFData, device=device)
            )
            shape_material_ke = (
                sap_model.shape_material_ke
                if sap_model is not None and sap_model.shape_material_ke is not None
                else wp.array(self.shape_material_ke, dtype=wp.float32, device=device, requires_grad=requires_grad)
            )
            shape_material_tau = (
                sap_model.shape_material_tau
                if sap_model is not None and sap_model.shape_material_tau is not None
                else wp.array(self.shape_material_tau, dtype=wp.float32, device=device, requires_grad=requires_grad)
            )
            shape_material_mu = (
                sap_model.shape_material_mu
                if sap_model is not None and sap_model.shape_material_mu is not None
                else wp.array(self.shape_material_mu, dtype=wp.float32, device=device, requires_grad=requires_grad)
            )

            return SapCollisionModel(
                device=wp.get_device(device),
                shape_count=shape_count,
                world_count=max(1, int(self.world_count)),
                requires_grad=requires_grad,
                rigid_contact_max=int(rigid_contact_max),
                shape_transform=shape_transform,
                shape_body=shape_body,
                shape_type=shape_type,
                shape_scale=shape_scale,
                shape_collision_radius=shape_collision_radius,
                shape_source_ptr=shape_source_ptr,
                shape_source_refs=list(self.shape_source_refs),
                shape_margin=shape_margin,
                shape_gap=shape_gap,
                shape_flags=shape_flags,
                shape_world=shape_world,
                shape_collision_group=wp.array(self.shape_collision_group, dtype=wp.int32, device=device),
                shape_sdf_index=shape_sdf_index,
                texture_sdf_data=texture_sdf_data,
                shape_collision_aabb_lower=wp.array(aabb_lower, dtype=wp.vec3, device=device),
                shape_collision_aabb_upper=wp.array(aabb_upper, dtype=wp.vec3, device=device),
                _shape_voxel_resolution=wp.array(voxel_resolution, dtype=wp.vec3i, device=device),
                shape_heightfield_index=wp.full(shape_count, -1, dtype=wp.int32, device=device),
                heightfield_data=wp.zeros(0, dtype=SapHeightfieldData, device=device),
                heightfield_elevations=wp.zeros(0, dtype=wp.float32, device=device),
                shape_contact_pairs=shape_contact_pairs,
                shape_contact_pair_count=int(shape_contact_pairs.shape[0]),
                shape_collision_filter_pairs=set(self.shape_collision_filter_pairs),
                shape_material_ke=shape_material_ke,
                shape_material_tau=shape_material_tau,
                shape_material_mu=shape_material_mu,
                shape_material_restitution=wp.array(
                    self.shape_material_restitution,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                shape_material_mu_torsional=wp.array(
                    self.shape_material_mu_torsional,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                shape_material_mu_rolling=wp.array(
                    self.shape_material_mu_rolling,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                shape_material_kh=wp.array(
                    self.shape_material_kh,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
            )

    def build_state(self, *, device: Any = None, requires_grad: bool = False) -> SapCollisionState:
        body_q = [wp.transform(xform.p, xform.q) for xform in self.body_q]
        with wp.ScopedDevice(device):
            return SapCollisionState(
                body_q=wp.array(body_q, dtype=wp.transform, device=device, requires_grad=requires_grad),
                requires_grad=requires_grad,
            )

    def build_sap_model(
        self,
        *,
        device: Any = None,
        requires_grad: bool = False,
        eval_initial_fk: bool = False,
    ) -> SapModel:
        body_q = [wp.transform(xform.p, xform.q) for xform in self.body_q]
        body_qd = list(self.body_qd)
        joint_q_start = list(self.joint_q_start)
        joint_q_start.append(len(self.joint_q))
        joint_qd_start = list(self.joint_qd_start)
        joint_qd_start.append(len(self.joint_qd))
        articulation_start = list(self.articulation_start)
        articulation_start.append(len(self.joint_type))

        child_to_joint = {child: joint for joint, child in enumerate(self.joint_child)}
        joint_ancestor = [child_to_joint.get(parent, -1) for parent in self.joint_parent]
        joint_dof_dim_np = np.asarray(self.joint_dof_dim, dtype=np.int32).reshape((-1, 2))
        gravity_vec = tuple(float(g) * float(self.gravity) for g in self.up_vector)
        gravity = [wp.vec3(*gravity_vec)] * max(1, int(self.world_count))

        with wp.ScopedDevice(device):
            body_mass = wp.array(self.body_mass, dtype=wp.float32, device=device, requires_grad=requires_grad)
            body_inertia = wp.array(self.body_inertia, dtype=wp.mat33, device=device, requires_grad=requires_grad)
            body_inv_mass = wp.array(self.body_inv_mass, dtype=wp.float32, device=device, requires_grad=requires_grad)
            body_inv_inertia = wp.array(
                self.body_inv_inertia,
                dtype=wp.mat33,
                device=device,
                requires_grad=requires_grad,
            )
            if len(self.body_mass) > 0:
                correction_count = wp.zeros(1, dtype=wp.int32, device=device)
                wp.launch(
                    kernel=_sap_validate_and_correct_inertia_kernel,
                    dim=len(self.body_mass),
                    inputs=[
                        body_mass,
                        body_inertia,
                        body_inv_mass,
                        body_inv_inertia,
                        True,
                        0.0,
                        0.0,
                        correction_count,
                    ],
                    device=device,
                )
            sap_model = SapModel(
                device=wp.get_device(device),
                joint_count=len(self.joint_type),
                joint_dof_count=len(self.joint_qd),
                joint_coord_count=len(self.joint_q),
                body_count=len(self.body_q),
                articulation_count=len(self.articulation_start),
                world_count=max(1, int(self.world_count)),
                shape_count=self.shape_count,
                requires_grad=requires_grad,
                joint_type=wp.array(self.joint_type, dtype=wp.int32, device=device),
                joint_articulation=wp.array(self.joint_articulation, dtype=wp.int32, device=device),
                joint_parent=wp.array(self.joint_parent, dtype=wp.int32, device=device),
                joint_child=wp.array(self.joint_child, dtype=wp.int32, device=device),
                joint_q_start=wp.array(joint_q_start, dtype=wp.int32, device=device),
                joint_qd_start=wp.array(joint_qd_start, dtype=wp.int32, device=device),
                joint_dof_dim=wp.array(joint_dof_dim_np, dtype=wp.int32, ndim=2, device=device),
                joint_axis=wp.array(self.joint_axis, dtype=wp.vec3, device=device, requires_grad=requires_grad),
                joint_X_p=wp.array(self.joint_X_p, dtype=wp.transform, device=device, requires_grad=requires_grad),
                joint_X_c=wp.array(self.joint_X_c, dtype=wp.transform, device=device, requires_grad=requires_grad),
                joint_ancestor=wp.array(joint_ancestor, dtype=wp.int32, device=device),
                articulation_start=wp.array(articulation_start, dtype=wp.int32, device=device),
                joint_q=wp.array(self.joint_q, dtype=wp.float32, device=device, requires_grad=requires_grad),
                joint_qd=wp.array(self.joint_qd, dtype=wp.float32, device=device, requires_grad=requires_grad),
                body_q=wp.array(body_q, dtype=wp.transform, device=device, requires_grad=requires_grad),
                body_qd=wp.array(body_qd, dtype=wp.spatial_vector, device=device, requires_grad=requires_grad),
                body_mass=body_mass,
                body_inertia=body_inertia,
                body_com=wp.array(self.body_com, dtype=wp.vec3, device=device, requires_grad=requires_grad),
                body_flags=wp.array(self.body_flags, dtype=wp.int32, device=device),
                body_world=wp.array(self.body_world, dtype=wp.int32, device=device),
                gravity=wp.array(gravity, dtype=wp.vec3, device=device, requires_grad=requires_grad),
                joint_armature=wp.array(self.joint_armature, dtype=wp.float32, device=device, requires_grad=requires_grad),
                shape_body=wp.array(self.shape_body, dtype=wp.int32, device=device),
                shape_material_mu=wp.array(
                    self.shape_material_mu,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                shape_material_ke=wp.array(
                    self.shape_material_ke,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                shape_material_tau=wp.array(
                    self.shape_material_tau,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                joint_target_mode=wp.array(self.joint_target_mode, dtype=wp.int32, device=device),
                joint_target_ke=wp.array(
                    self.joint_target_ke,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                joint_target_kd=wp.array(
                    self.joint_target_kd,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                joint_target_pos=wp.array(
                    self.joint_target_pos,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                joint_target_vel=wp.array(
                    self.joint_target_vel,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                joint_f=wp.array(self.joint_f, dtype=wp.float32, device=device, requires_grad=requires_grad),
                joint_act=wp.array(self.joint_act, dtype=wp.float32, device=device, requires_grad=requires_grad),
                joint_effort_limit=wp.array(
                    self.joint_effort_limit,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                joint_limit_lower=wp.array(
                    self.joint_limit_lower,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                joint_limit_upper=wp.array(
                    self.joint_limit_upper,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                joint_limit_ke=wp.array(
                    self.joint_limit_ke,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                joint_limit_kd=wp.array(
                    self.joint_limit_kd,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                up_axis=2,
                shape_transform=wp.array(
                    self.shape_transform,
                    dtype=wp.transform,
                    device=device,
                    requires_grad=requires_grad,
                ),
                shape_type=wp.array(self.shape_type, dtype=wp.int32, device=device),
                shape_scale=wp.array(
                    self.shape_scale,
                    dtype=wp.vec3,
                    device=device,
                    requires_grad=requires_grad,
                ),
                shape_flags=wp.array(self.shape_flags, dtype=wp.int32, device=device),
                shape_margin=wp.array(
                    self.shape_margin,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                shape_gap=wp.array(
                    self.shape_gap,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                shape_is_solid=wp.array(self.shape_is_solid, dtype=wp.bool, device=device),
                shape_source=list(self.shape_source_refs),
                shape_collision_radius=wp.array(
                    self.shape_collision_radius,
                    dtype=wp.float32,
                    device=device,
                    requires_grad=requires_grad,
                ),
                shape_world=wp.array(self.shape_world, dtype=wp.int32, device=device),
                shape_sdf_index=wp.full(self.shape_count, -1, dtype=wp.int32, device=device),
                texture_sdf_data=wp.zeros(0, dtype=SapTextureSDFData, device=device),
                texture_sdf_coarse_textures=[],
                texture_sdf_subgrid_textures=[],
                texture_sdf_subgrid_start_slots=[],
                body_label=tuple(self.body_labels),
                joint_label=tuple(self.joint_labels),
                articulation_label=tuple(self.articulation_labels),
                shape_label=tuple(self.shape_labels),
            )
            if eval_initial_fk and int(sap_model.articulation_count) > 0:
                sap_eval_fk(sap_model, sap_model.joint_q, sap_model.joint_qd, sap_model)
            return sap_model

    def build_sap_state(self, *, device: Any = None, requires_grad: bool = False) -> SapState:
        return self.build_sap_model(device=device, requires_grad=requires_grad).state(requires_grad=requires_grad)

    def build_sap_control(self, *, device: Any = None, requires_grad: bool = False) -> SapControl:
        return self.build_sap_model(device=device, requires_grad=requires_grad).control(requires_grad=requires_grad)

    def _finalize_shape_sources(self, *, device: Any, requires_grad: bool) -> wp.array:
        geo_sources: list[int] = []
        finalized_geos: dict[int, int] = {}
        for geo in self.shape_source_refs:
            if geo is None:
                geo_sources.append(0)
                continue
            geo_hash = hash(geo)
            if geo_hash not in finalized_geos:
                finalized_geos[geo_hash] = int(geo.finalize(device=device, requires_grad=requires_grad))
            geo_sources.append(finalized_geos[geo_hash])
        return wp.array(geo_sources, dtype=wp.uint64, device=device)

    def _build_local_aabbs(self) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int, int]]]:
        aabb_lower: list[np.ndarray] = []
        aabb_upper: list[np.ndarray] = []
        voxel_resolution: list[tuple[int, int, int]] = []
        cache: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray, int, int, int]] = {}

        for shape_type, src, scale in zip(self.shape_type, self.shape_source_refs, self.shape_scale, strict=True):
            if shape_type in (int(SapGeoType.MESH), int(SapGeoType.CONVEX_MESH)) and src is not None:
                cache_key = (shape_type, id(src), tuple(scale))
            else:
                cache_key = (shape_type, tuple(scale))

            if cache_key in cache:
                lower, upper, nx, ny, nz = cache[cache_key]
            else:
                lower, upper = _compute_local_aabb(shape_type, src, scale)
                if shape_type in {
                    int(SapGeoType.BOX),
                    int(SapGeoType.SPHERE),
                    int(SapGeoType.CAPSULE),
                    int(SapGeoType.CYLINDER),
                    int(SapGeoType.CONE),
                    int(SapGeoType.ELLIPSOID),
                    int(SapGeoType.MESH),
                    int(SapGeoType.CONVEX_MESH),
                }:
                    nx, ny, nz = _compute_voxel_resolution_from_aabb(lower, upper, 100)
                else:
                    nx, ny, nz = 1, 1, 1
                cache[cache_key] = (lower, upper, nx, ny, nz)

            aabb_lower.append(lower)
            aabb_upper.append(upper)
            voxel_resolution.append((nx, ny, nz))

        return aabb_lower, aabb_upper, voxel_resolution

    def _find_shape_contact_pairs(self, *, device: Any) -> wp.array:
        filters = set(self.shape_collision_filter_pairs)
        pairs: list[tuple[int, int]] = []
        colliding_indices = [
            index
            for index, flags in enumerate(self.shape_flags)
            if flags & int(SapShapeFlags.COLLIDE_SHAPES)
        ]
        sorted_indices = sorted(colliding_indices, key=lambda index: self.shape_world[index])
        for i1, s1 in enumerate(sorted_indices):
            world1 = self.shape_world[s1]
            group1 = self.shape_collision_group[s1]
            for s2 in sorted_indices[i1 + 1 :]:
                world2 = self.shape_world[s2]
                group2 = self.shape_collision_group[s2]
                if world1 != -1 and world2 != -1 and world1 != world2:
                    break
                if not _test_world_and_group_pair(world1, world2, group1, group2):
                    continue
                shape_a, shape_b = (s2, s1) if s1 > s2 else (s1, s2)
                if (shape_a, shape_b) not in filters:
                    pairs.append((shape_a, shape_b))

        pair_array = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        return wp.array(pair_array, dtype=wp.vec2i, device=device)


def _joint_dof_count(joint_type: int, num_axes: int) -> tuple[int, int]:
    if joint_type == _SAP_JOINT_BALL:
        return 3, 4
    if joint_type in {_SAP_JOINT_FREE, _SAP_JOINT_DISTANCE}:
        return 6, 7
    if joint_type == _SAP_JOINT_FIXED:
        return 0, 0
    return int(num_axes), int(num_axes)


def _joint_dof_config(value: _SapJointDofConfig | Any) -> _SapJointDofConfig:
    if isinstance(value, _SapJointDofConfig):
        return value
    return _SapJointDofConfig(axis=_vec3_from_any(value))


def _unlimited_joint_dof_config(value: Any) -> _SapJointDofConfig:
    return _SapJointDofConfig(axis=_vec3_from_any(value), limit_ke=0.0, limit_kd=0.0)


def _free_joint_linear_axes() -> list[_SapJointDofConfig]:
    return [
        _unlimited_joint_dof_config((1.0, 0.0, 0.0)),
        _unlimited_joint_dof_config((0.0, 1.0, 0.0)),
        _unlimited_joint_dof_config((0.0, 0.0, 1.0)),
    ]


def _free_joint_angular_axes() -> list[_SapJointDofConfig]:
    return [
        _unlimited_joint_dof_config((1.0, 0.0, 0.0)),
        _unlimited_joint_dof_config((0.0, 1.0, 0.0)),
        _unlimited_joint_dof_config((0.0, 0.0, 1.0)),
    ]


@wp.kernel(enable_backward=False, module="unique")
def _sap_validate_and_correct_inertia_kernel(
    body_mass: wp.array(dtype=wp.float32),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inv_mass: wp.array(dtype=wp.float32),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    balance_inertia: wp.bool,
    bound_mass: wp.float32,
    bound_inertia: wp.float32,
    correction_count: wp.array(dtype=wp.int32),
):
    tid = wp.tid()

    mass = body_mass[tid]
    inertia = body_inertia[tid]
    was_corrected = False

    if (
        not wp.isfinite(mass)
        or not wp.isfinite(inertia[0, 0])
        or not wp.isfinite(inertia[0, 1])
        or not wp.isfinite(inertia[0, 2])
        or not wp.isfinite(inertia[1, 0])
        or not wp.isfinite(inertia[1, 1])
        or not wp.isfinite(inertia[1, 2])
        or not wp.isfinite(inertia[2, 0])
        or not wp.isfinite(inertia[2, 1])
        or not wp.isfinite(inertia[2, 2])
    ):
        mass = 0.0
        inertia = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        was_corrected = True

    if mass < 0.0:
        mass = 0.0
        was_corrected = True

    if bound_mass > 0.0 and mass < bound_mass and mass > 0.0:
        mass = bound_mass
        was_corrected = True

    if mass == 0.0:
        was_corrected = was_corrected or (wp.ddot(inertia, inertia) > 0.0)
        inertia = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        sym = wp.mat33(
            inertia[0, 0],
            (inertia[0, 1] + inertia[1, 0]) * 0.5,
            (inertia[0, 2] + inertia[2, 0]) * 0.5,
            (inertia[0, 1] + inertia[1, 0]) * 0.5,
            inertia[1, 1],
            (inertia[1, 2] + inertia[2, 1]) * 0.5,
            (inertia[0, 2] + inertia[2, 0]) * 0.5,
            (inertia[1, 2] + inertia[2, 1]) * 0.5,
            inertia[2, 2],
        )
        if wp.ddot(inertia - sym, inertia - sym) > 0.0:
            was_corrected = True
        inertia = sym

        _eigvecs, eigvals = wp.eig3(inertia)

        I1 = eigvals[0]
        I2 = eigvals[1]
        I3 = eigvals[2]
        if I1 > I2:
            I1, I2 = I2, I1
        if I2 > I3:
            I2, I3 = I3, I2
            if I1 > I2:
                I1, I2 = I2, I1

        if I1 < 1.0e-6:
            adjustment = -I1 + 1.0e-6
            I1 += adjustment
            I2 += adjustment
            I3 += adjustment
            inertia = inertia + wp.mat33(adjustment, 0.0, 0.0, 0.0, adjustment, 0.0, 0.0, 0.0, adjustment)
            was_corrected = True

        if bound_inertia > 0.0 and I1 < bound_inertia:
            adjustment = bound_inertia - I1
            I1 += adjustment
            I2 += adjustment
            I3 += adjustment
            inertia = inertia + wp.mat33(adjustment, 0.0, 0.0, 0.0, adjustment, 0.0, 0.0, 0.0, adjustment)
            was_corrected = True

        if balance_inertia and (I1 + I2 < I3 - 1.0e-6):
            deficit = I3 - I1 - I2
            adjustment = deficit + 1.0e-6
            inertia = inertia + wp.mat33(adjustment, 0.0, 0.0, 0.0, adjustment, 0.0, 0.0, 0.0, adjustment)
            was_corrected = True

    body_mass[tid] = mass
    body_inertia[tid] = inertia

    if mass > 0.0:
        body_inv_mass[tid] = 1.0 / mass
    else:
        body_inv_mass[tid] = 0.0

    if mass > 0.0:
        body_inv_inertia[tid] = wp.inverse(inertia)
    else:
        body_inv_inertia[tid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    if was_corrected:
        wp.atomic_add(correction_count, 0, 1)


def _vec3_from_any(value: Any) -> wp.vec3:
    if isinstance(value, wp.vec3):
        return value
    if isinstance(value, str):
        return _sap_axis_vector(_sap_axis_from_token(value))
    if isinstance(value, (int, np.integer)):
        return _sap_axis_vector(int(value))
    return _vec3(value)


def load_sap_scene_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML or JSON scene file, validate its schema version, and return the raw configuration
    mapping.
    """
    scene_path = Path(path)
    with scene_path.open("r", encoding="utf-8") as f:
        data = json.load(f) if scene_path.suffix.lower() == ".json" else yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SapSceneLoaderError(f"Scene file {scene_path} must contain a mapping.")
    version = int(data.get("schema_version", SCENE_SCHEMA_VERSION))
    if version != SCENE_SCHEMA_VERSION:
        raise SapSceneLoaderError(f"Unsupported scene schema_version={version}; expected {SCENE_SCHEMA_VERSION}.")
    return data


def load_sap_scene(
    path: str | Path,
    *,
    device: Any = None,
    rigid_contact_max: int | None = None,
    requires_grad: bool = False,
    strict: bool = True,
    num_worlds: int | None = None,
    spacing: tuple[float, float, float] | None = None,
) -> SapLoadedScene:
    """Load a scene file into SAP collision and solver runtime objects ready for benchmark or direct
    stepping code.
    """
    scene_path = Path(path)
    config = load_sap_scene_config(scene_path)
    builder, unsupported = _build_sap_scene_builder(
        config,
        strict=strict,
        num_worlds=num_worlds,
        spacing=spacing,
    )

    simulation_cfg = config.get("simulation", {}) or {}
    if rigid_contact_max is None:
        rigid_contact_max = int(simulation_cfg.get("max_rigid_contact", 0) or 0)

    sap_model = builder.build_sap_model(
        device=device,
        requires_grad=requires_grad,
        eval_initial_fk=_scene_requires_initial_fk(config),
    )
    model = builder.finalize(
        device=device,
        requires_grad=requires_grad,
        rigid_contact_max=int(rigid_contact_max or 0),
        sap_model=sap_model,
    )
    state = SapCollisionState(
        body_q=wp.clone(sap_model.body_q, requires_grad=requires_grad),
        requires_grad=requires_grad,
    )
    sap_state = sap_model.state(requires_grad=requires_grad)
    sap_control = sap_model.control(requires_grad=requires_grad)
    return SapLoadedScene(
        collision_model=model,
        collision_state=state,
        sap_model=sap_model,
        sap_state=sap_state,
        sap_control=sap_control,
        body_labels=tuple(builder.body_labels),
        shape_labels=tuple(builder.shape_labels),
        shape_colors=tuple(builder.shape_colors),
        unsupported_features=tuple(unsupported),
        source_path=scene_path,
    )


def _build_sap_scene_builder(
    config: dict[str, Any],
    *,
    strict: bool,
    num_worlds: int | None,
    spacing: tuple[float, float, float] | None,
) -> tuple[_SapSceneBuilder, list[str]]:
    unsupported: list[str] = []
    replicate_cfg = config.get("replicate", {}) or {}
    simulation_cfg = config.get("simulation", {}) or {}
    scene_world_count = simulation_cfg.get("num_worlds", replicate_cfg.get("num_worlds", 1))
    world_count = int(num_worlds if num_worlds is not None else scene_world_count)
    world_spacing = spacing if spacing is not None else _tuple3(replicate_cfg.get("spacing", DEFAULT_REPLICATE_SPACING))

    builder_cfg = config.get("builder", {}) or {}
    builder = _SapSceneBuilder(rigid_gap=_float(builder_cfg.get("rigid_gap", 0.1)))
    if "gravity" in builder_cfg:
        builder.gravity = _float(builder_cfg["gravity"])
    if "default_body_armature" in builder_cfg:
        builder.default_body_armature = _float(builder_cfg["default_body_armature"])
    defaults = builder_cfg.get("defaults", {}) or {}
    joint_defaults = defaults.get("joint", {}) or {}
    shape_defaults = defaults.get("shape", {}) or {}
    builder.default_joint_cfg.apply(joint_defaults)
    builder.default_shape_cfg.apply(_normalize_shape_cfg(shape_defaults))

    for asset_index, asset_cfg in enumerate(config.get("assets", []) or []):
        _add_asset(builder, asset_cfg, asset_index, unsupported)

    body_ids: dict[str, int] = {}
    for body_cfg in config.get("bodies", []) or []:
        body_id = _required_str(body_cfg, "id")
        body_shape_cfg = _object_shape_cfg_overrides(body_cfg)
        body = builder.add_body(
            xform=_transform(body_cfg.get("transform")),
            label=body_cfg.get("label", body_id),
            mass=_float(body_cfg.get("mass", 0.0)),
        )
        body_ids[body_id] = body
        for shape_cfg in body_cfg.get("shapes", []) or []:
            _add_shape(builder, body, shape_cfg, default_cfg=body_shape_cfg)

    joint_ids: dict[str, int] = {}
    for joint_cfg in config.get("joints", []) or []:
        joint_id = _required_str(joint_cfg, "id")
        try:
            joint_ids[joint_id] = _add_inline_joint(builder, joint_cfg, body_ids, label=joint_id)
        except SapSceneLoaderError as exc:
            unsupported.append(str(exc))

    for articulation_cfg in config.get("articulations", []) or []:
        joints = [joint_ids[_required_joint_name(name, joint_ids)] for name in articulation_cfg.get("joints", [])]
        builder.add_articulation(joints, label=articulation_cfg.get("id"))

    for entry in config.get("initial_joint_q", []) or []:
        _set_joint_q(builder, entry, joint_ids)

    for op in config.get("post_build", []) or []:
        _apply_post_build_op(builder, op, body_ids, unsupported)

    builder.replicate(world_count, spacing=world_spacing)
    _add_ground(builder, config)

    if strict and unsupported:
        raise SapUnsupportedSceneFeature(unsupported)
    return builder, unsupported


def _post_build_op_writes_joint_state(op: Any) -> bool:
    if not isinstance(op, dict):
        return False
    op_type = str(op.get("op", "")).lower()
    if op_type == "set_joint_q":
        return True
    if op_type == "set_array":
        return str(op.get("array", "")) in {"joint_q", "joint_qd"}
    if op_type == "copy_array":
        return str(op.get("dst", "")) in {"joint_q", "joint_qd"}
    return False


def _scene_requires_initial_fk(config: dict[str, Any]) -> bool:
    if config.get("initial_joint_q"):
        return True
    return any(_post_build_op_writes_joint_state(op) for op in config.get("post_build", []) or [])


def _add_asset(
    builder: _SapSceneBuilder,
    cfg: dict[str, Any],
    asset_index: int,
    unsupported: list[str],
) -> None:
    asset_type = str(cfg.get("type", "")).lower()
    asset_id = cfg.get("id", asset_index)
    if asset_type == "urdf":
        gaps = _urdf_asset_gaps(cfg)
        if gaps:
            unsupported.extend(f"asset[{asset_id!r}] urdf {gap}" for gap in gaps)
            return

        try:
            source = _resolve_asset_source(cfg.get("source"))
        except SapSceneLoaderError as exc:
            unsupported.append(f"asset[{asset_id!r}] urdf source: {exc}")
            return

        shape_start = builder.shape_count
        shape_overrides = _asset_shape_cfg_overrides(cfg)
        original_shape_cfg = builder.default_shape_cfg.copy()
        try:
            if shape_overrides:
                builder.default_shape_cfg.apply(shape_overrides)
            _add_urdf_asset(builder, source, cfg)
        finally:
            builder.default_shape_cfg = original_shape_cfg
        _apply_shape_cfg_overrides(builder, range(shape_start, builder.shape_count), shape_overrides)
        return

    if asset_type == "mjcf":
        gaps = _mjcf_asset_gaps(cfg)
        if gaps:
            unsupported.extend(f"asset[{asset_id!r}] mjcf {gap}" for gap in gaps)
            return

        try:
            source = _resolve_asset_source(cfg.get("source"))
            source = _prepare_mjcf_asset_source(source, cfg)
        except SapSceneLoaderError as exc:
            unsupported.append(f"asset[{asset_id!r}] mjcf source: {exc}")
            return

        shape_start = builder.shape_count
        shape_overrides = _asset_shape_cfg_overrides(cfg)
        original_shape_cfg = builder.default_shape_cfg.copy()
        try:
            if shape_overrides:
                builder.default_shape_cfg.apply(shape_overrides)
            _add_mjcf_asset(builder, source, cfg)
        except SapSceneLoaderError as exc:
            unsupported.append(f"asset[{asset_id!r}] mjcf: {exc}")
            return
        finally:
            builder.default_shape_cfg = original_shape_cfg
        _apply_shape_cfg_overrides(builder, range(shape_start, builder.shape_count), shape_overrides)
        return

    if asset_type == "usd":
        gaps = _usd_asset_gaps(cfg)
        if gaps:
            unsupported.extend(f"asset[{asset_id!r}] usd {gap}" for gap in gaps)
            return

        try:
            source = _resolve_asset_source(cfg.get("source"))
        except SapSceneLoaderError as exc:
            unsupported.append(f"asset[{asset_id!r}] usd source: {exc}")
            return

        shape_start = builder.shape_count
        shape_overrides = _asset_shape_cfg_overrides(cfg)
        original_shape_cfg = builder.default_shape_cfg.copy()
        try:
            if shape_overrides:
                builder.default_shape_cfg.apply(shape_overrides)
            _add_usd_asset(builder, source, cfg)
        except SapSceneLoaderError as exc:
            unsupported.append(f"asset[{asset_id!r}] usd: {exc}")
            return
        finally:
            builder.default_shape_cfg = original_shape_cfg
        _apply_shape_cfg_overrides(builder, range(shape_start, builder.shape_count), shape_overrides)
        return

    unsupported.append(f"asset[{asset_id!r}] type={asset_type or '<missing>'}")
    return


def _urdf_asset_gaps(cfg: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    if cfg.get("base_joint") is not None:
        gaps.append("base_joint")
    if int(cfg.get("parent_body", -1)) != -1:
        gaps.append("parent_body")
    if bool(cfg.get("collapse_fixed_joints", False)):
        gaps.append("collapse_fixed_joints")
    if bool(cfg.get("split_bimanual_root_articulation", False)):
        gaps.append("split_bimanual_root_articulation")
    if bool(cfg.get("override_root_xform", False)):
        gaps.append("override_root_xform")
    up_axis = str(cfg.get("up_axis", "z")).lower()
    if up_axis not in {"z", "axis.z", "2"}:
        gaps.append(f"up_axis={cfg.get('up_axis')!r}")
    return gaps


def _mjcf_asset_gaps(cfg: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    if cfg.get("base_joint") is not None:
        gaps.append("base_joint")
    if int(cfg.get("parent_body", -1)) != -1:
        gaps.append("parent_body")
    if bool(cfg.get("override_root_xform", False)):
        gaps.append("override_root_xform")
    up_axis = str(cfg.get("up_axis", "z")).lower()
    if up_axis not in {"z", "axis.z", "2"}:
        gaps.append(f"up_axis={cfg.get('up_axis')!r}")
    return gaps


def _usd_asset_gaps(cfg: dict[str, Any]) -> list[str]:
    gaps: list[str] = []
    if cfg.get("base_joint") is not None:
        gaps.append("base_joint")
    if int(cfg.get("parent_body", -1)) != -1:
        gaps.append("parent_body")
    if bool(cfg.get("override_root_xform", False)):
        gaps.append("override_root_xform")
    up_axis = str(cfg.get("up_axis", "z")).lower()
    if up_axis not in {"z", "axis.z", "2"}:
        gaps.append(f"up_axis={cfg.get('up_axis')!r}")
    joint_ordering = cfg.get("joint_ordering", "dfs")
    if joint_ordering not in {"dfs", "bfs", None}:
        gaps.append(f"joint_ordering={joint_ordering!r}")
    return gaps


def _resolve_asset_source(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    if not isinstance(spec, dict):
        raise SapSceneLoaderError(f"Expected asset source string or mapping, got {spec!r}.")
    if "path" in spec:
        return str(Path(spec["path"]))
    if "git" in spec:
        return str(_resolve_git_asset_source(spec["git"]))
    if "examples_asset" in spec:
        raise SapSceneLoaderError(f"examples_asset={spec['examples_asset']!r}")
    raise SapSceneLoaderError(f"Unsupported asset source: {spec}")


def _resolve_git_asset_source(spec: Any) -> Path:
    if not isinstance(spec, dict):
        raise SapSceneLoaderError(f"source.git must be a mapping, got {spec!r}.")
    repo = str(spec.get("repo", "")).strip()
    rev = str(spec.get("rev", "")).strip()
    asset_path = Path(str(spec.get("path", "")).strip())
    if not repo:
        raise SapSceneLoaderError("source.git.repo is required.")
    if not rev:
        raise SapSceneLoaderError("source.git.rev is required.")
    if not str(asset_path) or asset_path.is_absolute() or ".." in asset_path.parts:
        raise SapSceneLoaderError(f"source.git.path must be a relative path inside the asset repo, got {asset_path!s}.")

    cache_root = Path(os.environ.get("SAP_WARP_ASSET_CACHE", Path.home() / ".cache" / "sap_warp" / "assets"))
    repo_key = _asset_cache_key(repo)
    checkout_root = cache_root / "git" / repo_key / rev
    target = checkout_root / asset_path
    if target.exists():
        return target

    if os.environ.get("SAP_WARP_ASSET_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise SapSceneLoaderError(f"Git asset {repo}@{rev}:{asset_path!s} is not cached and SAP_WARP_ASSET_OFFLINE is set.")
    if shutil.which("git") is None:
        raise SapSceneLoaderError("source.git requires the git executable.")

    sparse_paths = _normalize_git_sparse_paths(spec.get("sparse"), asset_path)
    checkout_root.parent.mkdir(parents=True, exist_ok=True)
    if not (checkout_root / ".git").exists():
        if checkout_root.exists():
            shutil.rmtree(checkout_root)
        _run_git(["clone", "--filter=blob:none", "--no-checkout", repo, str(checkout_root)])
    _run_git(["-C", str(checkout_root), "fetch", "--depth=1", "origin", rev])
    _run_git(["-C", str(checkout_root), "sparse-checkout", "init", "--cone"])
    _run_git(["-C", str(checkout_root), "sparse-checkout", "set", *sparse_paths])
    _run_git(["-C", str(checkout_root), "checkout", "FETCH_HEAD"])

    if not target.exists():
        raise SapSceneLoaderError(f"Git asset path {asset_path!s} was not found after fetching {repo}@{rev}.")
    return target


def _asset_cache_key(repo: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", repo).strip("_") or "repo"


def _normalize_git_sparse_paths(raw: Any, asset_path: Path) -> list[str]:
    if raw is None:
        paths = [asset_path.parent if asset_path.parent != Path(".") else asset_path]
    elif isinstance(raw, str):
        paths = [Path(raw)]
    else:
        paths = [Path(str(item)) for item in raw]
    normalized: list[str] = []
    for path in paths:
        if path.is_absolute() or ".." in path.parts:
            raise SapSceneLoaderError(f"source.git.sparse entries must be relative paths inside the asset repo, got {path!s}.")
        text = str(path)
        if text and text != ".":
            normalized.append(text)
    if not normalized:
        normalized.append(str(asset_path.parent if asset_path.parent != Path(".") else asset_path))
    return list(dict.fromkeys(normalized))


def _run_git(args: list[str]) -> None:
    command = ["git", *args]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise SapSceneLoaderError(f"Git asset command failed: {' '.join(command)}\n{detail}") from exc


def _prepare_mjcf_asset_source(source: str, cfg: dict[str, Any]) -> str:
    external_assets = cfg.get("external_assets")
    if not external_assets:
        return source
    if not isinstance(external_assets, list):
        raise SapSceneLoaderError("external_assets must be a list.")

    source_text = os.fspath(source)
    if _is_xml_content(source_text):
        raise SapSceneLoaderError("external_assets require an MJCF file path source, not inline XML.")

    source_path = Path(source_text).resolve()
    if not source_path.is_file():
        raise SapSceneLoaderError(f"MJCF source path does not exist: {source_text}")

    cache_root = Path(os.environ.get("SAP_WARP_ASSET_CACHE", Path.home() / ".cache" / "sap_warp" / "assets"))
    overlay_key_data = json.dumps(
        {
            "source_dir": str(source_path.parent),
            "external_assets": external_assets,
        },
        sort_keys=True,
        default=str,
    )
    overlay_key = hashlib.sha256(overlay_key_data.encode("utf-8")).hexdigest()[:16]
    overlay_dir = cache_root / "mjcf_overlay" / _asset_cache_key(source_path.stem) / overlay_key
    overlay_dir.mkdir(parents=True, exist_ok=True)

    links = [_normalize_mjcf_external_asset_link(entry) for entry in external_assets]
    reserved_roots = {target.parts[0] for target, _ in links if target.parts}
    for child in source_path.parent.iterdir():
        if child.name in reserved_roots:
            continue
        _replace_cache_path_with_link(overlay_dir / child.name, child)

    for target, asset_source in links:
        resolved_source = Path(_resolve_asset_source(asset_source)).resolve()
        if not resolved_source.exists():
            raise SapSceneLoaderError(f"external asset source does not exist after resolution: {resolved_source}")
        _replace_cache_path_with_link(overlay_dir / target, resolved_source)

    return str(overlay_dir / source_path.name)


def _normalize_mjcf_external_asset_link(entry: Any) -> tuple[Path, Any]:
    if not isinstance(entry, dict):
        raise SapSceneLoaderError(f"external_assets entries must be mappings, got {entry!r}.")
    if "target" not in entry:
        raise SapSceneLoaderError("external_assets entries require target.")
    target = Path(str(entry["target"]))
    if not str(target) or target.is_absolute() or ".." in target.parts:
        raise SapSceneLoaderError(f"external_assets target must be a relative path, got {target!s}.")

    if "source" in entry:
        asset_source = entry["source"]
    elif "git" in entry:
        asset_source = {"git": entry["git"]}
    elif "path" in entry:
        asset_source = {"path": entry["path"]}
    else:
        raise SapSceneLoaderError(f"external_assets entry for {target!s} requires source.")
    return target, asset_source


def _replace_cache_path_with_link(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = target.resolve()
    if path.is_symlink() and path.resolve() == target:
        return
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    try:
        os.symlink(target, path, target_is_directory=target.is_dir())
    except (NotImplementedError, OSError):
        if target.is_dir():
            shutil.copytree(target, path)
        else:
            shutil.copy2(target, path)


def _asset_mesh_scale(cfg: dict[str, Any]) -> tuple[float, float, float] | None:
    value = cfg.get("mesh_scale")
    source = cfg.get("source")
    if value is None and isinstance(source, dict):
        value = source.get("mesh_scale")
    if value is None:
        return None
    if isinstance(value, (int, float)):
        scale = float(value)
        return (scale, scale, scale)
    values = tuple(float(v) for v in value)
    if len(values) != 3:
        raise SapSceneLoaderError(f"mesh_scale must be a scalar or length-3 sequence, got {value!r}.")
    return values


def _asset_shape_cfg_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    overrides = _object_shape_cfg_overrides(cfg)
    raw_cfg = cfg.get("cfg")
    if isinstance(raw_cfg, dict):
        overrides.update(_normalize_shape_cfg(raw_cfg))
    return overrides


def _apply_shape_cfg_overrides(builder: _SapSceneBuilder, indices, overrides: dict[str, Any]) -> None:
    if not overrides:
        return
    builder_arrays = {
        "margin": "shape_margin",
        "gap": "shape_gap",
        "mu": "shape_material_mu",
        "ke": "shape_material_ke",
        "tau": "shape_material_tau",
    }
    for index in indices:
        for name, value in overrides.items():
            array_name = builder_arrays.get(name)
            if array_name is None:
                continue
            getattr(builder, array_name)[int(index)] = _coerce_config_value(value)


def _add_urdf_asset(builder: _SapSceneBuilder, source: str, cfg: dict[str, Any]) -> None:
    source_text = os.fspath(source)
    source_path = Path(source_text)
    if source_path.is_file():
        urdf_root = ET.parse(source_path).getroot()
        source_base = source_path.parent
    else:
        stripped = source_text.strip()
        if not stripped.startswith("<"):
            raise SapSceneLoaderError(f"URDF source path does not exist: {source}")
        urdf_root = ET.fromstring(stripped)
        source_base = None

    scale = _float(cfg.get("scale", 1.0))
    xform = _transform(cfg.get("xform")) or wp.transform_identity()
    mesh_scale = _asset_mesh_scale(cfg)
    mesh_maxhullvert = int(cfg.get("mesh_maxhullvert", SapMesh.MAX_HULL_VERTICES))
    load_visual_shapes = bool(cfg.get("load_visual_shapes", True))
    parse_visuals_as_colliders = bool(cfg.get("parse_visuals_as_colliders", False))
    force_show_colliders = bool(cfg.get("force_show_colliders", False))
    enable_self_collisions = bool(cfg.get("enable_self_collisions", True))
    ignore_inertial_definitions = bool(cfg.get("ignore_inertial_definitions", False))
    floating = cfg.get("floating")
    joint_ordering = cfg.get("joint_ordering", "dfs")
    bodies_follow_joint_ordering = bool(cfg.get("bodies_follow_joint_ordering", True))

    material_table = _parse_urdf_material_table(urdf_root, source_base)
    articulation_label = urdf_root.attrib.get("name")

    def make_label(name: str) -> str:
        return f"{articulation_label}/{name}" if articulation_label else name

    def parse_transform(element: ET.Element | None) -> wp.transform:
        if element is None or element.find("origin") is None:
            return wp.transform()
        origin = element.find("origin")
        assert origin is not None
        xyz = [float(x) * scale for x in (origin.get("xyz") or "0 0 0").split()]
        rpy = [float(x) for x in (origin.get("rpy") or "0 0 0").split()]
        return wp.transform(xyz, wp.quat_rpy(*rpy))

    def resolve_material(material_element: ET.Element | None) -> dict[str, Any]:
        if material_element is None:
            return {"color": None, "texture": None}
        mat_name = material_element.get("name")
        color, texture = _parse_urdf_material_properties(material_element, source_base)
        if mat_name and mat_name in material_table:
            resolved = dict(material_table[mat_name])
        else:
            resolved = {"color": None, "texture": None}
        if color is not None:
            resolved["color"] = color
        if texture is not None:
            resolved["texture"] = texture
        if mat_name and mat_name not in material_table and any(value is not None for value in resolved.values()):
            material_table[mat_name] = dict(resolved)
        return resolved

    def parse_shapes(
        link: int,
        geoms: Sequence[ET.Element],
        density: float,
        incoming_xform: wp.transform | None = None,
        visible: bool = True,
        just_visual: bool = False,
    ) -> list[int]:
        shape_cfg = builder.default_shape_cfg.copy()
        shape_cfg.density = density
        shape_cfg.is_visible = visible
        shape_cfg.has_shape_collision = not just_visual
        shape_cfg.has_particle_collision = not just_visual
        shapes: list[int] = []

        for geom_group in geoms:
            geo = geom_group.find("geometry")
            if geo is None:
                continue
            tf = parse_transform(geom_group)
            if incoming_xform is not None:
                tf = incoming_xform * tf

            material_info = {"color": None, "texture": None}
            if just_visual:
                material_info = resolve_material(geom_group.find("material"))

            for box in geo.findall("box"):
                size = [float(x) for x in (box.get("size") or "1 1 1").split()]
                shapes.append(
                    builder.add_shape(
                        body=link,
                        shape_type=int(SapGeoType.BOX),
                        xform=tf,
                        cfg=shape_cfg,
                        scale=(size[0] * 0.5 * scale, size[1] * 0.5 * scale, size[2] * 0.5 * scale),
                    )
                )

            for sphere in geo.findall("sphere"):
                shapes.append(
                    builder.add_shape(
                        body=link,
                        shape_type=int(SapGeoType.SPHERE),
                        xform=tf,
                        cfg=shape_cfg,
                        scale=(float(sphere.get("radius") or "1") * scale, 0.0, 0.0),
                    )
                )

            for cylinder in geo.findall("cylinder"):
                xform = wp.transform(tf.p, tf.q)
                shapes.append(
                    builder.add_shape(
                        body=link,
                        shape_type=int(SapGeoType.CYLINDER),
                        xform=xform,
                        cfg=shape_cfg,
                        scale=(
                            float(cylinder.get("radius") or "1") * scale,
                            float(cylinder.get("length") or "1") * 0.5 * scale,
                            0.0,
                        ),
                    )
                )

            for capsule in geo.findall("capsule"):
                xform = wp.transform(tf.p, tf.q)
                shapes.append(
                    builder.add_shape(
                        body=link,
                        shape_type=int(SapGeoType.CAPSULE),
                        xform=xform,
                        cfg=shape_cfg,
                        scale=(
                            float(capsule.get("radius") or "1") * scale,
                            float(capsule.get("height") or "1") * 0.5 * scale,
                            0.0,
                        ),
                    )
                )

            for mesh_node in geo.findall("mesh"):
                filename = mesh_node.get("filename")
                if filename is None:
                    continue
                scaling = np.asarray([float(x) * scale for x in (mesh_node.get("scale") or "1 1 1").split()])
                if mesh_scale is not None:
                    scaling = scaling * np.asarray(mesh_scale, dtype=np.float32)
                resolved = _resolve_urdf_asset(filename, source_text, source_base)
                if resolved is None:
                    continue
                for sap_mesh in _load_sap_meshes_from_file(
                    resolved,
                    scale=scaling,
                    maxhullvert=mesh_maxhullvert,
                    override_color=material_info["color"],
                    override_texture=material_info["texture"],
                ):
                    if sap_mesh.texture is not None and sap_mesh.uvs is None:
                        sap_mesh.texture = None
                    mesh_cfg = shape_cfg.copy()
                    mesh_cfg.sdf_max_resolution = None
                    mesh_cfg.sdf_target_voxel_size = None
                    mesh_cfg.sdf_narrow_band_range = (-0.1, 0.1)
                    shapes.append(
                        builder.add_shape(
                            body=link,
                            shape_type=int(SapGeoType.MESH),
                            xform=tf,
                            cfg=mesh_cfg,
                            scale=(1.0, 1.0, 1.0),
                            src=sap_mesh,
                        )
                    )

        return shapes

    joints: list[dict[str, Any]] = []
    for joint in urdf_root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        joint_data: dict[str, Any] = {
            "name": joint.get("name"),
            "parent": parent.get("link"),
            "child": child.get("link"),
            "type": joint.get("type"),
            "origin": parse_transform(joint),
            "axis": wp.vec3(1.0, 0.0, 0.0),
            "limit_lower": -1.0e10,
            "limit_upper": 1.0e10,
            "damping": 0.0,
        }
        el_axis = joint.find("axis")
        if el_axis is not None:
            ax = el_axis.get("xyz", "1 0 0").strip().split()
            joint_data["axis"] = wp.vec3(float(ax[0]), float(ax[1]), float(ax[2]))
        limit = joint.find("limit")
        if limit is not None:
            if limit.get("lower") is not None:
                joint_data["limit_lower"] = float(limit.get("lower", "0.0"))
            if limit.get("upper") is not None:
                joint_data["limit_upper"] = float(limit.get("upper", "0.0"))
        dynamics = joint.find("dynamics")
        if dynamics is not None:
            joint_data["damping"] = float(dynamics.get("damping", "0.0"))
        joints.append(joint_data)

    urdf_links: list[ET.Element] = []
    sorted_joints: list[dict[str, Any]] = []
    if joints:
        if joint_ordering is not None:
            joint_edges = [(joint["parent"], joint["child"]) for joint in joints]
            sorted_joint_ids = _topological_sort(joint_edges, use_dfs=joint_ordering == "dfs")
            sorted_joints = [joints[i] for i in sorted_joint_ids]
        else:
            sorted_joints = joints

        if bodies_follow_joint_ordering:
            body_order = [sorted_joints[0]["parent"]] + [joint["child"] for joint in sorted_joints]
            for body_name in body_order:
                urdf_link = urdf_root.find(f"link[@name='{body_name}']")
                if urdf_link is None:
                    raise SapSceneLoaderError(f"Link {body_name} not found in URDF {source}")
                urdf_links.append(urdf_link)
    if not urdf_links:
        urdf_links = list(urdf_root.findall("link"))

    link_index: dict[str, int] = {}
    visual_shapes: list[int] = []
    start_shape_count = builder.shape_count

    for urdf_link in urdf_links:
        name = urdf_link.get("name")
        if name is None:
            raise SapSceneLoaderError(f"Link has no name in URDF {source}")
        link = builder.add_body(xform=None, label=make_label(name))
        link_index[name] = link

        visuals = list(urdf_link.findall("visual"))
        colliders = list(urdf_link.findall("collision"))
        if parse_visuals_as_colliders:
            colliders = visuals
        elif load_visual_shapes:
            visual_shapes.extend(parse_shapes(link, visuals, density=0.0, just_visual=True, visible=True))

        show_colliders = _should_show_collider(
            force_show_colliders,
            has_visual_shapes=len(visuals) > 0 and load_visual_shapes,
            parse_visuals_as_colliders=parse_visuals_as_colliders,
        )
        parse_shapes(link, colliders, density=builder.default_shape_cfg.density, visible=show_colliders)
        inertial = urdf_link.find("inertial")
        if not ignore_inertial_definitions and inertial is not None:
            inertial_frame = parse_transform(inertial)
            builder.body_com[link] = inertial_frame.p
            inertia_node = inertial.find("inertia")
            if inertia_node is not None:
                inertia_np = np.zeros((3, 3), dtype=np.float32)
                inertia_np[0, 0] = float(inertia_node.get("ixx", 0.0)) * scale**2
                inertia_np[1, 1] = float(inertia_node.get("iyy", 0.0)) * scale**2
                inertia_np[2, 2] = float(inertia_node.get("izz", 0.0)) * scale**2
                inertia_np[0, 1] = float(inertia_node.get("ixy", 0.0)) * scale**2
                inertia_np[0, 2] = float(inertia_node.get("ixz", 0.0)) * scale**2
                inertia_np[1, 2] = float(inertia_node.get("iyz", 0.0)) * scale**2
                inertia_np[1, 0] = inertia_np[0, 1]
                inertia_np[2, 0] = inertia_np[0, 2]
                inertia_np[2, 1] = inertia_np[1, 2]
                rot = wp.quat_to_matrix(inertial_frame.q)
                inertia = rot @ wp.mat33(inertia_np)
                builder.body_inertia[link] = inertia
                builder.body_inv_inertia[link] = wp.inverse(inertia) if any(x for x in inertia) else inertia
            mass_node = inertial.find("mass")
            if mass_node is not None:
                mass = float(mass_node.get("value", 0.0))
                builder.body_mass[link] = mass
                builder.body_inv_mass[link] = 1.0 / mass if mass > 0.0 else 0.0

    end_shape_count = builder.shape_count

    base_link_name = sorted_joints[0]["parent"] if sorted_joints else next(iter(link_index.keys()))
    root = link_index[base_link_name]
    joint_indices: list[int] = []
    if floating and int(cfg.get("parent_body", -1)) == -1:
        base_joint = builder.add_joint(
            parent=-1,
            child=root,
            joint_type=_SAP_JOINT_FREE,
            linear_axes=_free_joint_linear_axes(),
            angular_axes=_free_joint_angular_axes(),
            label=make_label("floating_base"),
            collision_filter_parent=True,
        )
        start = builder.joint_q_start[base_joint]
        builder.joint_q[start + 0] = float(xform.p[0])
        builder.joint_q[start + 1] = float(xform.p[1])
        builder.joint_q[start + 2] = float(xform.p[2])
        builder.joint_q[start + 3] = float(xform.q[0])
        builder.joint_q[start + 4] = float(xform.q[1])
        builder.joint_q[start + 5] = float(xform.q[2])
        builder.joint_q[start + 6] = float(xform.q[3])
    else:
        base_joint = builder.add_joint(
            parent=-1,
            child=root,
            joint_type=_SAP_JOINT_FIXED,
            parent_xform=xform,
            label=make_label("fixed_base"),
            collision_filter_parent=True,
        )
    joint_indices.append(base_joint)

    for joint in sorted_joints:
        parent = link_index[joint["parent"]]
        child = link_index[joint["child"]]
        joint_type = str(joint.get("type", "fixed")).lower()
        common = {
            "parent": parent,
            "child": child,
            "parent_xform": joint["origin"],
            "label": make_label(joint.get("name") or f"joint_{len(builder.joint_type)}"),
            "collision_filter_parent": True,
        }
        if joint_type in {"revolute", "continuous"}:
            joint_id = builder.add_joint(
                joint_type=_SAP_JOINT_REVOLUTE,
                angular_axes=[
                    _SapJointDofConfig(
                        axis=joint["axis"],
                        target_kd=float(joint["damping"]),
                        limit_lower=float(joint["limit_lower"]),
                        limit_upper=float(joint["limit_upper"]),
                    )
                ],
                **common,
            )
        elif joint_type == "prismatic":
            joint_id = builder.add_joint(
                joint_type=_SAP_JOINT_PRISMATIC,
                linear_axes=[
                    _SapJointDofConfig(
                        axis=joint["axis"],
                        target_kd=float(joint["damping"]),
                        limit_lower=float(joint["limit_lower"]) * scale,
                        limit_upper=float(joint["limit_upper"]) * scale,
                    )
                ],
                **common,
            )
        elif joint_type == "floating":
            joint_id = builder.add_joint(
                joint_type=_SAP_JOINT_FREE,
                linear_axes=_free_joint_linear_axes(),
                angular_axes=_free_joint_angular_axes(),
                **common,
            )
        elif joint_type == "planar":
            axis = np.asarray(joint["axis"], dtype=np.float32)
            axis /= np.linalg.norm(axis)
            helper = np.array([1.0, 0.0, 0.0], dtype=np.float32) if np.allclose(axis, [0.0, 1.0, 0.0]) else np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32,
            )
            u = np.cross(helper, axis)
            u /= np.linalg.norm(u)
            v = np.cross(axis, u)
            v /= np.linalg.norm(v)
            joint_id = builder.add_joint(
                joint_type=_SAP_JOINT_D6,
                linear_axes=[
                    _SapJointDofConfig(axis=wp.vec3(*u), limit_lower=float(joint["limit_lower"]) * scale, limit_upper=float(joint["limit_upper"]) * scale),
                    _SapJointDofConfig(axis=wp.vec3(*v), limit_lower=float(joint["limit_lower"]) * scale, limit_upper=float(joint["limit_upper"]) * scale),
                ],
                **common,
            )
        else:
            joint_id = builder.add_joint(joint_type=_SAP_JOINT_FIXED, **common)
        joint_indices.append(joint_id)

    builder.add_articulation(joint_indices, label=articulation_label)

    for i in range(start_shape_count, end_shape_count):
        for j in visual_shapes:
            builder.add_shape_collision_filter_pair(i, j)

    if not enable_self_collisions:
        for i in range(start_shape_count, end_shape_count):
            if builder.shape_body[i] < 0:
                continue
            for j in range(i + 1, end_shape_count):
                if builder.shape_body[j] < 0:
                    continue
                builder.add_shape_collision_filter_pair(i, j)


def _should_show_collider(
    force_show_colliders: bool,
    has_visual_shapes: bool,
    parse_visuals_as_colliders: bool = False,
) -> bool:
    if force_show_colliders or parse_visuals_as_colliders:
        return True
    return not has_visual_shapes


@dataclass
class _SapUsdPhysicsMaterial:
    static_friction: float
    dynamic_friction: float
    torsional_friction: float
    rolling_friction: float
    restitution: float
    density: float


def _add_usd_asset(builder: _SapSceneBuilder, source: str, cfg: dict[str, Any]) -> None:
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise SapSceneLoaderError("pxr/usd-core is not available") from exc

    stage = Usd.Stage.Open(os.fspath(source), Usd.Stage.LoadAll)
    if stage is None:
        raise SapSceneLoaderError(f"USD stage could not be opened: {source}")

    xform = _transform(cfg.get("xform"))
    floating = cfg.get("floating")
    _ = floating
    only_load_enabled_rigid_bodies = bool(cfg.get("only_load_enabled_rigid_bodies", False))
    only_load_enabled_joints = bool(cfg.get("only_load_enabled_joints", True))
    ignore_paths = [str(path) for path in (cfg.get("ignore_paths") or [])]
    enable_self_collisions = bool(cfg.get("enable_self_collisions", True))
    apply_up_axis_from_stage = bool(cfg.get("apply_up_axis_from_stage", False))
    root_path = str(cfg.get("root_path", "/"))
    joint_ordering = cfg.get("joint_ordering", "dfs")
    bodies_follow_joint_ordering = bool(cfg.get("bodies_follow_joint_ordering", True))
    skip_mesh_approximation = bool(cfg.get("skip_mesh_approximation", False))
    load_sites = bool(cfg.get("load_sites", True))
    load_visual_shapes = bool(cfg.get("load_visual_shapes", True))
    hide_collision_shapes = bool(cfg.get("hide_collision_shapes", False))
    force_show_colliders = bool(cfg.get("force_show_colliders", False))
    mesh_maxhullvert = int(cfg.get("mesh_maxhullvert", SapMesh.MAX_HULL_VERTICES))

    stage_up_axis = _sap_axis_from_token(str(UsdGeom.GetStageUpAxis(stage)))
    if apply_up_axis_from_stage:
        axis_xform = wp.transform_identity()
    else:
        axis_xform = wp.transform(wp.vec3(0.0), _sap_quat_between_axes(stage_up_axis, 2))
    incoming_world_xform = axis_xform if xform is None else xform * axis_xform

    non_regex_ignore_paths = [path for path in ignore_paths if ".*" not in path]
    ret_dict = UsdPhysics.LoadUsdPhysicsFromRange(stage, [root_path], excludePaths=non_regex_ignore_paths)

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    path_body_map: dict[str, int] = {}
    path_shape_map: dict[str, int] = {}
    path_shape_scale: dict[str, wp.vec3] = {}
    path_joint_map: dict[str, int] = {}
    material_props_cache: dict[str, dict[str, Any]] = {}
    mesh_cache: dict[tuple[str, bool, bool], SapMesh] = {}
    bodies_with_visual_shapes: set[int] = set()
    degrees_to_radian = math.pi / 180.0
    force_position_velocity_actuation = bool(cfg.get("force_position_velocity_actuation", False))
    joint_drive_gains_scaling = _float(cfg.get("joint_drive_gains_scaling", 1.0))
    default_joint_limit_ke = builder.default_joint_cfg.limit_ke
    default_joint_limit_kd = builder.default_joint_cfg.limit_kd
    default_joint_armature = builder.default_joint_cfg.armature
    default_joint_velocity_limit = builder.default_joint_cfg.velocity_limit
    default_joint_friction = builder.default_joint_cfg.friction

    if UsdPhysics.ObjectType.Scene in ret_dict:
        scene_paths, scene_descs = ret_dict[UsdPhysics.ObjectType.Scene]
        if scene_paths:
            scene_desc = scene_descs[0]
            builder.gravity = -float(scene_desc.gravityMagnitude) if hasattr(builder, "gravity") else -float(
                scene_desc.gravityMagnitude
            )

    def ignored(path: str) -> bool:
        return any(re.match(pattern, path) for pattern in ignore_paths)

    def warn_invalid_desc(path: Any, descriptor: Any) -> bool:
        return hasattr(descriptor, "isValid") and not bool(descriptor.isValid)

    def data_for_key(key: Any):
        if key not in ret_dict:
            return
        yield from zip(*ret_dict[key], strict=False)

    def is_enabled_collider(prim: Any) -> bool:
        collider = UsdPhysics.CollisionAPI(prim)
        if collider:
            value = collider.GetCollisionEnabledAttr().Get()
            return bool(value)
        return False

    def get_material_props_cached(prim: Any) -> dict[str, Any]:
        prim_path = str(prim.GetPath())
        if prim_path not in material_props_cache:
            material_props_cache[prim_path] = _usd_resolve_material_properties_for_prim(prim)
        return material_props_cache[prim_path]

    def get_mesh_cached(prim: Any, *, load_uvs: bool = False, load_normals: bool = False) -> SapMesh:
        prim_path = str(prim.GetPath())
        key = (prim_path, load_uvs, load_normals)
        if key in mesh_cache:
            return mesh_cache[key]
        for cached_key in [
            (prim_path, True, True),
            (prim_path, load_uvs, True),
            (prim_path, True, load_normals),
        ]:
            if cached_key != key and cached_key in mesh_cache:
                return mesh_cache[cached_key]
        mesh = _usd_get_mesh(
            prim,
            load_uvs=load_uvs,
            load_normals=load_normals,
            maxhullvert=mesh_maxhullvert,
        )
        mesh_cache[key] = mesh
        return mesh

    def get_mesh_with_visual_material(prim: Any, *, path_name: str) -> SapMesh:
        material_props = get_material_props_cached(prim)
        texture = material_props.get("texture")
        physics_mesh = get_mesh_cached(prim)
        if texture is not None:
            render_mesh = get_mesh_cached(prim, load_uvs=True)
            mesh = SapMesh(
                render_mesh.vertices,
                render_mesh.indices,
                normals=render_mesh.normals,
                uvs=render_mesh.uvs,
                compute_inertia=False,
                is_solid=physics_mesh.is_solid,
                maxhullvert=physics_mesh.maxhullvert,
                sdf=physics_mesh.sdf,
            )
            mesh.mass = physics_mesh.mass
            mesh.com = physics_mesh.com
            mesh.inertia = physics_mesh.inertia
            mesh.has_inertia = physics_mesh.has_inertia
        else:
            mesh = physics_mesh.copy(recompute_inertia=False)
        if texture:
            mesh.texture = texture
        if mesh.texture is not None and mesh.uvs is None:
            warnings.warn(
                f"Warning: mesh {path_name} has a texture but no UVs; texture will be ignored.",
                stacklevel=2,
            )
            mesh.texture = None
        if material_props.get("color") is not None and mesh.texture is None:
            mesh.color = material_props["color"]
        if material_props.get("roughness") is not None:
            mesh.roughness = material_props["roughness"]
        if material_props.get("metallic") is not None:
            mesh.metallic = material_props["metallic"]
        return mesh

    def has_visual_material_properties(material_props: dict[str, Any]) -> bool:
        return any(material_props.get(key) is not None for key in ("texture", "roughness", "metallic"))

    def xform_to_mat44(xform_value: wp.transform) -> wp.mat44:
        return wp.transform_compose(xform_value.p, xform_value.q, wp.vec3(1.0))

    def get_prim_world_mat(prim: Any, articulation_root_xform: wp.transform | None, incoming_xform: wp.transform | None):
        prim_world_mat = _usd_get_transform_matrix(prim, local=False, xform_cache=xform_cache)
        if articulation_root_xform is not None:
            prim_world_mat = xform_to_mat44(wp.transform_inverse(articulation_root_xform)) @ prim_world_mat
        if incoming_xform is not None:
            prim_world_mat = xform_to_mat44(incoming_xform) @ prim_world_mat
        return prim_world_mat

    def visual_cfg(*, is_site: bool = False) -> _SapShapeConfig:
        shape_cfg = _SapShapeConfig()
        shape_cfg.density = 0.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False
        if is_site:
            shape_cfg.is_site = True
            shape_cfg.collision_group = 0
        return shape_cfg

    def load_visual_shapes_impl(
        parent_body_id: int,
        prim: Any,
        body_xform: wp.transform | None = None,
        articulation_root_xform: wp.transform | None = None,
    ) -> None:
        if is_enabled_collider(prim) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return
        path_name = str(prim.GetPath())
        if ignored(path_name):
            return

        prim_world_mat = get_prim_world_mat(
            prim,
            articulation_root_xform,
            incoming_world_xform if (parent_body_id == -1 or body_xform is not None) else None,
        )
        if body_xform is not None:
            body_world_mat = xform_to_mat44(body_xform)
            rel_mat = wp.inverse(body_world_mat) @ prim_world_mat
        else:
            rel_mat = prim_world_mat

        xform_pos, xform_rot, scale = wp.transform_decompose(rel_mat)
        shape_xform = wp.transform(xform_pos, xform_rot)

        if prim.IsInstance():
            proto = prim.GetPrototype()
            for child in proto.GetChildren():
                inst_path = child.GetPath().ReplacePrefix(proto.GetPath(), prim.GetPath())
                inst_child = stage.GetPrimAtPath(inst_path)
                load_visual_shapes_impl(parent_body_id, inst_child, body_xform, articulation_root_xform)
            return

        type_name = str(prim.GetTypeName()).lower()
        if type_name.endswith("joint"):
            return

        is_site = _usd_has_applied_api_schema(prim, "MjcSiteAPI")
        if is_site and not load_sites:
            return
        if not is_site and not load_visual_shapes:
            return

        shape_id = -1
        if path_name not in path_shape_map:
            cfg_visual = visual_cfg(is_site=is_site)
            if type_name == "cube":
                size = _usd_get_float(prim, "size", 2.0)
                side_lengths = scale * size
                shape_id = builder.add_shape(
                    body=parent_body_id,
                    shape_type=int(SapGeoType.BOX),
                    xform=shape_xform,
                    cfg=cfg_visual,
                    scale=(
                        float(side_lengths[0]) / 2.0,
                        float(side_lengths[1]) / 2.0,
                        float(side_lengths[2]) / 2.0,
                    ),
                    label=path_name,
                )
            elif type_name == "sphere":
                radius = _usd_get_float(prim, "radius", 1.0) * max(float(scale[0]), float(scale[1]), float(scale[2]))
                shape_id = builder.add_shape(
                    body=parent_body_id,
                    shape_type=int(SapGeoType.SPHERE),
                    xform=shape_xform,
                    cfg=cfg_visual,
                    scale=(float(radius), 0.0, 0.0),
                    label=path_name,
                )
            elif type_name == "plane":
                width = _usd_get_float(prim, "width", 0.0) * float(scale[0])
                length = _usd_get_float(prim, "length", 0.0) * float(scale[1])
                shape_id = builder.add_shape(
                    body=parent_body_id,
                    shape_type=int(SapGeoType.PLANE),
                    xform=shape_xform,
                    cfg=cfg_visual,
                    scale=(float(width), float(length), 0.0),
                    label=path_name,
                )
            elif type_name == "capsule":
                axis = _usd_get_gprim_axis(prim)
                corrected = wp.transform(shape_xform.p, shape_xform.q * _sap_quat_between_axes(2, axis))
                shape_id = builder.add_shape(
                    body=parent_body_id,
                    shape_type=int(SapGeoType.CAPSULE),
                    xform=corrected,
                    cfg=cfg_visual,
                    scale=(
                        float(_usd_get_float(prim, "radius", 0.5) * scale[0]),
                        float(_usd_get_float(prim, "height", 2.0) * 0.5 * scale[1]),
                        0.0,
                    ),
                    label=path_name,
                )
            elif type_name == "cylinder":
                axis = _usd_get_gprim_axis(prim)
                corrected = wp.transform(shape_xform.p, shape_xform.q * _sap_quat_between_axes(2, axis))
                shape_id = builder.add_shape(
                    body=parent_body_id,
                    shape_type=int(SapGeoType.CYLINDER),
                    xform=corrected,
                    cfg=cfg_visual,
                    scale=(
                        float(_usd_get_float(prim, "radius", 0.5) * scale[0]),
                        float(_usd_get_float(prim, "height", 2.0) * 0.5 * scale[1]),
                        0.0,
                    ),
                    label=path_name,
                )
            elif type_name == "cone":
                axis = _usd_get_gprim_axis(prim)
                corrected = wp.transform(shape_xform.p, shape_xform.q * _sap_quat_between_axes(2, axis))
                shape_id = builder.add_shape(
                    body=parent_body_id,
                    shape_type=int(SapGeoType.CONE),
                    xform=corrected,
                    cfg=cfg_visual,
                    scale=(
                        float(_usd_get_float(prim, "radius", 0.5) * scale[0]),
                        float(_usd_get_float(prim, "height", 2.0) * 0.5 * scale[1]),
                        0.0,
                    ),
                    label=path_name,
                )
            elif type_name == "mesh":
                mesh = get_mesh_with_visual_material(prim, path_name=path_name)
                shape_id = builder.add_shape(
                    body=parent_body_id,
                    shape_type=int(SapGeoType.MESH),
                    xform=shape_xform,
                    cfg=cfg_visual,
                    scale=(float(scale[0]), float(scale[1]), float(scale[2])),
                    src=mesh,
                    label=path_name,
                )

            if shape_id >= 0:
                path_shape_map[path_name] = shape_id
                path_shape_scale[path_name] = scale
                if not is_site:
                    bodies_with_visual_shapes.add(parent_body_id)

        for child in prim.GetChildren():
            load_visual_shapes_impl(parent_body_id, child, body_xform, articulation_root_xform)

    def add_body(
        prim: Any,
        xform_value: wp.transform,
        label: str,
        articulation_root_xform: wp.transform | None = None,
    ) -> int:
        body_id = builder.add_body(xform=xform_value, label=label)
        path_body_map[label] = body_id
        if load_sites or load_visual_shapes:
            for child in prim.GetChildren():
                load_visual_shapes_impl(body_id, child, body_xform=xform_value, articulation_root_xform=articulation_root_xform)
        return body_id

    def parse_body(
        rigid_body_desc: Any,
        prim: Any,
        incoming_xform: wp.transform | None = None,
        add_body_to_builder: bool = True,
        articulation_root_xform: wp.transform | None = None,
    ) -> int | dict[str, Any]:
        if not rigid_body_desc.rigidBodyEnabled and only_load_enabled_rigid_bodies:
            return -1
        origin = wp.transform(rigid_body_desc.position, _usd_value_to_warp(rigid_body_desc.rotation))
        if incoming_xform is not None:
            origin = incoming_xform * origin
        path = str(prim.GetPath())
        if add_body_to_builder:
            return add_body(prim, origin, path, articulation_root_xform=articulation_root_xform)
        result: dict[str, Any] = {
            "prim": prim,
            "xform_value": origin,
            "label": path,
            "articulation_root_xform": articulation_root_xform,
        }
        return result

    def resolve_joint_parent_child(
        joint_desc: Any,
        body_index_map: dict[str, int],
        *,
        get_transforms: bool = False,
    ):
        if get_transforms:
            parent_tf = wp.transform(joint_desc.localPose0Position, _usd_value_to_warp(joint_desc.localPose0Orientation))
            child_tf = wp.transform(joint_desc.localPose1Position, _usd_value_to_warp(joint_desc.localPose1Orientation))
        else:
            parent_tf = None
            child_tf = None
        parent_path = str(joint_desc.body0)
        child_path = str(joint_desc.body1)
        parent_id = body_index_map.get(parent_path, -1)
        child_id = body_index_map.get(child_path, -1)
        if child_id == -1:
            if parent_id == -1:
                raise SapSceneLoaderError(f"Unable to parse joint {joint_desc.primPath}: both bodies unresolved")
            parent_id, child_id = child_id, parent_id
            if get_transforms:
                parent_tf, child_tf = child_tf, parent_tf
        if get_transforms:
            return parent_id, child_id, parent_tf, child_tf
        return parent_id, child_id

    def parse_joint(joint_desc: Any, incoming_xform: wp.transform | None = None) -> int | None:
        if not joint_desc.jointEnabled and only_load_enabled_joints:
            return None
        parent_id, child_id, parent_tf, child_tf = resolve_joint_parent_child(
            joint_desc,
            path_body_map,
            get_transforms=True,
        )
        if incoming_xform is not None:
            parent_tf = incoming_xform * parent_tf
        key = joint_desc.type
        joint_path = str(joint_desc.primPath)
        joint_prim = stage.GetPrimAtPath(joint_desc.primPath)
        joint_armature = _usd_get_first_float(joint_prim, ("newton:armature", "newton:jointArmature"), default_joint_armature)
        joint_friction = _usd_get_first_float(joint_prim, ("newton:friction", "newton:jointFriction"), default_joint_friction)
        joint_velocity_limit = _usd_get_first_float(
            joint_prim,
            ("newton:velocityLimit", "newton:velocity_limit"),
            default_joint_velocity_limit,
        )
        common = {
            "parent": parent_id,
            "child": child_id,
            "parent_xform": parent_tf,
            "child_xform": child_tf,
            "label": joint_path,
            "collision_filter_parent": True,
        }

        if key == UsdPhysics.ObjectType.FixedJoint:
            joint_id = builder.add_joint(joint_type=_SAP_JOINT_FIXED, **common)
        elif key in (UsdPhysics.ObjectType.RevoluteJoint, UsdPhysics.ObjectType.PrismaticJoint):
            axis = _sap_axis_vector(_sap_axis_from_usd_physics_axis(joint_desc.axis, UsdPhysics))
            lower = float(joint_desc.limit.lower)
            upper = float(joint_desc.limit.upper)
            drive = getattr(joint_desc, "drive", None)
            has_drive = bool(getattr(drive, "enabled", False))
            target_pos = builder.default_joint_cfg.target_pos
            target_vel = builder.default_joint_cfg.target_vel
            target_ke = builder.default_joint_cfg.target_ke
            target_kd = builder.default_joint_cfg.target_kd
            effort_limit = builder.default_joint_cfg.effort_limit
            actuator_mode = 0
            if has_drive:
                target_pos = float(getattr(drive, "targetPosition", 0.0))
                target_vel = float(getattr(drive, "targetVelocity", 0.0))
                target_ke = float(getattr(drive, "stiffness", 0.0))
                target_kd = float(getattr(drive, "damping", 0.0))
                effort_limit = float(getattr(drive, "forceLimit", effort_limit))
                actuator_mode = _joint_target_mode_from_gains(
                    target_ke,
                    target_kd,
                    force_position_velocity_actuation,
                    has_drive=True,
                )
            if key == UsdPhysics.ObjectType.RevoluteJoint:
                limit_ke = _usd_get_first_float(
                    joint_prim,
                    ("newton:limitAngularKe", "newton:limit_angular_ke"),
                    default_joint_limit_ke * degrees_to_radian,
                )
                limit_kd = _usd_get_first_float(
                    joint_prim,
                    ("newton:limitAngularKd", "newton:limit_angular_kd"),
                    default_joint_limit_kd * degrees_to_radian,
                )
                angular_target_pos = target_pos
                angular_target_vel = target_vel
                angular_target_ke = target_ke
                angular_target_kd = target_kd
                if has_drive:
                    angular_target_pos *= degrees_to_radian
                    angular_target_vel *= degrees_to_radian
                    angular_target_ke /= degrees_to_radian / joint_drive_gains_scaling
                    angular_target_kd /= degrees_to_radian / joint_drive_gains_scaling
                if joint_velocity_limit is not None:
                    joint_velocity_limit = float(joint_velocity_limit) * degrees_to_radian
                axis_cfg = _SapJointDofConfig(
                    axis=axis,
                    limit_lower=lower * degrees_to_radian,
                    limit_upper=upper * degrees_to_radian,
                    limit_ke=float(limit_ke) / degrees_to_radian,
                    limit_kd=float(limit_kd) / degrees_to_radian,
                    target_pos=angular_target_pos,
                    target_vel=angular_target_vel,
                    target_ke=angular_target_ke,
                    target_kd=angular_target_kd,
                    armature=float(joint_armature),
                    effort_limit=effort_limit,
                    velocity_limit=float(joint_velocity_limit),
                    friction=float(joint_friction),
                    actuator_mode=actuator_mode,
                )
                joint_id = builder.add_joint(joint_type=_SAP_JOINT_REVOLUTE, angular_axes=[axis_cfg], **common)
                initial_position = _usd_get_first_float(
                    joint_prim,
                    (
                        "state:angular:physics:position",
                        "state:angular:position",
                        "newton:angularPosition",
                        "newton:angular_position",
                    ),
                    None,
                )
                initial_velocity = _usd_get_first_float(
                    joint_prim,
                    (
                        "state:angular:physics:velocity",
                        "state:angular:velocity",
                        "newton:angularVelocity",
                        "newton:angular_velocity",
                    ),
                    None,
                )
                if initial_position is not None:
                    builder.joint_q[builder.joint_q_start[joint_id]] = float(initial_position) * degrees_to_radian
                if initial_velocity is not None:
                    builder.joint_qd[builder.joint_qd_start[joint_id]] = float(initial_velocity)
            else:
                limit_ke = _usd_get_first_float(
                    joint_prim,
                    ("newton:limitLinearKe", "newton:limit_linear_ke"),
                    default_joint_limit_ke,
                )
                limit_kd = _usd_get_first_float(
                    joint_prim,
                    ("newton:limitLinearKd", "newton:limit_linear_kd"),
                    default_joint_limit_kd,
                )
                axis_cfg = _SapJointDofConfig(
                    axis=axis,
                    limit_lower=lower,
                    limit_upper=upper,
                    limit_ke=float(limit_ke),
                    limit_kd=float(limit_kd),
                    target_pos=target_pos,
                    target_vel=target_vel,
                    target_ke=target_ke,
                    target_kd=target_kd,
                    armature=float(joint_armature),
                    effort_limit=effort_limit,
                    velocity_limit=float(joint_velocity_limit),
                    friction=float(joint_friction),
                    actuator_mode=actuator_mode,
                )
                joint_id = builder.add_joint(joint_type=_SAP_JOINT_PRISMATIC, linear_axes=[axis_cfg], **common)
                initial_position = _usd_get_first_float(
                    joint_prim,
                    (
                        "state:linear:physics:position",
                        "state:linear:position",
                        "newton:linearPosition",
                        "newton:linear_position",
                    ),
                    None,
                )
                initial_velocity = _usd_get_first_float(
                    joint_prim,
                    (
                        "state:linear:physics:velocity",
                        "state:linear:velocity",
                        "newton:linearVelocity",
                        "newton:linear_velocity",
                    ),
                    None,
                )
                if initial_position is not None:
                    builder.joint_q[builder.joint_q_start[joint_id]] = float(initial_position)
                if initial_velocity is not None:
                    builder.joint_qd[builder.joint_qd_start[joint_id]] = float(initial_velocity)
        elif key == UsdPhysics.ObjectType.SphericalJoint:
            axes = [
                _SapJointDofConfig(axis=wp.vec3(1.0, 0.0, 0.0)),
                _SapJointDofConfig(axis=wp.vec3(0.0, 1.0, 0.0)),
                _SapJointDofConfig(axis=wp.vec3(0.0, 0.0, 1.0)),
            ]
            joint_id = builder.add_joint(joint_type=_SAP_JOINT_BALL, angular_axes=axes, **common)
        elif key == UsdPhysics.ObjectType.D6Joint:
            linear_axes: list[_SapJointDofConfig] = []
            angular_axes: list[_SapJointDofConfig] = []
            d6_axis_names: list[str] = []
            trans_axes = {
                UsdPhysics.JointDOF.TransX: ("transX", wp.vec3(1.0, 0.0, 0.0)),
                UsdPhysics.JointDOF.TransY: ("transY", wp.vec3(0.0, 1.0, 0.0)),
                UsdPhysics.JointDOF.TransZ: ("transZ", wp.vec3(0.0, 0.0, 1.0)),
            }
            rot_axes = {
                UsdPhysics.JointDOF.RotX: ("rotX", wp.vec3(1.0, 0.0, 0.0)),
                UsdPhysics.JointDOF.RotY: ("rotY", wp.vec3(0.0, 1.0, 0.0)),
                UsdPhysics.JointDOF.RotZ: ("rotZ", wp.vec3(0.0, 0.0, 1.0)),
            }
            for limit in joint_desc.jointLimits:
                if limit.second.enabled:
                    limit_lower = float(limit.second.lower)
                    limit_upper = float(limit.second.upper)
                else:
                    limit_lower = -1.0e10
                    limit_upper = 1.0e10
                if limit_lower >= limit_upper:
                    continue
                dof = limit.first
                if dof in trans_axes:
                    axis_name, axis = trans_axes[dof]
                    linear_axes.append(_SapJointDofConfig(axis=axis, limit_lower=limit_lower, limit_upper=limit_upper))
                    d6_axis_names.append(axis_name)
                elif dof in rot_axes:
                    axis_name, axis = rot_axes[dof]
                    angular_axes.append(
                        _SapJointDofConfig(
                            axis=axis,
                            limit_lower=limit_lower * degrees_to_radian,
                            limit_upper=limit_upper * degrees_to_radian,
                        )
                    )
                    d6_axis_names.append(axis_name)
            joint_id = builder.add_joint(joint_type=_SAP_JOINT_D6, linear_axes=linear_axes, angular_axes=angular_axes, **common)
            q_start = builder.joint_q_start[joint_id]
            qd_start = builder.joint_qd_start[joint_id]
            for offset, axis_name in enumerate(d6_axis_names):
                pos = _usd_get_first_float(
                    joint_prim,
                    (
                        f"state:{axis_name}:physics:position",
                        f"state:{axis_name}:position",
                        f"newton:{axis_name}Position",
                        f"newton:{axis_name}_position",
                    ),
                    None,
                )
                vel = _usd_get_first_float(
                    joint_prim,
                    (
                        f"state:{axis_name}:physics:velocity",
                        f"state:{axis_name}:velocity",
                        f"newton:{axis_name}Velocity",
                        f"newton:{axis_name}_velocity",
                    ),
                    None,
                )
                if pos is not None and q_start + offset < len(builder.joint_q):
                    builder.joint_q[q_start + offset] = float(pos) * degrees_to_radian if axis_name.startswith("rot") else float(pos)
                if vel is not None and qd_start + offset < len(builder.joint_qd):
                    builder.joint_qd[qd_start + offset] = float(vel)
        elif key == UsdPhysics.ObjectType.DistanceJoint:
            axes = [
                _SapJointDofConfig(axis=wp.vec3(1.0, 0.0, 0.0)),
                _SapJointDofConfig(axis=wp.vec3(0.0, 1.0, 0.0)),
                _SapJointDofConfig(axis=wp.vec3(0.0, 0.0, 1.0)),
            ]
            joint_id = builder.add_joint(joint_type=_SAP_JOINT_DISTANCE, linear_axes=axes, angular_axes=axes, **common)
        else:
            joint_id = builder.add_joint(joint_type=_SAP_JOINT_FIXED, **common)

        path_joint_map[joint_path] = joint_id
        return joint_id

    material_specs: dict[str, _SapUsdPhysicsMaterial] = {
        "": _SapUsdPhysicsMaterial(
            static_friction=builder.default_shape_cfg.mu,
            dynamic_friction=builder.default_shape_cfg.mu,
            torsional_friction=builder.default_shape_cfg.mu_torsional,
            rolling_friction=builder.default_shape_cfg.mu_rolling,
            restitution=builder.default_shape_cfg.restitution,
            density=builder.default_shape_cfg.density,
        )
    }
    for sdf_path, desc in data_for_key(UsdPhysics.ObjectType.RigidBodyMaterial) or []:
        if warn_invalid_desc(sdf_path, desc):
            continue
        prim = stage.GetPrimAtPath(sdf_path)
        material_specs[str(sdf_path)] = _SapUsdPhysicsMaterial(
            static_friction=float(desc.staticFriction),
            dynamic_friction=float(desc.dynamicFriction),
            torsional_friction=float(
                _usd_get_float(prim, "newton:mu_torsional", builder.default_shape_cfg.mu_torsional)
            ),
            rolling_friction=float(_usd_get_float(prim, "newton:mu_rolling", builder.default_shape_cfg.mu_rolling)),
            restitution=float(desc.restitution),
            density=float(desc.density if desc.density > 0.0 else builder.default_shape_cfg.density),
        )

    body_specs: dict[str, Any] = {}
    ignored_body_paths: set[str] = set()
    for prim_path, rigid_body_desc in data_for_key(UsdPhysics.ObjectType.RigidBody) or []:
        if warn_invalid_desc(prim_path, rigid_body_desc):
            continue
        body_path = str(prim_path)
        if ignored(body_path):
            ignored_body_paths.add(body_path)
            continue
        body_specs[body_path] = rigid_body_desc

    joint_descriptions: dict[str, Any] = {}
    joint_object_types = {
        UsdPhysics.ObjectType.FixedJoint,
        UsdPhysics.ObjectType.RevoluteJoint,
        UsdPhysics.ObjectType.PrismaticJoint,
        UsdPhysics.ObjectType.SphericalJoint,
        UsdPhysics.ObjectType.D6Joint,
        UsdPhysics.ObjectType.DistanceJoint,
    }
    for key, value in ret_dict.items():
        if key in joint_object_types:
            paths, joint_specs = value
            for path, joint_spec in zip(paths, joint_specs, strict=False):
                joint_descriptions[str(path)] = joint_spec

    processed_joints: set[str] = set()
    articulation_bodies: dict[int, list[int]] = {}
    articulation_has_self_collision: dict[int, bool] = {}
    articulation_id = 0

    for path, desc in data_for_key(UsdPhysics.ObjectType.Articulation) or []:
        if warn_invalid_desc(path, desc):
            continue
        articulation_path = str(path)
        if ignored(articulation_path):
            continue
        articulation_prim = stage.GetPrimAtPath(path)
        body_data: dict[int, dict[str, Any]] = {}
        body_ids: dict[str, int] = {}
        body_labels: list[str] = []
        current_body_id = 0
        art_bodies: list[int] = []

        for body_path_obj in desc.articulatedBodies:
            if body_path_obj == Sdf.Path.emptyPath:
                continue
            key = str(body_path_obj)
            if key in ignored_body_paths:
                continue
            usd_prim = stage.GetPrimAtPath(body_path_obj)
            if key in body_specs:
                body_desc = body_specs[key]
                desc_xform = wp.transform(body_desc.position, _usd_value_to_warp(body_desc.rotation))
                body_world = _usd_get_transform(usd_prim, local=False, xform_cache=xform_cache)
                desired_world = incoming_world_xform * body_world
                body_incoming_xform = desired_world * wp.transform_inverse(desc_xform)
                if bodies_follow_joint_ordering:
                    parsed = parse_body(
                        body_desc,
                        usd_prim,
                        incoming_xform=body_incoming_xform,
                        add_body_to_builder=False,
                    )
                    if isinstance(parsed, dict):
                        body_data[current_body_id] = parsed
                else:
                    body_id = parse_body(body_desc, usd_prim, incoming_xform=body_incoming_xform)
                    if isinstance(body_id, int) and body_id >= 0:
                        art_bodies.append(body_id)
                del body_specs[key]
            body_ids[key] = current_body_id
            body_labels.append(key)
            current_body_id += 1

        if not body_ids:
            continue

        joint_names: list[str] = []
        joint_edges: list[tuple[int, int]] = []
        joint_excluded: set[str] = set()
        for joint_path_obj in desc.articulatedJoints:
            joint_path = str(joint_path_obj)
            joint_desc = joint_descriptions[joint_path]
            if ignored(joint_path):
                continue
            if str(joint_desc.body0) in ignored_body_paths or str(joint_desc.body1) in ignored_body_paths:
                continue
            parent_id, child_id = resolve_joint_parent_child(joint_desc, body_ids)
            if getattr(joint_desc, "excludeFromArticulation", False):
                joint_excluded.add(joint_path)
            else:
                joint_edges.append((parent_id, child_id))
                joint_names.append(joint_path)

        if joint_edges:
            if joint_ordering is not None:
                sorted_joints, reversed_joints = _topological_sort_undirected(
                    joint_edges,
                    use_dfs=joint_ordering == "dfs",
                    ensure_single_root=True,
                )
                if reversed_joints:
                    names = ", ".join(joint_names[index] for index in reversed_joints)
                    raise SapSceneLoaderError(f"Reversed USD joints are not supported: {names}")
            else:
                sorted_joints = list(range(len(joint_names)))

            articulation_joint_indices: list[int] = []
            if bodies_follow_joint_ordering:
                inserted_bodies: set[int] = set()
                for joint_index in sorted_joints:
                    parent, child = joint_edges[joint_index]
                    if parent >= 0 and parent not in inserted_bodies:
                        data = body_data[parent]
                        b = add_body(
                            data["prim"],
                            data["xform_value"],
                            data["label"],
                            articulation_root_xform=data.get("articulation_root_xform"),
                        )
                        inserted_bodies.add(parent)
                        art_bodies.append(b)
                    if child >= 0 and child not in inserted_bodies:
                        data = body_data[child]
                        b = add_body(
                            data["prim"],
                            data["xform_value"],
                            data["label"],
                            articulation_root_xform=data.get("articulation_root_xform"),
                        )
                        inserted_bodies.add(child)
                        art_bodies.append(b)

            first_joint_parent = joint_edges[sorted_joints[0]][0]
            if first_joint_parent != -1:
                if bodies_follow_joint_ordering:
                    first_label = body_data[first_joint_parent]["label"]
                    child_body_id = path_body_map[first_label]
                else:
                    child_body_id = art_bodies[first_joint_parent]
                if floating is False:
                    base_joint = builder.add_joint(
                        parent=-1,
                        child=child_body_id,
                        joint_type=_SAP_JOINT_FIXED,
                        parent_xform=builder.body_q[child_body_id],
                        collision_filter_parent=True,
                    )
                else:
                    base_joint = builder.add_joint(
                        parent=-1,
                        child=child_body_id,
                        joint_type=_SAP_JOINT_FREE,
                        linear_axes=_free_joint_linear_axes(),
                        angular_axes=_free_joint_angular_axes(),
                        collision_filter_parent=True,
                    )
                articulation_joint_indices.append(base_joint)

            for joint_offset, joint_index in enumerate(sorted_joints):
                incoming_for_joint = None
                if joint_offset == 0 and first_joint_parent == -1:
                    root_joint_desc = joint_descriptions[joint_names[joint_index]]
                    b0 = str(root_joint_desc.body0)
                    b1 = str(root_joint_desc.body1)
                    if b0 not in body_ids:
                        world_body_path = b0
                    elif b1 not in body_ids:
                        world_body_path = b1
                    else:
                        world_body_path = b0
                    world_body_prim = stage.GetPrimAtPath(world_body_path) if world_body_path else None
                    if world_body_prim is not None and world_body_prim.IsValid():
                        world_body_xform = _usd_get_transform(world_body_prim, local=False, xform_cache=xform_cache)
                    else:
                        _, child_local_id, parent_tf, child_tf = resolve_joint_parent_child(
                            root_joint_desc,
                            body_ids,
                            get_transforms=True,
                        )
                        identity_tf = wp.transform_identity()
                        parent_pos = np.array(parent_tf.p, dtype=float)
                        parent_quat = np.array(parent_tf.q, dtype=float)
                        identity_pos = np.array(identity_tf.p, dtype=float)
                        identity_quat = np.array(identity_tf.q, dtype=float)
                        parent_pos_is_identity = np.allclose(parent_pos, identity_pos, atol=1.0e-6)
                        parent_rot_is_identity = abs(np.dot(parent_quat, identity_quat)) > 1.0 - 1.0e-6
                        if parent_pos_is_identity and parent_rot_is_identity and 0 <= child_local_id < len(body_labels):
                            child_prim = stage.GetPrimAtPath(body_labels[child_local_id])
                        else:
                            child_prim = None
                        if child_prim is not None and child_prim.IsValid():
                            child_world_xform = _usd_get_transform(child_prim, local=False, xform_cache=xform_cache)
                            world_body_xform = child_world_xform * child_tf * wp.transform_inverse(parent_tf)
                        else:
                            world_body_xform = wp.transform_identity()
                    incoming_for_joint = incoming_world_xform * world_body_xform

                joint = parse_joint(joint_descriptions[joint_names[joint_index]], incoming_xform=incoming_for_joint)
                if joint is not None:
                    articulation_joint_indices.append(joint)
                    processed_joints.add(joint_names[joint_index])

            for joint_path in joint_excluded:
                joint = parse_joint(joint_descriptions[joint_path])
                if joint is not None:
                    processed_joints.add(joint_path)
            if articulation_joint_indices:
                builder.add_articulation(articulation_joint_indices, label=articulation_path)
        else:
            for i in body_ids.values():
                if bodies_follow_joint_ordering:
                    data = body_data[i]
                    child_body_id = add_body(
                        data["prim"],
                        data["xform_value"],
                        data["label"],
                        articulation_root_xform=data.get("articulation_root_xform"),
                    )
                    art_bodies.append(child_body_id)
                else:
                    child_body_id = art_bodies[i]
                if floating is False:
                    base_joint = builder.add_joint(
                        parent=-1,
                        child=child_body_id,
                        joint_type=_SAP_JOINT_FIXED,
                        parent_xform=builder.body_q[child_body_id],
                        collision_filter_parent=True,
                    )
                else:
                    base_joint = builder.add_joint(
                        parent=-1,
                        child=child_body_id,
                        joint_type=_SAP_JOINT_FREE,
                        linear_axes=_free_joint_linear_axes(),
                        angular_axes=_free_joint_angular_axes(),
                        collision_filter_parent=True,
                    )
                builder.add_articulation([base_joint], label=builder.body_labels[child_body_id])

        articulation_bodies[articulation_id] = art_bodies
        authored_self_collision = _usd_get_attribute(articulation_prim, "newton:selfCollisionEnabled", None)
        articulation_has_self_collision[articulation_id] = (
            bool(enable_self_collisions) if authored_self_collision is None else bool(authored_self_collision)
        )
        articulation_id += 1

    no_articulations = UsdPhysics.ObjectType.Articulation not in ret_dict
    has_joints = any(
        (
            not (only_load_enabled_joints and not joint_desc.jointEnabled)
            and not ignored(joint_path)
            and str(joint_desc.body0) not in ignored_body_paths
            and str(joint_desc.body1) not in ignored_body_paths
        )
        for joint_path, joint_desc in joint_descriptions.items()
    )

    for path, rigid_body_desc in list(body_specs.items()):
        parse_body(
            rigid_body_desc,
            stage.GetPrimAtPath(path),
            incoming_xform=incoming_world_xform,
            add_body_to_builder=True,
        )

    orphan_joints: list[str] = []
    for joint_path, joint_desc in joint_descriptions.items():
        if joint_path in processed_joints:
            continue
        if only_load_enabled_joints and not joint_desc.jointEnabled:
            continue
        if ignored(joint_path):
            continue
        if str(joint_desc.body0) in ignored_body_paths or str(joint_desc.body1) in ignored_body_paths:
            continue
        body0_path = str(joint_desc.body0)
        body1_path = str(joint_desc.body1)
        is_body_to_world = body0_path in ("", "/") or body1_path in ("", "/")
        is_fixed_joint = joint_desc.type == UsdPhysics.ObjectType.FixedJoint
        free_joints_auto_inserted = not (no_articulations and has_joints)
        if is_body_to_world and free_joints_auto_inserted and not is_fixed_joint:
            continue
        try:
            if parse_joint(joint_desc) is not None and not (is_body_to_world and is_fixed_joint):
                orphan_joints.append(joint_path)
        except SapSceneLoaderError:
            continue
    _ = orphan_joints

    path_collision_filters: set[tuple[str, str]] = set()
    no_collision_shapes: set[int] = set()
    collision_group_ids: dict[str, int] = {}
    remeshing_queue: dict[str, list[int]] = {}
    approximation_to_remeshing_method = {
        "convexdecomposition": "coacd",
        "convexhull": "convex_hull",
        "boundingsphere": "bounding_sphere",
        "boundingcube": "bounding_box",
        "meshsimplification": "quadratic",
    }

    shape_object_types = {
        UsdPhysics.ObjectType.CubeShape,
        UsdPhysics.ObjectType.SphereShape,
        UsdPhysics.ObjectType.CapsuleShape,
        UsdPhysics.ObjectType.CylinderShape,
        UsdPhysics.ObjectType.ConeShape,
        UsdPhysics.ObjectType.MeshShape,
        UsdPhysics.ObjectType.PlaneShape,
    }
    for key, value in ret_dict.items():
        if key not in shape_object_types:
            continue
        paths, shape_specs = value
        for xpath, shape_spec in zip(paths, shape_specs, strict=False):
            if warn_invalid_desc(xpath, shape_spec):
                continue
            path = str(xpath)
            if ignored(path):
                continue
            prim = stage.GetPrimAtPath(xpath)
            if path in path_shape_map:
                continue
            body_path = str(shape_spec.rigidBody)
            body_id = path_body_map.get(body_path, -1)
            scale = _usd_get_scale(prim, local=False)

            collision_group = builder.default_shape_cfg.collision_group
            if len(shape_spec.collisionGroups) > 0:
                cgroup_name = str(shape_spec.collisionGroups[0])
                if cgroup_name not in collision_group_ids:
                    collision_group_ids[cgroup_name] = len(collision_group_ids) + 1
                collision_group = collision_group_ids[cgroup_name]

            material = material_specs[""]
            has_shape_material = len(shape_spec.materials) >= 1
            if has_shape_material:
                material = material_specs.get(str(shape_spec.materials[0]), material)
            shape_density = material.density if has_shape_material else builder.default_shape_cfg.density
            _ = shape_density

            local_xform = wp.transform(shape_spec.localPos, _usd_value_to_warp(shape_spec.localRot))
            shape_xform = incoming_world_xform * local_xform if body_id == -1 else local_xform

            margin_val = _usd_get_first_float(
                prim,
                ("newton:contactMargin", "newton:margin"),
                builder.default_shape_cfg.margin,
            )
            gap_val = _usd_get_first_float(prim, ("newton:contactGap", "newton:gap"), None)
            if gap_val is None:
                gap_val = builder.default_shape_cfg.gap
            shape_ke = _usd_get_first_float(
                prim,
                ("newton:contactKe", "newton:ke"),
                builder.default_shape_cfg.ke,
            )

            has_body_visual_shapes = load_visual_shapes and body_id in bodies_with_visual_shapes
            collider_has_visual_material = (
                key == UsdPhysics.ObjectType.MeshShape and has_visual_material_properties(get_material_props_cached(prim))
            )
            hide_collider_for_body = hide_collision_shapes and has_body_visual_shapes and not collider_has_visual_material
            show_collider_by_policy = _should_show_collider(force_show_colliders, has_visual_shapes=has_body_visual_shapes)
            collider_is_visible = (show_collider_by_policy or collider_has_visual_material) and not hide_collider_for_body

            shape_cfg = builder.default_shape_cfg.copy()
            shape_cfg.ke = float(shape_ke)
            shape_cfg.margin = float(margin_val)
            shape_cfg.gap = None if gap_val is None else float(gap_val)
            shape_cfg.mu = float(material.dynamic_friction)
            shape_cfg.restitution = float(material.restitution)
            shape_cfg.mu_torsional = float(material.torsional_friction)
            shape_cfg.mu_rolling = float(material.rolling_friction)
            shape_cfg.collision_group = int(collision_group)
            shape_cfg.is_visible = bool(collider_is_visible)

            shape_id = -1
            if key == UsdPhysics.ObjectType.CubeShape:
                hx, hy, hz = shape_spec.halfExtents
                shape_id = builder.add_shape(
                    body=body_id,
                    shape_type=int(SapGeoType.BOX),
                    xform=shape_xform,
                    cfg=shape_cfg,
                    scale=(float(hx), float(hy), float(hz)),
                    label=path,
                )
            elif key == UsdPhysics.ObjectType.SphereShape:
                shape_id = builder.add_shape(
                    body=body_id,
                    shape_type=int(SapGeoType.SPHERE),
                    xform=shape_xform,
                    cfg=shape_cfg,
                    scale=(float(shape_spec.radius), 0.0, 0.0),
                    label=path,
                )
            elif key == UsdPhysics.ObjectType.CapsuleShape:
                axis = _sap_axis_from_usd_physics_axis(shape_spec.axis, UsdPhysics)
                corrected = wp.transform(shape_xform.p, shape_xform.q * _sap_quat_between_axes(2, axis))
                shape_id = builder.add_shape(
                    body=body_id,
                    shape_type=int(SapGeoType.CAPSULE),
                    xform=corrected,
                    cfg=shape_cfg,
                    scale=(float(shape_spec.radius), float(shape_spec.halfHeight), 0.0),
                    label=path,
                )
            elif key == UsdPhysics.ObjectType.CylinderShape:
                axis = _sap_axis_from_usd_physics_axis(shape_spec.axis, UsdPhysics)
                corrected = wp.transform(shape_xform.p, shape_xform.q * _sap_quat_between_axes(2, axis))
                shape_id = builder.add_shape(
                    body=body_id,
                    shape_type=int(SapGeoType.CYLINDER),
                    xform=corrected,
                    cfg=shape_cfg,
                    scale=(float(shape_spec.radius), float(shape_spec.halfHeight), 0.0),
                    label=path,
                )
            elif key == UsdPhysics.ObjectType.ConeShape:
                axis = _sap_axis_from_usd_physics_axis(shape_spec.axis, UsdPhysics)
                corrected = wp.transform(shape_xform.p, shape_xform.q * _sap_quat_between_axes(2, axis))
                shape_id = builder.add_shape(
                    body=body_id,
                    shape_type=int(SapGeoType.CONE),
                    xform=corrected,
                    cfg=shape_cfg,
                    scale=(float(shape_spec.radius), float(shape_spec.halfHeight), 0.0),
                    label=path,
                )
            elif key == UsdPhysics.ObjectType.MeshShape:
                if collider_is_visible:
                    mesh = get_mesh_with_visual_material(prim, path_name=path)
                else:
                    mesh = get_mesh_cached(prim)
                mesh.maxhullvert = int(_usd_get_float(prim, "newton:max_hull_vertices", mesh_maxhullvert))
                mesh_scale = tuple(float(v) for v in shape_spec.meshScale)
                shape_id = builder.add_shape(
                    body=body_id,
                    shape_type=int(SapGeoType.MESH),
                    xform=shape_xform,
                    cfg=shape_cfg,
                    scale=mesh_scale,
                    src=mesh,
                    label=path,
                )
                if not skip_mesh_approximation:
                    approximation = _usd_get_attribute(prim, "physics:approximation", None)
                    if approximation is not None:
                        remeshing_method = approximation_to_remeshing_method.get(str(approximation).lower())
                        if remeshing_method is not None:
                            remeshing_queue.setdefault(remeshing_method, []).append(shape_id)
            elif key == UsdPhysics.ObjectType.PlaneShape:
                if shape_spec.axis != UsdPhysics.Axis.Z:
                    axis = _sap_axis_from_usd_physics_axis(shape_spec.axis, UsdPhysics)
                    shape_xform = wp.transform(shape_xform.p, shape_xform.q * _sap_quat_between_axes(2, axis))
                shape_id = builder.add_shape(
                    body=body_id,
                    shape_type=int(SapGeoType.PLANE),
                    xform=shape_xform,
                    cfg=shape_cfg,
                    scale=(0.0, 0.0, 0.0),
                    label=path,
                )

            if shape_id < 0:
                continue

            path_shape_map[path] = shape_id
            path_shape_scale[path] = scale

            if prim.HasRelationship("physics:filteredPairs"):
                for other_path in prim.GetRelationship("physics:filteredPairs").GetTargets():
                    path_collision_filters.add((path, str(other_path)))

            if not is_enabled_collider(prim):
                no_collision_shapes.add(shape_id)
                builder.shape_flags[shape_id] &= ~int(SapShapeFlags.COLLIDE_SHAPES)

    for remeshing_method, shape_ids in remeshing_queue.items():
        if remeshing_method == "bounding_box":
            _approximate_meshes_with_bounding_boxes(builder, shape_ids)
        else:
            raise SapSceneLoaderError(f"USD mesh approximation method={remeshing_method}")

    for body_path, body_id in path_body_map.items():
        prim = stage.GetPrimAtPath(body_path)
        if prim is None or not prim.IsValid() or not prim.HasAPI(UsdPhysics.MassAPI):
            continue
        mass_api = UsdPhysics.MassAPI(prim)
        mass_attr = mass_api.GetMassAttr()
        inertia_attr = mass_api.GetDiagonalInertiaAttr()
        com_attr = mass_api.GetCenterOfMassAttr()
        principal_axes_attr = mass_api.GetPrincipalAxesAttr()
        has_authored_mass = mass_attr.HasAuthoredValue()
        has_authored_inertia = inertia_attr.HasAuthoredValue()
        has_authored_com = com_attr.HasAuthoredValue()

        if has_authored_inertia:
            i_diag_np = np.array(inertia_attr.Get(), dtype=np.float32)
            if i_diag_np.size == 3 and np.all(i_diag_np >= 0.0) and np.linalg.norm(i_diag_np) > 0.0:
                if principal_axes_attr.HasAuthoredValue():
                    i_rot = _usd_value_to_warp(principal_axes_attr.Get(), wp.quat)
                else:
                    i_rot = wp.quat_identity()
                rot = np.array(wp.quat_to_matrix(i_rot), dtype=np.float32).reshape(3, 3)
                inertia_np = rot @ np.diag(i_diag_np) @ rot.T
                builder.body_inertia[body_id] = wp.mat33(inertia_np)
                builder.body_inv_inertia[body_id] = wp.inverse(builder.body_inertia[body_id])

        if has_authored_mass:
            mass = float(mass_attr.Get())
            shape_accumulated_mass = builder.body_mass[body_id]
            if not has_authored_inertia and shape_accumulated_mass > 0.0 and mass > 0.0:
                scale_mass = mass / shape_accumulated_mass
                inertia_np = np.array(builder.body_inertia[body_id], dtype=np.float32).reshape(3, 3) * scale_mass
                builder.body_inertia[body_id] = wp.mat33(inertia_np)
                if np.any(inertia_np):
                    builder.body_inv_inertia[body_id] = wp.inverse(builder.body_inertia[body_id])
                else:
                    builder.body_inv_inertia[body_id] = wp.mat33(0.0)
            builder.body_mass[body_id] = mass
            builder.body_inv_mass[body_id] = 1.0 / mass if mass > 0.0 else 0.0

        if has_authored_com:
            builder.body_com[body_id] = wp.vec3(*com_attr.Get())

        if builder.body_mass[body_id] > 0.0 and not np.any(np.array(builder.body_inertia[body_id])):
            density = builder.default_shape_cfg.density
            volume = builder.body_mass[body_id] / density if density > 0.0 else 0.0
            if volume > 0.0:
                radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
                _, _, default_inertia = _sap_compute_inertia_sphere(density, radius)
                com_np = np.array(builder.body_com[body_id], dtype=np.float32)
                if np.linalg.norm(com_np) > 1.0e-6:
                    default_inertia = wp.mat33(
                        np.array(default_inertia, dtype=np.float32).reshape(3, 3)
                        + builder.body_mass[body_id] * float(np.sum(com_np * com_np)) * np.eye(3, dtype=np.float32)
                    )
                builder.body_inertia[body_id] = default_inertia
                builder.body_inv_inertia[body_id] = wp.inverse(default_inertia)

        if has_authored_mass or has_authored_inertia or has_authored_com:
            builder.body_lock_inertia[body_id] = True

    for path1, path2 in path_collision_filters:
        if path1 in path_shape_map and path2 in path_shape_map:
            builder.add_shape_collision_filter_pair(path_shape_map[path1], path_shape_map[path2])

    for shape_id in no_collision_shapes:
        for other_shape_id in range(builder.shape_count):
            if other_shape_id != shape_id:
                builder.add_shape_collision_filter_pair(shape_id, other_shape_id)

    for art_id, bodies in articulation_bodies.items():
        if not articulation_has_self_collision[art_id]:
            for body1, body2 in itertools.combinations(bodies, 2):
                for shape1 in builder.body_shapes.get(body1, []):
                    for shape2 in builder.body_shapes.get(body2, []):
                        builder.add_shape_collision_filter_pair(shape1, shape2)


def _sap_axis_from_token(value: str) -> int:
    token = value.strip().upper()
    if token in {"X", "AXIS.X", "0"}:
        return 0
    if token in {"Y", "AXIS.Y", "1"}:
        return 1
    return 2


def _sap_axis_from_usd_physics_axis(value: Any, usd_physics: Any) -> int:
    if value == usd_physics.Axis.X or int(value) == int(usd_physics.Axis.X):
        return 0
    if value == usd_physics.Axis.Y or int(value) == int(usd_physics.Axis.Y):
        return 1
    return 2


def _sap_axis_vector(axis: int) -> wp.vec3:
    if axis == 0:
        return wp.vec3(1.0, 0.0, 0.0)
    if axis == 1:
        return wp.vec3(0.0, 1.0, 0.0)
    return wp.vec3(0.0, 0.0, 1.0)


def _sap_quat_between_axes(axis_from: int, axis_to: int) -> wp.quat:
    if axis_from == axis_to:
        return wp.quat_identity()
    a = _sap_axis_vector(axis_from)
    b = _sap_axis_vector(axis_to)
    cross = wp.cross(a, b)
    dot = max(-1.0, min(1.0, float(wp.dot(a, b))))
    length = float(wp.length(cross))
    if length < 1.0e-8:
        fallback = wp.vec3(1.0, 0.0, 0.0) if axis_from != 0 else wp.vec3(0.0, 1.0, 0.0)
        cross = wp.normalize(wp.cross(a, fallback))
        return wp.quat_from_axis_angle(cross, math.pi)
    return wp.quat_from_axis_angle(cross / length, math.acos(dot))


def _usd_get_attribute(prim: Any, name: str, default: Any = None) -> Any:
    attr = prim.GetAttribute(name)
    if not attr or not attr.HasAuthoredValue():
        return default
    return attr.Get()


def _usd_get_float(prim: Any, name: str, default: float | None = None) -> float | None:
    attr = prim.GetAttribute(name)
    if not attr or not attr.HasAuthoredValue():
        return default
    value = attr.Get()
    if np.isfinite(value):
        return float(value)
    return default


def _usd_get_first_float(prim: Any, names: Sequence[str], default: float | None = None) -> float | None:
    for name in names:
        value = _usd_get_float(prim, name, None)
        if value is not None:
            return value
    return default


def _usd_get_float_with_fallback(prims: Sequence[Any | None], name: str, default: float = 0.0) -> float:
    for prim in prims:
        if not prim:
            continue
        value = _usd_get_float(prim, name, None)
        if value is not None:
            return float(value)
    return float(default)


def _usd_has_applied_api_schema(prim: Any, schema_name: str) -> bool:
    if prim.HasAPI(schema_name):
        return True
    schemas_listop = prim.GetMetadata("apiSchemas")
    if schemas_listop:
        all_schemas = (
            list(schemas_listop.prependedItems)
            + list(schemas_listop.appendedItems)
            + list(schemas_listop.explicitItems)
        )
        return schema_name in all_schemas
    return False


def _usd_value_to_warp(value: Any, warp_dtype: Any | None = None) -> Any:
    if warp_dtype is wp.quat or (hasattr(value, "real") and hasattr(value, "imaginary")):
        return wp.normalize(wp.quat(*value.imaginary, value.real))
    if warp_dtype is not None:
        if hasattr(value, "__len__"):
            return warp_dtype(*value)
        return warp_dtype(value)
    if hasattr(value, "__len__"):
        if len(value) == 2:
            return wp.vec2(*value)
        if len(value) == 3:
            return wp.vec3(*value)
        if len(value) == 4:
            return wp.vec4(*value)
    return value


def _usd_get_xform_matrix(prim: Any, local: bool = True, xform_cache: Any | None = None) -> np.ndarray:
    from pxr import Usd, UsdGeom

    xform = UsdGeom.Xformable(prim)
    if local:
        matrix = xform.GetLocalTransformation()
        if isinstance(matrix, tuple):
            matrix = matrix[0]
    else:
        if xform_cache is None:
            matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        else:
            matrix = xform_cache.GetLocalToWorldTransform(prim)
    return np.array(matrix, dtype=np.float32)


def _usd_get_transform_matrix(prim: Any, local: bool = True, xform_cache: Any | None = None) -> wp.mat44:
    matrix = _usd_get_xform_matrix(prim, local=local, xform_cache=xform_cache)
    return wp.mat44(matrix.T)


def _usd_get_transform(prim: Any, local: bool = True, xform_cache: Any | None = None) -> wp.transform:
    matrix = _usd_get_xform_matrix(prim, local=local, xform_cache=xform_cache)
    xform_pos, xform_rot, _scale = wp.transform_decompose(wp.mat44(matrix.T))
    return wp.transform(xform_pos, xform_rot)


def _usd_get_scale(prim: Any, local: bool = True, xform_cache: Any | None = None) -> wp.vec3:
    _pos, _rot, scale = wp.transform_decompose(_usd_get_transform_matrix(prim, local=local, xform_cache=xform_cache))
    return wp.vec3(*scale)


def _usd_get_gprim_axis(prim: Any, name: str = "axis", default: str = "Z") -> int:
    return _sap_axis_from_token(str(_usd_get_attribute(prim, name, default)))


def _usd_fan_triangulate_faces(counts: Sequence[int], indices: Sequence[int]) -> np.ndarray:
    counts_i32 = np.asarray(counts, dtype=np.int32)
    indices_i32 = np.asarray(indices, dtype=np.int32)
    num_tris = int(np.sum(counts_i32 - 2))
    if num_tris == 0:
        return np.zeros((0, 3), dtype=np.int32)
    tri_face_ids = np.repeat(np.arange(len(counts_i32), dtype=np.int32), counts_i32 - 2)
    tri_local_ids = np.concatenate([np.arange(n - 2, dtype=np.int32) for n in counts_i32])
    face_bases = np.concatenate([[0], np.cumsum(counts_i32[:-1], dtype=np.int32)])
    out = np.empty((num_tris, 3), dtype=np.int32)
    out[:, 0] = indices_i32[face_bases[tri_face_ids]]
    out[:, 1] = indices_i32[face_bases[tri_face_ids] + tri_local_ids + 1]
    out[:, 2] = indices_i32[face_bases[tri_face_ids] + tri_local_ids + 2]
    return out


def _usd_triangulate_face_varying_indices(counts: Sequence[int], flip_winding: bool) -> np.ndarray:
    counts_i32 = np.asarray(counts, dtype=np.int32)
    num_tris = int(np.sum(counts_i32 - 2))
    if num_tris <= 0:
        return np.zeros((0,), dtype=np.int32)
    tri_face_ids = np.repeat(np.arange(len(counts_i32), dtype=np.int32), counts_i32 - 2)
    tri_local_ids = np.concatenate([np.arange(n - 2, dtype=np.int32) for n in counts_i32])
    face_bases = np.concatenate([[0], np.cumsum(counts_i32[:-1], dtype=np.int32)])
    corner_faces = np.empty((num_tris, 3), dtype=np.int32)
    corner_faces[:, 0] = face_bases[tri_face_ids]
    corner_faces[:, 1] = face_bases[tri_face_ids] + tri_local_ids + 1
    corner_faces[:, 2] = face_bases[tri_face_ids] + tri_local_ids + 2
    if flip_winding:
        corner_faces = corner_faces[:, ::-1]
    return corner_faces.reshape(-1)


def _usd_expand_indexed_primvar(values: np.ndarray, indices: Any, primvar_name: str, prim_path: str) -> np.ndarray:
    if indices is None or len(indices) == 0:
        return values
    indices_array = np.asarray(indices, dtype=np.int64)
    if indices_array.max() >= len(values):
        raise SapSceneLoaderError(
            f"{primvar_name} primvar index out of range for mesh {prim_path}: {indices_array.max()} >= {len(values)}"
        )
    if indices_array.min() < 0:
        raise SapSceneLoaderError(f"{primvar_name} primvar negative index for mesh {prim_path}: {indices_array.min()}")
    return values[indices_array]


def _usd_get_mesh(
    prim: Any,
    *,
    load_normals: bool = False,
    load_uvs: bool = False,
    maxhullvert: int | None = None,
) -> SapMesh:
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    counts = mesh.GetFaceVertexCountsAttr().Get()

    normals = None
    normals_interpolation = None
    normal_indices = None
    if load_normals:
        normals_primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("normals")
        if normals_primvar:
            normals = normals_primvar.Get()
            if normals is not None:
                normals_interpolation = normals_primvar.GetInterpolation()
                if normals_primvar.IsIndexed():
                    normal_indices = normals_primvar.GetIndices()
        if normals is None:
            normals_attr = mesh.GetNormalsAttr()
            if normals_attr:
                normals = normals_attr.Get()
                if normals is not None:
                    normals_interpolation = mesh.GetNormalsInterpolation()
        if normals is not None:
            normals = np.asarray(normals, dtype=np.float64)
            if normals_interpolation == UsdGeom.Tokens.faceVarying:
                normals = _usd_expand_indexed_primvar(normals, normal_indices, "Normal", str(prim.GetPath()))
                accum = np.zeros((len(points), 3), dtype=np.float64)
                for corner, vertex in enumerate(indices):
                    accum[int(vertex)] += normals[corner]
                lengths = np.linalg.norm(accum, axis=1, keepdims=True)
                lengths[lengths < 1.0e-20] = 1.0
                normals = (accum / lengths).astype(np.float32)
            else:
                normals = normals.astype(np.float32)

    uvs = None
    uvs_interpolation = None
    if load_uvs:
        uv_primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        if uv_primvar:
            uvs = uv_primvar.Get()
            if uvs is not None:
                uvs = np.asarray(uvs, dtype=np.float32)
                uvs_interpolation = uv_primvar.GetInterpolation()
                if uv_primvar.IsIndexed():
                    uvs = _usd_expand_indexed_primvar(uvs, uv_primvar.GetIndices(), "UV", str(prim.GetPath()))

    faces = _usd_fan_triangulate_faces(counts, indices)
    orientation_attr = mesh.GetOrientationAttr()
    flip_winding = False
    if orientation_attr:
        handedness = orientation_attr.Get()
        if handedness and str(handedness).lower() == "lefthanded":
            flip_winding = True
            faces = faces[:, ::-1]

    if uvs is not None:
        if uvs_interpolation == UsdGeom.Tokens.faceVarying:
            if len(uvs) != len(indices):
                warnings.warn(
                    f"UV primvar length ({len(uvs)}) does not match indices length ({len(indices)}) for mesh {prim.GetPath()}; dropping UVs.",
                    stacklevel=2,
                )
                uvs = None
            else:
                corner_flat = _usd_triangulate_face_varying_indices(counts, flip_winding)
                points_original = points
                points = points_original[indices[corner_flat]]
                if normals is not None and len(normals) == len(points_original):
                    normals = normals[indices[corner_flat]]
                uvs = uvs[corner_flat]
                faces = np.arange(len(corner_flat), dtype=np.int32).reshape(-1, 3)
        else:
            uvs = np.asarray(uvs, dtype=np.float32)

    material_props = _usd_resolve_material_properties_for_prim(prim)
    return SapMesh(
        points,
        faces.flatten(),
        normals=normals,
        uvs=uvs,
        maxhullvert=SapMesh.MAX_HULL_VERTICES if maxhullvert is None else int(maxhullvert),
        color=material_props.get("color"),
        texture=material_props.get("texture"),
        metallic=material_props.get("metallic"),
        roughness=material_props.get("roughness"),
    )


def _usd_resolve_asset_path(asset: Any, prim: Any, asset_attr: Any | None = None) -> str | None:
    from pxr import Sdf

    if asset is None:
        return None
    if asset_attr is not None:
        try:
            resolved_attr_path = asset_attr.GetResolvedPath()
        except Exception:
            resolved_attr_path = None
        if resolved_attr_path:
            return resolved_attr_path
    if isinstance(asset, Sdf.AssetPath):
        if asset.resolvedPath:
            return asset.resolvedPath
        asset_path = asset.path
    elif isinstance(asset, os.PathLike):
        asset_path = os.fspath(asset)
    elif isinstance(asset, str):
        asset_path = asset
    else:
        return None
    if not asset_path:
        return None
    if asset_path.startswith(("http://", "https://", "file:")):
        return asset_path
    if os.path.isabs(asset_path):
        return asset_path

    source_layer = None
    if asset_attr is not None:
        try:
            resolve_info = asset_attr.GetResolveInfo()
        except Exception:
            resolve_info = None
        if resolve_info is not None:
            for getter_name in ("GetSourceLayer", "GetLayer"):
                getter = getattr(resolve_info, getter_name, None)
                if getter is None:
                    continue
                try:
                    source_layer = getter()
                except Exception:
                    source_layer = None
                if source_layer is not None:
                    break
        if source_layer is None:
            try:
                spec = asset_attr.GetSpec()
            except Exception:
                spec = None
            if spec is not None:
                source_layer = getattr(spec, "layer", None)

    root_layer = prim.GetStage().GetRootLayer()
    base_layer = source_layer or root_layer
    if base_layer is not None:
        try:
            resolved = Sdf.ComputeAssetPathRelativeToLayer(base_layer, asset_path)
        except Exception:
            resolved = None
        if resolved:
            return resolved
        base_dir = os.path.dirname(base_layer.realPath or base_layer.identifier or "")
        if base_dir:
            return os.path.abspath(os.path.join(base_dir, asset_path))
    return asset_path


def _usd_empty_material_properties() -> dict[str, Any]:
    return {"color": None, "metallic": None, "roughness": None, "texture": None}


def _usd_coerce_color(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    color_np = np.array(value, dtype=np.float32).reshape(-1)
    if color_np.size >= 3:
        return (float(color_np[0]), float(color_np[1]), float(color_np[2]))
    return None


def _usd_coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _usd_find_texture_in_shader(shader: Any | None, prim: Any) -> str | None:
    from pxr import UsdShade

    if shader is None:
        return None
    shader_id = shader.GetIdAttr().Get()
    if shader_id == "UsdUVTexture":
        file_input = shader.GetInput("file")
        if file_input:
            attrs = UsdShade.Utils.GetValueProducingAttributes(file_input)
            if attrs:
                return _usd_resolve_asset_path(attrs[0].Get(), prim, attrs[0])
        return None
    if shader_id == "UsdPreviewSurface":
        for input_name in ("diffuseColor", "baseColor"):
            shader_input = shader.GetInput(input_name)
            if shader_input:
                source = shader_input.GetConnectedSource()
                if source:
                    texture = _usd_find_texture_in_shader(UsdShade.Shader(source[0].GetPrim()), prim)
                    if texture:
                        return texture
    return None


def _usd_get_input_value(shader: Any | None, names: tuple[str, ...]) -> Any | None:
    from pxr import UsdShade

    if shader is None:
        return None
    try:
        if not shader.GetPrim().IsValid():
            return None
    except Exception:
        return None
    for name in names:
        inp = shader.GetInput(name)
        if inp is None:
            continue
        try:
            attrs = UsdShade.Utils.GetValueProducingAttributes(inp)
        except Exception:
            continue
        if attrs:
            value = attrs[0].Get()
            if value is not None:
                return value
    return None


def _usd_extract_preview_surface_properties(shader: Any | None, prim: Any) -> dict[str, Any]:
    from pxr import UsdShade

    properties = _usd_empty_material_properties()
    if shader is None or shader.GetIdAttr().Get() != "UsdPreviewSurface":
        return properties
    color_input = shader.GetInput("baseColor") or shader.GetInput("diffuseColor")
    if color_input:
        source = color_input.GetConnectedSource()
        if source:
            source_shader = UsdShade.Shader(source[0].GetPrim())
            properties["texture"] = _usd_find_texture_in_shader(source_shader, prim)
            if properties["texture"] is None:
                properties["color"] = _usd_coerce_color(
                    _usd_get_input_value(
                        source_shader,
                        (
                            "diffuseColor",
                            "baseColor",
                            "diffuse_color",
                            "base_color",
                            "diffuse_color_constant",
                            "displayColor",
                        ),
                    )
                )
        else:
            properties["color"] = _usd_coerce_color(color_input.Get())

    metallic_input = shader.GetInput("metallic")
    if metallic_input:
        if metallic_input.HasConnectedSource():
            source = metallic_input.GetConnectedSource()
            source_shader = UsdShade.Shader(source[0].GetPrim()) if source else None
            properties["metallic"] = _usd_coerce_float(_usd_get_input_value(source_shader, ("metallic", "metallic_constant")))
        else:
            properties["metallic"] = _usd_coerce_float(metallic_input.Get())

    roughness_input = shader.GetInput("roughness")
    if roughness_input:
        if roughness_input.HasConnectedSource():
            source = roughness_input.GetConnectedSource()
            source_shader = UsdShade.Shader(source[0].GetPrim()) if source else None
            properties["roughness"] = _usd_coerce_float(
                _usd_get_input_value(
                    source_shader,
                    ("roughness", "roughness_constant", "reflection_roughness_constant"),
                )
            )
        else:
            properties["roughness"] = _usd_coerce_float(roughness_input.Get())
    return properties


def _usd_extract_shader_properties(shader: Any | None, prim: Any) -> dict[str, Any]:
    from pxr import UsdShade

    properties = _usd_extract_preview_surface_properties(shader, prim)
    if shader is None:
        return properties
    try:
        if not shader.GetPrim().IsValid():
            return properties
    except Exception:
        return properties

    if properties["color"] is None:
        properties["color"] = _usd_coerce_color(
            _usd_get_input_value(
                shader,
                (
                    "diffuse_color_constant",
                    "diffuse_color",
                    "diffuseColor",
                    "base_color",
                    "baseColor",
                    "displayColor",
                ),
            )
        )
    if properties["metallic"] is None:
        properties["metallic"] = _usd_coerce_float(_usd_get_input_value(shader, ("metallic_constant", "metallic")))
    if properties["roughness"] is None:
        properties["roughness"] = _usd_coerce_float(
            _usd_get_input_value(shader, ("reflection_roughness_constant", "roughness_constant", "roughness"))
        )
    if properties["texture"] is None:
        for inp in shader.GetInputs():
            name = inp.GetBaseName()
            if inp.HasConnectedSource():
                source = inp.GetConnectedSource()
                source_shader = UsdShade.Shader(source[0].GetPrim())
                texture = _usd_find_texture_in_shader(source_shader, prim)
                if texture:
                    properties["texture"] = texture
                    break
            elif "file" in name or "texture" in name:
                asset = inp.Get()
                if asset:
                    properties["texture"] = _usd_resolve_asset_path(asset, prim, inp.GetAttr())
                    break
    return properties


def _usd_extract_material_input_properties(material: Any | None, prim: Any) -> dict[str, Any]:
    properties = _usd_empty_material_properties()
    if material is None:
        return properties
    for inp in material.GetInputs():
        name_lower = inp.GetBaseName().lower()
        try:
            if inp.HasConnectedSource():
                continue
        except Exception:
            continue
        value = inp.Get()
        if value is None:
            continue
        if properties["texture"] is None and ("texture" in name_lower or "file" in name_lower):
            texture = _usd_resolve_asset_path(value, prim, inp.GetAttr())
            if texture:
                properties["texture"] = texture
                continue
        if properties["color"] is None and name_lower in (
            "diffusecolor",
            "basecolor",
            "diffuse_color",
            "base_color",
            "displaycolor",
        ):
            color = _usd_coerce_color(value)
            if color is not None:
                properties["color"] = color
                continue
        if properties["metallic"] is None and name_lower in ("metallic", "metallic_constant"):
            metallic = _usd_coerce_float(value)
            if metallic is not None:
                properties["metallic"] = metallic
                continue
        if properties["roughness"] is None and name_lower in (
            "roughness",
            "roughness_constant",
            "reflection_roughness_constant",
        ):
            roughness = _usd_coerce_float(value)
            if roughness is not None:
                properties["roughness"] = roughness
    return properties


def _usd_get_bound_material(target_prim: Any) -> Any | None:
    from pxr import UsdShade

    if not target_prim or not target_prim.IsValid():
        return None
    if target_prim.HasAPI(UsdShade.MaterialBindingAPI):
        binding_api = UsdShade.MaterialBindingAPI(target_prim)
        bound_material, _ = binding_api.ComputeBoundMaterial()
        return bound_material
    rels = [rel for rel in target_prim.GetRelationships() if rel.GetName().startswith("material:binding")]
    rels.sort(
        key=lambda rel: 0
        if rel.GetName() == "material:binding"
        else 1
        if rel.GetName() == "material:binding:preview"
        else 2
    )
    for rel in rels:
        targets = rel.GetTargets()
        if targets:
            mat_prim = target_prim.GetStage().GetPrimAtPath(targets[0])
            if mat_prim and mat_prim.IsValid():
                return UsdShade.Material(mat_prim)
    return None


def _usd_resolve_prim_material_properties(target_prim: Any) -> dict[str, Any] | None:
    from pxr import UsdShade

    material = _usd_get_bound_material(target_prim)
    if not material:
        return None
    surface_output = material.GetSurfaceOutput() or material.GetOutput("surface") or material.GetOutput("mdl:surface")
    source_shader = None
    if surface_output:
        source = surface_output.GetConnectedSource()
        if source:
            source_shader = UsdShade.Shader(source[0].GetPrim())
    if source_shader is None:
        for child in material.GetPrim().GetChildren():
            if child.IsA(UsdShade.Shader):
                source_shader = UsdShade.Shader(child)
                break
    if source_shader is None:
        material_props = _usd_extract_material_input_properties(material, target_prim)
        return material_props if any(value is not None for value in material_props.values()) else None
    properties = _usd_extract_shader_properties(source_shader, target_prim)
    material_props = _usd_extract_material_input_properties(material, target_prim)
    for key in ("texture", "color", "metallic", "roughness"):
        if properties.get(key) is None and material_props.get(key) is not None:
            properties[key] = material_props[key]
    if properties["color"] is None and properties["texture"] is None:
        from pxr import UsdGeom

        display_color = UsdGeom.PrimvarsAPI(target_prim).GetPrimvar("displayColor")
        if display_color:
            properties["color"] = _usd_coerce_color(display_color.Get())
    return properties


def _usd_resolve_material_properties_for_prim(prim: Any) -> dict[str, Any]:
    if not prim or not prim.IsValid():
        return _usd_empty_material_properties()
    properties = _usd_resolve_prim_material_properties(prim)
    if properties is not None:
        return properties
    proto_prim = None
    try:
        if prim.IsInstanceProxy():
            proto_prim = prim.GetPrimInPrototype()
        elif prim.IsInstance():
            proto_prim = prim.GetPrototype()
    except Exception:
        proto_prim = None
    if proto_prim and proto_prim.IsValid():
        properties = _usd_resolve_prim_material_properties(proto_prim)
        if properties is not None:
            return properties
    try:
        from pxr import UsdGeom

        is_mesh = prim.IsA(UsdGeom.Mesh)
    except Exception:
        is_mesh = False
    if is_mesh:
        fallback_props = None
        from pxr import UsdGeom

        for child in prim.GetChildren():
            try:
                is_subset = child.IsA(UsdGeom.Subset)
            except Exception:
                is_subset = False
            if not is_subset:
                continue
            subset_props = _usd_resolve_prim_material_properties(child)
            if subset_props is None:
                continue
            if subset_props.get("texture") is not None or subset_props.get("color") is not None:
                return subset_props
            if fallback_props is None:
                fallback_props = subset_props
        if fallback_props is not None:
            return fallback_props
    return _usd_empty_material_properties()


def _sanitize_xml_content(source: str) -> str:
    xml_content = source.strip()
    if xml_content.startswith("\ufeff"):
        xml_content = xml_content[1:]
    while xml_content.strip().startswith("<!--"):
        end_comment = xml_content.find("-->")
        if end_comment == -1:
            break
        xml_content = xml_content[end_comment + 3 :].strip()
    return xml_content.strip()


def _is_xml_content(source: str) -> bool:
    return any(char in source for char in "<>")


def _sanitize_name(name: str) -> str:
    return name.replace("-", "_")


def _default_mjcf_path_resolver(base_dir: str | None, file_path: str) -> str:
    if os.path.isabs(file_path):
        return os.path.normpath(file_path)
    if base_dir:
        return os.path.abspath(os.path.join(base_dir, file_path))
    raise SapSceneLoaderError(f"Cannot resolve relative path {file_path!r} without base directory")


def _load_and_expand_mjcf(
    source: str,
    *,
    included_files: set[str] | None = None,
) -> tuple[ET.Element, str | None]:
    if included_files is None:
        included_files = set()

    if _is_xml_content(source):
        base_dir = None
        root = ET.fromstring(_sanitize_xml_content(source))
    else:
        base_dir = os.path.dirname(source) or "."
        root = ET.parse(source).getroot()

    own_compiler = root.find("compiler")
    own_meshdir = own_compiler.attrib.get("meshdir", ".") if own_compiler is not None else "."
    own_texturedir = own_compiler.attrib.get("texturedir", own_meshdir) if own_compiler is not None else "."
    if own_compiler is not None:
        own_compiler.attrib.pop("meshdir", None)
        own_compiler.attrib.pop("texturedir", None)

    include_pairs = [(parent, child) for parent in root.iter() for child in parent if child.tag == "include"]
    for parent, include in include_pairs:
        file_attr = include.get("file")
        if not file_attr:
            continue
        resolved = _default_mjcf_path_resolver(base_dir, file_attr)
        if not _is_xml_content(resolved):
            if resolved in included_files:
                raise SapSceneLoaderError(f"Circular MJCF include detected: {resolved}")
            included_files.add(resolved)
        included_root, _ = _load_and_expand_mjcf(resolved, included_files=included_files)
        index = list(parent).index(include)
        parent.remove(include)
        for offset, child in enumerate(included_root):
            parent.insert(index + offset, child)

    asset_dir_tags = {"mesh": own_meshdir, "hfield": own_meshdir, "texture": own_texturedir}
    for elem in root.iter():
        file_attr = elem.get("file")
        if file_attr and not os.path.isabs(file_attr):
            asset_dir = asset_dir_tags.get(elem.tag, ".")
            resolved_path = os.path.join(asset_dir, file_attr) if asset_dir != "." else file_attr
            if base_dir is not None or os.path.isabs(resolved_path):
                elem.set("file", _default_mjcf_path_resolver(base_dir, resolved_path))

    return root, base_dir


def _add_mjcf_asset(builder: _SapSceneBuilder, source: str, cfg: dict[str, Any]) -> None:
    source_text = os.fspath(source)
    root, base_dir = _load_and_expand_mjcf(source_text)
    mjcf_dirname = base_dir or "."

    xform = _transform(cfg.get("xform")) or wp.transform_identity()
    scale = _float(cfg.get("scale", 1.0))
    armature_scale = _float(cfg.get("armature_scale", 1.0))
    mesh_maxhullvert = int(cfg.get("mesh_maxhullvert", SapMesh.MAX_HULL_VERTICES))
    parse_visuals_as_colliders = bool(cfg.get("parse_visuals_as_colliders", False))
    parse_meshes = bool(cfg.get("parse_meshes", True))
    parse_sites = bool(cfg.get("parse_sites", True))
    parse_visuals = bool(cfg.get("parse_visuals", True))
    force_show_colliders = bool(cfg.get("force_show_colliders", False))
    enable_self_collisions = bool(cfg.get("enable_self_collisions", True))
    no_class_as_colliders = bool(cfg.get("no_class_as_colliders", True))
    visual_classes = tuple(cfg.get("visual_classes", ("visual",)))
    collider_classes = tuple(cfg.get("collider_classes", ("collision",)))
    ignore_names = tuple(cfg.get("ignore_names", ()))
    ignore_classes = tuple(cfg.get("ignore_classes", ()))
    ignore_inertial_definitions = bool(cfg.get("ignore_inertial_definitions", False))
    default_joint_limit_lower = builder.default_joint_cfg.limit_lower
    default_joint_limit_upper = builder.default_joint_cfg.limit_upper
    default_joint_target_ke = builder.default_joint_cfg.target_ke
    default_joint_target_kd = builder.default_joint_cfg.target_kd
    default_joint_armature = builder.default_joint_cfg.armature
    default_joint_effort_limit = builder.default_joint_cfg.effort_limit

    compiler = root.find("compiler")
    if compiler is not None:
        use_degrees = compiler.attrib.get("angle", "degree").lower() == "degree"
        euler_seq = ["xyz".index(c) for c in compiler.attrib.get("eulerseq", "xyz").lower()]
        mesh_dir = compiler.attrib.get("meshdir", ".")
        texture_dir = compiler.attrib.get("texturedir", mesh_dir)
    else:
        use_degrees = True
        euler_seq = [0, 1, 2]
        mesh_dir = "."
        texture_dir = "."

    class_children: dict[str, list[str]] = {}
    class_defaults: dict[str, dict[str, dict[str, str]]] = {"__all__": {}}

    def get_class(element: ET.Element) -> str:
        return element.get("class", "__all__")

    def parse_default(node: ET.Element, parent: str | None) -> None:
        class_name = "__all__"
        if "class" in node.attrib:
            class_name = node.attrib["class"]
            parent = parent or "__all__"
            class_children.setdefault(parent, []).append(class_name)
        class_defaults.setdefault(class_name, {})
        for child in node:
            if child.tag == "default":
                parse_default(child, node.get("class"))
            else:
                class_defaults[class_name][child.tag] = dict(child.attrib)

    for default in root.findall("default"):
        parse_default(default, None)

    def merge_attrib(default_attrib: dict, incoming_attrib: dict) -> dict:
        attrib = default_attrib.copy()
        for key, value in incoming_attrib.items():
            if key in attrib and isinstance(attrib[key], dict) and isinstance(value, dict):
                attrib[key] = merge_attrib(attrib[key], value)
            else:
                attrib[key] = value
        return attrib

    def resolve_defaults(class_name: str) -> None:
        if class_name in class_children:
            for child_name in class_children[class_name]:
                if class_name in class_defaults and child_name in class_defaults:
                    class_defaults[child_name] = merge_attrib(class_defaults[class_name], class_defaults[child_name])
                resolve_defaults(child_name)

    resolve_defaults("__all__")

    mesh_assets: dict[str, dict[str, Any]] = {}
    texture_assets: dict[str, dict[str, str]] = {}
    material_assets: dict[str, dict[str, str | None]] = {}
    hfield_assets: dict[str, dict[str, Any]] = {}
    for asset in root.findall("asset"):
        for mesh in asset.findall("mesh"):
            if "file" not in mesh.attrib:
                continue
            fname = os.path.join(mesh_dir, mesh.attrib["file"])
            if not os.path.isabs(fname):
                fname = os.path.abspath(os.path.join(mjcf_dirname, fname))
            mesh_class = mesh.attrib.get("class", "__all__")
            mesh_defaults = class_defaults.get(mesh_class, {}).get("mesh", {})
            mesh_attrib = merge_attrib(mesh_defaults, dict(mesh.attrib))
            name = mesh.attrib.get("name", ".".join(os.path.basename(fname).split(".")[:-1]))
            mesh_scale = np.fromstring(mesh_attrib.get("scale", "1.0 1.0 1.0"), sep=" ", dtype=np.float32)
            mesh_assets[name] = {
                "file": fname,
                "scale": mesh_scale,
                "maxhullvert": int(mesh_attrib.get("maxhullvert", str(mesh_maxhullvert))),
            }
        for texture in asset.findall("texture"):
            tex_name = texture.attrib.get("name")
            tex_file = texture.attrib.get("file")
            if not tex_name or not tex_file:
                continue
            tex_path = os.path.join(texture_dir, tex_file)
            if not os.path.isabs(tex_path):
                tex_path = os.path.abspath(os.path.join(mjcf_dirname, tex_path))
            texture_assets[tex_name] = {"file": tex_path}
        for material in asset.findall("material"):
            mat_name = material.attrib.get("name")
            if mat_name:
                material_assets[mat_name] = {
                    "rgba": material.attrib.get("rgba"),
                    "texture": material.attrib.get("texture"),
                }
        for hfield in asset.findall("hfield"):
            hfield_name = hfield.attrib.get("name")
            if hfield_name:
                hfield_assets[hfield_name] = dict(hfield.attrib)

    def parse_float(attrib: dict[str, str], key: str, default: float) -> float:
        return float(attrib[key]) if key in attrib else default

    def parse_vec(attrib: dict[str, str], key: str, default: Sequence[float] | None) -> np.ndarray | None:
        if key in attrib:
            out = np.fromstring(attrib[key], sep=" ", dtype=np.float32)
        else:
            if default is None:
                return None
            out = np.array(default, dtype=np.float32)
        if len(out) == 1 and default is not None:
            return np.full(len(default), float(out[0]), dtype=np.float32)
        return out

    def quat_from_euler_mjcf(e: np.ndarray, i: int, j: int, k: int) -> wp.quat:
        half_e = e * 0.5
        cr = math.cos(float(half_e[i]))
        sr = math.sin(float(half_e[i]))
        cp = math.cos(float(half_e[j]))
        sp = math.sin(float(half_e[j]))
        cy = math.cos(float(half_e[k]))
        sy = math.sin(float(half_e[k]))
        return wp.quat(
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
            cy * cr * cp + sy * sr * sp,
        )

    def parse_orientation(attrib: dict[str, str]) -> wp.quat:
        if "quat" in attrib:
            wxyz = np.fromstring(attrib["quat"], sep=" ")
            return wp.normalize(wp.quat(*wxyz[1:], wxyz[0]))
        if "euler" in attrib:
            euler = np.fromstring(attrib["euler"], sep=" ")
            if use_degrees:
                euler *= np.pi / 180.0
            return quat_from_euler_mjcf(euler, *euler_seq)
        if "axisangle" in attrib:
            axisangle = np.fromstring(attrib["axisangle"], sep=" ")
            angle = float(axisangle[3])
            if use_degrees:
                angle *= np.pi / 180.0
            axis = wp.normalize(wp.vec3(*axisangle[:3]))
            return wp.quat_from_axis_angle(axis, angle)
        if "xyaxes" in attrib:
            xyaxes = np.fromstring(attrib["xyaxes"], sep=" ")
            xaxis = wp.normalize(wp.vec3(*xyaxes[:3]))
            zaxis = wp.normalize(wp.vec3(*xyaxes[3:]))
            yaxis = wp.normalize(wp.cross(zaxis, xaxis))
            return wp.quat_from_matrix(wp.mat33(np.array([xaxis, yaxis, zaxis]).T))
        if "zaxis" in attrib:
            zaxis_np = np.fromstring(attrib["zaxis"], sep=" ")
            zaxis = wp.normalize(wp.vec3(*zaxis_np))
            xaxis = wp.normalize(wp.cross(wp.vec3(0.0, 0.0, 1.0), zaxis))
            yaxis = wp.normalize(wp.cross(zaxis, xaxis))
            return wp.quat_from_matrix(wp.mat33(np.array([xaxis, yaxis, zaxis]).T))
        return wp.quat_identity()

    def parse_shapes(
        defaults: dict,
        body_name: str,
        link: int,
        geoms: Sequence[ET.Element],
        density: float,
        *,
        visible: bool = True,
        just_visual: bool = False,
        incoming_xform: wp.transform | None = None,
        label_prefix: str = "",
    ) -> list[int]:
        shapes: list[int] = []
        for geo_count, geom in enumerate(geoms):
            geom_defaults = defaults
            if "class" in geom.attrib:
                geom_class = geom.attrib["class"]
                if any(re.match(pattern, geom_class) for pattern in ignore_classes):
                    continue
                if geom_class in class_defaults:
                    geom_defaults = merge_attrib(defaults, class_defaults[geom_class])
            geom_attrib = merge_attrib(geom_defaults.get("geom", {}), dict(geom.attrib))

            geom_name = geom_attrib.get("name", f"{body_name}_geom_{geo_count}{'_visual' if just_visual else ''}")
            geom_type = geom_attrib.get("type", "sphere")
            fit_to_mesh = False
            if "mesh" in geom_attrib:
                if "type" in geom_attrib and geom_type in {"sphere", "capsule", "cylinder", "ellipsoid", "box"}:
                    fit_to_mesh = True
                else:
                    geom_type = "mesh"
            if "hfield" in geom_attrib:
                geom_type = "hfield"

            if any(re.match(pattern, geom_name) for pattern in ignore_names):
                continue

            geom_size = parse_vec(geom_attrib, "size", (1.0, 1.0, 1.0)) * scale
            geom_pos = parse_vec(geom_attrib, "pos", (0.0, 0.0, 0.0)) * scale
            geom_rot = parse_orientation(geom_attrib)
            tf = wp.transform(wp.vec3(*geom_pos[:3]), geom_rot)
            if incoming_xform is not None:
                tf = incoming_xform * tf

            shape_cfg = builder.default_shape_cfg.copy()
            shape_cfg.is_visible = visible
            shape_cfg.has_shape_collision = not just_visual
            shape_cfg.has_particle_collision = not just_visual
            shape_cfg.density = parse_float(geom_attrib, "density", density)

            contype = int(geom_attrib.get("contype", 1))
            conaffinity = int(geom_attrib.get("conaffinity", 1))
            if contype == 0 and conaffinity == 0 and not just_visual:
                shape_cfg.collision_group = 0

            if "friction" in geom_attrib:
                friction_values = np.fromstring(geom_attrib["friction"], sep=" ", dtype=np.float32)
                if len(friction_values) >= 1:
                    shape_cfg.mu = float(friction_values[0])
                if len(friction_values) >= 2:
                    shape_cfg.mu_torsional = float(friction_values[1])
                if len(friction_values) >= 3:
                    shape_cfg.mu_rolling = float(friction_values[2])

            if "solref" in geom_attrib:
                geom_ke, geom_kd = _sap_solref_to_stiffness_damping(parse_vec(geom_attrib, "solref", (0.02, 1.0)))
                if geom_ke is not None:
                    shape_cfg.ke = geom_ke
                if geom_ke is not None and geom_kd is not None and geom_ke > 0.0:
                    shape_cfg.tau = geom_kd / geom_ke

            mj_gap = float(geom_attrib.get("gap", "0")) * scale
            if "margin" in geom_attrib:
                mj_margin = float(geom_attrib["margin"]) * scale
                shape_cfg.margin = mj_margin - mj_gap
            if "gap" in geom_attrib:
                shape_cfg.gap = mj_gap

            material_name = geom_attrib.get("material")
            material_info = material_assets.get(material_name, {})
            rgba = geom_attrib.get("rgba", material_info.get("rgba"))
            material_color = None
            if rgba is not None:
                rgba_values = np.fromstring(str(rgba), sep=" ", dtype=np.float32)
                if len(rgba_values) >= 3:
                    material_color = (float(rgba_values[0]), float(rgba_values[1]), float(rgba_values[2]))
            texture = None
            texture_name = material_info.get("texture")
            if texture_name:
                texture_asset = texture_assets.get(str(texture_name))
                if texture_asset and "file" in texture_asset:
                    texture = texture_asset["file"]

            shape_label = f"{label_prefix}/{geom_name}" if label_prefix else geom_name
            common = {
                "body": link,
                "xform": tf,
                "cfg": shape_cfg,
                "label": shape_label,
            }

            if fit_to_mesh:
                raise SapSceneLoaderError(f"mesh fitting for geom {geom_name!r}")

            if geom_type == "sphere":
                shapes.append(
                    builder.add_shape(
                        shape_type=int(SapGeoType.SPHERE),
                        scale=(float(geom_size[0]), 0.0, 0.0),
                        **common,
                    )
                )
            elif geom_type == "box":
                shapes.append(
                    builder.add_shape(
                        shape_type=int(SapGeoType.BOX),
                        scale=(float(geom_size[0]), float(geom_size[1]), float(geom_size[2])),
                        **common,
                    )
                )
            elif geom_type == "mesh" and parse_meshes:
                mesh_name = geom_attrib.get("mesh")
                if mesh_name is None or mesh_name not in mesh_assets:
                    continue
                mesh_asset = mesh_assets[str(mesh_name)]
                scaling = np.asarray(mesh_asset["scale"], dtype=np.float32) * scale
                for sap_mesh in _load_sap_meshes_from_file(
                    mesh_asset["file"],
                    scale=scaling,
                    maxhullvert=int(mesh_asset.get("maxhullvert", mesh_maxhullvert)),
                    override_color=material_color,
                    override_texture=texture,
                ):
                    if sap_mesh.texture is not None and sap_mesh.uvs is None:
                        sap_mesh.texture = None
                    mesh_cfg = shape_cfg.copy()
                    mesh_cfg.sdf_max_resolution = None
                    mesh_cfg.sdf_target_voxel_size = None
                    mesh_cfg.sdf_narrow_band_range = (-0.1, 0.1)
                    mesh_common = dict(common)
                    mesh_common["cfg"] = mesh_cfg
                    shapes.append(
                        builder.add_shape(
                            shape_type=int(SapGeoType.MESH),
                            scale=(1.0, 1.0, 1.0),
                            src=sap_mesh,
                            **mesh_common,
                        )
                    )
            elif geom_type in {"capsule", "cylinder"}:
                geom_radius = float(geom_size[0])
                geom_height = float(geom_size[1])
                if "fromto" in geom_attrib:
                    geom_fromto = parse_vec(geom_attrib, "fromto", (0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
                    start = wp.vec3(*geom_fromto[0:3]) * scale
                    end = wp.vec3(*geom_fromto[3:6]) * scale
                    if incoming_xform is not None:
                        start = wp.transform_point(incoming_xform, start)
                        end = wp.transform_point(incoming_xform, end)
                    geom_pos_ft = (start + end) * 0.5
                    dir_vec = start - end
                    dir_len = wp.length(dir_vec)
                    if dir_len < 1.0e-6:
                        geom_rot_ft = wp.quat_identity()
                    else:
                        direction = dir_vec / dir_len
                        if float(direction[2]) < -0.999999:
                            geom_rot_ft = wp.quat(1.0, 0.0, 0.0, 0.0)
                        else:
                            geom_rot_ft = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), direction)
                    common["xform"] = wp.transform(geom_pos_ft, geom_rot_ft)
                    geom_height = float(dir_len) * 0.5
                shapes.append(
                    builder.add_shape(
                        shape_type=int(SapGeoType.CYLINDER if geom_type == "cylinder" else SapGeoType.CAPSULE),
                        scale=(geom_radius, geom_height, 0.0),
                        **common,
                    )
                )
            elif geom_type == "plane":
                shapes.append(
                    builder.add_shape(
                        shape_type=int(SapGeoType.PLANE),
                        scale=(float(geom_size[0]), float(geom_size[1]), 0.0),
                        **common,
                    )
                )
            elif geom_type == "ellipsoid":
                shapes.append(
                    builder.add_shape(
                        shape_type=int(SapGeoType.ELLIPSOID),
                        scale=(float(geom_size[0]), float(geom_size[1]), float(geom_size[2])),
                        **common,
                    )
                )
            elif geom_type == "hfield":
                if hfield_assets:
                    raise SapSceneLoaderError(f"hfield geom {geom_name!r}")

        return shapes

    def parse_sites_impl(
        defaults: dict,
        body_name: str,
        link: int,
        sites: Sequence[ET.Element],
        *,
        incoming_xform: wp.transform | None = None,
        label_prefix: str = "",
    ) -> list[int]:
        site_shapes: list[int] = []
        for site_count, site in enumerate(sites):
            site_defaults = defaults
            if "class" in site.attrib:
                site_class = site.attrib["class"]
                if any(re.match(pattern, site_class) for pattern in ignore_classes):
                    continue
                if site_class in class_defaults:
                    site_defaults = merge_attrib(defaults, class_defaults[site_class])
            site_attrib = merge_attrib(site_defaults.get("site", {}), dict(site.attrib))
            site_name = site_attrib.get("name", f"{body_name}_site_{site_count}")
            if any(re.match(pattern, site_name) for pattern in ignore_names):
                continue
            site_pos = parse_vec(site_attrib, "pos", (0.0, 0.0, 0.0)) * scale
            site_rot = parse_orientation(site_attrib)
            site_xform = wp.transform(wp.vec3(*site_pos[:3]), site_rot)
            if incoming_xform is not None:
                site_xform = incoming_xform * site_xform
            site_size = np.array([0.005, 0.005, 0.005], dtype=np.float32)
            if "size" in site_attrib:
                size_values = np.fromstring(site_attrib["size"], sep=" ", dtype=np.float32)
                for i, value in enumerate(size_values):
                    if i < 3:
                        site_size[i] = value
            site_size = site_size * scale
            site_type = {
                "sphere": int(SapGeoType.SPHERE),
                "box": int(SapGeoType.BOX),
                "capsule": int(SapGeoType.CAPSULE),
                "cylinder": int(SapGeoType.CYLINDER),
                "ellipsoid": int(SapGeoType.ELLIPSOID),
            }.get(site_attrib.get("type", "sphere"), int(SapGeoType.SPHERE))
            site_cfg = _SapShapeConfig()
            site_cfg.is_site = True
            site_cfg.is_visible = False
            site_cfg.has_shape_collision = False
            site_cfg.has_particle_collision = False
            site_cfg.density = 0.0
            site_cfg.collision_group = 0
            site_label = f"{label_prefix}/{site_name}" if label_prefix else site_name
            site_shapes.append(
                builder.add_shape(
                    body=link,
                    shape_type=site_type,
                    xform=site_xform,
                    cfg=site_cfg,
                    scale=(float(site_size[0]), float(site_size[1]), float(site_size[2])),
                    label=site_label,
                )
            )
        return site_shapes

    visual_shapes: list[int] = []
    start_shape_count = builder.shape_count
    body_name_to_idx: dict[str, int] = {}
    joint_indices: list[int] = []
    articulation_label = root.attrib.get("model")
    root_label_path = f"{articulation_label}/worldbody" if articulation_label else "worldbody"
    default_shape_density = builder.default_shape_cfg.density

    def process_body_geoms(
        geoms: Sequence[ET.Element],
        defaults: dict,
        body_name: str,
        link: int,
        *,
        incoming_xform: wp.transform | None = None,
        label_prefix: str = "",
    ) -> list[int]:
        visuals: list[ET.Element] = []
        colliders: list[ET.Element] = []
        for geom in geoms:
            geom_defaults = defaults
            geom_class = geom.attrib.get("class")
            if geom_class is not None:
                if any(re.match(pattern, geom_class) for pattern in ignore_classes):
                    continue
                if geom_class in class_defaults:
                    geom_defaults = merge_attrib(defaults, class_defaults[geom_class])
            geom_attrib = merge_attrib(geom_defaults.get("geom", {}), dict(geom.attrib))
            contype = int(geom_attrib.get("contype", 1))
            conaffinity = int(geom_attrib.get("conaffinity", 1))
            collides_with_anything = not (contype == 0 and conaffinity == 0)

            if geom_class is not None:
                neither_visual_nor_collider = True
                for pattern in visual_classes:
                    if re.match(pattern, geom_class):
                        visuals.append(geom)
                        neither_visual_nor_collider = False
                        break
                for pattern in collider_classes:
                    if re.match(pattern, geom_class):
                        colliders.append(geom)
                        neither_visual_nor_collider = False
                        break
                if neither_visual_nor_collider:
                    if no_class_as_colliders and collides_with_anything:
                        colliders.append(geom)
                    else:
                        visuals.append(geom)
            elif no_class_as_colliders and collides_with_anything:
                colliders.append(geom)
            else:
                visuals.append(geom)

        visual_shape_indices: list[int] = []
        if parse_visuals_as_colliders:
            colliders = visuals
        elif parse_visuals:
            visual_shape_indices.extend(
                parse_shapes(
                    defaults,
                    body_name,
                    link,
                    visuals,
                    default_shape_density,
                    just_visual=True,
                    visible=True,
                    incoming_xform=incoming_xform,
                    label_prefix=label_prefix,
                )
            )

        parse_shapes(
            defaults,
            body_name,
            link,
            colliders,
            default_shape_density,
            visible=_should_show_collider(
                force_show_colliders,
                has_visual_shapes=len(visuals) > 0 and parse_visuals,
                parse_visuals_as_colliders=parse_visuals_as_colliders,
            ),
            incoming_xform=incoming_xform,
            label_prefix=label_prefix,
        )
        return visual_shape_indices

    def parse_body(
        body: ET.Element,
        parent: int,
        incoming_defaults: dict,
        *,
        childclass: str | None = None,
        incoming_xform: wp.transform | None = None,
        parent_label_path: str = "",
    ) -> None:
        body_class = body.get("class") or body.get("childclass")
        if body_class is None:
            body_class = childclass
            defaults = incoming_defaults
        else:
            if any(re.match(pattern, body_class) for pattern in ignore_classes):
                return
            defaults = merge_attrib(incoming_defaults, class_defaults.get(body_class, {}))
        body_attrib = merge_attrib(defaults.get("body", {}), dict(body.attrib))
        body_name = _sanitize_name(body_attrib.get("name", f"body_{len(builder.body_q)}"))
        body_label_path = f"{parent_label_path}/{body_name}" if parent_label_path else body_name

        body_pos = parse_vec(body_attrib, "pos", (0.0, 0.0, 0.0))
        body_ori = parse_orientation(body_attrib)
        local_xform = wp.transform(wp.vec3(*(body_pos * scale)[:3]), body_ori)
        parent_xform = incoming_xform if incoming_xform is not None else xform
        world_xform = parent_xform * local_xform
        if parent >= 0:
            relative_xform = wp.transform_inverse(builder.body_q[parent]) * world_xform
            body_pos_for_joints = relative_xform.p
            body_ori_for_joints = relative_xform.q
        else:
            body_pos_for_joints = world_xform.p
            body_ori_for_joints = world_xform.q

        linear_axes: list[_SapJointDofConfig] = []
        angular_axes: list[_SapJointDofConfig] = []
        joint_positions: list[wp.vec3] = []
        joint_names: list[str] = []
        joint_type = None
        freejoint_tags = body.findall("freejoint")
        if freejoint_tags:
            joint_type = _SAP_JOINT_FREE
            joint_names.append(_sanitize_name(freejoint_tags[0].attrib.get("name", f"{body_name}_freejoint")))
        else:
            for joint_count, joint in enumerate(body.findall("joint")):
                joint_defaults = defaults
                if "class" in joint.attrib:
                    joint_class = joint.attrib["class"]
                    if joint_class in class_defaults:
                        joint_defaults = merge_attrib(defaults, class_defaults[joint_class])
                joint_attrib = merge_attrib(joint_defaults.get("joint", {}), dict(joint.attrib))
                joint_type_str = joint_attrib.get("type", "hinge")
                joint_names.append(_sanitize_name(joint_attrib.get("name") or f"{body_name}_joint_{joint_count}"))
                joint_positions.append(wp.vec3(*(parse_vec(joint_attrib, "pos", (0.0, 0.0, 0.0)) * scale)[:3]))
                joint_range = parse_vec(joint_attrib, "range", (default_joint_limit_lower, default_joint_limit_upper))
                if joint_type_str == "free":
                    joint_type = _SAP_JOINT_FREE
                    break
                if joint_type_str == "fixed":
                    joint_type = _SAP_JOINT_FIXED
                    break
                axis = wp.vec3(*parse_vec(joint_attrib, "axis", (0.0, 0.0, 1.0))[:3])
                solreflimit = parse_vec(joint_attrib, "solreflimit", (0.02, 1.0))
                limit_ke, limit_kd = _sap_solref_to_stiffness_damping(solreflimit)
                if limit_ke is None:
                    limit_ke = 2500.0
                if limit_kd is None:
                    limit_kd = 100.0
                effort_limit = default_joint_effort_limit
                if "actuatorfrcrange" in joint_attrib:
                    actuatorfrcrange = parse_vec(joint_attrib, "actuatorfrcrange", None)
                    if actuatorfrcrange is not None and len(actuatorfrcrange) == 2:
                        actuatorfrclimited = joint_attrib.get("actuatorfrclimited", "auto").lower()
                        if actuatorfrclimited == "true" or actuatorfrclimited == "auto":
                            effort_limit = max(abs(float(actuatorfrcrange[0])), abs(float(actuatorfrcrange[1])))
                if joint_type_str == "hinge":
                    lower = math.radians(float(joint_range[0])) if use_degrees else float(joint_range[0])
                    upper = math.radians(float(joint_range[1])) if use_degrees else float(joint_range[1])
                    angular_axes.append(
                        _SapJointDofConfig(
                            axis=axis,
                            limit_lower=lower,
                            limit_upper=upper,
                            limit_ke=limit_ke,
                            limit_kd=limit_kd,
                            target_ke=default_joint_target_ke,
                            target_kd=default_joint_target_kd,
                            armature=parse_float(joint_attrib, "armature", default_joint_armature) * armature_scale,
                            friction=parse_float(joint_attrib, "frictionloss", 0.0),
                            effort_limit=effort_limit,
                            actuator_mode=0,
                        )
                    )
                else:
                    linear_axes.append(
                        _SapJointDofConfig(
                            axis=axis,
                            limit_lower=float(joint_range[0]),
                            limit_upper=float(joint_range[1]),
                            limit_ke=limit_ke,
                            limit_kd=limit_kd,
                            target_ke=default_joint_target_ke,
                            target_kd=default_joint_target_kd,
                            armature=parse_float(joint_attrib, "armature", default_joint_armature) * armature_scale,
                            friction=parse_float(joint_attrib, "frictionloss", 0.0),
                            effort_limit=effort_limit,
                            actuator_mode=0,
                        )
                    )

        if joint_type is None:
            if not linear_axes and not angular_axes:
                joint_type = _SAP_JOINT_FIXED
            elif not linear_axes and len(angular_axes) == 1:
                joint_type = _SAP_JOINT_REVOLUTE
            elif len(linear_axes) == 1 and not angular_axes:
                joint_type = _SAP_JOINT_PRISMATIC
            else:
                joint_type = _SAP_JOINT_D6

        link = builder.add_body(xform=world_xform, label=body_label_path)
        body_name_to_idx[body_name] = link
        joint_pos = joint_positions[0] if joint_positions else wp.vec3(0.0, 0.0, 0.0)
        if joint_names:
            joint_label_name = "_".join(joint_names)
        else:
            joint_label_name = f"{body_name}_joint"
        joint_label = f"{body_label_path}/{joint_label_name}"
        if joint_type == _SAP_JOINT_FREE:
            joint_id = builder.add_joint(
                parent=parent,
                child=link,
                joint_type=_SAP_JOINT_FREE,
                linear_axes=_free_joint_linear_axes(),
                angular_axes=_free_joint_angular_axes(),
                label=joint_label,
                collision_filter_parent=True,
            )
        else:
            if parent == -1:
                parent_xform_for_joint = world_xform * wp.transform(joint_pos, wp.quat_identity())
            else:
                rotated_joint_pos = wp.quat_rotate(body_ori_for_joints, joint_pos)
                parent_xform_for_joint = wp.transform(body_pos_for_joints + rotated_joint_pos, body_ori_for_joints)
            joint_id = builder.add_joint(
                parent=parent,
                child=link,
                joint_type=joint_type,
                linear_axes=linear_axes,
                angular_axes=angular_axes,
                parent_xform=parent_xform_for_joint,
                child_xform=wp.transform(joint_pos, wp.quat_identity()),
                label=joint_label,
                collision_filter_parent=True,
            )
        joint_indices.append(joint_id)

        body_visual_shapes = process_body_geoms(
            body.findall("geom"),
            defaults,
            body_name,
            link,
            label_prefix=body_label_path,
        )
        visual_shapes.extend(body_visual_shapes)

        if parse_sites:
            parse_sites_impl(defaults, body_name, link, body.findall("site"), label_prefix=body_label_path)

        inertial = body.find("inertial")
        if not ignore_inertial_definitions and inertial is not None:
            inertial_attrib = merge_attrib(defaults.get("inertial", {}), dict(inertial.attrib))
            inertial_pos = parse_vec(inertial_attrib, "pos", (0.0, 0.0, 0.0)) * scale
            inertial_rot = parse_orientation(inertial_attrib)
            inertial_frame = wp.transform(wp.vec3(*inertial_pos[:3]), inertial_rot)
            if inertial_attrib.get("diaginertia") is not None:
                diaginertia = parse_vec(inertial_attrib, "diaginertia", (0.0, 0.0, 0.0))
                inertia_np = np.zeros((3, 3), dtype=np.float32)
                inertia_np[0, 0] = diaginertia[0] * scale**2
                inertia_np[1, 1] = diaginertia[1] * scale**2
                inertia_np[2, 2] = diaginertia[2] * scale**2
            else:
                fullinertia = inertial_attrib.get("fullinertia")
                if fullinertia is None:
                    raise SapSceneLoaderError(f"MJCF body {body_name!r} inertial lacks diaginertia/fullinertia")
                values = np.fromstring(fullinertia, sep=" ", dtype=np.float32)
                inertia_np = np.zeros((3, 3), dtype=np.float32)
                inertia_np[0, 0] = values[0] * scale**2
                inertia_np[1, 1] = values[1] * scale**2
                inertia_np[2, 2] = values[2] * scale**2
                inertia_np[0, 1] = values[3] * scale**2
                inertia_np[0, 2] = values[4] * scale**2
                inertia_np[1, 2] = values[5] * scale**2
                inertia_np[1, 0] = inertia_np[0, 1]
                inertia_np[2, 0] = inertia_np[0, 2]
                inertia_np[2, 1] = inertia_np[1, 2]
            rot_np = np.asarray(wp.quat_to_matrix(inertial_frame.q), dtype=np.float32).reshape(3, 3)
            inertia = wp.mat33(rot_np @ inertia_np @ rot_np.T)
            mass = float(inertial_attrib.get("mass", "0"))
            builder.body_mass[link] = mass
            builder.body_inv_mass[link] = 1.0 / mass if mass > 0.0 else 0.0
            builder.body_com[link] = inertial_frame.p
            builder.body_inertia[link] = inertia
            builder.body_inv_inertia[link] = wp.inverse(inertia) if any(x for x in inertia) else inertia
            builder.body_lock_inertia[link] = True

        for child in body.findall("body"):
            next_childclass = body.get("childclass")
            if next_childclass is None:
                next_childclass = childclass
                child_defaults = defaults
            else:
                child_defaults = merge_attrib(defaults, class_defaults.get(next_childclass, {}))
            parse_body(
                child,
                link,
                child_defaults,
                childclass=next_childclass,
                incoming_xform=world_xform,
                parent_label_path=body_label_path,
            )

    for world in root.findall("worldbody"):
        world_class = get_class(world)
        world_defaults = merge_attrib(class_defaults["__all__"], class_defaults.get(world_class, {}))
        for body in world.findall("body"):
            parse_body(body, -1, world_defaults, incoming_xform=xform, parent_label_path=root_label_path)

        parse_shapes(
            defaults=world_defaults,
            body_name="world",
            link=-1,
            geoms=world.findall("geom"),
            density=default_shape_density,
            incoming_xform=xform,
            label_prefix=root_label_path,
        )
        if parse_sites:
            parse_sites_impl(
                world_defaults,
                "world",
                -1,
                world.findall("site"),
                incoming_xform=xform,
                label_prefix=root_label_path,
            )

    if joint_indices:
        builder.add_articulation(joint_indices, label=articulation_label)

    contact = root.find("contact")
    if contact is not None:
        for exclude in contact.findall("exclude"):
            body1_name = exclude.attrib.get("body1")
            body2_name = exclude.attrib.get("body2")
            if not body1_name or not body2_name:
                continue
            body1_idx = body_name_to_idx.get(body1_name.replace("-", "_"))
            body2_idx = body_name_to_idx.get(body2_name.replace("-", "_"))
            if body1_idx is None or body2_idx is None:
                continue
            body1_shapes = [i for i, body_id in enumerate(builder.shape_body) if body_id == body1_idx]
            body2_shapes = [i for i, body_id in enumerate(builder.shape_body) if body_id == body2_idx]
            for shape1_idx in body1_shapes:
                for shape2_idx in body2_shapes:
                    builder.add_shape_collision_filter_pair(shape1_idx, shape2_idx)

    end_shape_count = builder.shape_count
    for i in range(start_shape_count, end_shape_count):
        for j in visual_shapes:
            builder.add_shape_collision_filter_pair(i, j)

    if not enable_self_collisions:
        for i in range(start_shape_count, end_shape_count):
            if builder.shape_body[i] < 0:
                continue
            for j in range(i + 1, end_shape_count):
                if builder.shape_body[j] < 0:
                    continue
                builder.add_shape_collision_filter_pair(i, j)


def _add_shape(
    builder: _SapSceneBuilder,
    body: int,
    cfg: dict[str, Any],
    *,
    default_cfg: dict[str, Any] | None = None,
) -> int:
    shape_type = str(cfg.get("type", "box")).lower()
    shape_cfg = _shape_config(builder, _effective_shape_cfg(cfg, default_cfg))
    common = {
        "body": body,
        "xform": _transform(cfg.get("transform")),
        "cfg": shape_cfg,
        "label": cfg.get("label"),
        "color": cfg.get("color"),
    }
    if shape_type == "box":
        return builder.add_shape(
            shape_type=int(SapGeoType.BOX),
            scale=(_float(cfg.get("hx", 0.5)), _float(cfg.get("hy", 0.5)), _float(cfg.get("hz", 0.5))),
            **common,
        )
    if shape_type == "sphere":
        return builder.add_shape(
            shape_type=int(SapGeoType.SPHERE),
            scale=(_float(cfg.get("radius", 1.0)), 0.0, 0.0),
            **common,
        )
    if shape_type == "capsule":
        return builder.add_shape(
            shape_type=int(SapGeoType.CAPSULE),
            scale=(_float(cfg.get("radius", 1.0)), _float(cfg.get("half_height", 0.5)), 0.0),
            **common,
        )
    if shape_type == "cylinder":
        return builder.add_shape(
            shape_type=int(SapGeoType.CYLINDER),
            scale=(_float(cfg.get("radius", 1.0)), _float(cfg.get("half_height", 0.5)), 0.0),
            **common,
        )
    if shape_type == "cone":
        return builder.add_shape(
            shape_type=int(SapGeoType.CONE),
            scale=(_float(cfg.get("radius", 1.0)), _float(cfg.get("half_height", 0.5)), 0.0),
            **common,
        )
    if shape_type == "ellipsoid":
        return builder.add_shape(
            shape_type=int(SapGeoType.ELLIPSOID),
            scale=(_float(cfg.get("a", 1.0)), _float(cfg.get("b", 0.75)), _float(cfg.get("c", 0.5))),
            **common,
        )
    if shape_type == "mesh":
        vertices, faces = _load_mesh_geometry(cfg)
        mesh = SapMesh(vertices, faces.reshape(-1), is_solid=bool(cfg.get("is_solid", True)))
        return builder.add_shape(
            shape_type=int(SapGeoType.MESH),
            scale=(1.0, 1.0, 1.0),
            src=mesh,
            **common,
        )
    raise SapSceneLoaderError(f"Unsupported inline shape type: {shape_type}")


def _add_inline_joint(
    builder: _SapSceneBuilder,
    cfg: dict[str, Any],
    body_ids: dict[str, int],
    *,
    label: str,
) -> int:
    joint_type = str(cfg.get("type", "")).lower()
    common = {
        "parent": _body_ref(cfg.get("parent", -1), body_ids),
        "child": _body_ref(cfg["child"], body_ids),
        "parent_xform": _transform(cfg.get("parent_xform")),
        "child_xform": _transform(cfg.get("child_xform")),
        "label": cfg.get("label", label),
        "collision_filter_parent": bool(cfg.get("collision_filter_parent", True)),
    }
    if joint_type == "fixed":
        return builder.add_joint(joint_type=_SAP_JOINT_FIXED, **common)
    if joint_type == "free":
        return builder.add_joint(
            joint_type=_SAP_JOINT_FREE,
            linear_axes=_free_joint_linear_axes(),
            angular_axes=_free_joint_angular_axes(),
            **common,
        )
    if joint_type == "revolute":
        return builder.add_joint(
            joint_type=_SAP_JOINT_REVOLUTE,
            angular_axes=[_joint_dof_config_from_cfg(cfg, default_cfg=builder.default_joint_cfg)],
            **common,
        )
    if joint_type == "prismatic":
        return builder.add_joint(
            joint_type=_SAP_JOINT_PRISMATIC,
            linear_axes=[_joint_dof_config_from_cfg(cfg, default_cfg=builder.default_joint_cfg)],
            **common,
        )
    raise SapSceneLoaderError(f"joint type={joint_type or '<missing>'}")


def _joint_dof_config_from_cfg(
    cfg: dict[str, Any],
    *,
    axis_default: Any = None,
    default_cfg: _SapJointDofConfig | None = None,
) -> _SapJointDofConfig:
    if default_cfg is None:
        default_cfg = _SapJointDofConfig(axis=wp.vec3(1.0, 0.0, 0.0))
    if axis_default is None:
        axis_default = default_cfg.axis
    return _SapJointDofConfig(
        axis=_vec3_from_any(cfg.get("axis", axis_default)),
        target_pos=_float(cfg.get("target_pos", default_cfg.target_pos)),
        target_vel=_float(cfg.get("target_vel", default_cfg.target_vel)),
        target_ke=_float(cfg.get("target_ke", default_cfg.target_ke)),
        target_kd=_float(cfg.get("target_kd", default_cfg.target_kd)),
        limit_lower=_float(cfg.get("limit_lower", default_cfg.limit_lower)),
        limit_upper=_float(cfg.get("limit_upper", default_cfg.limit_upper)),
        limit_ke=_float(cfg.get("limit_ke", default_cfg.limit_ke)),
        limit_kd=_float(cfg.get("limit_kd", default_cfg.limit_kd)),
        armature=_float(cfg.get("armature", default_cfg.armature)),
        effort_limit=_float(cfg.get("effort_limit", default_cfg.effort_limit)),
        velocity_limit=_float(cfg.get("velocity_limit", default_cfg.velocity_limit)),
        friction=_float(cfg.get("friction", default_cfg.friction)),
        actuator_mode=int(cfg.get("actuator_mode", default_cfg.actuator_mode) or 0),
    )


def _add_ground(builder: _SapSceneBuilder, config: dict[str, Any]) -> None:
    ground = config.get("ground", {"enabled": True})
    if ground is False:
        return
    if ground is True or ground is None:
        ground = {"enabled": True}
    if not bool(ground.get("enabled", True)):
        return
    builder.add_ground_plane(
        height=_float(ground.get("height", 0.0)),
        cfg=_shape_config(builder, ground.get("cfg")),
        label=ground.get("label"),
    )


def _required_joint_name(value: Any, joint_ids: dict[str, int]) -> str:
    name = str(value)
    if name not in joint_ids:
        raise SapSceneLoaderError(f"Unknown joint reference: {value!r}")
    return name


def _set_joint_q(builder: _SapSceneBuilder, entry: dict[str, Any], joint_ids: dict[str, int]) -> None:
    if "index" in entry:
        index = int(entry["index"])
    else:
        joint_name = entry.get("joint", entry.get("joint_label"))
        if joint_name in joint_ids:
            joint_index = joint_ids[_required_joint_name(joint_name, joint_ids)]
        else:
            joint_index = _joint_index_from_ref(builder, joint_name)
        index = int(builder.joint_q_start[joint_index]) + int(entry.get("offset", 0))
    builder.joint_q[index] = _float(entry["value"])


def _joint_target_mode(value: Any) -> int:
    if isinstance(value, int):
        return int(value)
    modes = {
        "none": 0,
        "position": 1,
        "velocity": 2,
        "position_velocity": 3,
        "effort": 4,
    }
    key = str(value).lower()
    if key not in modes:
        raise SapSceneLoaderError(f"Unknown joint target mode: {value!r}")
    return modes[key]


def _joint_target_mode_from_gains(
    target_ke: float,
    target_kd: float,
    force_position_velocity_actuation: bool,
    *,
    has_drive: bool,
) -> int:
    if not has_drive:
        return 0
    if force_position_velocity_actuation and target_ke != 0.0 and target_kd != 0.0:
        return 3
    if target_ke != 0.0:
        return 1
    if target_kd != 0.0:
        return 2
    return 4


def _has_joint_selector(spec: dict[str, Any]) -> bool:
    return any(name in spec for name in ("joint", "joints", "joint_label", "joint_labels", "match", "label_prefix"))


def _joint_index_from_ref(builder: _SapSceneBuilder, value: Any) -> int:
    if isinstance(value, int):
        return int(value)
    text = str(value)
    exact = [index for index, label in enumerate(builder.joint_labels) if label == text]
    if len(exact) == 1:
        return exact[0]
    suffix = [index for index, label in enumerate(builder.joint_labels) if label.endswith(f"/{text}")]
    if len(suffix) == 1:
        return suffix[0]
    matches = exact or suffix
    if matches:
        labels = [builder.joint_labels[index] for index in matches]
        raise SapSceneLoaderError(f"Ambiguous joint reference {value!r}: {labels}")
    raise SapSceneLoaderError(f"Unknown joint reference: {value!r}")


def _joint_indices_for_spec(builder: _SapSceneBuilder, spec: dict[str, Any]) -> list[int]:
    if "joint" in spec or "joint_label" in spec:
        return [_joint_index_from_ref(builder, spec.get("joint", spec.get("joint_label")))]
    if "joints" in spec or "joint_labels" in spec:
        values = spec.get("joints", spec.get("joint_labels")) or ()
        return [_joint_index_from_ref(builder, value) for value in values]
    if "label_prefix" in spec:
        prefix = str(spec["label_prefix"])
        return [index for index, label in enumerate(builder.joint_labels) if label.startswith(prefix)]
    if "match" in spec:
        match = str(spec["match"])
        return [index for index, label in enumerate(builder.joint_labels) if match in label]
    raise SapSceneLoaderError(f"Expected a joint selector in op: {spec!r}")


def _joint_q_range(builder: _SapSceneBuilder, joint_index: int) -> tuple[int, int]:
    start = int(builder.joint_q_start[joint_index])
    end = int(builder.joint_q_start[joint_index + 1]) if joint_index + 1 < len(builder.joint_q_start) else len(builder.joint_q)
    return start, end


def _joint_qd_range(builder: _SapSceneBuilder, joint_index: int) -> tuple[int, int]:
    start = int(builder.joint_qd_start[joint_index])
    end = (
        int(builder.joint_qd_start[joint_index + 1])
        if joint_index + 1 < len(builder.joint_qd_start)
        else len(builder.joint_qd)
    )
    return start, end


def _joint_dof_indices_for_spec(builder: _SapSceneBuilder, spec: dict[str, Any]) -> list[int]:
    indices: list[int] = []
    for joint_index in _joint_indices_for_spec(builder, spec):
        start, end = _joint_qd_range(builder, joint_index)
        indices.extend(range(start, end))
    return indices


def _copy_joint_q_to_joint_targets(builder: _SapSceneBuilder, spec: dict[str, Any]) -> None:
    for joint_index in _joint_indices_for_spec(builder, spec):
        q_start, q_end = _joint_q_range(builder, joint_index)
        qd_start, qd_end = _joint_qd_range(builder, joint_index)
        count = min(q_end - q_start, qd_end - qd_start)
        for offset in range(count):
            builder.joint_target_pos[qd_start + offset] = builder.joint_q[q_start + offset]


def _apply_post_build_op(
    builder: _SapSceneBuilder,
    op: dict[str, Any],
    body_ids: dict[str, int],
    unsupported: list[str],
) -> None:
    op_type = str(op.get("op", "")).lower()
    if op_type == "add_shape":
        _add_shape(builder, _body_ref(op.get("body", -1), body_ids), op)
        return
    if op_type == "set_array":
        _apply_set_array(builder, op, unsupported)
        return
    if op_type == "copy_array":
        _apply_copy_array(builder, op, unsupported)
        return
    if op_type == "scale_sphere_shapes":
        factor = _float(op.get("factor", 1.0))
        for index, shape_type in enumerate(builder.shape_type):
            if shape_type == int(SapGeoType.SPHERE):
                radius = builder.shape_scale[index][0]
                builder.shape_scale[index] = (float(radius) * factor, 0.0, 0.0)
        return
    if op_type == "approximate_meshes":
        method = str(op.get("method", "convex_hull"))
        if method != "bounding_box":
            unsupported.append(f"post_build approximate_meshes method={method}")
            return
        _approximate_meshes_with_bounding_boxes(builder)
        return
    if op_type == "set_attr":
        name = str(op.get("name", ""))
        if not name or not hasattr(builder, name):
            unsupported.append(f"post_build set_attr {name or '<missing>'}")
            return
        setattr(builder, name, _coerce_config_value(op["value"]))
        return
    if op_type == "set_joint_targets":
        try:
            if _has_joint_selector(op):
                indices = _joint_dof_indices_for_spec(builder, op)
            else:
                indices = _indices_for(builder.joint_target_ke, op)
        except SapSceneLoaderError as exc:
            unsupported.append(f"post_build set_joint_targets: {exc}")
            return
        mode = _joint_target_mode(op.get("mode", "position"))
        ke = _float(op.get("ke", 0.0))
        kd = _float(op.get("kd", 0.0))
        for index in indices:
            builder.joint_target_ke[index] = ke
            builder.joint_target_kd[index] = kd
            builder.joint_target_mode[index] = mode
        return
    if op_type == "set_joint_q":
        try:
            _set_joint_q(builder, op, {})
        except SapSceneLoaderError as exc:
            unsupported.append(f"post_build set_joint_q: {exc}")
        return
    if op_type == "copy_joint_q_to_joint_targets":
        try:
            _copy_joint_q_to_joint_targets(builder, op)
        except SapSceneLoaderError as exc:
            unsupported.append(f"post_build copy_joint_q_to_joint_targets: {exc}")
        return
    if op_type == "set_joint_armature":
        indices = _indices_for(builder.joint_armature, op)
        value = _float(op["value"])
        for index in indices:
            builder.joint_armature[index] = value
        return
    unsupported.append(f"post_build op={op_type or '<missing>'}")


_POST_BUILD_ARRAYS: dict[str, str] = {
    "shape_margin": "shape_margin",
    "shape_gap": "shape_gap",
    "shape_material_ke": "shape_material_ke",
    "shape_material_tau": "shape_material_tau",
    "shape_material_mu": "shape_material_mu",
    "shape_material_restitution": "shape_material_restitution",
    "shape_material_mu_torsional": "shape_material_mu_torsional",
    "shape_material_mu_rolling": "shape_material_mu_rolling",
    "shape_material_kh": "shape_material_kh",
    "shape_collision_group": "shape_collision_group",
    "shape_flags": "shape_flags",
    "joint_q": "joint_q",
    "joint_qd": "joint_qd",
    "joint_target_pos": "joint_target_pos",
    "joint_target_vel": "joint_target_vel",
    "joint_target_ke": "joint_target_ke",
    "joint_target_kd": "joint_target_kd",
    "joint_target_mode": "joint_target_mode",
    "joint_armature": "joint_armature",
}


def _apply_set_array(builder: _SapSceneBuilder, op: dict[str, Any], unsupported: list[str]) -> None:
    array_name = str(op.get("array", ""))
    field_name = _POST_BUILD_ARRAYS.get(array_name)
    if field_name is None:
        unsupported.append(f"post_build set_array {array_name or '<missing>'}")
        return
    array = getattr(builder, field_name)
    indices = _indices_for(array, op)
    values = _values_for(builder, op, len(indices), unsupported)
    if values is None:
        return
    for index, value in zip(indices, values, strict=True):
        array[index] = _coerce_config_value(value)
        if field_name in {"shape_flags", "shape_collision_group"}:
            array[index] = int(array[index])


def _apply_copy_array(builder: _SapSceneBuilder, op: dict[str, Any], unsupported: list[str]) -> None:
    src_name = str(op.get("src", ""))
    dst_name = str(op.get("dst", ""))
    src_field = _POST_BUILD_ARRAYS.get(src_name)
    dst_field = _POST_BUILD_ARRAYS.get(dst_name)
    if src_field is None or dst_field is None:
        unsupported.append(f"post_build copy_array {src_name or '<missing>'}->{dst_name or '<missing>'}")
        return
    src = getattr(builder, src_field)
    dst = getattr(builder, dst_field)
    src_indices = _indices_for(src, op.get("src_range", op))
    dst_indices = _indices_for(dst, op.get("dst_range", op))
    if len(src_indices) != len(dst_indices):
        raise SapSceneLoaderError(f"copy_array length mismatch: {len(src_indices)} != {len(dst_indices)}")
    for src_index, dst_index in zip(src_indices, dst_indices, strict=True):
        dst[dst_index] = src[src_index]


def _approximate_meshes_with_bounding_boxes(
    builder: _SapSceneBuilder,
    shape_indices: Sequence[int] | None = None,
) -> None:
    if shape_indices is None:
        shape_indices = [
            i
            for i, shape_type in enumerate(builder.shape_type)
            if shape_type == int(SapGeoType.MESH) and builder.shape_flags[i] & int(SapShapeFlags.COLLIDE_SHAPES)
        ]
    for shape in shape_indices:
        mesh = builder.shape_source_refs[shape]
        if mesh is None:
            continue
        scale = np.asarray(builder.shape_scale[shape], dtype=np.float32)
        vertices = mesh.vertices * scale.reshape(1, 3)
        tf, box_scale = _sap_compute_inertia_obb(vertices)
        builder.shape_type[shape] = int(SapGeoType.BOX)
        builder.shape_source_refs[shape] = None
        builder.shape_scale[shape] = (float(box_scale[0]), float(box_scale[1]), float(box_scale[2]))
        builder.shape_transform[shape] = builder.shape_transform[shape] * tf


def _values_for(
    builder: _SapSceneBuilder,
    op: dict[str, Any],
    count: int,
    unsupported: list[str],
) -> list[Any] | None:
    if "from_array" in op:
        source_name = str(op["from_array"])
        source_field = _POST_BUILD_ARRAYS.get(source_name)
        if source_field is None:
            unsupported.append(f"post_build set_array from_array {source_name}")
            return None
        src = getattr(builder, source_field)
        src_indices = _indices_for(src, op.get("from_range", op))
        if len(src_indices) != count:
            raise SapSceneLoaderError(f"set_array from_array length mismatch: {len(src_indices)} != {count}")
        return [src[index] for index in src_indices]
    if "values" in op:
        values = [_coerce_config_value(value) for value in op["values"]]
        if len(values) != count:
            raise SapSceneLoaderError(f"set_array expected {count} values, got {len(values)}")
        return values
    value = _coerce_config_value(op["value"])
    return [value for _ in range(count)]


def _indices_for(array: Any, spec: dict[str, Any]) -> list[int]:
    n = len(array)
    if "index" in spec:
        return [_normalize_index(int(spec["index"]), n)]
    range_spec = spec.get("range", "all")
    if range_spec == "all":
        return list(range(n))
    if isinstance(range_spec, str):
        if range_spec.startswith("head:"):
            return list(range(min(int(range_spec.split(":", 1)[1]), n)))
        if range_spec.startswith("tail:"):
            count = min(int(range_spec.split(":", 1)[1]), n)
            return list(range(n - count, n))
        if range_spec.startswith("from:"):
            start = _normalize_index(int(range_spec.split(":", 1)[1]), n)
            return list(range(start, n))
    if isinstance(range_spec, (list, tuple)) and len(range_spec) == 2:
        start = _normalize_index(int(range_spec[0]), n)
        end = _normalize_index(int(range_spec[1]), n) if range_spec[1] is not None else n
        return list(range(start, min(end, n)))
    raise SapSceneLoaderError(f"Unsupported range spec: {range_spec!r}")


def _normalize_index(index: int, n: int) -> int:
    return n + index if index < 0 else index


def _shape_config(builder: _SapSceneBuilder, cfg: dict[str, Any] | None) -> _SapShapeConfig:
    shape_cfg = builder.default_shape_cfg.copy()
    if cfg is not None:
        shape_cfg.apply(_normalize_shape_cfg(cfg))
    return shape_cfg


_SHAPE_CFG_ALIASES: dict[str, tuple[str, ...]] = {
    "margin": ("contact_margin", "shape_margin", "margin"),
    "gap": ("shape_gap", "gap"),
    "mu": ("mu", "shape_mu"),
    "ke": ("ke", "shape_ke"),
    "tau": ("tau", "shape_tau", "relaxation_time"),
}


def _normalize_shape_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    alias_to_name = {alias: canonical for canonical, aliases in _SHAPE_CFG_ALIASES.items() for alias in aliases}
    normalized: dict[str, Any] = {}
    for name, value in cfg.items():
        if value is None:
            continue
        normalized[alias_to_name.get(str(name), str(name))] = value
    return normalized


def _object_shape_cfg_overrides(cfg: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    overrides: dict[str, Any] = {}
    for source_name in ("physics", "shape"):
        source = cfg.get(source_name)
        if isinstance(source, dict):
            for name, value in _normalize_shape_cfg(source).items():
                if name in _SHAPE_CFG_ALIASES:
                    overrides[name] = value
    for name, value in _normalize_shape_cfg(cfg).items():
        if name in _SHAPE_CFG_ALIASES:
            overrides[name] = value
    return overrides


def _effective_shape_cfg(cfg: dict[str, Any], default_cfg: dict[str, Any] | None = None) -> dict[str, Any] | None:
    effective: dict[str, Any] = {}
    if default_cfg:
        effective.update(default_cfg)
    effective.update(_object_shape_cfg_overrides(cfg))
    raw_cfg = cfg.get("cfg")
    if isinstance(raw_cfg, dict):
        effective.update(_normalize_shape_cfg(raw_cfg))
    return effective or None


def _body_ref(value: Any, body_ids: dict[str, int]) -> int:
    if value in (-1, "world", None):
        return -1
    if isinstance(value, int):
        return int(value)
    if value not in body_ids:
        raise SapSceneLoaderError(f"Unknown body reference: {value}")
    return body_ids[value]


def _required_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise SapSceneLoaderError(f"Expected non-empty string field {key!r}.")
    return value


def _compute_sap_world_offsets(
    world_count: int,
    spacing: tuple[float, float, float],
    up_axis: int = 2,
) -> np.ndarray:
    if world_count <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    spacing_np = np.asarray(spacing, dtype=np.float32)
    nonzeros = np.nonzero(spacing_np)[0]
    dim_count = int(nonzeros.shape[0])
    if dim_count == 0:
        offsets = np.zeros((world_count, 3), dtype=np.float32)
    else:
        side_length = int(np.ceil(world_count ** (1.0 / dim_count)))
        offsets = np.zeros((world_count, 3), dtype=np.float32)
        if dim_count == 1:
            for i in range(world_count):
                offsets[i] = i * spacing_np
        elif dim_count == 2:
            for i in range(world_count):
                offsets[i, nonzeros[0]] = (i % side_length) * spacing_np[nonzeros[0]]
                offsets[i, nonzeros[1]] = (i // side_length) * spacing_np[nonzeros[1]]
        else:
            for i in range(world_count):
                offsets[i, 0] = (i % side_length) * spacing_np[0]
                offsets[i, 1] = ((i // side_length) % side_length) * spacing_np[1]
                offsets[i, 2] = (i // (side_length * side_length)) * spacing_np[2]

    min_offsets = np.min(offsets, axis=0)
    correction = min_offsets + (np.max(offsets, axis=0) - min_offsets) / 2.0
    correction[int(up_axis)] = 0.0
    return offsets - correction


def _offset_transform(xform: wp.transform, offset: Any) -> wp.transform:
    return wp.transform(
        wp.vec3(
            float(xform.p[0]) + float(offset[0]),
            float(xform.p[1]) + float(offset[1]),
            float(xform.p[2]) + float(offset[2]),
        ),
        xform.q,
    )


def _transform(spec: Any) -> wp.transform | None:
    if spec is None:
        return None
    if spec == "identity":
        return wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity())
    return wp.transform(p=_vec3(spec.get("p", (0.0, 0.0, 0.0))), q=_quat(spec.get("q", "identity")))


def _as_transform(value: Any) -> wp.transform:
    if value is None:
        return wp.transform()
    if isinstance(value, wp.transform):
        return value
    return wp.transform(*value)


def _quat(spec: Any) -> wp.quat:
    if spec is None or spec == "identity":
        return wp.quat_identity()
    if isinstance(spec, (list, tuple)):
        if len(spec) != 4:
            raise SapSceneLoaderError("Quaternion lists must be [x, y, z, w].")
        return wp.quat(*(_float(v) for v in spec))
    if isinstance(spec, dict) and "axis_angle" in spec:
        axis_angle = spec["axis_angle"]
        return wp.quat_from_axis_angle(_vec3(axis_angle["axis"]), _float(axis_angle["angle"]))
    raise SapSceneLoaderError(f"Unsupported quaternion spec: {spec}")


def _vec3(value: Any) -> wp.vec3:
    x, y, z = _tuple3(value)
    return wp.vec3(x, y, z)


def _tuple3(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SapSceneLoaderError(f"Expected a 3-vector, got {value!r}.")
    return (_float(value[0]), _float(value[1]), _float(value[2]))


def _coerce_config_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, str):
        try:
            return _float(value)
        except ValueError:
            return value
    return value


def _float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(_eval_number_expr(value))
    raise SapSceneLoaderError(f"Expected numeric value, got {value!r}.")


def _eval_number_expr(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return math.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError(f"Unsupported numeric expression: {expr!r}")

    return visit(tree)


def _test_group_pair(group_a: int, group_b: int) -> bool:
    if group_a == 0 or group_b == 0:
        return False
    if group_a > 0:
        return group_a == group_b or group_b < 0
    if group_a < 0:
        return group_a != group_b
    return False


def _test_world_and_group_pair(world_a: int, world_b: int, group_a: int, group_b: int) -> bool:
    if world_a != -1 and world_b != -1 and world_a != world_b:
        return False
    return _test_group_pair(group_a, group_b)


def _compute_shape_radius(shape_type: int, scale: tuple[float, float, float], src: Any) -> float:
    if shape_type == int(SapGeoType.SPHERE):
        return float(scale[0])
    if shape_type == int(SapGeoType.BOX):
        return float(np.linalg.norm(scale))
    if shape_type in (int(SapGeoType.CAPSULE), int(SapGeoType.CYLINDER), int(SapGeoType.CONE)):
        return float(scale[0] + scale[1])
    if shape_type == int(SapGeoType.ELLIPSOID):
        return float(max(scale[0], scale[1], scale[2]))
    if shape_type in (int(SapGeoType.MESH), int(SapGeoType.CONVEX_MESH)) and src is not None:
        vmax = np.max(np.abs(src.vertices), axis=0) * np.max(scale)
        return float(np.linalg.norm(vmax))
    if shape_type == int(SapGeoType.PLANE):
        if scale[0] > 0.0 and scale[1] > 0.0:
            return float(np.linalg.norm(scale))
        return 1.0e6
    return 10.0


def _compute_local_aabb(shape_type: int, src: Any, scale: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    if shape_type in (int(SapGeoType.MESH), int(SapGeoType.CONVEX_MESH)) and src is not None:
        vertices = src.vertices
        lower = vertices.min(axis=0) * np.asarray(scale, dtype=np.float32)
        upper = vertices.max(axis=0) * np.asarray(scale, dtype=np.float32)
        return lower, upper
    if shape_type == int(SapGeoType.ELLIPSOID):
        sx, sy, sz = scale
        return np.asarray([-sx, -sy, -sz]), np.asarray([sx, sy, sz])
    if shape_type == int(SapGeoType.BOX):
        hx, hy, hz = scale
        return np.asarray([-hx, -hy, -hz]), np.asarray([hx, hy, hz])
    if shape_type == int(SapGeoType.SPHERE):
        r = scale[0]
        return np.asarray([-r, -r, -r]), np.asarray([r, r, r])
    if shape_type == int(SapGeoType.CAPSULE):
        r, half_height, _ = scale
        return np.asarray([-r, -r, -half_height - r]), np.asarray([r, r, half_height + r])
    if shape_type in (int(SapGeoType.CYLINDER), int(SapGeoType.CONE)):
        r, half_height, _ = scale
        return np.asarray([-r, -r, -half_height]), np.asarray([r, r, half_height])
    return np.asarray([-1.0, -1.0, -1.0]), np.asarray([1.0, 1.0, 1.0])


def _sap_compute_inertia_sphere(density: float, radius: float) -> tuple[float, wp.vec3, wp.mat33]:
    volume = 4.0 / 3.0 * wp.pi * radius * radius * radius
    mass = density * volume
    inertia_axis = 2.0 / 5.0 * mass * radius * radius
    return mass, wp.vec3(), wp.mat33([[inertia_axis, 0.0, 0.0], [0.0, inertia_axis, 0.0], [0.0, 0.0, inertia_axis]])


def _sap_compute_inertia_capsule(density: float, radius: float, half_height: float) -> tuple[float, wp.vec3, wp.mat33]:
    height = 2.0 * half_height
    sphere_mass = density * (4.0 / 3.0) * wp.pi * radius * radius * radius
    cylinder_mass = density * wp.pi * radius * radius * height
    mass = sphere_mass + cylinder_mass
    inertia_xy = cylinder_mass * (0.25 * radius * radius + (1.0 / 12.0) * height * height) + sphere_mass * (
        0.4 * radius * radius + 0.375 * radius * height + 0.25 * height * height
    )
    inertia_z = (cylinder_mass * 0.5 + sphere_mass * 0.4) * radius * radius
    return mass, wp.vec3(), wp.mat33([[inertia_xy, 0.0, 0.0], [0.0, inertia_xy, 0.0], [0.0, 0.0, inertia_z]])


def _sap_compute_inertia_cylinder(
    density: float,
    radius: float,
    half_height: float,
) -> tuple[float, wp.vec3, wp.mat33]:
    height = 2.0 * half_height
    mass = density * wp.pi * radius * radius * height
    inertia_xy = 1.0 / 12.0 * mass * (3.0 * radius * radius + height * height)
    inertia_z = 0.5 * mass * radius * radius
    return mass, wp.vec3(), wp.mat33([[inertia_xy, 0.0, 0.0], [0.0, inertia_xy, 0.0], [0.0, 0.0, inertia_z]])


def _sap_compute_inertia_cone(density: float, radius: float, half_height: float) -> tuple[float, wp.vec3, wp.mat33]:
    height = 2.0 * half_height
    mass = density * wp.pi * radius * radius * height / 3.0
    com = wp.vec3(0.0, 0.0, -height / 4.0)
    inertia_xy = 3.0 / 20.0 * mass * radius * radius + 3.0 / 80.0 * mass * height * height
    inertia_z = 3.0 / 10.0 * mass * radius * radius
    return mass, com, wp.mat33([[inertia_xy, 0.0, 0.0], [0.0, inertia_xy, 0.0], [0.0, 0.0, inertia_z]])


def _sap_compute_inertia_ellipsoid(
    density: float,
    rx: float,
    ry: float,
    rz: float,
) -> tuple[float, wp.vec3, wp.mat33]:
    volume = 4.0 / 3.0 * wp.pi * rx * ry * rz
    mass = density * volume
    ixx = 1.0 / 5.0 * mass * (ry * ry + rz * rz)
    iyy = 1.0 / 5.0 * mass * (rx * rx + rz * rz)
    izz = 1.0 / 5.0 * mass * (rx * rx + ry * ry)
    return mass, wp.vec3(), wp.mat33([[ixx, 0.0, 0.0], [0.0, iyy, 0.0], [0.0, 0.0, izz]])


def _sap_compute_inertia_box_from_mass(mass: float, hx: float, hy: float, hz: float) -> wp.mat33:
    ixx = 1.0 / 3.0 * mass * (hy * hy + hz * hz)
    iyy = 1.0 / 3.0 * mass * (hx * hx + hz * hz)
    izz = 1.0 / 3.0 * mass * (hx * hx + hy * hy)
    return wp.mat33([[ixx, 0.0, 0.0], [0.0, iyy, 0.0], [0.0, 0.0, izz]])


def _sap_compute_inertia_box(density: float, hx: float, hy: float, hz: float) -> tuple[float, wp.vec3, wp.mat33]:
    mass = density * 8.0 * hx * hy * hz
    return mass, wp.vec3(), _sap_compute_inertia_box_from_mass(mass, hx, hy, hz)


def _sap_transform_inertia(mass: float, inertia: wp.mat33, offset: wp.vec3, quat: wp.quat) -> wp.mat33:
    rot = wp.quat_to_matrix(quat)
    return rot @ inertia @ wp.transpose(rot) + mass * (
        wp.dot(offset, offset) * wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        - wp.outer(offset, offset)
    )


def _sap_shell_inertia(
    solid: tuple[float, wp.vec3, wp.mat33],
    hollow: tuple[float, wp.vec3, wp.mat33],
) -> tuple[float, wp.vec3, wp.mat33]:
    return solid[0] - hollow[0], solid[1], solid[2] - hollow[2]


def _sap_compute_inertia_shape(
    shape_type: int,
    scale: tuple[float, float, float],
    src: Any,
    density: float,
    is_solid: bool = True,
    thickness: float = 0.001,
) -> tuple[float, wp.vec3, wp.mat33]:
    if density == 0.0 or shape_type == int(SapGeoType.PLANE):
        return 0.0, wp.vec3(), wp.mat33()

    if shape_type == int(SapGeoType.SPHERE):
        solid = _sap_compute_inertia_sphere(density, scale[0])
        if is_solid:
            return solid
        return _sap_shell_inertia(solid, _sap_compute_inertia_sphere(density, scale[0] - thickness))

    if shape_type == int(SapGeoType.BOX):
        solid = _sap_compute_inertia_box(density, scale[0], scale[1], scale[2])
        if is_solid:
            return solid
        return _sap_shell_inertia(
            solid,
            _sap_compute_inertia_box(density, scale[0] - thickness, scale[1] - thickness, scale[2] - thickness),
        )

    if shape_type == int(SapGeoType.CAPSULE):
        solid = _sap_compute_inertia_capsule(density, scale[0], scale[1])
        if is_solid:
            return solid
        return _sap_shell_inertia(solid, _sap_compute_inertia_capsule(density, scale[0] - thickness, scale[1] - thickness))

    if shape_type == int(SapGeoType.CYLINDER):
        solid = _sap_compute_inertia_cylinder(density, scale[0], scale[1])
        if is_solid:
            return solid
        return _sap_shell_inertia(
            solid,
            _sap_compute_inertia_cylinder(density, scale[0] - thickness, scale[1] - thickness),
        )

    if shape_type == int(SapGeoType.CONE):
        solid = _sap_compute_inertia_cone(density, scale[0], scale[1])
        if is_solid:
            return solid
        hollow = _sap_compute_inertia_cone(density, scale[0] - thickness, scale[1] - thickness)
        shell_mass = solid[0] - hollow[0]
        if shell_mass <= 0.0:
            raise SapSceneLoaderError(f"Hollow cone shell has non-positive mass ({shell_mass:.6g}).")
        solid_com = np.asarray(solid[1], dtype=np.float64)
        hollow_com = np.asarray(hollow[1], dtype=np.float64)
        shell_com = (solid[0] * solid_com - hollow[0] * hollow_com) / shell_mass

        def shift(mass: float, inertia_value: wp.mat33, source_com: np.ndarray, target_com: np.ndarray) -> np.ndarray:
            offset = target_com - source_com
            return np.asarray(inertia_value, dtype=np.float64).reshape(3, 3) + mass * (
                np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
            )

        inertia = shift(solid[0], solid[2], solid_com, shell_com) - shift(hollow[0], hollow[2], hollow_com, shell_com)
        return shell_mass, wp.vec3(*shell_com), wp.mat33(*inertia.flatten())

    if shape_type == int(SapGeoType.ELLIPSOID):
        solid = _sap_compute_inertia_ellipsoid(density, scale[0], scale[1], scale[2])
        if is_solid:
            return solid
        return _sap_shell_inertia(
            solid,
            _sap_compute_inertia_ellipsoid(
                density,
                scale[0] - thickness,
                scale[1] - thickness,
                scale[2] - thickness,
            ),
        )

    if shape_type == int(SapGeoType.HFIELD):
        return 0.0, wp.vec3(), wp.mat33()

    if shape_type in (int(SapGeoType.MESH), int(SapGeoType.CONVEX_MESH)):
        if src is None:
            raise SapSceneLoaderError("mesh inertia requires shape source")
        if getattr(src, "has_inertia", False) and src.mass > 0.0 and bool(src.is_solid) == bool(is_solid):
            sx, sy, sz = scale
            mass_ratio = sx * sy * sz * density
            mass = float(src.mass) * mass_ratio
            com = wp.vec3(float(src.com[0]) * sx, float(src.com[1]) * sy, float(src.com[2]) * sz)
            inertia = src.inertia
            ixx = inertia[0, 0] * (sy**2 + sz**2) / 2.0 * mass_ratio
            iyy = inertia[1, 1] * (sx**2 + sz**2) / 2.0 * mass_ratio
            izz = inertia[2, 2] * (sx**2 + sy**2) / 2.0 * mass_ratio
            ixy = inertia[0, 1] * sx * sy * mass_ratio
            ixz = inertia[0, 2] * sx * sz * mass_ratio
            iyz = inertia[1, 2] * sy * sz * mass_ratio
            return mass, com, wp.mat33([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])

        vertices = np.asarray(src.vertices, dtype=np.float32) * np.asarray(scale, dtype=np.float32).reshape(1, 3)
        mass, com, inertia, _ = _sap_compute_inertia_mesh(density, vertices, src.indices, is_solid, thickness)
        return mass, com, inertia

    raise SapSceneLoaderError(f"Unsupported inertia shape type: {shape_type}")


def _compute_voxel_resolution_from_aabb(
    aabb_lower: np.ndarray,
    aabb_upper: np.ndarray,
    voxel_budget: int,
) -> tuple[int, int, int]:
    size = np.maximum(aabb_upper - aabb_lower, 1.0e-6)
    volume = size[0] * size[1] * size[2]
    voxel_size = max(float((volume / voxel_budget) ** (1.0 / 3.0)), 1.0e-6)
    nx = max(1, round(float(size[0] / voxel_size)))
    ny = max(1, round(float(size[1] / voxel_size)))
    nz = max(1, round(float(size[2] / voxel_size)))
    while nx * ny * nz > voxel_budget:
        if nx >= ny and nx >= nz and nx > 1:
            nx -= 1
        elif ny >= nz and ny > 1:
            ny -= 1
        elif nz > 1:
            nz -= 1
        else:
            break
    return int(nx), int(ny), int(nz)


def _load_mesh_geometry(cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Mesh scene shapes require trimesh; install project dependencies with uv.") from exc

    path = Path(str(cfg["path"]))
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.suffix.lower() == ".urdf":
        vertices, faces = _load_urdf_mesh_geometry(path)
    else:
        mesh = trimesh.load(str(path), force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise SapSceneLoaderError(f"Mesh shape must be triangular XYZ geometry, got {vertices.shape=} {faces.shape=}")
    scale = cfg.get("mesh_scale", cfg.get("scale", 1.0))
    if isinstance(scale, (int, float)):
        scale_values = np.asarray([float(scale), float(scale), float(scale)], dtype=np.float32)
    else:
        scale_values = np.asarray([float(value) for value in scale], dtype=np.float32).reshape(3)
    vertices = vertices * scale_values.reshape(1, 3)
    axes = cfg.get("axis_order")
    if axes is not None:
        axis_indices = [int(axis) for axis in axes]
        if len(axis_indices) != 3:
            raise SapSceneLoaderError(f"axis_order must contain three indices, got {axes!r}")
        vertices = vertices[:, axis_indices]
    face_order = cfg.get("face_order")
    if face_order is not None:
        order = [int(axis) for axis in face_order]
        if len(order) != 3:
            raise SapSceneLoaderError(f"face_order must contain three indices, got {face_order!r}")
        faces = faces[:, order]
    return vertices.astype(np.float32, copy=False), faces.astype(np.int32, copy=False)


def _load_urdf_mesh_geometry(urdf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    node = root.find("./link/collision")
    if node is None:
        node = root.find("./link/visual")
    if node is None:
        raise SapSceneLoaderError(f"URDF has no collision or visual mesh: {urdf_path}")

    origin = node.find("origin")
    offset = _parse_xyz(origin.get("xyz") if origin is not None else None)
    rotation = _rotation_matrix_from_rpy(_parse_xyz(origin.get("rpy") if origin is not None else None))

    mesh_node = node.find("./geometry/mesh")
    if mesh_node is None:
        raise SapSceneLoaderError(f"URDF collision/visual geometry is not a mesh: {urdf_path}")
    mesh_path = Path(str(mesh_node.get("filename", ""))).expanduser()
    if not mesh_path.is_absolute():
        mesh_path = (urdf_path.parent / mesh_path).resolve()
    scale = _parse_xyz(mesh_node.get("scale"), default=1.0)

    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Mesh scene shapes require trimesh; install project dependencies with uv.") from exc

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale.reshape(1, 3)
    vertices = (vertices @ rotation.T) + offset.reshape(1, 3)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return vertices.astype(np.float32, copy=False), faces.astype(np.int32, copy=False)


def _resolve_urdf_asset(filename: str | None, source: str, source_base: Path | None) -> str | None:
    if filename is None:
        return None
    if filename.startswith(("package://", "model://")):
        if source_base is None:
            return None
        if filename.startswith("package://"):
            stripped = filename.replace("package://", "")
            package_name = stripped.split("/", 1)[0]
            source_parts = source_base.resolve().parts
            if package_name in source_parts:
                package_index = source_parts.index(package_name)
                candidate = Path(*source_parts[:package_index]) / stripped
                return str(candidate) if candidate.exists() else None
        return None
    if filename.startswith(("http://", "https://")):
        return None
    path = Path(filename).expanduser()
    if not path.is_absolute():
        if source_base is None or not Path(source).exists():
            return None
        path = source_base / path
    return str(path) if path.exists() else None


def _parse_urdf_material_table(
    urdf_root: ET.Element,
    source_base: Path | None,
) -> dict[str, dict[str, Any]]:
    materials: dict[str, dict[str, Any]] = {}
    for material in urdf_root.findall("material"):
        mat_name = material.get("name")
        if not mat_name:
            continue
        color, texture = _parse_urdf_material_properties(material, source_base)
        materials[mat_name] = {"color": color, "texture": texture}
    return materials


def _parse_urdf_material_properties(
    material_element: ET.Element | None,
    source_base: Path | None,
) -> tuple[tuple[float, float, float] | None, str | None]:
    if material_element is None:
        return None, None

    color = None
    texture = None
    color_el = material_element.find("color")
    if color_el is not None:
        rgba = color_el.get("rgba")
        if rgba:
            values = np.fromstring(rgba, sep=" ", dtype=np.float32)
            if len(values) >= 3:
                color = (float(values[0]), float(values[1]), float(values[2]))

    texture_el = material_element.find("texture")
    if texture_el is not None:
        texture_name = texture_el.get("filename")
        if texture_name:
            path = Path(texture_name).expanduser()
            if not path.is_absolute() and source_base is not None:
                path = source_base / path
            if path.exists():
                texture = str(path)
    return color, texture


def _normalize_color(color: Any) -> tuple[float, float, float] | None:
    if color is None:
        return None
    values = np.asarray(color, dtype=np.float32).flatten()
    if values.size < 3:
        return None
    if np.max(values) > 1.0:
        values = values / 255.0
    return (float(values[0]), float(values[1]), float(values[2]))


def _extract_trimesh_texture(visual_or_material: Any, base_dir: str) -> np.ndarray | str | None:
    material = getattr(visual_or_material, "material", visual_or_material)
    if material is None:
        return None

    image = getattr(material, "image", None)
    image_path = getattr(material, "image_path", None)
    if image is None:
        base_color_texture = getattr(material, "baseColorTexture", None)
        if base_color_texture is not None:
            image = getattr(base_color_texture, "image", None)
            image_path = image_path or getattr(base_color_texture, "image_path", None)

    if image is not None:
        try:
            return np.asarray(image)
        except Exception:
            pass

    if image_path:
        if not os.path.isabs(image_path):
            image_path = os.path.abspath(os.path.join(base_dir, image_path))
        return image_path
    return None


def _extract_trimesh_material_params(
    material: Any,
) -> tuple[float | None, float | None, tuple[float, float, float] | None]:
    if material is None:
        return None, None, None

    base_color = None
    metallic = None
    roughness = None
    for candidate in (
        getattr(material, "baseColorFactor", None),
        getattr(material, "diffuse", None),
        getattr(material, "diffuseColor", None),
    ):
        if candidate is not None:
            base_color = _normalize_color(candidate)
            break

    for attr_name in ("metallicFactor", "metallic"):
        value = getattr(material, attr_name, None)
        if value is not None:
            metallic = float(value)
            break

    for attr_name in ("roughnessFactor", "roughness"):
        value = getattr(material, attr_name, None)
        if value is not None:
            roughness = float(value)
            break

    if roughness is None:
        for attr_name in ("glossiness", "shininess"):
            value = getattr(material, attr_name, None)
            if value is not None:
                gloss = float(value)
                if attr_name == "shininess":
                    gloss = min(max(gloss / 1000.0, 0.0), 1.0)
                roughness = 1.0 - min(max(gloss, 0.0), 1.0)
                break
    return roughness, metallic, base_color


def _compute_vertex_normals_np(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    normals = np.zeros_like(vertices, dtype=np.float32)
    if len(vertices) == 0 or len(faces) == 0:
        return normals
    face_normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1.0e-8)


def _smooth_vertex_normals_by_position(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    eps: float = 1.0e-6,
) -> np.ndarray:
    normals = _compute_vertex_normals_np(mesh_vertices, mesh_faces)
    if len(mesh_vertices) == 0:
        return normals
    keys = np.round(mesh_vertices / eps).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    accum = np.zeros((len(unique_keys), 3), dtype=np.float32)
    np.add.at(accum, inverse, normals)
    lengths = np.linalg.norm(accum, axis=1, keepdims=True)
    accum = accum / np.maximum(lengths, 1.0e-8)
    return accum[inverse]


def _load_sap_meshes_from_file(
    filename: str,
    *,
    scale: np.ndarray | list[float] | tuple[float, ...] = (1.0, 1.0, 1.0),
    maxhullvert: int,
    override_color: np.ndarray | list[float] | tuple[float, float, float] | None = None,
    override_texture: np.ndarray | str | None = None,
) -> list[SapMesh]:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("URDF mesh assets require trimesh; install project dependencies with uv.") from exc

    filename = os.fspath(filename)
    scale = np.asarray(scale, dtype=np.float32)
    base_dir = os.path.dirname(filename)

    def parse_dae_material_colors(
        path: str,
    ) -> tuple[list[str], dict[str, dict[str, float | str | tuple[float, float, float] | None]]]:
        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except Exception:
            return [], {}

        def strip(tag: str) -> str:
            return tag.split("}", 1)[-1] if "}" in tag else tag

        image_paths: dict[str, str] = {}
        for image in root.iter():
            if strip(image.tag) != "image":
                continue
            image_id = image.attrib.get("id") or image.attrib.get("name")
            init_from = None
            for child in image:
                if strip(child.tag) == "init_from" and child.text:
                    init_from = child.text.strip()
                    break
            if not image_id or not init_from:
                continue
            image_path = Path(init_from)
            if not image_path.is_absolute():
                image_path = Path(path).parent / image_path
            image_paths[image_id] = os.fspath(image_path)

        effect_props: dict[str, dict[str, float | str | tuple[float, float, float] | None]] = {}
        for effect in root.iter():
            if strip(effect.tag) != "effect":
                continue
            effect_id = effect.attrib.get("id")
            if not effect_id:
                continue
            surface_images: dict[str, str] = {}
            sampler_surfaces: dict[str, str] = {}
            for newparam in effect.iter():
                if strip(newparam.tag) != "newparam":
                    continue
                sid = newparam.attrib.get("sid")
                if not sid:
                    continue
                for node in newparam.iter():
                    tag = strip(node.tag)
                    if tag == "surface":
                        for child in node:
                            if strip(child.tag) == "init_from" and child.text:
                                surface_images[sid] = child.text.strip()
                                break
                    elif tag == "sampler2D":
                        for child in node:
                            if strip(child.tag) == "source" and child.text:
                                sampler_surfaces[sid] = child.text.strip()
                                break
            diffuse_color = None
            diffuse_texture = None
            specular_color = None
            specular_intensity = None
            shininess = None
            for shader_tag in ("phong", "lambert", "blinn"):
                shader = None
                for elem in effect.iter():
                    if strip(elem.tag) == shader_tag:
                        shader = elem
                        break
                if shader is None:
                    continue
                for node in shader.iter():
                    tag = strip(node.tag)
                    if tag == "diffuse":
                        for col in node.iter():
                            if strip(col.tag) == "color" and col.text:
                                values = [float(x) for x in col.text.strip().split()]
                                if len(values) >= 3:
                                    diffuse = np.clip(values[:3], 0.0, 1.0)
                                    srgb = np.power(diffuse, 1.0 / 2.2)
                                    diffuse_color = (float(srgb[0]), float(srgb[1]), float(srgb[2]))
                                    break
                            if strip(col.tag) == "texture":
                                sampler = col.attrib.get("texture")
                                surface = sampler_surfaces.get(str(sampler), str(sampler))
                                image_id = surface_images.get(surface, surface)
                                diffuse_texture = image_paths.get(image_id)
                        continue
                    if tag == "specular":
                        for col in node.iter():
                            if strip(col.tag) == "color" and col.text:
                                values = [float(x) for x in col.text.strip().split()]
                                if len(values) >= 3:
                                    specular_color = (values[0], values[1], values[2])
                                    break
                        continue
                    if tag == "reflectivity":
                        for val in node.iter():
                            if strip(val.tag) == "float" and val.text:
                                try:
                                    specular_intensity = float(val.text.strip())
                                except ValueError:
                                    specular_intensity = None
                                break
                        continue
                    if tag == "shininess":
                        for val in node.iter():
                            if strip(val.tag) == "float" and val.text:
                                try:
                                    shininess = float(val.text.strip())
                                except ValueError:
                                    shininess = None
                                break
                        continue
                if diffuse_color is not None or diffuse_texture is not None:
                    break
            metallic = None
            if specular_color is not None:
                metallic = float(np.clip(np.max(specular_color), 0.0, 1.0))
            elif specular_intensity is not None:
                metallic = float(np.clip(specular_intensity, 0.0, 1.0))
            roughness = None
            if shininess is not None:
                if shininess > 1.0:
                    shininess = min(shininess / 128.0, 1.0)
                roughness = float(np.clip(1.0 - shininess, 0.0, 1.0))
            if diffuse_color is not None or diffuse_texture is not None:
                effect_props[effect_id] = {
                    "color": diffuse_color,
                    "texture": diffuse_texture,
                    "metallic": metallic,
                    "roughness": roughness,
                }

        material_colors: dict[str, dict[str, float | str | tuple[float, float, float] | None]] = {}
        for material in root.iter():
            if strip(material.tag) != "material":
                continue
            mat_id = material.attrib.get("id") or material.attrib.get("name")
            effect_url = None
            for inst in material.iter():
                if strip(inst.tag) == "instance_effect":
                    effect_url = inst.attrib.get("url")
                    break
            if mat_id and effect_url and effect_url.startswith("#"):
                effect_id = effect_url[1:]
                if effect_id in effect_props:
                    material_colors[mat_id] = effect_props[effect_id]

        face_materials: list[str] = []
        for triangles in root.iter():
            if strip(triangles.tag) != "triangles":
                continue
            mat = triangles.attrib.get("material")
            count = triangles.attrib.get("count")
            if not mat or count is None:
                continue
            try:
                tri_count = int(count)
            except ValueError:
                continue
            face_materials.extend([mat] * tri_count)
        return face_materials, material_colors

    dae_face_materials: list[str] = []
    dae_material_colors: dict[str, dict[str, float | str | tuple[float, float, float] | None]] = {}
    if filename.lower().endswith(".dae"):
        dae_face_materials, dae_material_colors = parse_dae_material_colors(filename)

    tri = trimesh.load(filename, force="mesh")
    tri_meshes = tri.geometry.values() if hasattr(tri, "geometry") else [tri]

    meshes: list[SapMesh] = []
    for tri_mesh in tri_meshes:
        vertices = np.asarray(tri_mesh.vertices, dtype=np.float32) * scale
        faces = np.asarray(tri_mesh.faces, dtype=np.int32)
        normals = (
            np.asarray(tri_mesh.vertex_normals, dtype=np.float32)
            if getattr(tri_mesh, "vertex_normals", None) is not None
            else None
        )
        if normals is None or not np.isfinite(normals).all() or np.allclose(normals, 0.0):
            normals = _compute_vertex_normals_np(vertices, faces)

        uvs = None
        if hasattr(tri_mesh, "visual") and getattr(tri_mesh.visual, "uv", None) is not None:
            uvs = np.asarray(tri_mesh.visual.uv, dtype=np.float32)

        color = _normalize_color(override_color) if override_color is not None else None
        texture = override_texture

        def add_mesh_from_faces(
            face_indices: np.ndarray,
            *,
            mat_color: tuple[float, float, float] | None = None,
            mat_roughness: float | None = None,
            mat_metallic: float | None = None,
            mesh_vertices: np.ndarray,
            mesh_normals: np.ndarray | None = None,
            mesh_uvs: np.ndarray | None = None,
            mesh_texture: np.ndarray | str | None = None,
        ) -> None:
            used = np.unique(face_indices.flatten())
            remap = {int(old): i for i, old in enumerate(used)}
            remapped_faces = np.vectorize(remap.get)(face_indices).astype(np.int32)
            sub_vertices = mesh_vertices[used]
            sub_normals = mesh_normals[used] if mesh_normals is not None else None
            force_smooth = bool((mat_metallic is not None and mat_metallic > 0.0) or (mat_roughness is not None and mat_roughness < 0.6))
            if sub_normals is None or force_smooth:
                sub_normals = _smooth_vertex_normals_by_position(sub_vertices, remapped_faces)
            sub_uvs = mesh_uvs[used] if mesh_uvs is not None else None
            meshes.append(
                SapMesh(
                    sub_vertices,
                    remapped_faces.flatten(),
                    normals=sub_normals,
                    uvs=sub_uvs,
                    maxhullvert=maxhullvert,
                    color=mat_color,
                    texture=mesh_texture,
                    roughness=mat_roughness,
                    metallic=mat_metallic,
                )
            )

        if color is not None or texture is not None:
            add_mesh_from_faces(
                faces,
                mat_color=color,
                mesh_vertices=vertices,
                mesh_normals=normals,
                mesh_uvs=uvs,
                mesh_texture=texture,
            )
            continue

        face_materials = getattr(tri_mesh.visual, "face_materials", None) if hasattr(tri_mesh, "visual") else None
        materials = getattr(tri_mesh.visual, "materials", None) if hasattr(tri_mesh, "visual") else None
        if face_materials is not None and materials is not None:
            face_materials = np.asarray(face_materials, dtype=np.int32).flatten()
            for mat_index in np.unique(face_materials):
                mat_faces = faces[face_materials == mat_index]
                material = materials[int(mat_index)] if int(mat_index) < len(materials) else None
                roughness, metallic, base_color = _extract_trimesh_material_params(material)
                mat_color = base_color
                mat_texture = _extract_trimesh_texture(material, base_dir)
                if mat_color is None and hasattr(tri_mesh.visual, "main_color"):
                    mat_color = _normalize_color(tri_mesh.visual.main_color)
                add_mesh_from_faces(
                    mat_faces,
                    mat_color=mat_color,
                    mat_roughness=roughness,
                    mat_metallic=metallic,
                    mesh_vertices=vertices,
                    mesh_normals=normals,
                    mesh_uvs=uvs,
                    mesh_texture=mat_texture,
                )
            continue

        if dae_face_materials and len(dae_face_materials) == len(faces):
            face_materials = np.asarray(dae_face_materials, dtype=object)
            for mat_name in np.unique(face_materials):
                mat_faces = faces[face_materials == mat_name]
                mat_props = dae_material_colors.get(str(mat_name), {})
                add_mesh_from_faces(
                    mat_faces,
                    mat_color=mat_props.get("color"),
                    mat_roughness=mat_props.get("roughness"),
                    mat_metallic=mat_props.get("metallic"),
                    mesh_vertices=vertices,
                    mesh_normals=normals,
                    mesh_uvs=uvs,
                    mesh_texture=mat_props.get("texture"),
                )
            continue

        face_colors = getattr(tri_mesh.visual, "face_colors", None) if hasattr(tri_mesh, "visual") else None
        if face_colors is not None:
            face_colors = np.asarray(face_colors, dtype=np.float32)
            if face_colors.shape[0] == faces.shape[0]:
                if np.max(face_colors) > 1.0:
                    face_colors = face_colors / 255.0
                rgb = np.round(face_colors[:, :3], 4)
                unique_colors, inverse = np.unique(rgb, axis=0, return_inverse=True)
                for color_idx, mat_color in enumerate(unique_colors):
                    mat_faces = faces[inverse == color_idx]
                    add_mesh_from_faces(
                        mat_faces,
                        mat_color=(float(mat_color[0]), float(mat_color[1]), float(mat_color[2])),
                        mesh_vertices=vertices,
                        mesh_normals=normals,
                        mesh_uvs=uvs,
                        mesh_texture=texture,
                    )
                continue

        vertex_colors = getattr(tri_mesh.visual, "vertex_colors", None) if hasattr(tri_mesh, "visual") else None
        if vertex_colors is not None:
            vertex_colors = np.asarray(vertex_colors, dtype=np.float32)
            if vertex_colors.size and np.max(vertex_colors) > 1.0:
                vertex_colors = vertex_colors / 255.0
            if vertex_colors.shape[0] == vertices.shape[0]:
                rgb = vertex_colors[:, :3]
                face_rgb = np.round(rgb[faces].mean(axis=1), 4)
                unique_colors, inverse = np.unique(face_rgb, axis=0, return_inverse=True)
                for color_idx, mat_color in enumerate(unique_colors):
                    mat_faces = faces[inverse == color_idx]
                    add_mesh_from_faces(
                        mat_faces,
                        mat_color=(float(mat_color[0]), float(mat_color[1]), float(mat_color[2])),
                        mesh_vertices=vertices,
                        mesh_normals=normals,
                        mesh_uvs=uvs,
                        mesh_texture=texture,
                    )
                continue

        roughness = None
        metallic = None
        if color is None and hasattr(tri_mesh, "visual") and hasattr(tri_mesh.visual, "main_color"):
            color = _normalize_color(tri_mesh.visual.main_color)
        if hasattr(tri_mesh, "visual") and texture is None:
            texture = _extract_trimesh_texture(tri_mesh.visual, base_dir)
            material = getattr(tri_mesh.visual, "material", None)
            roughness, metallic, base_color = _extract_trimesh_material_params(material)
            if color is None and base_color is not None:
                color = base_color
        meshes.append(
            SapMesh(
                vertices,
                faces.flatten(),
                normals=normals,
                uvs=uvs,
                maxhullvert=maxhullvert,
                color=color,
                texture=texture,
                roughness=roughness,
                metallic=metallic,
            )
        )

    return meshes


def _sap_solref_to_stiffness_damping(solref: Sequence[float] | None) -> tuple[float | None, float | None]:
    if solref is None:
        return None, None
    try:
        timeconst = float(solref[0])
        dampratio = float(solref[1])
    except (TypeError, ValueError, IndexError):
        return None, None
    if timeconst < 0.0 and dampratio < 0.0:
        return -timeconst, -dampratio
    if timeconst <= 0.0 or dampratio <= 0.0:
        return None, None
    return 1.0 / (timeconst * timeconst * dampratio * dampratio), 2.0 / timeconst


@wp.func
def _sap_triangle_inertia(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3):
    vol = wp.dot(v0, wp.cross(v1, v2)) / 6.0
    first = vol * (v0 + v1 + v2) / 4.0

    o00 = wp.outer(v0, v0)
    o11 = wp.outer(v1, v1)
    o22 = wp.outer(v2, v2)
    o01 = wp.outer(v0, v1)
    o02 = wp.outer(v0, v2)
    o12 = wp.outer(v1, v2)
    second = (vol / 10.0) * (o00 + o11 + o22)
    second += (vol / 20.0) * (
        o01 + wp.transpose(o01) + o02 + wp.transpose(o02) + o12 + wp.transpose(o12)
    )

    return vol, first, second


@wp.kernel
def _sap_compute_solid_mesh_inertia_kernel(
    indices: wp.array(dtype=int),
    vertices: wp.array(dtype=wp.vec3),
    volume: wp.array(dtype=float),
    first: wp.array(dtype=wp.vec3),
    second: wp.array(dtype=wp.mat33),
):
    i = wp.tid()
    p = vertices[indices[i * 3 + 0]]
    q = vertices[indices[i * 3 + 1]]
    r = vertices[indices[i * 3 + 2]]

    v, f, s = _sap_triangle_inertia(p, q, r)
    wp.atomic_add(volume, 0, v)
    wp.atomic_add(first, 0, f)
    wp.atomic_add(second, 0, s)


@wp.kernel
def _sap_compute_hollow_mesh_inertia_kernel(
    indices: wp.array(dtype=int),
    vertices: wp.array(dtype=wp.vec3),
    thickness: wp.array(dtype=float),
    volume: wp.array(dtype=float),
    first: wp.array(dtype=wp.vec3),
    second: wp.array(dtype=wp.mat33),
):
    tid = wp.tid()
    i = indices[tid * 3 + 0]
    j = indices[tid * 3 + 1]
    k = indices[tid * 3 + 2]

    vi = vertices[i]
    vj = vertices[j]
    vk = vertices[k]
    normal = -wp.normalize(wp.cross(vj - vi, vk - vi))
    ti = normal * thickness[i]
    tj = normal * thickness[j]
    tk = normal * thickness[k]

    vi0 = vi - ti
    vi1 = vi + ti
    vj0 = vj - tj
    vj1 = vj + tj
    vk0 = vk - tk
    vk1 = vk + tk

    v_total = 0.0
    f_total = wp.vec3(0.0)
    s_total = wp.mat33(0.0)

    v, f, s = _sap_triangle_inertia(vi0, vj0, vk0)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = _sap_triangle_inertia(vj0, vk1, vk0)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = _sap_triangle_inertia(vj0, vj1, vk1)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = _sap_triangle_inertia(vj0, vi1, vj1)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = _sap_triangle_inertia(vj0, vi0, vi1)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = _sap_triangle_inertia(vj1, vi1, vk1)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = _sap_triangle_inertia(vi1, vi0, vk0)
    v_total += v
    f_total += f
    s_total += s
    v, f, s = _sap_triangle_inertia(vi1, vk0, vk1)
    v_total += v
    f_total += f
    s_total += s

    wp.atomic_add(volume, 0, v_total)
    wp.atomic_add(first, 0, f_total)
    wp.atomic_add(second, 0, s_total)


@wp.kernel(enable_backward=False)
def _sap_compute_obb_candidates(
    vertices: wp.array(dtype=wp.vec3),
    base_quat: wp.quat,
    volumes: wp.array2d(dtype=float),
    transforms: wp.array2d(dtype=wp.transform),
    extents: wp.array2d(dtype=wp.vec3),
):
    angle_idx, axis_idx = wp.tid()
    num_angles_per_axis = volumes.shape[0]
    angle = float(angle_idx) * (2.0 * wp.pi) / float(num_angles_per_axis)

    local_axis = wp.vec3(0.0, 0.0, 0.0)
    local_axis[axis_idx] = 1.0
    incremental_quat = wp.quat_from_axis_angle(local_axis, angle)
    quat = base_quat * incremental_quat

    min_bounds = wp.vec3(1.0e10, 1.0e10, 1.0e10)
    max_bounds = wp.vec3(-1.0e10, -1.0e10, -1.0e10)

    for i in range(vertices.shape[0]):
        rotated = wp.quat_rotate(quat, vertices[i])
        min_bounds = wp.min(min_bounds, rotated)
        max_bounds = wp.max(max_bounds, rotated)

    box_extents = (max_bounds - min_bounds) * 0.5
    volume = box_extents[0] * box_extents[1] * box_extents[2]
    center = (max_bounds + min_bounds) * 0.5
    world_center = wp.quat_rotate_inv(quat, center)

    volumes[angle_idx, axis_idx] = volume
    extents[angle_idx, axis_idx] = box_extents
    transforms[angle_idx, axis_idx] = wp.transform(world_center, wp.quat_inverse(quat))


def _sap_compute_inertia_mesh(
    density: float,
    vertices: np.ndarray,
    indices: np.ndarray,
    is_solid: bool = True,
    thickness: Sequence[float] | float = 0.001,
) -> tuple[float, wp.vec3, wp.mat33, float]:
    indices = np.array(indices).flatten()
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    num_tris = len(indices) // 3

    com_warp = wp.zeros(1, dtype=wp.vec3)
    inertia_warp = wp.zeros(1, dtype=wp.mat33)
    volume_warp = wp.zeros(1, dtype=float)

    wp_indices = wp.array(indices.tolist(), dtype=int)
    wp_vertices = wp.array(vertices.tolist(), dtype=wp.vec3)
    if is_solid:
        wp.launch(
            kernel=_sap_compute_solid_mesh_inertia_kernel,
            dim=num_tris,
            inputs=[wp_indices, wp_vertices],
            outputs=[volume_warp, com_warp, inertia_warp],
        )
    else:
        if isinstance(thickness, (int, float)):
            thickness = [float(thickness)] * len(vertices)
        wp.launch(
            kernel=_sap_compute_hollow_mesh_inertia_kernel,
            dim=num_tris,
            inputs=[wp_indices, wp_vertices, wp.array(thickness, dtype=float)],
            outputs=[volume_warp, com_warp, inertia_warp],
        )

    volume_total = float(volume_warp.numpy()[0])
    first_total = com_warp.numpy()[0]
    second_total = inertia_warp.numpy()[0]

    if volume_total < 0.0:
        volume_total = -volume_total
        first_total = -first_total
        second_total = -second_total

    mass = density * volume_total
    if volume_total > 0.0:
        com = first_total / volume_total
    else:
        com = first_total

    second_total *= density
    inertia_origin = np.trace(second_total) * np.eye(3) - second_total
    r = com
    inertia_com = inertia_origin - mass * ((r @ r) * np.eye(3) - np.outer(r, r))
    return mass, wp.vec3(*com), wp.mat33(*inertia_com), volume_total


def _sap_remesh_convex_hull(vertices: np.ndarray, maxhullvert: int = 0) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import ConvexHull

    qhull_options = "Qt"
    if maxhullvert > 0:
        qhull_options += f" TA{maxhullvert - 4}"
    hull = ConvexHull(vertices, qhull_options=qhull_options)
    verts = hull.points.copy().astype(np.float32)
    faces = hull.simplices.astype(np.int32)

    centre = verts.mean(0)
    for i, tri in enumerate(faces):
        a, b, c = verts[tri]
        normal = np.cross(b - a, c - a)
        if np.dot(normal, a - centre) < 0:
            faces[i] = tri[[0, 2, 1]]

    unique_verts = np.unique(faces.flatten())
    verts = verts[unique_verts]
    mapping = {v: i for i, v in enumerate(unique_verts)}
    faces = np.array([mapping[v] for v in faces.flatten()], dtype=np.int32).reshape(faces.shape)
    return verts, faces


def _sap_compute_inertia_obb(vertices: np.ndarray, num_angle_steps: int = 360) -> tuple[wp.transform, wp.vec3]:
    vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
    if len(vertices) == 0:
        return wp.transform_identity(), wp.vec3(0.0, 0.0, 0.0)
    if len(vertices) == 1:
        return wp.transform(wp.vec3(vertices[0]), wp.quat_identity()), wp.vec3(0.0, 0.0, 0.0)

    hull_vertices, hull_faces = _sap_remesh_convex_hull(vertices, maxhullvert=0)
    _, com, inertia_tensor, _ = _sap_compute_inertia_mesh(1.0, hull_vertices, hull_faces.flatten())

    center = np.array(com)
    centered_vertices = hull_vertices - center
    inertia = np.array(inertia_tensor).reshape(3, 3)
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    sorted_indices = np.argsort(eigenvalues)
    eigenvectors = eigenvectors[:, sorted_indices]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 2] *= -1

    base_quat = wp.quat_from_matrix(wp.mat33(eigenvectors.T.flatten()))
    vertices_wp = wp.array(centered_vertices, dtype=wp.vec3)
    volumes = wp.zeros((num_angle_steps, 3), dtype=float)
    transforms = wp.zeros((num_angle_steps, 3), dtype=wp.transform)
    extents = wp.zeros((num_angle_steps, 3), dtype=wp.vec3)

    wp.launch(
        _sap_compute_obb_candidates,
        dim=(num_angle_steps, 3),
        inputs=[vertices_wp, base_quat, volumes, transforms, extents],
    )

    best_idx = np.unravel_index(np.argmin(volumes.numpy()), volumes.shape)
    best_transform = transforms.numpy()[best_idx]
    best_extents = extents.numpy()[best_idx]
    best_transform[0:3] += center
    return wp.transform(*best_transform), wp.vec3(*best_extents)


def _parse_xyz(text: str | None, *, default: float = 0.0) -> np.ndarray:
    if text is None:
        return np.full(3, default, dtype=np.float32)
    values = [float(value) for value in text.split()]
    if len(values) != 3:
        raise SapSceneLoaderError(f"Expected 3-vector, got {text!r}")
    return np.asarray(values, dtype=np.float32)


def _rotation_matrix_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in np.asarray(rpy, dtype=np.float64).reshape(3))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (rz @ ry @ rx).astype(np.float32)


def _joint_key(item: tuple[int, Any]) -> int:
    return item[0]


def _topological_sort(
    joints: Sequence[tuple[Any, Any]],
    *,
    use_dfs: bool = True,
) -> list[int]:
    incoming: dict[Any, set[tuple[int, Any]]] = defaultdict(set)
    outgoing: dict[Any, set[tuple[int, Any]]] = defaultdict(set)
    nodes: set[Any] = set()
    for joint_id, (parent, child) in enumerate(joints):
        if len(incoming[child]) == 1:
            raise SapSceneLoaderError(f"Multiple joints lead to body {child}")
        incoming[child].add((joint_id, parent))
        outgoing[parent].add((joint_id, child))
        nodes.add(parent)
        nodes.add(child)

    roots = nodes - set(incoming.keys())
    if len(roots) == 0:
        raise SapSceneLoaderError("No root found in the joint graph.")

    joint_order: list[int] = []
    visited = set()

    if use_dfs:

        def visit(node: Any) -> None:
            visited.add(node)
            for joint_id, child in sorted(outgoing[node], key=_joint_key):
                if child in visited:
                    raise SapSceneLoaderError(f"Joint graph contains a cycle at body {child}")
                joint_order.append(joint_id)
                visit(child)

        for root in sorted(roots):
            visit(root)
    else:
        queue = deque(sorted(roots))
        while queue:
            node = queue.popleft()
            visited.add(node)
            for joint_id, child in sorted(outgoing[node], key=_joint_key):
                if child in visited:
                    raise SapSceneLoaderError(f"Joint graph contains a cycle at body {child}")
                joint_order.append(joint_id)
                queue.append(child)

    return joint_order


def _topological_sort_undirected(
    joints: Sequence[tuple[Any, Any]],
    *,
    use_dfs: bool = True,
    ensure_single_root: bool = False,
) -> tuple[list[int], list[int]]:
    try:
        return _topological_sort(joints, use_dfs=use_dfs), []
    except SapSceneLoaderError:
        pass

    adjacency: dict[Any, list[tuple[int, Any]]] = defaultdict(list)
    nodes: set[Any] = set()
    for joint_id, (parent, child) in enumerate(joints):
        adjacency[parent].append((joint_id, child))
        adjacency[child].append((joint_id, parent))
        nodes.add(parent)
        nodes.add(child)

    if not nodes:
        return [], []

    joint_order: list[int] = []
    reversed_joints: list[int] = []
    visited: set[Any] = set()

    def record_edge(node: Any, neighbor: Any, joint_id: int) -> None:
        original_parent, original_child = joints[joint_id]
        if original_parent == node and original_child == neighbor:
            reversed_edge = False
        elif original_parent == neighbor and original_child == node:
            reversed_edge = True
        else:
            raise SapSceneLoaderError(f"Joint {joint_id} does not connect {node} and {neighbor}")
        if reversed_edge:
            reversed_joints.append(joint_id)
        joint_order.append(joint_id)

    def sorted_roots() -> list[Any]:
        roots = sorted(nodes)
        if -1 in nodes:
            roots = [-1] + [node for node in roots if node != -1]
        return roots

    if use_dfs:

        def visit(node: Any, parent: Any | None = None) -> None:
            visited.add(node)
            for joint_id, neighbor in sorted(adjacency[node], key=_joint_key):
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    raise SapSceneLoaderError(f"Joint graph contains a cycle at body {neighbor}")
                record_edge(node, neighbor, joint_id)
                visit(neighbor, node)

        for root in sorted_roots():
            if root in visited:
                continue
            if ensure_single_root and visited:
                raise SapSceneLoaderError("Multiple roots found in the joint graph.")
            visit(root)
    else:
        for root in sorted_roots():
            if root in visited:
                continue
            if ensure_single_root and visited:
                raise SapSceneLoaderError("Multiple roots found in the joint graph.")
            queue = deque([(root, None)])
            visited.add(root)
            while queue:
                node, parent = queue.popleft()
                for joint_id, neighbor in sorted(adjacency[node], key=_joint_key):
                    if neighbor == parent:
                        continue
                    if neighbor in visited:
                        raise SapSceneLoaderError(f"Joint graph contains a cycle at body {neighbor}")
                    record_edge(node, neighbor, joint_id)
                    visited.add(neighbor)
                    queue.append((neighbor, node))

    return joint_order, reversed_joints
