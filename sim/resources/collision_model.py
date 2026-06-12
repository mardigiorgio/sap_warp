"""Collision-model resources consumed by SAP collision generation.

Source note: the SAP modifications in this module are based on Newton's shape
and collision metadata code and adapted for compatibility with
Newton-authored assets and runtime data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from sim.collision.flags import SapShapeFlags
from sim.collision.heightfield import SapHeightfieldData
from sim.collision.sdf_texture import SapTextureSDFData, sap_create_empty_texture_sdf_data


@dataclass(frozen=True)
class SapCollisionModel:
    """Newton-compatible collision-side arrays consumed by SapCollisionPipeline.

    The pipeline reads these arrays to build AABBs, candidate pairs,
    narrow-phase inputs, material data, and SAP-owned rigid contacts. The layout
    preserves Newton collision metadata, including hydroelastic flags used by
    the Drake-style hydroelastic merge work.
    """
    device: Any
    shape_count: int
    world_count: int
    requires_grad: bool
    rigid_contact_max: int

    shape_transform: wp.array
    shape_body: wp.array
    shape_type: wp.array
    shape_scale: wp.array
    shape_collision_radius: wp.array
    shape_source_ptr: wp.array
    shape_source_refs: list[Any]
    shape_margin: wp.array
    shape_gap: wp.array
    shape_flags: wp.array
    shape_world: wp.array
    shape_collision_group: wp.array
    shape_sdf_index: wp.array
    texture_sdf_data: wp.array | None
    shape_collision_aabb_lower: wp.array
    shape_collision_aabb_upper: wp.array
    _shape_voxel_resolution: wp.array
    shape_heightfield_index: wp.array
    heightfield_data: wp.array | None
    heightfield_elevations: wp.array | None
    shape_contact_pairs: wp.array | None
    shape_contact_pair_count: int
    texture_sdf_coarse_textures: list[Any] = field(default_factory=list)
    texture_sdf_subgrid_textures: list[Any] = field(default_factory=list)
    texture_sdf_subgrid_start_slots: list[Any] = field(default_factory=list)
    shape_collision_filter_pairs: set[tuple[int, int]] = field(default_factory=set)
    requested_contact_attributes: set[str] = field(default_factory=set)
    shape_material_ke: wp.array | None = None
    shape_material_tau: wp.array | None = None
    shape_material_mu: wp.array | None = None
    shape_material_restitution: wp.array | None = None
    shape_material_mu_torsional: wp.array | None = None
    shape_material_mu_rolling: wp.array | None = None
    shape_material_kh: wp.array | None = None

    def get_requested_contact_attributes(self) -> set[str]:
        """Return a copy of the optional per-contact attributes requested by downstream solver or viewer
        code.
        """
        return set(self.requested_contact_attributes)


@dataclass(frozen=True)
class SapCollisionState:
    """Collision-side state containing body poses in the shape model frame expected by
    SapCollisionPipeline.
    """
    body_q: wp.array
    requires_grad: bool = False


def _array_size(arr: wp.array | None) -> int:
    if arr is None:
        return 0
    try:
        return int(np.asarray(arr.numpy()).reshape(-1).shape[0])
    except Exception:
        return int(getattr(arr, "shape", (0,))[0] or 0)


def _required_array(model: Any, name: str) -> wp.array:
    value = getattr(model, name, None)
    if value is None:
        raise ValueError(f"collision model requires {name}")
    return value


def _array_or_default(model: Any, name: str, default: wp.array) -> wp.array:
    value = getattr(model, name, None)
    return default if value is None else value


def _contact_pair_count(shape_contact_pairs: wp.array | None, model: Any) -> int:
    explicit_count = int(getattr(model, "shape_contact_pair_count", 0) or 0)
    if explicit_count > 0:
        return explicit_count
    if shape_contact_pairs is None:
        return 0
    return int(shape_contact_pairs.shape[0])


def _sap_texture_sdf_from_row(row: np.void, coarse_texture: Any, subgrid_texture: Any, subgrid_start_slots: Any):
    if coarse_texture is None or subgrid_texture is None or subgrid_start_slots is None:
        return sap_create_empty_texture_sdf_data()

    data = SapTextureSDFData()
    data.coarse_texture = coarse_texture
    data.subgrid_texture = subgrid_texture
    data.subgrid_start_slots = subgrid_start_slots
    data.sdf_box_lower = wp.vec3(row["sdf_box_lower"])
    data.sdf_box_upper = wp.vec3(row["sdf_box_upper"])
    data.inv_sdf_dx = wp.vec3(row["inv_sdf_dx"])
    data.subgrid_size = int(row["subgrid_size"])
    data.subgrid_size_f = float(row["subgrid_size_f"])
    data.subgrid_samples_f = float(row["subgrid_samples_f"])
    data.fine_to_coarse = float(row["fine_to_coarse"])
    data.voxel_size = wp.vec3(row["voxel_size"])
    data.voxel_radius = float(row["voxel_radius"])
    data.subgrids_min_sdf_value = float(row["subgrids_min_sdf_value"])
    data.subgrids_sdf_value_range = float(row["subgrids_sdf_value_range"])
    data.scale_baked = bool(row["scale_baked"])
    return data


def _sap_texture_sdf_data_from_model(model: Any, *, device: Any) -> tuple[wp.array, list[Any], list[Any], list[Any]]:
    source_data = getattr(model, "texture_sdf_data", None)
    if source_data is None:
        return wp.zeros(0, dtype=SapTextureSDFData, device=device), [], [], []

    count = int(getattr(source_data, "shape", (0,))[0] or 0)
    if count == 0:
        return wp.zeros(0, dtype=SapTextureSDFData, device=device), [], [], []

    coarse_textures = list(getattr(model, "texture_sdf_coarse_textures", []) or [])
    subgrid_textures = list(getattr(model, "texture_sdf_subgrid_textures", []) or [])
    subgrid_start_slots = list(getattr(model, "texture_sdf_subgrid_start_slots", []) or [])

    if getattr(source_data, "dtype", None) == SapTextureSDFData:
        return source_data, coarse_textures, subgrid_textures, subgrid_start_slots

    host_rows = source_data.numpy()
    sap_rows = []
    sap_coarse_refs = []
    sap_subgrid_refs = []
    sap_slot_refs = []
    for i in range(count):
        coarse_texture = coarse_textures[i] if i < len(coarse_textures) else None
        subgrid_texture = subgrid_textures[i] if i < len(subgrid_textures) else None
        slots = subgrid_start_slots[i] if i < len(subgrid_start_slots) else None
        sap_rows.append(_sap_texture_sdf_from_row(host_rows[i], coarse_texture, subgrid_texture, slots))
        sap_coarse_refs.append(coarse_texture)
        sap_subgrid_refs.append(subgrid_texture)
        sap_slot_refs.append(slots)

    return (
        wp.array(sap_rows, dtype=SapTextureSDFData, device=device),
        sap_coarse_refs,
        sap_subgrid_refs,
        sap_slot_refs,
    )


def _sap_heightfield_from_row(row: np.void) -> SapHeightfieldData:
    data = SapHeightfieldData()
    data.data_offset = int(row["data_offset"])
    data.nrow = int(row["nrow"])
    data.ncol = int(row["ncol"])
    data.hx = float(row["hx"])
    data.hy = float(row["hy"])
    data.min_z = float(row["min_z"])
    data.max_z = float(row["max_z"])
    return data


def _sap_heightfield_data_from_model(model: Any, *, device: Any) -> wp.array | None:
    source_data = getattr(model, "heightfield_data", None)
    if source_data is None:
        return None

    count = int(getattr(source_data, "shape", (0,))[0] or 0)
    if count == 0:
        return wp.zeros(0, dtype=SapHeightfieldData, device=device)

    if getattr(source_data, "dtype", None) == SapHeightfieldData:
        return source_data

    host_rows = source_data.numpy()
    return wp.array(
        [_sap_heightfield_from_row(host_rows[i]) for i in range(count)],
        dtype=SapHeightfieldData,
        device=device,
    )


def sap_collision_model_from_model(model: Any, *, rigid_contact_max: int | None = None) -> SapCollisionModel:
    """Build a SapCollisionModel from a richer model object, filling missing optional collision arrays
    with safe defaults.
    """
    shape_transform = _required_array(model, "shape_transform")
    shape_body = _required_array(model, "shape_body")
    shape_type = _required_array(model, "shape_type")
    shape_scale = _required_array(model, "shape_scale")

    shape_count = int(getattr(model, "shape_count", 0) or 0)
    if shape_count <= 0:
        shape_count = max(_array_size(shape_body), _array_size(shape_type), _array_size(shape_transform))
    if shape_count <= 0:
        raise ValueError("collision model requires at least one shape")

    device = getattr(model, "device", shape_transform.device)
    if rigid_contact_max is None:
        rigid_contact_max = int(getattr(model, "rigid_contact_max", 0) or 0)

    with wp.ScopedDevice(device):
        default_radius = wp.zeros(shape_count, dtype=wp.float32, device=device)
        default_source = wp.zeros(shape_count, dtype=wp.uint64, device=device)
        default_margin = wp.zeros(shape_count, dtype=wp.float32, device=device)
        default_gap = wp.zeros(shape_count, dtype=wp.float32, device=device)
        default_flags = wp.full(shape_count, int(SapShapeFlags.COLLIDE_SHAPES), dtype=wp.int32, device=device)
        default_world = wp.zeros(shape_count, dtype=wp.int32, device=device)
        default_group = wp.full(shape_count, -1, dtype=wp.int32, device=device)
        default_sdf_index = wp.full(shape_count, -1, dtype=wp.int32, device=device)
        default_aabb_lower = wp.zeros(shape_count, dtype=wp.vec3, device=device)
        default_aabb_upper = wp.zeros(shape_count, dtype=wp.vec3, device=device)
        default_voxel_resolution = wp.zeros(shape_count, dtype=wp.vec3i, device=device)
        default_heightfield_index = wp.full(shape_count, -1, dtype=wp.int32, device=device)

    shape_contact_pairs = getattr(model, "shape_contact_pairs", None)
    filters = getattr(model, "shape_collision_filter_pairs", None) or set()
    requested_attributes = set()
    get_requested_contact_attributes = getattr(model, "get_requested_contact_attributes", None)
    if get_requested_contact_attributes is not None:
        requested_attributes = set(get_requested_contact_attributes() or set())

    texture_sdf_data, texture_coarse_refs, texture_subgrid_refs, texture_slot_refs = _sap_texture_sdf_data_from_model(
        model,
        device=device,
    )
    heightfield_data = _sap_heightfield_data_from_model(model, device=device)

    return SapCollisionModel(
        device=device,
        shape_count=shape_count,
        world_count=int(getattr(model, "world_count", 1) or 1),
        requires_grad=bool(getattr(model, "requires_grad", False)),
        rigid_contact_max=int(rigid_contact_max or 0),
        shape_transform=shape_transform,
        shape_body=shape_body,
        shape_type=shape_type,
        shape_scale=shape_scale,
        shape_collision_radius=_array_or_default(model, "shape_collision_radius", default_radius),
        shape_source_ptr=_array_or_default(model, "shape_source_ptr", default_source),
        shape_source_refs=list(getattr(model, "shape_source", []) or []),
        shape_margin=_array_or_default(model, "shape_margin", default_margin),
        shape_gap=_array_or_default(model, "shape_gap", default_gap),
        shape_flags=_array_or_default(model, "shape_flags", default_flags),
        shape_world=_array_or_default(model, "shape_world", default_world),
        shape_collision_group=_array_or_default(model, "shape_collision_group", default_group),
        shape_sdf_index=_array_or_default(model, "shape_sdf_index", default_sdf_index),
        texture_sdf_data=texture_sdf_data,
        texture_sdf_coarse_textures=texture_coarse_refs,
        texture_sdf_subgrid_textures=texture_subgrid_refs,
        texture_sdf_subgrid_start_slots=texture_slot_refs,
        shape_collision_aabb_lower=_array_or_default(model, "shape_collision_aabb_lower", default_aabb_lower),
        shape_collision_aabb_upper=_array_or_default(model, "shape_collision_aabb_upper", default_aabb_upper),
        _shape_voxel_resolution=_array_or_default(model, "_shape_voxel_resolution", default_voxel_resolution),
        shape_heightfield_index=_array_or_default(model, "shape_heightfield_index", default_heightfield_index),
        heightfield_data=heightfield_data,
        heightfield_elevations=getattr(model, "heightfield_elevations", None),
        shape_contact_pairs=shape_contact_pairs,
        shape_contact_pair_count=_contact_pair_count(shape_contact_pairs, model),
        shape_collision_filter_pairs=set(filters),
        requested_contact_attributes=requested_attributes,
        shape_material_ke=getattr(model, "shape_material_ke", None),
        shape_material_tau=getattr(model, "shape_material_tau", None),
        shape_material_mu=getattr(model, "shape_material_mu", None),
        shape_material_restitution=getattr(model, "shape_material_restitution", None),
        shape_material_mu_torsional=getattr(model, "shape_material_mu_torsional", None),
        shape_material_mu_rolling=getattr(model, "shape_material_mu_rolling", None),
        shape_material_kh=getattr(model, "shape_material_kh", None),
    )


def sap_collision_state_from_state(state: Any) -> SapCollisionState:
    """Extract the collision body-pose state from a SapState for use by SapCollisionPipeline.collide."""
    body_q = getattr(state, "body_q", None)
    if body_q is None:
        raise ValueError("collision state requires body_q")
    return SapCollisionState(
        body_q=body_q,
        requires_grad=bool(getattr(state, "requires_grad", False)),
    )


__all__ = [
    "SapCollisionModel",
    "SapCollisionState",
    "sap_collision_model_from_model",
    "sap_collision_state_from_state",
]
