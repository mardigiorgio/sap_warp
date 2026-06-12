"""Collision pipeline that writes SAP contacts from Newton-compatible data.

Source note: the SAP modifications in this module are based on Newton's
collision pipeline code and adapted for SAP Warp contact buffers and solver
boundaries.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import warp as wp
from warp import DeviceLike as SapDeviceLike

from .broad_phase_nxn import SapBroadPhaseAllPairs, SapBroadPhaseExplicit
from .broad_phase_sap import SapBroadPhaseSAP
from .contact_data import SapContactData
from .contacts import SapContacts
from .core import sap_compute_tight_aabb_from_support
from .narrow_phase import SapNarrowPhase
from .support import SapGenericShapeData, SapSupportMapDataProvider, sap_pack_mesh_ptr
from .types import SapGeoType


@wp.struct
class SapCollisionContactWriterData:
    """Warp struct passed to narrow-phase writers so contact kernels can append contacts into the
    active SapContacts buffer.
    """
    contact_max: int
    body_q: wp.array(dtype=wp.transform)
    shape_body: wp.array(dtype=int)
    shape_gap: wp.array(dtype=float)
    contact_count: wp.array(dtype=int)
    out_shape0: wp.array(dtype=int)
    out_shape1: wp.array(dtype=int)
    out_point0: wp.array(dtype=wp.vec3)
    out_point1: wp.array(dtype=wp.vec3)
    out_offset0: wp.array(dtype=wp.vec3)
    out_offset1: wp.array(dtype=wp.vec3)
    out_normal: wp.array(dtype=wp.vec3)
    out_margin0: wp.array(dtype=float)
    out_margin1: wp.array(dtype=float)
    out_tids: wp.array(dtype=int)
    out_stiffness: wp.array(dtype=float)
    out_damping: wp.array(dtype=float)
    out_friction: wp.array(dtype=float)


@wp.func
def sap_write_contact(
    contact_data: SapContactData,
    writer_data: SapCollisionContactWriterData,
    output_index: int,
):
    """Write one generated rigid contact into the output contact buffer with witness points, normal,
    margins, and optional material values.
    """
    total_separation_needed = (
        contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.margin_a + contact_data.margin_b
    )

    offset_mag_a = contact_data.radius_eff_a + contact_data.margin_a
    offset_mag_b = contact_data.radius_eff_b + contact_data.margin_b

    contact_normal_a_to_b = wp.normalize(contact_data.contact_normal_a_to_b)

    a_contact_world = contact_data.contact_point_center - contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_a
    )
    b_contact_world = contact_data.contact_point_center + contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_b
    )

    diff = b_contact_world - a_contact_world
    distance = wp.dot(diff, contact_normal_a_to_b)
    d = distance - total_separation_needed

    gap_a = writer_data.shape_gap[contact_data.shape_a]
    gap_b = writer_data.shape_gap[contact_data.shape_b]
    contact_gap = gap_a + gap_b

    index = output_index

    if index < 0:
        if d > contact_gap:
            return
        index = wp.atomic_add(writer_data.contact_count, 0, 1)
    if index >= writer_data.contact_max:
        return

    writer_data.out_shape0[index] = contact_data.shape_a
    writer_data.out_shape1[index] = contact_data.shape_b

    body0 = writer_data.shape_body[contact_data.shape_a]
    body1 = writer_data.shape_body[contact_data.shape_b]

    X_bw_a = wp.transform_identity() if body0 == -1 else wp.transform_inverse(writer_data.body_q[body0])
    X_bw_b = wp.transform_identity() if body1 == -1 else wp.transform_inverse(writer_data.body_q[body1])

    writer_data.out_point0[index] = wp.transform_point(X_bw_a, a_contact_world)
    writer_data.out_point1[index] = wp.transform_point(X_bw_b, b_contact_world)

    contact_normal = contact_normal_a_to_b

    writer_data.out_offset0[index] = wp.transform_vector(X_bw_a, offset_mag_a * contact_normal)
    writer_data.out_offset1[index] = wp.transform_vector(X_bw_b, -offset_mag_b * contact_normal)

    writer_data.out_normal[index] = contact_normal
    writer_data.out_margin0[index] = offset_mag_a
    writer_data.out_margin1[index] = offset_mag_b
    writer_data.out_tids[index] = 0

    if writer_data.out_stiffness.shape[0] > 0:
        writer_data.out_stiffness[index] = contact_data.contact_stiffness
        writer_data.out_damping[index] = contact_data.contact_damping
        writer_data.out_friction[index] = contact_data.contact_friction_scale


@wp.kernel
def sap_compute_shape_aabbs(
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_collision_radius: wp.array(dtype=float),
    shape_source_ptr: wp.array(dtype=wp.uint64),
    shape_margin: wp.array(dtype=float),
    shape_gap: wp.array(dtype=float),
    aabb_lower: wp.array(dtype=wp.vec3),
    aabb_upper: wp.array(dtype=wp.vec3),
):
    """Compute world-space shape AABBs from body poses, shape transforms, scales, margins, and geometry
    metadata.
    """
    shape_id = wp.tid()

    rigid_id = shape_body[shape_id]
    geo_type = shape_type[shape_id]

    if rigid_id == -1:
        X_ws = shape_transform[shape_id]
    else:
        X_ws = wp.transform_multiply(body_q[rigid_id], shape_transform[shape_id])

    pos = wp.transform_get_translation(X_ws)
    orientation = wp.transform_get_rotation(X_ws)

    effective_gap = shape_margin[shape_id] + shape_gap[shape_id]
    margin_vec = wp.vec3(effective_gap, effective_gap, effective_gap)

    scale = shape_scale[shape_id]
    is_infinite_plane = (geo_type == SapGeoType.PLANE) and (scale[0] == 0.0 and scale[1] == 0.0)
    is_mesh = geo_type == SapGeoType.MESH
    is_hfield = geo_type == SapGeoType.HFIELD

    if is_infinite_plane or is_mesh or is_hfield:
        radius = shape_collision_radius[shape_id]
        half_extents = wp.vec3(radius, radius, radius)
        aabb_lower[shape_id] = pos - half_extents - margin_vec
        aabb_upper[shape_id] = pos + half_extents + margin_vec
    else:
        shape_data = SapGenericShapeData()
        shape_data.shape_type = geo_type
        shape_data.scale = scale
        shape_data.auxiliary = wp.vec3(0.0, 0.0, 0.0)

        if geo_type == SapGeoType.CONVEX_MESH:
            shape_data.auxiliary = sap_pack_mesh_ptr(shape_source_ptr[shape_id])

        data_provider = SapSupportMapDataProvider()
        aabb_min_world, aabb_max_world = sap_compute_tight_aabb_from_support(
            shape_data, orientation, pos, data_provider
        )

        aabb_lower[shape_id] = aabb_min_world - margin_vec
        aabb_upper[shape_id] = aabb_max_world + margin_vec


@wp.kernel
def sap_prepare_geom_data_kernel(
    shape_transform: wp.array(dtype=wp.transform),
    shape_body: wp.array(dtype=int),
    shape_type: wp.array(dtype=int),
    shape_scale: wp.array(dtype=wp.vec3),
    shape_margin: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    geom_data: wp.array(dtype=wp.vec4),
    geom_transform: wp.array(dtype=wp.transform),
):
    """Prepare per-shape geometry helper data used by broad-phase and narrow-phase collision kernels."""
    idx = wp.tid()

    scale = shape_scale[idx]
    margin = shape_margin[idx]
    geom_data[idx] = wp.vec4(scale[0], scale[1], scale[2], margin)

    body_idx = shape_body[idx]
    if body_idx >= 0:
        geom_transform[idx] = wp.transform_multiply(body_q[body_idx], shape_transform[idx])
    else:
        geom_transform[idx] = shape_transform[idx]


def sap_estimate_rigid_contact_max(model: Any) -> int:
    """Estimate a conservative rigid-contact capacity for a collision model when the caller did not
    provide one.
    """
    if not hasattr(model, "shape_type") or model.shape_type is None:
        return 1000

    shape_types = model.shape_type.numpy()

    primitive_cpp = 5
    mesh_cpp = 40
    max_neighbors_per_shape = 20

    mesh_mask = (shape_types == int(SapGeoType.MESH)) | (shape_types == int(SapGeoType.HFIELD))
    plane_mask = shape_types == int(SapGeoType.PLANE)
    non_plane_mask = ~plane_mask
    num_meshes = int(np.count_nonzero(mesh_mask))
    num_non_planes = int(np.count_nonzero(non_plane_mask))
    num_primitives = num_non_planes - num_meshes
    num_planes = int(np.count_nonzero(plane_mask))

    non_plane_contacts = (
        num_primitives * max_neighbors_per_shape * primitive_cpp + num_meshes * max_neighbors_per_shape * mesh_cpp
    ) // 2

    avg_cpp = (
        (num_primitives * primitive_cpp + num_meshes * mesh_cpp) // max(num_non_planes, 1)
        if num_non_planes > 0
        else 0
    )

    plane_contacts = 0
    if num_planes > 0 and num_non_planes > 0:
        has_world_info = (
            hasattr(model, "shape_world")
            and model.shape_world is not None
            and hasattr(model, "world_count")
            and model.world_count > 0
        )
        shape_world = model.shape_world.numpy() if has_world_info else None

        if shape_world is not None and len(shape_world) == len(shape_types):
            global_mask = shape_world == -1
            local_mask = ~global_mask
            n_worlds = model.world_count

            global_planes = int(np.count_nonzero(global_mask & plane_mask))
            global_non_planes = int(np.count_nonzero(global_mask & non_plane_mask))

            local_plane_counts = np.bincount(shape_world[local_mask & plane_mask], minlength=n_worlds)[:n_worlds]
            local_non_plane_counts = np.bincount(shape_world[local_mask & non_plane_mask], minlength=n_worlds)[
                :n_worlds
            ]

            per_world_planes = local_plane_counts + global_planes
            per_world_non_planes = local_non_plane_counts + global_non_planes

            plane_pair_count = int(np.sum(per_world_planes * per_world_non_planes))
            if n_worlds > 1:
                plane_pair_count -= (n_worlds - 1) * global_planes * global_non_planes
            plane_contacts = plane_pair_count * avg_cpp
        else:
            plane_contacts = num_planes * (num_primitives * primitive_cpp + num_meshes * mesh_cpp)

    total_contacts = non_plane_contacts + plane_contacts

    if hasattr(model, "shape_contact_pair_count") and model.shape_contact_pair_count > 0:
        weighted_cpp = max(avg_cpp, primitive_cpp)
        pair_contacts = int(model.shape_contact_pair_count) * weighted_cpp
        total_contacts = min(total_contacts, pair_contacts)

    return max(1000, total_contacts)


SAP_BROAD_PHASE_MODES = ("nxn", "sap", "explicit")


def sap_normalize_broad_phase_mode(mode: str) -> str:
    """Normalize a user-provided broad-phase mode string into the canonical mode used by
    SapCollisionPipeline.
    """
    mode_str = str(mode).lower()
    if mode_str not in SAP_BROAD_PHASE_MODES:
        raise ValueError(f"Unsupported broad phase mode: {mode!r}")
    return mode_str


def sap_infer_broad_phase_mode_from_instance(
    broad_phase: SapBroadPhaseAllPairs | SapBroadPhaseSAP | SapBroadPhaseExplicit,
) -> str:
    """Infer the broad-phase mode name from an existing broad-phase implementation instance."""
    if isinstance(broad_phase, SapBroadPhaseAllPairs):
        return "nxn"
    if isinstance(broad_phase, SapBroadPhaseSAP):
        return "sap"
    if isinstance(broad_phase, SapBroadPhaseExplicit):
        return "explicit"
    raise TypeError(f"Unsupported broad phase instance: {type(broad_phase)!r}")


def _sap_array_or_default(owner: Any, name: str, default: wp.array) -> wp.array:
    value = getattr(owner, name, None)
    return default if value is None else value


class SapCollisionPipeline:
    """
    Newton-compatible collision front end that writes SAP-owned rigid contacts.

    The broad-phase, narrow-phase, and geometry data layout intentionally track
    Newton's collision implementation for asset and behavior compatibility.
    Drake-style hydroelastic support is being merged; this class documents and
    exposes the rigid-contact path used by the SAP solver today.
    """

    def __init__(
        self,
        model: Any,
        *,
        reduce_contacts: bool = True,
        rigid_contact_max: int | None = None,
        max_triangle_pairs: int = 1000000,
        shape_pairs_filtered: wp.array(dtype=wp.vec2i) | None = None,
        requires_grad: bool | None = None,
        broad_phase: Literal["nxn", "sap", "explicit"]
        | SapBroadPhaseAllPairs
        | SapBroadPhaseSAP
        | SapBroadPhaseExplicit
        | None = None,
        narrow_phase: SapNarrowPhase | None = None,
    ) -> None:
        mode_from_broad_phase: str | None = None
        broad_phase_instance: SapBroadPhaseAllPairs | SapBroadPhaseSAP | SapBroadPhaseExplicit | None = None
        if broad_phase is not None:
            if isinstance(broad_phase, str):
                mode_from_broad_phase = sap_normalize_broad_phase_mode(broad_phase)
            else:
                broad_phase_instance = broad_phase

        shape_count = int(model.shape_count)
        device = model.device

        if rigid_contact_max is None:
            model_rigid_contact_max = int(getattr(model, "rigid_contact_max", 0) or 0)
            if model_rigid_contact_max > 0:
                rigid_contact_max = model_rigid_contact_max
            else:
                rigid_contact_max = sap_estimate_rigid_contact_max(model)
        self._rigid_contact_max = rigid_contact_max

        if max_triangle_pairs <= 0:
            raise ValueError("max_triangle_pairs must be > 0")

        try:
            model.rigid_contact_max = rigid_contact_max
        except Exception:
            pass

        if requires_grad is None:
            requires_grad = bool(getattr(model, "requires_grad", False))

        shape_world = getattr(model, "shape_world", None)
        shape_flags = getattr(model, "shape_flags", None)
        with wp.ScopedDevice(device):
            shape_aabb_lower = wp.zeros(shape_count, dtype=wp.vec3, device=device)
            shape_aabb_upper = wp.zeros(shape_count, dtype=wp.vec3, device=device)
            self._empty_shape_source = wp.zeros(shape_count, dtype=wp.uint64, device=device)
            self._empty_shape_sdf_index = wp.full(shape_count, -1, dtype=wp.int32, device=device)
            self._empty_shape_heightfield_index = wp.full(shape_count, -1, dtype=wp.int32, device=device)
            self._empty_shape_flags = wp.zeros(shape_count, dtype=wp.int32, device=device)
            self._empty_shape_collision_group = wp.zeros(shape_count, dtype=wp.int32, device=device)
            self._empty_shape_world = wp.zeros(shape_count, dtype=wp.int32, device=device)
            self._empty_shape_collision_aabb_lower = wp.zeros(shape_count, dtype=wp.vec3, device=device)
            self._empty_shape_collision_aabb_upper = wp.zeros(shape_count, dtype=wp.vec3, device=device)
            self._empty_shape_voxel_resolution = wp.zeros(shape_count, dtype=wp.vec3i, device=device)

        self.model = model
        self.shape_count = shape_count
        self.device = device
        self.reduce_contacts = reduce_contacts
        self.requires_grad = requires_grad

        using_expert_components = broad_phase_instance is not None or narrow_phase is not None
        if using_expert_components:
            if broad_phase_instance is None or narrow_phase is None:
                raise ValueError("Provide both broad_phase and narrow_phase for expert component construction")

            self.broad_phase_mode = sap_infer_broad_phase_mode_from_instance(broad_phase_instance)
            self.broad_phase = broad_phase_instance

            if self.broad_phase_mode == "explicit":
                if shape_pairs_filtered is None:
                    shape_pairs_filtered = getattr(model, "shape_contact_pairs", None)
                if shape_pairs_filtered is None:
                    raise ValueError("shape_pairs_filtered must be provided for explicit broad phase")
                self.shape_pairs_filtered = shape_pairs_filtered
                self.shape_pairs_max = len(shape_pairs_filtered)
                self.shape_pairs_excluded = None
                self.shape_pairs_excluded_count = 0
            else:
                self.shape_pairs_filtered = None
                self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )

            if narrow_phase.max_candidate_pairs < self.shape_pairs_max:
                raise ValueError(
                    "Provided narrow_phase.max_candidate_pairs is too small for this model and broad phase mode "
                    f"(required at least {self.shape_pairs_max}, got {narrow_phase.max_candidate_pairs})"
                )
            self.narrow_phase = narrow_phase
        else:
            self.broad_phase_mode = mode_from_broad_phase if mode_from_broad_phase is not None else "explicit"

            if self.broad_phase_mode == "explicit":
                if shape_pairs_filtered is None:
                    shape_pairs_filtered = getattr(model, "shape_contact_pairs", None)
                if shape_pairs_filtered is None:
                    raise ValueError("shape_pairs_filtered must be provided for broad_phase=explicit")
                self.broad_phase = SapBroadPhaseExplicit()
                self.shape_pairs_filtered = shape_pairs_filtered
                self.shape_pairs_max = len(shape_pairs_filtered)
                self.shape_pairs_excluded = None
                self.shape_pairs_excluded_count = 0
            elif self.broad_phase_mode == "nxn":
                if shape_world is None:
                    raise ValueError("model.shape_world is required for broad_phase=nxn")
                self.broad_phase = SapBroadPhaseAllPairs(shape_world, shape_flags=shape_flags, device=device)
                self.shape_pairs_filtered = None
                self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )
            elif self.broad_phase_mode == "sap":
                if shape_world is None:
                    raise ValueError("model.shape_world is required for broad_phase=sap")
                self.broad_phase = SapBroadPhaseSAP(shape_world, shape_flags=shape_flags, device=device)
                self.shape_pairs_filtered = None
                self.shape_pairs_max = (shape_count * (shape_count - 1)) // 2
                self.shape_pairs_excluded = self._build_excluded_pairs(model)
                self.shape_pairs_excluded_count = (
                    self.shape_pairs_excluded.shape[0] if self.shape_pairs_excluded is not None else 0
                )
            else:
                raise ValueError(f"Unsupported broad phase mode: {self.broad_phase_mode}")

            has_meshes = False
            has_heightfields = False
            use_lean_gjk_mpr = False
            if hasattr(model, "shape_type") and model.shape_type is not None:
                shape_types = model.shape_type.numpy()
                has_heightfields = bool((shape_types == int(SapGeoType.HFIELD)).any())
                has_meshes = bool((shape_types == int(SapGeoType.MESH)).any())
                lean_unsupported = {
                    int(SapGeoType.CAPSULE),
                    int(SapGeoType.ELLIPSOID),
                    int(SapGeoType.CYLINDER),
                    int(SapGeoType.CONE),
                }
                use_lean_gjk_mpr = not bool(lean_unsupported & set(shape_types.tolist()))

            shape_voxel_resolution = getattr(model, "_shape_voxel_resolution", None)
            if shape_voxel_resolution is None:
                shape_voxel_resolution = self._empty_shape_voxel_resolution
            self.narrow_phase = SapNarrowPhase(
                max_candidate_pairs=self.shape_pairs_max,
                max_triangle_pairs=max_triangle_pairs,
                reduce_contacts=self.reduce_contacts,
                device=device,
                shape_aabb_lower=shape_aabb_lower,
                shape_aabb_upper=shape_aabb_upper,
                contact_writer_warp_func=sap_write_contact,
                shape_voxel_resolution=shape_voxel_resolution,
                has_meshes=has_meshes,
                has_heightfields=has_heightfields,
                use_lean_gjk_mpr=use_lean_gjk_mpr,
            )

        with wp.ScopedDevice(device):
            self.broad_phase_pair_count = wp.zeros(1, dtype=wp.int32, device=device)
            self.broad_phase_shape_pairs = wp.zeros(self.shape_pairs_max, dtype=wp.vec2i, device=device)
            self.geom_data = wp.zeros(shape_count, dtype=wp.vec4, device=device)
            self.geom_transform = wp.zeros(shape_count, dtype=wp.transform, device=device)

        if (
            getattr(self.narrow_phase, "shape_aabb_lower", None) is None
            or getattr(self.narrow_phase, "shape_aabb_upper", None) is None
        ):
            raise ValueError("narrow_phase must expose shape_aabb_lower and shape_aabb_upper arrays")
        if self.narrow_phase.shape_aabb_lower.shape[0] != shape_count:
            raise ValueError(
                "narrow_phase.shape_aabb_lower must have one entry per model shape "
                f"(expected {shape_count}, got {self.narrow_phase.shape_aabb_lower.shape[0]})"
            )
        if self.narrow_phase.shape_aabb_upper.shape[0] != shape_count:
            raise ValueError(
                "narrow_phase.shape_aabb_upper must have one entry per model shape "
                f"(expected {shape_count}, got {self.narrow_phase.shape_aabb_upper.shape[0]})"
            )

    @property
    def rigid_contact_max(self) -> int:
        """Return the configured flat rigid-contact capacity for this collision pipeline."""
        return self._rigid_contact_max

    def contacts(self) -> SapContacts:
        """Allocate a SapContacts buffer sized for this pipeline and carrying any contact attributes
        requested by the model.
        """
        requested_attributes: set[str] = set()
        get_requested_contact_attributes = getattr(self.model, "get_requested_contact_attributes", None)
        if get_requested_contact_attributes is not None:
            requested_attributes = set(get_requested_contact_attributes() or set())

        return SapContacts(
            self.rigid_contact_max,
            requires_grad=self.requires_grad,
            device=self.model.device,
            per_contact_shape_properties=False,
            requested_attributes=requested_attributes,
        )

    @staticmethod
    def _build_excluded_pairs(model: Any) -> wp.array(dtype=wp.vec2i) | None:
        if not hasattr(model, "shape_collision_filter_pairs"):
            return None
        filters = model.shape_collision_filter_pairs
        if not filters:
            return None
        sorted_pairs = sorted(filters)
        return wp.array(
            np.array(sorted_pairs),
            dtype=wp.vec2i,
            device=model.device,
        )

    def collide(
        self,
        state: Any,
        contacts: SapContacts,
    ) -> None:
        """Populate a SapContacts buffer from the current collision state by running AABB setup, broad
        phase, and narrow phase.
        """
        contacts.clear()
        self.broad_phase_pair_count.zero_()

        model = self.model

        if self.requires_grad:
            return

        shape_source_ptr = _sap_array_or_default(model, "shape_source_ptr", self._empty_shape_source)
        shape_flags = _sap_array_or_default(model, "shape_flags", self._empty_shape_flags)
        shape_collision_group = _sap_array_or_default(
            model,
            "shape_collision_group",
            self._empty_shape_collision_group,
        )
        shape_world = _sap_array_or_default(model, "shape_world", self._empty_shape_world)
        shape_sdf_index = _sap_array_or_default(model, "shape_sdf_index", self._empty_shape_sdf_index)
        shape_collision_aabb_lower = _sap_array_or_default(
            model,
            "shape_collision_aabb_lower",
            self._empty_shape_collision_aabb_lower,
        )
        shape_collision_aabb_upper = _sap_array_or_default(
            model,
            "shape_collision_aabb_upper",
            self._empty_shape_collision_aabb_upper,
        )
        shape_voxel_resolution = getattr(self.narrow_phase, "shape_voxel_resolution", None)
        if shape_voxel_resolution is None:
            shape_voxel_resolution = self._empty_shape_voxel_resolution
        shape_heightfield_index = _sap_array_or_default(
            model,
            "shape_heightfield_index",
            self._empty_shape_heightfield_index,
        )

        wp.launch(
            kernel=sap_compute_shape_aabbs,
            dim=model.shape_count,
            inputs=[
                state.body_q,
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_collision_radius,
                shape_source_ptr,
                model.shape_margin,
                model.shape_gap,
            ],
            outputs=[
                self.narrow_phase.shape_aabb_lower,
                self.narrow_phase.shape_aabb_upper,
            ],
            device=self.device,
        )

        if isinstance(self.broad_phase, SapBroadPhaseAllPairs):
            self.broad_phase.launch(
                self.narrow_phase.shape_aabb_lower,
                self.narrow_phase.shape_aabb_upper,
                None,
                shape_collision_group,
                shape_world,
                model.shape_count,
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                device=self.device,
                filter_pairs=self.shape_pairs_excluded,
                num_filter_pairs=self.shape_pairs_excluded_count,
            )
        elif isinstance(self.broad_phase, SapBroadPhaseSAP):
            self.broad_phase.launch(
                self.narrow_phase.shape_aabb_lower,
                self.narrow_phase.shape_aabb_upper,
                None,
                shape_collision_group,
                shape_world,
                model.shape_count,
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                device=self.device,
                filter_pairs=self.shape_pairs_excluded,
                num_filter_pairs=self.shape_pairs_excluded_count,
            )
        else:
            self.broad_phase.launch(
                self.narrow_phase.shape_aabb_lower,
                self.narrow_phase.shape_aabb_upper,
                None,
                self.shape_pairs_filtered,
                len(self.shape_pairs_filtered),
                self.broad_phase_shape_pairs,
                self.broad_phase_pair_count,
                device=self.device,
            )

        wp.launch(
            kernel=sap_prepare_geom_data_kernel,
            dim=model.shape_count,
            inputs=[
                model.shape_transform,
                model.shape_body,
                model.shape_type,
                model.shape_scale,
                model.shape_margin,
                state.body_q,
            ],
            outputs=[
                self.geom_data,
                self.geom_transform,
            ],
            device=self.device,
        )

        writer_data = SapCollisionContactWriterData()
        writer_data.contact_max = contacts.rigid_contact_max
        writer_data.body_q = state.body_q
        writer_data.shape_body = model.shape_body
        writer_data.shape_gap = model.shape_gap
        writer_data.contact_count = contacts.rigid_contact_count
        writer_data.out_shape0 = contacts.rigid_contact_shape0
        writer_data.out_shape1 = contacts.rigid_contact_shape1
        writer_data.out_point0 = contacts.rigid_contact_point0
        writer_data.out_point1 = contacts.rigid_contact_point1
        writer_data.out_offset0 = contacts.rigid_contact_offset0
        writer_data.out_offset1 = contacts.rigid_contact_offset1
        writer_data.out_normal = contacts.rigid_contact_normal
        writer_data.out_margin0 = contacts.rigid_contact_margin0
        writer_data.out_margin1 = contacts.rigid_contact_margin1
        writer_data.out_tids = contacts.rigid_contact_tids
        writer_data.out_stiffness = contacts.rigid_contact_stiffness
        writer_data.out_damping = contacts.rigid_contact_damping
        writer_data.out_friction = contacts.rigid_contact_friction

        self.narrow_phase.launch_custom_write(
            candidate_pair=self.broad_phase_shape_pairs,
            candidate_pair_count=self.broad_phase_pair_count,
            shape_types=model.shape_type,
            shape_data=self.geom_data,
            shape_transform=self.geom_transform,
            shape_source=shape_source_ptr,
            shape_sdf_index=shape_sdf_index,
            texture_sdf_data=getattr(model, "texture_sdf_data", None),
            shape_gap=model.shape_gap,
            shape_collision_radius=model.shape_collision_radius,
            shape_flags=shape_flags,
            shape_collision_aabb_lower=shape_collision_aabb_lower,
            shape_collision_aabb_upper=shape_collision_aabb_upper,
            shape_voxel_resolution=shape_voxel_resolution,
            shape_heightfield_index=shape_heightfield_index,
            heightfield_data=getattr(model, "heightfield_data", None),
            heightfield_elevations=getattr(model, "heightfield_elevations", None),
            writer_data=writer_data,
            device=self.device,
        )


__all__ = [
    "SAP_BROAD_PHASE_MODES",
    "SapCollisionContactWriterData",
    "SapCollisionPipeline",
    "sap_compute_shape_aabbs",
    "sap_estimate_rigid_contact_max",
    "sap_infer_broad_phase_mode_from_instance",
    "sap_normalize_broad_phase_mode",
    "sap_prepare_geom_data_kernel",
    "sap_write_contact",
]
