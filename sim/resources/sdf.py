"""Signed-distance-field resources used by SAP collision generation.

Source note: the SAP modifications in this module are based on Newton's SDF
resource code and adapted for compatibility with imported assets and collision
data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from sim.collision.math import SAP_MAXVAL
from sim.collision.types import SapGeoType

from .mesh import SapMesh
from .sdf_kernels import (
    SAP_AXIS_Z,
    sap_get_primitive_extents,
    sap_sdf_box,
    sap_sdf_capsule,
    sap_sdf_cone,
    sap_sdf_cylinder,
    sap_sdf_ellipsoid,
    sap_sdf_sphere,
)


@wp.struct
class SapSDFData:
    sparse_sdf_ptr: wp.uint64
    sparse_voxel_size: wp.vec3
    sparse_voxel_radius: wp.float32
    coarse_sdf_ptr: wp.uint64
    coarse_voxel_size: wp.vec3
    center: wp.vec3
    half_extents: wp.vec3
    background_value: wp.float32
    scale_baked: wp.bool


@wp.func
def sap_sample_sdf_extrapolated(
    sdf_data: SapSDFData,
    sdf_pos: wp.vec3,
) -> float:
    lower = sdf_data.center - sdf_data.half_extents
    upper = sdf_data.center + sdf_data.half_extents

    inside_extent = (
        sdf_pos[0] >= lower[0]
        and sdf_pos[0] <= upper[0]
        and sdf_pos[1] >= lower[1]
        and sdf_pos[1] <= upper[1]
        and sdf_pos[2] >= lower[2]
        and sdf_pos[2] <= upper[2]
    )

    if inside_extent:
        sparse_idx = wp.volume_world_to_index(sdf_data.sparse_sdf_ptr, sdf_pos)
        sparse_dist = wp.volume_sample_f(sdf_data.sparse_sdf_ptr, sparse_idx, wp.Volume.LINEAR)

        if sparse_dist >= sdf_data.background_value * 0.99 or wp.isnan(sparse_dist):
            coarse_idx = wp.volume_world_to_index(sdf_data.coarse_sdf_ptr, sdf_pos)
            return wp.volume_sample_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR)
        else:
            return sparse_dist
    else:
        eps = 1e-2 * sdf_data.sparse_voxel_size
        clamped_pos = wp.min(wp.max(sdf_pos, lower + eps), upper - eps)
        dist_to_boundary = wp.length(sdf_pos - clamped_pos)

        coarse_idx = wp.volume_world_to_index(sdf_data.coarse_sdf_ptr, clamped_pos)
        boundary_dist = wp.volume_sample_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR)

        return boundary_dist + dist_to_boundary


@wp.func
def sap_sample_sdf_grad_extrapolated(
    sdf_data: SapSDFData,
    sdf_pos: wp.vec3,
) -> tuple[float, wp.vec3]:
    lower = sdf_data.center - sdf_data.half_extents
    upper = sdf_data.center + sdf_data.half_extents

    gradient = wp.vec3(0.0, 0.0, 0.0)

    inside_extent = (
        sdf_pos[0] >= lower[0]
        and sdf_pos[0] <= upper[0]
        and sdf_pos[1] >= lower[1]
        and sdf_pos[1] <= upper[1]
        and sdf_pos[2] >= lower[2]
        and sdf_pos[2] <= upper[2]
    )

    if inside_extent:
        sparse_idx = wp.volume_world_to_index(sdf_data.sparse_sdf_ptr, sdf_pos)
        sparse_dist = wp.volume_sample_grad_f(sdf_data.sparse_sdf_ptr, sparse_idx, wp.Volume.LINEAR, gradient)

        if sparse_dist >= sdf_data.background_value * 0.99 or wp.isnan(sparse_dist):
            coarse_idx = wp.volume_world_to_index(sdf_data.coarse_sdf_ptr, sdf_pos)
            coarse_dist = wp.volume_sample_grad_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR, gradient)
            return coarse_dist, gradient
        else:
            return sparse_dist, gradient
    else:
        eps = 1e-2 * sdf_data.sparse_voxel_size
        clamped_pos = wp.min(wp.max(sdf_pos, lower + eps), upper - eps)
        diff = sdf_pos - clamped_pos
        dist_to_boundary = wp.length(diff)

        coarse_idx = wp.volume_world_to_index(sdf_data.coarse_sdf_ptr, clamped_pos)
        boundary_dist = wp.volume_sample_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR)

        extrapolated_dist = boundary_dist + dist_to_boundary

        if dist_to_boundary > 0.0:
            gradient = diff / dist_to_boundary
        else:
            wp.volume_sample_grad_f(sdf_data.coarse_sdf_ptr, coarse_idx, wp.Volume.LINEAR, gradient)

        return extrapolated_dist, gradient


class SapSDF:
    def __init__(
        self,
        *,
        data: SapSDFData,
        sparse_volume: wp.Volume | None = None,
        coarse_volume: wp.Volume | None = None,
        block_coords: Sequence[wp.vec3us] | None = None,
        texture_block_coords: Sequence[wp.vec3us] | None = None,
        texture_data: Any | None = None,
        _coarse_texture: wp.Texture3D | None = None,
        _subgrid_texture: wp.Texture3D | None = None,
        _internal: bool = False,
    ) -> None:
        if not _internal:
            raise RuntimeError("SapSDF objects are created through SAP resource constructors.")
        self.data = data
        self.sparse_volume = sparse_volume
        self.coarse_volume = coarse_volume
        self.block_coords = block_coords
        self.texture_block_coords = texture_block_coords
        self.texture_data = texture_data
        self._coarse_texture = _coarse_texture
        self._subgrid_texture = _subgrid_texture

    def to_kernel_data(self) -> SapSDFData:
        return self.data

    def to_texture_kernel_data(self) -> Any | None:
        return self.texture_data

    def is_empty(self) -> bool:
        return int(self.data.sparse_sdf_ptr) == 0 and int(self.data.coarse_sdf_ptr) == 0

    def validate(self) -> None:
        if int(self.data.sparse_sdf_ptr) == 0 and self.sparse_volume is not None:
            raise ValueError("SapSDFData sparse pointer is empty but sparse_volume is set.")
        if int(self.data.coarse_sdf_ptr) == 0 and self.coarse_volume is not None:
            raise ValueError("SapSDFData coarse pointer is empty but coarse_volume is set.")

    def __copy__(self) -> "SapSDF":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "SapSDF":
        memo[id(self)] = self
        return self

    @staticmethod
    def create_from_points(
        points: np.ndarray | Sequence[Sequence[float]],
        indices: np.ndarray | Sequence[int],
        *,
        device: object | None = None,
        narrow_band_range: tuple[float, float] = (-0.1, 0.1),
        target_voxel_size: float | None = None,
        max_resolution: int | None = None,
        margin: float = 0.05,
        shape_margin: float = 0.0,
        scale: tuple[float, float, float] | None = None,
    ) -> "SapSDF":
        mesh = SapMesh(points, indices, compute_inertia=False)
        return SapSDF.create_from_mesh(
            mesh,
            device=device,
            narrow_band_range=narrow_band_range,
            target_voxel_size=target_voxel_size,
            max_resolution=max_resolution,
            margin=margin,
            shape_margin=shape_margin,
            scale=scale,
        )

    @staticmethod
    def create_from_mesh(
        mesh: SapMesh,
        *,
        device: object | None = None,
        narrow_band_range: tuple[float, float] = (-0.1, 0.1),
        target_voxel_size: float | None = None,
        max_resolution: int | None = None,
        margin: float = 0.05,
        shape_margin: float = 0.0,
        scale: tuple[float, float, float] | None = None,
        texture_format: str = "uint16",
    ) -> "SapSDF":
        effective_max_resolution = 64 if max_resolution is None and target_voxel_size is None else max_resolution
        bake_scale = scale is not None
        effective_scale = scale if scale is not None else (1.0, 1.0, 1.0)
        sdf_data, sparse_volume, coarse_volume, block_coords = _sap_compute_sdf_from_shape_impl(
            shape_type=int(SapGeoType.MESH),
            shape_geo=mesh,
            shape_scale=effective_scale,
            shape_margin=shape_margin,
            narrow_band_distance=narrow_band_range,
            margin=margin,
            target_voxel_size=target_voxel_size,
            max_resolution=effective_max_resolution if effective_max_resolution is not None else 64,
            bake_scale=bake_scale,
            device=device,
        )

        texture_data = None
        coarse_texture = None
        subgrid_texture = None
        texture_block_coords = None
        if wp.is_cuda_available():
            from .texture_sdf import SapTextureSDFQuantizationMode, sap_create_texture_sdf_from_mesh

            texture_format_map = {
                "float32": SapTextureSDFQuantizationMode.FLOAT32,
                "uint16": SapTextureSDFQuantizationMode.UINT16,
                "uint8": SapTextureSDFQuantizationMode.UINT8,
            }
            if texture_format not in texture_format_map:
                raise ValueError(f"Unknown texture_format {texture_format!r}. Expected one of {list(texture_format_map)}.")
            quantization_mode = texture_format_map[texture_format]

            with wp.ScopedDevice(device):
                verts = mesh.vertices * np.array(effective_scale)[None, :]
                pos = wp.array(verts, dtype=wp.vec3)
                indices = wp.array(mesh.indices, dtype=wp.int32)
                texture_mesh = wp.Mesh(points=pos, indices=indices, support_winding_number=True)

                signed_volume = sap_compute_mesh_signed_volume(pos, indices)
                winding_threshold = 0.5 if signed_volume >= 0.0 else -0.5

                resolution = effective_max_resolution if effective_max_resolution is not None else 64
                texture_data, coarse_texture, subgrid_texture, texture_block_coords = sap_create_texture_sdf_from_mesh(
                    texture_mesh,
                    margin=margin,
                    narrow_band_range=narrow_band_range,
                    max_resolution=resolution,
                    quantization_mode=quantization_mode,
                    winding_threshold=winding_threshold,
                    scale_baked=bake_scale,
                )
                wp.synchronize()

        sdf = SapSDF(
            data=sdf_data,
            sparse_volume=sparse_volume,
            coarse_volume=coarse_volume,
            block_coords=block_coords,
            texture_block_coords=texture_block_coords,
            texture_data=texture_data,
            _coarse_texture=coarse_texture,
            _subgrid_texture=subgrid_texture,
            _internal=True,
        )
        sdf.validate()
        return sdf

    @staticmethod
    def create_from_data(
        *,
        sparse_volume: wp.Volume | None = None,
        coarse_volume: wp.Volume | None = None,
        block_coords: Sequence[wp.vec3us] | None = None,
        center: Sequence[float] | None = None,
        half_extents: Sequence[float] | None = None,
        background_value: float = SAP_MAXVAL,
        scale_baked: bool = False,
        texture_data: Any | None = None,
    ) -> "SapSDF":
        sdf_data = sap_create_empty_sdf_data()
        if sparse_volume is not None:
            sdf_data.sparse_sdf_ptr = sparse_volume.id
            sparse_voxel_size = np.asarray(sparse_volume.get_voxel_size(), dtype=np.float32)
            sdf_data.sparse_voxel_size = wp.vec3(sparse_voxel_size)
            sdf_data.sparse_voxel_radius = 0.5 * float(np.linalg.norm(sparse_voxel_size))
        if coarse_volume is not None:
            sdf_data.coarse_sdf_ptr = coarse_volume.id
            coarse_voxel_size = np.asarray(coarse_volume.get_voxel_size(), dtype=np.float32)
            sdf_data.coarse_voxel_size = wp.vec3(coarse_voxel_size)

        sdf_data.center = wp.vec3(center) if center is not None else wp.vec3(0.0, 0.0, 0.0)
        sdf_data.half_extents = wp.vec3(half_extents) if half_extents is not None else wp.vec3(0.0, 0.0, 0.0)
        sdf_data.background_value = background_value
        sdf_data.scale_baked = scale_baked

        sdf = SapSDF(
            data=sdf_data,
            sparse_volume=sparse_volume,
            coarse_volume=coarse_volume,
            block_coords=block_coords,
            texture_data=texture_data,
            _internal=True,
        )
        sdf.validate()
        return sdf


SAP_SDF_BACKGROUND_VALUE = SAP_MAXVAL


@wp.kernel
def sap_compute_mesh_signed_volume_kernel(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    volume_sum: wp.array(dtype=wp.float32),
):
    tri_idx = wp.tid()
    v0 = points[indices[tri_idx * 3 + 0]]
    v1 = points[indices[tri_idx * 3 + 1]]
    v2 = points[indices[tri_idx * 3 + 2]]
    wp.atomic_add(volume_sum, 0, wp.dot(v0, wp.cross(v1, v2)) / 6.0)


def sap_compute_mesh_signed_volume(points: wp.array, indices: wp.array) -> float:
    num_tris = indices.shape[0] // 3
    volume_sum = wp.zeros(1, dtype=wp.float32)
    wp.launch(sap_compute_mesh_signed_volume_kernel, dim=num_tris, inputs=[points, indices, volume_sum])
    return float(volume_sum.numpy()[0])


@wp.func
def sap_get_distance_to_mesh(mesh: wp.uint64, point: wp.vec3, max_dist: wp.float32, winding_threshold: wp.float32):
    res = wp.mesh_query_point_sign_winding_number(mesh, point, max_dist, 2.0, winding_threshold)
    if res.result:
        closest = wp.mesh_eval_position(mesh, res.face, res.u, res.v)
        vec_to_surface = closest - point
        sign = res.sign
        if winding_threshold < 0.0:
            sign = -sign
        return sign * wp.length(vec_to_surface)
    return max_dist


@wp.func
def sap_int_to_vec3f(x: int, y: int, z: int) -> wp.vec3:
    return wp.vec3(float(x), float(y), float(z))


@wp.kernel
def sap_sdf_from_mesh_kernel(
    mesh: wp.uint64,
    sdf: wp.uint64,
    tile_points: wp.array(dtype=wp.vec3i),
    shape_margin: wp.float32,
    winding_threshold: wp.float32,
):
    tile_idx, local_x, local_y, local_z = wp.tid()

    tile_origin = tile_points[tile_idx]
    x_id = tile_origin[0] + local_x
    y_id = tile_origin[1] + local_y
    z_id = tile_origin[2] + local_z

    sample_pos = wp.volume_index_to_world(sdf, sap_int_to_vec3f(x_id, y_id, z_id))
    signed_distance = sap_get_distance_to_mesh(mesh, sample_pos, 10000.0, winding_threshold)
    signed_distance -= shape_margin
    wp.volume_store(sdf, x_id, y_id, z_id, signed_distance)


@wp.kernel(enable_backward=False)
def sap_sdf_from_primitive_kernel(
    shape_type: wp.int32,
    shape_scale: wp.vec3,
    sdf: wp.uint64,
    tile_points: wp.array(dtype=wp.vec3i),
    shape_margin: wp.float32,
):
    tile_idx, local_x, local_y, local_z = wp.tid()

    tile_origin = tile_points[tile_idx]
    x_id = tile_origin[0] + local_x
    y_id = tile_origin[1] + local_y
    z_id = tile_origin[2] + local_z

    sample_pos = wp.volume_index_to_world(sdf, sap_int_to_vec3f(x_id, y_id, z_id))
    signed_distance = float(1.0e6)
    if shape_type == SapGeoType.SPHERE:
        signed_distance = sap_sdf_sphere(sample_pos, shape_scale[0])
    elif shape_type == SapGeoType.BOX:
        signed_distance = sap_sdf_box(sample_pos, shape_scale[0], shape_scale[1], shape_scale[2])
    elif shape_type == SapGeoType.CAPSULE:
        signed_distance = sap_sdf_capsule(sample_pos, shape_scale[0], shape_scale[1], int(SAP_AXIS_Z))
    elif shape_type == SapGeoType.CYLINDER:
        signed_distance = sap_sdf_cylinder(sample_pos, shape_scale[0], shape_scale[1], int(SAP_AXIS_Z))
    elif shape_type == SapGeoType.ELLIPSOID:
        signed_distance = sap_sdf_ellipsoid(sample_pos, shape_scale)
    elif shape_type == SapGeoType.CONE:
        signed_distance = sap_sdf_cone(sample_pos, shape_scale[0], shape_scale[1], int(SAP_AXIS_Z))
    signed_distance -= shape_margin
    wp.volume_store(sdf, x_id, y_id, z_id, signed_distance)


@wp.kernel
def sap_check_tile_occupied_mesh_kernel(
    mesh: wp.uint64,
    tile_points: wp.array(dtype=wp.vec3f),
    threshold: wp.vec2f,
    winding_threshold: wp.float32,
    tile_occupied: wp.array(dtype=bool),
):
    tid = wp.tid()
    sample_pos = tile_points[tid]

    signed_distance = sap_get_distance_to_mesh(mesh, sample_pos, 10000.0, winding_threshold)
    is_occupied = wp.bool(False)
    if wp.sign(signed_distance) > 0.0:
        is_occupied = signed_distance < threshold[1]
    else:
        is_occupied = signed_distance > threshold[0]
    tile_occupied[tid] = is_occupied


@wp.kernel(enable_backward=False)
def sap_check_tile_occupied_primitive_kernel(
    shape_type: wp.int32,
    shape_scale: wp.vec3,
    tile_points: wp.array(dtype=wp.vec3f),
    threshold: wp.vec2f,
    tile_occupied: wp.array(dtype=bool),
):
    tid = wp.tid()
    sample_pos = tile_points[tid]

    signed_distance = float(1.0e6)
    if shape_type == SapGeoType.SPHERE:
        signed_distance = sap_sdf_sphere(sample_pos, shape_scale[0])
    elif shape_type == SapGeoType.BOX:
        signed_distance = sap_sdf_box(sample_pos, shape_scale[0], shape_scale[1], shape_scale[2])
    elif shape_type == SapGeoType.CAPSULE:
        signed_distance = sap_sdf_capsule(sample_pos, shape_scale[0], shape_scale[1], int(SAP_AXIS_Z))
    elif shape_type == SapGeoType.CYLINDER:
        signed_distance = sap_sdf_cylinder(sample_pos, shape_scale[0], shape_scale[1], int(SAP_AXIS_Z))
    elif shape_type == SapGeoType.ELLIPSOID:
        signed_distance = sap_sdf_ellipsoid(sample_pos, shape_scale)
    elif shape_type == SapGeoType.CONE:
        signed_distance = sap_sdf_cone(sample_pos, shape_scale[0], shape_scale[1], int(SAP_AXIS_Z))

    is_occupied = wp.bool(False)
    if wp.sign(signed_distance) > 0.0:
        is_occupied = signed_distance < threshold[1]
    else:
        is_occupied = signed_distance > threshold[0]
    tile_occupied[tid] = is_occupied


def _sap_compute_sdf_from_shape_impl(
    shape_type: int,
    shape_geo: SapMesh | None = None,
    shape_scale: Sequence[float] = (1.0, 1.0, 1.0),
    shape_margin: float = 0.0,
    narrow_band_distance: Sequence[float] = (-0.1, 0.1),
    margin: float = 0.05,
    target_voxel_size: float | None = None,
    max_resolution: int = 64,
    bake_scale: bool = False,
    verbose: bool = False,
    device: object | None = None,
) -> tuple[SapSDFData, wp.Volume | None, wp.Volume | None, Sequence[wp.vec3us]]:
    if not wp.is_cuda_available():
        raise RuntimeError("sap_compute_sdf_from_shape requires CUDA but no CUDA device is available")

    if shape_type == SapGeoType.PLANE or shape_type == SapGeoType.HFIELD:
        return sap_create_empty_sdf_data(), None, None, []

    with wp.ScopedDevice(device):
        assert isinstance(narrow_band_distance, Sequence), "narrow_band_distance must be a tuple of two floats"
        assert len(narrow_band_distance) == 2, "narrow_band_distance must be a tuple of two floats"
        assert narrow_band_distance[0] < 0.0 < narrow_band_distance[1], (
            "narrow_band_distance[0] must be less than 0.0 and narrow_band_distance[1] must be greater than 0.0"
        )
        assert margin > 0, "margin must be > 0"

        effective_scale = tuple(shape_scale) if bake_scale else (1.0, 1.0, 1.0)
        effective_scale_vec = wp.vec3(effective_scale)

        offset = margin + shape_margin

        if shape_type == SapGeoType.MESH:
            if shape_geo is None:
                raise ValueError("shape_geo must be provided for SapGeoType.MESH.")
            verts = shape_geo.vertices * np.array(effective_scale)[None, :]
            pos = wp.array(verts, dtype=wp.vec3)
            indices = wp.array(shape_geo.indices, dtype=wp.int32)

            mesh = wp.Mesh(points=pos, indices=indices, support_winding_number=True)
            mesh_id = mesh.id

            signed_volume = sap_compute_mesh_signed_volume(pos, indices)
            winding_threshold = 0.5 if signed_volume >= 0.0 else -0.5
            if verbose and signed_volume < 0:
                print("Mesh has inverted winding, using threshold -0.5")

            min_ext = np.min(verts, axis=0).tolist()
            max_ext = np.max(verts, axis=0).tolist()
        else:
            min_ext, max_ext = sap_get_primitive_extents(shape_type, effective_scale)

        min_ext = np.array(min_ext) - offset
        max_ext = np.array(max_ext) + offset
        ext = max_ext - min_ext

        center = (min_ext + max_ext) * 0.5
        half_extents = (max_ext - min_ext) * 0.5

        max_extent = np.max(ext)
        if target_voxel_size is None:
            assert max_resolution % 8 == 0, "max_resolution must be divisible by 8 for SDF volume allocation"
            assert max_resolution < 1 << 16, f"max_resolution must be less than {1 << 16}"
            target_voxel_size = max_extent / max_resolution
        voxel_size_max_ext = target_voxel_size
        grid_tile_nums = (ext / voxel_size_max_ext).astype(int) // 8
        grid_tile_nums = np.maximum(grid_tile_nums, 1)
        grid_dims = grid_tile_nums * 8

        actual_voxel_size = ext / (grid_dims - 1)

        if verbose:
            print(
                f"Extent: {ext}, Grid dims: {grid_dims}, voxel size: {actual_voxel_size} target_voxel_size: {target_voxel_size}"
            )

        tile_max = np.around((max_ext - min_ext) / actual_voxel_size).astype(np.int32) // 8
        tiles = np.array(
            [[i, j, k] for i in range(tile_max[0] + 1) for j in range(tile_max[1] + 1) for k in range(tile_max[2] + 1)],
            dtype=np.int32,
        )

        tile_points = tiles * 8

        tile_center_points_world = (tile_points + 4) * actual_voxel_size + min_ext
        tile_center_points_world = wp.array(tile_center_points_world, dtype=wp.vec3f)
        tile_occupied = wp.zeros(len(tile_points), dtype=bool)

        tile_radius = np.linalg.norm(4 * actual_voxel_size)
        threshold = wp.vec2f(narrow_band_distance[0] - tile_radius, narrow_band_distance[1] + tile_radius)

        if shape_type == SapGeoType.MESH:
            wp.launch(
                sap_check_tile_occupied_mesh_kernel,
                dim=(len(tile_points)),
                inputs=[mesh_id, tile_center_points_world, threshold, winding_threshold],
                outputs=[tile_occupied],
            )
        else:
            wp.launch(
                sap_check_tile_occupied_primitive_kernel,
                dim=(len(tile_points)),
                inputs=[shape_type, effective_scale_vec, tile_center_points_world, threshold],
                outputs=[tile_occupied],
            )

        if verbose:
            print("Occupancy: ", tile_occupied.numpy().sum() / len(tile_points))

        tile_points = tile_points[tile_occupied.numpy()]
        tile_points_wp = wp.array(tile_points, dtype=wp.vec3i)

        sparse_volume = wp.Volume.allocate_by_tiles(
            tile_points=tile_points_wp,
            voxel_size=wp.vec3(actual_voxel_size),
            translation=wp.vec3(min_ext),
            bg_value=SAP_SDF_BACKGROUND_VALUE,
        )

        num_allocated_tiles = len(tile_points)
        if shape_type == SapGeoType.MESH:
            wp.launch(
                sap_sdf_from_mesh_kernel,
                dim=(num_allocated_tiles, 8, 8, 8),
                inputs=[mesh_id, sparse_volume.id, tile_points_wp, shape_margin, winding_threshold],
            )
        else:
            wp.launch(
                sap_sdf_from_primitive_kernel,
                dim=(num_allocated_tiles, 8, 8, 8),
                inputs=[shape_type, effective_scale_vec, sparse_volume.id, tile_points_wp, shape_margin],
            )

        tiles = sparse_volume.get_tiles().numpy()
        block_coords = [wp.vec3us(t_coords) for t_coords in tiles]

        coarse_dims = 8
        coarse_voxel_size = ext / (coarse_dims - 1)
        coarse_tile_points = np.array([[0, 0, 0]], dtype=np.int32)

        coarse_tile_points_wp = wp.array(coarse_tile_points, dtype=wp.vec3i)
        coarse_volume = wp.Volume.allocate_by_tiles(
            tile_points=coarse_tile_points_wp,
            voxel_size=wp.vec3(coarse_voxel_size),
            translation=wp.vec3(min_ext),
            bg_value=SAP_SDF_BACKGROUND_VALUE,
        )

        if shape_type == SapGeoType.MESH:
            wp.launch(
                sap_sdf_from_mesh_kernel,
                dim=(1, 8, 8, 8),
                inputs=[mesh_id, coarse_volume.id, coarse_tile_points_wp, shape_margin, winding_threshold],
            )
        else:
            wp.launch(
                sap_sdf_from_primitive_kernel,
                dim=(1, 8, 8, 8),
                inputs=[shape_type, effective_scale_vec, coarse_volume.id, coarse_tile_points_wp, shape_margin],
            )

        if shape_type == SapGeoType.MESH:
            wp.synchronize()

        if verbose:
            print(f"Coarse SDF: dims={coarse_dims}x{coarse_dims}x{coarse_dims}, voxel size: {coarse_voxel_size}")

        sdf_data = SapSDFData()
        sdf_data.sparse_sdf_ptr = sparse_volume.id
        sdf_data.sparse_voxel_size = wp.vec3(actual_voxel_size)
        sdf_data.sparse_voxel_radius = 0.5 * float(np.linalg.norm(actual_voxel_size))
        sdf_data.coarse_sdf_ptr = coarse_volume.id
        sdf_data.coarse_voxel_size = wp.vec3(coarse_voxel_size)
        sdf_data.center = wp.vec3(center)
        sdf_data.half_extents = wp.vec3(half_extents)
        sdf_data.background_value = SAP_SDF_BACKGROUND_VALUE
        sdf_data.scale_baked = bake_scale

        return sdf_data, sparse_volume, coarse_volume, block_coords


def sap_compute_sdf_from_shape(
    shape_type: int,
    shape_geo: SapMesh | None = None,
    shape_scale: Sequence[float] = (1.0, 1.0, 1.0),
    shape_margin: float = 0.0,
    narrow_band_distance: Sequence[float] = (-0.1, 0.1),
    margin: float = 0.05,
    target_voxel_size: float | None = None,
    max_resolution: int = 64,
    bake_scale: bool = False,
    verbose: bool = False,
    device: object | None = None,
) -> tuple[SapSDFData, wp.Volume | None, wp.Volume | None, Sequence[wp.vec3us]]:
    if shape_type == SapGeoType.MESH:
        if shape_geo is None:
            raise ValueError("shape_geo must be provided for SapGeoType.MESH.")
        sdf = SapSDF.create_from_mesh(
            shape_geo,
            device=device,
            narrow_band_range=tuple(narrow_band_distance),
            target_voxel_size=target_voxel_size,
            max_resolution=max_resolution,
            margin=margin,
            shape_margin=shape_margin,
            scale=tuple(shape_scale) if bake_scale else None,
        )
        return sdf.to_kernel_data(), sdf.sparse_volume, sdf.coarse_volume, (sdf.block_coords or [])

    return _sap_compute_sdf_from_shape_impl(
        shape_type=shape_type,
        shape_geo=shape_geo,
        shape_scale=shape_scale,
        shape_margin=shape_margin,
        narrow_band_distance=narrow_band_distance,
        margin=margin,
        target_voxel_size=target_voxel_size,
        max_resolution=max_resolution,
        bake_scale=bake_scale,
        verbose=verbose,
        device=device,
    )


def sap_create_empty_sdf_data() -> SapSDFData:
    sdf_data = SapSDFData()
    sdf_data.sparse_sdf_ptr = wp.uint64(0)
    sdf_data.sparse_voxel_size = wp.vec3(0.0, 0.0, 0.0)
    sdf_data.sparse_voxel_radius = 0.0
    sdf_data.coarse_sdf_ptr = wp.uint64(0)
    sdf_data.coarse_voxel_size = wp.vec3(0.0, 0.0, 0.0)
    sdf_data.center = wp.vec3(0.0, 0.0, 0.0)
    sdf_data.half_extents = wp.vec3(0.0, 0.0, 0.0)
    sdf_data.background_value = SAP_SDF_BACKGROUND_VALUE
    sdf_data.scale_baked = False
    return sdf_data


__all__ = [
    "SAP_SDF_BACKGROUND_VALUE",
    "SapSDF",
    "SapSDFData",
    "sap_check_tile_occupied_mesh_kernel",
    "sap_check_tile_occupied_primitive_kernel",
    "sap_compute_mesh_signed_volume",
    "sap_compute_mesh_signed_volume_kernel",
    "sap_compute_sdf_from_shape",
    "sap_create_empty_sdf_data",
    "sap_get_distance_to_mesh",
    "sap_int_to_vec3f",
    "sap_sample_sdf_extrapolated",
    "sap_sample_sdf_grad_extrapolated",
    "sap_sdf_from_mesh_kernel",
    "sap_sdf_from_primitive_kernel",
]
