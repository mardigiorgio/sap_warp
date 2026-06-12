"""Shared-memory and reduction helpers for SAP collision kernels.

Source note: the SAP modifications in this module are based on Newton's
collision reduction/support code and adapted for SAP Warp contact buffers.
"""

from __future__ import annotations

import warp as wp


@wp.func_native("""
uint32_t i = reinterpret_cast<uint32_t&>(f);
uint32_t mask = (uint32_t)(-(int)(i >> 31)) | 0x80000000;
return i ^ mask;
""")
def sap_float_flip(f: float) -> wp.uint32: ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def sap_synchronize(): ...


_sap_mat20x3 = wp.types.matrix(shape=(20, 3), dtype=wp.float32)

SAP_ICOSAHEDRON_FACE_NORMALS = _sap_mat20x3(
    0.49112338,
    0.79465455,
    0.35682216,
    -0.18759243,
    0.7946545,
    0.57735026,
    -0.6070619,
    0.7946545,
    0.0,
    -0.18759237,
    0.7946545,
    -0.57735026,
    0.4911234,
    0.79465455,
    -0.3568221,
    0.9822469,
    -0.18759257,
    0.0,
    0.7946544,
    0.18759239,
    -0.5773503,
    0.30353096,
    -0.18759252,
    0.93417233,
    0.7946544,
    0.18759243,
    0.5773503,
    -0.7946545,
    -0.18759249,
    0.5773503,
    -0.30353105,
    0.18759243,
    0.9341724,
    -0.7946544,
    -0.1875924,
    -0.5773503,
    -0.9822469,
    0.18759254,
    0.0,
    0.30353096,
    -0.1875925,
    -0.93417233,
    -0.30353084,
    0.18759246,
    -0.9341724,
    0.18759249,
    -0.7946544,
    0.57735026,
    -0.49112338,
    -0.7946545,
    0.35682213,
    -0.49112338,
    -0.79465455,
    -0.35682213,
    0.18759243,
    -0.7946544,
    -0.57735026,
    0.607062,
    -0.7946544,
    0.0,
)


@wp.func
def sap_get_slot(normal: wp.vec3) -> int:
    up_dot = normal[1]

    if up_dot > 0.65:
        start_idx = 0
        end_idx = 5
    elif up_dot < -0.65:
        start_idx = 15
        end_idx = 20
    elif up_dot >= 0.0:
        start_idx = 0
        end_idx = 15
    else:
        start_idx = 5
        end_idx = 20

    best_slot = start_idx
    max_dot = wp.dot(normal, SAP_ICOSAHEDRON_FACE_NORMALS[start_idx])

    for i in range(start_idx + 1, end_idx):
        d = wp.dot(normal, SAP_ICOSAHEDRON_FACE_NORMALS[i])
        if d > max_dot:
            max_dot = d
            best_slot = i

    return best_slot


@wp.func
def sap_project_point_to_plane(bin_normal_idx: wp.int32, point: wp.vec3) -> wp.vec2:
    face_normal = SAP_ICOSAHEDRON_FACE_NORMALS[bin_normal_idx]

    if wp.abs(face_normal[1]) < 0.9:
        ref = wp.vec3(0.0, 1.0, 0.0)
    else:
        ref = wp.vec3(1.0, 0.0, 0.0)

    u = wp.normalize(ref - wp.dot(ref, face_normal) * face_normal)
    v = wp.cross(face_normal, u)

    return wp.vec2(wp.dot(point, u), wp.dot(point, v))


@wp.func
def sap_get_spatial_direction_2d(dir_idx: int) -> wp.vec2:
    angle = float(dir_idx) * (2.0 * wp.pi / 6.0)
    return wp.vec2(wp.cos(angle), wp.sin(angle))


SAP_NUM_SPATIAL_DIRECTIONS = 6
SAP_NUM_NORMAL_BINS = 20
SAP_NUM_VOXEL_DEPTH_SLOTS = 100


def sap_compute_num_reduction_slots() -> int:
    return SAP_NUM_NORMAL_BINS * (SAP_NUM_SPATIAL_DIRECTIONS + 1) + SAP_NUM_VOXEL_DEPTH_SLOTS


@wp.func
def sap_compute_voxel_index(
    pos_local: wp.vec3,
    aabb_lower: wp.vec3,
    aabb_upper: wp.vec3,
    resolution: wp.vec3i,
) -> int:
    size = aabb_upper - aabb_lower
    rel = wp.vec3(0.0, 0.0, 0.0)
    if size[0] > 1e-6:
        rel = wp.vec3((pos_local[0] - aabb_lower[0]) / size[0], rel[1], rel[2])
    if size[1] > 1e-6:
        rel = wp.vec3(rel[0], (pos_local[1] - aabb_lower[1]) / size[1], rel[2])
    if size[2] > 1e-6:
        rel = wp.vec3(rel[0], rel[1], (pos_local[2] - aabb_lower[2]) / size[2])

    nx = resolution[0]
    ny = resolution[1]
    nz = resolution[2]

    vx = wp.clamp(int(rel[0] * float(nx)), 0, nx - 1)
    vy = wp.clamp(int(rel[1] * float(ny)), 0, ny - 1)
    vz = wp.clamp(int(rel[2] * float(nz)), 0, nz - 1)

    return vx + vy * nx + vz * nx * ny


def sap_create_shared_memory_pointer_block_dim_func(
    add: int,
):
    snippet = f"""
#if defined(__CUDA_ARCH__)
    constexpr int array_size = WP_TILE_BLOCK_DIM +{add};
    __shared__ int s[array_size];
    auto ptr = &s[0];
    return (uint64_t)ptr;
#else
    return (uint64_t)0;
#endif
    """

    @wp.func_native(snippet)
    def get_shared_memory_pointer() -> wp.uint64: ...

    return get_shared_memory_pointer


def sap_create_shared_memory_pointer_block_dim_mul_func(
    mul: int,
):
    snippet = f"""
#if defined(__CUDA_ARCH__)
    constexpr int array_size = WP_TILE_BLOCK_DIM * {mul};
    __shared__ int s[array_size];
    auto ptr = &s[0];
    return (uint64_t)ptr;
#else
    return (uint64_t)0;
#endif
    """

    @wp.func_native(snippet)
    def get_shared_memory_pointer() -> wp.uint64: ...

    return get_shared_memory_pointer


sap_get_shared_memory_pointer_block_dim_plus_2_ints = sap_create_shared_memory_pointer_block_dim_func(2)


__all__ = [
    "SAP_ICOSAHEDRON_FACE_NORMALS",
    "SAP_NUM_NORMAL_BINS",
    "SAP_NUM_SPATIAL_DIRECTIONS",
    "SAP_NUM_VOXEL_DEPTH_SLOTS",
    "sap_compute_num_reduction_slots",
    "sap_compute_voxel_index",
    "sap_create_shared_memory_pointer_block_dim_func",
    "sap_create_shared_memory_pointer_block_dim_mul_func",
    "sap_float_flip",
    "sap_get_shared_memory_pointer_block_dim_plus_2_ints",
    "sap_get_slot",
    "sap_get_spatial_direction_2d",
    "sap_project_point_to_plane",
    "sap_synchronize",
]
