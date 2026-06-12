"""Texture-backed SDF resources used by SAP collision generation.

Source note: the SAP modifications in this module are based on Newton's
texture SDF resource code and adapted for compatibility with imported assets
and collision data.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from sim.collision.sdf_texture import (
    SAP_TEXTURE_SDF_SLOT_EMPTY,
    SAP_TEXTURE_SDF_SLOT_LINEAR,
    SapTextureSDFData,
    sap_create_empty_texture_sdf_data,
)

from .sdf import sap_get_distance_to_mesh


class SapTextureSDFQuantizationMode:
    FLOAT32 = 4
    UINT16 = 2
    UINT8 = 1


@wp.func
def sap_texture_idx3d(x: int, y: int, z: int, size_x: int, size_y: int) -> int:
    return z * size_x * size_y + y * size_x + x


@wp.func
def sap_texture_id_to_xyz(idx: int, size_x: int, size_y: int) -> wp.vec3i:
    z = idx // (size_x * size_y)
    rem = idx - z * size_x * size_y
    y = rem // size_x
    x = rem - y * size_x
    return wp.vec3i(x, y, z)


@wp.kernel
def sap_check_subgrid_occupied_kernel(
    mesh: wp.uint64,
    subgrid_centers: wp.array(dtype=wp.vec3),
    threshold: wp.vec2f,
    winding_threshold: float,
    subgrid_required: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    sample_pos = subgrid_centers[tid]

    signed_distance = sap_get_distance_to_mesh(mesh, sample_pos, 10000.0, winding_threshold)
    is_occupied = wp.bool(False)
    if wp.sign(signed_distance) > 0.0:
        is_occupied = signed_distance < threshold[1]
    else:
        is_occupied = signed_distance > threshold[0]

    if is_occupied:
        subgrid_required[tid] = 1
    else:
        subgrid_required[tid] = 0


@wp.kernel
def sap_check_subgrid_linearity_kernel(
    mesh: wp.uint64,
    background_sdf: wp.array(dtype=float),
    subgrid_required: wp.array(dtype=wp.int32),
    subgrid_is_linear: wp.array(dtype=wp.int32),
    cells_per_subgrid: int,
    min_corner: wp.vec3,
    cell_size: wp.vec3,
    winding_threshold: float,
    num_subgrids_x: int,
    num_subgrids_y: int,
    bg_size_x: int,
    bg_size_y: int,
    bg_size_z: int,
    error_threshold: float,
):
    tid = wp.tid()
    if subgrid_required[tid] == 0:
        return

    coords = sap_texture_id_to_xyz(tid, num_subgrids_x, num_subgrids_y)
    block_x = coords[0]
    block_y = coords[1]
    block_z = coords[2]

    s = 1.0 / float(cells_per_subgrid)
    samples_per_dim = cells_per_subgrid + 1
    max_abs_error = float(0.0)

    for lz in range(samples_per_dim):
        for ly in range(samples_per_dim):
            for lx in range(samples_per_dim):
                gx = block_x * cells_per_subgrid + lx
                gy = block_y * cells_per_subgrid + ly
                gz = block_z * cells_per_subgrid + lz

                pos = min_corner + wp.vec3(
                    float(gx) * cell_size[0],
                    float(gy) * cell_size[1],
                    float(gz) * cell_size[2],
                )
                mesh_val = sap_get_distance_to_mesh(mesh, pos, 10000.0, winding_threshold)

                coarse_fx = float(block_x) + float(lx) * s
                coarse_fy = float(block_y) + float(ly) * s
                coarse_fz = float(block_z) + float(lz) * s

                x0 = int(wp.floor(coarse_fx))
                y0 = int(wp.floor(coarse_fy))
                z0 = int(wp.floor(coarse_fz))
                x0 = wp.clamp(x0, 0, bg_size_x - 2)
                y0 = wp.clamp(y0, 0, bg_size_y - 2)
                z0 = wp.clamp(z0, 0, bg_size_z - 2)

                tx = wp.clamp(coarse_fx - float(x0), 0.0, 1.0)
                ty = wp.clamp(coarse_fy - float(y0), 0.0, 1.0)
                tz = wp.clamp(coarse_fz - float(z0), 0.0, 1.0)

                v000 = background_sdf[sap_texture_idx3d(x0, y0, z0, bg_size_x, bg_size_y)]
                v100 = background_sdf[sap_texture_idx3d(x0 + 1, y0, z0, bg_size_x, bg_size_y)]
                v010 = background_sdf[sap_texture_idx3d(x0, y0 + 1, z0, bg_size_x, bg_size_y)]
                v110 = background_sdf[sap_texture_idx3d(x0 + 1, y0 + 1, z0, bg_size_x, bg_size_y)]
                v001 = background_sdf[sap_texture_idx3d(x0, y0, z0 + 1, bg_size_x, bg_size_y)]
                v101 = background_sdf[sap_texture_idx3d(x0 + 1, y0, z0 + 1, bg_size_x, bg_size_y)]
                v011 = background_sdf[sap_texture_idx3d(x0, y0 + 1, z0 + 1, bg_size_x, bg_size_y)]
                v111 = background_sdf[sap_texture_idx3d(x0 + 1, y0 + 1, z0 + 1, bg_size_x, bg_size_y)]

                c00 = v000 * (1.0 - tx) + v100 * tx
                c10 = v010 * (1.0 - tx) + v110 * tx
                c01 = v001 * (1.0 - tx) + v101 * tx
                c11 = v011 * (1.0 - tx) + v111 * tx
                c0 = c00 * (1.0 - ty) + c10 * ty
                c1 = c01 * (1.0 - ty) + c11 * ty
                coarse_val = c0 * (1.0 - tz) + c1 * tz

                max_abs_error = wp.max(max_abs_error, wp.abs(mesh_val - coarse_val))

    if max_abs_error < error_threshold:
        subgrid_is_linear[tid] = 1
        subgrid_required[tid] = 0


@wp.kernel
def sap_build_coarse_sdf_from_mesh_kernel(
    mesh: wp.uint64,
    background_sdf: wp.array(dtype=float),
    min_corner: wp.vec3,
    cell_size: wp.vec3,
    cells_per_subgrid: int,
    bg_size_x: int,
    bg_size_y: int,
    bg_size_z: int,
    winding_threshold: float,
):
    tid = wp.tid()

    total_bg = bg_size_x * bg_size_y * bg_size_z
    if tid >= total_bg:
        return

    coords = sap_texture_id_to_xyz(tid, bg_size_x, bg_size_y)
    x_block = coords[0]
    y_block = coords[1]
    z_block = coords[2]

    pos = min_corner + wp.vec3(
        float(x_block * cells_per_subgrid) * cell_size[0],
        float(y_block * cells_per_subgrid) * cell_size[1],
        float(z_block * cells_per_subgrid) * cell_size[2],
    )

    background_sdf[tid] = sap_get_distance_to_mesh(mesh, pos, 10000.0, winding_threshold)


@wp.kernel
def sap_populate_subgrid_texture_float32_kernel(
    mesh: wp.uint64,
    subgrid_required: wp.array(dtype=wp.int32),
    subgrid_addresses: wp.array(dtype=wp.int32),
    subgrid_start_slots: wp.array(dtype=wp.uint32, ndim=3),
    subgrid_texture: wp.array(dtype=float),
    cells_per_subgrid: int,
    min_corner: wp.vec3,
    cell_size: wp.vec3,
    winding_threshold: float,
    num_subgrids_x: int,
    num_subgrids_y: int,
    num_subgrids_z: int,
    tex_blocks_per_dim: int,
    tex_size: int,
):
    tid = wp.tid()

    total_subgrids = num_subgrids_x * num_subgrids_y * num_subgrids_z
    samples_per_dim = cells_per_subgrid + 1
    samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim

    subgrid_idx = tid // samples_per_subgrid
    local_sample = tid - subgrid_idx * samples_per_subgrid

    if subgrid_idx >= total_subgrids:
        return
    if subgrid_required[subgrid_idx] == 0:
        return

    subgrid_coords = sap_texture_id_to_xyz(subgrid_idx, num_subgrids_x, num_subgrids_y)
    block_x = subgrid_coords[0]
    block_y = subgrid_coords[1]
    block_z = subgrid_coords[2]

    local_coords = sap_texture_id_to_xyz(local_sample, samples_per_dim, samples_per_dim)
    lx = local_coords[0]
    ly = local_coords[1]
    lz = local_coords[2]

    gx = block_x * cells_per_subgrid + lx
    gy = block_y * cells_per_subgrid + ly
    gz = block_z * cells_per_subgrid + lz

    pos = min_corner + wp.vec3(
        float(gx) * cell_size[0],
        float(gy) * cell_size[1],
        float(gz) * cell_size[2],
    )
    sdf_val = sap_get_distance_to_mesh(mesh, pos, 10000.0, winding_threshold)

    address = subgrid_addresses[subgrid_idx]
    if address < 0:
        return

    addr_coords = sap_texture_id_to_xyz(address, tex_blocks_per_dim, tex_blocks_per_dim)
    addr_x = addr_coords[0]
    addr_y = addr_coords[1]
    addr_z = addr_coords[2]

    if local_sample == 0:
        start_slot = wp.uint32(addr_x) | (wp.uint32(addr_y) << wp.uint32(10)) | (wp.uint32(addr_z) << wp.uint32(20))
        subgrid_start_slots[block_x, block_y, block_z] = start_slot

    tex_x = addr_x * samples_per_dim + lx
    tex_y = addr_y * samples_per_dim + ly
    tex_z = addr_z * samples_per_dim + lz

    tex_idx = sap_texture_idx3d(tex_x, tex_y, tex_z, tex_size, tex_size)
    subgrid_texture[tex_idx] = sdf_val


@wp.kernel
def sap_populate_subgrid_texture_uint16_kernel(
    mesh: wp.uint64,
    subgrid_required: wp.array(dtype=wp.int32),
    subgrid_addresses: wp.array(dtype=wp.int32),
    subgrid_start_slots: wp.array(dtype=wp.uint32, ndim=3),
    subgrid_texture: wp.array(dtype=wp.uint16),
    cells_per_subgrid: int,
    min_corner: wp.vec3,
    cell_size: wp.vec3,
    winding_threshold: float,
    num_subgrids_x: int,
    num_subgrids_y: int,
    num_subgrids_z: int,
    tex_blocks_per_dim: int,
    tex_size: int,
    sdf_min: float,
    sdf_range_inv: float,
):
    tid = wp.tid()

    total_subgrids = num_subgrids_x * num_subgrids_y * num_subgrids_z
    samples_per_dim = cells_per_subgrid + 1
    samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim

    subgrid_idx = tid // samples_per_subgrid
    local_sample = tid - subgrid_idx * samples_per_subgrid

    if subgrid_idx >= total_subgrids:
        return
    if subgrid_required[subgrid_idx] == 0:
        return

    subgrid_coords = sap_texture_id_to_xyz(subgrid_idx, num_subgrids_x, num_subgrids_y)
    block_x = subgrid_coords[0]
    block_y = subgrid_coords[1]
    block_z = subgrid_coords[2]

    local_coords = sap_texture_id_to_xyz(local_sample, samples_per_dim, samples_per_dim)
    lx = local_coords[0]
    ly = local_coords[1]
    lz = local_coords[2]

    gx = block_x * cells_per_subgrid + lx
    gy = block_y * cells_per_subgrid + ly
    gz = block_z * cells_per_subgrid + lz

    pos = min_corner + wp.vec3(
        float(gx) * cell_size[0],
        float(gy) * cell_size[1],
        float(gz) * cell_size[2],
    )
    sdf_val = sap_get_distance_to_mesh(mesh, pos, 10000.0, winding_threshold)

    address = subgrid_addresses[subgrid_idx]
    if address < 0:
        return

    addr_coords = sap_texture_id_to_xyz(address, tex_blocks_per_dim, tex_blocks_per_dim)
    addr_x = addr_coords[0]
    addr_y = addr_coords[1]
    addr_z = addr_coords[2]

    if local_sample == 0:
        start_slot = wp.uint32(addr_x) | (wp.uint32(addr_y) << wp.uint32(10)) | (wp.uint32(addr_z) << wp.uint32(20))
        subgrid_start_slots[block_x, block_y, block_z] = start_slot

    tex_x = addr_x * samples_per_dim + lx
    tex_y = addr_y * samples_per_dim + ly
    tex_z = addr_z * samples_per_dim + lz

    v_normalized = wp.clamp((sdf_val - sdf_min) * sdf_range_inv, 0.0, 1.0)
    quantized = wp.uint16(v_normalized * 65535.0)

    tex_idx = sap_texture_idx3d(tex_x, tex_y, tex_z, tex_size, tex_size)
    subgrid_texture[tex_idx] = quantized


@wp.kernel
def sap_populate_subgrid_texture_uint8_kernel(
    mesh: wp.uint64,
    subgrid_required: wp.array(dtype=wp.int32),
    subgrid_addresses: wp.array(dtype=wp.int32),
    subgrid_start_slots: wp.array(dtype=wp.uint32, ndim=3),
    subgrid_texture: wp.array(dtype=wp.uint8),
    cells_per_subgrid: int,
    min_corner: wp.vec3,
    cell_size: wp.vec3,
    winding_threshold: float,
    num_subgrids_x: int,
    num_subgrids_y: int,
    num_subgrids_z: int,
    tex_blocks_per_dim: int,
    tex_size: int,
    sdf_min: float,
    sdf_range_inv: float,
):
    tid = wp.tid()

    total_subgrids = num_subgrids_x * num_subgrids_y * num_subgrids_z
    samples_per_dim = cells_per_subgrid + 1
    samples_per_subgrid = samples_per_dim * samples_per_dim * samples_per_dim

    subgrid_idx = tid // samples_per_subgrid
    local_sample = tid - subgrid_idx * samples_per_subgrid

    if subgrid_idx >= total_subgrids:
        return
    if subgrid_required[subgrid_idx] == 0:
        return

    subgrid_coords = sap_texture_id_to_xyz(subgrid_idx, num_subgrids_x, num_subgrids_y)
    block_x = subgrid_coords[0]
    block_y = subgrid_coords[1]
    block_z = subgrid_coords[2]

    local_coords = sap_texture_id_to_xyz(local_sample, samples_per_dim, samples_per_dim)
    lx = local_coords[0]
    ly = local_coords[1]
    lz = local_coords[2]

    gx = block_x * cells_per_subgrid + lx
    gy = block_y * cells_per_subgrid + ly
    gz = block_z * cells_per_subgrid + lz

    pos = min_corner + wp.vec3(
        float(gx) * cell_size[0],
        float(gy) * cell_size[1],
        float(gz) * cell_size[2],
    )
    sdf_val = sap_get_distance_to_mesh(mesh, pos, 10000.0, winding_threshold)

    address = subgrid_addresses[subgrid_idx]
    if address < 0:
        return

    addr_coords = sap_texture_id_to_xyz(address, tex_blocks_per_dim, tex_blocks_per_dim)
    addr_x = addr_coords[0]
    addr_y = addr_coords[1]
    addr_z = addr_coords[2]

    if local_sample == 0:
        start_slot = wp.uint32(addr_x) | (wp.uint32(addr_y) << wp.uint32(10)) | (wp.uint32(addr_z) << wp.uint32(20))
        subgrid_start_slots[block_x, block_y, block_z] = start_slot

    tex_x = addr_x * samples_per_dim + lx
    tex_y = addr_y * samples_per_dim + ly
    tex_z = addr_z * samples_per_dim + lz

    v_normalized = wp.clamp((sdf_val - sdf_min) * sdf_range_inv, 0.0, 1.0)
    quantized = wp.uint8(v_normalized * 255.0)

    tex_idx = sap_texture_idx3d(tex_x, tex_y, tex_z, tex_size, tex_size)
    subgrid_texture[tex_idx] = quantized


@wp.kernel
def sap_sample_volume_at_positions_kernel(
    volume: wp.uint64,
    positions: wp.array(dtype=wp.vec3),
    out_values: wp.array(dtype=float),
):
    tid = wp.tid()
    pos = positions[tid]
    idx = wp.volume_world_to_index(volume, pos)
    out_values[tid] = wp.volume_sample_f(volume, idx, wp.Volume.LINEAR)


def sap_build_sparse_sdf_from_mesh(
    mesh: wp.Mesh,
    grid_size_x: int,
    grid_size_y: int,
    grid_size_z: int,
    cell_size: np.ndarray,
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    subgrid_size: int = 8,
    narrow_band_thickness: float = 0.1,
    quantization_mode: int = SapTextureSDFQuantizationMode.UINT16,
    winding_threshold: float = 0.5,
    linearization_error_threshold: float | None = None,
    device: str = "cuda",
) -> dict:
    num_cells_x = grid_size_x - 1
    num_cells_y = grid_size_y - 1
    num_cells_z = grid_size_z - 1
    w = (num_cells_x + subgrid_size - 1) // subgrid_size
    h = (num_cells_y + subgrid_size - 1) // subgrid_size
    d = (num_cells_z + subgrid_size - 1) // subgrid_size
    total_subgrids = w * h * d

    min_corner_wp = wp.vec3(float(min_corner[0]), float(min_corner[1]), float(min_corner[2]))
    cell_size_wp = wp.vec3(float(cell_size[0]), float(cell_size[1]), float(cell_size[2]))

    bg_size_x = w + 1
    bg_size_y = h + 1
    bg_size_z = d + 1
    total_bg = bg_size_x * bg_size_y * bg_size_z

    background_sdf = wp.zeros(total_bg, dtype=float, device=device)

    wp.launch(
        sap_build_coarse_sdf_from_mesh_kernel,
        dim=total_bg,
        inputs=[mesh.id, background_sdf, min_corner_wp, cell_size_wp, subgrid_size, bg_size_x, bg_size_y, bg_size_z, winding_threshold],
        device=device,
    )

    subgrid_centers = np.empty((total_subgrids, 3), dtype=np.float32)
    for idx in range(total_subgrids):
        bz = idx // (w * h)
        rem = idx - bz * w * h
        by = rem // w
        bx = rem - by * w
        subgrid_centers[idx, 0] = (bx * subgrid_size + subgrid_size * 0.5) * cell_size[0] + min_corner[0]
        subgrid_centers[idx, 1] = (by * subgrid_size + subgrid_size * 0.5) * cell_size[1] + min_corner[1]
        subgrid_centers[idx, 2] = (bz * subgrid_size + subgrid_size * 0.5) * cell_size[2] + min_corner[2]

    subgrid_centers_gpu = wp.array(subgrid_centers, dtype=wp.vec3, device=device)

    half_subgrid = subgrid_size * 0.5 * cell_size
    subgrid_radius = float(np.linalg.norm(half_subgrid))
    threshold = wp.vec2f(-narrow_band_thickness - subgrid_radius, narrow_band_thickness + subgrid_radius)

    subgrid_required = wp.zeros(total_subgrids, dtype=wp.int32, device=device)

    wp.launch(
        sap_check_subgrid_occupied_kernel,
        dim=total_subgrids,
        inputs=[mesh.id, subgrid_centers_gpu, threshold, winding_threshold, subgrid_required],
        device=device,
    )
    wp.synchronize()

    subgrid_occupied = subgrid_required.numpy().copy()

    if linearization_error_threshold is None:
        extents = max_corner - min_corner
        linearization_error_threshold = float(1e-6 * np.linalg.norm(extents))
    subgrid_is_linear = wp.zeros(total_subgrids, dtype=wp.int32, device=device)
    if linearization_error_threshold > 0.0:
        wp.launch(
            sap_check_subgrid_linearity_kernel,
            dim=total_subgrids,
            inputs=[
                mesh.id,
                background_sdf,
                subgrid_required,
                subgrid_is_linear,
                subgrid_size,
                min_corner_wp,
                cell_size_wp,
                winding_threshold,
                w,
                h,
                bg_size_x,
                bg_size_y,
                bg_size_z,
                linearization_error_threshold,
            ],
            device=device,
        )
        wp.synchronize()

    subgrid_addresses = wp.zeros(total_subgrids, dtype=wp.int32, device=device)
    wp._src.utils.array_scan(subgrid_required, subgrid_addresses, inclusive=False)
    wp.synchronize()

    required_np = subgrid_required.numpy()
    num_required = int(np.sum(required_np))

    global_sdf_min = -narrow_band_thickness - subgrid_radius
    global_sdf_max = narrow_band_thickness + subgrid_radius
    sdf_range = global_sdf_max - global_sdf_min
    if sdf_range < 1e-10:
        sdf_range = 1.0

    if num_required == 0:
        subgrid_start_slots = np.full((w, h, d), SAP_TEXTURE_SDF_SLOT_EMPTY, dtype=np.uint32)
        subgrid_texture_data = np.zeros((1, 1, 1), dtype=np.float32)
        tex_size = 1
        final_sdf_min = 0.0
        final_sdf_range = 1.0
    else:
        cubic_root = num_required ** (1.0 / 3.0)
        tex_blocks_per_dim = max(1, int(np.ceil(cubic_root)))
        while tex_blocks_per_dim**3 < num_required:
            tex_blocks_per_dim += 1

        samples_per_dim = subgrid_size + 1
        tex_size = tex_blocks_per_dim * samples_per_dim

        subgrid_start_slots = np.full((w, h, d), SAP_TEXTURE_SDF_SLOT_EMPTY, dtype=np.uint32)
        subgrid_start_slots_gpu = wp.array(subgrid_start_slots, dtype=wp.uint32, device=device)

        total_tex_samples = tex_size * tex_size * tex_size
        samples_per_subgrid = samples_per_dim**3
        total_work = total_subgrids * samples_per_subgrid

        sdf_range_inv = 1.0 / sdf_range

        if quantization_mode == SapTextureSDFQuantizationMode.FLOAT32:
            subgrid_texture_gpu = wp.zeros(total_tex_samples, dtype=float, device=device)
            wp.launch(
                sap_populate_subgrid_texture_float32_kernel,
                dim=total_work,
                inputs=[
                    mesh.id,
                    subgrid_required,
                    subgrid_addresses,
                    subgrid_start_slots_gpu,
                    subgrid_texture_gpu,
                    subgrid_size,
                    min_corner_wp,
                    cell_size_wp,
                    winding_threshold,
                    w,
                    h,
                    d,
                    tex_blocks_per_dim,
                    tex_size,
                ],
                device=device,
            )
            final_sdf_min = 0.0
            final_sdf_range = 1.0
            subgrid_texture_data = subgrid_texture_gpu.numpy().reshape((tex_size, tex_size, tex_size))

        elif quantization_mode == SapTextureSDFQuantizationMode.UINT16:
            subgrid_texture_gpu = wp.zeros(total_tex_samples, dtype=wp.uint16, device=device)
            wp.launch(
                sap_populate_subgrid_texture_uint16_kernel,
                dim=total_work,
                inputs=[
                    mesh.id,
                    subgrid_required,
                    subgrid_addresses,
                    subgrid_start_slots_gpu,
                    subgrid_texture_gpu,
                    subgrid_size,
                    min_corner_wp,
                    cell_size_wp,
                    winding_threshold,
                    w,
                    h,
                    d,
                    tex_blocks_per_dim,
                    tex_size,
                    global_sdf_min,
                    sdf_range_inv,
                ],
                device=device,
            )
            final_sdf_min = global_sdf_min
            final_sdf_range = sdf_range
            subgrid_texture_data = subgrid_texture_gpu.numpy().reshape((tex_size, tex_size, tex_size))

        elif quantization_mode == SapTextureSDFQuantizationMode.UINT8:
            subgrid_texture_gpu = wp.zeros(total_tex_samples, dtype=wp.uint8, device=device)
            wp.launch(
                sap_populate_subgrid_texture_uint8_kernel,
                dim=total_work,
                inputs=[
                    mesh.id,
                    subgrid_required,
                    subgrid_addresses,
                    subgrid_start_slots_gpu,
                    subgrid_texture_gpu,
                    subgrid_size,
                    min_corner_wp,
                    cell_size_wp,
                    winding_threshold,
                    w,
                    h,
                    d,
                    tex_blocks_per_dim,
                    tex_size,
                    global_sdf_min,
                    sdf_range_inv,
                ],
                device=device,
            )
            final_sdf_min = global_sdf_min
            final_sdf_range = sdf_range
            subgrid_texture_data = subgrid_texture_gpu.numpy().reshape((tex_size, tex_size, tex_size))

        else:
            raise ValueError(f"Unknown quantization mode: {quantization_mode}")

        wp.synchronize()
        subgrid_start_slots = subgrid_start_slots_gpu.numpy()

    is_linear_np = subgrid_is_linear.numpy()
    if np.any(is_linear_np):
        for idx in range(total_subgrids):
            if is_linear_np[idx]:
                bz = idx // (w * h)
                rem = idx - bz * w * h
                by = rem // w
                bx = rem - by * w
                subgrid_start_slots[bx, by, bz] = SAP_TEXTURE_SDF_SLOT_LINEAR

    background_sdf_np = background_sdf.numpy().reshape((bg_size_z, bg_size_y, bg_size_x))

    padded_max = min_corner + np.array([w, h, d], dtype=float) * subgrid_size * cell_size

    return {
        "coarse_sdf": background_sdf_np.astype(np.float32),
        "subgrid_data": subgrid_texture_data,
        "subgrid_start_slots": subgrid_start_slots,
        "coarse_dims": (w, h, d),
        "subgrid_tex_size": tex_size,
        "num_subgrids": num_required,
        "min_extents": min_corner,
        "max_extents": padded_max,
        "cell_size": cell_size,
        "subgrid_size": subgrid_size,
        "quantization_mode": quantization_mode,
        "subgrids_min_sdf_value": final_sdf_min,
        "subgrids_sdf_value_range": final_sdf_range,
        "subgrid_required": required_np,
        "subgrid_occupied": subgrid_occupied,
    }


def sap_create_sparse_sdf_textures(
    sparse_data: dict,
    device: str = "cuda",
) -> tuple[SapTextureSDFData, wp.Texture3D, wp.Texture3D]:
    coarse_tex = wp.Texture3D(
        sparse_data["coarse_sdf"],
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=False,
        device=device,
    )

    subgrid_tex = wp.Texture3D(
        sparse_data["subgrid_data"],
        filter_mode=wp.TextureFilterMode.CLOSEST,
        address_mode=wp.TextureAddressMode.CLAMP,
        normalized_coords=False,
        device=device,
    )

    subgrid_slots = wp.array(sparse_data["subgrid_start_slots"], dtype=wp.uint32, device=device)

    cell_size = sparse_data["cell_size"]

    min_ext = sparse_data["min_extents"]
    max_ext = sparse_data["max_extents"]

    sdf_params = SapTextureSDFData()
    sdf_params.coarse_texture = coarse_tex
    sdf_params.subgrid_texture = subgrid_tex
    sdf_params.subgrid_start_slots = subgrid_slots
    sdf_params.sdf_box_lower = wp.vec3(float(min_ext[0]), float(min_ext[1]), float(min_ext[2]))
    sdf_params.sdf_box_upper = wp.vec3(float(max_ext[0]), float(max_ext[1]), float(max_ext[2]))
    sdf_params.inv_sdf_dx = wp.vec3(1.0 / float(cell_size[0]), 1.0 / float(cell_size[1]), 1.0 / float(cell_size[2]))
    sdf_params.subgrid_size = sparse_data["subgrid_size"]
    sdf_params.subgrid_size_f = float(sparse_data["subgrid_size"])
    sdf_params.subgrid_samples_f = float(sparse_data["subgrid_size"] + 1)
    sdf_params.fine_to_coarse = 1.0 / sparse_data["subgrid_size"]

    sdf_params.voxel_size = wp.vec3(float(cell_size[0]), float(cell_size[1]), float(cell_size[2]))
    sdf_params.voxel_radius = float(0.5 * np.linalg.norm(cell_size))

    sdf_params.subgrids_min_sdf_value = sparse_data["subgrids_min_sdf_value"]
    sdf_params.subgrids_sdf_value_range = sparse_data["subgrids_sdf_value_range"]
    sdf_params.scale_baked = False

    return sdf_params, coarse_tex, subgrid_tex


def sap_block_coords_from_subgrid_required(
    subgrid_required: np.ndarray,
    coarse_dims: tuple[int, int, int],
    subgrid_size: int,
    subgrid_occupied: np.ndarray | None = None,
) -> list:
    w, h, _d = coarse_dims
    include = subgrid_occupied if subgrid_occupied is not None else subgrid_required

    coords = []
    for idx in range(len(include)):
        if include[idx]:
            bz = idx // (w * h)
            rem = idx - bz * w * h
            by = rem // w
            bx = rem - by * w
            coords.append(wp.vec3us(bx * subgrid_size, by * subgrid_size, bz * subgrid_size))
    return coords


def sap_create_texture_sdf_from_mesh(
    mesh: wp.Mesh,
    *,
    margin: float = 0.05,
    narrow_band_range: tuple[float, float] = (-0.1, 0.1),
    max_resolution: int = 64,
    subgrid_size: int = 8,
    quantization_mode: int = SapTextureSDFQuantizationMode.UINT16,
    winding_threshold: float = 0.5,
    scale_baked: bool = False,
    device: str | None = None,
) -> tuple[SapTextureSDFData, wp.Texture3D | None, wp.Texture3D | None, list]:
    if device is None:
        device = str(mesh.device)

    points_np = mesh.points.numpy()
    mesh_min = np.min(points_np, axis=0)
    mesh_max = np.max(points_np, axis=0)

    min_ext = mesh_min - margin
    max_ext = mesh_max + margin

    ext = max_ext - min_ext
    max_ext_scalar = np.max(ext)
    if max_ext_scalar < 1e-10:
        return sap_create_empty_texture_sdf_data(), None, None, []
    cell_size_scalar = max_ext_scalar / max_resolution
    dims = np.ceil(ext / cell_size_scalar).astype(int) + 1
    grid_x, grid_y, grid_z = int(dims[0]), int(dims[1]), int(dims[2])
    cell_size = ext / (dims - 1)

    narrow_band_thickness = max(abs(narrow_band_range[0]), abs(narrow_band_range[1]))

    sparse_data = sap_build_sparse_sdf_from_mesh(
        mesh,
        grid_x,
        grid_y,
        grid_z,
        cell_size,
        min_ext,
        max_ext,
        subgrid_size=subgrid_size,
        narrow_band_thickness=narrow_band_thickness,
        quantization_mode=quantization_mode,
        winding_threshold=winding_threshold,
        device=device,
    )

    sdf_params, coarse_tex, subgrid_tex = sap_create_sparse_sdf_textures(sparse_data, device)
    sdf_params.scale_baked = scale_baked

    block_coords = sap_block_coords_from_subgrid_required(
        sparse_data["subgrid_required"],
        sparse_data["coarse_dims"],
        sparse_data["subgrid_size"],
        subgrid_occupied=sparse_data["subgrid_occupied"],
    )

    return sdf_params, coarse_tex, subgrid_tex, block_coords


def sap_create_texture_sdf_from_volume(
    sparse_volume: wp.Volume,
    coarse_volume: wp.Volume,
    *,
    min_ext: np.ndarray,
    max_ext: np.ndarray,
    voxel_size: np.ndarray,
    narrow_band_range: tuple[float, float] = (-0.1, 0.1),
    subgrid_size: int = 8,
    scale_baked: bool = False,
    linearization_error_threshold: float | None = None,
    device: str = "cuda",
) -> tuple[SapTextureSDFData, wp.Texture3D, wp.Texture3D]:
    ext = max_ext - min_ext
    cells_per_axis = np.round(ext / voxel_size).astype(int)
    w = int((cells_per_axis[0] + subgrid_size - 1) // subgrid_size)
    h = int((cells_per_axis[1] + subgrid_size - 1) // subgrid_size)
    d = int((cells_per_axis[2] + subgrid_size - 1) // subgrid_size)
    total_subgrids = w * h * d

    cell_size = voxel_size.copy()
    padded_max = min_ext + np.array([w, h, d], dtype=float) * subgrid_size * cell_size

    bg_size_x = w + 1
    bg_size_y = h + 1
    bg_size_z = d + 1
    total_bg = bg_size_x * bg_size_y * bg_size_z

    bg_positions = np.zeros((total_bg, 3), dtype=np.float32)
    for idx in range(total_bg):
        z_block = idx // (bg_size_x * bg_size_y)
        rem = idx - z_block * bg_size_x * bg_size_y
        y_block = rem // bg_size_x
        x_block = rem - y_block * bg_size_x
        bg_positions[idx] = min_ext + np.array(
            [
                float(x_block * subgrid_size) * cell_size[0],
                float(y_block * subgrid_size) * cell_size[1],
                float(z_block * subgrid_size) * cell_size[2],
            ]
        )

    bg_positions_gpu = wp.array(bg_positions, dtype=wp.vec3, device=device)
    bg_sdf_gpu = wp.zeros(total_bg, dtype=float, device=device)
    wp.launch(
        sap_sample_volume_at_positions_kernel,
        dim=total_bg,
        inputs=[coarse_volume.id, bg_positions_gpu, bg_sdf_gpu],
        device=device,
    )

    narrow_band_thickness = max(abs(narrow_band_range[0]), abs(narrow_band_range[1]))
    half_subgrid = subgrid_size * 0.5 * cell_size
    subgrid_radius = float(np.linalg.norm(half_subgrid))

    subgrid_centers = np.empty((total_subgrids, 3), dtype=np.float32)
    for idx in range(total_subgrids):
        bz = idx // (w * h)
        rem = idx - bz * w * h
        by = rem // w
        bx = rem - by * w
        subgrid_centers[idx, 0] = (bx * subgrid_size + subgrid_size * 0.5) * cell_size[0] + min_ext[0]
        subgrid_centers[idx, 1] = (by * subgrid_size + subgrid_size * 0.5) * cell_size[1] + min_ext[1]
        subgrid_centers[idx, 2] = (bz * subgrid_size + subgrid_size * 0.5) * cell_size[2] + min_ext[2]

    center_positions_gpu = wp.array(subgrid_centers, dtype=wp.vec3, device=device)
    center_sdf_gpu = wp.zeros(total_subgrids, dtype=float, device=device)
    wp.launch(
        sap_sample_volume_at_positions_kernel,
        dim=total_subgrids,
        inputs=[sparse_volume.id, center_positions_gpu, center_sdf_gpu],
        device=device,
    )
    wp.synchronize()

    center_sdf_np = center_sdf_gpu.numpy()
    threshold_inner = -narrow_band_thickness - subgrid_radius
    threshold_outer = narrow_band_thickness + subgrid_radius

    subgrid_required = np.zeros(total_subgrids, dtype=np.int32)
    for idx in range(total_subgrids):
        val = center_sdf_np[idx]
        if val > 0:
            subgrid_required[idx] = 1 if val < threshold_outer else 0
        else:
            subgrid_required[idx] = 1 if val > threshold_inner else 0

    subgrid_occupied = subgrid_required.copy()

    if linearization_error_threshold is None:
        linearization_error_threshold = float(1e-6 * np.linalg.norm(ext))
    subgrid_is_linear = np.zeros(total_subgrids, dtype=np.int32)
    if linearization_error_threshold > 0.0:
        bg_sdf_np = bg_sdf_gpu.numpy()
        samples_per_dim_lin = subgrid_size + 1
        s_inv = 1.0 / float(subgrid_size)

        occupied_indices = np.nonzero(subgrid_required)[0]
        if len(occupied_indices) > 0:
            all_positions = []
            for idx in occupied_indices:
                bz = idx // (w * h)
                rem = idx - bz * w * h
                by = rem // w
                bx = rem - by * w
                for lz in range(samples_per_dim_lin):
                    for ly in range(samples_per_dim_lin):
                        for lx in range(samples_per_dim_lin):
                            gx = bx * subgrid_size + lx
                            gy = by * subgrid_size + ly
                            gz = bz * subgrid_size + lz
                            pos = min_ext + np.array(
                                [
                                    float(gx) * cell_size[0],
                                    float(gy) * cell_size[1],
                                    float(gz) * cell_size[2],
                                ]
                            )
                            all_positions.append(pos)

            all_positions_gpu = wp.array(np.array(all_positions, dtype=np.float32), dtype=wp.vec3, device=device)
            all_sdf_gpu = wp.zeros(len(all_positions), dtype=float, device=device)
            wp.launch(
                sap_sample_volume_at_positions_kernel,
                dim=len(all_positions),
                inputs=[sparse_volume.id, all_positions_gpu, all_sdf_gpu],
                device=device,
            )
            wp.synchronize()
            all_sdf_np = all_sdf_gpu.numpy()

            samples_per_subgrid = samples_per_dim_lin**3
            for i, idx in enumerate(occupied_indices):
                bz_i = idx // (w * h)
                rem_i = idx - bz_i * w * h
                by_i = rem_i // w
                bx_i = rem_i - by_i * w
                max_err = 0.0
                base = i * samples_per_subgrid
                for lz in range(samples_per_dim_lin):
                    for ly in range(samples_per_dim_lin):
                        for lx in range(samples_per_dim_lin):
                            local_idx = lz * samples_per_dim_lin * samples_per_dim_lin + ly * samples_per_dim_lin + lx
                            vol_val = all_sdf_np[base + local_idx]

                            cfx = float(bx_i) + float(lx) * s_inv
                            cfy = float(by_i) + float(ly) * s_inv
                            cfz = float(bz_i) + float(lz) * s_inv

                            x0 = max(0, min(int(np.floor(cfx)), bg_size_x - 2))
                            y0 = max(0, min(int(np.floor(cfy)), bg_size_y - 2))
                            z0 = max(0, min(int(np.floor(cfz)), bg_size_z - 2))
                            tx = np.clip(cfx - float(x0), 0.0, 1.0)
                            ty = np.clip(cfy - float(y0), 0.0, 1.0)
                            tz = np.clip(cfz - float(z0), 0.0, 1.0)

                            def _bg(xi, yi, zi):
                                return float(bg_sdf_np[zi * bg_size_x * bg_size_y + yi * bg_size_x + xi])

                            c00 = _bg(x0, y0, z0) * (1.0 - tx) + _bg(x0 + 1, y0, z0) * tx
                            c10 = _bg(x0, y0 + 1, z0) * (1.0 - tx) + _bg(x0 + 1, y0 + 1, z0) * tx
                            c01 = _bg(x0, y0, z0 + 1) * (1.0 - tx) + _bg(x0 + 1, y0, z0 + 1) * tx
                            c11 = _bg(x0, y0 + 1, z0 + 1) * (1.0 - tx) + _bg(x0 + 1, y0 + 1, z0 + 1) * tx
                            c0 = c00 * (1.0 - ty) + c10 * ty
                            c1 = c01 * (1.0 - ty) + c11 * ty
                            coarse_val = c0 * (1.0 - tz) + c1 * tz

                            max_err = max(max_err, abs(vol_val - coarse_val))

                if max_err < linearization_error_threshold:
                    subgrid_is_linear[idx] = 1
                    subgrid_required[idx] = 0

    num_required = int(np.sum(subgrid_required))

    global_sdf_min = threshold_inner
    global_sdf_max = threshold_outer
    sdf_range = global_sdf_max - global_sdf_min
    if sdf_range < 1e-10:
        sdf_range = 1.0

    if num_required == 0:
        subgrid_start_slots = np.full((w, h, d), SAP_TEXTURE_SDF_SLOT_EMPTY, dtype=np.uint32)
        subgrid_texture_data = np.zeros((1, 1, 1), dtype=np.float32)
        tex_size = 1
    else:
        cubic_root = num_required ** (1.0 / 3.0)
        tex_blocks_per_dim = max(1, int(np.ceil(cubic_root)))
        while tex_blocks_per_dim**3 < num_required:
            tex_blocks_per_dim += 1

        samples_per_dim = subgrid_size + 1
        tex_size = tex_blocks_per_dim * samples_per_dim

        subgrid_start_slots = np.full((w, h, d), SAP_TEXTURE_SDF_SLOT_EMPTY, dtype=np.uint32)
        address = 0
        for idx in range(total_subgrids):
            if subgrid_required[idx]:
                addr_z = address // (tex_blocks_per_dim * tex_blocks_per_dim)
                addr_rem = address - addr_z * tex_blocks_per_dim * tex_blocks_per_dim
                addr_y = addr_rem // tex_blocks_per_dim
                addr_x = addr_rem - addr_y * tex_blocks_per_dim
                bz = idx // (w * h)
                rem = idx - bz * w * h
                by = rem // w
                bx = rem - by * w
                subgrid_start_slots[bx, by, bz] = int(addr_x) | (int(addr_y) << 10) | (int(addr_z) << 20)
                address += 1

        total_texel_work = num_required * samples_per_dim**3
        texel_positions = np.empty((total_texel_work, 3), dtype=np.float32)
        texel_tex_indices = np.empty(total_texel_work, dtype=np.int32)

        work_idx = 0
        subgrid_texture_data = np.zeros((tex_size, tex_size, tex_size), dtype=np.float32)
        for sg_idx in range(total_subgrids):
            if not subgrid_required[sg_idx]:
                continue
            sg_z = sg_idx // (w * h)
            sg_rem = sg_idx - sg_z * w * h
            sg_y = sg_rem // w
            sg_x = sg_rem - sg_y * w

            slot = subgrid_start_slots[sg_x, sg_y, sg_z]
            addr_x = int(slot & 0x3FF)
            addr_y = int((slot >> 10) & 0x3FF)
            addr_z = int((slot >> 20) & 0x3FF)

            for lz in range(samples_per_dim):
                for ly in range(samples_per_dim):
                    for lx in range(samples_per_dim):
                        gx = sg_x * subgrid_size + lx
                        gy = sg_y * subgrid_size + ly
                        gz = sg_z * subgrid_size + lz
                        pos = min_ext + np.array(
                            [
                                float(gx) * cell_size[0],
                                float(gy) * cell_size[1],
                                float(gz) * cell_size[2],
                            ]
                        )
                        tex_x = addr_x * samples_per_dim + lx
                        tex_y = addr_y * samples_per_dim + ly
                        tex_z = addr_z * samples_per_dim + lz
                        texel_positions[work_idx] = pos
                        texel_tex_indices[work_idx] = tex_z * tex_size * tex_size + tex_y * tex_size + tex_x
                        work_idx += 1

        texel_positions_gpu = wp.array(texel_positions, dtype=wp.vec3, device=device)
        texel_sdf_gpu = wp.zeros(total_texel_work, dtype=float, device=device)
        wp.launch(
            sap_sample_volume_at_positions_kernel,
            dim=total_texel_work,
            inputs=[sparse_volume.id, texel_positions_gpu, texel_sdf_gpu],
            device=device,
        )
        wp.synchronize()

        texel_sdf_np = texel_sdf_gpu.numpy()

        bg_threshold = threshold_outer * 2.0
        outlier_mask = (texel_sdf_np > bg_threshold) | (texel_sdf_np < -bg_threshold)
        if np.any(outlier_mask):
            outlier_positions = texel_positions[outlier_mask]
            outlier_gpu = wp.array(outlier_positions, dtype=wp.vec3, device=device)
            outlier_sdf_gpu = wp.zeros(len(outlier_positions), dtype=float, device=device)
            wp.launch(
                sap_sample_volume_at_positions_kernel,
                dim=len(outlier_positions),
                inputs=[coarse_volume.id, outlier_gpu, outlier_sdf_gpu],
                device=device,
            )
            wp.synchronize()
            texel_sdf_np[outlier_mask] = outlier_sdf_gpu.numpy()
        flat_texture = subgrid_texture_data.ravel()
        for i in range(total_texel_work):
            flat_texture[texel_tex_indices[i]] = texel_sdf_np[i]
        subgrid_texture_data = flat_texture.reshape((tex_size, tex_size, tex_size))

    wp.synchronize()

    if np.any(subgrid_is_linear):
        for idx in range(total_subgrids):
            if subgrid_is_linear[idx]:
                bz = idx // (w * h)
                rem = idx - bz * w * h
                by = rem // w
                bx = rem - by * w
                subgrid_start_slots[bx, by, bz] = SAP_TEXTURE_SDF_SLOT_LINEAR

    background_sdf_np = bg_sdf_gpu.numpy().reshape((bg_size_z, bg_size_y, bg_size_x))

    sparse_data = {
        "coarse_sdf": background_sdf_np.astype(np.float32),
        "subgrid_data": subgrid_texture_data,
        "subgrid_start_slots": subgrid_start_slots,
        "coarse_dims": (w, h, d),
        "subgrid_tex_size": tex_size,
        "num_subgrids": num_required,
        "min_extents": min_ext,
        "max_extents": padded_max,
        "cell_size": cell_size,
        "subgrid_size": subgrid_size,
        "quantization_mode": SapTextureSDFQuantizationMode.FLOAT32,
        "subgrids_min_sdf_value": 0.0,
        "subgrids_sdf_value_range": 1.0,
        "subgrid_required": subgrid_required,
        "subgrid_occupied": subgrid_occupied,
    }

    sdf_params, coarse_tex, subgrid_tex = sap_create_sparse_sdf_textures(sparse_data, device)
    sdf_params.scale_baked = scale_baked

    return sdf_params, coarse_tex, subgrid_tex


__all__ = [
    "SapTextureSDFQuantizationMode",
    "sap_block_coords_from_subgrid_required",
    "sap_build_coarse_sdf_from_mesh_kernel",
    "sap_build_sparse_sdf_from_mesh",
    "sap_check_subgrid_linearity_kernel",
    "sap_check_subgrid_occupied_kernel",
    "sap_create_sparse_sdf_textures",
    "sap_create_texture_sdf_from_mesh",
    "sap_create_texture_sdf_from_volume",
    "sap_populate_subgrid_texture_float32_kernel",
    "sap_populate_subgrid_texture_uint16_kernel",
    "sap_populate_subgrid_texture_uint8_kernel",
    "sap_sample_volume_at_positions_kernel",
    "sap_texture_id_to_xyz",
    "sap_texture_idx3d",
]
