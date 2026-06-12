"""Texture-backed SDF metadata used by SAP collision kernels.

Source note: the SAP modifications in this module are based on Newton's SDF
collision support code and adapted for SAP Warp's narrow-phase stages.
"""

from __future__ import annotations

import warp as wp


SAP_TEXTURE_SDF_SLOT_EMPTY = 0xFFFFFFFF
SAP_TEXTURE_SDF_SLOT_LINEAR = 0xFFFFFFFE


@wp.struct
class SapTextureSDFData:
    """Sparse SDF stored in 3D textures with a coarse grid and subgrid indirection."""

    coarse_texture: wp.Texture3D
    subgrid_texture: wp.Texture3D
    subgrid_start_slots: wp.array(dtype=wp.uint32, ndim=3)

    sdf_box_lower: wp.vec3
    sdf_box_upper: wp.vec3
    inv_sdf_dx: wp.vec3
    subgrid_size: int
    subgrid_size_f: float
    subgrid_samples_f: float
    fine_to_coarse: float

    voxel_size: wp.vec3
    voxel_radius: wp.float32

    subgrids_min_sdf_value: float
    subgrids_sdf_value_range: float

    scale_baked: wp.bool


@wp.func
def sap_apply_subgrid_start(start_slot: wp.uint32, local_f: wp.vec3, subgrid_samples_f: float) -> wp.vec3:
    block_x = float(start_slot & wp.uint32(0x3FF))
    block_y = float((start_slot >> wp.uint32(10)) & wp.uint32(0x3FF))
    block_z = float((start_slot >> wp.uint32(20)) & wp.uint32(0x3FF))

    return wp.vec3(
        local_f[0] + block_x * subgrid_samples_f,
        local_f[1] + block_y * subgrid_samples_f,
        local_f[2] + block_z * subgrid_samples_f,
    )


@wp.func
def sap_apply_subgrid_sdf_scale(raw_value: float, min_value: float, value_range: float) -> float:
    return raw_value * value_range + min_value


SapTextureSDFVec8f = wp.types.vector(length=8, dtype=wp.float32)


@wp.func
def _sap_read_cell_corners(
    sdf: SapTextureSDFData,
    f: wp.vec3,
) -> tuple[SapTextureSDFVec8f, float, float, float]:
    coarse_x = sdf.coarse_texture.width - 1
    coarse_y = sdf.coarse_texture.height - 1
    coarse_z = sdf.coarse_texture.depth - 1

    fine_verts_x = float(coarse_x) * sdf.subgrid_size_f
    fine_verts_y = float(coarse_y) * sdf.subgrid_size_f
    fine_verts_z = float(coarse_z) * sdf.subgrid_size_f

    fx = wp.clamp(f[0], 0.0, fine_verts_x)
    fy = wp.clamp(f[1], 0.0, fine_verts_y)
    fz = wp.clamp(f[2], 0.0, fine_verts_z)

    num_fine_cells_x = int(fine_verts_x)
    num_fine_cells_y = int(fine_verts_y)
    num_fine_cells_z = int(fine_verts_z)
    ix = wp.clamp(int(wp.floor(fx)), 0, num_fine_cells_x - 1)
    iy = wp.clamp(int(wp.floor(fy)), 0, num_fine_cells_y - 1)
    iz = wp.clamp(int(wp.floor(fz)), 0, num_fine_cells_z - 1)
    tx = fx - float(ix)
    ty = fy - float(iy)
    tz = fz - float(iz)

    x_base = wp.clamp(int(float(ix) * sdf.fine_to_coarse), 0, coarse_x - 1)
    y_base = wp.clamp(int(float(iy) * sdf.fine_to_coarse), 0, coarse_y - 1)
    z_base = wp.clamp(int(float(iz) * sdf.fine_to_coarse), 0, coarse_z - 1)

    start_slot = sdf.subgrid_start_slots[x_base, y_base, z_base]

    v000 = float(0.0)
    v100 = float(0.0)
    v010 = float(0.0)
    v110 = float(0.0)
    v001 = float(0.0)
    v101 = float(0.0)
    v011 = float(0.0)
    v111 = float(0.0)

    if start_slot >= wp.static(SAP_TEXTURE_SDF_SLOT_LINEAR):
        cx = float(x_base)
        cy = float(y_base)
        cz = float(z_base)
        coarse_f = wp.vec3(fx, fy, fz) * sdf.fine_to_coarse
        tx = coarse_f[0] - cx
        ty = coarse_f[1] - cy
        tz = coarse_f[2] - cz
        v000 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 0.5, cz + 0.5), dtype=float)
        v100 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 0.5, cz + 0.5), dtype=float)
        v010 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 1.5, cz + 0.5), dtype=float)
        v110 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 1.5, cz + 0.5), dtype=float)
        v001 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 0.5, cz + 1.5), dtype=float)
        v101 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 0.5, cz + 1.5), dtype=float)
        v011 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 1.5, cz + 1.5), dtype=float)
        v111 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 1.5, cz + 1.5), dtype=float)
    else:
        block_x = float(start_slot & wp.uint32(0x3FF))
        block_y = float((start_slot >> wp.uint32(10)) & wp.uint32(0x3FF))
        block_z = float((start_slot >> wp.uint32(20)) & wp.uint32(0x3FF))
        tex_ox = block_x * sdf.subgrid_samples_f
        tex_oy = block_y * sdf.subgrid_samples_f
        tex_oz = block_z * sdf.subgrid_samples_f
        lx = float(ix) - float(x_base) * sdf.subgrid_size_f
        ly = float(iy) - float(y_base) * sdf.subgrid_size_f
        lz = float(iz) - float(z_base) * sdf.subgrid_size_f
        ox = tex_ox + lx + 0.5
        oy = tex_oy + ly + 0.5
        oz = tex_oz + lz + 0.5
        v000 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy, oz), dtype=float)
        v100 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy, oz), dtype=float)
        v010 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy + 1.0, oz), dtype=float)
        v110 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy + 1.0, oz), dtype=float)
        v001 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy, oz + 1.0), dtype=float)
        v101 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy, oz + 1.0), dtype=float)
        v011 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy + 1.0, oz + 1.0), dtype=float)
        v111 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy + 1.0, oz + 1.0), dtype=float)
        v000 = sap_apply_subgrid_sdf_scale(v000, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v100 = sap_apply_subgrid_sdf_scale(v100, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v010 = sap_apply_subgrid_sdf_scale(v010, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v110 = sap_apply_subgrid_sdf_scale(v110, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v001 = sap_apply_subgrid_sdf_scale(v001, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v101 = sap_apply_subgrid_sdf_scale(v101, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v011 = sap_apply_subgrid_sdf_scale(v011, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)
        v111 = sap_apply_subgrid_sdf_scale(v111, sdf.subgrids_min_sdf_value, sdf.subgrids_sdf_value_range)

    corners = SapTextureSDFVec8f(v000, v100, v010, v110, v001, v101, v011, v111)
    return corners, tx, ty, tz


@wp.func
def _sap_trilinear(corners: SapTextureSDFVec8f, tx: float, ty: float, tz: float) -> float:
    c00 = corners[0] + (corners[1] - corners[0]) * tx
    c10 = corners[2] + (corners[3] - corners[2]) * tx
    c01 = corners[4] + (corners[5] - corners[4]) * tx
    c11 = corners[6] + (corners[7] - corners[6]) * tx
    c0 = c00 + (c10 - c00) * ty
    c1 = c01 + (c11 - c01) * ty
    return c0 + (c1 - c0) * tz


@wp.func
def sap_texture_sample_sdf(
    sdf: SapTextureSDFData,
    local_pos: wp.vec3,
) -> float:
    clamped = wp.vec3(
        wp.clamp(local_pos[0], sdf.sdf_box_lower[0], sdf.sdf_box_upper[0]),
        wp.clamp(local_pos[1], sdf.sdf_box_lower[1], sdf.sdf_box_upper[1]),
        wp.clamp(local_pos[2], sdf.sdf_box_lower[2], sdf.sdf_box_upper[2]),
    )
    diff_mag = wp.length(local_pos - clamped)

    f = wp.cw_mul(clamped - sdf.sdf_box_lower, sdf.inv_sdf_dx)

    coarse_x = sdf.coarse_texture.width - 1
    coarse_y = sdf.coarse_texture.height - 1
    coarse_z = sdf.coarse_texture.depth - 1

    fine_verts_x = float(coarse_x) * sdf.subgrid_size_f
    fine_verts_y = float(coarse_y) * sdf.subgrid_size_f
    fine_verts_z = float(coarse_z) * sdf.subgrid_size_f

    fx = wp.clamp(f[0], 0.0, fine_verts_x)
    fy = wp.clamp(f[1], 0.0, fine_verts_y)
    fz = wp.clamp(f[2], 0.0, fine_verts_z)

    num_fine_cells_x = int(fine_verts_x)
    num_fine_cells_y = int(fine_verts_y)
    num_fine_cells_z = int(fine_verts_z)
    ix = wp.clamp(int(wp.floor(fx)), 0, num_fine_cells_x - 1)
    iy = wp.clamp(int(wp.floor(fy)), 0, num_fine_cells_y - 1)
    iz = wp.clamp(int(wp.floor(fz)), 0, num_fine_cells_z - 1)
    tx = fx - float(ix)
    ty = fy - float(iy)
    tz = fz - float(iz)

    x_base = wp.clamp(int(float(ix) * sdf.fine_to_coarse), 0, coarse_x - 1)
    y_base = wp.clamp(int(float(iy) * sdf.fine_to_coarse), 0, coarse_y - 1)
    z_base = wp.clamp(int(float(iz) * sdf.fine_to_coarse), 0, coarse_z - 1)

    start_slot = sdf.subgrid_start_slots[x_base, y_base, z_base]

    v000 = float(0.0)
    v100 = float(0.0)
    v010 = float(0.0)
    v110 = float(0.0)
    v001 = float(0.0)
    v101 = float(0.0)
    v011 = float(0.0)
    v111 = float(0.0)

    needs_scale = False

    if start_slot >= wp.static(SAP_TEXTURE_SDF_SLOT_LINEAR):
        cx = float(x_base)
        cy = float(y_base)
        cz = float(z_base)
        coarse_f = wp.vec3(fx, fy, fz) * sdf.fine_to_coarse
        tx = coarse_f[0] - cx
        ty = coarse_f[1] - cy
        tz = coarse_f[2] - cz
        v000 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 0.5, cz + 0.5), dtype=float)
        v100 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 0.5, cz + 0.5), dtype=float)
        v010 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 1.5, cz + 0.5), dtype=float)
        v110 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 1.5, cz + 0.5), dtype=float)
        v001 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 0.5, cz + 1.5), dtype=float)
        v101 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 0.5, cz + 1.5), dtype=float)
        v011 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 0.5, cy + 1.5, cz + 1.5), dtype=float)
        v111 = wp.texture_sample(sdf.coarse_texture, wp.vec3f(cx + 1.5, cy + 1.5, cz + 1.5), dtype=float)
    else:
        needs_scale = True
        block_x = float(start_slot & wp.uint32(0x3FF))
        block_y = float((start_slot >> wp.uint32(10)) & wp.uint32(0x3FF))
        block_z = float((start_slot >> wp.uint32(20)) & wp.uint32(0x3FF))
        tex_ox = block_x * sdf.subgrid_samples_f
        tex_oy = block_y * sdf.subgrid_samples_f
        tex_oz = block_z * sdf.subgrid_samples_f
        lx = float(ix) - float(x_base) * sdf.subgrid_size_f
        ly = float(iy) - float(y_base) * sdf.subgrid_size_f
        lz = float(iz) - float(z_base) * sdf.subgrid_size_f
        ox = tex_ox + lx + 0.5
        oy = tex_oy + ly + 0.5
        oz = tex_oz + lz + 0.5
        v000 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy, oz), dtype=float)
        v100 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy, oz), dtype=float)
        v010 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy + 1.0, oz), dtype=float)
        v110 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy + 1.0, oz), dtype=float)
        v001 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy, oz + 1.0), dtype=float)
        v101 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy, oz + 1.0), dtype=float)
        v011 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox, oy + 1.0, oz + 1.0), dtype=float)
        v111 = wp.texture_sample(sdf.subgrid_texture, wp.vec3f(ox + 1.0, oy + 1.0, oz + 1.0), dtype=float)

    c00 = v000 + (v100 - v000) * tx
    c10 = v010 + (v110 - v010) * tx
    c01 = v001 + (v101 - v001) * tx
    c11 = v011 + (v111 - v011) * tx
    c0 = c00 + (c10 - c00) * ty
    c1 = c01 + (c11 - c01) * ty
    sdf_val = c0 + (c1 - c0) * tz

    if needs_scale:
        sdf_val = sdf_val * sdf.subgrids_sdf_value_range + sdf.subgrids_min_sdf_value

    return sdf_val + diff_mag


@wp.func
def sap_texture_sample_sdf_grad(
    sdf: SapTextureSDFData,
    local_pos: wp.vec3,
) -> tuple[float, wp.vec3]:
    clamped = wp.vec3(
        wp.clamp(local_pos[0], sdf.sdf_box_lower[0], sdf.sdf_box_upper[0]),
        wp.clamp(local_pos[1], sdf.sdf_box_lower[1], sdf.sdf_box_upper[1]),
        wp.clamp(local_pos[2], sdf.sdf_box_lower[2], sdf.sdf_box_upper[2]),
    )
    diff = local_pos - clamped
    diff_mag = wp.length(diff)

    f = wp.cw_mul(clamped - sdf.sdf_box_lower, sdf.inv_sdf_dx)
    corners, tx, ty, tz = _sap_read_cell_corners(sdf, f)

    sdf_val = _sap_trilinear(corners, tx, ty, tz)

    omtx = 1.0 - tx
    omty = 1.0 - ty
    omtz = 1.0 - tz

    v000 = corners[0]
    v100 = corners[1]
    v010 = corners[2]
    v110 = corners[3]
    v001 = corners[4]
    v101 = corners[5]
    v011 = corners[6]
    v111 = corners[7]

    gx = (
        omty * omtz * (v100 - v000)
        + ty * omtz * (v110 - v010)
        + omty * tz * (v101 - v001)
        + ty * tz * (v111 - v011)
    )
    gy = (
        omtx * omtz * (v010 - v000)
        + tx * omtz * (v110 - v100)
        + omtx * tz * (v011 - v001)
        + tx * tz * (v111 - v101)
    )
    gz = (
        omtx * omty * (v001 - v000)
        + tx * omty * (v101 - v100)
        + omtx * ty * (v011 - v010)
        + tx * ty * (v111 - v110)
    )

    grad = wp.cw_mul(wp.vec3(gx, gy, gz), sdf.inv_sdf_dx)

    if diff_mag > 0.0:
        sdf_val = sdf_val + diff_mag
        grad = diff / diff_mag

    return sdf_val, grad


def sap_create_empty_texture_sdf_data() -> SapTextureSDFData:
    sdf = SapTextureSDFData()
    sdf.subgrid_size = 0
    sdf.subgrid_size_f = 0.0
    sdf.subgrid_samples_f = 0.0
    sdf.fine_to_coarse = 0.0
    sdf.inv_sdf_dx = wp.vec3(0.0, 0.0, 0.0)
    sdf.sdf_box_lower = wp.vec3(0.0, 0.0, 0.0)
    sdf.sdf_box_upper = wp.vec3(0.0, 0.0, 0.0)
    sdf.voxel_size = wp.vec3(0.0, 0.0, 0.0)
    sdf.voxel_radius = 0.0
    sdf.subgrids_min_sdf_value = 0.0
    sdf.subgrids_sdf_value_range = 1.0
    sdf.scale_baked = False
    return sdf


__all__ = [
    "SAP_TEXTURE_SDF_SLOT_EMPTY",
    "SAP_TEXTURE_SDF_SLOT_LINEAR",
    "SapTextureSDFData",
    "SapTextureSDFVec8f",
    "_sap_read_cell_corners",
    "_sap_trilinear",
    "sap_apply_subgrid_sdf_scale",
    "sap_apply_subgrid_start",
    "sap_create_empty_texture_sdf_data",
    "sap_texture_sample_sdf",
    "sap_texture_sample_sdf_grad",
]
