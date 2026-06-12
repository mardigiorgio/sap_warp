"""Heightfield collision helpers for the Newton-compatible collision path.

Source note: the SAP modifications in this module are based on Newton's
heightfield collision code and adapted for SAP Warp compatibility.
"""

from __future__ import annotations

import warp as wp

from .support import SapGenericShapeData, SapGeoTypeEx

@wp.struct
class SapHeightfieldData:
    """Per-shape heightfield metadata for collision kernels.

    The actual elevation data is stored in a separate concatenated array
    passed to kernels. ``data_offset`` is the starting index into that array.
    """

    data_offset: wp.int32  # Offset into the concatenated elevation array
    nrow: wp.int32
    ncol: wp.int32
    hx: wp.float32  # Half-extent X
    hy: wp.float32  # Half-extent Y
    min_z: wp.float32
    max_z: wp.float32


def sap_create_empty_heightfield_data() -> SapHeightfieldData:
    """Create an empty SapHeightfieldData for non-heightfield shapes."""
    hd = SapHeightfieldData()
    hd.data_offset = 0
    hd.nrow = 0
    hd.ncol = 0
    hd.hx = 0.0
    hd.hy = 0.0
    hd.min_z = 0.0
    hd.max_z = 0.0
    return hd


@wp.func
def _sap_heightfield_surface_query(
    hfd: SapHeightfieldData,
    elevation_data: wp.array(dtype=wp.float32),
    pos: wp.vec3,
) -> tuple[float, wp.vec3, float]:
    """Core heightfield surface query returning (plane_dist, normal, lateral_dist_sq).

    Computes the signed distance to the nearest triangle plane at the closest
    point within the heightfield XY extent, plus the squared lateral distance
    from the query point to that extent boundary.
    """
    if hfd.nrow <= 1 or hfd.ncol <= 1:
        return 1.0e10, wp.vec3(0.0, 0.0, 1.0), 0.0

    dx = 2.0 * hfd.hx / wp.float32(hfd.ncol - 1)
    dy = 2.0 * hfd.hy / wp.float32(hfd.nrow - 1)
    z_range = hfd.max_z - hfd.min_z

    # Clamp to heightfield XY extent and track lateral overshoot
    cx = wp.clamp(pos[0], -hfd.hx, hfd.hx)
    cy = wp.clamp(pos[1], -hfd.hy, hfd.hy)
    out_x = pos[0] - cx
    out_y = pos[1] - cy
    lateral_dist_sq = out_x * out_x + out_y * out_y

    col_f = (cx + hfd.hx) / dx
    row_f = (cy + hfd.hy) / dy
    col_f = wp.clamp(col_f, 0.0, wp.float32(hfd.ncol - 1))
    row_f = wp.clamp(row_f, 0.0, wp.float32(hfd.nrow - 1))

    col = wp.min(wp.int32(col_f), hfd.ncol - 2)
    row = wp.min(wp.int32(row_f), hfd.nrow - 2)
    fx = col_f - wp.float32(col)
    fy = row_f - wp.float32(row)

    base = hfd.data_offset
    h00 = hfd.min_z + elevation_data[base + row * hfd.ncol + col] * z_range
    h10 = hfd.min_z + elevation_data[base + row * hfd.ncol + col + 1] * z_range
    h01 = hfd.min_z + elevation_data[base + (row + 1) * hfd.ncol + col] * z_range
    h11 = hfd.min_z + elevation_data[base + (row + 1) * hfd.ncol + col + 1] * z_range

    x0 = -hfd.hx + wp.float32(col) * dx
    y0 = -hfd.hy + wp.float32(row) * dy

    if fx >= fy:
        v0 = wp.vec3(x0, y0, h00)
        e1 = wp.vec3(dx, 0.0, h10 - h00)
        e2 = wp.vec3(dx, dy, h11 - h00)
    else:
        v0 = wp.vec3(x0, y0, h00)
        e1 = wp.vec3(dx, dy, h11 - h00)
        e2 = wp.vec3(0.0, dy, h01 - h00)

    normal = wp.normalize(wp.cross(e1, e2))
    d_plane = wp.dot(pos - v0, normal)
    return d_plane, normal, lateral_dist_sq


@wp.func
def sap_sample_sdf_heightfield(
    hfd: SapHeightfieldData,
    elevation_data: wp.array(dtype=wp.float32),
    pos: wp.vec3,
) -> float:
    """On-the-fly signed distance to a piecewise-planar heightfield surface.

    Positive above the surface, negative below. Exact for the piecewise-linear
    triangulation when the query point projects inside the heightfield XY extent.
    Outside the extent the lateral gap is folded in, yielding a positive distance
    that prevents false contacts.

    Note: This means objects penetrating near the boundary will experience a
    discontinuous contact loss at the edge (the distance jumps from negative to
    positive). This is an intentional tradeoff to avoid ghost contacts outside
    the heightfield footprint.
    """
    d_plane, _normal, lateral_dist_sq = _sap_heightfield_surface_query(hfd, elevation_data, pos)
    if lateral_dist_sq > 0.0:
        return wp.sqrt(lateral_dist_sq + d_plane * d_plane)
    return d_plane


@wp.func
def sap_sample_sdf_grad_heightfield(
    hfd: SapHeightfieldData,
    elevation_data: wp.array(dtype=wp.float32),
    pos: wp.vec3,
) -> tuple[float, wp.vec3]:
    """On-the-fly signed distance and gradient for a heightfield surface.

    Inside the XY extent the gradient is the triangle face normal. Outside,
    it blends the face normal with the lateral displacement direction.
    """
    d_plane, normal, lateral_dist_sq = _sap_heightfield_surface_query(hfd, elevation_data, pos)
    if lateral_dist_sq > 0.0:
        dist = wp.sqrt(lateral_dist_sq + d_plane * d_plane)
        cx = wp.clamp(pos[0], -hfd.hx, hfd.hx)
        cy = wp.clamp(pos[1], -hfd.hy, hfd.hy)
        lateral = wp.vec3(pos[0] - cx, pos[1] - cy, 0.0)
        raw_grad = lateral + d_plane * normal
        if wp.length_sq(raw_grad) > 1.0e-20:
            grad = wp.normalize(raw_grad)
        else:
            grad = wp.vec3(0.0, 0.0, 1.0)
        return dist, grad
    return d_plane, normal


@wp.func
def sap_get_triangle_shape_from_heightfield(
    hfd: SapHeightfieldData,
    elevation_data: wp.array(dtype=wp.float32),
    X_ws: wp.transform,
    tri_idx: int,
) -> tuple[SapGenericShapeData, wp.vec3]:
    """Extract a triangle from a heightfield by packed triangle index.

    ``tri_idx`` encodes ``(row * (ncol - 1) + col) * 2 + tri_sub``.
    Returns ``(SapGenericShapeData, v0_world)`` in the same format as
    :func:`sap_get_triangle_shape_from_mesh`, so GJK/MPR works unchanged.

    Triangle layout for cell (row, col)::

        p01 --- p11
         |  \\ 1  |
         | 0  \\  |
        p00 --- p10

        tri_sub=0: (p00, p10, p11)
        tri_sub=1: (p00, p11, p01)
    """
    # Decode packed triangle index
    cell_idx = tri_idx // 2
    tri_sub = tri_idx - cell_idx * 2
    cols = hfd.ncol - 1
    row = cell_idx // cols
    col = cell_idx - row * cols

    # Grid spacing
    dx = 2.0 * hfd.hx / wp.float32(hfd.ncol - 1)
    dy = 2.0 * hfd.hy / wp.float32(hfd.nrow - 1)
    z_range = hfd.max_z - hfd.min_z

    # Corner positions in local space
    x0 = -hfd.hx + wp.float32(col) * dx
    x1 = x0 + dx
    y0 = -hfd.hy + wp.float32(row) * dy
    y1 = y0 + dy

    # Read elevation values from concatenated array
    base = hfd.data_offset
    h00 = elevation_data[base + row * hfd.ncol + col]
    h10 = elevation_data[base + row * hfd.ncol + (col + 1)]
    h01 = elevation_data[base + (row + 1) * hfd.ncol + col]
    h11 = elevation_data[base + (row + 1) * hfd.ncol + (col + 1)]

    # Convert to world Z: min_z + h * (max_z - min_z)
    z00 = hfd.min_z + h00 * z_range
    z10 = hfd.min_z + h10 * z_range
    z01 = hfd.min_z + h01 * z_range
    z11 = hfd.min_z + h11 * z_range

    # Local-space corner positions
    p00 = wp.vec3(x0, y0, z00)
    p10 = wp.vec3(x1, y0, z10)
    p01 = wp.vec3(x0, y1, z01)
    p11 = wp.vec3(x1, y1, z11)

    # Select triangle vertices
    if tri_sub == 0:
        v0_local = p00
        v1_local = p10
        v2_local = p11
    else:
        v0_local = p00
        v1_local = p11
        v2_local = p01

    # Transform to world space
    v0_world = wp.transform_point(X_ws, v0_local)
    v1_world = wp.transform_point(X_ws, v1_local)
    v2_world = wp.transform_point(X_ws, v2_local)

    # Create triangle shape data (same convention as sap_get_triangle_shape_from_mesh)
    shape_data = SapGenericShapeData()
    shape_data.shape_type = int(SapGeoTypeEx.TRIANGLE)
    shape_data.scale = v1_world - v0_world  # B - A
    shape_data.auxiliary = v2_world - v0_world  # C - A

    return shape_data, v0_world


@wp.func
def sap_heightfield_vs_convex_midphase(
    hfield_shape: int,
    other_shape: int,
    hfd: SapHeightfieldData,
    shape_transform: wp.array(dtype=wp.transform),
    shape_collision_radius: wp.array(dtype=float),
    shape_gap: wp.array(dtype=float),
    triangle_pairs: wp.array(dtype=wp.vec3i),
    triangle_pairs_count: wp.array(dtype=int),
):
    """Find heightfield triangles that overlap with a convex shape's bounding sphere.

    Projects the convex shape onto the heightfield grid and emits triangle pairs
    for each overlapping cell (two triangles per cell).

    Args:
        hfield_shape: Index of the heightfield shape.
        other_shape: Index of the convex shape.
        hfd: Heightfield data struct.
        shape_transform: World-space transforms for all shapes.
        shape_collision_radius: Bounding-sphere radii for all shapes.
        shape_gap: Per-shape contact gaps.
        triangle_pairs: Output buffer for ``(hfield_shape, other_shape, tri_idx)`` triples.
        triangle_pairs_count: Atomic counter for emitted triangle pairs.
    """
    # Transform other shape's position to heightfield local space
    X_hfield_ws = shape_transform[hfield_shape]
    X_hfield_inv = wp.transform_inverse(X_hfield_ws)
    X_other_ws = shape_transform[other_shape]
    pos_in_hfield = wp.transform_point(X_hfield_inv, wp.transform_get_translation(X_other_ws))

    # Conservative AABB using bounding sphere radius
    radius = shape_collision_radius[other_shape]
    gap_sum = shape_gap[hfield_shape] + shape_gap[other_shape]
    extent = radius + gap_sum

    aabb_lower = pos_in_hfield - wp.vec3(extent, extent, extent)
    aabb_upper = pos_in_hfield + wp.vec3(extent, extent, extent)

    # Map AABB to grid cell indices
    dx = 2.0 * hfd.hx / wp.float32(hfd.ncol - 1)
    dy = 2.0 * hfd.hy / wp.float32(hfd.nrow - 1)

    col_min_f = (aabb_lower[0] + hfd.hx) / dx
    col_max_f = (aabb_upper[0] + hfd.hx) / dx
    row_min_f = (aabb_lower[1] + hfd.hy) / dy
    row_max_f = (aabb_upper[1] + hfd.hy) / dy

    col_min = wp.max(wp.int32(wp.floor(col_min_f)), 0)
    col_max = wp.min(wp.int32(wp.floor(col_max_f)), hfd.ncol - 2)
    row_min = wp.max(wp.int32(wp.floor(row_min_f)), 0)
    row_max = wp.min(wp.int32(wp.floor(row_max_f)), hfd.nrow - 2)

    cols = hfd.ncol - 1
    for r in range(row_min, row_max + 1):
        for c in range(col_min, col_max + 1):
            for tri_sub in range(2):
                tri_idx = (r * cols + c) * 2 + tri_sub
                out_idx = wp.atomic_add(triangle_pairs_count, 0, 1)
                if out_idx < triangle_pairs.shape[0]:
                    triangle_pairs[out_idx] = wp.vec3i(hfield_shape, other_shape, tri_idx)

__all__ = [
    "SapHeightfieldData",
    "_sap_heightfield_surface_query",
    "sap_create_empty_heightfield_data",
    "sap_get_triangle_shape_from_heightfield",
    "sap_heightfield_vs_convex_midphase",
    "sap_sample_sdf_grad_heightfield",
    "sap_sample_sdf_heightfield",
]
