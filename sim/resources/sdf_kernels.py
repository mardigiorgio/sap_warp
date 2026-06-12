"""SDF construction kernels used by SAP runtime resources.

Source note: the SAP modifications in this module are based on Newton's SDF
resource/kernel code and adapted for compatibility with imported assets and
collision data.
"""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from sim.collision.types import SapGeoType


SAP_AXIS_X = wp.constant(0)
SAP_AXIS_Y = wp.constant(1)
SAP_AXIS_Z = wp.constant(2)


@wp.func
def sap_sdf_point_to_z_up(point: wp.vec3, up_axis: int):
    if up_axis == int(SAP_AXIS_X):
        return wp.vec3(point[1], point[2], point[0])
    if up_axis == int(SAP_AXIS_Y):
        return wp.vec3(point[0], point[2], point[1])
    return point


@wp.func
def sap_sdf_capped_cone_z(bottom_radius: float, top_radius: float, half_height: float, point_z_up: wp.vec3):
    q = wp.vec2(wp.length(wp.vec2(point_z_up[0], point_z_up[1])), point_z_up[2])
    k1 = wp.vec2(top_radius, half_height)
    k2 = wp.vec2(top_radius - bottom_radius, 2.0 * half_height)

    if q[1] < 0.0:
        ca = wp.vec2(q[0] - wp.min(q[0], bottom_radius), wp.abs(q[1]) - half_height)
    else:
        ca = wp.vec2(q[0] - wp.min(q[0], top_radius), wp.abs(q[1]) - half_height)

    denom = wp.dot(k2, k2)
    t = 0.0
    if denom > 0.0:
        t = wp.clamp(wp.dot(k1 - q, k2) / denom, 0.0, 1.0)
    cb = q - k1 + k2 * t

    sign = 1.0
    if cb[0] < 0.0 and ca[1] < 0.0:
        sign = -1.0

    return sign * wp.sqrt(wp.min(wp.dot(ca, ca), wp.dot(cb, cb)))


@wp.func
def sap_sdf_sphere(point: wp.vec3, radius: float):
    return wp.length(point) - radius


@wp.func
def sap_sdf_sphere_grad(point: wp.vec3, radius: float):
    _ = radius
    eps = 1.0e-8
    p_len = wp.length(point)
    if p_len > eps:
        return point / p_len
    return wp.vec3(0.0, 0.0, 1.0)


@wp.func
def sap_sdf_box(point: wp.vec3, hx: float, hy: float, hz: float):
    qx = wp.abs(point[0]) - hx
    qy = wp.abs(point[1]) - hy
    qz = wp.abs(point[2]) - hz

    e = wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))

    return wp.length(e) + wp.min(wp.max(qx, wp.max(qy, qz)), 0.0)


@wp.func
def sap_sdf_box_grad(point: wp.vec3, hx: float, hy: float, hz: float):
    qx = wp.abs(point[0]) - hx
    qy = wp.abs(point[1]) - hy
    qz = wp.abs(point[2]) - hz

    if qx > 0.0 or qy > 0.0 or qz > 0.0:
        x = wp.clamp(point[0], -hx, hx)
        y = wp.clamp(point[1], -hy, hy)
        z = wp.clamp(point[2], -hz, hz)

        return wp.normalize(point - wp.vec3(x, y, z))

    sx = wp.sign(point[0])
    sy = wp.sign(point[1])
    sz = wp.sign(point[2])

    if (qx > qy and qx > qz) or (qy == 0.0 and qz == 0.0):
        return wp.vec3(sx, 0.0, 0.0)

    if (qy > qx and qy > qz) or (qx == 0.0 and qz == 0.0):
        return wp.vec3(0.0, sy, 0.0)

    return wp.vec3(0.0, 0.0, sz)


@wp.func
def sap_sdf_capsule(point: wp.vec3, radius: float, half_height: float, up_axis: int = int(SAP_AXIS_Y)):
    point_z_up = sap_sdf_point_to_z_up(point, up_axis)
    if point_z_up[2] > half_height:
        return wp.length(wp.vec3(point_z_up[0], point_z_up[1], point_z_up[2] - half_height)) - radius

    if point_z_up[2] < -half_height:
        return wp.length(wp.vec3(point_z_up[0], point_z_up[1], point_z_up[2] + half_height)) - radius

    return wp.length(wp.vec3(point_z_up[0], point_z_up[1], 0.0)) - radius


@wp.func
def sap_sdf_vector_from_z_up(v: wp.vec3, up_axis: int):
    if up_axis == int(SAP_AXIS_X):
        return wp.vec3(v[2], v[0], v[1])
    if up_axis == int(SAP_AXIS_Y):
        return wp.vec3(v[0], v[2], v[1])
    return v


@wp.func
def sap_sdf_capsule_grad(point: wp.vec3, radius: float, half_height: float, up_axis: int = int(SAP_AXIS_Y)):
    _ = radius
    eps = 1.0e-8
    point_z_up = sap_sdf_point_to_z_up(point, up_axis)
    grad_z_up = wp.vec3()
    if point_z_up[2] > half_height:
        v = wp.vec3(point_z_up[0], point_z_up[1], point_z_up[2] - half_height)
        v_len = wp.length(v)
        grad_z_up = wp.vec3(0.0, 0.0, 1.0)
        if v_len > eps:
            grad_z_up = v / v_len
    elif point_z_up[2] < -half_height:
        v = wp.vec3(point_z_up[0], point_z_up[1], point_z_up[2] + half_height)
        v_len = wp.length(v)
        grad_z_up = wp.vec3(0.0, 0.0, -1.0)
        if v_len > eps:
            grad_z_up = v / v_len
    else:
        v = wp.vec3(point_z_up[0], point_z_up[1], 0.0)
        v_len = wp.length(v)
        grad_z_up = wp.vec3(0.0, 0.0, 1.0)
        if v_len > eps:
            grad_z_up = v / v_len
    return sap_sdf_vector_from_z_up(grad_z_up, up_axis)


@wp.func
def sap_sdf_cylinder(
    point: wp.vec3,
    radius: float,
    half_height: float,
    up_axis: int = int(SAP_AXIS_Y),
    top_radius: float = -1.0,
):
    point_z_up = sap_sdf_point_to_z_up(point, up_axis)
    if top_radius < 0.0 or wp.abs(top_radius - radius) <= 1.0e-6:
        dx = wp.length(wp.vec3(point_z_up[0], point_z_up[1], 0.0)) - radius
        dy = wp.abs(point_z_up[2]) - half_height
        return wp.min(wp.max(dx, dy), 0.0) + wp.length(wp.vec2(wp.max(dx, 0.0), wp.max(dy, 0.0)))
    return sap_sdf_capped_cone_z(radius, top_radius, half_height, point_z_up)


@wp.func
def sap_sdf_cylinder_grad(
    point: wp.vec3,
    radius: float,
    half_height: float,
    up_axis: int = int(SAP_AXIS_Y),
    top_radius: float = -1.0,
):
    eps = 1.0e-8
    point_z_up = sap_sdf_point_to_z_up(point, up_axis)
    if top_radius >= 0.0 and wp.abs(top_radius - radius) > 1.0e-6:
        fd_eps = 1.0e-4
        dx = sap_sdf_capped_cone_z(
            radius,
            top_radius,
            half_height,
            point_z_up + wp.vec3(fd_eps, 0.0, 0.0),
        ) - sap_sdf_capped_cone_z(
            radius,
            top_radius,
            half_height,
            point_z_up - wp.vec3(fd_eps, 0.0, 0.0),
        )
        dy = sap_sdf_capped_cone_z(
            radius,
            top_radius,
            half_height,
            point_z_up + wp.vec3(0.0, fd_eps, 0.0),
        ) - sap_sdf_capped_cone_z(
            radius,
            top_radius,
            half_height,
            point_z_up - wp.vec3(0.0, fd_eps, 0.0),
        )
        dz = sap_sdf_capped_cone_z(
            radius,
            top_radius,
            half_height,
            point_z_up + wp.vec3(0.0, 0.0, fd_eps),
        ) - sap_sdf_capped_cone_z(
            radius,
            top_radius,
            half_height,
            point_z_up - wp.vec3(0.0, 0.0, fd_eps),
        )
        grad_z_up = wp.vec3(dx, dy, dz)
        grad_len = wp.length(grad_z_up)
        if grad_len > eps:
            grad_z_up = grad_z_up / grad_len
        else:
            grad_z_up = wp.vec3(0.0, 0.0, 1.0)
        return sap_sdf_vector_from_z_up(grad_z_up, up_axis)

    v = wp.vec3(point_z_up[0], point_z_up[1], 0.0)
    v_len = wp.length(v)
    radial = wp.vec3(0.0, 0.0, 1.0)
    if v_len > eps:
        radial = v / v_len
    axial = wp.vec3(0.0, 0.0, wp.sign(point_z_up[2]))
    dx = v_len - radius
    dy = wp.abs(point_z_up[2]) - half_height
    grad_z_up = wp.vec3()
    if dx > 0.0 and dy > 0.0:
        g = radial * dx + axial * dy
        g_len = wp.length(g)
        if g_len > eps:
            grad_z_up = g / g_len
        else:
            grad_z_up = radial
    elif dx > dy:
        grad_z_up = radial
    else:
        grad_z_up = axial
    return sap_sdf_vector_from_z_up(grad_z_up, up_axis)


@wp.func
def sap_sdf_ellipsoid(point: wp.vec3, radii: wp.vec3):
    eps = 1.0e-8
    r = wp.vec3(
        wp.max(wp.abs(radii[0]), eps),
        wp.max(wp.abs(radii[1]), eps),
        wp.max(wp.abs(radii[2]), eps),
    )
    inv_r = wp.cw_div(wp.vec3(1.0, 1.0, 1.0), r)
    inv_r2 = wp.cw_mul(inv_r, inv_r)
    q0 = wp.cw_mul(point, inv_r)
    q1 = wp.cw_mul(point, inv_r2)
    k0 = wp.length(q0)
    k1 = wp.length(q1)
    if k1 > eps:
        return k0 * (k0 - 1.0) / k1
    return -wp.min(wp.min(r[0], r[1]), r[2])


@wp.func
def sap_sdf_ellipsoid_grad(point: wp.vec3, radii: wp.vec3):
    eps = 1.0e-8
    r = wp.vec3(
        wp.max(wp.abs(radii[0]), eps),
        wp.max(wp.abs(radii[1]), eps),
        wp.max(wp.abs(radii[2]), eps),
    )
    inv_r = wp.cw_div(wp.vec3(1.0, 1.0, 1.0), r)
    inv_r2 = wp.cw_mul(inv_r, inv_r)
    q0 = wp.cw_mul(point, inv_r)
    q1 = wp.cw_mul(point, inv_r2)
    k0 = wp.length(q0)
    k1 = wp.length(q1)
    if k1 < eps:
        return wp.vec3(0.0, 0.0, 1.0)
    grad = q1 * (k0 / k1)
    grad_len = wp.length(grad)
    if grad_len > eps:
        return grad / grad_len
    return wp.vec3(0.0, 0.0, 1.0)


@wp.func
def sap_sdf_cone(point: wp.vec3, radius: float, half_height: float, up_axis: int = int(SAP_AXIS_Y)):
    point_z_up = sap_sdf_point_to_z_up(point, up_axis)
    return sap_sdf_capped_cone_z(radius, 0.0, half_height, point_z_up)


@wp.func
def sap_sdf_cone_grad(point: wp.vec3, radius: float, half_height: float, up_axis: int = int(SAP_AXIS_Y)):
    point_z_up = sap_sdf_point_to_z_up(point, up_axis)
    if half_height <= 0.0:
        return sap_sdf_vector_from_z_up(wp.vec3(0.0, 0.0, wp.sign(point_z_up[2])), up_axis)

    r = wp.length(wp.vec3(point_z_up[0], point_z_up[1], 0.0))
    dx = r - radius * (half_height - point_z_up[2]) / (2.0 * half_height)
    dy = wp.abs(point_z_up[2]) - half_height
    grad_z_up = wp.vec3()
    if dx > dy:
        if r > 0.0:
            radial_dir = wp.vec3(point_z_up[0], point_z_up[1], 0.0) / r
            grad_z_up = wp.normalize(radial_dir + wp.vec3(0.0, 0.0, radius / (2.0 * half_height)))
        else:
            grad_z_up = wp.vec3(0.0, 0.0, 1.0)
    else:
        grad_z_up = wp.vec3(0.0, 0.0, wp.sign(point_z_up[2]))
    return sap_sdf_vector_from_z_up(grad_z_up, up_axis)


@wp.func
def sap_sdf_plane(point: wp.vec3, width: float, length: float):
    if width > 0.0 and length > 0.0:
        d = wp.max(wp.abs(point[0]) - width, wp.abs(point[1]) - length)
        return wp.max(d, wp.abs(point[2]))
    return point[2]


@wp.func
def sap_sdf_plane_grad(point: wp.vec3, width: float, length: float):
    _ = (width, length, point)
    return wp.vec3(0.0, 0.0, 1.0)


def sap_get_primitive_extents(shape_type: int, shape_scale: Sequence[float]) -> tuple[list[float], list[float]]:
    if shape_type == SapGeoType.SPHERE:
        min_ext = [-shape_scale[0], -shape_scale[0], -shape_scale[0]]
        max_ext = [shape_scale[0], shape_scale[0], shape_scale[0]]
    elif shape_type == SapGeoType.BOX:
        min_ext = [-shape_scale[0], -shape_scale[1], -shape_scale[2]]
        max_ext = [shape_scale[0], shape_scale[1], shape_scale[2]]
    elif shape_type == SapGeoType.CAPSULE:
        min_ext = [-shape_scale[0], -shape_scale[0], -shape_scale[1] - shape_scale[0]]
        max_ext = [shape_scale[0], shape_scale[0], shape_scale[1] + shape_scale[0]]
    elif shape_type == SapGeoType.CYLINDER:
        min_ext = [-shape_scale[0], -shape_scale[0], -shape_scale[1]]
        max_ext = [shape_scale[0], shape_scale[0], shape_scale[1]]
    elif shape_type == SapGeoType.ELLIPSOID:
        min_ext = [-shape_scale[0], -shape_scale[1], -shape_scale[2]]
        max_ext = [shape_scale[0], shape_scale[1], shape_scale[2]]
    elif shape_type == SapGeoType.CONE:
        min_ext = [-shape_scale[0], -shape_scale[0], -shape_scale[1]]
        max_ext = [shape_scale[0], shape_scale[0], shape_scale[1]]
    else:
        raise NotImplementedError(f"Extents not implemented for shape type: {shape_type}")
    return min_ext, max_ext


__all__ = [
    "SAP_AXIS_X",
    "SAP_AXIS_Y",
    "SAP_AXIS_Z",
    "sap_get_primitive_extents",
    "sap_sdf_box",
    "sap_sdf_box_grad",
    "sap_sdf_capsule",
    "sap_sdf_capsule_grad",
    "sap_sdf_cone",
    "sap_sdf_cone_grad",
    "sap_sdf_cylinder",
    "sap_sdf_cylinder_grad",
    "sap_sdf_ellipsoid",
    "sap_sdf_ellipsoid_grad",
    "sap_sdf_plane",
    "sap_sdf_plane_grad",
    "sap_sdf_sphere",
    "sap_sdf_sphere_grad",
]
