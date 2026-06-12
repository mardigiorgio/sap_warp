"""Conversion and material helpers shared by the SAP runtime stages. The documented helpers move
generalized velocities and forces between the public model convention and the angular-first SAP
convention used internally by free motion, contact Jacobian assembly, and contact solve code.

Source note: the SAP modifications in this module are based on Newton's
runtime helper code and adapted so SAP Warp can wrap Newton-owned runtime
arrays while preserving public generalized-coordinate and force ordering.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from sim.sap_runtime import (
    Model,
    SAP_JOINT_BALL,
    SAP_JOINT_D6,
    SAP_JOINT_DISTANCE,
    SAP_JOINT_FIXED,
    SAP_JOINT_FREE,
    SAP_JOINT_PRISMATIC,
    SAP_JOINT_REVOLUTE,
)

_PI = 3.141592653589793
_CONTACT_SOFT_NORM_TOL = 1.0e-7
_SAP_PD_BETA = 0.1


def _shape_mu_numpy(model: Model) -> np.ndarray:
    arr = model.shape_material_mu
    if arr is None:
        return np.empty((0,), dtype=np.float32)
    try:
        return np.asarray(arr.numpy(), dtype=np.float32).reshape(-1)
    except Exception:
        return np.empty((0,), dtype=np.float32)


def _infer_contact_mu_fallback(model: Model, default: float = 0.1) -> float:
    mu_np = _shape_mu_numpy(model)
    positive = mu_np[mu_np > 0.0]
    if positive.size == 0:
        return float(default)
    return float(np.min(positive))


def _resolve_contact_shape_mu(model: Model, mu: float | None) -> wp.array:
    model_mu = model.shape_material_mu
    mu_np = _shape_mu_numpy(model)
    fallback = float(mu) if mu is not None else _infer_contact_mu_fallback(model)
    device = getattr(model_mu, "device", model.device)
    if model_mu is None or mu_np.size == 0:
        return wp.full((0,), fallback, dtype=wp.float32, device=device)

    if mu is not None and bool(np.allclose(mu_np, mu_np[0], rtol=0.0, atol=1.0e-7)):
        return wp.full(mu_np.shape, fallback, dtype=wp.float32, device=device)
    return model_mu


def _shape_stiffness_numpy(model: Model) -> np.ndarray:
    arr = model.shape_material_ke
    if arr is None:
        return np.empty((0,), dtype=np.float32)
    try:
        return np.asarray(arr.numpy(), dtype=np.float32).reshape(-1)
    except Exception:
        return np.empty((0,), dtype=np.float32)


def _resolve_contact_shape_stiffness(model: Model, fallback_k: float) -> wp.array:
    model_ke = model.shape_material_ke
    ke_np = _shape_stiffness_numpy(model)
    device = getattr(model_ke, "device", model.device)
    if model_ke is None or ke_np.size == 0:
        return wp.full((max(int(model.shape_count), 1),), float(fallback_k), dtype=wp.float32, device=device)
    return model_ke


@wp.func
def _v3d_zero() -> wp.vec3d:
    return wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))


@wp.func
def _vec3_to_vec3d(v: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(v.x), wp.float64(v.y), wp.float64(v.z))


@wp.func
def _safe_normalize(v: wp.vec3d) -> wp.vec3d:
    n2 = wp.dot(v, v)
    if n2 <= wp.float64(1.0e-32):
        return _v3d_zero()
    return v / wp.sqrt(n2)


@wp.func
def _v3_zero_f32() -> wp.vec3:
    return wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))


@wp.func
def _safe_normalize_f32(v: wp.vec3) -> wp.vec3:
    n2 = wp.dot(v, v)
    if n2 <= wp.float32(1.0e-32):
        return _v3_zero_f32()
    return v / wp.sqrt(n2)


@wp.func
def _quat_rotate_vec3_f32(q: wp.quatd, v: wp.vec3) -> wp.vec3:
    qx = wp.float32(q.x)
    qy = wp.float32(q.y)
    qz = wp.float32(q.z)
    qw = wp.float32(q.w)
    norm = wp.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= wp.float32(0.0):
        return v
    inv_norm = wp.float32(1.0) / norm
    x = qx * inv_norm
    y = qy * inv_norm
    z = qz * inv_norm
    w = qw * inv_norm
    qv = wp.vec3(x, y, z)
    t = wp.float32(2.0) * wp.cross(qv, v)
    return v + w * t + wp.cross(qv, t)


@wp.func
def _quat_inverse_f32(q: wp.quatd) -> wp.quatd:
    qx = wp.float32(q.x)
    qy = wp.float32(q.y)
    qz = wp.float32(q.z)
    qw = wp.float32(q.w)
    norm2 = qx * qx + qy * qy + qz * qz + qw * qw
    if norm2 <= wp.float32(0.0):
        return wp.quatd(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    inv_norm2 = wp.float32(1.0) / norm2
    return wp.quatd(
        wp.float64(-qx * inv_norm2),
        wp.float64(-qy * inv_norm2),
        wp.float64(-qz * inv_norm2),
        wp.float64(qw * inv_norm2),
    )


@wp.func
def _shape_mu_or_fallback(
    shape_material_mu: wp.array(dtype=wp.float64),
    shape: int,
    fallback_mu: float,
) -> wp.float64:
    value = wp.float64(fallback_mu)
    if shape >= 0:
        value = wp.float64(shape_material_mu[shape])
    if value < wp.float64(0.0):
        value = wp.float64(0.0)
    return value


@wp.func
def _sap_combine_mu(mu0: wp.float64, mu1: wp.float64) -> wp.float64:
    denom = mu0 + mu1
    if denom <= wp.float64(1.0e-12):
        return wp.float64(0.0)
    return wp.float64(2.0) * mu0 * mu1 / denom


@wp.func
def _contact_mu_pair(
    shape_material_mu: wp.array(dtype=wp.float64),
    shape0: int,
    shape1: int,
    fallback_mu: float,
) -> wp.float64:
    mu0 = _shape_mu_or_fallback(shape_material_mu, shape0, fallback_mu)
    mu1 = _shape_mu_or_fallback(shape_material_mu, shape1, fallback_mu)
    return _sap_combine_mu(mu0, mu1)


@wp.func
def _shape_stiffness_or_fallback(
    shape_material_ke: wp.array(dtype=wp.float64),
    shape: int,
    fallback_k: float,
) -> wp.float64:
    value = wp.float64(fallback_k)
    if shape >= 0:
        value = wp.float64(shape_material_ke[shape])
    if not wp.isfinite(value) or value <= wp.float64(0.0):
        value = wp.float64(fallback_k)
    if not wp.isfinite(value) or value <= wp.float64(0.0):
        value = wp.float64(1.0)
    return value


@wp.func
def _sap_combine_stiffness(k0: wp.float64, k1: wp.float64) -> wp.float64:
    if not wp.isfinite(k0):
        return k1
    if not wp.isfinite(k1):
        return k0
    denom = k0 + k1
    if denom <= wp.float64(1.0e-12):
        return wp.max(k0, k1)
    return k0 * k1 / denom


@wp.func
def _shape_tau_or_fallback(
    shape_material_tau: wp.array(dtype=wp.float64),
    shape: int,
    fallback_tau: float,
) -> wp.float64:
    value = wp.float64(fallback_tau)
    if shape >= 0:
        value = wp.float64(shape_material_tau[shape])
    if not wp.isfinite(value) or value < wp.float64(0.0):
        value = wp.float64(fallback_tau)
    if not wp.isfinite(value) or value < wp.float64(0.0):
        value = wp.float64(0.0)
    return value


@wp.func
def _contact_tau_pair(
    shape_material_tau: wp.array(dtype=wp.float64),
    shape0: int,
    shape1: int,
    fallback_tau: float,
) -> wp.float64:
    return _shape_tau_or_fallback(shape_material_tau, shape0, fallback_tau) + _shape_tau_or_fallback(
        shape_material_tau,
        shape1,
        fallback_tau,
    )


@wp.func
def _quat_rotate_vec3d(q: wp.quatd, v: wp.vec3d) -> wp.vec3d:
    norm = wp.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm <= wp.float64(0.0):
        return v
    inv_norm = wp.float64(1.0) / norm
    x = q.x * inv_norm
    y = q.y * inv_norm
    z = q.z * inv_norm
    w = q.w * inv_norm
    qv = wp.vec3d(x, y, z)
    t = wp.float64(2.0) * wp.cross(qv, v)
    return v + w * t + wp.cross(qv, t)


@wp.func
def _quat_inverse_unit_d(q: wp.quatd) -> wp.quatd:
    norm2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w
    if norm2 <= wp.float64(0.0):
        return wp.quatd(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))
    inv_norm2 = wp.float64(1.0) / norm2
    return wp.quatd(-q.x * inv_norm2, -q.y * inv_norm2, -q.z * inv_norm2, q.w * inv_norm2)


@wp.func
def _transform_point_d(x: wp.transformd, p_local: wp.vec3d) -> wp.vec3d:
    return wp.transform_get_translation(x) + _quat_rotate_vec3d(wp.transform_get_rotation(x), p_local)


@wp.func
def _quatd_from_quat(q: wp.quat) -> wp.quatd:
    return wp.quatd(wp.float64(q.x), wp.float64(q.y), wp.float64(q.z), wp.float64(q.w))


@wp.func
def _transform_point_f32_pose_d(x: wp.transform, p_local: wp.vec3d) -> wp.vec3d:
    p = wp.transform_get_translation(x)
    return wp.vec3d(wp.float64(p.x), wp.float64(p.y), wp.float64(p.z)) + _quat_rotate_vec3d(
        _quatd_from_quat(wp.transform_get_rotation(x)),
        p_local,
    )


@wp.func
def _transform_translation_f32_pose_d(x: wp.transform) -> wp.vec3d:
    p = wp.transform_get_translation(x)
    return wp.vec3d(wp.float64(p.x), wp.float64(p.y), wp.float64(p.z))


@wp.func
def _transform_point_f32_pose(x: wp.transform, p_local: wp.vec3) -> wp.vec3:
    return wp.transform_get_translation(x) + _quat_rotate_vec3_f32(
        _quatd_from_quat(wp.transform_get_rotation(x)),
        p_local,
    )


@wp.func
def _body_com_world_d(
    body: int,
    body_q: wp.array(dtype=wp.transformd),
    body_com: wp.array(dtype=wp.vec3d),
) -> wp.vec3d:
    x = body_q[body]
    return wp.transform_get_translation(x) + _quat_rotate_vec3d(wp.transform_get_rotation(x), body_com[body])


@wp.func
def _sap_body_direction_weight_d(
    body: int,
    r_world: wp.vec3d,
    direction_world: wp.vec3d,
    body_q: wp.array(dtype=wp.transformd),
    body_mass: wp.array(dtype=wp.float64),
    body_inertia: wp.array(dtype=wp.mat33d),
) -> wp.float64:
    if body < 0:
        return wp.float64(0.0)

    mass = body_mass[body]
    if mass <= wp.float64(1.0e-12):
        return wp.float64(0.0)

    c_world = wp.cross(r_world, direction_world)
    q_inv = _quat_inverse_unit_d(wp.transform_get_rotation(body_q[body]))
    c_body = _quat_rotate_vec3d(q_inv, c_world)
    inertia = body_inertia[body]
    ix = inertia[0, 0]
    iy = inertia[1, 1]
    iz = inertia[2, 2]
    if ix < wp.float64(1.0e-12):
        ix = wp.float64(1.0e-12)
    if iy < wp.float64(1.0e-12):
        iy = wp.float64(1.0e-12)
    if iz < wp.float64(1.0e-12):
        iz = wp.float64(1.0e-12)

    angular = c_body.x * c_body.x / ix + c_body.y * c_body.y / iy + c_body.z * c_body.z / iz
    return wp.float64(1.0) / mass + angular


@wp.func
def _body_com_world_f32(
    body: int,
    body_q: wp.array(dtype=wp.transformd),
    body_com: wp.array(dtype=wp.vec3d),
) -> wp.vec3:
    x = body_q[body]
    p = wp.transform_get_translation(x)
    com = body_com[body]
    return wp.vec3(wp.float32(p.x), wp.float32(p.y), wp.float32(p.z)) + _quat_rotate_vec3_f32(
        wp.transform_get_rotation(x),
        wp.vec3(wp.float32(com.x), wp.float32(com.y), wp.float32(com.z)),
    )


@wp.func
def _body_com_world_f32_pose(
    body: int,
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3d),
) -> wp.vec3:
    x = body_q[body]
    p = wp.transform_get_translation(x)
    com = body_com[body]
    return p + _quat_rotate_vec3_f32(
        _quatd_from_quat(wp.transform_get_rotation(x)),
        wp.vec3(wp.float32(com.x), wp.float32(com.y), wp.float32(com.z)),
    )


@wp.func
def _sap_body_direction_weight_f32(
    body: int,
    r_world: wp.vec3,
    direction_world: wp.vec3,
    body_q: wp.array(dtype=wp.transformd),
    body_mass: wp.array(dtype=wp.float64),
    body_inertia: wp.array(dtype=wp.mat33d),
) -> wp.float32:
    if body < 0:
        return wp.float32(0.0)

    mass = wp.float32(body_mass[body])
    if mass <= wp.float32(1.0e-12):
        return wp.float32(0.0)

    c_world = wp.cross(r_world, direction_world)
    q_inv = _quat_inverse_f32(wp.transform_get_rotation(body_q[body]))
    c_body = _quat_rotate_vec3_f32(q_inv, c_world)
    inertia = body_inertia[body]
    ix = wp.float32(inertia[0, 0])
    iy = wp.float32(inertia[1, 1])
    iz = wp.float32(inertia[2, 2])
    if ix < wp.float32(1.0e-12):
        ix = wp.float32(1.0e-12)
    if iy < wp.float32(1.0e-12):
        iy = wp.float32(1.0e-12)
    if iz < wp.float32(1.0e-12):
        iz = wp.float32(1.0e-12)

    angular = c_body.x * c_body.x / ix + c_body.y * c_body.y / iy + c_body.z * c_body.z / iz
    return wp.float32(1.0) / mass + angular


@wp.func
def _sap_body_direction_weight_f32_pose(
    body: int,
    r_world: wp.vec3,
    direction_world: wp.vec3,
    body_q: wp.array(dtype=wp.transform),
    body_mass: wp.array(dtype=wp.float64),
    body_inertia: wp.array(dtype=wp.mat33d),
) -> wp.float32:
    if body < 0:
        return wp.float32(0.0)

    mass = wp.float32(body_mass[body])
    if mass <= wp.float32(1.0e-12):
        return wp.float32(0.0)

    c_world = wp.cross(r_world, direction_world)
    q_inv = _quat_inverse_f32(_quatd_from_quat(wp.transform_get_rotation(body_q[body])))
    c_body = _quat_rotate_vec3_f32(q_inv, c_world)
    inertia = body_inertia[body]
    ix = wp.float32(inertia[0, 0])
    iy = wp.float32(inertia[1, 1])
    iz = wp.float32(inertia[2, 2])
    if ix < wp.float32(1.0e-12):
        ix = wp.float32(1.0e-12)
    if iy < wp.float32(1.0e-12):
        iy = wp.float32(1.0e-12)
    if iz < wp.float32(1.0e-12):
        iz = wp.float32(1.0e-12)

    angular = c_body.x * c_body.x / ix + c_body.y * c_body.y / iy + c_body.z * c_body.z / iz
    return wp.float32(1.0) / mass + angular


@wp.func
def _contact_env_from_shapes(
    bodies_per_env: int,
    shape_body: wp.array(dtype=int),
    shape0: int,
    shape1: int,
) -> int:
    b0 = -1
    b1 = -1
    if shape0 >= 0:
        b0 = shape_body[shape0]
    if shape1 >= 0:
        b1 = shape_body[shape1]

    env = -1
    if b0 >= 0:
        env = b0 // bodies_per_env
    elif b1 >= 0:
        env = b1 // bodies_per_env
    if env < 0:
        return -1
    if b0 >= 0 and b0 // bodies_per_env != env:
        return -1
    if b1 >= 0 and b1 // bodies_per_env != env:
        return -1
    return env


@wp.func
def _sap_make_from_one_unit_vector_z(n: wp.vec3d):
    abs_x = wp.abs(n.x)
    abs_y = wp.abs(n.y)
    abs_z = wp.abs(n.z)
    i = int(0)
    min_abs = abs_x
    if abs_y < min_abs:
        i = int(1)
        min_abs = abs_y
    if abs_z < min_abs:
        i = int(2)
        min_abs = abs_z

    mag = wp.sqrt(wp.float64(1.0) - min_abs * min_abs)
    r = wp.float64(1.0) / mag
    u_min = n.x
    if i == 1:
        u_min = n.y
    elif i == 2:
        u_min = n.z
    s = -r * u_min

    v = _v3d_zero()
    w = _v3d_zero()

    if i == 0:
        v = wp.vec3d(wp.float64(0.0), -r * n.z, r * n.y)
        w = wp.vec3d(mag, s * n.y, s * n.z)
    elif i == 1:
        v = wp.vec3d(r * n.z, wp.float64(0.0), -r * n.x)
        w = wp.vec3d(s * n.x, mag, s * n.z)
    else:
        v = wp.vec3d(-r * n.y, r * n.x, wp.float64(0.0))
        w = wp.vec3d(s * n.x, s * n.y, mag)

    return v, w, n


@wp.func
def _sap_make_from_one_unit_vector_z_f32(n: wp.vec3):
    abs_x = wp.abs(n.x)
    abs_y = wp.abs(n.y)
    abs_z = wp.abs(n.z)
    i = int(0)
    min_abs = abs_x
    if abs_y < min_abs:
        i = int(1)
        min_abs = abs_y
    if abs_z < min_abs:
        i = int(2)
        min_abs = abs_z

    mag = wp.sqrt(wp.float32(1.0) - min_abs * min_abs)
    r = wp.float32(1.0) / mag
    u_min = n.x
    if i == 1:
        u_min = n.y
    elif i == 2:
        u_min = n.z
    s = -r * u_min

    v = _v3_zero_f32()
    w = _v3_zero_f32()
    if i == 0:
        v = wp.vec3(wp.float32(0.0), -r * n.z, r * n.y)
        w = wp.vec3(mag, s * n.y, s * n.z)
    elif i == 1:
        v = wp.vec3(r * n.z, wp.float32(0.0), -r * n.x)
        w = wp.vec3(s * n.x, mag, s * n.z)
    else:
        v = wp.vec3(-r * n.y, r * n.x, wp.float32(0.0))
        w = wp.vec3(s * n.x, s * n.y, mag)
    return v, w, n


@wp.func
def _vec3d_from_vec3(v: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(v.x), wp.float64(v.y), wp.float64(v.z))


@wp.func
def _transformd_from_transform(x: wp.transform) -> wp.transformd:
    return wp.transformd(
        p=_vec3d_from_vec3(wp.transform_get_translation(x)),
        q=_quatd_from_quat(wp.transform_get_rotation(x)),
    )


@wp.func
def _mat33d_from_mat33(m: wp.mat33) -> wp.mat33d:
    return wp.mat33d(
        wp.float64(m[0, 0]), wp.float64(m[0, 1]), wp.float64(m[0, 2]),
        wp.float64(m[1, 0]), wp.float64(m[1, 1]), wp.float64(m[1, 2]),
        wp.float64(m[2, 0]), wp.float64(m[2, 1]), wp.float64(m[2, 2]),
    )


@wp.func
def _vec3d_zero() -> wp.vec3d:
    return wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))


@wp.func
def _quatd_identity() -> wp.quatd:
    return wp.quatd(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))


@wp.func
def _quatd_from_axis_angle(axis: wp.vec3d, angle: wp.float64) -> wp.quatd:
    half = wp.float64(0.5) * angle
    s = wp.sin(half)
    c = wp.cos(half)
    return wp.quatd(axis[0] * s, axis[1] * s, axis[2] * s, c)


@wp.func
def _quatd_mul(a: wp.quatd, b: wp.quatd) -> wp.quatd:
    return wp.quatd(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    )


@wp.func
def _quatd_conjugate(q: wp.quatd) -> wp.quatd:
    return wp.quatd(-q.x, -q.y, -q.z, q.w)


@wp.func
def _quatd_normalize(q: wp.quatd) -> wp.quatd:
    norm = wp.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm <= wp.float64(0.0):
        return _quatd_identity()
    inv_norm = wp.float64(1.0) / norm
    return wp.quatd(q.x * inv_norm, q.y * inv_norm, q.z * inv_norm, q.w * inv_norm)


@wp.func
def _quatd_from_matrix_cols(c0: wp.vec3d, c1: wp.vec3d, c2: wp.vec3d) -> wp.quatd:
    r00 = c0[0]
    r10 = c0[1]
    r20 = c0[2]
    r01 = c1[0]
    r11 = c1[1]
    r21 = c1[2]
    r02 = c2[0]
    r12 = c2[1]
    r22 = c2[2]
    trace = r00 + r11 + r22
    q = _quatd_identity()
    if trace > wp.float64(0.0):
        s = wp.sqrt(trace + wp.float64(1.0)) * wp.float64(2.0)
        q = wp.quatd(
            (r21 - r12) / s,
            (r02 - r20) / s,
            (r10 - r01) / s,
            wp.float64(0.25) * s,
        )
    elif r00 > r11 and r00 > r22:
        s = wp.sqrt(wp.float64(1.0) + r00 - r11 - r22) * wp.float64(2.0)
        q = wp.quatd(
            wp.float64(0.25) * s,
            (r01 + r10) / s,
            (r02 + r20) / s,
            (r21 - r12) / s,
        )
    elif r11 > r22:
        s = wp.sqrt(wp.float64(1.0) + r11 - r00 - r22) * wp.float64(2.0)
        q = wp.quatd(
            (r01 + r10) / s,
            wp.float64(0.25) * s,
            (r12 + r21) / s,
            (r02 - r20) / s,
        )
    else:
        s = wp.sqrt(wp.float64(1.0) + r22 - r00 - r11) * wp.float64(2.0)
        q = wp.quatd(
            (r02 + r20) / s,
            (r12 + r21) / s,
            wp.float64(0.25) * s,
            (r10 - r01) / s,
        )
    return _quatd_normalize(q)


@wp.func
def _quatd_rotate(q: wp.quatd, v: wp.vec3d) -> wp.vec3d:
    q = _quatd_normalize(q)
    qv = wp.vec3d(q.x, q.y, q.z)
    t = wp.float64(2.0) * wp.cross(qv, v)
    return v + q.w * t + wp.cross(qv, t)


@wp.func
def _quatd_rotate_unit(q: wp.quatd, v: wp.vec3d) -> wp.vec3d:
    qv = wp.vec3d(q.x, q.y, q.z)
    t = wp.float64(2.0) * wp.cross(qv, v)
    return v + q.w * t + wp.cross(qv, t)


@wp.func
def _transformd_compose(a: wp.transformd, b: wp.transformd) -> wp.transformd:
    ap = wp.transform_get_translation(a)
    aq = _quatd_normalize(wp.transform_get_rotation(a))
    bp = wp.transform_get_translation(b)
    bq = _quatd_normalize(wp.transform_get_rotation(b))
    return wp.transformd(ap + _quatd_rotate_unit(aq, bp), _quatd_normalize(_quatd_mul(aq, bq)))


@wp.func
def _transformd_inverse(x: wp.transformd) -> wp.transformd:
    p = wp.transform_get_translation(x)
    q_inv = _quatd_conjugate(_quatd_normalize(wp.transform_get_rotation(x)))
    return wp.transformd(-_quatd_rotate_unit(q_inv, p), q_inv)


@wp.func
def _transformd_identity() -> wp.transformd:
    return wp.transformd(p=_vec3d_zero(), q=_quatd_identity())


@wp.func
def _sap_compose_child_body_transform(
    X_WF: wp.transformd,
    X_FM: wp.transformd,
    X_BM: wp.transformd,
    child_frame_identity: int,
) -> wp.transformd:
    X_WM = _transformd_compose(X_WF, X_FM)
    if child_frame_identity != 0:
        return X_WM
    return _transformd_compose(X_WM, _transformd_inverse(X_BM))


@wp.func
def _sap_revolute_identity_child_kinematics(
    joint_axis: wp.array(dtype=wp.vec3d),
    X_WF: wp.transformd,
    joint_q: wp.array(dtype=wp.float64),
    q_start: int,
    joint_qd_sap: wp.array(dtype=wp.float64),
    qd_start: int,
    joint_S_s: wp.array(dtype=wp.spatial_vectord),
):
    axis = joint_axis[qd_start]
    q_FM = _quatd_from_axis_angle(axis, wp.float64(joint_q[q_start]))
    R_WF = wp.transform_get_rotation(X_WF)
    X_WB = wp.transformd(wp.transform_get_translation(X_WF), _quatd_mul(R_WF, q_FM))
    S_s = _sap_spatial(_quatd_rotate_unit(R_WF, axis), _vec3d_zero())
    joint_S_s[qd_start] = S_s
    return X_WB, S_s * joint_qd_sap[qd_start], _sap_spatial(_vec3d_zero(), _vec3d_zero())


@wp.func
def _sap_revolute_identity_child_kinematics_f32_math(
    joint_axis: wp.array(dtype=wp.vec3d),
    X_WF: wp.transformd,
    joint_q: wp.array(dtype=wp.float64),
    q_start: int,
    joint_qd_sap: wp.array(dtype=wp.float64),
    qd_start: int,
    joint_S_s: wp.array(dtype=wp.spatial_vectord),
):
    axis = _vec3f_from_vec3d(joint_axis[qd_start])
    q_FM = _quatf_from_axis_angle(axis, wp.float32(joint_q[q_start]))
    R_WF = _quatf_from_quatd(wp.transform_get_rotation(X_WF))
    X_WB = wp.transformd(
        _vec3d_from_vec3f(_vec3f_from_vec3d(wp.transform_get_translation(X_WF))),
        _quatd_from_quatf(_quatf_mul(R_WF, q_FM)),
    )
    S_f = _sap_spatialf(_quatf_rotate_unit(R_WF, axis), wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)))
    S_s = _spatiald_from_spatialf(S_f)
    joint_S_s[qd_start] = S_s
    return X_WB, S_s * joint_qd_sap[qd_start], _sap_spatial(_vec3d_zero(), _vec3d_zero())


@wp.func
def _sap_revolute_identity_child_kinematics_f32_core(
    joint_axis: wp.array(dtype=wp.vec3d),
    X_WF: wp.transform,
    joint_q: wp.array(dtype=wp.float64),
    q_start: int,
    joint_qd_sap: wp.array(dtype=wp.float64),
    qd_start: int,
    joint_S_s: wp.array(dtype=wp.spatial_vectord),
):
    axis = _vec3f_from_vec3d(joint_axis[qd_start])
    q_FM = _quatf_from_axis_angle(axis, wp.float32(joint_q[q_start]))
    R_WF = wp.transform_get_rotation(X_WF)
    X_WB = wp.transform(wp.transform_get_translation(X_WF), _quatf_mul(R_WF, q_FM))
    S_f = _sap_spatialf(
        _quatf_rotate_unit(R_WF, axis),
        wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)),
    )
    joint_S_s[qd_start] = _spatiald_from_spatialf(S_f)
    return X_WB, S_f * wp.float32(joint_qd_sap[qd_start]), _sap_spatialf(
        wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)),
        wp.vec3(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0)),
    )


@wp.func
def _sap_w(V: wp.spatial_vectord) -> wp.vec3d:
    # SAP spatial coefficient order is [angular, translational].  Warp's
    # `spatial_top` is only storage here; semantically it is angular.
    return wp.spatial_top(V)


@wp.func
def _sap_v(V: wp.spatial_vectord) -> wp.vec3d:
    return wp.spatial_bottom(V)


@wp.func
def _sap_spatial(w: wp.vec3d, v: wp.vec3d) -> wp.spatial_vectord:
    return wp.spatial_vectord(w.x, w.y, w.z, v.x, v.y, v.z)


@wp.func
def _vec3f_from_vec3d(v: wp.vec3d) -> wp.vec3:
    return wp.vec3(wp.float32(v.x), wp.float32(v.y), wp.float32(v.z))


@wp.func
def _vec3d_from_vec3f(v: wp.vec3) -> wp.vec3d:
    return wp.vec3d(wp.float64(v.x), wp.float64(v.y), wp.float64(v.z))


@wp.func
def _quatf_from_quatd(q: wp.quatd) -> wp.quat:
    return wp.quat(wp.float32(q.x), wp.float32(q.y), wp.float32(q.z), wp.float32(q.w))


@wp.func
def _quatf_from_axis_angle(axis: wp.vec3, angle: wp.float32) -> wp.quat:
    half = wp.float32(0.5) * angle
    s = wp.sin(half)
    c = wp.cos(half)
    return wp.quat(axis[0] * s, axis[1] * s, axis[2] * s, c)


@wp.func
def _quatf_mul(a: wp.quat, b: wp.quat) -> wp.quat:
    return wp.quat(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    )


@wp.func
def _quatf_normalize(q: wp.quat) -> wp.quat:
    norm = wp.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm <= wp.float32(0.0):
        return wp.quat(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0), wp.float32(1.0))
    inv_norm = wp.float32(1.0) / norm
    return wp.quat(q.x * inv_norm, q.y * inv_norm, q.z * inv_norm, q.w * inv_norm)


@wp.func
def _quatf_rotate_unit(q: wp.quat, v: wp.vec3) -> wp.vec3:
    qv = wp.vec3(q.x, q.y, q.z)
    t = wp.float32(2.0) * wp.cross(qv, v)
    return v + q.w * t + wp.cross(qv, t)


@wp.func
def _quatd_from_quatf(q: wp.quat) -> wp.quatd:
    return wp.quatd(wp.float64(q.x), wp.float64(q.y), wp.float64(q.z), wp.float64(q.w))


@wp.func
def _transformf_from_transformd(x: wp.transformd) -> wp.transform:
    return wp.transform(
        _vec3f_from_vec3d(wp.transform_get_translation(x)),
        _quatf_from_quatd(wp.transform_get_rotation(x)),
    )


@wp.func
def _transformd_from_transformf(x: wp.transform) -> wp.transformd:
    return wp.transformd(
        _vec3d_from_vec3f(wp.transform_get_translation(x)),
        _quatd_from_quatf(wp.transform_get_rotation(x)),
    )


@wp.func
def _transformf_compose(a: wp.transform, b: wp.transform) -> wp.transform:
    ap = wp.transform_get_translation(a)
    aq = _quatf_normalize(wp.transform_get_rotation(a))
    bp = wp.transform_get_translation(b)
    bq = _quatf_normalize(wp.transform_get_rotation(b))
    return wp.transform(ap + _quatf_rotate_unit(aq, bp), _quatf_normalize(_quatf_mul(aq, bq)))


@wp.func
def _spatialf_from_spatiald(v: wp.spatial_vectord) -> wp.spatial_vector:
    return wp.spatial_vector(
        wp.float32(v[0]),
        wp.float32(v[1]),
        wp.float32(v[2]),
        wp.float32(v[3]),
        wp.float32(v[4]),
        wp.float32(v[5]),
    )


@wp.func
def _spatiald_from_spatialf(v: wp.spatial_vector) -> wp.spatial_vectord:
    return wp.spatial_vectord(
        wp.float64(v[0]),
        wp.float64(v[1]),
        wp.float64(v[2]),
        wp.float64(v[3]),
        wp.float64(v[4]),
        wp.float64(v[5]),
    )


@wp.func
def _sap_wf(V: wp.spatial_vector) -> wp.vec3:
    return wp.spatial_top(V)


@wp.func
def _sap_vf(V: wp.spatial_vector) -> wp.vec3:
    return wp.spatial_bottom(V)


@wp.func
def _sap_spatialf(w: wp.vec3, v: wp.vec3) -> wp.spatial_vector:
    return wp.spatial_vector(w.x, w.y, w.z, v.x, v.y, v.z)


@wp.func
def _sap_shift_velocity_f32(V_Bp: wp.spatial_vector, p_BpBq: wp.vec3) -> wp.spatial_vector:
    w = _sap_wf(V_Bp)
    v = _sap_vf(V_Bp)
    return _sap_spatialf(w, v + wp.cross(w, p_BpBq))


@wp.func
def _sap_shift_acceleration_f32(A_Bp: wp.spatial_vector, p_BpBq: wp.vec3, w_Bp: wp.vec3) -> wp.spatial_vector:
    alpha = _sap_wf(A_Bp)
    a = _sap_vf(A_Bp) + wp.cross(alpha, p_BpBq) + wp.cross(w_Bp, wp.cross(w_Bp, p_BpBq))
    return _sap_spatialf(alpha, a)


@wp.func
def _sap_compose_velocity_f32(
    V_WP: wp.spatial_vector,
    p_PB_W: wp.vec3,
    V_PB_W: wp.spatial_vector,
) -> wp.spatial_vector:
    w_WP = _sap_wf(V_WP)
    v_WPo = _sap_vf(V_WP)
    w_PB = _sap_wf(V_PB_W)
    v_PBo = _sap_vf(V_PB_W)
    return _sap_spatialf(w_WP + w_PB, v_WPo + wp.cross(w_WP, p_PB_W) + v_PBo)


@wp.func
def _sap_compose_acceleration_f32(
    A_WP: wp.spatial_vector,
    V_WP: wp.spatial_vector,
    p_PB_W: wp.vec3,
    V_PB_W: wp.spatial_vector,
    A_PB_W: wp.spatial_vector,
) -> wp.spatial_vector:
    w_WP = _sap_wf(V_WP)
    A_shifted = _sap_shift_acceleration_f32(A_WP, p_PB_W, w_WP)
    alpha = _sap_wf(A_shifted) + _sap_wf(A_PB_W) + wp.cross(w_WP, _sap_wf(V_PB_W))
    a = _sap_vf(A_shifted) + _sap_vf(A_PB_W) + wp.float32(2.0) * wp.cross(w_WP, _sap_vf(V_PB_W))
    return _sap_spatialf(alpha, a)


@wp.func
def _sap_shift_velocity(V_Bp: wp.spatial_vectord, p_BpBq: wp.vec3d) -> wp.spatial_vectord:
    w = _sap_w(V_Bp)
    v = _sap_v(V_Bp)
    return _sap_spatial(w, v + wp.cross(w, p_BpBq))


@wp.func
def _sap_shift_acceleration(A_Bp: wp.spatial_vectord, p_BpBq: wp.vec3d, w_Bp: wp.vec3d) -> wp.spatial_vectord:
    alpha = _sap_w(A_Bp)
    a = _sap_v(A_Bp) + wp.cross(alpha, p_BpBq) + wp.cross(w_Bp, wp.cross(w_Bp, p_BpBq))
    return _sap_spatial(alpha, a)


@wp.func
def _sap_shift_force(F_Bp: wp.spatial_vectord, p_BpBq: wp.vec3d) -> wp.spatial_vectord:
    tau = _sap_w(F_Bp)
    force = _sap_v(F_Bp)
    return _sap_spatial(tau - wp.cross(p_BpBq, force), force)


@wp.func
def _sap_compose_velocity(
    V_WP: wp.spatial_vectord,
    p_PB_W: wp.vec3d,
    V_PB_W: wp.spatial_vectord,
) -> wp.spatial_vectord:
    w_WP = _sap_w(V_WP)
    v_WPo = _sap_v(V_WP)
    w_PB = _sap_w(V_PB_W)
    v_PBo = _sap_v(V_PB_W)
    return _sap_spatial(w_WP + w_PB, v_WPo + wp.cross(w_WP, p_PB_W) + v_PBo)


@wp.func
def _sap_compose_acceleration(
    A_WP: wp.spatial_vectord,
    V_WP: wp.spatial_vectord,
    p_PB_W: wp.vec3d,
    V_PB_W: wp.spatial_vectord,
    A_PB_W: wp.spatial_vectord,
) -> wp.spatial_vectord:
    w_WP = _sap_w(V_WP)
    A_shifted = _sap_shift_acceleration(A_WP, p_PB_W, w_WP)
    alpha = _sap_w(A_shifted) + _sap_w(A_PB_W) + wp.cross(w_WP, _sap_w(V_PB_W))
    a = _sap_v(A_shifted) + _sap_v(A_PB_W) + wp.float64(2.0) * wp.cross(w_WP, _sap_v(V_PB_W))
    return _sap_spatial(alpha, a)


@wp.func
def _sap_transform_mobilizer_velocity_to_body(
    X_WF: wp.transformd,
    p_MoBo_F: wp.vec3d,
    V_FM_F: wp.spatial_vectord,
) -> wp.spatial_vectord:
    """Transform `V_FM_F` to SAP's `V_PB_W` at body origin `Bo`."""
    R_WF = wp.transform_get_rotation(X_WF)
    w_F = _sap_w(V_FM_F)
    v_Mo_F = _sap_v(V_FM_F)
    w_W = _quatd_rotate_unit(R_WF, w_F)
    v_Bo_W = _quatd_rotate_unit(R_WF, v_Mo_F + wp.cross(w_F, p_MoBo_F))
    return _sap_spatial(w_W, v_Bo_W)


@wp.func
def _sap_transform_mobilizer_accel_bias_to_body(
    X_WF: wp.transformd,
    p_MoBo_F: wp.vec3d,
    V_FM_F: wp.spatial_vectord,
    A_FM_F: wp.spatial_vectord,
) -> wp.spatial_vectord:
    R_WF = wp.transform_get_rotation(X_WF)
    shifted = _sap_shift_acceleration(A_FM_F, p_MoBo_F, _sap_w(V_FM_F))
    return _sap_spatial(
        _quatd_rotate_unit(R_WF, _sap_w(shifted)),
        _quatd_rotate_unit(R_WF, _sap_v(shifted)),
    )


@wp.func
def _sap_spatial_inertia_body_origin_world(
    body_q: wp.transformd,
    body_inertia_com: wp.mat33d,
    body_mass: wp.float64,
    body_com: wp.vec3d,
) -> wp.spatial_matrixd:
    R_WB_q = wp.transform_get_rotation(body_q)
    r0 = _quatd_rotate_unit(R_WB_q, wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0)))
    r1 = _quatd_rotate_unit(R_WB_q, wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0)))
    r2 = _quatd_rotate_unit(R_WB_q, wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0)))
    R_WB = wp.matrix_from_cols(r0, r1, r2)
    I_com_W = R_WB @ body_inertia_com @ wp.transpose(R_WB)

    c = _quatd_rotate_unit(R_WB_q, body_com)
    c_dot_c = wp.dot(c, c)
    c_outer = wp.outer(c, c)
    I3 = wp.mat33d(
        wp.float64(1.0), wp.float64(0.0), wp.float64(0.0),
        wp.float64(0.0), wp.float64(1.0), wp.float64(0.0),
        wp.float64(0.0), wp.float64(0.0), wp.float64(1.0),
    )
    I_Bo_W = I_com_W + body_mass * (c_dot_c * I3 - c_outer)

    cx = wp.skew(c)
    mcx = body_mass * cx
    # SAP order: [angular, translational].  Spatial inertia maps
    # [alpha, a_Bo] or [w, v_Bo] to [tau_Bo, f].
    return wp.spatial_matrixd(
        I_Bo_W[0, 0], I_Bo_W[0, 1], I_Bo_W[0, 2], mcx[0, 0], mcx[0, 1], mcx[0, 2],
        I_Bo_W[1, 0], I_Bo_W[1, 1], I_Bo_W[1, 2], mcx[1, 0], mcx[1, 1], mcx[1, 2],
        I_Bo_W[2, 0], I_Bo_W[2, 1], I_Bo_W[2, 2], mcx[2, 0], mcx[2, 1], mcx[2, 2],
        -mcx[0, 0], -mcx[0, 1], -mcx[0, 2], body_mass, 0.0, 0.0,
        -mcx[1, 0], -mcx[1, 1], -mcx[1, 2], 0.0, body_mass, 0.0,
        -mcx[2, 0], -mcx[2, 1], -mcx[2, 2], 0.0, 0.0, body_mass,
    )


@wp.func
def _sap_dynamic_bias_force_body_origin_world(
    body_q: wp.transformd,
    body_inertia_com: wp.mat33d,
    body_mass: wp.float64,
    body_com: wp.vec3d,
    V_WB: wp.spatial_vectord,
) -> wp.spatial_vectord:
    R_WB_q = wp.transform_get_rotation(body_q)
    r0 = _quatd_rotate_unit(R_WB_q, wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0)))
    r1 = _quatd_rotate_unit(R_WB_q, wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0)))
    r2 = _quatd_rotate_unit(R_WB_q, wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0)))
    R_WB = wp.matrix_from_cols(r0, r1, r2)
    I_com_W = R_WB @ body_inertia_com @ wp.transpose(R_WB)

    c = _quatd_rotate_unit(R_WB_q, body_com)
    c_dot_c = wp.dot(c, c)
    I3 = wp.mat33d(
        wp.float64(1.0), wp.float64(0.0), wp.float64(0.0),
        wp.float64(0.0), wp.float64(1.0), wp.float64(0.0),
        wp.float64(0.0), wp.float64(0.0), wp.float64(1.0),
    )
    I_Bo_W = I_com_W + body_mass * (c_dot_c * I3 - wp.outer(c, c))

    w = _sap_w(V_WB)
    return _sap_spatial(
        wp.cross(w, I_Bo_W @ w),
        body_mass * wp.cross(w, wp.cross(w, c)),
    )


@wp.func
def _sap_gravity_force_body_origin_world(
    body_q: wp.transformd,
    body_mass: wp.float64,
    body_com: wp.vec3d,
    gravity: wp.vec3d,
) -> wp.spatial_vectord:
    c = _quatd_rotate_unit(wp.transform_get_rotation(body_q), body_com)
    f = body_mass * gravity
    return _sap_spatial(wp.cross(c, f), f)


@wp.func
def _sap_compute_2d_rotational_dofs(
    axis_0: wp.vec3d,
    axis_1: wp.vec3d,
    q0: wp.float64,
    q1: wp.float64,
    qd0: wp.float64,
    qd1: wp.float64,
):
    q_off = _quatd_from_matrix_cols(axis_0, axis_1, wp.cross(axis_0, axis_1))
    local_0 = _quatd_rotate(q_off, wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0)))
    local_1 = _quatd_rotate(q_off, wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0)))

    axis_0 = local_0
    q_0 = _quatd_from_axis_angle(axis_0, q0)

    axis_1 = _quatd_rotate(q_0, local_1)
    q_1 = _quatd_from_axis_angle(axis_1, q1)

    rot = _quatd_mul(q_1, q_0)
    vel = axis_0 * qd0 + axis_1 * qd1
    return rot, vel


@wp.func
def _sap_compute_3d_rotational_dofs(
    axis_0: wp.vec3d,
    axis_1: wp.vec3d,
    axis_2: wp.vec3d,
    q0: wp.float64,
    q1: wp.float64,
    q2: wp.float64,
    qd0: wp.float64,
    qd1: wp.float64,
    qd2: wp.float64,
):
    q_off = _quatd_from_matrix_cols(axis_0, axis_1, axis_2)
    local_0 = _quatd_rotate(q_off, wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0)))
    local_1 = _quatd_rotate(q_off, wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0)))
    local_2 = _quatd_rotate(q_off, wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0)))

    axis_0 = local_0
    q_0 = _quatd_from_axis_angle(axis_0, q0)

    axis_1 = _quatd_rotate(q_0, local_1)
    q_1 = _quatd_from_axis_angle(axis_1, q1)

    q_10 = _quatd_mul(q_1, q_0)
    axis_2 = _quatd_rotate(q_10, local_2)
    q_2 = _quatd_from_axis_angle(axis_2, q2)

    rot = _quatd_mul(q_2, q_10)
    vel = axis_0 * qd0 + axis_1 * qd1 + axis_2 * qd2
    return rot, vel


@wp.func
def _sap_calc_across_mobilizer_transform(
    joint_type: int,
    joint_axis: wp.array(dtype=wp.vec3d),
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    joint_q: wp.array(dtype=wp.float64),
    q_start: int,
) -> wp.transformd:
    if joint_type == SAP_JOINT_PRISMATIC:
        q = wp.float64(joint_q[q_start])
        axis = joint_axis[axis_start]
        return wp.transformd(axis * q, _quatd_identity())

    if joint_type == SAP_JOINT_REVOLUTE:
        q = wp.float64(joint_q[q_start])
        axis = joint_axis[axis_start]
        return wp.transformd(_vec3d_zero(), _quatd_from_axis_angle(axis, q))

    if joint_type == SAP_JOINT_BALL:
        qx = wp.float64(joint_q[q_start + 0])
        qy = wp.float64(joint_q[q_start + 1])
        qz = wp.float64(joint_q[q_start + 2])
        qw = wp.float64(joint_q[q_start + 3])
        return wp.transformd(_vec3d_zero(), wp.quatd(qx, qy, qz, qw))

    if joint_type == SAP_JOINT_FIXED:
        return _transformd_identity()

    if joint_type == SAP_JOINT_FREE or joint_type == SAP_JOINT_DISTANCE:
        px = wp.float64(joint_q[q_start + 0])
        py = wp.float64(joint_q[q_start + 1])
        pz = wp.float64(joint_q[q_start + 2])
        qx = wp.float64(joint_q[q_start + 3])
        qy = wp.float64(joint_q[q_start + 4])
        qz = wp.float64(joint_q[q_start + 5])
        qw = wp.float64(joint_q[q_start + 6])
        return wp.transformd(wp.vec3d(px, py, pz), wp.quatd(qx, qy, qz, qw))

    if joint_type == SAP_JOINT_D6:
        pos = _vec3d_zero()
        rot = _quatd_identity()

        if lin_axis_count > 0:
            pos = pos + joint_axis[axis_start + 0] * wp.float64(joint_q[q_start + 0])
        if lin_axis_count > 1:
            pos = pos + joint_axis[axis_start + 1] * wp.float64(joint_q[q_start + 1])
        if lin_axis_count > 2:
            pos = pos + joint_axis[axis_start + 2] * wp.float64(joint_q[q_start + 2])

        ia = axis_start + lin_axis_count
        iq = q_start + lin_axis_count
        if ang_axis_count == 1:
            rot = _quatd_from_axis_angle(joint_axis[ia], wp.float64(joint_q[iq]))
        if ang_axis_count == 2:
            rot, _ = _sap_compute_2d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                wp.float64(joint_q[iq + 0]),
                wp.float64(joint_q[iq + 1]),
                wp.float64(0.0),
                wp.float64(0.0),
            )
        if ang_axis_count == 3:
            rot, _ = _sap_compute_3d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_axis[ia + 2],
                wp.float64(joint_q[iq + 0]),
                wp.float64(joint_q[iq + 1]),
                wp.float64(joint_q[iq + 2]),
                wp.float64(0.0),
                wp.float64(0.0),
                wp.float64(0.0),
            )

        return wp.transformd(pos, rot)

    return _transformd_identity()


@wp.func
def _d6_angular_axis(
    joint_axis: wp.array(dtype=wp.vec3d),
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    joint_q: wp.array(dtype=wp.float64),
    q_start: int,
    axis: int,
) -> wp.vec3d:
    ia = axis_start + lin_axis_count
    iq = q_start + lin_axis_count
    if ang_axis_count == 1:
        return joint_axis[ia]

    if ang_axis_count == 2:
        joint_axis0 = joint_axis[ia + 0]
        joint_axis1 = joint_axis[ia + 1]
        q_off = _quatd_from_matrix_cols(joint_axis0, joint_axis1, wp.cross(joint_axis0, joint_axis1))
        axis_0 = _quatd_rotate(q_off, wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0)))
        axis_1 = _quatd_rotate(q_off, wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0)))
        if axis == 0:
            return axis_0
        q_0 = _quatd_from_axis_angle(axis_0, wp.float64(joint_q[iq + 0]))
        return _quatd_rotate(q_0, axis_1)

    if ang_axis_count == 3:
        q_off = _quatd_from_matrix_cols(joint_axis[ia + 0], joint_axis[ia + 1], joint_axis[ia + 2])
        axis_0 = _quatd_rotate(q_off, wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0)))
        axis_1 = _quatd_rotate(q_off, wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0)))
        axis_2 = _quatd_rotate(q_off, wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0)))
        if axis == 0:
            return axis_0
        q_0 = _quatd_from_axis_angle(axis_0, wp.float64(joint_q[iq + 0]))
        axis_1_q = _quatd_rotate(q_0, axis_1)
        if axis == 1:
            return axis_1_q
        q_1 = _quatd_from_axis_angle(axis_1_q, wp.float64(joint_q[iq + 1]))
        return _quatd_rotate(_quatd_mul(q_1, q_0), axis_2)

    return _vec3d_zero()


@wp.func
def _sap_calc_mobilizer_velocity_and_bias_body_world(
    joint_type: int,
    joint_axis: wp.array(dtype=wp.vec3d),
    lin_axis_count: int,
    ang_axis_count: int,
    X_WF: wp.transformd,
    X_FM: wp.transformd,
    X_BM: wp.transformd,
    child_frame_identity: int,
    joint_q: wp.array(dtype=wp.float64),
    q_start: int,
    joint_qd_sap: wp.array(dtype=wp.float64),
    qd_start: int,
    joint_S_s: wp.array(dtype=wp.spatial_vectord),
):
    if joint_type == SAP_JOINT_REVOLUTE and child_frame_identity != 0:
        axis = joint_axis[qd_start]
        R_WF = wp.transform_get_rotation(X_WF)
        S_s = _sap_spatial(_quatd_rotate_unit(R_WF, axis), _vec3d_zero())
        joint_S_s[qd_start] = S_s
        return S_s * joint_qd_sap[qd_start], _sap_spatial(_vec3d_zero(), _vec3d_zero())

    X_MB = _transformd_inverse(X_BM)
    p_MoBo_M = wp.transform_get_translation(X_MB)
    R_FM = wp.transform_get_rotation(X_FM)
    p_MoBo_F = _quatd_rotate_unit(R_FM, p_MoBo_M)

    if joint_type == SAP_JOINT_PRISMATIC:
        axis = joint_axis[qd_start]
        S_s = _sap_transform_mobilizer_velocity_to_body(
            X_WF,
            p_MoBo_F,
            _sap_spatial(_vec3d_zero(), axis),
        )
        joint_S_s[qd_start] = S_s
        V_FM_F = _sap_spatial(_vec3d_zero(), axis * joint_qd_sap[qd_start])
        A_PB_W = _sap_transform_mobilizer_accel_bias_to_body(X_WF, p_MoBo_F, V_FM_F, _sap_spatial(_vec3d_zero(), _vec3d_zero()))
        return S_s * joint_qd_sap[qd_start], A_PB_W

    if joint_type == SAP_JOINT_REVOLUTE:
        axis = joint_axis[qd_start]
        R_WF = wp.transform_get_rotation(X_WF)
        S_s = _sap_spatial(
            _quatd_rotate_unit(R_WF, axis),
            _quatd_rotate_unit(R_WF, wp.cross(axis, p_MoBo_F)),
        )
        joint_S_s[qd_start] = S_s
        qd = joint_qd_sap[qd_start]
        w_FM_F = axis * qd
        A_PB_W = _sap_spatial(
            _vec3d_zero(),
            _quatd_rotate_unit(R_WF, wp.cross(w_FM_F, wp.cross(w_FM_F, p_MoBo_F))),
        )
        return S_s * qd, A_PB_W

    if joint_type == SAP_JOINT_BALL:
        S_0 = _sap_transform_mobilizer_velocity_to_body(X_WF, p_MoBo_F, _sap_spatial(wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0)), _vec3d_zero()))
        S_1 = _sap_transform_mobilizer_velocity_to_body(X_WF, p_MoBo_F, _sap_spatial(wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0)), _vec3d_zero()))
        S_2 = _sap_transform_mobilizer_velocity_to_body(X_WF, p_MoBo_F, _sap_spatial(wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0)), _vec3d_zero()))
        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2
        w_FM_F = wp.vec3d(
            joint_qd_sap[qd_start + 0],
            joint_qd_sap[qd_start + 1],
            joint_qd_sap[qd_start + 2],
        )
        V_FM_F = _sap_spatial(w_FM_F, _vec3d_zero())
        A_PB_W = _sap_transform_mobilizer_accel_bias_to_body(X_WF, p_MoBo_F, V_FM_F, _sap_spatial(_vec3d_zero(), _vec3d_zero()))
        return S_0 * joint_qd_sap[qd_start + 0] + S_1 * joint_qd_sap[qd_start + 1] + S_2 * joint_qd_sap[qd_start + 2], A_PB_W

    if joint_type == SAP_JOINT_FIXED:
        return _sap_spatial(_vec3d_zero(), _vec3d_zero()), _sap_spatial(_vec3d_zero(), _vec3d_zero())

    if joint_type == SAP_JOINT_FREE or joint_type == SAP_JOINT_DISTANCE:
        V_FM_F = _sap_spatial(
            wp.vec3d(
                joint_qd_sap[qd_start + 0],
                joint_qd_sap[qd_start + 1],
                joint_qd_sap[qd_start + 2],
            ),
            wp.vec3d(
                joint_qd_sap[qd_start + 3],
                joint_qd_sap[qd_start + 4],
                joint_qd_sap[qd_start + 5],
            ),
        )
        v_j_s = _sap_transform_mobilizer_velocity_to_body(X_WF, p_MoBo_F, V_FM_F)
        joint_S_s[qd_start + 0] = _sap_transform_mobilizer_velocity_to_body(X_WF, p_MoBo_F, _sap_spatial(wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0)), _vec3d_zero()))
        joint_S_s[qd_start + 1] = _sap_transform_mobilizer_velocity_to_body(X_WF, p_MoBo_F, _sap_spatial(wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0)), _vec3d_zero()))
        joint_S_s[qd_start + 2] = _sap_transform_mobilizer_velocity_to_body(X_WF, p_MoBo_F, _sap_spatial(wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0)), _vec3d_zero()))
        joint_S_s[qd_start + 3] = _sap_transform_mobilizer_velocity_to_body(X_WF, p_MoBo_F, _sap_spatial(_vec3d_zero(), wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))))
        joint_S_s[qd_start + 4] = _sap_transform_mobilizer_velocity_to_body(X_WF, p_MoBo_F, _sap_spatial(_vec3d_zero(), wp.vec3d(wp.float64(0.0), wp.float64(1.0), wp.float64(0.0))))
        joint_S_s[qd_start + 5] = _sap_transform_mobilizer_velocity_to_body(X_WF, p_MoBo_F, _sap_spatial(_vec3d_zero(), wp.vec3d(wp.float64(0.0), wp.float64(0.0), wp.float64(1.0))))
        A_PB_W = _sap_transform_mobilizer_accel_bias_to_body(X_WF, p_MoBo_F, V_FM_F, _sap_spatial(_vec3d_zero(), _vec3d_zero()))
        return v_j_s, A_PB_W

    if joint_type == SAP_JOINT_D6:
        v_j_s = _sap_spatial(_vec3d_zero(), _vec3d_zero())
        V_FM_F = _sap_spatial(_vec3d_zero(), _vec3d_zero())
        if lin_axis_count > 0:
            axis = joint_axis[qd_start + 0]
            S_s = _sap_transform_mobilizer_velocity_to_body(X_WF, p_MoBo_F, _sap_spatial(_vec3d_zero(), axis))
            v_j_s = v_j_s + S_s * joint_qd_sap[qd_start + 0]
            V_FM_F = V_FM_F + _sap_spatial(_vec3d_zero(), axis * joint_qd_sap[qd_start + 0])
            joint_S_s[qd_start + 0] = S_s
        if lin_axis_count > 1:
            axis = joint_axis[qd_start + 1]
            S_s = _sap_transform_mobilizer_velocity_to_body(X_WF, p_MoBo_F, _sap_spatial(_vec3d_zero(), axis))
            v_j_s = v_j_s + S_s * joint_qd_sap[qd_start + 1]
            V_FM_F = V_FM_F + _sap_spatial(_vec3d_zero(), axis * joint_qd_sap[qd_start + 1])
            joint_S_s[qd_start + 1] = S_s
        if lin_axis_count > 2:
            axis = joint_axis[qd_start + 2]
            S_s = _sap_transform_mobilizer_velocity_to_body(X_WF, p_MoBo_F, _sap_spatial(_vec3d_zero(), axis))
            v_j_s = v_j_s + S_s * joint_qd_sap[qd_start + 2]
            V_FM_F = V_FM_F + _sap_spatial(_vec3d_zero(), axis * joint_qd_sap[qd_start + 2])
            joint_S_s[qd_start + 2] = S_s
        if ang_axis_count > 0:
            axis = _d6_angular_axis(joint_axis, qd_start, lin_axis_count, ang_axis_count, joint_q, q_start, 0)
            dof = qd_start + lin_axis_count + 0
            S_s = _sap_transform_mobilizer_velocity_to_body(X_WF, p_MoBo_F, _sap_spatial(axis, _vec3d_zero()))
            v_j_s = v_j_s + S_s * joint_qd_sap[dof]
            V_FM_F = V_FM_F + _sap_spatial(axis * joint_qd_sap[dof], _vec3d_zero())
            joint_S_s[dof] = S_s
        if ang_axis_count > 1:
            axis = _d6_angular_axis(joint_axis, qd_start, lin_axis_count, ang_axis_count, joint_q, q_start, 1)
            dof = qd_start + lin_axis_count + 1
            S_s = _sap_transform_mobilizer_velocity_to_body(X_WF, p_MoBo_F, _sap_spatial(axis, _vec3d_zero()))
            v_j_s = v_j_s + S_s * joint_qd_sap[dof]
            V_FM_F = V_FM_F + _sap_spatial(axis * joint_qd_sap[dof], _vec3d_zero())
            joint_S_s[dof] = S_s
        if ang_axis_count > 2:
            axis = _d6_angular_axis(joint_axis, qd_start, lin_axis_count, ang_axis_count, joint_q, q_start, 2)
            dof = qd_start + lin_axis_count + 2
            S_s = _sap_transform_mobilizer_velocity_to_body(X_WF, p_MoBo_F, _sap_spatial(axis, _vec3d_zero()))
            v_j_s = v_j_s + S_s * joint_qd_sap[dof]
            V_FM_F = V_FM_F + _sap_spatial(axis * joint_qd_sap[dof], _vec3d_zero())
            joint_S_s[dof] = S_s
        A_PB_W = _sap_transform_mobilizer_accel_bias_to_body(X_WF, p_MoBo_F, V_FM_F, _sap_spatial(_vec3d_zero(), _vec3d_zero()))
        return v_j_s, A_PB_W

    wp.printf("SapFreeMotion motion subspace not implemented for joint type %d\n", joint_type)
    return _sap_spatial(_vec3d_zero(), _vec3d_zero()), _sap_spatial(_vec3d_zero(), _vec3d_zero())


@wp.func
def _free_distance_child_origin_world(
    joint: int,
    joint_child: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
) -> wp.vec3d:
    child = joint_child[joint]
    return _vec3d_from_vec3(wp.transform_get_translation(body_q[child]))


@wp.func
def _free_distance_child_com_offset_world(
    joint: int,
    joint_child: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
) -> wp.vec3d:
    child = joint_child[joint]
    x = _transformd_from_transform(body_q[child])
    return _quatd_rotate(wp.transform_get_rotation(x), _vec3d_from_vec3(body_com[child]))


@wp.func
def _free_distance_child_com_offset_world_d(
    joint: int,
    joint_child: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transformd),
    body_com: wp.array(dtype=wp.vec3d),
) -> wp.vec3d:
    child = joint_child[joint]
    return _quatd_rotate(wp.transform_get_rotation(body_q[child]), body_com[child])


@wp.kernel
def _copy_f32_to_f64(src: wp.array(dtype=wp.float32), dst: wp.array(dtype=wp.float64)):
    i = wp.tid()
    dst[i] = wp.float64(src[i])


@wp.kernel
def _copy_f64(src: wp.array(dtype=wp.float64), dst: wp.array(dtype=wp.float64)):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def copy_public_to_sap_velocity_f32(
    joint_type: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    public_v: wp.array(dtype=wp.float32),
    sap_v: wp.array(dtype=wp.float64),
):
    """Copy public-order velocities into SAP-order f32 buffers, shifting free-joint linear velocity to
    the body origin.
    """
    joint = wp.tid()
    dof_start = joint_qd_start[joint]
    axis_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
    jtype = joint_type[joint]

    if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
        r_com = _free_distance_child_com_offset_world(joint, joint_child, body_q, body_com)
        v_com = wp.vec3d(
            wp.float64(public_v[dof_start + 0]),
            wp.float64(public_v[dof_start + 1]),
            wp.float64(public_v[dof_start + 2]),
        )
        w = wp.vec3d(
            wp.float64(public_v[dof_start + 3]),
            wp.float64(public_v[dof_start + 4]),
            wp.float64(public_v[dof_start + 5]),
        )
        v_body_origin = v_com - wp.cross(w, r_com)
        sap_v[dof_start + 0] = w.x
        sap_v[dof_start + 1] = w.y
        sap_v[dof_start + 2] = w.z
        sap_v[dof_start + 3] = v_body_origin.x
        sap_v[dof_start + 4] = v_body_origin.y
        sap_v[dof_start + 5] = v_body_origin.z
        return

    for axis in range(axis_count):
        sap_v[dof_start + axis] = wp.float64(public_v[dof_start + axis])


@wp.kernel
def copy_public_to_sap_velocity_f64(
    joint_type: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_q: wp.array(dtype=wp.transformd),
    body_com: wp.array(dtype=wp.vec3d),
    public_v: wp.array(dtype=wp.float64),
    sap_v: wp.array(dtype=wp.float64),
):
    """Copy public-order velocities into SAP-order f64 buffers, shifting free-joint linear velocity to
    the body origin.
    """
    joint = wp.tid()
    dof_start = joint_qd_start[joint]
    axis_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
    jtype = joint_type[joint]

    if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
        r_com = _free_distance_child_com_offset_world_d(joint, joint_child, body_q, body_com)
        v_com = wp.vec3d(
            public_v[dof_start + 0],
            public_v[dof_start + 1],
            public_v[dof_start + 2],
        )
        w = wp.vec3d(
            public_v[dof_start + 3],
            public_v[dof_start + 4],
            public_v[dof_start + 5],
        )
        v_body_origin = v_com - wp.cross(w, r_com)
        sap_v[dof_start + 0] = w.x
        sap_v[dof_start + 1] = w.y
        sap_v[dof_start + 2] = w.z
        sap_v[dof_start + 3] = v_body_origin.x
        sap_v[dof_start + 4] = v_body_origin.y
        sap_v[dof_start + 5] = v_body_origin.z
        return

    for axis in range(axis_count):
        sap_v[dof_start + axis] = public_v[dof_start + axis]


@wp.kernel
def copy_public_to_sap_velocity_f64_from_joint_pose(
    joint_type: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3d),
    joint_q: wp.array(dtype=wp.float64),
    public_v: wp.array(dtype=wp.float64),
    sap_v: wp.array(dtype=wp.float64),
):
    """Copy public-order velocities into SAP-order f64 buffers using joint pose data to compute
    reference-point shifts.
    """
    joint = wp.tid()
    q_start = joint_q_start[joint]
    dof_start = joint_qd_start[joint]
    axis_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
    jtype = joint_type[joint]

    if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
        child = joint_child[joint]
        q = wp.quatd(
            joint_q[q_start + 3],
            joint_q[q_start + 4],
            joint_q[q_start + 5],
            joint_q[q_start + 6],
        )
        r_com = _quatd_rotate(q, body_com[child])
        v_com = wp.vec3d(
            public_v[dof_start + 0],
            public_v[dof_start + 1],
            public_v[dof_start + 2],
        )
        w = wp.vec3d(
            public_v[dof_start + 3],
            public_v[dof_start + 4],
            public_v[dof_start + 5],
        )
        v_body_origin = v_com - wp.cross(w, r_com)
        sap_v[dof_start + 0] = w.x
        sap_v[dof_start + 1] = w.y
        sap_v[dof_start + 2] = w.z
        sap_v[dof_start + 3] = v_body_origin.x
        sap_v[dof_start + 4] = v_body_origin.y
        sap_v[dof_start + 5] = v_body_origin.z
        return

    for axis in range(axis_count):
        sap_v[dof_start + axis] = public_v[dof_start + axis]


@wp.kernel
def copy_sap_to_public_velocity_f32(
    joint_type: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    sap_v: wp.array(dtype=wp.float64),
    public_v: wp.array(dtype=wp.float32),
):
    """Copy SAP-order velocities back to public-order f32 buffers, shifting free-joint linear velocity
    to the center of mass.
    """
    joint = wp.tid()
    dof_start = joint_qd_start[joint]
    axis_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
    jtype = joint_type[joint]

    if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
        r_com = _free_distance_child_com_offset_world(joint, joint_child, body_q, body_com)
        w = wp.vec3d(
            sap_v[dof_start + 0],
            sap_v[dof_start + 1],
            sap_v[dof_start + 2],
        )
        v_body_origin = wp.vec3d(
            sap_v[dof_start + 3],
            sap_v[dof_start + 4],
            sap_v[dof_start + 5],
        )
        v_com = v_body_origin + wp.cross(w, r_com)
        public_v[dof_start + 0] = wp.float32(v_com.x)
        public_v[dof_start + 1] = wp.float32(v_com.y)
        public_v[dof_start + 2] = wp.float32(v_com.z)
        public_v[dof_start + 3] = wp.float32(w.x)
        public_v[dof_start + 4] = wp.float32(w.y)
        public_v[dof_start + 5] = wp.float32(w.z)
        return

    for axis in range(axis_count):
        public_v[dof_start + axis] = wp.float32(sap_v[dof_start + axis])


@wp.kernel
def copy_sap_to_public_velocity_f64(
    joint_type: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_q: wp.array(dtype=wp.transformd),
    body_com: wp.array(dtype=wp.vec3d),
    sap_v: wp.array(dtype=wp.float64),
    public_v: wp.array(dtype=wp.float64),
):
    """Copy SAP-order velocities back to public-order f64 buffers, shifting free-joint linear velocity
    to the center of mass.
    """
    joint = wp.tid()
    dof_start = joint_qd_start[joint]
    axis_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
    jtype = joint_type[joint]

    if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
        r_com = _free_distance_child_com_offset_world_d(joint, joint_child, body_q, body_com)
        w = wp.vec3d(
            sap_v[dof_start + 0],
            sap_v[dof_start + 1],
            sap_v[dof_start + 2],
        )
        v_body_origin = wp.vec3d(
            sap_v[dof_start + 3],
            sap_v[dof_start + 4],
            sap_v[dof_start + 5],
        )
        v_com = v_body_origin + wp.cross(w, r_com)
        public_v[dof_start + 0] = v_com.x
        public_v[dof_start + 1] = v_com.y
        public_v[dof_start + 2] = v_com.z
        public_v[dof_start + 3] = w.x
        public_v[dof_start + 4] = w.y
        public_v[dof_start + 5] = w.z
        return

    for axis in range(axis_count):
        public_v[dof_start + axis] = sap_v[dof_start + axis]


@wp.kernel
def _copy_sap_to_public_velocity_f64_from_joint_pose(
    joint_type: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3d),
    joint_q: wp.array(dtype=wp.float64),
    sap_v: wp.array(dtype=wp.float64),
    public_v: wp.array(dtype=wp.float64),
):
    joint = wp.tid()
    q_start = joint_q_start[joint]
    dof_start = joint_qd_start[joint]
    axis_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
    jtype = joint_type[joint]

    if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
        child = joint_child[joint]
        q = wp.quatd(
            joint_q[q_start + 3],
            joint_q[q_start + 4],
            joint_q[q_start + 5],
            joint_q[q_start + 6],
        )
        r_com = _quat_rotate_vec3d(q, body_com[child])
        w = wp.vec3d(
            sap_v[dof_start + 0],
            sap_v[dof_start + 1],
            sap_v[dof_start + 2],
        )
        v_body_origin = wp.vec3d(
            sap_v[dof_start + 3],
            sap_v[dof_start + 4],
            sap_v[dof_start + 5],
        )
        v_com = v_body_origin + wp.cross(w, r_com)
        public_v[dof_start + 0] = v_com.x
        public_v[dof_start + 1] = v_com.y
        public_v[dof_start + 2] = v_com.z
        public_v[dof_start + 3] = w.x
        public_v[dof_start + 4] = w.y
        public_v[dof_start + 5] = w.z
        return

    for axis in range(axis_count):
        public_v[dof_start + axis] = sap_v[dof_start + axis]


@wp.kernel
def _copy_sap_to_public_velocity_f32_from_joint_pose(
    joint_type: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3d),
    joint_q: wp.array(dtype=wp.float32),
    sap_v: wp.array(dtype=wp.float64),
    public_v: wp.array(dtype=wp.float32),
):
    joint = wp.tid()
    q_start = joint_q_start[joint]
    dof_start = joint_qd_start[joint]
    axis_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
    jtype = joint_type[joint]

    if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
        child = joint_child[joint]
        q = wp.quatd(
            wp.float64(joint_q[q_start + 3]),
            wp.float64(joint_q[q_start + 4]),
            wp.float64(joint_q[q_start + 5]),
            wp.float64(joint_q[q_start + 6]),
        )
        r_com = _quat_rotate_vec3d(q, body_com[child])
        w = wp.vec3d(
            sap_v[dof_start + 0],
            sap_v[dof_start + 1],
            sap_v[dof_start + 2],
        )
        v_body_origin = wp.vec3d(
            sap_v[dof_start + 3],
            sap_v[dof_start + 4],
            sap_v[dof_start + 5],
        )
        v_com = v_body_origin + wp.cross(w, r_com)
        public_v[dof_start + 0] = wp.float32(v_com.x)
        public_v[dof_start + 1] = wp.float32(v_com.y)
        public_v[dof_start + 2] = wp.float32(v_com.z)
        public_v[dof_start + 3] = wp.float32(w.x)
        public_v[dof_start + 4] = wp.float32(w.y)
        public_v[dof_start + 5] = wp.float32(w.z)
        return

    for axis in range(axis_count):
        public_v[dof_start + axis] = wp.float32(sap_v[dof_start + axis])


@wp.kernel
def _copy_public_to_sap_force_kernel(
    joint_type: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    public_f: wp.array(dtype=wp.float32),
    sap_f: wp.array(dtype=wp.float64),
):
    joint = wp.tid()
    dof_start = joint_qd_start[joint]
    axis_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
    jtype = joint_type[joint]

    if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
        r_com = _free_distance_child_com_offset_world(joint, joint_child, body_q, body_com)
        f_com = wp.vec3d(
            wp.float64(public_f[dof_start + 0]),
            wp.float64(public_f[dof_start + 1]),
            wp.float64(public_f[dof_start + 2]),
        )
        tau_com = wp.vec3d(
            wp.float64(public_f[dof_start + 3]),
            wp.float64(public_f[dof_start + 4]),
            wp.float64(public_f[dof_start + 5]),
        )
        tau_body_origin = tau_com + wp.cross(r_com, f_com)
        sap_f[dof_start + 0] = tau_body_origin.x
        sap_f[dof_start + 1] = tau_body_origin.y
        sap_f[dof_start + 2] = tau_body_origin.z
        sap_f[dof_start + 3] = f_com.x
        sap_f[dof_start + 4] = f_com.y
        sap_f[dof_start + 5] = f_com.z
        return

    for axis in range(axis_count):
        sap_f[dof_start + axis] = wp.float64(public_f[dof_start + axis])


@wp.kernel
def _copy_public_boundary_to_sap_f32_kernel(
    joint_count: int,
    body_count: int,
    joint_type: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    public_v: wp.array(dtype=wp.float32),
    public_f: wp.array(dtype=wp.float32),
    public_body_f: wp.array(dtype=wp.spatial_vector),
    sap_v: wp.array(dtype=wp.float64),
    sap_f: wp.array(dtype=wp.float64),
    sap_body_f: wp.array(dtype=wp.spatial_vectord),
):
    tid = wp.tid()
    if tid < joint_count:
        dof_start = joint_qd_start[tid]
        axis_count = joint_dof_dim[tid, 0] + joint_dof_dim[tid, 1]
        jtype = joint_type[tid]

        if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
            r_com = _free_distance_child_com_offset_world(tid, joint_child, body_q, body_com)
            v_com = wp.vec3d(
                wp.float64(public_v[dof_start + 0]),
                wp.float64(public_v[dof_start + 1]),
                wp.float64(public_v[dof_start + 2]),
            )
            w = wp.vec3d(
                wp.float64(public_v[dof_start + 3]),
                wp.float64(public_v[dof_start + 4]),
                wp.float64(public_v[dof_start + 5]),
            )
            v_body_origin = v_com - wp.cross(w, r_com)
            sap_v[dof_start + 0] = w.x
            sap_v[dof_start + 1] = w.y
            sap_v[dof_start + 2] = w.z
            sap_v[dof_start + 3] = v_body_origin.x
            sap_v[dof_start + 4] = v_body_origin.y
            sap_v[dof_start + 5] = v_body_origin.z

            f_com = wp.vec3d(
                wp.float64(public_f[dof_start + 0]),
                wp.float64(public_f[dof_start + 1]),
                wp.float64(public_f[dof_start + 2]),
            )
            tau_com = wp.vec3d(
                wp.float64(public_f[dof_start + 3]),
                wp.float64(public_f[dof_start + 4]),
                wp.float64(public_f[dof_start + 5]),
            )
            tau_body_origin = tau_com + wp.cross(r_com, f_com)
            sap_f[dof_start + 0] = tau_body_origin.x
            sap_f[dof_start + 1] = tau_body_origin.y
            sap_f[dof_start + 2] = tau_body_origin.z
            sap_f[dof_start + 3] = f_com.x
            sap_f[dof_start + 4] = f_com.y
            sap_f[dof_start + 5] = f_com.z
        else:
            for axis in range(axis_count):
                sap_v[dof_start + axis] = wp.float64(public_v[dof_start + axis])
                sap_f[dof_start + axis] = wp.float64(public_f[dof_start + axis])

    if tid < body_count:
        f_com = _vec3d_from_vec3(wp.spatial_top(public_body_f[tid]))
        tau_com = _vec3d_from_vec3(wp.spatial_bottom(public_body_f[tid]))
        x = _transformd_from_transform(body_q[tid])
        c = _quatd_rotate(wp.transform_get_rotation(x), _vec3d_from_vec3(body_com[tid]))
        tau_body_origin = tau_com + wp.cross(c, f_com)
        sap_body_f[tid] = _sap_spatial(-tau_body_origin, -f_com)


@wp.kernel
def _copy_public_to_sap_force_kernel_f32_body_qd(
    joint_type: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_q: wp.array(dtype=wp.transformd),
    body_com: wp.array(dtype=wp.vec3d),
    public_f: wp.array(dtype=wp.float32),
    sap_f: wp.array(dtype=wp.float64),
):
    joint = wp.tid()
    dof_start = joint_qd_start[joint]
    axis_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
    jtype = joint_type[joint]

    if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
        r_com = _free_distance_child_com_offset_world_d(joint, joint_child, body_q, body_com)
        f_com = wp.vec3d(
            wp.float64(public_f[dof_start + 0]),
            wp.float64(public_f[dof_start + 1]),
            wp.float64(public_f[dof_start + 2]),
        )
        tau_com = wp.vec3d(
            wp.float64(public_f[dof_start + 3]),
            wp.float64(public_f[dof_start + 4]),
            wp.float64(public_f[dof_start + 5]),
        )
        tau_body_origin = tau_com + wp.cross(r_com, f_com)
        sap_f[dof_start + 0] = tau_body_origin.x
        sap_f[dof_start + 1] = tau_body_origin.y
        sap_f[dof_start + 2] = tau_body_origin.z
        sap_f[dof_start + 3] = f_com.x
        sap_f[dof_start + 4] = f_com.y
        sap_f[dof_start + 5] = f_com.z
        return

    for axis in range(axis_count):
        sap_f[dof_start + axis] = wp.float64(public_f[dof_start + axis])


@wp.kernel
def _copy_public_to_sap_force_kernel_f64(
    joint_type: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_q: wp.array(dtype=wp.transformd),
    body_com: wp.array(dtype=wp.vec3d),
    public_f: wp.array(dtype=wp.float64),
    sap_f: wp.array(dtype=wp.float64),
):
    joint = wp.tid()
    dof_start = joint_qd_start[joint]
    axis_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
    jtype = joint_type[joint]

    if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
        r_com = _free_distance_child_com_offset_world_d(joint, joint_child, body_q, body_com)
        f_com = wp.vec3d(
            public_f[dof_start + 0],
            public_f[dof_start + 1],
            public_f[dof_start + 2],
        )
        tau_com = wp.vec3d(
            public_f[dof_start + 3],
            public_f[dof_start + 4],
            public_f[dof_start + 5],
        )
        tau_body_origin = tau_com + wp.cross(r_com, f_com)
        sap_f[dof_start + 0] = tau_body_origin.x
        sap_f[dof_start + 1] = tau_body_origin.y
        sap_f[dof_start + 2] = tau_body_origin.z
        sap_f[dof_start + 3] = f_com.x
        sap_f[dof_start + 4] = f_com.y
        sap_f[dof_start + 5] = f_com.z
        return

    for axis in range(axis_count):
        sap_f[dof_start + axis] = public_f[dof_start + axis]


@wp.kernel
def _copy_public_to_sap_force_kernel_f64_from_joint_pose(
    joint_type: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3d),
    joint_q: wp.array(dtype=wp.float64),
    public_f: wp.array(dtype=wp.float64),
    sap_f: wp.array(dtype=wp.float64),
):
    joint = wp.tid()
    q_start = joint_q_start[joint]
    dof_start = joint_qd_start[joint]
    axis_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
    jtype = joint_type[joint]

    if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
        child = joint_child[joint]
        q = wp.quatd(
            joint_q[q_start + 3],
            joint_q[q_start + 4],
            joint_q[q_start + 5],
            joint_q[q_start + 6],
        )
        r_com = _quatd_rotate(q, body_com[child])
        f_com = wp.vec3d(
            public_f[dof_start + 0],
            public_f[dof_start + 1],
            public_f[dof_start + 2],
        )
        tau_com = wp.vec3d(
            public_f[dof_start + 3],
            public_f[dof_start + 4],
            public_f[dof_start + 5],
        )
        tau_body_origin = tau_com + wp.cross(r_com, f_com)
        sap_f[dof_start + 0] = tau_body_origin.x
        sap_f[dof_start + 1] = tau_body_origin.y
        sap_f[dof_start + 2] = tau_body_origin.z
        sap_f[dof_start + 3] = f_com.x
        sap_f[dof_start + 4] = f_com.y
        sap_f[dof_start + 5] = f_com.z
        return

    for axis in range(axis_count):
        sap_f[dof_start + axis] = public_f[dof_start + axis]


@wp.kernel
def _copy_public_body_force_to_sap_body_origin_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    public_body_f: wp.array(dtype=wp.spatial_vector),
    sap_body_f: wp.array(dtype=wp.spatial_vectord),
):
    body = wp.tid()
    f_com = _vec3d_from_vec3(wp.spatial_top(public_body_f[body]))
    tau_com = _vec3d_from_vec3(wp.spatial_bottom(public_body_f[body]))
    x = _transformd_from_transform(body_q[body])
    c = _quatd_rotate(wp.transform_get_rotation(x), _vec3d_from_vec3(body_com[body]))
    tau_body_origin = tau_com + wp.cross(c, f_com)
    # The inverse-dynamics residual stores -Fapp so that projection produces
    # +J^T Fapp in generalized free motion.
    sap_body_f[body] = _sap_spatial(-tau_body_origin, -f_com)


@wp.kernel
def _copy_public_body_force_to_sap_body_origin_kernel_body_qd(
    body_q: wp.array(dtype=wp.transformd),
    body_com: wp.array(dtype=wp.vec3d),
    public_body_f: wp.array(dtype=wp.spatial_vector),
    sap_body_f: wp.array(dtype=wp.spatial_vectord),
):
    body = wp.tid()
    f_com = _vec3d_from_vec3(wp.spatial_top(public_body_f[body]))
    tau_com = _vec3d_from_vec3(wp.spatial_bottom(public_body_f[body]))
    c = _quatd_rotate(wp.transform_get_rotation(body_q[body]), body_com[body])
    tau_body_origin = tau_com + wp.cross(c, f_com)
    sap_body_f[body] = _sap_spatial(-tau_body_origin, -f_com)


@wp.func
def _project_tau_no_drives(
    joint_type: int,
    joint_S_s: wp.array(dtype=wp.spatial_vectord),
    joint_f: wp.array(dtype=wp.float64),
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    body_f_s: wp.spatial_vectord,
    tau: wp.array(dtype=wp.float64),
):
    if joint_type == SAP_JOINT_BALL:
        for i in range(3):
            S_s = joint_S_s[dof_start + i]
            tau[dof_start + i] = -wp.dot(S_s, body_f_s) + joint_f[dof_start + i]
        return

    if joint_type == SAP_JOINT_FREE or joint_type == SAP_JOINT_DISTANCE:
        for i in range(6):
            S_s = joint_S_s[dof_start + i]
            tau[dof_start + i] = -wp.dot(S_s, body_f_s) + joint_f[dof_start + i]
        return

    if joint_type == SAP_JOINT_PRISMATIC or joint_type == SAP_JOINT_REVOLUTE or joint_type == SAP_JOINT_D6:
        axis_count = lin_axis_count + ang_axis_count
        for i in range(axis_count):
            j = dof_start + i
            S_s = joint_S_s[j]
            tau[j] = -wp.dot(S_s, body_f_s) + joint_f[j]


@wp.func
def _dense_index_f64(n: int, i: int, j: int) -> int:
    return i * n + j


@wp.func
def _dense_cholesky_f64(
    n: int,
    A: wp.array(dtype=wp.float64),
    R: wp.array(dtype=wp.float64),
    A_start: int,
    R_start: int,
    L: wp.array(dtype=wp.float64),
):
    for j in range(n):
        s = A[A_start + _dense_index_f64(n, j, j)] + R[R_start + j]

        for k in range(j):
            r = L[A_start + _dense_index_f64(n, j, k)]
            s = s - r * r

        s = wp.sqrt(s)
        inv_s = wp.float64(1.0) / s
        L[A_start + _dense_index_f64(n, j, j)] = s

        for i in range(j + 1, n):
            t = A[A_start + _dense_index_f64(n, i, j)]

            for k in range(j):
                t = t - L[A_start + _dense_index_f64(n, i, k)] * L[A_start + _dense_index_f64(n, j, k)]

            L[A_start + _dense_index_f64(n, i, j)] = t * inv_s


@wp.func
def _dense_subs_f64(
    n: int,
    L_start: int,
    b_start: int,
    L: wp.array(dtype=wp.float64),
    b: wp.array(dtype=wp.float64),
    x: wp.array(dtype=wp.float64),
):
    for i in range(n):
        s = b[b_start + i]

        for j in range(i):
            s = s - L[L_start + _dense_index_f64(n, i, j)] * x[b_start + j]

        x[b_start + i] = s / L[L_start + _dense_index_f64(n, i, i)]

    for i in range(n - 1, -1, -1):
        s = x[b_start + i]

        for j in range(i + 1, n):
            s = s - L[L_start + _dense_index_f64(n, j, i)] * x[b_start + j]

        x[b_start + i] = s / L[L_start + _dense_index_f64(n, i, i)]


@wp.func
def _v3d(x: wp.float64, y: wp.float64, z: wp.float64) -> wp.vec3d:
    return wp.vec3d(x, y, z)


@wp.func
def _m33d(
    a00: wp.float64,
    a01: wp.float64,
    a02: wp.float64,
    a10: wp.float64,
    a11: wp.float64,
    a12: wp.float64,
    a20: wp.float64,
    a21: wp.float64,
    a22: wp.float64,
) -> wp.mat33d:
    return wp.mat33d(a00, a01, a02, a10, a11, a12, a20, a21, a22)


@wp.func
def _zero_m33d() -> wp.mat33d:
    return _m33d(
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
        wp.float64(0.0),
    )


@wp.func
def _v3f(x: wp.float32, y: wp.float32, z: wp.float32) -> wp.vec3:
    return wp.vec3(x, y, z)


@wp.func
def _m33f(
    a00: wp.float32,
    a01: wp.float32,
    a02: wp.float32,
    a10: wp.float32,
    a11: wp.float32,
    a12: wp.float32,
    a20: wp.float32,
    a21: wp.float32,
    a22: wp.float32,
) -> wp.mat33:
    return wp.mat33(a00, a01, a02, a10, a11, a12, a20, a21, a22)


@wp.func
def _zero_m33f() -> wp.mat33:
    return _m33f(
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
        wp.float32(0.0),
    )


@wp.func
def _clamp_scalar_f64(x: wp.float64, limit: wp.float64) -> wp.float64:
    if limit <= wp.float64(0.0) or not wp.isfinite(limit):
        return x
    return wp.max(-limit, wp.min(limit, x))


@wp.func
def _clamp_derivative_f64(x: wp.float64, limit: wp.float64) -> wp.float64:
    if limit <= wp.float64(0.0) or not wp.isfinite(limit):
        return wp.float64(1.0)
    if -limit <= x and x <= limit:
        return wp.float64(1.0)
    return wp.float64(0.0)


@wp.func
def _clamp_antiderivative_f64(x: wp.float64, limit: wp.float64) -> wp.float64:
    if limit <= wp.float64(0.0) or not wp.isfinite(limit):
        return wp.float64(0.5) * x * x
    if x < -limit:
        return -limit * (x + wp.float64(0.5) * limit)
    if x <= limit:
        return wp.float64(0.5) * x * x
    return limit * (x - wp.float64(0.5) * limit)


@wp.func
def _clamp_scalar_f32(x: wp.float32, limit: wp.float32) -> wp.float32:
    if limit <= wp.float32(0.0) or not wp.isfinite(limit):
        return x
    return wp.max(-limit, wp.min(limit, x))


@wp.func
def _clamp_derivative_f32(x: wp.float32, limit: wp.float32) -> wp.float32:
    if limit <= wp.float32(0.0) or not wp.isfinite(limit):
        return wp.float32(1.0)
    if -limit <= x and x <= limit:
        return wp.float32(1.0)
    return wp.float32(0.0)


@wp.func
def _clamp_antiderivative_f32(x: wp.float32, limit: wp.float32) -> wp.float32:
    if limit <= wp.float32(0.0) or not wp.isfinite(limit):
        return wp.float32(0.5) * x * x
    if x < -limit:
        return -limit * (x + wp.float32(0.5) * limit)
    if x <= limit:
        return wp.float32(0.5) * x * x
    return limit * (x - wp.float32(0.5) * limit)


@wp.func
def _compute_effective_pd_gains_sap(
    kp: wp.float64,
    kd: wp.float64,
    dt: wp.float64,
    delassus: wp.float64,
) -> wp.vec2d:
    if kp <= wp.float64(0.0) and kd <= wp.float64(0.0):
        return wp.vec2d(wp.float64(0.0), wp.float64(0.0))

    beta = wp.float64(_SAP_PD_BETA)
    beta_factor = beta * beta / (wp.float64(4.0) * wp.float64(_PI) * wp.float64(_PI))
    r_nr = beta_factor * wp.max(delassus, wp.float64(1.0e-12))
    common = dt * kp + kd
    if common <= wp.float64(0.0):
        return wp.vec2d(wp.float64(0.0), wp.float64(0.0))

    r = wp.float64(1.0) / (dt * common)
    if r >= r_nr:
        return wp.vec2d(kp, kd)

    kp_eff = kp
    kd_eff = kd
    if dt * kp > kd and kp > wp.float64(0.0):
        tau = kd / kp
        kp_eff = wp.float64(1.0) / (dt * (dt + tau) * r_nr)
        kd_eff = tau * kp_eff
    elif kd > wp.float64(0.0):
        tau_inv = kp / kd
        kd_eff = wp.float64(1.0) / (dt * (wp.float64(1.0) + dt * tau_inv) * r_nr)
        kp_eff = tau_inv * kd_eff
    else:
        kp_eff = wp.float64(1.0) / (dt * dt * r_nr)
        kd_eff = wp.float64(0.0)

    return wp.vec2d(kp_eff, kd_eff)


@wp.func
def _compute_effective_pd_gains_sap_f32(
    kp: wp.float32,
    kd: wp.float32,
    dt: wp.float32,
    delassus: wp.float32,
) -> wp.vec2:
    if kp <= wp.float32(0.0) and kd <= wp.float32(0.0):
        return wp.vec2(wp.float32(0.0), wp.float32(0.0))

    beta = wp.float32(_SAP_PD_BETA)
    beta_factor = beta * beta / (wp.float32(4.0) * wp.float32(_PI) * wp.float32(_PI))
    r_nr = beta_factor * wp.max(delassus, wp.float32(1.0e-12))
    common = dt * kp + kd
    if common <= wp.float32(0.0):
        return wp.vec2(wp.float32(0.0), wp.float32(0.0))

    r = wp.float32(1.0) / (dt * common)
    if r >= r_nr:
        return wp.vec2(kp, kd)

    kp_eff = kp
    kd_eff = kd
    if dt * kp > kd and kp > wp.float32(0.0):
        tau = kd / kp
        kp_eff = wp.float32(1.0) / (dt * (dt + tau) * r_nr)
        kd_eff = tau * kp_eff
    elif kd > wp.float32(0.0):
        tau_inv = kp / kd
        kd_eff = wp.float32(1.0) / (dt * (wp.float32(1.0) + dt * tau_inv) * r_nr)
        kp_eff = tau_inv * kd_eff
    else:
        kp_eff = wp.float32(1.0) / (dt * dt * r_nr)
        kd_eff = wp.float32(0.0)

    return wp.vec2(kp_eff, kd_eff)


@wp.func
def _contact_projection_cost_from_vc_sap(
    env: int,
    c: int,
    vc: wp.vec3d,
    contact_phi0: wp.array(dtype=wp.float64, ndim=2),
    contact_w_eff: wp.array(dtype=wp.float64, ndim=2),
    contact_mu: wp.array(dtype=wp.float64, ndim=2),
    contact_k: wp.array(dtype=wp.float64, ndim=2),
    contact_tau_d: wp.array(dtype=wp.float64, ndim=2),
    beta: wp.float64,
    sigma: wp.float64,
    dt: wp.float64,
) -> wp.float64:
    wi = contact_w_eff[env, c]
    if wi < wp.float64(1.0e-12) or not wp.isfinite(wi):
        wi = wp.float64(1.0e-12)

    beta_factor = beta * beta / (wp.float64(4.0) * wp.float64(_PI) * wp.float64(_PI))
    rn_hard = beta_factor * wi
    k_c = contact_k[env, c]
    if k_c <= wp.float64(0.0) or not wp.isfinite(k_c):
        k_c = wp.float64(1.0)
    tau_c = wp.max(contact_tau_d[env, c], wp.float64(0.0))
    rn_soft = wp.float64(1.0) / (dt * k_c * (dt + tau_c))
    rn = wp.max(rn_hard, rn_soft)
    rt = sigma * wi
    if rt < wp.float64(1.0e-30):
        rt = wp.float64(1.0e-30)
    if rn < wp.float64(1.0e-30):
        rn = wp.float64(1.0e-30)

    rt_inv = wp.float64(1.0) / rt
    rn_inv = wp.float64(1.0) / rn
    vhat_n = -contact_phi0[env, c] / (dt + tau_c)
    y = wp.vec3d(-rt_inv * vc.x, -rt_inv * vc.y, rn_inv * (vhat_n - vc.z))

    mu = contact_mu[env, c]
    if mu < wp.float64(0.0) or not wp.isfinite(mu):
        mu = wp.float64(0.0)

    yr = wp.sqrt(
        y.x * y.x
        + y.y * y.y
        + wp.float64(_CONTACT_SOFT_NORM_TOL) * wp.float64(_CONTACT_SOFT_NORM_TOL)
    )
    t_hat = _v3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    if yr > wp.float64(0.0):
        t_hat = _v3d(y.x / yr, y.y / yr, wp.float64(0.0))

    gamma = _v3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    if mu <= wp.float64(1.0e-12):
        if y.z > wp.float64(0.0):
            gamma = _v3d(wp.float64(0.0), wp.float64(0.0), y.z)
    else:
        mu_tilde = mu * wp.sqrt(rt / rn)
        mu_hat = mu * rt / rn
        factor = wp.float64(1.0) / (wp.float64(1.0) + mu_tilde * mu_tilde)

        if yr <= mu * y.z:
            gamma = y
        elif (-mu_hat * yr < y.z) and (y.z < yr / mu):
            gamma_n = (y.z + mu_hat * yr) * factor
            gamma = _v3d(mu * gamma_n * t_hat.x, mu * gamma_n * t_hat.y, gamma_n)

    return wp.float64(0.5) * (
        rt * (gamma.x * gamma.x + gamma.y * gamma.y) + rn * gamma.z * gamma.z
    )


@wp.func
def _contact_projection_cost_from_velocity_sap(
    env: int,
    c: int,
    dof_per_env: int,
    contact_jac: wp.array(dtype=wp.float64, ndim=4),
    contact_phi0: wp.array(dtype=wp.float64, ndim=2),
    contact_w_eff: wp.array(dtype=wp.float64, ndim=2),
    contact_mu: wp.array(dtype=wp.float64, ndim=2),
    contact_k: wp.array(dtype=wp.float64, ndim=2),
    contact_tau_d: wp.array(dtype=wp.float64, ndim=2),
    v: wp.array(dtype=wp.float64, ndim=2),
    dv: wp.array(dtype=wp.float64, ndim=2),
    alpha: wp.float64,
    beta: wp.float64,
    sigma: wp.float64,
    dt: wp.float64,
) -> wp.float64:
    vc = _v3d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    for j in range(dof_per_env):
        vj = v[env, j] + alpha * dv[env, j]
        vc = wp.vec3d(
            vc.x + contact_jac[env, c, 0, j] * vj,
            vc.y + contact_jac[env, c, 1, j] * vj,
            vc.z + contact_jac[env, c, 2, j] * vj,
        )
    return _contact_projection_cost_from_vc_sap(
        env,
        c,
        vc,
        contact_phi0,
        contact_w_eff,
        contact_mu,
        contact_k,
        contact_tau_d,
        beta,
        sigma,
        dt,
    )


@wp.func
def _contact_projection_cost_from_vc_sap_f32(
    env: int,
    c: int,
    vc: wp.vec3,
    contact_phi0: wp.array(dtype=wp.float32, ndim=2),
    contact_w_eff: wp.array(dtype=wp.float32, ndim=2),
    contact_mu: wp.array(dtype=wp.float32, ndim=2),
    contact_k: wp.array(dtype=wp.float32, ndim=2),
    contact_tau_d: wp.array(dtype=wp.float32, ndim=2),
    beta: wp.float32,
    sigma: wp.float32,
    dt: wp.float32,
) -> wp.float32:
    wi = contact_w_eff[env, c]
    if wi < wp.float32(1.0e-12) or not wp.isfinite(wi):
        wi = wp.float32(1.0e-12)

    beta_factor = beta * beta / (wp.float32(4.0) * wp.float32(_PI) * wp.float32(_PI))
    rn_hard = beta_factor * wi
    k_c = contact_k[env, c]
    if k_c <= wp.float32(0.0) or not wp.isfinite(k_c):
        k_c = wp.float32(1.0)
    tau_c = wp.max(contact_tau_d[env, c], wp.float32(0.0))
    rn_soft = wp.float32(1.0) / (dt * k_c * (dt + tau_c))
    rn = wp.max(rn_hard, rn_soft)
    rt = sigma * wi
    if rt < wp.float32(1.0e-30):
        rt = wp.float32(1.0e-30)
    if rn < wp.float32(1.0e-30):
        rn = wp.float32(1.0e-30)

    rt_inv = wp.float32(1.0) / rt
    rn_inv = wp.float32(1.0) / rn
    vhat_n = -contact_phi0[env, c] / (dt + tau_c)
    y = wp.vec3(-rt_inv * vc.x, -rt_inv * vc.y, rn_inv * (vhat_n - vc.z))

    mu = contact_mu[env, c]
    if mu < wp.float32(0.0) or not wp.isfinite(mu):
        mu = wp.float32(0.0)

    yr = wp.sqrt(
        y.x * y.x
        + y.y * y.y
        + wp.float32(_CONTACT_SOFT_NORM_TOL) * wp.float32(_CONTACT_SOFT_NORM_TOL)
    )
    t_hat = _v3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if yr > wp.float32(0.0):
        t_hat = _v3f(y.x / yr, y.y / yr, wp.float32(0.0))

    gamma = _v3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    if mu <= wp.float32(1.0e-12):
        if y.z > wp.float32(0.0):
            gamma = _v3f(wp.float32(0.0), wp.float32(0.0), y.z)
    else:
        mu_tilde = mu * wp.sqrt(rt / rn)
        mu_hat = mu * rt / rn
        factor = wp.float32(1.0) / (wp.float32(1.0) + mu_tilde * mu_tilde)

        if yr <= mu * y.z:
            gamma = y
        elif (-mu_hat * yr < y.z) and (y.z < yr / mu):
            gamma_n = (y.z + mu_hat * yr) * factor
            gamma = _v3f(mu * gamma_n * t_hat.x, mu * gamma_n * t_hat.y, gamma_n)

    return wp.float32(0.5) * (
        rt * (gamma.x * gamma.x + gamma.y * gamma.y) + rn * gamma.z * gamma.z
    )


@wp.func
def _contact_projection_cost_from_velocity_sap_f32(
    env: int,
    c: int,
    dof_per_env: int,
    contact_jac: wp.array(dtype=wp.float32, ndim=4),
    contact_phi0: wp.array(dtype=wp.float32, ndim=2),
    contact_w_eff: wp.array(dtype=wp.float32, ndim=2),
    contact_mu: wp.array(dtype=wp.float32, ndim=2),
    contact_k: wp.array(dtype=wp.float32, ndim=2),
    contact_tau_d: wp.array(dtype=wp.float32, ndim=2),
    v: wp.array(dtype=wp.float32, ndim=2),
    dv: wp.array(dtype=wp.float32, ndim=2),
    alpha: wp.float32,
    beta: wp.float32,
    sigma: wp.float32,
    dt: wp.float32,
) -> wp.float32:
    vc = _v3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    for j in range(dof_per_env):
        vj = v[env, j] + alpha * dv[env, j]
        vc = wp.vec3(
            vc.x + contact_jac[env, c, 0, j] * vj,
            vc.y + contact_jac[env, c, 1, j] * vj,
            vc.z + contact_jac[env, c, 2, j] * vj,
        )
    return _contact_projection_cost_from_vc_sap_f32(
        env,
        c,
        vc,
        contact_phi0,
        contact_w_eff,
        contact_mu,
        contact_k,
        contact_tau_d,
        beta,
        sigma,
        dt,
    )


@wp.func
def _sap_armijo_ok(
    alpha: wp.float64,
    ell: wp.float64,
    ell0: wp.float64,
    dell0: wp.float64,
    c: wp.float64,
) -> bool:
    return ell < ell0 + c * alpha * dell0


@wp.func
def _sap_armijo_ok_f32(
    alpha: wp.float32,
    ell: wp.float32,
    ell0: wp.float32,
    dell0: wp.float32,
    c: wp.float32,
) -> bool:
    return ell < ell0 + c * alpha * dell0
