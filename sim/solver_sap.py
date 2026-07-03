from __future__ import annotations

import warnings

import numpy as np
import warp as wp

from sim.contact_jacobian import SapContactJacobian, SapContactJacobianResult
from sim.contact_solve import (
    SapContactSolve,
    SapContactSolveResult,
    normalize_sap_line_search_mode,
)
from sim.sap_helpers import (
    _copy_public_body_force_to_sap_body_origin_kernel,
    _copy_public_body_force_to_sap_body_origin_kernel_body_qd,
    _copy_public_to_sap_force_kernel,
    _copy_public_to_sap_force_kernel_f64,
    _copy_public_to_sap_force_kernel_f64_from_joint_pose,
    _copy_sap_to_public_velocity_f32_from_joint_pose,
    _copy_sap_to_public_velocity_f64_from_joint_pose,
    copy_public_to_sap_velocity_f32,
    copy_public_to_sap_velocity_f64,
    copy_public_to_sap_velocity_f64_from_joint_pose,
)
from sim.sap_kinematics import sap_eval_fk
from sim.sap_runtime import (
    Model,
    SAP_JOINT_BALL,
    SAP_JOINT_D6,
    SAP_JOINT_DISTANCE,
    SAP_JOINT_FIXED,
    SAP_JOINT_FREE,
    SAP_JOINT_PRISMATIC,
    SAP_JOINT_REVOLUTE,
    SapContacts,
    SapControl,
    SapModel,
    SapState,
    sap_contacts_from_newton,
)

wp.config.enable_backward = False


@wp.kernel
def _copy_f64_to_f32(
    src: wp.array(dtype=wp.float64),
    dst: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    dst[i] = wp.float32(src[i])


@wp.kernel
def _copy_f32_to_f64(
    src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    dst[i] = wp.float64(src[i])


def _normalize_precision_knob(value: str, *, option_name: str) -> str:
    precision = str(value).strip().lower()
    if precision == "f32":
        precision = "fp32"
    elif precision == "f64":
        precision = "fp64"
    if precision not in {"fp32", "fp64"}:
        raise ValueError(f"{option_name} must be 'fp32'/'f32' or 'fp64'/'f64', got {value!r}.")
    return precision


SAP_CONTACT_PRESET_VARIANTS = ("approx32", "approx64", "drake")

_CONTACT_PRESET_KWARGS = {
    "approx32": {
        "use_f64_boundary_pose": False,
        "free_motion_solve_precision": "fp32",
        "contact_solve_precision": "fp64",
        "contact_linear_solve_precision": "fp32",
        "sap_contact_weight_precision": "fp32",
        "contact_weight_mode": "body_inertia",
        "contact_point_mode": "witness_point",
        "position_integration": "midpoint",
    },
    "approx64": {
        "use_f64_boundary_pose": True,
        "free_motion_solve_precision": "fp64",
        "contact_solve_precision": "fp64",
        "contact_linear_solve_precision": "fp64",
        "sap_contact_weight_precision": "fp64",
        "contact_weight_mode": "body_inertia",
        "contact_point_mode": "witness_point",
        "position_integration": "midpoint",
    },
    "drake": {
        "use_f64_boundary_pose": True,
        "free_motion_solve_precision": "fp64",
        "contact_solve_precision": "fp64",
        "contact_linear_solve_precision": "fp64",
        "sap_contact_weight_precision": "fp64",
        "contact_weight_mode": "diag_delassus",
        "contact_point_mode": "contact_midpoint",
        "position_integration": "sap_euler",
    },
}


def _normalize_contact_preset_variant(value: str | None) -> str:
    if value is None:
        return "approx32"
    variant = str(value).strip().lower().replace("-", "_")
    if variant == "approx_32":
        variant = "approx32"
    elif variant == "approx_64":
        variant = "approx64"
    if variant not in _CONTACT_PRESET_KWARGS:
        choices = ", ".join(SAP_CONTACT_PRESET_VARIANTS)
        raise ValueError(f"contact_preset_variant must be one of {choices}, got {value!r}.")
    return variant


def _contact_preset_kwargs(contact_preset_variant: str | None) -> dict[str, object]:
    return dict(_CONTACT_PRESET_KWARGS[_normalize_contact_preset_variant(contact_preset_variant)])


def _infer_contact_tau_d_fallback(model: Model) -> float:
    for name in ("sap_debug_shape_material_tau", "shape_material_tau"):
        src = getattr(model, name, None)
        if src is None:
            continue
        if isinstance(src, wp.array):
            values = np.asarray(src.numpy(), dtype=np.float64).reshape(-1)
        else:
            values = np.asarray(src, dtype=np.float64).reshape(-1)
        explicit = values[np.isfinite(values) & (values >= 0.0)]
        if explicit.size > 0:
            return float(explicit[0])
    return 0.0


@wp.func
def _quat_rotate_vec3(q: wp.quat, v: wp.vec3) -> wp.vec3:
    norm = wp.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm <= wp.float32(0.0):
        return v
    inv_norm = wp.float32(1.0) / norm
    x = q.x * inv_norm
    y = q.y * inv_norm
    z = q.z * inv_norm
    w = q.w * inv_norm
    qv = wp.vec3(x, y, z)
    t = wp.float32(2.0) * wp.cross(qv, v)
    return v + w * t + wp.cross(qv, t)

@wp.func
def _quat_sap_euler_xyzw(q: wp.quat, w: wp.vec3d, dt: wp.float64) -> wp.quat:
    # SAP's quaternion floating mobilizer maps angular velocity with
    # qdot = 0.5 * [0, w] * q. Runtime quaternions are stored as xyzw.
    qx = wp.float64(q.x)
    qy = wp.float64(q.y)
    qz = wp.float64(q.z)
    qw = wp.float64(q.w)
    half_dt = wp.float64(0.5) * dt
    x = qx + half_dt * (qw * w.x + w.y * qz - w.z * qy)
    y = qy + half_dt * (qw * w.y + w.z * qx - w.x * qz)
    z = qz + half_dt * (qw * w.z + w.x * qy - w.y * qx)
    s = qw - half_dt * (w.x * qx + w.y * qy + w.z * qz)
    norm = wp.sqrt(x * x + y * y + z * z + s * s)
    if norm > wp.float64(0.0):
        inv_norm = wp.float64(1.0) / norm
        x = x * inv_norm
        y = y * inv_norm
        z = z * inv_norm
        s = s * inv_norm
    return wp.quat(wp.float32(x), wp.float32(y), wp.float32(z), wp.float32(s))


@wp.func
def _quat_sap_euler_xyzw_f32(q: wp.quat, w: wp.vec3, dt: wp.float32) -> wp.quat:
    # SAP's quaternion floating mobilizer maps angular velocity with
    # qdot = 0.5 * [0, w] * q. Runtime quaternions are stored as xyzw.
    half_dt = wp.float32(0.5) * dt
    x = q.x + half_dt * (q.w * w.x + w.y * q.z - w.z * q.y)
    y = q.y + half_dt * (q.w * w.y + w.z * q.x - w.x * q.z)
    z = q.z + half_dt * (q.w * w.z + w.x * q.y - w.y * q.x)
    s = q.w - half_dt * (w.x * q.x + w.y * q.y + w.z * q.z)
    norm = wp.sqrt(x * x + y * y + z * z + s * s)
    if norm > wp.float32(0.0):
        inv_norm = wp.float32(1.0) / norm
        x = x * inv_norm
        y = y * inv_norm
        z = z * inv_norm
        s = s * inv_norm
    return wp.quat(x, y, z, s)


@wp.func
def _quat_sap_euler_xyzw_d(q: wp.quatd, w: wp.vec3d, dt: wp.float64) -> wp.quatd:
    qx = wp.float64(q.x)
    qy = wp.float64(q.y)
    qz = wp.float64(q.z)
    qw = wp.float64(q.w)
    half_dt = wp.float64(0.5) * dt
    x = qx + half_dt * (qw * w.x + w.y * qz - w.z * qy)
    y = qy + half_dt * (qw * w.y + w.z * qx - w.x * qz)
    z = qz + half_dt * (qw * w.z + w.x * qy - w.y * qx)
    s = qw - half_dt * (w.x * qx + w.y * qy + w.z * qz)
    norm = wp.sqrt(x * x + y * y + z * z + s * s)
    if norm > wp.float64(0.0):
        inv_norm = wp.float64(1.0) / norm
        x = x * inv_norm
        y = y * inv_norm
        z = z * inv_norm
        s = s * inv_norm
    return wp.quatd(x, y, z, s)


@wp.func
def _quat_midpoint_xyzw(q: wp.quat, w_mid: wp.vec3d, dt: wp.float64) -> wp.quat:
    angle = wp.length(w_mid) * dt
    q_next = q
    if angle > wp.float64(1.0e-12):
        axis = wp.normalize(w_mid)
        half_angle = wp.float64(0.5) * angle
        s = wp.sin(half_angle)
        c = wp.cos(half_angle)
        r = wp.quatd(axis.x * s, axis.y * s, axis.z * s, c)
        qd = wp.quatd(
            wp.float64(q.x),
            wp.float64(q.y),
            wp.float64(q.z),
            wp.float64(q.w),
        )
        out = r * qd
        norm = wp.sqrt(out.x * out.x + out.y * out.y + out.z * out.z + out.w * out.w)
        if norm > wp.float64(0.0):
            inv_norm = wp.float64(1.0) / norm
            out = wp.quatd(out.x * inv_norm, out.y * inv_norm, out.z * inv_norm, out.w * inv_norm)
        q_next = wp.quat(wp.float32(out.x), wp.float32(out.y), wp.float32(out.z), wp.float32(out.w))
    return q_next


@wp.func
def _quat_midpoint_xyzw_f32(q: wp.quat, w_mid: wp.vec3, dt: wp.float32) -> wp.quat:
    angle = wp.length(w_mid) * dt
    q_next = q
    if angle > wp.float32(1.0e-12):
        axis = wp.normalize(w_mid)
        half_angle = wp.float32(0.5) * angle
        s = wp.sin(half_angle)
        c = wp.cos(half_angle)
        r = wp.quat(axis.x * s, axis.y * s, axis.z * s, c)
        out = r * q
        norm = wp.sqrt(out.x * out.x + out.y * out.y + out.z * out.z + out.w * out.w)
        if norm > wp.float32(0.0):
            inv_norm = wp.float32(1.0) / norm
            out = wp.quat(out.x * inv_norm, out.y * inv_norm, out.z * inv_norm, out.w * inv_norm)
        q_next = out
    return q_next


@wp.func
def _quat_midpoint_xyzw_d(q: wp.quatd, w_mid: wp.vec3d, dt: wp.float64) -> wp.quatd:
    angle = wp.length(w_mid) * dt
    q_next = q
    if angle > wp.float64(1.0e-12):
        axis = wp.normalize(w_mid)
        half_angle = wp.float64(0.5) * angle
        s = wp.sin(half_angle)
        c = wp.cos(half_angle)
        r = wp.quatd(axis.x * s, axis.y * s, axis.z * s, c)
        q_next = r * q
        norm = wp.sqrt(q_next.x * q_next.x + q_next.y * q_next.y + q_next.z * q_next.z + q_next.w * q_next.w)
        if norm > wp.float64(0.0):
            inv_norm = wp.float64(1.0) / norm
            q_next = wp.quatd(
                q_next.x * inv_norm,
                q_next.y * inv_norm,
                q_next.z * inv_norm,
                q_next.w * inv_norm,
            )
    return q_next


@wp.kernel
def _integrate_generalized_positions_sap_euler(
    joint_type: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3d),
    joint_q_in: wp.array(dtype=wp.float32),
    v_sap: wp.array(dtype=wp.float64),
    joint_world: wp.array(dtype=int),
    dt: wp.array(dtype=wp.float64),
    joint_q_out: wp.array(dtype=wp.float32),
    joint_qd_out: wp.array(dtype=wp.float32),
):
    joint = wp.tid()
    q_start = joint_q_start[joint]
    qd_start = joint_qd_start[joint]
    axis_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
    jtype = joint_type[joint]
    h = wp.float32(dt[joint_world[joint]])

    if jtype == SAP_JOINT_PRISMATIC or jtype == SAP_JOINT_REVOLUTE:
        qd = wp.float32(v_sap[qd_start])
        joint_q_out[q_start] = joint_q_in[q_start] + h * qd
        joint_qd_out[qd_start] = qd
        return

    if jtype == SAP_JOINT_BALL:
        q = wp.quat(
            joint_q_in[q_start + 0],
            joint_q_in[q_start + 1],
            joint_q_in[q_start + 2],
            joint_q_in[q_start + 3],
        )
        w = wp.vec3(
            wp.float32(v_sap[qd_start + 0]),
            wp.float32(v_sap[qd_start + 1]),
            wp.float32(v_sap[qd_start + 2]),
        )
        q_next = _quat_sap_euler_xyzw_f32(q, w, h)
        joint_q_out[q_start + 0] = q_next.x
        joint_q_out[q_start + 1] = q_next.y
        joint_q_out[q_start + 2] = q_next.z
        joint_q_out[q_start + 3] = q_next.w
        joint_qd_out[qd_start + 0] = wp.float32(w.x)
        joint_qd_out[qd_start + 1] = wp.float32(w.y)
        joint_qd_out[qd_start + 2] = wp.float32(w.z)
        return

    if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
        q = wp.quat(
            joint_q_in[q_start + 3],
            joint_q_in[q_start + 4],
            joint_q_in[q_start + 5],
            joint_q_in[q_start + 6],
        )
        w = wp.vec3(
            wp.float32(v_sap[qd_start + 0]),
            wp.float32(v_sap[qd_start + 1]),
            wp.float32(v_sap[qd_start + 2]),
        )
        v_origin = wp.vec3(
            wp.float32(v_sap[qd_start + 3]),
            wp.float32(v_sap[qd_start + 4]),
            wp.float32(v_sap[qd_start + 5]),
        )
        q_next = _quat_sap_euler_xyzw_f32(q, w, h)
        joint_q_out[q_start + 0] = joint_q_in[q_start + 0] + h * v_origin.x
        joint_q_out[q_start + 1] = joint_q_in[q_start + 1] + h * v_origin.y
        joint_q_out[q_start + 2] = joint_q_in[q_start + 2] + h * v_origin.z
        joint_q_out[q_start + 3] = q_next.x
        joint_q_out[q_start + 4] = q_next.y
        joint_q_out[q_start + 5] = q_next.z
        joint_q_out[q_start + 6] = q_next.w
        child = joint_child[joint]
        body_com_f = wp.vec3(
            wp.float32(body_com[child].x),
            wp.float32(body_com[child].y),
            wp.float32(body_com[child].z),
        )
        r_com = _quat_rotate_vec3(q_next, body_com_f)
        v_com = v_origin + wp.cross(w, r_com)
        joint_qd_out[qd_start + 0] = v_com.x
        joint_qd_out[qd_start + 1] = v_com.y
        joint_qd_out[qd_start + 2] = v_com.z
        joint_qd_out[qd_start + 3] = w.x
        joint_qd_out[qd_start + 4] = w.y
        joint_qd_out[qd_start + 5] = w.z
        return

    if jtype == SAP_JOINT_D6:
        for axis in range(axis_count):
            qd = wp.float32(v_sap[qd_start + axis])
            joint_q_out[q_start + axis] = joint_q_in[q_start + axis] + h * qd
            joint_qd_out[qd_start + axis] = qd
        return

    if jtype == SAP_JOINT_FIXED:
        return

    for axis in range(axis_count):
        qd = wp.float32(v_sap[qd_start + axis])
        joint_q_out[q_start + axis] = joint_q_in[q_start + axis] + h * qd
        joint_qd_out[qd_start + axis] = qd


@wp.kernel
def _integrate_generalized_positions_midpoint(
    joint_type: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3d),
    joint_q_in: wp.array(dtype=wp.float32),
    v_prev_sap: wp.array(dtype=wp.float64),
    v_new_sap: wp.array(dtype=wp.float64),
    joint_world: wp.array(dtype=int),
    dt: wp.array(dtype=wp.float64),
    joint_q_out: wp.array(dtype=wp.float32),
    joint_qd_out: wp.array(dtype=wp.float32),
):
    joint = wp.tid()
    q_start = joint_q_start[joint]
    qd_start = joint_qd_start[joint]
    axis_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
    jtype = joint_type[joint]
    h = wp.float32(dt[joint_world[joint]])

    if jtype == SAP_JOINT_PRISMATIC or jtype == SAP_JOINT_REVOLUTE:
        qd = wp.float32(v_new_sap[qd_start])
        qd_mid = wp.float32(0.5) * (wp.float32(v_prev_sap[qd_start]) + qd)
        joint_q_out[q_start] = joint_q_in[q_start] + h * qd_mid
        joint_qd_out[qd_start] = qd
        return

    if jtype == SAP_JOINT_BALL:
        q = wp.quat(
            joint_q_in[q_start + 0],
            joint_q_in[q_start + 1],
            joint_q_in[q_start + 2],
            joint_q_in[q_start + 3],
        )
        w_old = wp.vec3(
            wp.float32(v_prev_sap[qd_start + 0]),
            wp.float32(v_prev_sap[qd_start + 1]),
            wp.float32(v_prev_sap[qd_start + 2]),
        )
        w_new = wp.vec3(
            wp.float32(v_new_sap[qd_start + 0]),
            wp.float32(v_new_sap[qd_start + 1]),
            wp.float32(v_new_sap[qd_start + 2]),
        )
        q_next = _quat_midpoint_xyzw_f32(q, wp.float32(0.5) * (w_old + w_new), h)
        joint_q_out[q_start + 0] = q_next.x
        joint_q_out[q_start + 1] = q_next.y
        joint_q_out[q_start + 2] = q_next.z
        joint_q_out[q_start + 3] = q_next.w
        joint_qd_out[qd_start + 0] = wp.float32(w_new.x)
        joint_qd_out[qd_start + 1] = wp.float32(w_new.y)
        joint_qd_out[qd_start + 2] = wp.float32(w_new.z)
        return

    if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
        q = wp.quat(
            joint_q_in[q_start + 3],
            joint_q_in[q_start + 4],
            joint_q_in[q_start + 5],
            joint_q_in[q_start + 6],
        )
        w_old = wp.vec3(
            wp.float32(v_prev_sap[qd_start + 0]),
            wp.float32(v_prev_sap[qd_start + 1]),
            wp.float32(v_prev_sap[qd_start + 2]),
        )
        v_origin_old = wp.vec3(
            wp.float32(v_prev_sap[qd_start + 3]),
            wp.float32(v_prev_sap[qd_start + 4]),
            wp.float32(v_prev_sap[qd_start + 5]),
        )
        w_new = wp.vec3(
            wp.float32(v_new_sap[qd_start + 0]),
            wp.float32(v_new_sap[qd_start + 1]),
            wp.float32(v_new_sap[qd_start + 2]),
        )
        v_origin_new = wp.vec3(
            wp.float32(v_new_sap[qd_start + 3]),
            wp.float32(v_new_sap[qd_start + 4]),
            wp.float32(v_new_sap[qd_start + 5]),
        )
        q_next = _quat_midpoint_xyzw_f32(q, wp.float32(0.5) * (w_old + w_new), h)
        v_origin_mid = wp.float32(0.5) * (v_origin_old + v_origin_new)
        joint_q_out[q_start + 0] = joint_q_in[q_start + 0] + h * v_origin_mid.x
        joint_q_out[q_start + 1] = joint_q_in[q_start + 1] + h * v_origin_mid.y
        joint_q_out[q_start + 2] = joint_q_in[q_start + 2] + h * v_origin_mid.z
        joint_q_out[q_start + 3] = q_next.x
        joint_q_out[q_start + 4] = q_next.y
        joint_q_out[q_start + 5] = q_next.z
        joint_q_out[q_start + 6] = q_next.w
        child = joint_child[joint]
        body_com_f = wp.vec3(
            wp.float32(body_com[child].x),
            wp.float32(body_com[child].y),
            wp.float32(body_com[child].z),
        )
        r_com = _quat_rotate_vec3(q_next, body_com_f)
        v_com = v_origin_new + wp.cross(w_new, r_com)
        joint_qd_out[qd_start + 0] = v_com.x
        joint_qd_out[qd_start + 1] = v_com.y
        joint_qd_out[qd_start + 2] = v_com.z
        joint_qd_out[qd_start + 3] = w_new.x
        joint_qd_out[qd_start + 4] = w_new.y
        joint_qd_out[qd_start + 5] = w_new.z
        return

    if jtype == SAP_JOINT_D6:
        for axis in range(axis_count):
            qd = wp.float32(v_new_sap[qd_start + axis])
            qd_mid = wp.float32(0.5) * (wp.float32(v_prev_sap[qd_start + axis]) + qd)
            joint_q_out[q_start + axis] = joint_q_in[q_start + axis] + h * qd_mid
            joint_qd_out[qd_start + axis] = qd
        return

    if jtype == SAP_JOINT_FIXED:
        return

    for axis in range(axis_count):
        qd = wp.float32(v_new_sap[qd_start + axis])
        qd_mid = wp.float32(0.5) * (wp.float32(v_prev_sap[qd_start + axis]) + qd)
        joint_q_out[q_start + axis] = joint_q_in[q_start + axis] + h * qd_mid
        joint_qd_out[qd_start + axis] = qd


@wp.kernel
def _integrate_generalized_positions_sap_euler_f64(
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_q_in: wp.array(dtype=wp.float64),
    v_sap: wp.array(dtype=wp.float64),
    joint_world: wp.array(dtype=int),
    dt: wp.array(dtype=wp.float64),
    joint_q_out: wp.array(dtype=wp.float64),
    joint_qd_out: wp.array(dtype=wp.float64),
):
    joint = wp.tid()
    q_start = joint_q_start[joint]
    qd_start = joint_qd_start[joint]
    axis_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
    jtype = joint_type[joint]
    h = wp.float64(dt[joint_world[joint]])

    if jtype == SAP_JOINT_PRISMATIC or jtype == SAP_JOINT_REVOLUTE:
        qd = v_sap[qd_start]
        joint_q_out[q_start] = joint_q_in[q_start] + h * qd
        joint_qd_out[qd_start] = qd
        return

    if jtype == SAP_JOINT_BALL:
        q = wp.quatd(
            joint_q_in[q_start + 0],
            joint_q_in[q_start + 1],
            joint_q_in[q_start + 2],
            joint_q_in[q_start + 3],
        )
        w = wp.vec3d(
            v_sap[qd_start + 0],
            v_sap[qd_start + 1],
            v_sap[qd_start + 2],
        )
        q_next = _quat_sap_euler_xyzw_d(q, w, h)
        joint_q_out[q_start + 0] = q_next.x
        joint_q_out[q_start + 1] = q_next.y
        joint_q_out[q_start + 2] = q_next.z
        joint_q_out[q_start + 3] = q_next.w
        joint_qd_out[qd_start + 0] = w.x
        joint_qd_out[qd_start + 1] = w.y
        joint_qd_out[qd_start + 2] = w.z
        return

    if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
        q = wp.quatd(
            joint_q_in[q_start + 3],
            joint_q_in[q_start + 4],
            joint_q_in[q_start + 5],
            joint_q_in[q_start + 6],
        )
        w = wp.vec3d(
            v_sap[qd_start + 0],
            v_sap[qd_start + 1],
            v_sap[qd_start + 2],
        )
        v_origin = wp.vec3d(
            v_sap[qd_start + 3],
            v_sap[qd_start + 4],
            v_sap[qd_start + 5],
        )
        q_next = _quat_sap_euler_xyzw_d(q, w, h)
        joint_q_out[q_start + 0] = joint_q_in[q_start + 0] + h * v_origin.x
        joint_q_out[q_start + 1] = joint_q_in[q_start + 1] + h * v_origin.y
        joint_q_out[q_start + 2] = joint_q_in[q_start + 2] + h * v_origin.z
        joint_q_out[q_start + 3] = q_next.x
        joint_q_out[q_start + 4] = q_next.y
        joint_q_out[q_start + 5] = q_next.z
        joint_q_out[q_start + 6] = q_next.w
        joint_qd_out[qd_start + 0] = v_origin.x
        joint_qd_out[qd_start + 1] = v_origin.y
        joint_qd_out[qd_start + 2] = v_origin.z
        joint_qd_out[qd_start + 3] = w.x
        joint_qd_out[qd_start + 4] = w.y
        joint_qd_out[qd_start + 5] = w.z
        return

    if jtype == SAP_JOINT_D6:
        for axis in range(axis_count):
            qd = v_sap[qd_start + axis]
            joint_q_out[q_start + axis] = joint_q_in[q_start + axis] + h * qd
            joint_qd_out[qd_start + axis] = qd
        return

    if jtype == SAP_JOINT_FIXED:
        return

    for axis in range(axis_count):
        qd = v_sap[qd_start + axis]
        joint_q_out[q_start + axis] = joint_q_in[q_start + axis] + h * qd
        joint_qd_out[qd_start + axis] = qd


@wp.kernel
def _integrate_generalized_positions_midpoint_f64(
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_q_in: wp.array(dtype=wp.float64),
    v_prev_sap: wp.array(dtype=wp.float64),
    v_new_sap: wp.array(dtype=wp.float64),
    joint_world: wp.array(dtype=int),
    dt: wp.array(dtype=wp.float64),
    joint_q_out: wp.array(dtype=wp.float64),
    joint_qd_out: wp.array(dtype=wp.float64),
):
    joint = wp.tid()
    q_start = joint_q_start[joint]
    qd_start = joint_qd_start[joint]
    axis_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]
    jtype = joint_type[joint]
    h = wp.float64(dt[joint_world[joint]])

    if jtype == SAP_JOINT_PRISMATIC or jtype == SAP_JOINT_REVOLUTE:
        qd = v_new_sap[qd_start]
        qd_mid = wp.float64(0.5) * (v_prev_sap[qd_start] + qd)
        joint_q_out[q_start] = joint_q_in[q_start] + h * qd_mid
        joint_qd_out[qd_start] = qd
        return

    if jtype == SAP_JOINT_BALL:
        q = wp.quatd(
            joint_q_in[q_start + 0],
            joint_q_in[q_start + 1],
            joint_q_in[q_start + 2],
            joint_q_in[q_start + 3],
        )
        w_old = wp.vec3d(
            v_prev_sap[qd_start + 0],
            v_prev_sap[qd_start + 1],
            v_prev_sap[qd_start + 2],
        )
        w_new = wp.vec3d(
            v_new_sap[qd_start + 0],
            v_new_sap[qd_start + 1],
            v_new_sap[qd_start + 2],
        )
        q_next = _quat_midpoint_xyzw_d(q, wp.float64(0.5) * (w_old + w_new), h)
        joint_q_out[q_start + 0] = q_next.x
        joint_q_out[q_start + 1] = q_next.y
        joint_q_out[q_start + 2] = q_next.z
        joint_q_out[q_start + 3] = q_next.w
        joint_qd_out[qd_start + 0] = w_new.x
        joint_qd_out[qd_start + 1] = w_new.y
        joint_qd_out[qd_start + 2] = w_new.z
        return

    if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
        q = wp.quatd(
            joint_q_in[q_start + 3],
            joint_q_in[q_start + 4],
            joint_q_in[q_start + 5],
            joint_q_in[q_start + 6],
        )
        w_old = wp.vec3d(
            v_prev_sap[qd_start + 0],
            v_prev_sap[qd_start + 1],
            v_prev_sap[qd_start + 2],
        )
        v_origin_old = wp.vec3d(
            v_prev_sap[qd_start + 3],
            v_prev_sap[qd_start + 4],
            v_prev_sap[qd_start + 5],
        )
        w_new = wp.vec3d(
            v_new_sap[qd_start + 0],
            v_new_sap[qd_start + 1],
            v_new_sap[qd_start + 2],
        )
        v_origin_new = wp.vec3d(
            v_new_sap[qd_start + 3],
            v_new_sap[qd_start + 4],
            v_new_sap[qd_start + 5],
        )
        q_next = _quat_midpoint_xyzw_d(q, wp.float64(0.5) * (w_old + w_new), h)
        v_origin_mid = wp.float64(0.5) * (v_origin_old + v_origin_new)
        joint_q_out[q_start + 0] = joint_q_in[q_start + 0] + h * v_origin_mid.x
        joint_q_out[q_start + 1] = joint_q_in[q_start + 1] + h * v_origin_mid.y
        joint_q_out[q_start + 2] = joint_q_in[q_start + 2] + h * v_origin_mid.z
        joint_q_out[q_start + 3] = q_next.x
        joint_q_out[q_start + 4] = q_next.y
        joint_q_out[q_start + 5] = q_next.z
        joint_q_out[q_start + 6] = q_next.w
        joint_qd_out[qd_start + 0] = v_origin_new.x
        joint_qd_out[qd_start + 1] = v_origin_new.y
        joint_qd_out[qd_start + 2] = v_origin_new.z
        joint_qd_out[qd_start + 3] = w_new.x
        joint_qd_out[qd_start + 4] = w_new.y
        joint_qd_out[qd_start + 5] = w_new.z
        return

    if jtype == SAP_JOINT_D6:
        for axis in range(axis_count):
            qd = v_new_sap[qd_start + axis]
            qd_mid = wp.float64(0.5) * (v_prev_sap[qd_start + axis] + qd)
            joint_q_out[q_start + axis] = joint_q_in[q_start + axis] + h * qd_mid
            joint_qd_out[qd_start + axis] = qd
        return

    if jtype == SAP_JOINT_FIXED:
        return

    for axis in range(axis_count):
        qd = v_new_sap[qd_start + axis]
        qd_mid = wp.float64(0.5) * (v_prev_sap[qd_start + axis] + qd)
        joint_q_out[q_start + axis] = joint_q_in[q_start + axis] + h * qd_mid
        joint_qd_out[qd_start + axis] = qd


class SolverSAP:
    """SAP-native solver pipeline.

    This solver wires together the SAP runtime components:

    1. `SapFreeMotion.compute()` computes SAP-order free motion.
    2. `SapContactJacobian.compute()` computes contact Jacobians and
       env-local dynamics matrices from SAP runtime data.
    3. `SapContactSolve.solve()` solves the SAP stage2 problem in SAP-order
       generalized velocities.
    4. SAP-native integration writes `state_out`.
    """

    def __init__(
        self,
        model: Model,
        *,
        max_rigid_contact: int = 128,
        fallback_mu: float | None = None,
        fallback_stiffness: float = 1.0e10,
        contact_beta: float = 1.0,
        contact_sigma: float = 1.0e-3,
        contact_tau_d: float | None = None,
        block_size: int | None = None,
        diag_shift: float = 0.0,
        max_iterations: int = 100,
        optimality_abs_tol: float = 1.0e-14,
        optimality_rel_tol: float = 1.0e-6,
        cost_abs_tol: float | None = None,
        cost_rel_tol: float | None = None,
        line_search_max_iterations: int = 40,
        armijo_c: float = 1.0e-4,
        rho: float = 0.8,
        line_search_relative_slop: float | None = None,
        # armijo_decay pairs with the Drake-tight cost tolerances (1e-30/1e-15) chosen
        # below; monotone_decay's loose 5e-3 cost early-exit can stop Newton on a
        # linesearch plateau before the gradient converges, leaving unconverged
        # (jittery) contact forces exactly in hard stiction/impact states.
        line_search_variant: str = "armijo_decay",
        contact_preset_variant: str | None = None,
        contact_weight_mode: str | None = None,
        contact_point_mode: str | None = None,
        capture_contact_jacobian_snapshots: bool = False,
        use_f64_boundary_pose: bool | None = None,
        free_motion_solve_precision: str | None = None,
        contact_solve_precision: str | None = None,
        contact_linear_solve_precision: str | None = None,
        sap_contact_weight_precision: str | None = None,
        collect_iteration_stats: bool = False,
        check_line_search_errors: bool = False,
        graph_conditional: bool = True,
        static_substep: bool = False,
        static_substep_iterations: int = 8,
        static_substep_line_search_iterations: int = 6,
        position_integration: str | None = None,
    ):
        if not isinstance(model, SapModel):
            raise TypeError("SolverSAP requires SapModel; convert frontend data before constructing the solver.")
        if int(model.joint_count) <= 0 or int(model.joint_dof_count) <= 0:
            raise ValueError("SolverSAP requires a model with articulated joint DOFs.")

        preset_kwargs = _contact_preset_kwargs(contact_preset_variant)
        self.contact_preset_variant = _normalize_contact_preset_variant(contact_preset_variant)
        if use_f64_boundary_pose is None:
            use_f64_boundary_pose = bool(preset_kwargs["use_f64_boundary_pose"])
        if free_motion_solve_precision is None:
            free_motion_solve_precision = str(preset_kwargs["free_motion_solve_precision"])
        if contact_solve_precision is None:
            contact_solve_precision = str(preset_kwargs["contact_solve_precision"])
        if contact_linear_solve_precision is None:
            contact_linear_solve_precision = str(preset_kwargs["contact_linear_solve_precision"])
        if sap_contact_weight_precision is None:
            sap_contact_weight_precision = str(preset_kwargs["sap_contact_weight_precision"])
        if contact_weight_mode is None:
            contact_weight_mode = str(preset_kwargs["contact_weight_mode"])
        if contact_point_mode is None:
            contact_point_mode = str(preset_kwargs["contact_point_mode"])
        if position_integration is None:
            position_integration = str(preset_kwargs["position_integration"])

        self.model = model
        self.max_rigid_contact = int(max_rigid_contact)
        self.max_iterations = int(max_iterations)
        self.optimality_abs_tol = float(optimality_abs_tol)
        self.optimality_rel_tol = float(optimality_rel_tol)
        self.armijo_c = float(armijo_c)
        self.rho = float(rho)
        self.line_search_relative_slop = line_search_relative_slop
        self.line_search_variant = normalize_sap_line_search_mode(line_search_variant)
        if self.line_search_variant == "armijo_decay" and not (0.0 < float(rho) < 1.0):
            raise ValueError(f"rho must lie in (0, 1), got {rho!r}.")
        if cost_abs_tol is None:
            cost_abs_tol = 0.0 if self.line_search_variant == "monotone_decay" else 1.0e-30
        if cost_rel_tol is None:
            cost_rel_tol = 5.0e-3 if self.line_search_variant == "monotone_decay" else 1.0e-15
        self.cost_abs_tol = float(cost_abs_tol)
        self.cost_rel_tol = float(cost_rel_tol)
        self.line_search_max_iterations = int(line_search_max_iterations)
        if self.line_search_variant == "exact_root" and self.line_search_max_iterations == 40:
            self.line_search_max_iterations = 100
        self.capture_contact_jacobian_snapshots = bool(capture_contact_jacobian_snapshots)
        self.use_f64_boundary_pose = bool(use_f64_boundary_pose)
        self.free_motion_solve_precision = _normalize_precision_knob(
            free_motion_solve_precision,
            option_name="free_motion_solve_precision",
        )
        self.contact_solve_precision = _normalize_precision_knob(
            contact_solve_precision,
            option_name="contact_solve_precision",
        )
        self.contact_linear_solve_precision = _normalize_precision_knob(
            contact_linear_solve_precision,
            option_name="contact_linear_solve_precision",
        )
        self.sap_contact_weight_precision = _normalize_precision_knob(
            sap_contact_weight_precision,
            option_name="sap_contact_weight_precision",
        )
        self.collect_iteration_stats = bool(collect_iteration_stats)
        self.check_line_search_errors = bool(check_line_search_errors)
        if not bool(graph_conditional):
            raise ValueError(
                "SolverSAP only supports graph_conditional=True; "
                "the Python contact-solve loop has been removed."
            )
        self.graph_conditional = True
        # static_substep: when True the inner contact solve runs a FIXED, NON-conditional
        # iteration count (Python for-loops instead of wp.capture_while / wp.capture_if), so
        # SolverSAP.step records as a flat launch stream with no nested conditional CUDA-graph
        # subgraphs. This is what lets an OUTER ScopedCapture (e.g. SolverSAPAdaptive's per-N
        # step-doubling graph) record SolverSAP.step without the nested-conditional-capture
        # node explosion that SIGABRTs at >=1024 envs. Default False preserves legacy behavior.
        self.static_substep = bool(static_substep)
        # When static_substep, the inner solve runs a FIXED iteration count. Unlike the conditional
        # path (which early-exits per env at convergence, so a large cap is cheap), a static loop
        # ALWAYS runs the full count and UNROLLS it into the captured graph -- using the conditional
        # cap (max_iterations=30, line_search=40) makes the per-N graph enormous and OOMs the graph
        # exec at >=1024 envs with many contacts. CENIC Sec. VI-B: under outer error control the
        # inner solve needs only ~kappa*eps_acc, so a small fixed count is numerically sufficient
        # (residual is absorbed by the outer error controller, which simply subdivides more). These
        # are the fixed counts used when static_substep=True.
        self.static_substep_iterations = int(static_substep_iterations)
        self.static_substep_line_search_iterations = int(static_substep_line_search_iterations)
        self.position_integration = str(position_integration).strip().lower().replace("-", "_")
        if self.position_integration not in {"sap_euler", "midpoint"}:
            raise ValueError(
                "position_integration must be 'sap_euler' or 'midpoint', "
                f"got {position_integration!r}."
            )

        shape_fallback_tau_d = (
            _infer_contact_tau_d_fallback(model) if contact_tau_d is None else float(contact_tau_d)
        )
        if contact_tau_d is None and shape_fallback_tau_d == 0.0:
            warnings.warn(
                "No contact dissipation authored on the model and no contact_tau_d given: "
                "tau_d falls back to 0.0, making contacts undamped springs at near-rigid "
                "stiffness (bounce/rattle). contact_tau_d is a PER-SHAPE fallback "
                "(contacting pairs use the sum of both shapes' tau); pass a value on the "
                "order of the simulation step (Drake guidance: tau_d ~ dt) for stable "
                "contact.",
                stacklevel=2,
            )
        self.sap_model = model
        self.contact_jacobian = SapContactJacobian(
            self.sap_model,
            max_rigid_contact=self.max_rigid_contact,
            fallback_mu=fallback_mu,
            fallback_stiffness=fallback_stiffness,
            fallback_tau_d=shape_fallback_tau_d,
            contact_weight_mode=contact_weight_mode,
            contact_point_mode=contact_point_mode,
            capture_local_snapshots=self.capture_contact_jacobian_snapshots,
            use_f64_boundary_pose=self.use_f64_boundary_pose,
            free_motion_solve_precision=self.free_motion_solve_precision,
            sap_contact_weight_precision=self.sap_contact_weight_precision,
        )
        # `SapContactJacobian` owns and reuses the free-motion component so
        # contact preparation and solver v* share the exact same buffers/basis.
        self.free_motion = self.contact_jacobian.free_motion
        self.contact_solve = SapContactSolve(
            self.sap_model,
            max_rigid_contact=self.max_rigid_contact,
            contact_beta=contact_beta,
            contact_sigma=contact_sigma,
            # SapContactSolve.contact_tau_d is PER-PAIR: the Jacobian combines
            # dissipation per contact as tau(shape0) + tau(shape1) (_contact_tau_pair),
            # so the solve-side fallback for two unauthored shapes is 2x the per-shape
            # fallback. Not a typo.
            contact_tau_d=shape_fallback_tau_d + shape_fallback_tau_d,
            block_size=block_size,
            diag_shift=diag_shift,
            solve_precision=self.contact_solve_precision,
            linear_solve_precision=self.contact_linear_solve_precision,
        )
        self._contact_solve_v_guess_active = wp.zeros(1, dtype=int, device=model.device)
        self._zero_joint_f = wp.zeros((model.joint_dof_count,), dtype=wp.float64, device=model.device)
        self._zero_control = SapControl(joint_f=self._zero_joint_f)
        self._integrate_joint_qd_f32 = wp.zeros((model.joint_dof_count,), dtype=wp.float32, device=model.device)
        self._integrate_v_f64 = wp.zeros((model.joint_dof_count,), dtype=wp.float64, device=model.device)
        self._boundary_joint_qd_in_sap = wp.zeros((model.joint_dof_count,), dtype=wp.float64, device=model.device)
        self._boundary_joint_f_sap = wp.zeros((model.joint_dof_count,), dtype=wp.float64, device=model.device)
        self._boundary_joint_qd_out_sap_f32 = wp.zeros(
            (model.joint_dof_count,),
            dtype=wp.float32,
            device=model.device,
        )
        self._boundary_body_f_sap = wp.zeros((model.body_count,), dtype=wp.spatial_vectord, device=model.device)
        self.dof_count = int(model.joint_dof_count)
        self.num_envs = int(getattr(model, "world_count", 1))
        self.dof_per_env = self.dof_count // max(self.num_envs, 1)
        # Per-world timestep buffer -- the single dt source for the position
        # integration kernels.  _update_dt_world() fills it from this step's dt:
        # a scalar fills uniformly (reproducing the legacy float(dt) path
        # bit-for-bit), a per-world array gives each env its own dt.  f64 to
        # match the legacy scalar precision.
        self._dt_world = wp.zeros(max(self.num_envs, 1), dtype=wp.float64, device=model.device)
        # joint -> world map (the world of each joint's child body), precomputed
        # once so joint-indexed kernels can read self._dt_world[joint_world[j]].
        _body_world = getattr(model, "body_world", None)
        if _body_world is not None and int(model.joint_count) > 0:
            _jc = model.joint_child.numpy()
            _bw = _body_world.numpy()
            _jw = np.zeros(int(model.joint_count), dtype=np.int32)
            _valid = _jc >= 0
            _jw[_valid] = _bw[_jc[_valid]]
            self._joint_world = wp.array(_jw, dtype=wp.int32, device=model.device)
        else:
            self._joint_world = wp.zeros(max(int(model.joint_count), 1), dtype=wp.int32, device=model.device)
        self.last_contacts: object | None = None
        self.last_contact_jacobian_result: SapContactJacobianResult | None = None
        self.last_contact_solve_result: SapContactSolveResult | None = None
        self.reset_runtime_state()

    @property
    def device(self):
        """Return the Warp device used by the solver model and scratch buffers."""
        return self.model.device

    def _update_dt_world(self, dt) -> None:
        """Fill self._dt_world (shape [num_envs], f64) with this step's timestep.

        A Python scalar fills every world uniformly -- what fixed-step and
        non-adaptive callers pass -- and reproduces the legacy float(dt) path
        bit-for-bit.  A per-world wp.array (length num_envs) gives each env its
        own dt (cast to f64 when the caller's array is f32).
        """
        if isinstance(dt, wp.array):
            n = int(self.num_envs)
            if int(dt.shape[0]) != n:
                raise ValueError(f"per-world dt length {int(dt.shape[0])} != num_envs {n}")
            if dt.dtype == wp.float64:
                wp.copy(self._dt_world, dt)
            else:
                wp.launch(_copy_f32_to_f64, dim=n, inputs=[dt, self._dt_world], device=self.model.device)
            return
        self._dt_world.fill_(float(dt))

    def get_max_contact_count(self) -> int:
        """Return the per-environment rigid-contact capacity used by the contact solve."""
        return int(self.max_rigid_contact) * max(int(self.num_envs), 1)

    def close(self) -> None:
        """Release solver-owned resources. The current implementation is a no-op placeholder for lifecycle
        symmetry.
        """
        pass

    def reset_runtime_state(self, state: SapState | None = None) -> None:
        """Reset timestep counters and cached contact-solve guesses before starting a fresh rollout."""
        self.sim_time = 0.0
        self.frame_id = 0
        self.last_contact_count = 0
        self.last_truncated_contact_count = 0
        self.last_solve_iterations = 0
        self.last_line_search_iterations = 0
        self.last_converged = True
        self._has_contact_solve_v_guess = False
        self._contact_solve_v_guess_active.zero_()
        self.last_contacts = None
        self.last_contact_jacobian_result = None
        self.last_contact_solve_result = None

    def _copy_public_joint_velocity_to_sap(self, state_in: SapState, dst: wp.array) -> None:
        model = self.model
        if state_in.joint_qd.dtype == wp.float64:
            if state_in.joint_q.dtype == wp.float64:
                wp.launch(
                    copy_public_to_sap_velocity_f64_from_joint_pose,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_child,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        self.free_motion.model_body_com,
                        state_in.joint_q,
                        state_in.joint_qd,
                        dst,
                    ],
                    device=model.device,
                )
                return
            if getattr(state_in.body_q, "dtype", None) == wp.transformd:
                wp.launch(
                    copy_public_to_sap_velocity_f64,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_child,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        state_in.body_q,
                        self.free_motion.model_body_com,
                        state_in.joint_qd,
                        dst,
                    ],
                    device=model.device,
                )
                return
            raise TypeError("Public float64 joint velocities require float64 joint_q or transformd body_q.")

        wp.launch(
            copy_public_to_sap_velocity_f32,
            dim=model.joint_count,
            inputs=[
                model.joint_type,
                model.joint_child,
                model.joint_qd_start,
                model.joint_dof_dim,
                state_in.body_q,
                model.body_com,
                state_in.joint_qd,
                dst,
            ],
            device=model.device,
        )

    def _copy_public_joint_force_to_sap(self, state_in: SapState, control: SapControl, dst: wp.array) -> None:
        model = self.model
        if control.joint_f.dtype == wp.float64:
            if state_in.joint_q.dtype == wp.float64:
                wp.launch(
                    _copy_public_to_sap_force_kernel_f64_from_joint_pose,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_child,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        self.free_motion.model_body_com,
                        state_in.joint_q,
                        control.joint_f,
                        dst,
                    ],
                    device=model.device,
                )
                return
            if getattr(state_in.body_q, "dtype", None) == wp.transformd:
                wp.launch(
                    _copy_public_to_sap_force_kernel_f64,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_child,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        state_in.body_q,
                        self.free_motion.model_body_com,
                        control.joint_f,
                        dst,
                    ],
                    device=model.device,
                )
                return
            raise TypeError("Public float64 joint forces require float64 joint_q or transformd body_q.")

        wp.launch(
            _copy_public_to_sap_force_kernel,
            dim=model.joint_count,
            inputs=[
                model.joint_type,
                model.joint_child,
                model.joint_qd_start,
                model.joint_dof_dim,
                state_in.body_q,
                model.body_com,
                control.joint_f,
                dst,
            ],
            device=model.device,
        )

    def _copy_public_body_force_to_sap(self, state_in: SapState, dst: wp.array) -> None:
        model = self.model
        body_f = getattr(state_in, "body_f", None)
        if not int(model.body_count) or body_f is None:
            dst.zero_()
            return
        if body_f.dtype != wp.spatial_vector:
            raise TypeError("Public body_f conversion requires wp.spatial_vector input.")
        if getattr(state_in.body_q, "dtype", None) == wp.transformd:
            wp.launch(
                _copy_public_body_force_to_sap_body_origin_kernel_body_qd,
                dim=model.body_count,
                inputs=[state_in.body_q, self.free_motion.model_body_com, body_f, dst],
                device=model.device,
            )
            return
        wp.launch(
            _copy_public_body_force_to_sap_body_origin_kernel,
            dim=model.body_count,
            inputs=[state_in.body_q, model.body_com, body_f, dst],
            device=model.device,
        )

    def _copy_sap_joint_velocity_to_public(
        self,
        state_out: SapState,
        sap_v: wp.array,
    ) -> None:
        model = self.model
        if state_out.joint_qd.dtype == wp.float64:
            if state_out.joint_q.dtype != wp.float64:
                raise TypeError("Public float64 output joint_qd requires float64 joint_q.")
            wp.launch(
                _copy_sap_to_public_velocity_f64_from_joint_pose,
                dim=model.joint_count,
                inputs=[
                    model.joint_type,
                    model.joint_child,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    self.free_motion.model_body_com,
                    state_out.joint_q,
                    sap_v,
                    state_out.joint_qd,
                ],
                device=model.device,
            )
            return

        wp.launch(
            _copy_sap_to_public_velocity_f32_from_joint_pose,
            dim=model.joint_count,
            inputs=[
                model.joint_type,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
                self.free_motion.model_body_com,
                state_out.joint_q,
                sap_v,
                state_out.joint_qd,
            ],
            device=model.device,
        )

    def _prepare_step_boundary(
        self,
        state_in: SapState,
        state_out: SapState,
        control: SapControl,
    ) -> tuple[SapState, SapState, SapControl, bool]:
        state_order = getattr(state_in, "joint_qd_order", "sap")
        force_order = getattr(control, "joint_f_order", "sap")
        body_force_order = getattr(state_in, "body_f_order", "sap")
        output_order = getattr(state_out, "joint_qd_order", "sap")

        solve_state = state_in
        solve_control = control
        integrate_state_out = state_out
        output_is_public = output_order == "public"

        if state_order == "public":
            self._copy_public_joint_velocity_to_sap(state_in, self._boundary_joint_qd_in_sap)
            body_f = getattr(state_in, "body_f", None)
            if body_force_order == "public":
                self._copy_public_body_force_to_sap(state_in, self._boundary_body_f_sap)
                body_f = self._boundary_body_f_sap
            solve_state = SapState(
                joint_q=state_in.joint_q,
                joint_qd=self._boundary_joint_qd_in_sap,
                body_q=state_in.body_q,
                body_qd=getattr(state_in, "body_qd", None),
                body_f=body_f,
                joint_qd_order="sap",
                body_f_order="sap",
                requires_grad=bool(getattr(state_in, "requires_grad", False)),
            )
        elif state_order != "sap":
            raise ValueError(f"Unsupported SapState joint_qd_order={state_order!r}.")

        if force_order == "public":
            self._copy_public_joint_force_to_sap(state_in, control, self._boundary_joint_f_sap)
            solve_control = SapControl(
                joint_f=self._boundary_joint_f_sap,
                joint_target_pos=control.joint_target_pos,
                joint_target_vel=control.joint_target_vel,
                joint_act=control.joint_act,
                joint_f_order="sap",
            )
        elif force_order != "sap":
            raise ValueError(f"Unsupported SapControl joint_f_order={force_order!r}.")

        if output_is_public:
            if state_out.joint_q.dtype != wp.float32 or state_out.body_q.dtype != wp.transform:
                raise TypeError("Public SolverSAP output currently requires float32 joint_q and body_q.")
            integrate_state_out = SapState(
                joint_q=state_out.joint_q,
                joint_qd=self._boundary_joint_qd_out_sap_f32,
                body_q=state_out.body_q,
                body_qd=getattr(state_out, "body_qd", None),
                body_f=None,
                joint_qd_order="sap",
                body_f_order="sap",
                requires_grad=False,
            )
        elif output_order != "sap":
            raise ValueError(f"Unsupported SapState output joint_qd_order={output_order!r}.")

        return solve_state, integrate_state_out, solve_control, output_is_public

    def integrate_particles(self, model: SapModel, state_in: SapState, state_out: SapState, dt: float) -> None:
        """Integrate particle state with semi-implicit Euler using the model gravity and timestep."""
        return

    def _integrate_state(
        self,
        state_in: SapState,
        state_out: SapState,
        solved_v_sap: wp.array,
        dt: float,
    ) -> None:
        model = self.model
        if state_out.joint_q.dtype == wp.float64:
            if self.position_integration == "midpoint":
                wp.launch(
                    _integrate_generalized_positions_midpoint_f64,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        state_in.joint_q,
                        self.free_motion.joint_qd_sap_input,
                        solved_v_sap,
                        self._joint_world,
                        self._dt_world,
                        state_out.joint_q,
                        state_out.joint_qd,
                    ],
                    device=model.device,
                )
            else:
                wp.launch(
                    _integrate_generalized_positions_sap_euler_f64,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        state_in.joint_q,
                        solved_v_sap,
                        self._joint_world,
                        self._dt_world,
                        state_out.joint_q,
                        state_out.joint_qd,
                    ],
                    device=model.device,
                )
            if state_out.joint_qd is not solved_v_sap:
                wp.copy(dest=state_out.joint_qd, src=solved_v_sap)
            if (
                int(model.body_count) > 0
                and getattr(state_in, "body_f", None) is not None
                and getattr(state_out, "body_f", None) is not None
            ):
                wp.copy(dest=state_out.body_f, src=state_in.body_f)
            self.integrate_particles(model, state_in, state_out, dt)
            return

        joint_qd_out_f32 = (
            state_out.joint_qd
            if state_out.joint_qd.dtype == wp.float32
            else self._integrate_joint_qd_f32
        )
        if self.position_integration == "midpoint":
            wp.launch(
                _integrate_generalized_positions_midpoint,
                dim=model.joint_count,
                inputs=[
                    model.joint_type,
                    model.joint_child,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    self.free_motion.model_body_com,
                    state_in.joint_q,
                    self.free_motion.joint_qd_sap_input,
                    solved_v_sap,
                    self._joint_world,
                    self._dt_world,
                    state_out.joint_q,
                    joint_qd_out_f32,
                ],
                device=model.device,
            )
        else:
            wp.launch(
                _integrate_generalized_positions_sap_euler,
                dim=model.joint_count,
                inputs=[
                    model.joint_type,
                    model.joint_child,
                    model.joint_q_start,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    self.free_motion.model_body_com,
                    state_in.joint_q,
                    solved_v_sap,
                    self._joint_world,
                    self._dt_world,
                    state_out.joint_q,
                    joint_qd_out_f32,
                ],
                device=model.device,
            )

        if state_out.joint_qd is not solved_v_sap:
            if state_out.joint_qd.dtype == wp.float64:
                wp.copy(dest=state_out.joint_qd, src=solved_v_sap)
            else:
                wp.launch(
                    _copy_f64_to_f32,
                    dim=model.joint_dof_count,
                    inputs=[solved_v_sap, state_out.joint_qd],
                    device=model.device,
                )
        if (
            int(model.body_count) > 0
            and getattr(state_in, "body_f", None) is not None
            and getattr(state_out, "body_f", None) is not None
        ):
            wp.copy(dest=state_out.body_f, src=state_in.body_f)

    def step(
        self,
        state_in: SapState,
        state_out: SapState,
        control: SapControl | None,
        contacts: SapContacts | None,
        dt: float,
    ) -> SapState:
        """Advance one SAP timestep from state_in to state_out using control inputs, active contacts, and
        the configured solver settings.
        """
        if getattr(state_in, "requires_grad", False):
            raise NotImplementedError("SolverSAP does not support grad states.")
        if not isinstance(state_in, SapState):
            raise TypeError("SolverSAP.step requires SapState input.")
        if not isinstance(state_out, SapState):
            raise TypeError("SolverSAP.step requires SapState output.")
        if control is None:
            control = self._zero_control
        if not isinstance(control, SapControl):
            raise TypeError("SolverSAP.step requires SapControl.")
        if contacts is None:
            contacts = SapContacts()
        if not isinstance(contacts, SapContacts):
            contacts = sap_contacts_from_newton(contacts)

        # Per-world timestep source for the integration kernels.  Contact
        # jacobian/solve still take the scalar dt during the per-world
        # migration; under uniform dt the two are consistent.
        self._update_dt_world(dt)
        self.last_contacts = contacts
        solve_state, integrate_state_out, solve_control, output_is_public = self._prepare_step_boundary(
            state_in,
            state_out,
            control,
        )

        contact_result = self.contact_jacobian.compute(
            solve_state,
            contacts,
            control=solve_control,
            dt=dt,
        )

        # Static substep runs a FIXED (small) iteration count that is unrolled into the captured
        # graph; the conditional path keeps the larger early-exit cap. See __init__ rationale.
        if self.static_substep:
            eff_max_iterations = self.static_substep_iterations
            eff_line_search_max_iterations = self.static_substep_line_search_iterations
        else:
            eff_max_iterations = self.max_iterations
            eff_line_search_max_iterations = self.line_search_max_iterations

        v0 = self.free_motion.joint_qd_sap_input
        solve_result = self.contact_solve.solve(
            contact_result,
            solve_state,
            solve_control,
            dt,
            self.free_motion.free_motion_joint_qd_sap,
            v0=v0,
            v_guess=self.contact_solve.v_flat,
            v_guess_active=self._contact_solve_v_guess_active,
            max_iterations=eff_max_iterations,
            optimality_abs_tol=self.optimality_abs_tol,
            optimality_rel_tol=self.optimality_rel_tol,
            cost_abs_tol=self.cost_abs_tol,
            cost_rel_tol=self.cost_rel_tol,
            line_search_max_iterations=eff_line_search_max_iterations,
            armijo_c=self.armijo_c,
            rho=self.rho,
            line_search_relative_slop=self.line_search_relative_slop,
            line_search_variant=self.line_search_variant,
            collect_iteration_stats=self.collect_iteration_stats,
            check_line_search_errors=self.check_line_search_errors,
            graph_conditional=self.graph_conditional,
            static_loop=self.static_substep,
        )

        v_integrate = solve_result.v_flat
        if v_integrate.dtype == wp.float32:
            wp.launch(
                _copy_f32_to_f64,
                dim=self.dof_count,
                inputs=[v_integrate, self._integrate_v_f64],
                device=self.sap_model.device,
            )
            v_integrate = self._integrate_v_f64
        self._integrate_state(solve_state, integrate_state_out, v_integrate, dt)
        if output_is_public:
            self._copy_sap_joint_velocity_to_public(state_out, v_integrate)
            sap_eval_fk(self.model, state_out.joint_q, state_out.joint_qd, state_out)

        self.last_contact_jacobian_result = contact_result
        self.last_contact_solve_result = solve_result
        self.last_contact_count = int(contact_result.contact_count)
        self.last_truncated_contact_count = int(contact_result.truncated_contact_count)
        self.last_solve_iterations = int(solve_result.iterations)
        self.last_line_search_iterations = int(solve_result.line_search_iterations)
        self.last_converged = bool(solve_result.converged)
        self._has_contact_solve_v_guess = True
        # Per-world (array) dt advances each world's own clock in the adaptive
        # driver; this inner scalar bookkeeping only applies to fixed/scalar dt.
        if not isinstance(dt, wp.array):
            self.sim_time += float(dt)
        self.frame_id += 1
        return state_out

    def notify_model_changed(self, flags: int) -> None:
        # The standalone components cache model-topology-dependent buffers.  If
        # topology changes, construct a new solver; scalar model data is read
        # directly from model arrays during each step.
        """Refresh solver caches and work buffers after the underlying model arrays or topology have
        changed.
        """
        return

    def update_contacts(self, contacts: SapContacts) -> None:
        """Update the solver contact buffer reference and resize dependent contact-stage data when needed."""
        raise NotImplementedError("SolverSAP does not expose contact-force writeback yet.")


__all__ = [
    "SolverSAP",
]
