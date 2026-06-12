"""Kinematics kernels for SAP runtime model data.

Source note: the SAP modifications in this module are based on Newton's
kinematics/runtime code and adapted for compatibility with Newton-owned Warp
arrays.
"""

from __future__ import annotations

import warp as wp

from sim.sap_runtime import (
    SAP_BODY_FLAG_ALL,
    SAP_JOINT_BALL,
    SAP_JOINT_D6,
    SAP_JOINT_DISTANCE,
    SAP_JOINT_FREE,
    SAP_JOINT_PRISMATIC,
    SAP_JOINT_REVOLUTE,
    SapModel,
    SapState,
)


@wp.func
def sap_compute_2d_rotational_dofs(
    axis_0: wp.vec3,
    axis_1: wp.vec3,
    q0: float,
    q1: float,
    qd0: float,
    qd1: float,
):
    q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, wp.cross(axis_0, axis_1)))

    local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
    local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))

    axis_0 = local_0
    q_0 = wp.quat_from_axis_angle(axis_0, q0)

    axis_1 = wp.quat_rotate(q_0, local_1)
    q_1 = wp.quat_from_axis_angle(axis_1, q1)

    rot = q_1 * q_0
    vel = axis_0 * qd0 + axis_1 * qd1

    return rot, vel


@wp.func
def sap_compute_3d_rotational_dofs(
    axis_0: wp.vec3,
    axis_1: wp.vec3,
    axis_2: wp.vec3,
    q0: float,
    q1: float,
    q2: float,
    qd0: float,
    qd1: float,
    qd2: float,
):
    q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, axis_2))

    local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
    local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))
    local_2 = wp.quat_rotate(q_off, wp.vec3(0.0, 0.0, 1.0))

    axis_0 = local_0
    q_0 = wp.quat_from_axis_angle(axis_0, q0)

    axis_1 = wp.quat_rotate(q_0, local_1)
    q_1 = wp.quat_from_axis_angle(axis_1, q1)

    axis_2 = wp.quat_rotate(q_1 * q_0, local_2)
    q_2 = wp.quat_from_axis_angle(axis_2, q2)

    rot = q_2 * q_1 * q_0
    vel = axis_0 * qd0 + axis_1 * qd1 + axis_2 * qd2

    return rot, vel


@wp.func
def sap_eval_single_articulation_fk(
    joint_start: int,
    joint_end: int,
    joint_articulation: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3),
    body_flags: wp.array(dtype=wp.int32),
    body_flag_filter: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    for i in range(joint_start, joint_end):
        articulation = joint_articulation[i]
        if articulation == -1:
            continue

        parent = joint_parent[i]
        child = joint_child[i]
        jtype = joint_type[i]

        X_pj = joint_X_p[i]
        X_cj = joint_X_c[i]

        X_wpj = X_pj
        v_wpj = wp.spatial_vector()
        if parent >= 0:
            X_wp = body_q[parent]
            X_wpj = X_wp * X_wpj
            r_p = wp.transform_get_translation(X_wpj) - wp.transform_point(X_wp, body_com[parent])

            v_wp = body_qd[parent]
            w_p = wp.spatial_bottom(v_wp)
            v_p = wp.spatial_top(v_wp) + wp.cross(w_p, r_p)
            v_wpj = wp.spatial_vector(v_p, w_p)

        q_start = joint_q_start[i]
        qd_start = joint_qd_start[i]
        lin_axis_count = joint_dof_dim[i, 0]
        ang_axis_count = joint_dof_dim[i, 1]

        X_j = wp.transform_identity()
        v_j = wp.spatial_vector(wp.vec3(), wp.vec3())

        if jtype == SAP_JOINT_PRISMATIC:
            axis = joint_axis[qd_start]
            q = joint_q[q_start]
            qd = joint_qd[qd_start]
            X_j = wp.transform(axis * q, wp.quat_identity())
            v_j = wp.spatial_vector(axis * qd, wp.vec3())

        if jtype == SAP_JOINT_REVOLUTE:
            axis = joint_axis[qd_start]
            q = joint_q[q_start]
            qd = joint_qd[qd_start]
            X_j = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
            v_j = wp.spatial_vector(wp.vec3(), axis * qd)

        if jtype == SAP_JOINT_BALL:
            r = wp.quat(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2], joint_q[q_start + 3])
            w = wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2])
            X_j = wp.transform(wp.vec3(), r)
            v_j = wp.spatial_vector(wp.vec3(), w)

        if jtype == SAP_JOINT_FREE or jtype == SAP_JOINT_DISTANCE:
            t = wp.transform(
                wp.vec3(joint_q[q_start + 0], joint_q[q_start + 1], joint_q[q_start + 2]),
                wp.quat(joint_q[q_start + 3], joint_q[q_start + 4], joint_q[q_start + 5], joint_q[q_start + 6]),
            )
            v = wp.spatial_vector(
                wp.vec3(joint_qd[qd_start + 0], joint_qd[qd_start + 1], joint_qd[qd_start + 2]),
                wp.vec3(joint_qd[qd_start + 3], joint_qd[qd_start + 4], joint_qd[qd_start + 5]),
            )
            X_j = t
            v_j = v

        if jtype == SAP_JOINT_D6:
            pos = wp.vec3(0.0)
            rot = wp.quat_identity()
            vel_v = wp.vec3(0.0)
            vel_w = wp.vec3(0.0)

            if lin_axis_count > 0:
                axis = joint_axis[qd_start + 0]
                pos += axis * joint_q[q_start + 0]
                vel_v += axis * joint_qd[qd_start + 0]
            if lin_axis_count > 1:
                axis = joint_axis[qd_start + 1]
                pos += axis * joint_q[q_start + 1]
                vel_v += axis * joint_qd[qd_start + 1]
            if lin_axis_count > 2:
                axis = joint_axis[qd_start + 2]
                pos += axis * joint_q[q_start + 2]
                vel_v += axis * joint_qd[qd_start + 2]

            iq = q_start + lin_axis_count
            iqd = qd_start + lin_axis_count
            if ang_axis_count == 1:
                axis = joint_axis[iqd]
                rot = wp.quat_from_axis_angle(axis, joint_q[iq])
                vel_w = joint_qd[iqd] * axis
            if ang_axis_count == 2:
                rot, vel_w = sap_compute_2d_rotational_dofs(
                    joint_axis[iqd + 0],
                    joint_axis[iqd + 1],
                    joint_q[iq + 0],
                    joint_q[iq + 1],
                    joint_qd[iqd + 0],
                    joint_qd[iqd + 1],
                )
            if ang_axis_count == 3:
                rot, vel_w = sap_compute_3d_rotational_dofs(
                    joint_axis[iqd + 0],
                    joint_axis[iqd + 1],
                    joint_axis[iqd + 2],
                    joint_q[iq + 0],
                    joint_q[iq + 1],
                    joint_q[iq + 2],
                    joint_qd[iqd + 0],
                    joint_qd[iqd + 1],
                    joint_qd[iqd + 2],
                )

            X_j = wp.transform(pos, rot)
            v_j = wp.spatial_vector(vel_v, vel_w)

        X_wcj = X_wpj * X_j
        X_wc = X_wcj * wp.transform_inverse(X_cj)

        linear_vel = wp.transform_vector(X_wpj, wp.spatial_top(v_j))
        angular_vel = wp.transform_vector(X_wpj, wp.spatial_bottom(v_j))
        v_wc = v_wpj + wp.spatial_vector(linear_vel, angular_vel)

        if (body_flags[child] & body_flag_filter) != 0:
            body_q[child] = X_wc
            body_qd[child] = v_wc


@wp.kernel
def sap_eval_articulation_fk(
    articulation_start: wp.array(dtype=int),
    articulation_count: int,
    articulation_mask: wp.array(dtype=bool),
    articulation_indices: wp.array(dtype=int),
    joint_articulation: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_com: wp.array(dtype=wp.vec3),
    body_flags: wp.array(dtype=wp.int32),
    body_flag_filter: int,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    if articulation_indices:
        articulation_id = articulation_indices[tid]
    else:
        articulation_id = tid

    if articulation_id < 0 or articulation_id >= articulation_count:
        return

    if articulation_mask:
        if not articulation_mask[articulation_id]:
            return

    joint_start = articulation_start[articulation_id]
    joint_end = articulation_start[articulation_id + 1]

    sap_eval_single_articulation_fk(
        joint_start,
        joint_end,
        joint_articulation,
        joint_q,
        joint_qd,
        joint_q_start,
        joint_qd_start,
        joint_type,
        joint_parent,
        joint_child,
        joint_X_p,
        joint_X_c,
        joint_axis,
        joint_dof_dim,
        body_com,
        body_flags,
        body_flag_filter,
        body_q,
        body_qd,
    )


def sap_eval_fk(
    model: SapModel,
    joint_q: wp.array,
    joint_qd: wp.array,
    state: SapState,
    mask: wp.array | None = None,
    indices: wp.array | None = None,
    body_flag_filter: int = int(SAP_BODY_FLAG_ALL),
) -> None:
    if mask is not None and indices is not None:
        raise ValueError("Cannot specify both mask and indices.")
    if state.body_q is None or state.body_qd is None:
        raise ValueError("sap_eval_fk requires state.body_q and state.body_qd.")
    if joint_qd.dtype != wp.float32 or joint_q.dtype != wp.float32:
        raise TypeError("sap_eval_fk currently requires float32 joint_q and joint_qd arrays.")
    if state.body_q.dtype != wp.transform or state.body_qd.dtype != wp.spatial_vector:
        raise TypeError("sap_eval_fk currently requires float32 body transform and spatial-vector arrays.")

    dim = len(indices) if indices is not None else int(model.articulation_count)
    wp.launch(
        kernel=sap_eval_articulation_fk,
        dim=dim,
        inputs=[
            model.articulation_start,
            int(model.articulation_count),
            mask,
            indices,
            model.joint_articulation,
            joint_q,
            joint_qd,
            model.joint_q_start,
            model.joint_qd_start,
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_X_p,
            model.joint_X_c,
            model.joint_axis,
            model.joint_dof_dim,
            model.body_com,
            model.body_flags,
            int(body_flag_filter),
        ],
        outputs=[state.body_q, state.body_qd],
        device=model.device,
    )
