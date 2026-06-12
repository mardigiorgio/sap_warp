from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np
import warp as wp

wp.config.enable_backward = False

from sim.blocked_cholesky import BlockCholeskySolverBatched
from sim.sap_helpers import (
    _copy_f32_to_f64,
    _copy_f64,
    _dense_cholesky_f64,
    _dense_subs_f64,
    _sap_calc_across_mobilizer_transform,
    _sap_calc_mobilizer_velocity_and_bias_body_world,
    _sap_compose_acceleration,
    _sap_compose_acceleration_f32,
    _sap_compose_child_body_transform,
    _sap_compose_velocity,
    _sap_compose_velocity_f32,
    _sap_dynamic_bias_force_body_origin_world,
    _sap_gravity_force_body_origin_world,
    _sap_revolute_identity_child_kinematics,
    _sap_revolute_identity_child_kinematics_f32_core,
    _sap_shift_force,
    _sap_shift_velocity,
    _sap_shift_velocity_f32,
    _sap_spatial,
    _sap_spatial_inertia_body_origin_world,
    _project_tau_no_drives,
    _spatiald_from_spatialf,
    _spatialf_from_spatiald,
    _transformd_compose,
    _transformd_from_transformf,
    _transformd_identity,
    _transformf_compose,
    _transformf_from_transformd,
    _vec3d_zero,
    _vec3f_from_vec3d,
)
from sim.sap_runtime import (
    Control,
    Model,
    SAP_JOINT_BALL,
    SAP_JOINT_D6,
    SAP_JOINT_DISTANCE,
    SAP_JOINT_FIXED,
    SAP_JOINT_FREE,
    SAP_JOINT_PRISMATIC,
    SAP_JOINT_REVOLUTE,
    State,
)

_GEMM_COL_BLOCK = wp.constant(4)
_JTP_GEMM_TILE_M = 8
_JTP_GEMM_TILE_N = 8
_JTP_GEMM_TILE_K = 32


@dataclass(frozen=True)
class SapFreeMotionResult:
    """Views into the mutable output buffers owned by `SapFreeMotion`."""

    v_star: wp.array
    vdot: wp.array
    dynamics_matrix: wp.array | None

@wp.kernel
def _assemble_sap_free_motion_outputs_kernel(
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    sap_v0: wp.array(dtype=wp.float64),
    sap_vdot_solve: wp.array(dtype=wp.float64),
    dt: wp.float64,
    sap_v_star: wp.array(dtype=wp.float64),
    sap_vdot: wp.array(dtype=wp.float64),
):
    joint = wp.tid()
    dof_start = joint_qd_start[joint]
    axis_count = joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1]

    for axis in range(axis_count):
        vdot = sap_vdot_solve[dof_start + axis]
        sap_vdot[dof_start + axis] = vdot
        sap_v_star[dof_start + axis] = sap_v0[dof_start + axis] + dt * vdot


@wp.kernel
def _copy_spatial_vector_to_spatial_vectord(
    src: wp.array(dtype=wp.spatial_vector),
    dst: wp.array(dtype=wp.spatial_vectord),
):
    i = wp.tid()
    dst[i] = _spatiald_from_spatialf(src[i])


@wp.kernel
def _eval_rigid_tau_no_drives(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_f: wp.array(dtype=wp.float64),
    joint_S_s: wp.array(dtype=wp.spatial_vectord),
    body_q: wp.array(dtype=wp.transformd),
    body_fb_s: wp.array(dtype=wp.spatial_vectord),
    body_f_ext: wp.array(dtype=wp.spatial_vectord),
    body_ft_s: wp.array(dtype=wp.spatial_vectord),
    tau: wp.array(dtype=wp.float64),
):
    art = wp.tid()
    start = articulation_start[art]
    end = articulation_start[art + 1]
    count = end - start

    for offset in range(count):
        joint = end - offset - 1
        jtype = joint_type[joint]
        parent = joint_parent[joint]
        child = joint_child[joint]
        dof_start = joint_qd_start[joint]
        lin_axis_count = joint_dof_dim[joint, 0]
        ang_axis_count = joint_dof_dim[joint, 1]

        f_b_s = body_fb_s[child]
        f_t_s = body_ft_s[child]
        f_ext = body_f_ext[child]
        f_s = f_b_s + f_t_s + f_ext

        _project_tau_no_drives(
            jtype,
            joint_S_s,
            joint_f,
            dof_start,
            lin_axis_count,
            ang_axis_count,
            f_s,
            tau,
        )

        if parent >= 0:
            p_child = wp.transform_get_translation(body_q[child])
            p_parent = wp.transform_get_translation(body_q[parent])
            wp.atomic_add(body_ft_s, parent, _sap_shift_force(f_s, p_parent - p_child))


@cache
def _make_eval_rigid_tau_no_drives_tiled(tile_size: int):
    @wp.kernel(enable_backward=False)
    def _eval_rigid_tau_no_drives_tiled(
        articulation_level_joint_index: wp.array(dtype=wp.int32),
        max_articulation_level_count: int,
        max_articulation_level_width: int,
        joint_type: wp.array(dtype=int),
        joint_parent: wp.array(dtype=int),
        joint_child: wp.array(dtype=int),
        joint_qd_start: wp.array(dtype=int),
        joint_dof_dim: wp.array(dtype=int, ndim=2),
        joint_f: wp.array(dtype=wp.float64),
        joint_S_s: wp.array(dtype=wp.spatial_vectord),
        body_q: wp.array(dtype=wp.transformd),
        body_fb_s: wp.array(dtype=wp.spatial_vectord),
        body_f_ext: wp.array(dtype=wp.spatial_vectord),
        body_ft_s: wp.array(dtype=wp.spatial_vectord),
        tau: wp.array(dtype=wp.float64),
    ):
        art, tid = wp.tid()
        for reverse_level in range(max_articulation_level_count):
            level_index = max_articulation_level_count - reverse_level - 1
            stride = wp.block_dim()
            slot = tid
            while slot < max_articulation_level_width:
                flat_index = (
                    (art * max_articulation_level_count + level_index) * max_articulation_level_width + slot
                )
                joint = articulation_level_joint_index[flat_index]
                if joint >= 0:
                    jtype = joint_type[joint]
                    parent = joint_parent[joint]
                    child = joint_child[joint]
                    dof_start = joint_qd_start[joint]
                    lin_axis_count = joint_dof_dim[joint, 0]
                    ang_axis_count = joint_dof_dim[joint, 1]

                    f_b_s = body_fb_s[child]
                    f_t_s = body_ft_s[child]
                    f_ext = body_f_ext[child]
                    f_s = f_b_s + f_t_s + f_ext

                    _project_tau_no_drives(
                        jtype,
                        joint_S_s,
                        joint_f,
                        dof_start,
                        lin_axis_count,
                        ang_axis_count,
                        f_s,
                        tau,
                    )

                    if parent >= 0:
                        p_child = wp.transform_get_translation(body_q[child])
                        p_parent = wp.transform_get_translation(body_q[parent])
                        wp.atomic_add(body_ft_s, parent, _sap_shift_force(f_s, p_parent - p_child))

                slot = slot + stride

            if tile_size > 1:
                sync_values = wp.tile_zeros((tile_size,), dtype=wp.int32, storage="shared")
                sync_values[tid] = wp.int32(0)
                _ = wp.tile_sum(sync_values)

    return _eval_rigid_tau_no_drives_tiled


@wp.kernel
def _eval_rigid_id_root_level_kernel(
    articulation_level_joint_index: wp.array(dtype=wp.int32),
    max_articulation_level_count: int,
    max_articulation_level_width: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=wp.float64),
    joint_qd: wp.array(dtype=wp.float64),
    joint_axis: wp.array(dtype=wp.vec3d),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_inertia: wp.array(dtype=wp.mat33d),
    body_mass: wp.array(dtype=wp.float64),
    body_com: wp.array(dtype=wp.vec3d),
    body_q: wp.array(dtype=wp.transformd),
    joint_X_p: wp.array(dtype=wp.transformd),
    joint_X_c: wp.array(dtype=wp.transformd),
    joint_X_c_identity: wp.array(dtype=wp.int32),
    body_world: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3d),
    joint_S_s: wp.array(dtype=wp.spatial_vectord),
    body_I_s: wp.array(dtype=wp.spatial_matrixd),
    body_v_s: wp.array(dtype=wp.spatial_vectord),
    body_f_s: wp.array(dtype=wp.spatial_vectord),
    body_a_s: wp.array(dtype=wp.spatial_vectord),
):
    art = wp.tid()
    flat_index = art * max_articulation_level_count * max_articulation_level_width
    joint = articulation_level_joint_index[flat_index]
    if joint < 0:
        return

    child = joint_child[joint]
    jtype = joint_type[joint]
    q_start = joint_q_start[joint]
    qd_start = joint_qd_start[joint]
    lin_axis_count = joint_dof_dim[joint, 0]
    ang_axis_count = joint_dof_dim[joint, 1]
    X_wpj = joint_X_p[joint]
    X_wc = _transformd_identity()
    v_PB_W = _sap_spatial(_vec3d_zero(), _vec3d_zero())
    A_PB_W = _sap_spatial(_vec3d_zero(), _vec3d_zero())
    if jtype == SAP_JOINT_REVOLUTE and joint_X_c_identity[joint] != 0:
        X_wc, v_PB_W, A_PB_W = _sap_revolute_identity_child_kinematics(
            joint_axis,
            X_wpj,
            joint_q,
            q_start,
            joint_qd,
            qd_start,
            joint_S_s,
        )
    else:
        X_j = _sap_calc_across_mobilizer_transform(
            jtype,
            joint_axis,
            qd_start,
            lin_axis_count,
            ang_axis_count,
            joint_q,
            q_start,
        )
        X_wc = _sap_compose_child_body_transform(X_wpj, X_j, joint_X_c[joint], joint_X_c_identity[joint])
        v_PB_W, A_PB_W = _sap_calc_mobilizer_velocity_and_bias_body_world(
            jtype,
            joint_axis,
            lin_axis_count,
            ang_axis_count,
            X_wpj,
            X_j,
            joint_X_c[joint],
            joint_X_c_identity[joint],
            joint_q,
            q_start,
            joint_qd,
            qd_start,
            joint_S_s,
        )
    body_q[child] = X_wc

    v_s = v_PB_W
    a_s = A_PB_W

    body_v_s[child] = v_s
    body_a_s[child] = a_s


@wp.kernel
def _eval_rigid_id_level_parallel_1d_nonroot(
    articulation_level_joint_index: wp.array(dtype=wp.int32),
    max_articulation_level_count: int,
    max_articulation_level_width: int,
    level_index: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=wp.float64),
    joint_qd: wp.array(dtype=wp.float64),
    joint_axis: wp.array(dtype=wp.vec3d),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_inertia: wp.array(dtype=wp.mat33d),
    body_mass: wp.array(dtype=wp.float64),
    body_com: wp.array(dtype=wp.vec3d),
    body_q: wp.array(dtype=wp.transformd),
    joint_X_p: wp.array(dtype=wp.transformd),
    joint_X_c: wp.array(dtype=wp.transformd),
    joint_X_c_identity: wp.array(dtype=wp.int32),
    body_world: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3d),
    joint_S_s: wp.array(dtype=wp.spatial_vectord),
    body_I_s: wp.array(dtype=wp.spatial_matrixd),
    body_v_s: wp.array(dtype=wp.spatial_vectord),
    body_f_s: wp.array(dtype=wp.spatial_vectord),
    body_a_s: wp.array(dtype=wp.spatial_vectord),
):
    art, slot = wp.tid()
    flat_index = (
        (art * max_articulation_level_count + level_index) * max_articulation_level_width + slot
    )
    joint = articulation_level_joint_index[flat_index]
    if joint < 0:
        return

    parent = joint_parent[joint]
    child = joint_child[joint]
    q_start = joint_q_start[joint]
    qd_start = joint_qd_start[joint]
    jtype = joint_type[joint]
    lin_axis_count = joint_dof_dim[joint, 0]
    ang_axis_count = joint_dof_dim[joint, 1]
    X_wpj = _transformd_compose(body_q[parent], joint_X_p[joint])
    X_wc = _transformd_identity()
    v_PB_W = _sap_spatial(_vec3d_zero(), _vec3d_zero())
    A_PB_W = _sap_spatial(_vec3d_zero(), _vec3d_zero())
    if jtype == SAP_JOINT_REVOLUTE and joint_X_c_identity[joint] != 0:
        X_wc, v_PB_W, A_PB_W = _sap_revolute_identity_child_kinematics(
            joint_axis,
            X_wpj,
            joint_q,
            q_start,
            joint_qd,
            qd_start,
            joint_S_s,
        )
    else:
        X_j = _sap_calc_across_mobilizer_transform(
            jtype,
            joint_axis,
            qd_start,
            lin_axis_count,
            ang_axis_count,
            joint_q,
            q_start,
        )
        X_wc = _sap_compose_child_body_transform(X_wpj, X_j, joint_X_c[joint], joint_X_c_identity[joint])
        v_PB_W, A_PB_W = _sap_calc_mobilizer_velocity_and_bias_body_world(
            jtype,
            joint_axis,
            lin_axis_count,
            ang_axis_count,
            X_wpj,
            X_j,
            joint_X_c[joint],
            joint_X_c_identity[joint],
            joint_q,
            q_start,
            joint_qd,
            qd_start,
            joint_S_s,
        )
    body_q[child] = X_wc

    v_parent_s = body_v_s[parent]
    a_parent_s = body_a_s[parent]
    p_parent = wp.transform_get_translation(body_q[parent])
    p_child = wp.transform_get_translation(X_wc)
    p_PB_W = p_child - p_parent
    v_s = _sap_compose_velocity(v_parent_s, p_PB_W, v_PB_W)
    a_s = _sap_compose_acceleration(a_parent_s, v_parent_s, p_PB_W, v_PB_W, A_PB_W)

    body_v_s[child] = v_s
    body_a_s[child] = a_s


@cache
def _make_eval_rigid_id_tiled_articulations(tile_size: int):
    @wp.kernel(enable_backward=False)
    def _eval_rigid_id_tiled_articulations(
        articulation_level_joint_index: wp.array(dtype=wp.int32),
        max_articulation_level_count: int,
        max_articulation_level_width: int,
        joint_type: wp.array(dtype=int),
        joint_parent: wp.array(dtype=int),
        joint_child: wp.array(dtype=int),
        joint_q_start: wp.array(dtype=int),
        joint_qd_start: wp.array(dtype=int),
        joint_q: wp.array(dtype=wp.float64),
        joint_qd: wp.array(dtype=wp.float64),
        joint_axis: wp.array(dtype=wp.vec3d),
        joint_dof_dim: wp.array(dtype=int, ndim=2),
        body_inertia: wp.array(dtype=wp.mat33d),
        body_mass: wp.array(dtype=wp.float64),
        body_com: wp.array(dtype=wp.vec3d),
        body_q: wp.array(dtype=wp.transformd),
        joint_X_p: wp.array(dtype=wp.transformd),
        joint_X_c: wp.array(dtype=wp.transformd),
        joint_X_c_identity: wp.array(dtype=wp.int32),
        body_world: wp.array(dtype=wp.int32),
        gravity: wp.array(dtype=wp.vec3d),
        joint_S_s: wp.array(dtype=wp.spatial_vectord),
        body_I_s: wp.array(dtype=wp.spatial_matrixd),
        body_v_s: wp.array(dtype=wp.spatial_vectord),
        body_f_s: wp.array(dtype=wp.spatial_vectord),
        body_a_s: wp.array(dtype=wp.spatial_vectord),
    ):
        art, tid = wp.tid()
        for level_index in range(max_articulation_level_count):
            stride = wp.block_dim()
            slot = tid
            while slot < max_articulation_level_width:
                flat_index = (
                    (art * max_articulation_level_count + level_index) * max_articulation_level_width + slot
                )
                joint = articulation_level_joint_index[flat_index]
                if joint >= 0:
                    parent = joint_parent[joint]
                    child = joint_child[joint]
                    q_start = joint_q_start[joint]
                    qd_start = joint_qd_start[joint]
                    jtype = joint_type[joint]
                    lin_axis_count = joint_dof_dim[joint, 0]
                    ang_axis_count = joint_dof_dim[joint, 1]

                    X_wpj = joint_X_p[joint]
                    if parent >= 0:
                        X_wpj = _transformd_compose(body_q[parent], joint_X_p[joint])
                    X_wc = _transformd_identity()
                    v_PB_W = _sap_spatial(_vec3d_zero(), _vec3d_zero())
                    A_PB_W = _sap_spatial(_vec3d_zero(), _vec3d_zero())
                    if jtype == SAP_JOINT_REVOLUTE and joint_X_c_identity[joint] != 0:
                        X_wc, v_PB_W, A_PB_W = _sap_revolute_identity_child_kinematics(
                            joint_axis,
                            X_wpj,
                            joint_q,
                            q_start,
                            joint_qd,
                            qd_start,
                            joint_S_s,
                        )
                    else:
                        X_j = _sap_calc_across_mobilizer_transform(
                            jtype,
                            joint_axis,
                            qd_start,
                            lin_axis_count,
                            ang_axis_count,
                            joint_q,
                            q_start,
                        )
                        X_wc = _sap_compose_child_body_transform(
                            X_wpj,
                            X_j,
                            joint_X_c[joint],
                            joint_X_c_identity[joint],
                        )
                        v_PB_W, A_PB_W = _sap_calc_mobilizer_velocity_and_bias_body_world(
                            jtype,
                            joint_axis,
                            lin_axis_count,
                            ang_axis_count,
                            X_wpj,
                            X_j,
                            joint_X_c[joint],
                            joint_X_c_identity[joint],
                            joint_q,
                            q_start,
                            joint_qd,
                            qd_start,
                            joint_S_s,
                        )
                    body_q[child] = X_wc

                    v_s = v_PB_W
                    a_s = A_PB_W
                    if parent >= 0:
                        v_parent_s = body_v_s[parent]
                        a_parent_s = body_a_s[parent]
                        p_parent = wp.transform_get_translation(body_q[parent])
                        p_child = wp.transform_get_translation(X_wc)
                        p_PB_W = p_child - p_parent
                        v_s = _sap_compose_velocity(v_parent_s, p_PB_W, v_PB_W)
                        a_s = _sap_compose_acceleration(a_parent_s, v_parent_s, p_PB_W, v_PB_W, A_PB_W)

                    body_v_s[child] = v_s
                    body_a_s[child] = a_s

                slot = slot + stride

            sync_values = wp.tile_zeros((tile_size,), dtype=wp.int32, storage="shared")
            sync_values[tid] = wp.int32(0)
            _ = wp.tile_sum(sync_values)

    return _eval_rigid_id_tiled_articulations


@cache
def _make_eval_rigid_id_tiled_articulations_f32_revolute(tile_size: int):
    @wp.kernel(enable_backward=False)
    def _eval_rigid_id_tiled_articulations_f32_revolute(
        articulation_level_joint_index: wp.array(dtype=wp.int32),
        max_articulation_level_count: int,
        max_articulation_level_width: int,
        joint_type: wp.array(dtype=int),
        joint_parent: wp.array(dtype=int),
        joint_child: wp.array(dtype=int),
        joint_q_start: wp.array(dtype=int),
        joint_qd_start: wp.array(dtype=int),
        joint_q: wp.array(dtype=wp.float64),
        joint_qd: wp.array(dtype=wp.float64),
        joint_axis: wp.array(dtype=wp.vec3d),
        joint_dof_dim: wp.array(dtype=int, ndim=2),
        body_inertia: wp.array(dtype=wp.mat33d),
        body_mass: wp.array(dtype=wp.float64),
        body_com: wp.array(dtype=wp.vec3d),
        body_q: wp.array(dtype=wp.transformd),
        joint_X_p: wp.array(dtype=wp.transformd),
        joint_X_c: wp.array(dtype=wp.transformd),
        joint_X_c_identity: wp.array(dtype=wp.int32),
        body_world: wp.array(dtype=wp.int32),
        gravity: wp.array(dtype=wp.vec3d),
        joint_S_s: wp.array(dtype=wp.spatial_vectord),
        body_I_s: wp.array(dtype=wp.spatial_matrixd),
        body_v_s: wp.array(dtype=wp.spatial_vectord),
        body_f_s: wp.array(dtype=wp.spatial_vectord),
        body_a_s: wp.array(dtype=wp.spatial_vectord),
        body_q_f: wp.array(dtype=wp.transform),
        body_v_s_f: wp.array(dtype=wp.spatial_vector),
        body_a_s_f: wp.array(dtype=wp.spatial_vector),
    ):
        art, tid = wp.tid()
        for level_index in range(max_articulation_level_count):
            stride = wp.block_dim()
            slot = tid
            while slot < max_articulation_level_width:
                flat_index = (
                    (art * max_articulation_level_count + level_index) * max_articulation_level_width + slot
                )
                joint = articulation_level_joint_index[flat_index]
                if joint >= 0:
                    parent = joint_parent[joint]
                    child = joint_child[joint]
                    q_start = joint_q_start[joint]
                    qd_start = joint_qd_start[joint]
                    jtype = joint_type[joint]
                    lin_axis_count = joint_dof_dim[joint, 0]
                    ang_axis_count = joint_dof_dim[joint, 1]

                    if jtype == SAP_JOINT_REVOLUTE and joint_X_c_identity[joint] != 0:
                        X_wpj_f = _transformf_from_transformd(joint_X_p[joint])
                        if parent >= 0:
                            X_wpj_f = _transformf_compose(body_q_f[parent], X_wpj_f)
                        X_wc_f, v_PB_f, A_PB_f = _sap_revolute_identity_child_kinematics_f32_core(
                            joint_axis,
                            X_wpj_f,
                            joint_q,
                            q_start,
                            joint_qd,
                            qd_start,
                            joint_S_s,
                        )
                        v_s_f = v_PB_f
                        a_s_f = A_PB_f
                        if parent >= 0:
                            v_parent_f = body_v_s_f[parent]
                            a_parent_f = body_a_s_f[parent]
                            p_parent_f = wp.transform_get_translation(body_q_f[parent])
                            p_child_f = wp.transform_get_translation(X_wc_f)
                            p_PB_W_f = p_child_f - p_parent_f
                            v_s_f = _sap_compose_velocity_f32(v_parent_f, p_PB_W_f, v_PB_f)
                            a_s_f = _sap_compose_acceleration_f32(a_parent_f, v_parent_f, p_PB_W_f, v_PB_f, A_PB_f)

                        body_q_f[child] = X_wc_f
                        body_v_s_f[child] = v_s_f
                        body_a_s_f[child] = a_s_f
                        body_q[child] = _transformd_from_transformf(X_wc_f)
                        body_v_s[child] = _spatiald_from_spatialf(v_s_f)
                        body_a_s[child] = _spatiald_from_spatialf(a_s_f)
                    else:
                        X_wpj = joint_X_p[joint]
                        if parent >= 0:
                            X_wpj = _transformd_compose(body_q[parent], joint_X_p[joint])
                        X_wc = _transformd_identity()
                        v_PB_W = _sap_spatial(_vec3d_zero(), _vec3d_zero())
                        A_PB_W = _sap_spatial(_vec3d_zero(), _vec3d_zero())
                        X_j = _sap_calc_across_mobilizer_transform(
                            jtype,
                            joint_axis,
                            qd_start,
                            lin_axis_count,
                            ang_axis_count,
                            joint_q,
                            q_start,
                        )
                        X_wc = _sap_compose_child_body_transform(
                            X_wpj,
                            X_j,
                            joint_X_c[joint],
                            joint_X_c_identity[joint],
                        )
                        v_PB_W, A_PB_W = _sap_calc_mobilizer_velocity_and_bias_body_world(
                            jtype,
                            joint_axis,
                            lin_axis_count,
                            ang_axis_count,
                            X_wpj,
                            X_j,
                            joint_X_c[joint],
                            joint_X_c_identity[joint],
                            joint_q,
                            q_start,
                            joint_qd,
                            qd_start,
                            joint_S_s,
                        )
                        body_q[child] = X_wc

                        v_s = v_PB_W
                        a_s = A_PB_W
                        if parent >= 0:
                            v_parent_s = body_v_s[parent]
                            a_parent_s = body_a_s[parent]
                            p_parent = wp.transform_get_translation(body_q[parent])
                            p_child = wp.transform_get_translation(X_wc)
                            p_PB_W = p_child - p_parent
                            v_s = _sap_compose_velocity(v_parent_s, p_PB_W, v_PB_W)
                            a_s = _sap_compose_acceleration(a_parent_s, v_parent_s, p_PB_W, v_PB_W, A_PB_W)

                        body_v_s[child] = v_s
                        body_a_s[child] = a_s
                        body_q_f[child] = _transformf_from_transformd(X_wc)
                        body_v_s_f[child] = _spatialf_from_spatiald(v_s)
                        body_a_s_f[child] = _spatialf_from_spatiald(a_s)

                slot = slot + stride

            sync_values = wp.tile_zeros((tile_size,), dtype=wp.int32, storage="shared")
            sync_values[tid] = wp.int32(0)
            _ = wp.tile_sum(sync_values)

    return _eval_rigid_id_tiled_articulations_f32_revolute


@wp.kernel
def _eval_rigid_body_dynamics_parallel(
    joint_child: wp.array(dtype=int),
    body_inertia: wp.array(dtype=wp.mat33d),
    body_mass: wp.array(dtype=wp.float64),
    body_com: wp.array(dtype=wp.vec3d),
    body_q: wp.array(dtype=wp.transformd),
    body_world: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3d),
    body_I_s: wp.array(dtype=wp.spatial_matrixd),
    body_v_s: wp.array(dtype=wp.spatial_vectord),
    body_f_s: wp.array(dtype=wp.spatial_vectord),
    body_a_s: wp.array(dtype=wp.spatial_vectord),
):
    joint = wp.tid()
    child = joint_child[joint]
    if child < 0:
        return

    X_wc = body_q[child]
    v_s = body_v_s[child]
    a_s = body_a_s[child]
    mass = body_mass[child]
    body_inertia_d = body_inertia[child]
    body_com_d = body_com[child]
    world_idx = body_world[child]
    world_g = gravity[wp.max(world_idx, 0)]

    I_s = _sap_spatial_inertia_body_origin_world(X_wc, body_inertia_d, mass, body_com_d)
    f_b_s = I_s * a_s + _sap_dynamic_bias_force_body_origin_world(
        X_wc,
        body_inertia_d,
        mass,
        body_com_d,
        v_s,
    )
    f_g_s = _sap_gravity_force_body_origin_world(X_wc, mass, body_com_d, world_g)

    body_f_s[child] = f_b_s - f_g_s
    body_I_s[child] = I_s


@wp.kernel
def _eval_rigid_jacobian_parallel(
    articulation_start: wp.array(dtype=int),
    articulation_J_start: wp.array(dtype=int),
    joint_ancestor: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_S_s: wp.array(dtype=wp.spatial_vectord),
    body_q: wp.array(dtype=wp.transformd),
    max_articulation_joint_count: int,
    J: wp.array(dtype=wp.float64),
):
    art, joint_slot, axis = wp.tid()

    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]
    joint_count = joint_end - joint_start
    if joint_slot >= joint_count:
        return

    articulation_dof_start = joint_qd_start[joint_start]
    articulation_dof_end = joint_qd_start[joint_end]
    articulation_dof_count = articulation_dof_end - articulation_dof_start
    J_offset = articulation_J_start[art]

    row = joint_slot * 6 + axis
    joint = joint_start + joint_slot
    row_body = joint_child[joint]
    p_row = wp.transform_get_translation(body_q[row_body])
    while joint != -1:
        col_body = joint_child[joint]
        p_col = wp.transform_get_translation(body_q[col_body])
        p_col_row = p_row - p_col
        joint_dof_start = joint_qd_start[joint]
        joint_dof_end = joint_qd_start[joint + 1]
        joint_dof_count = joint_dof_end - joint_dof_start

        for dof in range(joint_dof_count):
            col = (joint_dof_start - articulation_dof_start) + dof
            S_Bc = joint_S_s[joint_dof_start + dof]
            S_Brow = _sap_shift_velocity(S_Bc, p_col_row)
            J[J_offset + row * articulation_dof_count + col] = wp.float64(S_Brow[axis])

        joint = joint_ancestor[joint]


@wp.kernel
def _eval_rigid_jacobian_parallel_f32_math(
    articulation_start: wp.array(dtype=int),
    articulation_J_start: wp.array(dtype=int),
    joint_ancestor: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_S_s: wp.array(dtype=wp.spatial_vectord),
    body_q: wp.array(dtype=wp.transformd),
    max_articulation_joint_count: int,
    J: wp.array(dtype=wp.float64),
):
    art, joint_slot, axis = wp.tid()

    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]
    joint_count = joint_end - joint_start
    if joint_slot >= joint_count:
        return

    articulation_dof_start = joint_qd_start[joint_start]
    articulation_dof_end = joint_qd_start[joint_end]
    articulation_dof_count = articulation_dof_end - articulation_dof_start
    J_offset = articulation_J_start[art]

    row = joint_slot * 6 + axis
    joint = joint_start + joint_slot
    row_body = joint_child[joint]
    p_row = _vec3f_from_vec3d(wp.transform_get_translation(body_q[row_body]))
    while joint != -1:
        col_body = joint_child[joint]
        p_col = _vec3f_from_vec3d(wp.transform_get_translation(body_q[col_body]))
        p_col_row = p_row - p_col
        joint_dof_start = joint_qd_start[joint]
        joint_dof_end = joint_qd_start[joint + 1]
        joint_dof_count = joint_dof_end - joint_dof_start

        for dof in range(joint_dof_count):
            col = (joint_dof_start - articulation_dof_start) + dof
            S_Bc = _spatialf_from_spatiald(joint_S_s[joint_dof_start + dof])
            S_Brow = _sap_shift_velocity_f32(S_Bc, p_col_row)
            J[J_offset + row * articulation_dof_count + col] = wp.float64(S_Brow[axis])

        joint = joint_ancestor[joint]


@wp.kernel
def _eval_rigid_mass_parallel(
    articulation_start: wp.array(dtype=int),
    articulation_M_start: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    body_I_s: wp.array(dtype=wp.spatial_matrixd),
    max_articulation_joint_count: int,
    M: wp.array(dtype=wp.float64),
):
    art, joint_slot, element = wp.tid()

    joint_start = articulation_start[art]
    joint_end = articulation_start[art + 1]
    joint_count = joint_end - joint_start
    if joint_slot >= joint_count:
        return

    row = element // 6
    col = element - row * 6
    stride = joint_count * 6
    joint = joint_start + joint_slot
    body = joint_child[joint]
    if body < 0:
        return
    I = body_I_s[body]
    M_offset = articulation_M_start[art]
    M[M_offset + (joint_slot * 6 + row) * stride + joint_slot * 6 + col] = wp.float64(I[row, col])


@wp.kernel
def _eval_mass_times_jacobian_batched(
    articulation_m_rows: wp.array(dtype=wp.int32),
    articulation_j_cols: wp.array(dtype=wp.int32),
    articulation_m_start: wp.array(dtype=wp.int32),
    articulation_j_start: wp.array(dtype=wp.int32),
    m: wp.array(dtype=wp.float64),
    j: wp.array(dtype=wp.float64),
    p: wp.array(dtype=wp.float64),
):
    art, row, col_block = wp.tid()
    m_rows = articulation_m_rows[art]
    j_cols = articulation_j_cols[art]
    col = col_block * _GEMM_COL_BLOCK
    if row >= m_rows or col >= j_cols:
        return

    m_start = articulation_m_start[art]
    j_start = articulation_j_start[art]
    acc0 = wp.float64(0.0)
    acc1 = wp.float64(0.0)
    acc2 = wp.float64(0.0)
    acc3 = wp.float64(0.0)
    block_row = (row // 6) * 6
    m_row_start = m_start + row * m_rows + block_row
    j_block_start = j_start + block_row * j_cols + col
    for k_local in range(6):
        mval = m[m_row_start + k_local]
        base = j_block_start + k_local * j_cols
        j0 = j[base]
        j1 = wp.float64(0.0)
        j2 = wp.float64(0.0)
        j3 = wp.float64(0.0)
        if col + 1 < j_cols:
            j1 = j[base + 1]
        if col + 2 < j_cols:
            j2 = j[base + 2]
        if col + 3 < j_cols:
            j3 = j[base + 3]
        acc0 = acc0 + mval * j0
        acc1 = acc1 + mval * j1
        acc2 = acc2 + mval * j2
        acc3 = acc3 + mval * j3

    out = j_start + row * j_cols + col
    p[out] = acc0
    if col + 1 < j_cols:
        p[out + 1] = acc1
    if col + 2 < j_cols:
        p[out + 2] = acc2
    if col + 3 < j_cols:
        p[out + 3] = acc3


@wp.kernel
def _eval_mass_times_jacobian_batched_f32_math(
    articulation_m_rows: wp.array(dtype=wp.int32),
    articulation_j_cols: wp.array(dtype=wp.int32),
    articulation_m_start: wp.array(dtype=wp.int32),
    articulation_j_start: wp.array(dtype=wp.int32),
    m: wp.array(dtype=wp.float64),
    j: wp.array(dtype=wp.float64),
    p: wp.array(dtype=wp.float64),
):
    art, row, col_block = wp.tid()
    m_rows = articulation_m_rows[art]
    j_cols = articulation_j_cols[art]
    col = col_block * _GEMM_COL_BLOCK
    if row >= m_rows or col >= j_cols:
        return

    m_start = articulation_m_start[art]
    j_start = articulation_j_start[art]
    acc0 = wp.float32(0.0)
    acc1 = wp.float32(0.0)
    acc2 = wp.float32(0.0)
    acc3 = wp.float32(0.0)
    block_row = (row // 6) * 6
    m_row_start = m_start + row * m_rows + block_row
    j_block_start = j_start + block_row * j_cols + col
    for k_local in range(6):
        mval = wp.float32(m[m_row_start + k_local])
        base = j_block_start + k_local * j_cols
        j0 = wp.float32(j[base])
        j1 = wp.float32(0.0)
        j2 = wp.float32(0.0)
        j3 = wp.float32(0.0)
        if col + 1 < j_cols:
            j1 = wp.float32(j[base + 1])
        if col + 2 < j_cols:
            j2 = wp.float32(j[base + 2])
        if col + 3 < j_cols:
            j3 = wp.float32(j[base + 3])
        acc0 = acc0 + mval * j0
        acc1 = acc1 + mval * j1
        acc2 = acc2 + mval * j2
        acc3 = acc3 + mval * j3

    out = j_start + row * j_cols + col
    p[out] = wp.float64(acc0)
    if col + 1 < j_cols:
        p[out + 1] = wp.float64(acc1)
    if col + 2 < j_cols:
        p[out + 2] = wp.float64(acc2)
    if col + 3 < j_cols:
        p[out + 3] = wp.float64(acc3)


@wp.kernel
def _eval_jacobian_transpose_times_p_batched(
    articulation_j_rows: wp.array(dtype=wp.int32),
    articulation_j_cols: wp.array(dtype=wp.int32),
    articulation_j_start: wp.array(dtype=wp.int32),
    articulation_h_start: wp.array(dtype=wp.int32),
    j: wp.array(dtype=wp.float64),
    p: wp.array(dtype=wp.float64),
    h: wp.array(dtype=wp.float64),
):
    art, row, col_block = wp.tid()
    j_rows = articulation_j_rows[art]
    j_cols = articulation_j_cols[art]
    col = col_block * _GEMM_COL_BLOCK
    if row >= j_cols or col >= j_cols:
        return
    if col > row:
        return

    j_start = articulation_j_start[art]
    h_start = articulation_h_start[art]
    acc0 = wp.float64(0.0)
    acc1 = wp.float64(0.0)
    acc2 = wp.float64(0.0)
    acc3 = wp.float64(0.0)
    for k in range(j_rows):
        jval = j[j_start + k * j_cols + row]
        base = j_start + k * j_cols + col
        p0 = p[base]
        p1 = wp.float64(0.0)
        p2 = wp.float64(0.0)
        p3 = wp.float64(0.0)
        if col + 1 < j_cols and col + 1 <= row:
            p1 = p[base + 1]
        if col + 2 < j_cols and col + 2 <= row:
            p2 = p[base + 2]
        if col + 3 < j_cols and col + 3 <= row:
            p3 = p[base + 3]
        acc0 = acc0 + jval * p0
        acc1 = acc1 + jval * p1
        acc2 = acc2 + jval * p2
        acc3 = acc3 + jval * p3

    out = h_start + row * j_cols + col
    h[out] = acc0
    if col != row:
        h[h_start + col * j_cols + row] = acc0
    if col + 1 < j_cols and col + 1 <= row:
        c = col + 1
        h[out + 1] = acc1
        if c != row:
            h[h_start + c * j_cols + row] = acc1
    if col + 2 < j_cols and col + 2 <= row:
        c = col + 2
        h[out + 2] = acc2
        if c != row:
            h[h_start + c * j_cols + row] = acc2
    if col + 3 < j_cols and col + 3 <= row:
        c = col + 3
        h[out + 3] = acc3
        if c != row:
            h[h_start + c * j_cols + row] = acc3


@cache
def _make_eval_jacobian_transpose_times_p_tiled(tile_size: int):
    @wp.kernel(enable_backward=False)
    def _eval_jacobian_transpose_times_p_tiled(
        articulation_j_rows: wp.array(dtype=wp.int32),
        articulation_j_cols: wp.array(dtype=wp.int32),
        articulation_j_start: wp.array(dtype=wp.int32),
        articulation_h_start: wp.array(dtype=wp.int32),
        j: wp.array(dtype=wp.float64),
        p: wp.array(dtype=wp.float64),
        h: wp.array(dtype=wp.float64),
    ):
        art, upper, tid = wp.tid()
        j_rows = articulation_j_rows[art]
        j_cols = articulation_j_cols[art]
        upper_count = (j_cols * (j_cols + 1)) // 2
        if upper >= upper_count:
            return

        row = wp.int32(0)
        rem = wp.int32(upper)
        row_count = wp.int32(j_cols)
        while rem >= row_count:
            rem = rem - row_count
            row = row + 1
            row_count = row_count - 1
        col = row + rem

        j_start = articulation_j_start[art]
        acc = wp.float64(0.0)
        k = tid
        stride = wp.block_dim()
        while k < j_rows:
            base = j_start + k * j_cols
            acc = acc + j[base + row] * p[base + col]
            k = k + stride

        values = wp.tile_zeros((tile_size,), dtype=wp.float64, storage="shared")
        values[tid] = acc
        total = wp.tile_sum(values)
        h_start = articulation_h_start[art]
        wp.tile_store(h, total, offset=h_start + row * j_cols + col)
        if col != row:
            wp.tile_store(h, total, offset=h_start + col * j_cols + row)

    return _eval_jacobian_transpose_times_p_tiled


@cache
def _make_eval_jacobian_transpose_times_p_tiled_f32_math(tile_size: int):
    @wp.kernel(enable_backward=False)
    def _eval_jacobian_transpose_times_p_tiled_f32_math(
        articulation_j_rows: wp.array(dtype=wp.int32),
        articulation_j_cols: wp.array(dtype=wp.int32),
        articulation_j_start: wp.array(dtype=wp.int32),
        articulation_h_start: wp.array(dtype=wp.int32),
        j: wp.array(dtype=wp.float64),
        p: wp.array(dtype=wp.float64),
        h: wp.array(dtype=wp.float64),
    ):
        art, upper, tid = wp.tid()
        j_rows = articulation_j_rows[art]
        j_cols = articulation_j_cols[art]
        upper_count = (j_cols * (j_cols + 1)) // 2
        if upper >= upper_count:
            return

        row = wp.int32(0)
        rem = wp.int32(upper)
        row_count = wp.int32(j_cols)
        while rem >= row_count:
            rem = rem - row_count
            row = row + 1
            row_count = row_count - 1
        col = row + rem

        j_start = articulation_j_start[art]
        acc = wp.float32(0.0)
        k = tid
        stride = wp.block_dim()
        while k < j_rows:
            base = j_start + k * j_cols
            acc = acc + wp.float32(j[base + row]) * wp.float32(p[base + col])
            k = k + stride

        values = wp.tile_zeros((tile_size,), dtype=wp.float64, storage="shared")
        values[tid] = wp.float64(acc)
        total = wp.tile_sum(values)
        h_start = articulation_h_start[art]
        wp.tile_store(h, total, offset=h_start + row * j_cols + col)
        if col != row:
            wp.tile_store(h, total, offset=h_start + col * j_cols + row)

    return _eval_jacobian_transpose_times_p_tiled_f32_math


@cache
def _make_eval_jacobian_transpose_times_p_gemm_tile(tile_m: int, tile_n: int, tile_k: int):
    @wp.kernel(enable_backward=False, module="unique")
    def _eval_jacobian_transpose_times_p_gemm_tile(
        articulation_j_rows: wp.array(dtype=wp.int32),
        articulation_j_cols: wp.array(dtype=wp.int32),
        articulation_j_start: wp.array(dtype=wp.int32),
        articulation_h_start: wp.array(dtype=wp.int32),
        j: wp.array(dtype=wp.float64),
        p: wp.array(dtype=wp.float64),
        h: wp.array(dtype=wp.float64),
    ):
        art, row_tile, col_tile, tid = wp.tid()
        j_rows = articulation_j_rows[art]
        j_cols = articulation_j_cols[art]
        row0 = row_tile * tile_m
        col0 = col_tile * tile_n
        if row0 >= j_cols or col0 >= j_cols or col0 > row0 + tile_m - 1:
            return

        j_start = articulation_j_start[art]
        acc = wp.tile_zeros((tile_m, tile_n), dtype=wp.float64)
        k0 = wp.int32(0)
        while k0 < j_rows:
            j_tile = wp.tile_zeros((tile_k, tile_m), dtype=wp.float64, storage="shared")
            p_tile = wp.tile_zeros((tile_k, tile_n), dtype=wp.float64, storage="shared")

            linear = tid
            while linear < tile_k * tile_m:
                kk = linear // tile_m
                rr = linear - kk * tile_m
                k = k0 + kk
                row = row0 + rr
                value = wp.float64(0.0)
                if k < j_rows and row < j_cols:
                    value = j[j_start + k * j_cols + row]
                j_tile[kk, rr] = value
                linear = linear + wp.block_dim()

            linear = tid
            while linear < tile_k * tile_n:
                kk = linear // tile_n
                cc = linear - kk * tile_n
                k = k0 + kk
                col = col0 + cc
                value = wp.float64(0.0)
                if k < j_rows and col < j_cols:
                    value = p[j_start + k * j_cols + col]
                p_tile[kk, cc] = value
                linear = linear + wp.block_dim()

            wp.tile_matmul(wp.tile_transpose(j_tile), p_tile, acc)
            k0 = k0 + tile_k

        h_start = articulation_h_start[art]
        linear = tid
        while linear < tile_m * tile_n:
            rr = linear // tile_n
            cc = linear - rr * tile_n
            row = row0 + rr
            col = col0 + cc
            if row < j_cols and col < j_cols and col <= row:
                value = acc[rr, cc]
                h[h_start + row * j_cols + col] = value
                if col != row:
                    h[h_start + col * j_cols + row] = value
            linear = linear + wp.block_dim()

    return _eval_jacobian_transpose_times_p_gemm_tile


@cache
def _make_eval_jacobian_transpose_times_p_gemm_tile_f32_math(tile_m: int, tile_n: int, tile_k: int):
    @wp.kernel(enable_backward=False, module="unique")
    def _eval_jacobian_transpose_times_p_gemm_tile_f32_math(
        articulation_j_rows: wp.array(dtype=wp.int32),
        articulation_j_cols: wp.array(dtype=wp.int32),
        articulation_j_start: wp.array(dtype=wp.int32),
        articulation_h_start: wp.array(dtype=wp.int32),
        j: wp.array(dtype=wp.float64),
        p: wp.array(dtype=wp.float64),
        h: wp.array(dtype=wp.float64),
    ):
        art, row_tile, col_tile, tid = wp.tid()
        j_rows = articulation_j_rows[art]
        j_cols = articulation_j_cols[art]
        row0 = row_tile * tile_m
        col0 = col_tile * tile_n
        if row0 >= j_cols or col0 >= j_cols or col0 > row0 + tile_m - 1:
            return

        j_start = articulation_j_start[art]
        acc = wp.tile_zeros((tile_m, tile_n), dtype=wp.float32)
        k0 = wp.int32(0)
        while k0 < j_rows:
            j_tile = wp.tile_zeros((tile_k, tile_m), dtype=wp.float32, storage="shared")
            p_tile = wp.tile_zeros((tile_k, tile_n), dtype=wp.float32, storage="shared")

            linear = tid
            while linear < tile_k * tile_m:
                kk = linear // tile_m
                rr = linear - kk * tile_m
                k = k0 + kk
                row = row0 + rr
                value = wp.float32(0.0)
                if k < j_rows and row < j_cols:
                    value = wp.float32(j[j_start + k * j_cols + row])
                j_tile[kk, rr] = value
                linear = linear + wp.block_dim()

            linear = tid
            while linear < tile_k * tile_n:
                kk = linear // tile_n
                cc = linear - kk * tile_n
                k = k0 + kk
                col = col0 + cc
                value = wp.float32(0.0)
                if k < j_rows and col < j_cols:
                    value = wp.float32(p[j_start + k * j_cols + col])
                p_tile[kk, cc] = value
                linear = linear + wp.block_dim()

            wp.tile_matmul(wp.tile_transpose(j_tile), p_tile, acc)
            k0 = k0 + tile_k

        h_start = articulation_h_start[art]
        linear = tid
        while linear < tile_m * tile_n:
            rr = linear // tile_n
            cc = linear - rr * tile_n
            row = row0 + rr
            col = col0 + cc
            if row < j_cols and col < j_cols and col <= row:
                out_value = wp.float64(acc[rr, cc])
                h[h_start + row * j_cols + col] = out_value
                if col != row:
                    h[h_start + col * j_cols + row] = out_value
            linear = linear + wp.block_dim()

    return _eval_jacobian_transpose_times_p_gemm_tile_f32_math


@wp.kernel
def _integrate_joint_velocity_kernel(
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    dt: float,
    joint_qd_out: wp.array(dtype=float),
):
    i = wp.tid()
    joint_qd_out[i] = joint_qd[i] + wp.float32(dt) * joint_qdd[i]


@wp.kernel
def _assemble_global_sap_dynamics_matrix_kernel(
    h_flat: wp.array(dtype=wp.float64),
    joint_armature: wp.array(dtype=wp.float64),
    dof_articulation_index: wp.array(dtype=wp.int32),
    dof_articulation_local_index: wp.array(dtype=wp.int32),
    articulation_dof_start: wp.array(dtype=wp.int32),
    articulation_h_start: wp.array(dtype=wp.int32),
    articulation_h_rows: wp.array(dtype=wp.int32),
    dynamics_matrix: wp.array(dtype=wp.float64, ndim=2),
):
    sap_i, sap_j = wp.tid()
    art_i = dof_articulation_index[sap_i]
    art_j = dof_articulation_index[sap_j]

    acc = wp.float64(0.0)
    if art_i >= 0 and art_i == art_j:
        rows = articulation_h_rows[art_i]
        h_start = articulation_h_start[art_i]
        local_i = dof_articulation_local_index[sap_i]
        local_j = dof_articulation_local_index[sap_j]
        if local_i >= 0 and local_i < rows and local_j >= 0 and local_j < rows:
            acc = h_flat[h_start + local_i * rows + local_j]
            if sap_i == sap_j:
                acc = acc + joint_armature[sap_i]

    dynamics_matrix[sap_i, sap_j] = acc


@wp.kernel
def _eval_dense_cholesky_batched_f64(
    A_starts: wp.array(dtype=int),
    A_dim: wp.array(dtype=int),
    R_starts: wp.array(dtype=int),
    A: wp.array(dtype=wp.float64),
    R: wp.array(dtype=wp.float64),
    L: wp.array(dtype=wp.float64),
):
    batch = wp.tid()
    _dense_cholesky_f64(A_dim[batch], A, R, A_starts[batch], R_starts[batch], L)


@wp.kernel
def _eval_dense_solve_batched_f64(
    L_start: wp.array(dtype=int),
    L_dim: wp.array(dtype=int),
    b_start: wp.array(dtype=int),
    A: wp.array(dtype=wp.float64),
    L: wp.array(dtype=wp.float64),
    b: wp.array(dtype=wp.float64),
    x: wp.array(dtype=wp.float64),
    tmp: wp.array(dtype=wp.float64),
):
    batch = wp.tid()
    _dense_subs_f64(L_dim[batch], L_start[batch], b_start[batch], L, b, x)


@wp.kernel
def _pack_articulation_h_to_padded_batched_f64(
    H_start: wp.array(dtype=int),
    H_rows: wp.array(dtype=int),
    dof_start: wp.array(dtype=int),
    H: wp.array(dtype=wp.float64),
    armature: wp.array(dtype=wp.float64),
    max_rows: int,
    out: wp.array(dtype=wp.float64, ndim=3),
):
    art, i, j = wp.tid()
    rows = H_rows[art]
    value = wp.float64(0.0)
    if i < max_rows and j < max_rows:
        if i < rows and j < rows:
            value = H[H_start[art] + i * rows + j]
            if i == j:
                value = value + armature[dof_start[art] + i]
        elif i == j:
            value = wp.float64(1.0)
    out[art, i, j] = value


@wp.kernel
def _pack_articulation_tau_to_padded_batched_f64(
    H_rows: wp.array(dtype=int),
    dof_start: wp.array(dtype=int),
    tau: wp.array(dtype=wp.float64),
    max_rows: int,
    out: wp.array(dtype=wp.float64, ndim=3),
):
    art, i = wp.tid()
    rows = H_rows[art]
    value = wp.float64(0.0)
    if i < max_rows and i < rows:
        value = tau[dof_start[art] + i]
    out[art, i, 0] = value


@wp.kernel
def _unpack_articulation_solution_from_padded_batched_f64(
    H_rows: wp.array(dtype=int),
    dof_start: wp.array(dtype=int),
    max_rows: int,
    x: wp.array(dtype=wp.float64, ndim=3),
    out: wp.array(dtype=wp.float64),
):
    art, i = wp.tid()
    rows = H_rows[art]
    if i < max_rows and i < rows:
        out[dof_start[art] + i] = x[art, i, 0]


@wp.kernel
def _pack_articulation_h_to_padded_batched_f32(
    H_start: wp.array(dtype=int),
    H_rows: wp.array(dtype=int),
    dof_start: wp.array(dtype=int),
    H: wp.array(dtype=wp.float64),
    armature: wp.array(dtype=wp.float64),
    max_rows: int,
    out: wp.array(dtype=wp.float32, ndim=3),
):
    art, i, j = wp.tid()
    rows = H_rows[art]
    value = wp.float32(0.0)
    if i < max_rows and j < max_rows:
        if i < rows and j < rows:
            value = wp.float32(H[H_start[art] + i * rows + j])
            if i == j:
                value = value + wp.float32(armature[dof_start[art] + i])
        elif i == j:
            value = wp.float32(1.0)
    out[art, i, j] = value


@wp.kernel
def _pack_articulation_tau_to_padded_batched_f32(
    H_rows: wp.array(dtype=int),
    dof_start: wp.array(dtype=int),
    tau: wp.array(dtype=wp.float64),
    max_rows: int,
    out: wp.array(dtype=wp.float32, ndim=3),
):
    art, i = wp.tid()
    rows = H_rows[art]
    value = wp.float32(0.0)
    if i < max_rows and i < rows:
        value = wp.float32(tau[dof_start[art] + i])
    out[art, i, 0] = value


@wp.kernel
def _unpack_articulation_solution_from_padded_batched_f32(
    H_rows: wp.array(dtype=int),
    dof_start: wp.array(dtype=int),
    max_rows: int,
    x: wp.array(dtype=wp.float32, ndim=3),
    out: wp.array(dtype=wp.float64),
):
    art, i = wp.tid()
    rows = H_rows[art]
    if i < max_rows and i < rows:
        out[dof_start[art] + i] = wp.float64(x[art, i, 0])


class SapFreeMotion:
    """Standalone SAP-style free-motion calculation for articulated models.

    Inputs and outputs use SAP's floating mobilizer convention for
    FREE/DISTANCE joints: `[w, v_body_origin]`. All other joint DOFs keep the
    model's declared order. Frontend-specific convention conversion belongs in
    the caller before constructing `SapState` / `SapControl`.
    """

    def __init__(
        self,
        model: Model,
        *,
        allocate_dynamics_matrix: bool = False,
        use_f64_boundary_pose: bool = True,
        linear_solve_precision: str = "fp64",
    ):
        if not isinstance(model, Model):
            raise TypeError("SapFreeMotion requires SapModel; convert in the frontend adapter before construction.")
        if int(model.joint_count) <= 0 or int(model.joint_dof_count) <= 0:
            raise ValueError("SapFreeMotion requires a model with articulated joint DOFs.")

        self.model = model
        self.use_f64_boundary_pose = bool(use_f64_boundary_pose)
        self.linear_solve_precision = self._normalize_linear_solve_precision(linear_solve_precision)
        self._compute_articulation_indices(model)
        self.rigid_tile_size = max(32, int(self.max_articulation_level_width))
        self._rigid_id_tiled = _make_eval_rigid_id_tiled_articulations(
            int(self.rigid_tile_size)
        )
        self._rigid_id_tiled_f32_revolute = _make_eval_rigid_id_tiled_articulations_f32_revolute(
            int(self.rigid_tile_size)
        )
        self._rigid_tau_tiled = _make_eval_rigid_tau_no_drives_tiled(
            int(self.rigid_tile_size)
        )
        self._jtp_gemm_tile = _make_eval_jacobian_transpose_times_p_gemm_tile(
            _JTP_GEMM_TILE_M,
            _JTP_GEMM_TILE_N,
            _JTP_GEMM_TILE_K,
        )
        self._jtp_gemm_tile_f32_math = _make_eval_jacobian_transpose_times_p_gemm_tile_f32_math(
            _JTP_GEMM_TILE_M,
            _JTP_GEMM_TILE_N,
            _JTP_GEMM_TILE_K,
        )
        self._build_dof_maps(model)
        self._allocate_buffers(model, allocate_dynamics_matrix=allocate_dynamics_matrix)

    @property
    def device(self):
        """Return the Warp device that owns the free-motion work buffers."""
        return self.model.device

    @staticmethod
    def _normalize_linear_solve_precision(value: str) -> str:
        precision = str(value).strip().lower()
        if precision == "f32":
            precision = "fp32"
        elif precision == "f64":
            precision = "fp64"
        if precision not in {"fp32", "fp64"}:
            raise ValueError(
                "linear_solve_precision must be 'fp32'/'f32' or 'fp64'/'f64', "
                f"got {value!r}."
            )
        return precision

    def _compute_articulation_indices(self, model: Model) -> None:
        self.max_articulation_level_count = 0
        self.max_articulation_level_width = 0
        self.max_articulation_joint_count = 0
        self.max_articulation_m_rows = 0
        self.max_articulation_j_cols = 0

        self.J_size = 0
        self.M_size = 0
        self.H_size = 0

        articulation_J_start = []
        articulation_M_start = []
        articulation_H_start = []
        articulation_M_rows = []
        articulation_H_rows = []
        articulation_J_rows = []
        articulation_J_cols = []
        articulation_dof_start = []
        articulation_level_lists = []

        articulation_start = model.articulation_start.numpy()
        joint_parent = model.joint_parent.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        for art in range(int(model.articulation_count)):
            first_joint = int(articulation_start[art])
            last_joint = int(articulation_start[art + 1])
            joint_count = last_joint - first_joint
            first_dof = int(joint_qd_start[first_joint])
            last_dof = int(joint_qd_start[last_joint])
            dof_count = last_dof - first_dof

            articulation_J_start.append(self.J_size)
            articulation_M_start.append(self.M_size)
            articulation_H_start.append(self.H_size)
            articulation_M_rows.append(joint_count * 6)
            articulation_H_rows.append(dof_count)
            articulation_J_rows.append(joint_count * 6)
            articulation_J_cols.append(dof_count)
            articulation_dof_start.append(first_dof)

            self.max_articulation_joint_count = max(self.max_articulation_joint_count, joint_count)
            self.max_articulation_m_rows = max(self.max_articulation_m_rows, joint_count * 6)
            self.max_articulation_j_cols = max(self.max_articulation_j_cols, dof_count)

            local_children = [[] for _ in range(joint_count)]
            local_depth = [-1] * joint_count
            queue = []
            for joint in range(first_joint, last_joint):
                parent_joint = int(joint_parent[joint])
                local_joint = joint - first_joint
                if parent_joint < first_joint or parent_joint >= last_joint:
                    local_depth[local_joint] = 0
                    queue.append(local_joint)
                else:
                    local_children[parent_joint - first_joint].append(local_joint)

            q_head = 0
            while q_head < len(queue):
                local_joint = queue[q_head]
                q_head += 1
                for child_local in local_children[local_joint]:
                    local_depth[child_local] = local_depth[local_joint] + 1
                    queue.append(child_local)

            level_count = max(local_depth) + 1 if local_depth else 0
            level_lists = [[] for _ in range(level_count)]
            for local_joint, depth in enumerate(local_depth):
                level_lists[depth].append(first_joint + local_joint)
            articulation_level_lists.append(level_lists)
            self.max_articulation_level_count = max(self.max_articulation_level_count, level_count)
            for joints_at_level in level_lists:
                self.max_articulation_level_width = max(
                    self.max_articulation_level_width,
                    len(joints_at_level),
                )

            self.J_size += 6 * joint_count * dof_count
            self.M_size += 6 * joint_count * 6 * joint_count
            self.H_size += dof_count * dof_count

        self.max_articulation_level_count = max(self.max_articulation_level_count, 1)
        self.max_articulation_level_width = max(self.max_articulation_level_width, 1)
        self.max_articulation_joint_count = max(self.max_articulation_joint_count, 1)
        self.max_articulation_m_rows = max(self.max_articulation_m_rows, 1)
        self.max_articulation_j_cols = max(self.max_articulation_j_cols, 1)

        self.articulation_J_start = wp.array(articulation_J_start, dtype=wp.int32, device=model.device)
        self.articulation_M_start = wp.array(articulation_M_start, dtype=wp.int32, device=model.device)
        self.articulation_H_start = wp.array(articulation_H_start, dtype=wp.int32, device=model.device)
        self.articulation_M_rows = wp.array(articulation_M_rows, dtype=wp.int32, device=model.device)
        self.articulation_H_rows = wp.array(articulation_H_rows, dtype=wp.int32, device=model.device)
        self.articulation_J_rows = wp.array(articulation_J_rows, dtype=wp.int32, device=model.device)
        self.articulation_J_cols = wp.array(articulation_J_cols, dtype=wp.int32, device=model.device)
        self.articulation_dof_start = wp.array(articulation_dof_start, dtype=wp.int32, device=model.device)

        level_index = np.full(
            int(model.articulation_count) * self.max_articulation_level_count * self.max_articulation_level_width,
            -1,
            dtype=np.int32,
        )
        for art, level_lists in enumerate(articulation_level_lists):
            for level, joints_at_level in enumerate(level_lists):
                base = (art * self.max_articulation_level_count + level) * self.max_articulation_level_width
                level_index[base : base + len(joints_at_level)] = joints_at_level
        self.articulation_level_joint_index = wp.array(level_index, dtype=wp.int32, device=model.device)

    def _build_dof_maps(self, model: Model) -> None:
        dof_count = int(model.joint_dof_count)
        joint_for_dof = np.full(dof_count, -1, dtype=np.int32)
        axis_for_dof = np.full(dof_count, -1, dtype=np.int32)
        dof_articulation = np.full(dof_count, -1, dtype=np.int32)
        dof_articulation_local = np.full(dof_count, -1, dtype=np.int32)

        joint_type = model.joint_type.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_dof_dim = model.joint_dof_dim.numpy()
        articulation_start = model.articulation_start.numpy()

        for art in range(int(model.articulation_count)):
            first_joint = int(articulation_start[art])
            last_joint = int(articulation_start[art + 1])
            first_dof = int(joint_qd_start[first_joint])
            last_dof = int(joint_qd_start[last_joint])
            for dof in range(first_dof, last_dof):
                dof_articulation[dof] = art
                dof_articulation_local[dof] = dof - first_dof

        for joint in range(int(model.joint_count)):
            start = int(joint_qd_start[joint])
            axis_count = int(joint_dof_dim[joint, 0] + joint_dof_dim[joint, 1])
            for axis in range(axis_count):
                dof = start + axis
                joint_for_dof[dof] = joint
                axis_for_dof[dof] = axis

            if int(joint_type[joint]) in (int(SAP_JOINT_FREE), int(SAP_JOINT_DISTANCE)):
                if axis_count != 6:
                    raise ValueError("FREE/DISTANCE joints must have 6 velocity DOFs.")

        self.dof_joint_index = wp.array(joint_for_dof, dtype=wp.int32, device=model.device)
        self.dof_axis_index = wp.array(axis_for_dof, dtype=wp.int32, device=model.device)
        self.dof_articulation_index = wp.array(dof_articulation, dtype=wp.int32, device=model.device)
        self.dof_articulation_local_index = wp.array(dof_articulation_local, dtype=wp.int32, device=model.device)

    def _allocate_buffers(self, model: Model, *, allocate_dynamics_matrix: bool) -> None:
        self.model_body_mass = self._make_model_array_f64(model, "sap_debug_body_mass", "body_mass", wp.float64)
        self.model_body_inertia = self._make_model_array_f64(model, "sap_debug_body_inertia", "body_inertia", wp.mat33d)
        self.model_body_com = self._make_model_array_f64(model, "sap_debug_body_com", "body_com", wp.vec3d)
        self.model_joint_axis = self._make_model_array_f64(model, "sap_debug_joint_axis", "joint_axis", wp.vec3d)
        self.model_joint_X_p = self._make_model_array_f64(model, "sap_debug_joint_X_p", "joint_X_p", wp.transformd)
        self.model_joint_X_c = self._make_model_array_f64(model, "sap_debug_joint_X_c", "joint_X_c", wp.transformd)
        self.model_joint_X_c_identity = self._make_transform_identity_flags(
            model,
            "sap_debug_joint_X_c",
            "joint_X_c",
        )
        self.model_gravity = self._make_model_array_f64(model, "sap_debug_gravity", "gravity", wp.vec3d)
        self.model_joint_armature = self._make_model_array_f64(
            model,
            "sap_debug_joint_armature",
            "joint_armature",
            wp.float64,
        )

        self.M = wp.zeros((self.M_size,), dtype=wp.float64, device=model.device)
        self.J = wp.zeros((self.J_size,), dtype=wp.float64, device=model.device)
        self.P = wp.zeros_like(self.J)
        self.H = wp.zeros((self.H_size,), dtype=wp.float64, device=model.device)
        self.L = wp.zeros_like(self.H)

        self.block_solver = None
        self.block_chol_a = None
        self.block_chol_rhs = None
        self.block_chol_x = None
        block_dtype = wp.float32 if self.linear_solve_precision == "fp32" else wp.float64
        self._block_chol_uses_f32 = block_dtype == wp.float32
        self.block_solver = BlockCholeskySolverBatched(
            max_num_equations=int(self.max_articulation_j_cols),
            batch_size=int(model.articulation_count),
            block_size=32,
            device=model.device,
            dtype=block_dtype,
        )
        padded = int(self.block_solver.max_num_equations)
        self.block_chol_a = wp.zeros(
            (int(model.articulation_count), padded, padded),
            dtype=block_dtype,
            device=model.device,
        )
        self.block_chol_rhs = wp.zeros(
            (int(model.articulation_count), padded, 1),
            dtype=block_dtype,
            device=model.device,
        )
        self.block_chol_x = wp.zeros_like(self.block_chol_rhs)

        self.joint_qdd_sap_solve = wp.zeros((model.joint_dof_count,), dtype=wp.float64, device=model.device)
        self.joint_tau = wp.zeros((model.joint_dof_count,), dtype=wp.float64, device=model.device)
        self.joint_solve_tmp = wp.zeros((model.joint_dof_count,), dtype=wp.float64, device=model.device)
        self.joint_S_s = wp.empty((model.joint_dof_count,), dtype=wp.spatial_vectord, device=model.device)

        self.joint_q_input = wp.zeros((model.joint_coord_count,), dtype=wp.float64, device=model.device)
        self.joint_qd_sap_input = wp.zeros((model.joint_dof_count,), dtype=wp.float64, device=model.device)
        self.joint_f_sap_input = wp.zeros((model.joint_dof_count,), dtype=wp.float64, device=model.device)
        self.free_motion_joint_qd_sap = wp.zeros((model.joint_dof_count,), dtype=wp.float64, device=model.device)
        self.free_motion_joint_qdd_sap = wp.zeros((model.joint_dof_count,), dtype=wp.float64, device=model.device)

        self.body_q = wp.empty((model.body_count,), dtype=wp.transformd, device=model.device)
        self.body_q_f = wp.empty((model.body_count,), dtype=wp.transform, device=model.device)
        self.body_I_s = wp.empty((model.body_count,), dtype=wp.spatial_matrixd, device=model.device)
        self.body_v_s = wp.empty((model.body_count,), dtype=wp.spatial_vectord, device=model.device)
        self.body_v_s_f = wp.empty((model.body_count,), dtype=wp.spatial_vector, device=model.device)
        self.body_a_s = wp.empty((model.body_count,), dtype=wp.spatial_vectord, device=model.device)
        self.body_a_s_f = wp.empty((model.body_count,), dtype=wp.spatial_vector, device=model.device)
        self.body_f_s = wp.zeros((model.body_count,), dtype=wp.spatial_vectord, device=model.device)
        self.body_ft_s = wp.zeros((model.body_count,), dtype=wp.spatial_vectord, device=model.device)
        self.body_f_ext_s = wp.zeros((model.body_count,), dtype=wp.spatial_vectord, device=model.device)

        self.dynamics_matrix_sap = None
        if allocate_dynamics_matrix:
            self._ensure_dynamics_matrix_allocated()
        self._result = SapFreeMotionResult(
            v_star=self.free_motion_joint_qd_sap,
            vdot=self.free_motion_joint_qdd_sap,
            dynamics_matrix=self.dynamics_matrix_sap,
        )

    @staticmethod
    def _make_model_array_f64(model: Model, exact_name: str, fallback_name: str, dtype) -> wp.array:
        src = getattr(model, exact_name, None)
        if src is None:
            src = getattr(model, fallback_name)
        if isinstance(src, wp.array):
            src_np = src.numpy()
        else:
            src_np = np.asarray(src)
        return wp.array(np.asarray(src_np, dtype=np.float64), dtype=dtype, device=model.device)

    @staticmethod
    def _make_transform_identity_flags(model: Model, exact_name: str, fallback_name: str) -> wp.array:
        src = getattr(model, exact_name, None)
        if src is None:
            src = getattr(model, fallback_name)
        if isinstance(src, wp.array):
            src_np = src.numpy()
        else:
            src_np = np.asarray(src)
        transforms = np.asarray(src_np, dtype=np.float64).reshape((-1, 7))
        identity = (
            (np.linalg.norm(transforms[:, 0:3], axis=1) <= 1.0e-12)
            & (np.linalg.norm(transforms[:, 3:6], axis=1) <= 1.0e-12)
            & (np.abs(transforms[:, 6] - 1.0) <= 1.0e-12)
        )
        return wp.array(identity.astype(np.int32), dtype=wp.int32, device=model.device)

    def _ensure_dynamics_matrix_allocated(self) -> None:
        if self.dynamics_matrix_sap is not None:
            return
        model = self.model
        self.dynamics_matrix_sap = wp.zeros(
            (model.joint_dof_count, model.joint_dof_count),
            dtype=wp.float64,
            device=model.device,
        )
        self._result = SapFreeMotionResult(
            v_star=self.free_motion_joint_qd_sap,
            vdot=self.free_motion_joint_qdd_sap,
            dynamics_matrix=self.dynamics_matrix_sap,
        )

    def _prepare_joint_q_input(self, state_in: State) -> None:
        if state_in.joint_q.dtype == wp.float64:
            kernel = _copy_f64
        else:
            kernel = _copy_f32_to_f64
        wp.launch(
            kernel,
            dim=self.model.joint_coord_count,
            inputs=[state_in.joint_q, self.joint_q_input],
            device=self.model.device,
        )

    def _launch_rigid_id(self, state_in: State) -> None:
        model = self.model
        self._prepare_joint_q_input(state_in)
        use_f32_rigid_id = self.linear_solve_precision == "fp32"
        rigid_id_inputs = [
            self.articulation_level_joint_index,
            int(self.max_articulation_level_count),
            int(self.max_articulation_level_width),
            model.joint_type,
            model.joint_parent,
            model.joint_child,
            model.joint_q_start,
            model.joint_qd_start,
            self.joint_q_input,
            self.joint_qd_sap_input,
            self.model_joint_axis,
            model.joint_dof_dim,
            self.model_body_inertia,
            self.model_body_mass,
            self.model_body_com,
            self.body_q,
            self.model_joint_X_p,
            self.model_joint_X_c,
            self.model_joint_X_c_identity,
            model.body_world,
            self.model_gravity,
            self.joint_S_s,
            self.body_I_s,
            self.body_v_s,
            self.body_f_s,
            self.body_a_s,
        ]
        if use_f32_rigid_id:
            rigid_id_inputs.extend(
                [
                    self.body_q_f,
                    self.body_v_s_f,
                    self.body_a_s_f,
                ]
            )
        wp.launch_tiled(
            self._rigid_id_tiled_f32_revolute if use_f32_rigid_id else self._rigid_id_tiled,
            dim=model.articulation_count,
            block_dim=int(self.rigid_tile_size),
            inputs=rigid_id_inputs,
            device=model.device,
        )
        self._launch_rigid_body_dynamics()

    def _launch_rigid_body_dynamics(self) -> None:
        model = self.model
        wp.launch(
            _eval_rigid_body_dynamics_parallel,
            dim=model.joint_count,
            inputs=[
                model.joint_child,
                self.model_body_inertia,
                self.model_body_mass,
                self.model_body_com,
                self.body_q,
                model.body_world,
                self.model_gravity,
                self.body_I_s,
                self.body_v_s,
                self.body_f_s,
                self.body_a_s,
            ],
            device=model.device,
        )

    def _assemble_articulation_matrices(self) -> None:
        model = self.model
        self.J.zero_()
        self.M.zero_()
        self.H.zero_()
        use_f32_math = self.linear_solve_precision == "fp32"

        wp.launch(
            _eval_rigid_jacobian_parallel_f32_math if use_f32_math else _eval_rigid_jacobian_parallel,
            dim=(model.articulation_count, self.max_articulation_joint_count, 6),
            inputs=[
                model.articulation_start,
                self.articulation_J_start,
                model.joint_ancestor,
                model.joint_child,
                model.joint_qd_start,
                self.joint_S_s,
                self.body_q,
                int(self.max_articulation_joint_count),
            ],
            outputs=[self.J],
            device=model.device,
        )

        wp.launch(
            _eval_rigid_mass_parallel,
            dim=(model.articulation_count, self.max_articulation_joint_count, 36),
            inputs=[
                model.articulation_start,
                self.articulation_M_start,
                model.joint_child,
                self.body_I_s,
                int(self.max_articulation_joint_count),
            ],
            outputs=[self.M],
            device=model.device,
        )

        wp.launch(
            _eval_mass_times_jacobian_batched_f32_math if use_f32_math else _eval_mass_times_jacobian_batched,
            dim=(
                model.articulation_count,
                self.max_articulation_m_rows,
                (self.max_articulation_j_cols + 3) // 4,
            ),
            inputs=[
                self.articulation_M_rows,
                self.articulation_J_cols,
                self.articulation_M_start,
                self.articulation_J_start,
                self.M,
                self.J,
            ],
            outputs=[self.P],
            device=model.device,
        )

        wp.launch_tiled(
            self._jtp_gemm_tile_f32_math if use_f32_math else self._jtp_gemm_tile,
            dim=(
                model.articulation_count,
                (self.max_articulation_j_cols + _JTP_GEMM_TILE_M - 1) // _JTP_GEMM_TILE_M,
                (self.max_articulation_j_cols + _JTP_GEMM_TILE_N - 1) // _JTP_GEMM_TILE_N,
            ),
            block_dim=128,
            inputs=[
                self.articulation_J_rows,
                self.articulation_J_cols,
                self.articulation_J_start,
                self.articulation_H_start,
                self.J,
                self.P,
            ],
            outputs=[self.H],
            device=model.device,
        )

    def _factor_dynamics_matrix(self) -> None:
        model = self.model
        wp.launch(
            _eval_dense_cholesky_batched_f64,
            dim=model.articulation_count,
            inputs=[
                self.articulation_H_start,
                self.articulation_H_rows,
                self.articulation_dof_start,
                self.H,
                self.model_joint_armature,
            ],
            outputs=[self.L],
            device=model.device,
        )

    def _can_use_block_dynamics_solve(self) -> bool:
        return self.block_solver is not None

    def _solve_articulation_accelerations_blocked(self) -> None:
        model = self.model
        assert self.block_solver is not None
        assert self.block_chol_a is not None
        assert self.block_chol_rhs is not None
        assert self.block_chol_x is not None
        max_rows = int(self.max_articulation_j_cols)
        pack_h_kernel = (
            _pack_articulation_h_to_padded_batched_f32
            if self._block_chol_uses_f32
            else _pack_articulation_h_to_padded_batched_f64
        )
        pack_tau_kernel = (
            _pack_articulation_tau_to_padded_batched_f32
            if self._block_chol_uses_f32
            else _pack_articulation_tau_to_padded_batched_f64
        )
        unpack_kernel = (
            _unpack_articulation_solution_from_padded_batched_f32
            if self._block_chol_uses_f32
            else _unpack_articulation_solution_from_padded_batched_f64
        )
        wp.launch(
            pack_h_kernel,
            dim=(model.articulation_count, max_rows, max_rows),
            inputs=[
                self.articulation_H_start,
                self.articulation_H_rows,
                self.articulation_dof_start,
                self.H,
                self.model_joint_armature,
                max_rows,
                self.block_chol_a,
            ],
            device=model.device,
        )
        wp.launch(
            pack_tau_kernel,
            dim=(model.articulation_count, max_rows),
            inputs=[
                self.articulation_H_rows,
                self.articulation_dof_start,
                self.joint_tau,
                max_rows,
                self.block_chol_rhs,
            ],
            device=model.device,
        )
        self.block_solver.factorize(self.block_chol_a, max_rows)
        self.block_solver.solve(self.block_chol_rhs, self.block_chol_x)
        wp.launch(
            unpack_kernel,
            dim=(model.articulation_count, max_rows),
            inputs=[
                self.articulation_H_rows,
                self.articulation_dof_start,
                max_rows,
                self.block_chol_x,
                self.joint_qdd_sap_solve,
            ],
            device=model.device,
        )

    def _assemble_sap_dynamics_matrix(self) -> None:
        model = self.model
        self._ensure_dynamics_matrix_allocated()
        wp.launch(
            _assemble_global_sap_dynamics_matrix_kernel,
            dim=(model.joint_dof_count, model.joint_dof_count),
            inputs=[
                self.H,
                self.model_joint_armature,
                self.dof_articulation_index,
                self.dof_articulation_local_index,
                self.articulation_dof_start,
                self.articulation_H_start,
                self.articulation_H_rows,
                self.dynamics_matrix_sap,
            ],
            device=model.device,
        )

    def _prepare_sap_boundary(self, state_in: State, control: Control) -> None:
        """Copy SAP-native state/control views into free-motion work buffers."""
        model = self.model

        if state_in.joint_qd is not self.joint_qd_sap_input:
            kernel = _copy_f64 if state_in.joint_qd.dtype == wp.float64 else _copy_f32_to_f64
            wp.launch(
                kernel,
                dim=model.joint_dof_count,
                inputs=[state_in.joint_qd, self.joint_qd_sap_input],
                device=model.device,
            )

        if control.joint_f is not self.joint_f_sap_input:
            kernel = _copy_f64 if control.joint_f.dtype == wp.float64 else _copy_f32_to_f64
            wp.launch(
                kernel,
                dim=model.joint_dof_count,
                inputs=[control.joint_f, self.joint_f_sap_input],
                device=model.device,
            )

        body_f = getattr(state_in, "body_f", None)
        if not int(model.body_count) or body_f is None:
            self.body_f_ext_s.zero_()
        elif body_f is self.body_f_ext_s:
            return
        elif body_f.dtype == wp.spatial_vectord:
            wp.copy(dest=self.body_f_ext_s, src=body_f)
        elif body_f.dtype == wp.spatial_vector:
            wp.launch(
                _copy_spatial_vector_to_spatial_vectord,
                dim=model.body_count,
                inputs=[body_f, self.body_f_ext_s],
                device=model.device,
            )
        else:
            raise TypeError(
                "SapState.body_f must be SAP body-origin forces with dtype "
                "wp.spatial_vectord or wp.spatial_vector."
            )

    def _compute_sap_core(
        self,
        state_in: State,
        dt: float,
        *,
        assemble_dynamics_matrix: bool,
    ) -> SapFreeMotionResult:
        """Run the SAP free-motion solve after boundary conversion.

        Inputs at this layer are the SAP-order `joint_qd_sap_input` and
        `joint_f_sap_input`.  Spatial buffers follow SAP's body-origin,
        world-expressed convention.
        """
        model = self.model
        self.body_f_s.zero_()
        self._launch_rigid_id(state_in)

        self.body_ft_s.zero_()
        wp.launch_tiled(
            self._rigid_tau_tiled,
            dim=model.articulation_count,
            block_dim=int(self.rigid_tile_size),
            inputs=[
                self.articulation_level_joint_index,
                int(self.max_articulation_level_count),
                int(self.max_articulation_level_width),
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_qd_start,
                model.joint_dof_dim,
                self.joint_f_sap_input,
                self.joint_S_s,
                self.body_q,
                self.body_f_s,
                self.body_f_ext_s,
                self.body_ft_s,
                self.joint_tau,
            ],
            device=model.device,
        )

        self._assemble_articulation_matrices()

        self.joint_qdd_sap_solve.zero_()
        if self._can_use_block_dynamics_solve():
            self._solve_articulation_accelerations_blocked()
        else:
            self._factor_dynamics_matrix()
            wp.launch(
                _eval_dense_solve_batched_f64,
                dim=model.articulation_count,
                inputs=[
                    self.articulation_H_start,
                    self.articulation_H_rows,
                    self.articulation_dof_start,
                    self.H,
                    self.L,
                    self.joint_tau,
                    self.joint_qdd_sap_solve,
                ],
                outputs=[self.joint_solve_tmp],
                device=model.device,
            )

        wp.launch(
            _assemble_sap_free_motion_outputs_kernel,
            dim=model.joint_count,
            inputs=[
                model.joint_qd_start,
                model.joint_dof_dim,
                self.joint_qd_sap_input,
                self.joint_qdd_sap_solve,
                float(dt),
                self.free_motion_joint_qd_sap,
                self.free_motion_joint_qdd_sap,
            ],
            device=model.device,
        )
        if assemble_dynamics_matrix:
            self._assemble_sap_dynamics_matrix()
        return self._result

    def compute(
        self,
        state_in: State,
        control: Control | None,
        dt: float,
        *,
        assemble_dynamics_matrix: bool = False,
    ) -> SapFreeMotionResult:
        """Compute SAP free motion from SAP-native state/control.

        The returned buffers satisfy `v_star = v0 + dt * vdot0` in SAP order.
        Set `assemble_dynamics_matrix=True` only for parity/debug code that
        needs the global dense `A = M + R` in SAP velocity order. Runtime
        contact/solver paths should use env-local matrix assembly instead.
        """
        if getattr(state_in, "requires_grad", False):
            raise NotImplementedError("SapFreeMotion does not support grad states.")

        if not isinstance(state_in, State):
            raise TypeError("SapFreeMotion.compute requires SapState; convert before entering SAP components.")
        if control is None or not isinstance(control, Control):
            raise TypeError("SapFreeMotion.compute requires SapControl; convert before entering SAP components.")

        self._prepare_sap_boundary(state_in, control)
        return self._compute_sap_core(
            state_in,
            dt,
            assemble_dynamics_matrix=assemble_dynamics_matrix,
        )

    def compute_articulation_free_motion(
        self,
        state_in: State,
        control: Control | None,
        dt: float,
        *,
        assemble_dynamics_matrix: bool = False,
    ):
        """Compatibility alias returning `(v_star, vdot, A_or_None)` in SAP order."""
        result = self.compute(
            state_in,
            control,
            dt,
            assemble_dynamics_matrix=assemble_dynamics_matrix,
        )
        return result.v_star, result.vdot, result.dynamics_matrix
