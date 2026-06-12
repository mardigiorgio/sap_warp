from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import viser
import warp as wp

from sim.sap_kinematics import sap_eval_fk


def _as_numpy(values: Any, *, dtype=np.float64) -> np.ndarray:
    if hasattr(values, "numpy"):
        values = values.numpy()
    return np.asarray(values, dtype=dtype)


def _normalize_xyzw(q: Any) -> np.ndarray:
    quat = np.asarray(q, dtype=np.float64).reshape(4).copy()
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12 or not np.isfinite(norm):
        return np.array((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    return quat / norm


def _normalize_wxyz(q: Any) -> np.ndarray:
    quat = np.asarray(q, dtype=np.float64).reshape(4).copy()
    norm = float(np.linalg.norm(quat))
    if norm <= 1.0e-12 or not np.isfinite(norm):
        return np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    return quat / norm


def _xyzw_to_wxyz(q: Any) -> np.ndarray:
    x, y, z, w = _normalize_xyzw(q)
    return np.array((w, x, y, z), dtype=np.float64)


def _wxyz_to_xyzw(q: Any) -> np.ndarray:
    w, x, y, z = _normalize_wxyz(q)
    return np.array((x, y, z, w), dtype=np.float64)


def _quat_conjugate_xyzw(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).reshape(4).copy()
    out[:3] *= -1.0
    return out


def _quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = _normalize_xyzw(a)
    bx, by, bz, bw = _normalize_xyzw(b)
    return _normalize_xyzw(
        np.array(
            (
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ),
            dtype=np.float64,
        )
    )


def _pose_error(current_pose: np.ndarray, target_position: np.ndarray, target_wxyz: np.ndarray) -> np.ndarray:
    current_position = current_pose[:3]
    current_xyzw = _normalize_xyzw(current_pose[3:7])
    target_xyzw = _wxyz_to_xyzw(target_wxyz)
    q_err = _quat_multiply_xyzw(target_xyzw, _quat_conjugate_xyzw(current_xyzw))
    if q_err[3] < 0.0:
        q_err *= -1.0
    return np.concatenate((target_position - current_position, 2.0 * q_err[:3]), axis=0)


def _find_label_index(labels: list[str], *tails: str) -> int | None:
    for tail in tails:
        for index, label in enumerate(labels):
            label_tail = str(label).rsplit("/", 1)[-1]
            if label_tail == tail or str(label) == tail:
                return index
    return None


def _find_label_indices(labels: list[str], *tails: str) -> list[int]:
    matches: list[int] = []
    for index, label in enumerate(labels):
        label_text = str(label)
        label_tail = label_text.rsplit("/", 1)[-1]
        if any(label_tail == tail or label_text == tail for tail in tails):
            matches.append(index)
    return matches


@dataclass(slots=True)
class _ArmTarget:
    name: str
    joint_indices: np.ndarray
    joint_q_indices: np.ndarray
    joint_dof_indices: np.ndarray
    ee_body_index: int
    lower: np.ndarray
    upper: np.ndarray
    target_position: np.ndarray
    target_wxyz: np.ndarray
    dirty: bool = False
    last_status: str = "idle"
    last_solve_ms: float = 0.0


class ViewerMouseControl:
    """EE pose mouse control only.

    Body force dragging is intentionally disabled. This class only creates Viser
    transform controls for recognized robot-arm end effectors and writes joint
    position targets when the handle moves.
    """

    def __init__(self, server: viser.ViserServer, model: Any, state: Any, control: Any, _render_state: Any) -> None:
        self.server = server
        self.model = model
        self.state = state
        self.control = control
        self.enabled = True
        self.status = server.gui.add_text("Mouse Control", "EE pose only; force drag disabled", disabled=True)
        self._scratch_state = model.state(requires_grad=False)
        self._arms = self._detect_arms(state)
        self._handles = []
        if not self._arms:
            self.status.value = "No supported EE targets found; force drag disabled"
            return
        with server.gui.add_folder("EE Targets", expand_by_default=True):
            for arm in self._arms:
                self._add_target_handle(arm)
        self.refresh_status()

    def _detect_arms(self, state: Any) -> list[_ArmTarget]:
        joint_labels = [str(label) for label in (getattr(self.model, "joint_label", None) or ())]
        body_labels = [str(label) for label in (getattr(self.model, "body_label", None) or ())]
        if not joint_labels or not body_labels:
            return []

        specs: list[tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = [
            (
                "right_piper",
                tuple(f"right_joint{i}" for i in range(1, 7)),
                ("right_tool_tip_link", "right_gripper_base", "right_link6"),
                ("right_base_link",),
            ),
            (
                "left_piper",
                tuple(f"left_joint{i}" for i in range(1, 7)),
                ("left_tool_tip_link", "left_gripper_base", "left_link6"),
                ("left_base_link",),
            ),
            (
                "franka",
                tuple(f"fr3_joint{i}" for i in range(1, 8)),
                ("fr3_hand_tcp", "fr3_hand", "fr3_link8"),
                ("fr3_link0", "base"),
            ),
            (
                "ur10",
                ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"),
                ("ee_link", "wrist_3_link"),
                ("base_link",),
            ),
        ]
        joint_q_start = _as_numpy(self.model.joint_q_start, dtype=np.int32).reshape(-1)
        joint_qd_start = _as_numpy(self.model.joint_qd_start, dtype=np.int32).reshape(-1)
        joint_q = _as_numpy(state.joint_q, dtype=np.float64).reshape(-1)
        body_q = _as_numpy(state.body_q, dtype=np.float64).reshape(-1, 7)
        lower_all = _as_numpy(self.model.joint_limit_lower, dtype=np.float64).reshape(-1)
        upper_all = _as_numpy(self.model.joint_limit_upper, dtype=np.float64).reshape(-1)

        arms: list[_ArmTarget] = []
        used_names: set[str] = set()
        for name, joint_tails, ee_tails, base_tails in specs:
            joint_matches = [_find_label_indices(joint_labels, tail) for tail in joint_tails]
            if any(not matches for matches in joint_matches):
                continue
            ee_matches = _find_label_indices(body_labels, *ee_tails)
            if not ee_matches:
                continue
            occurrences = min([len(matches) for matches in joint_matches] + [len(ee_matches)])
            base_matches = _find_label_indices(body_labels, *base_tails)
            for occurrence in range(occurrences):
                joint_indices_np = np.asarray([int(matches[occurrence]) for matches in joint_matches], dtype=np.int32)
                joint_q_indices = joint_q_start[joint_indices_np].astype(np.int32, copy=True)
                joint_dof_indices = joint_qd_start[joint_indices_np].astype(np.int32, copy=True)
                if len(np.unique(joint_q_indices)) != len(joint_q_indices) or len(np.unique(joint_dof_indices)) != len(joint_dof_indices):
                    continue
                lower = np.full(len(joint_dof_indices), -2.0 * np.pi, dtype=np.float64)
                upper = np.full(len(joint_dof_indices), 2.0 * np.pi, dtype=np.float64)
                for local_index, dof_index in enumerate(joint_dof_indices):
                    if 0 <= int(dof_index) < lower_all.size and np.isfinite(lower_all[int(dof_index)]):
                        lower[local_index] = lower_all[int(dof_index)]
                    if 0 <= int(dof_index) < upper_all.size and np.isfinite(upper_all[int(dof_index)]):
                        upper[local_index] = upper_all[int(dof_index)]
                arm_name = name
                if occurrences > 1:
                    arm_name = f"{name}_{occurrence}"
                    if name == "right_piper" and occurrence < len(base_matches):
                        base_y = float(body_q[int(base_matches[occurrence]), 1])
                        arm_name = "left_piper" if base_y >= 0.0 else "right_piper"
                if arm_name in used_names:
                    arm_name = f"{arm_name}_{occurrence}"
                used_names.add(arm_name)
                seed = np.clip(joint_q[joint_q_indices], lower, upper)
                pose = body_q[int(ee_matches[occurrence])]
                arms.append(
                    _ArmTarget(
                        name=arm_name,
                        joint_indices=joint_indices_np,
                        joint_q_indices=joint_q_indices,
                        joint_dof_indices=joint_dof_indices,
                        ee_body_index=int(ee_matches[occurrence]),
                        lower=lower,
                        upper=upper,
                        target_position=pose[:3].astype(np.float64, copy=True),
                        target_wxyz=_xyzw_to_wxyz(pose[3:7]),
                        last_status=f"ready q={np.array2string(seed, precision=2)}",
                    )
                )
        return arms

    def _add_target_handle(self, arm: _ArmTarget) -> None:
        handle = self.server.scene.add_transform_controls(
            f"/sap/mouse_control/{arm.name}_ee",
            scale=0.25,
            line_width=3.0,
            position=arm.target_position,
            wxyz=arm.target_wxyz,
        )

        @handle.on_update
        def _(_) -> None:
            arm.target_position = np.asarray(handle.position, dtype=np.float64).reshape(3).copy()
            arm.target_wxyz = _normalize_wxyz(handle.wxyz)
            arm.dirty = True

        self._handles.append(handle)

    def reset(self, state: Any) -> None:
        body_q = _as_numpy(state.body_q, dtype=np.float64).reshape(-1, 7)
        for arm, handle in zip(self._arms, self._handles):
            pose = body_q[arm.ee_body_index]
            arm.target_position = pose[:3].astype(np.float64, copy=True)
            arm.target_wxyz = _xyzw_to_wxyz(pose[3:7])
            arm.dirty = False
            handle.position = arm.target_position
            handle.wxyz = arm.target_wxyz
            arm.last_status = "idle"
        self.refresh_status()

    def apply_targets(self, state: Any, control: Any) -> bool:
        if not self.enabled or not self._arms:
            return False
        dirty = [arm for arm in self._arms if arm.dirty]
        if not dirty:
            return False
        target_pos = _as_numpy(control.joint_target_pos, dtype=np.float32).reshape(-1).copy()
        target_vel = _as_numpy(control.joint_target_vel, dtype=np.float32).reshape(-1).copy()
        changed = False
        for arm in dirty:
            solved = self._solve_arm(state, arm)
            arm.dirty = False
            if solved is None:
                continue
            for dof_index, value in zip(arm.joint_dof_indices, solved):
                if 0 <= int(dof_index) < target_pos.size:
                    target_pos[int(dof_index)] = np.float32(value)
                    if int(dof_index) < target_vel.size:
                        target_vel[int(dof_index)] = np.float32(0.0)
            changed = True
        if changed:
            control.joint_target_pos.assign(target_pos)
            control.joint_target_vel.assign(target_vel)
            self.refresh_status()
        return changed

    def _solve_arm(self, state: Any, arm: _ArmTarget) -> np.ndarray | None:
        start = time.perf_counter()
        joint_q = _as_numpy(state.joint_q, dtype=np.float32).reshape(-1).copy()
        joint_qd = _as_numpy(state.joint_qd, dtype=np.float32).reshape(-1).copy()
        q = np.asarray(joint_q[arm.joint_q_indices], dtype=np.float64)
        q = np.clip(q, arm.lower, arm.upper)

        best_error = np.zeros(6, dtype=np.float64)
        for iteration in range(16):
            pose = self._fk_pose(joint_q, joint_qd, arm, q)
            error = _pose_error(pose, arm.target_position, arm.target_wxyz)
            best_error = error
            if float(np.linalg.norm(error[:3])) < 0.003 and float(np.linalg.norm(error[3:])) < 0.05:
                break

            jacobian = np.zeros((6, len(q)), dtype=np.float64)
            eps = 1.0e-4
            for col in range(len(q)):
                q_plus = q.copy()
                q_plus[col] += eps
                pose_plus = self._fk_pose(joint_q, joint_qd, arm, q_plus)
                jacobian[:, col] = (_pose_error(pose, pose_plus[:3], _xyzw_to_wxyz(pose_plus[3:7])) / eps)

            damping = 5.0e-2
            lhs = jacobian @ jacobian.T + damping * damping * np.eye(6, dtype=np.float64)
            try:
                step = jacobian.T @ np.linalg.solve(lhs, error)
            except np.linalg.LinAlgError:
                arm.last_status = "IK solve failed"
                return None
            q = np.clip(q + np.clip(step, -0.12, 0.12), arm.lower, arm.upper)

        arm.last_solve_ms = (time.perf_counter() - start) * 1000.0
        arm.last_status = f"{np.linalg.norm(best_error[:3]) * 1000.0:.1f} mm, {arm.last_solve_ms:.1f} ms"
        return q

    def _fk_pose(self, base_joint_q: np.ndarray, base_joint_qd: np.ndarray, arm: _ArmTarget, q: np.ndarray) -> np.ndarray:
        scratch_q = base_joint_q.copy()
        scratch_q[arm.joint_q_indices] = np.asarray(q, dtype=np.float32)
        scratch_qd = base_joint_qd.copy()
        scratch_qd[arm.joint_dof_indices] = 0.0
        self._scratch_state.joint_q.assign(scratch_q)
        self._scratch_state.joint_qd.assign(scratch_qd)
        sap_eval_fk(self.model, self._scratch_state.joint_q, self._scratch_state.joint_qd, self._scratch_state)
        wp.synchronize_device(self.model.device)
        return _as_numpy(self._scratch_state.body_q, dtype=np.float64).reshape(-1, 7)[arm.ee_body_index].copy()

    def refresh_status(self) -> None:
        if not self._arms:
            self.status.value = "No supported EE targets found; force drag disabled"
            return
        lines = ["EE pose enabled; force drag disabled"]
        lines.extend(f"{arm.name}: {arm.last_status}" for arm in self._arms)
        self.status.value = "\n".join(lines)
