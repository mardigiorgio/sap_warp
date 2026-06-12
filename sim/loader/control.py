"""Control-sequence loading for SAP Warp scenes.

Source note: the SAP modifications in this module are based on Newton's
control/runtime code and adapted for compatibility with Newton-owned Warp
arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from sim.loader.scene import load_sap_scene_config
from sim.sap_kinematics import sap_eval_fk


@dataclass(frozen=True)
class SapControlSequence:
    """Runtime playback data loaded from a scene control file."""

    joint_target_pos: wp.array
    sample_dt: float
    frame_count: int
    dof_count: int
    initial_joint_q: wp.array | None = None
    initial_joint_qd: wp.array | None = None
    loop: bool = False

    def apply_initial_state(self, model: Any, state: Any) -> None:
        """Copy the sequence's first state sample into a SapState and refresh FK body poses."""
        wrote_state = False
        if self.initial_joint_q is not None:
            wp.copy(dest=state.joint_q, src=self.initial_joint_q)
            wrote_state = True
        if self.initial_joint_qd is not None:
            wp.copy(dest=state.joint_qd, src=self.initial_joint_qd)
            wrote_state = True
        if wrote_state:
            sap_eval_fk(model, state.joint_q, state.joint_qd, state)

    def apply(self, control: Any, frame_index: int, sim_dt: float) -> None:
        """Apply the zero-order-held target sample for a simulation frame."""
        if control.joint_target_pos is None:
            raise ValueError("Control sequence requires control.joint_target_pos.")
        sample = int(math.floor((frame_index * sim_dt) / self.sample_dt + 1.0e-9))
        if self.loop:
            sample %= self.frame_count
        else:
            sample = min(sample, self.frame_count - 1)
        wp.copy(
            dest=control.joint_target_pos,
            src=self.joint_target_pos,
            src_offset=sample * self.dof_count,
            count=self.dof_count,
        )


def load_sap_control_sequence(
    scene_path: str | Path,
    model: Any,
    state: Any,
    control: Any,
    *,
    device: Any = None,
) -> SapControlSequence | None:
    """Load and apply the optional ``control`` block referenced by a scene file.

    The helper mutates ``state`` to the control file's initial ``joint_q`` and
    ``joint_qd`` samples when present, mutates ``control`` to the first
    ``joint_target_pos`` sample, and returns a sequence object for per-frame
    playback. Scenes without a ``control.source`` return ``None``.
    """
    scene_path = Path(scene_path)
    control_cfg = _scene_control_config(scene_path)
    if not control_cfg:
        return None
    source = control_cfg.get("source")
    if source is None:
        return None

    control_path = _resolve_scene_relative_path(scene_path, source)
    with control_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"Control file {control_path} must contain a mapping.")
    if int(doc.get("schema_version", 1)) != 1:
        raise ValueError(f"Unsupported control schema_version={doc.get('schema_version')!r} in {control_path}.")

    target_section = doc.get("joint_target_pos")
    if not isinstance(target_section, dict):
        raise ValueError(f"Control file {control_path} must contain joint_target_pos.")
    target_values = _array_section_values(doc, "joint_target_pos", ndim=2)
    assert target_values is not None
    dof_count = int(model.joint_dof_count)
    target_values = _expand_matrix_width(target_values, dof_count, "joint_target_pos")
    if target_values.shape[0] <= 0:
        raise ValueError(f"Control file {control_path} contains no joint_target_pos frames.")

    sequence = SapControlSequence(
        joint_target_pos=wp.array(target_values.reshape(-1), dtype=wp.float32, device=device),
        sample_dt=_section_sample_dt(target_section),
        frame_count=int(target_values.shape[0]),
        dof_count=dof_count,
        initial_joint_q=_initial_state_array(doc, "joint_q", state.joint_q, device=device),
        initial_joint_qd=_initial_state_array(doc, "joint_qd", state.joint_qd, device=device),
        loop=bool(control_cfg.get("loop", doc.get("loop", False))),
    )
    sequence.apply_initial_state(model, state)
    sequence.apply(control, frame_index=0, sim_dt=sequence.sample_dt)
    return sequence


def _scene_control_config(scene_path: Path) -> dict[str, Any]:
    config = load_sap_scene_config(scene_path)
    raw_control = config.get("control", {}) or {}
    if not isinstance(raw_control, dict):
        raise ValueError(f"Scene file {scene_path} control must be a mapping.")
    return raw_control


def _resolve_scene_relative_path(scene_path: Path, value: str | Path) -> Path:
    resolved = Path(value).expanduser()
    if not resolved.is_absolute():
        resolved = scene_path.parent / resolved
    return resolved.resolve()


def _array_section_values(doc: dict[str, Any], section_name: str, *, ndim: int) -> np.ndarray | None:
    section = doc.get(section_name)
    if section is None:
        return None
    if not isinstance(section, dict) or "values" not in section:
        raise ValueError(f"Control file section {section_name!r} must contain values.")
    values = np.asarray(section["values"], dtype=np.float32)
    if values.ndim != ndim:
        raise ValueError(f"Control file section {section_name!r} must be {ndim}D, got shape {values.shape}.")
    return values


def _section_sample_dt(section: dict[str, Any]) -> float:
    if "dt" in section:
        sample_dt = float(section["dt"])
    else:
        times = np.asarray(section.get("times", []), dtype=np.float64)
        sample_dt = float(times[1] - times[0]) if len(times) > 1 else 0.0
    if sample_dt <= 0.0:
        raise ValueError("Control sequence dt must be positive.")
    return sample_dt


def _expand_matrix_width(values: np.ndarray, target_width: int, section_name: str) -> np.ndarray:
    if values.shape[1] == target_width:
        return np.ascontiguousarray(values, dtype=np.float32)
    if target_width % values.shape[1] != 0:
        raise ValueError(
            f"Control file section {section_name!r} width {values.shape[1]} does not match model width "
            f"{target_width}."
        )
    return np.ascontiguousarray(np.tile(values, (1, target_width // values.shape[1])), dtype=np.float32)


def _initial_state_array(doc: dict[str, Any], section_name: str, current_array: wp.array, *, device: Any) -> wp.array | None:
    values = _array_section_values(doc, section_name, ndim=2)
    if values is None:
        return None
    current = current_array.numpy().astype(np.float32, copy=False)
    expanded = _expand_state_row(values[0], current, section_name)
    return wp.array(expanded, dtype=wp.float32, device=device)


def _expand_state_row(row: np.ndarray, current: np.ndarray, section_name: str) -> np.ndarray:
    if row.shape[0] == current.shape[0]:
        return np.ascontiguousarray(row, dtype=np.float32)
    if current.shape[0] % row.shape[0] != 0:
        raise ValueError(
            f"Control file section {section_name!r} width {row.shape[0]} does not match model width "
            f"{current.shape[0]}."
        )
    repeat = current.shape[0] // row.shape[0]
    expanded = np.tile(row, repeat).reshape(repeat, row.shape[0])
    current_blocks = current.reshape(repeat, row.shape[0])
    if section_name == "joint_q" and row.shape[0] >= 7:
        root_offsets = current_blocks[:, :3] - current_blocks[0:1, :3]
        expanded[:, :3] += root_offsets
    return np.ascontiguousarray(expanded.reshape(-1), dtype=np.float32)


__all__ = ["SapControlSequence", "load_sap_control_sequence"]
