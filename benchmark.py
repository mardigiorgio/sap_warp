from __future__ import annotations

# ruff: noqa: E402

import argparse
from contextlib import contextmanager
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import warp as wp

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.collision.pipeline import SapCollisionPipeline
from sim.loader.control import load_sap_control_sequence
from sim.loader.scene import load_sap_scene, load_sap_scene_config
from sim.resources.collision_model import sap_collision_state_from_state
from sim.solver_sap import SolverSAP


DEFAULT_SCENE = ROOT / "assets" / "yaml" / "unitree_g1_usd.yaml"


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve()


@contextmanager
def _temporary_env(values: dict[str, str]) -> Any:
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _scene_simulation_config(scene_path: Path) -> dict[str, Any]:
    config = load_sap_scene_config(scene_path)
    simulation = config.get("simulation", {}) or {}
    return simulation if isinstance(simulation, dict) else {}


def _scene_default_shape_tau(scene_path: Path) -> float | None:
    config = load_sap_scene_config(scene_path)
    builder = config.get("builder", {}) or {}
    if not isinstance(builder, dict):
        return None
    defaults = builder.get("defaults", {}) or {}
    if not isinstance(defaults, dict):
        return None
    shape = defaults.get("shape", {}) or {}
    if not isinstance(shape, dict) or "tau" not in shape or shape["tau"] is None:
        return None
    return float(shape["tau"])


def _effective_dt(args: argparse.Namespace, simulation: dict[str, Any]) -> float:
    value = args.dt if args.dt is not None else simulation.get("dt", 0.003)
    dt = float(value)
    if dt <= 0.0:
        raise ValueError("--dt must be positive.")
    return dt


def _frame_count(args: argparse.Namespace, dt: float) -> int:
    if args.frames is not None:
        return max(int(args.frames), 1)
    return max(int(math.ceil(max(float(args.duration), 0.0) / dt)), 1)


def _scene_max_rigid_contact_per_env(scene_path: Path, simulation: dict[str, Any]) -> int:
    if "max_rigid_contact" not in simulation:
        raise ValueError(f"Scene file {scene_path} must define simulation.max_rigid_contact.")
    return max(int(simulation["max_rigid_contact"]), 1)


def _total_rigid_contact_capacity(max_rigid_contact_per_env: int, num_worlds: int) -> int:
    return max(int(max_rigid_contact_per_env), 1) * max(int(num_worlds), 1)


def _scene_num_worlds(args: argparse.Namespace, scene_path: Path, simulation: dict[str, Any]) -> int:
    if args.num_worlds is not None:
        return max(int(args.num_worlds), 1)
    if "num_worlds" not in simulation:
        raise ValueError(f"Scene file {scene_path} must define simulation.num_worlds.")
    return max(int(simulation["num_worlds"]), 1)


def _scene_solver_kwargs(scene_path: Path, simulation: dict[str, Any]) -> dict[str, Any]:
    raw_solver = simulation.get("solver", {}) or {}
    if not isinstance(raw_solver, dict):
        raise ValueError(f"Scene file {scene_path} simulation.solver must be a mapping.")
    return dict(raw_solver)


def _step_native(
    *,
    solver: SolverSAP,
    collision_pipeline: SapCollisionPipeline,
    state_0,
    state_1,
    control,
    contacts,
    dt: float,
):
    state_0.clear_forces()
    collision_pipeline.collide(sap_collision_state_from_state(state_0), contacts)
    solver.step(state_0, state_1, control, contacts, dt)
    return state_1, state_0


def _cuda_graph_supported(device) -> bool:
    return (
        bool(getattr(device, "is_cuda", False))
        and hasattr(wp, "ScopedCapture")
        and hasattr(wp, "capture_launch")
    )


def _capture_native_step_graph(
    *,
    solver: SolverSAP,
    collision_pipeline: SapCollisionPipeline,
    state_0,
    state_1,
    control,
    contacts,
    dt: float,
    device,
):
    if not _cuda_graph_supported(device):
        return None

    sim_time = solver.sim_time
    frame_id = solver.frame_id
    has_contact_solve_v_guess = solver._has_contact_solve_v_guess
    try:
        with wp.ScopedCapture(device=device) as capture:
            next_state, prev_state = _step_native(
                solver=solver,
                collision_pipeline=collision_pipeline,
                state_0=state_0,
                state_1=state_1,
                control=control,
                contacts=contacts,
                dt=dt,
            )
            prev_state.assign(next_state)
    except Exception:
        return None
    finally:
        solver.sim_time = sim_time
        solver.frame_id = frame_id
        solver._has_contact_solve_v_guess = has_contact_solve_v_guess

    return capture.graph


def run_native(args: argparse.Namespace) -> None:
    """Run the native SAP benchmark loop from parsed command-line arguments. It loads a scene, builds
    the collision pipeline and solver, optionally captures a CUDA graph, and prints timing
    statistics.
    """
    scene_path = _resolve_path(args.scene)
    simulation = _scene_simulation_config(scene_path)
    dt = _effective_dt(args, simulation)
    frames = _frame_count(args, dt)
    max_rigid_contact_per_env = _scene_max_rigid_contact_per_env(scene_path, simulation)
    num_worlds = _scene_num_worlds(args, scene_path, simulation)
    rigid_contact_capacity = _total_rigid_contact_capacity(max_rigid_contact_per_env, num_worlds)

    device = wp.get_device(args.device)
    loaded = load_sap_scene(
        scene_path,
        device=device,
        rigid_contact_max=rigid_contact_capacity,
        strict=True,
        num_worlds=num_worlds,
    )
    model = loaded.sap_model
    if int(model.joint_count) <= 0 or int(model.joint_dof_count) <= 0:
        raise ValueError("SolverSAP requires a scene with at least one joint DOF.")

    solver = SolverSAP(
        model,
        max_rigid_contact=max_rigid_contact_per_env,
        contact_tau_d=_scene_default_shape_tau(scene_path),
        **_scene_solver_kwargs(scene_path, simulation),
    )
    collision_pipeline = SapCollisionPipeline(loaded.collision_model, rigid_contact_max=rigid_contact_capacity)
    contacts = collision_pipeline.contacts()
    state_0 = loaded.sap_state
    state_1 = model.state()
    control = loaded.sap_control
    control_sequence = load_sap_control_sequence(scene_path, model, state_0, control, device=device)

    graph = None
    graph = _capture_native_step_graph(
        solver=solver,
        collision_pipeline=collision_pipeline,
        state_0=state_0,
        state_1=state_1,
        control=control,
        contacts=contacts,
        dt=dt,
        device=device,
    )
    wp.synchronize_device(device)
    t0 = time.time()
    if graph is not None:
        for frame_index in range(frames):
            if control_sequence is not None:
                control_sequence.apply(control, frame_index, dt)
            wp.capture_launch(graph)
            solver.sim_time += dt
            solver.frame_id += 1
            print("frame", solver.frame_id, "sim_time", solver.sim_time)
    else:
        for frame_index in range(frames):
            if control_sequence is not None:
                control_sequence.apply(control, frame_index, dt)
            state_0, state_1 = _step_native(
                solver=solver,
                collision_pipeline=collision_pipeline,
                state_0=state_0,
                state_1=state_1,
                control=control,
                contacts=contacts,
                dt=dt,
            )
            solver.sim_time += dt
            solver.frame_id += 1
            print("frame", solver.frame_id, "sim_time", solver.sim_time)
    wp.synchronize_device(device)
    t1 = time.time()
    fps = frames / (t1 - t0) if t1 > t0 else float("inf")
    realtime_ratio = (frames * dt) / (t1 - t0) if t1 > t0 else float("inf")

    print(
        f"scene={scene_path}: device={device} dt={dt:.6f} "
        f"frames={frames} num_worlds={num_worlds} "
        f"max_rigid_contact_per_env={max_rigid_contact_per_env} "
        f"rigid_contact_capacity={rigid_contact_capacity} "
        f"cuda_graph={graph is not None}",
        f"elapsed={t1 - t0:.3f}s fps={fps:.1f}",
        f"realtime_ratio={realtime_ratio:.3f}x",
        flush=True,
    )

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the benchmark entry point."""
    parser = argparse.ArgumentParser(description="Benchmark.")
    parser.add_argument("--scene", type=str, default=str(DEFAULT_SCENE), help="YAML scene file.")
    parser.add_argument("--duration", type=float, default=2.0, help="Simulation duration in seconds.")
    parser.add_argument("--frames", type=int, default=None, help="Number of simulation frames. Overrides --duration.")
    parser.add_argument("--dt", type=float, default=None, help="Simulation timestep. Defaults to simulation.dt.")
    parser.add_argument("--num-worlds", type=int, default=None, help="Number of replicated worlds.")
    parser.add_argument("--device", type=str, default=None, help="Warp device, for example cuda:0 or cpu.")
    return parser


def main() -> None:
    """Parse command-line options and dispatch the requested benchmark mode."""
    args = build_parser().parse_args()
    run_native(args)


if __name__ == "__main__":
    main()
