from __future__ import annotations

# ruff: noqa: E402

import argparse
import asyncio
import base64
from dataclasses import dataclass
import io
import json
import logging
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Sequence
import urllib.request

import numpy as np
import viser
from viser import _messages as viser_messages
import warp as wp
import websockets
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim.collision.pipeline import SapCollisionPipeline
from sim.collision.flags import SapShapeFlags
from sim.collision.types import SapGeoType
from sim.loader.control import load_sap_control_sequence
from sim.loader.scene import load_sap_scene, load_sap_scene_config
from sim.resources.collision_model import sap_collision_state_from_state
from sim.solver_sap import SolverSAP
from viewer.viser.mouse_control import ViewerMouseControl


DEFAULT_SCENE = ROOT / "assets" / "yaml" / "unitree_g1_usd.yaml"
_TEXTURE_COLOR_CACHE: dict[str, tuple[int, int, int] | None] = {}
logging.getLogger("websockets.server").setLevel(logging.CRITICAL)


def _find_free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _find_chrome_binary() -> str:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path is not None:
            return path
    raise RuntimeError("MP4 recording requires Google Chrome or Chromium on PATH.")


def _wait_for_cdp_page(debug_port: int, timeout: float = 10.0) -> str:
    deadline = time.perf_counter() + timeout
    last_error: Exception | None = None
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{debug_port}/json/list", timeout=0.5) as response:
                pages = json.loads(response.read().decode("utf-8"))
            for page in pages:
                ws_url = page.get("webSocketDebuggerUrl")
                if page.get("type") == "page" and ws_url:
                    return str(ws_url)
        except Exception as exc:
            last_error = exc
        time.sleep(0.05)
    raise RuntimeError(f"Timed out waiting for Chrome DevTools page: {last_error}")


class _ChromeMp4Recorder:
    def __init__(
        self,
        *,
        url: str,
        output: Path,
        fps: float,
        width: int,
        height: int,
        webgl_backend: str,
    ) -> None:
        self.url = str(url)
        self.output = Path(output)
        self.fps = max(float(fps), 1.0)
        self.width = max(int(width), 64)
        self.height = max(int(height), 64)
        self.webgl_backend = str(webgl_backend)
        self.webgl_renderer = "unknown"
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._chrome: subprocess.Popen | None = None
        self._tmpdir: tempfile.TemporaryDirectory | None = None

    def start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._thread_main, name="viser-mp4-recorder", daemon=True)
        self._thread.start()
        if not self._ready_event.wait(timeout=15.0):
            self.stop()
            raise RuntimeError("Timed out starting MP4 recorder.")
        if self._error is not None:
            self.stop()
            raise RuntimeError(f"Failed to start MP4 recorder: {self._error}") from self._error

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        if self._chrome is not None and self._chrome.poll() is None:
            self._chrome.terminate()
            try:
                self._chrome.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._chrome.kill()
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._record())
        except BaseException as exc:
            self._error = exc
            self._ready_event.set()

    async def _record(self) -> None:
        try:
            import imageio_ffmpeg
        except ImportError as exc:
            raise RuntimeError("MP4 recording requires dependency imageio-ffmpeg. Run `uv sync`.") from exc

        chrome = _find_chrome_binary()
        debug_port = _find_free_tcp_port()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="sap-warp-chrome-")
        if self.webgl_backend == "swiftshader":
            webgl_flags = [
                "--enable-webgl",
                "--ignore-gpu-blocklist",
                "--enable-unsafe-swiftshader",
                "--disable-gpu-sandbox",
                "--use-gl=angle",
                "--use-angle=swiftshader",
            ]
        else:
            webgl_flags = [
                "--enable-webgl",
                "--ignore-gpu-blocklist",
                "--enable-gpu-rasterization",
                "--use-angle=vulkan",
            ]
        self._chrome = subprocess.Popen(
            [
                chrome,
                "--headless=new",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-dev-shm-usage",
                *webgl_flags,
                f"--remote-debugging-port={debug_port}",
                f"--user-data-dir={self._tmpdir.name}",
                f"--window-size={self.width},{self.height}",
                self.url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        ws_url = _wait_for_cdp_page(debug_port)
        writer = imageio_ffmpeg.write_frames(
            str(self.output),
            size=(self.width, self.height),
            fps=self.fps,
            codec="libx264",
            macro_block_size=1,
            ffmpeg_log_level="error",
            output_params=["-movflags", "+faststart"],
        )
        writer.send(None)
        frame_interval = 1.0 / self.fps
        next_frame_time = time.perf_counter()

        try:
            async with websockets.connect(ws_url, max_size=None) as ws:
                command_id = 0

                async def command(method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
                    nonlocal command_id
                    command_id += 1
                    await ws.send(json.dumps({"id": command_id, "method": method, "params": params or {}}))
                    while True:
                        message = json.loads(await ws.recv())
                        if message.get("id") == command_id:
                            if "error" in message:
                                raise RuntimeError(f"Chrome DevTools {method} failed: {message['error']}")
                            return dict(message.get("result", {}))

                await command("Page.enable")
                await command(
                    "Emulation.setDeviceMetricsOverride",
                    {"width": self.width, "height": self.height, "deviceScaleFactor": 1, "mobile": False},
                )
                await command("Page.navigate", {"url": self.url})
                await asyncio.sleep(2.0)
                webgl_status = await command(
                    "Runtime.evaluate",
                    {
                        "returnByValue": True,
                        "expression": """
(() => {
  const canvas = document.createElement("canvas");
  const gl2 = canvas.getContext("webgl2");
  const gl = gl2 || canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
  if (!gl) return {ok: false, webgl2: false, renderer: null};
  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  const renderer = debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  return {ok: true, webgl2: !!gl2, renderer};
})()
""",
                    },
                )
                webgl_value = webgl_status.get("result", {}).get("value", {})
                if not webgl_value.get("ok"):
                    raise RuntimeError(
                        "Headless Chrome could not create a WebGL context. "
                        "Try --record-webgl swiftshader, or run without --record-mp4."
                    )
                self.webgl_renderer = str(webgl_value.get("renderer") or "unknown")
                renderer_lower = self.webgl_renderer.lower()
                if self.webgl_backend == "gpu" and ("swiftshader" in renderer_lower or "software" in renderer_lower):
                    raise RuntimeError(
                        "Requested GPU WebGL recording, but Chrome fell back to software rendering "
                        f"({self.webgl_renderer}). Use --record-webgl swiftshader to allow software recording."
                    )
                canvas_ready_deadline = time.perf_counter() + 10.0
                while time.perf_counter() < canvas_ready_deadline:
                    canvas_status = await command(
                        "Runtime.evaluate",
                        {
                            "returnByValue": True,
                            "expression": """
(() => {
  const canvas = document.querySelector("canvas");
  if (!canvas) return {ok: false, width: 0, height: 0};
  return {ok: canvas.width > 0 && canvas.height > 0, width: canvas.width, height: canvas.height};
})()
""",
                        },
                    )
                    canvas_value = canvas_status.get("result", {}).get("value", {})
                    if canvas_value.get("ok"):
                        break
                    await asyncio.sleep(0.1)
                else:
                    raise RuntimeError("Timed out waiting for the Viser WebGL canvas before recording.")
                screenshot = await command(
                    "Page.captureScreenshot",
                    {
                        "format": "png",
                        "fromSurface": True,
                        "clip": {"x": 0, "y": 0, "width": self.width, "height": self.height, "scale": 1},
                    },
                )
                png = base64.b64decode(str(screenshot["data"]))
                with Image.open(io.BytesIO(png)) as image:
                    writer.send(np.asarray(image.convert("RGB"), dtype=np.uint8))
                self._ready_event.set()

                while not self._stop_event.is_set():
                    now = time.perf_counter()
                    if now < next_frame_time:
                        await asyncio.sleep(min(next_frame_time - now, 0.01))
                        continue
                    screenshot = await command(
                        "Page.captureScreenshot",
                        {
                            "format": "png",
                            "fromSurface": True,
                            "clip": {"x": 0, "y": 0, "width": self.width, "height": self.height, "scale": 1},
                        },
                    )
                    png = base64.b64decode(str(screenshot["data"]))
                    with Image.open(io.BytesIO(png)) as image:
                        frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
                    writer.send(frame)
                    next_frame_time += frame_interval
        finally:
            writer.close()


@dataclass
class ViewerShapeBatch:
    """Viewer-side batch of shape handles and geometry metadata used to update repeated instances
    efficiently.
    """
    handle: Any
    body_ids: np.ndarray
    local_transforms: np.ndarray
    dynamic_indices: np.ndarray
    dynamic_body_ids: np.ndarray
    local_positions: np.ndarray
    local_q_xyzw: np.ndarray
    positions: np.ndarray
    q_xyzw: np.ndarray
    wxyzs: np.ndarray


@dataclass
class ViewerRenderState:
    """Mutable viewer render state that tracks handles, colors, transforms, and render bookkeeping."""
    shape_batches: tuple[ViewerShapeBatch, ...]
    visible_shape_count: int
    visual_instance_count: int


@dataclass
class ViewerControls:
    """Interactive controls exposed by the Viser viewer."""
    running: dict[str, bool]
    status_text: Any
    perf_text: Any
    reset_requested: dict[str, bool]


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved.resolve()


def _effective_dt(args: argparse.Namespace, simulation: dict[str, Any]) -> float:
    value = args.dt if args.dt is not None else simulation.get("dt", 0.003)
    dt = float(value)
    if dt <= 0.0:
        raise ValueError("--dt must be positive.")
    return dt


def _frame_limit(args: argparse.Namespace, dt: float) -> int | None:
    if args.frames is not None:
        return max(int(args.frames), 1)
    if args.duration is None or math.isinf(float(args.duration)):
        return None
    return max(int(math.ceil(max(float(args.duration), 0.0) / dt)), 1)


def _effective_substeps_per_frame(args: argparse.Namespace, dt: float) -> int:
    if args.substeps_per_frame is not None:
        return max(int(args.substeps_per_frame), 1)
    return 1


def _scene_max_rigid_contact(scene_path: Path, simulation: dict[str, Any]) -> int:
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


def _nested_config_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


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


def _assign_control(dst: Any, src: Any) -> None:
    for name in ("joint_f", "joint_target_pos", "joint_target_vel", "joint_act"):
        dst_array = getattr(dst, name, None)
        src_array = getattr(src, name, None)
        if dst_array is None and src_array is None:
            continue
        if dst_array is None or src_array is None:
            raise ValueError(f"SapControl assign mismatch for {name}")
        wp.copy(dest=dst_array, src=src_array)


def _cuda_graph_supported(device: Any) -> bool:
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
    device: Any,
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
    except Exception as exc:
        print(f"CUDA graph capture failed; falling back to direct stepping: {exc}", flush=True)
        return None
    finally:
        solver.sim_time = sim_time
        solver.frame_id = frame_id
        solver._has_contact_solve_v_guess = has_contact_solve_v_guess

    return capture.graph


def _launch_step_graph(graph: Any) -> None:
    wp.capture_launch(graph)


def _as_numpy(arr: Any) -> np.ndarray:
    return np.asarray(arr.numpy())


def _quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q = _normalize_quat_xyzw(q_xyzw)
    return np.asarray([q[3], q[0], q[1], q[2]], dtype=np.float64)


def _normalize_quat_xyzw(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm <= 0.0:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def _quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.asarray(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def _quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = _normalize_quat_xyzw(q)
    q_vec = np.asarray(q[:3], dtype=np.float64)
    vec = np.asarray(v, dtype=np.float64)
    return vec + 2.0 * np.cross(q_vec, np.cross(q_vec, vec) + q[3] * vec)


def _quat_xyzw_to_wxyz_batch_inplace(q_xyzw: np.ndarray, out: np.ndarray) -> None:
    norms = np.linalg.norm(q_xyzw, axis=1)
    valid = norms > 0.0
    out[valid, 0] = q_xyzw[valid, 3] / norms[valid]
    out[valid, 1] = q_xyzw[valid, 0] / norms[valid]
    out[valid, 2] = q_xyzw[valid, 1] / norms[valid]
    out[valid, 3] = q_xyzw[valid, 2] / norms[valid]
    if np.any(~valid):
        out[~valid] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _compose_transform(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    parent_p = parent[:3]
    parent_q = parent[3:7]
    child_p = child[:3]
    child_q = child[3:7]
    out = np.empty(7, dtype=np.float64)
    out[:3] = parent_p + _quat_rotate_xyzw(parent_q, child_p)
    out[3:7] = _quat_multiply_xyzw(parent_q, child_q)
    return out


def _make_viewer_shape_batch(
    handle: Any,
    body_ids: np.ndarray,
    local_transforms: np.ndarray,
    body_tfs: np.ndarray,
) -> ViewerShapeBatch:
    body_ids = np.asarray(body_ids, dtype=np.int32)
    local_transforms = np.asarray(local_transforms, dtype=np.float64).reshape(-1, 7)
    local_positions = local_transforms[:, :3].astype(np.float32, copy=True)
    local_q_xyzw = local_transforms[:, 3:7].astype(np.float32, copy=True)
    positions = local_positions.copy()
    q_xyzw = local_q_xyzw.copy()
    wxyzs = np.empty((body_ids.shape[0], 4), dtype=np.float32)
    dynamic_indices = np.flatnonzero(body_ids >= 0).astype(np.int32)
    batch = ViewerShapeBatch(
        handle=handle,
        body_ids=body_ids,
        local_transforms=local_transforms,
        dynamic_indices=dynamic_indices,
        dynamic_body_ids=body_ids[dynamic_indices],
        local_positions=local_positions,
        local_q_xyzw=local_q_xyzw,
        positions=positions,
        q_xyzw=q_xyzw,
        wxyzs=wxyzs,
    )
    _update_shape_batch_buffers(body_tfs, batch, force=True)
    return batch


def _write_dynamic_instance_transforms(
    parent: np.ndarray,
    local_positions: np.ndarray,
    local_q_xyzw: np.ndarray,
    out_positions: np.ndarray,
    out_q_xyzw: np.ndarray,
) -> None:
    ax = parent[:, 3]
    ay = parent[:, 4]
    az = parent[:, 5]
    aw = parent[:, 6]
    vx = local_positions[:, 0]
    vy = local_positions[:, 1]
    vz = local_positions[:, 2]

    tx = 2.0 * (ay * vz - az * vy)
    ty = 2.0 * (az * vx - ax * vz)
    tz = 2.0 * (ax * vy - ay * vx)
    out_positions[:, 0] = parent[:, 0] + vx + aw * tx + ay * tz - az * ty
    out_positions[:, 1] = parent[:, 1] + vy + aw * ty + az * tx - ax * tz
    out_positions[:, 2] = parent[:, 2] + vz + aw * tz + ax * ty - ay * tx

    bx = local_q_xyzw[:, 0]
    by = local_q_xyzw[:, 1]
    bz = local_q_xyzw[:, 2]
    bw = local_q_xyzw[:, 3]
    out_q_xyzw[:, 0] = aw * bx + ax * bw + ay * bz - az * by
    out_q_xyzw[:, 1] = aw * by - ax * bz + ay * bw + az * bx
    out_q_xyzw[:, 2] = aw * bz + ax * by - ay * bx + az * bw
    out_q_xyzw[:, 3] = aw * bw - ax * bx - ay * by - az * bz


def _update_shape_batch_buffers(body_tfs: np.ndarray, batch: ViewerShapeBatch, *, force: bool = False) -> bool:
    if batch.dynamic_indices.size == 0:
        if force:
            _quat_xyzw_to_wxyz_batch_inplace(batch.q_xyzw, batch.wxyzs)
        return False

    parent = body_tfs[batch.dynamic_body_ids]
    if batch.dynamic_indices.shape[0] == batch.body_ids.shape[0]:
        _write_dynamic_instance_transforms(
            parent,
            batch.local_positions,
            batch.local_q_xyzw,
            batch.positions,
            batch.q_xyzw,
        )
    else:
        indices = batch.dynamic_indices
        dynamic_positions = np.empty((indices.shape[0], 3), dtype=np.float32)
        dynamic_q_xyzw = np.empty((indices.shape[0], 4), dtype=np.float32)
        _write_dynamic_instance_transforms(
            parent,
            batch.local_positions[indices],
            batch.local_q_xyzw[indices],
            dynamic_positions,
            dynamic_q_xyzw,
        )
        batch.positions[indices] = dynamic_positions
        batch.q_xyzw[indices] = dynamic_q_xyzw
    _quat_xyzw_to_wxyz_batch_inplace(batch.q_xyzw, batch.wxyzs)
    return True


def _queue_shape_batch_pose_update(batch: ViewerShapeBatch) -> None:
    batch.handle._impl.props.batched_positions[:] = batch.positions
    batch.handle._impl.props.batched_wxyzs[:] = batch.wxyzs
    batch.handle._impl.api._websock_interface.queue_message(
        viser_messages.SceneNodeUpdateMessage(
            batch.handle._impl.name,
            {
                "batched_positions": batch.positions,
                "batched_wxyzs": batch.wxyzs,
            },
        )
    )


def _shape_world_transforms(model: Any, state: Any) -> np.ndarray:
    body_q = _as_numpy(state.body_q).reshape(-1, 7).astype(np.float64)
    shape_transform = _as_numpy(model.shape_transform).reshape(-1, 7).astype(np.float64)
    shape_body = _as_numpy(model.shape_body).reshape(-1).astype(np.int32)

    world = np.empty_like(shape_transform)
    for shape_id, body_id in enumerate(shape_body):
        if body_id >= 0:
            world[shape_id] = _compose_transform(body_q[int(body_id)], shape_transform[shape_id])
        else:
            world[shape_id] = shape_transform[shape_id]
    return world


def _body_transforms(state: Any) -> np.ndarray:
    return _as_numpy(state.body_q).reshape(-1, 7).astype(np.float32, copy=False)


def _box_mesh(scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = [float(v) for v in scale]
    vertices = np.asarray(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _ellipsoid_mesh(scale: np.ndarray, segments: int = 16, rings: int = 8) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    rx, ry, rz = [max(float(v), 1.0e-4) for v in scale]
    for i in range(rings + 1):
        theta = math.pi * float(i) / float(rings)
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        for j in range(segments):
            phi = 2.0 * math.pi * float(j) / float(segments)
            vertices.append((rx * sin_t * math.cos(phi), ry * sin_t * math.sin(phi), rz * cos_t))
    for i in range(rings):
        for j in range(segments):
            a = i * segments + j
            b = i * segments + (j + 1) % segments
            c = (i + 1) * segments + j
            d = (i + 1) * segments + (j + 1) % segments
            faces.append((a, c, b))
            faces.append((b, c, d))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _cylinder_mesh(radius: float, half_height: float, segments: int = 20) -> tuple[np.ndarray, np.ndarray]:
    radius = max(float(radius), 1.0e-4)
    half_height = max(float(half_height), 1.0e-4)
    vertices: list[tuple[float, float, float]] = []
    for z in (-half_height, half_height):
        for j in range(segments):
            phi = 2.0 * math.pi * float(j) / float(segments)
            vertices.append((radius * math.cos(phi), radius * math.sin(phi), z))
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, -half_height))
    top_center = len(vertices)
    vertices.append((0.0, 0.0, half_height))

    faces: list[tuple[int, int, int]] = []
    for j in range(segments):
        b0 = j
        b1 = (j + 1) % segments
        t0 = segments + j
        t1 = segments + (j + 1) % segments
        faces.append((b0, b1, t1))
        faces.append((b0, t1, t0))
        faces.append((bottom_center, b1, b0))
        faces.append((top_center, t0, t1))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _cone_mesh(radius: float, half_height: float, segments: int = 20) -> tuple[np.ndarray, np.ndarray]:
    radius = max(float(radius), 1.0e-4)
    half_height = max(float(half_height), 1.0e-4)
    vertices: list[tuple[float, float, float]] = []
    for j in range(segments):
        phi = 2.0 * math.pi * float(j) / float(segments)
        vertices.append((radius * math.cos(phi), radius * math.sin(phi), -half_height))
    apex = len(vertices)
    vertices.append((0.0, 0.0, half_height))
    base_center = len(vertices)
    vertices.append((0.0, 0.0, -half_height))

    faces: list[tuple[int, int, int]] = []
    for j in range(segments):
        j_next = (j + 1) % segments
        faces.append((j, j_next, apex))
        faces.append((base_center, j_next, j))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _safe_name(label: str, fallback: str) -> str:
    text = label.strip().replace(" ", "_")
    text = text.replace("/", "_")
    return text or fallback


def _shape_color(shape_type: int) -> tuple[int, int, int]:
    if shape_type == int(SapGeoType.BOX):
        return (120, 185, 255)
    if shape_type == int(SapGeoType.SPHERE):
        return (255, 170, 100)
    if shape_type in (int(SapGeoType.CYLINDER), int(SapGeoType.CAPSULE), int(SapGeoType.CONE)):
        return (140, 220, 160)
    if shape_type == int(SapGeoType.ELLIPSOID):
        return (220, 150, 240)
    return (180, 180, 190)


def _mesh_color(mesh: Any, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    color = getattr(mesh, "color", None)
    if color is None:
        texture_color = _texture_average_color(getattr(mesh, "texture", None))
        return texture_color if texture_color is not None else fallback
    values = np.asarray(color, dtype=np.float64).reshape(-1)
    if values.size < 3:
        return fallback
    rgb = values[:3]
    if np.max(rgb) <= 1.0:
        rgb = rgb * 255.0
    return tuple(int(np.clip(v, 0, 255)) for v in rgb)


def _viewer_color(color: Any, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if color is None:
        return fallback
    values = np.asarray(color, dtype=np.float64).reshape(-1)
    if values.size < 3:
        return fallback
    rgb = values[:3]
    if np.max(rgb) <= 1.0:
        rgb = rgb * 255.0
    return tuple(int(np.clip(v, 0, 255)) for v in rgb)


def _texture_average_color(texture: Any) -> tuple[int, int, int] | None:
    if texture is None:
        return None
    try:
        if isinstance(texture, str):
            cached = _TEXTURE_COLOR_CACHE.get(texture)
            if texture in _TEXTURE_COLOR_CACHE:
                return cached
            from PIL import Image

            with Image.open(texture) as image:
                pixels = np.asarray(image.convert("RGB").resize((1, 1), resample=Image.Resampling.BOX), dtype=np.uint8)
            color = tuple(int(v) for v in pixels.reshape(3))
            _TEXTURE_COLOR_CACHE[texture] = color
            return color
        pixels = np.asarray(texture)
        if pixels.size == 0:
            return None
        pixels = pixels.reshape(-1, pixels.shape[-1])
        rgb = pixels[:, :3].astype(np.float64)
        if np.max(rgb) <= 1.0:
            rgb = rgb * 255.0
        return tuple(int(np.clip(v, 0, 255)) for v in np.mean(rgb, axis=0))
    except Exception:
        if isinstance(texture, str):
            _TEXTURE_COLOR_CACHE[texture] = None
        return None


def _shape_source(model: Any, shape_id: int) -> Any | None:
    sources = getattr(model, "shape_source", None)
    if sources is None or shape_id >= len(sources):
        return None
    return sources[shape_id]


def _visible_shape_ids(model: Any) -> list[int]:
    shape_count = int(getattr(model, "shape_count", 0) or _as_numpy(model.shape_type).reshape(-1).shape[0])
    if getattr(model, "shape_flags", None) is None:
        return list(range(shape_count))

    flags = _as_numpy(model.shape_flags).reshape(-1).astype(np.int32)
    visible_ids = [i for i, flag in enumerate(flags[:shape_count]) if flag & int(SapShapeFlags.VISIBLE)]
    return visible_ids or list(range(shape_count))


def _shape_node_name(parent_path: str, shape_id: int, label: str) -> str:
    return f"{parent_path}/{shape_id:04d}_{_safe_name(label, f'shape_{shape_id}')}"


@dataclass(frozen=True)
class _BatchPrototype:
    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[int, int, int]
    flat_shading: bool
    side: str


def _shape_batch_key(
    shape_type: int,
    scale: np.ndarray,
    source: Any,
    variant: str,
    color: tuple[int, int, int],
) -> tuple:
    return (
        int(shape_type),
        variant,
        id(source),
        tuple(round(float(v), 8) for v in np.asarray(scale, dtype=np.float64).reshape(-1)),
        tuple(int(c) for c in color),
    )


def _offset_transform_xyzw(parent_tf: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = parent_tf[:3] + _quat_rotate_xyzw(parent_tf[3:7], offset)
    wxyz = _quat_xyzw_to_wxyz(parent_tf[3:7])
    return position, wxyz


def _offset_local_transform_xyzw(parent_tf: np.ndarray, offset: np.ndarray) -> np.ndarray:
    child = np.asarray([float(offset[0]), float(offset[1]), float(offset[2]), 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return _compose_transform(parent_tf, child)


def _batched_shape_visuals(
    model: Any,
    shape_id: int,
    local_tf: np.ndarray,
    shape_type: int,
    scale: np.ndarray,
    color_override: Any = None,
) -> list[tuple[tuple, _BatchPrototype, np.ndarray]]:
    shape_type = int(shape_type)
    scale = np.asarray(scale, dtype=np.float64).reshape(3)
    color = _viewer_color(color_override, _shape_color(shape_type))

    if shape_type in (int(SapGeoType.MESH), int(SapGeoType.CONVEX_MESH)):
        mesh = _shape_source(model, shape_id)
        if mesh is None or not hasattr(mesh, "vertices") or not hasattr(mesh, "indices"):
            return []
        vertices = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3) * scale.reshape(1, 3).astype(np.float32)
        faces = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
        color = _mesh_color(mesh, color)
        key = _shape_batch_key(shape_type, scale, mesh, "mesh", color)
        return [(key, _BatchPrototype(vertices, faces, color, False, "double"), local_tf)]

    if shape_type == int(SapGeoType.PLANE):
        return []

    if shape_type == int(SapGeoType.SPHERE):
        radius = max(float(scale[0]), 1.0e-4)
        vertices, faces = _ellipsoid_mesh(np.asarray([radius, radius, radius], dtype=np.float64), segments=16, rings=8)
        key = _shape_batch_key(shape_type, np.asarray([radius, radius, radius]), None, "sphere", color)
        return [(key, _BatchPrototype(vertices, faces, color, False, "double"), local_tf)]

    if shape_type == int(SapGeoType.BOX):
        vertices, faces = _box_mesh(scale)
        key = _shape_batch_key(shape_type, scale, None, "box", color)
        return [(key, _BatchPrototype(vertices, faces, color, True, "double"), local_tf)]

    if shape_type == int(SapGeoType.CYLINDER):
        vertices, faces = _cylinder_mesh(scale[0], scale[1])
        key = _shape_batch_key(shape_type, scale, None, "cylinder", color)
        return [(key, _BatchPrototype(vertices, faces, color, True, "double"), local_tf)]

    if shape_type == int(SapGeoType.CONE):
        vertices, faces = _cone_mesh(scale[0], scale[1])
        key = _shape_batch_key(shape_type, scale, None, "cone", color)
        return [(key, _BatchPrototype(vertices, faces, color, True, "double"), local_tf)]

    if shape_type == int(SapGeoType.ELLIPSOID):
        vertices, faces = _ellipsoid_mesh(scale)
        key = _shape_batch_key(shape_type, scale, None, "ellipsoid", color)
        return [(key, _BatchPrototype(vertices, faces, color, True, "double"), local_tf)]

    if shape_type == int(SapGeoType.CAPSULE):
        radius = max(float(scale[0]), 1.0e-4)
        half_height = max(float(scale[1]), 1.0e-4)
        cylinder_vertices, cylinder_faces = _cylinder_mesh(radius, half_height)
        cylinder_key = _shape_batch_key(shape_type, scale, None, "capsule_cylinder", color)
        cap_vertices, cap_faces = _ellipsoid_mesh(
            np.asarray([radius, radius, radius], dtype=np.float64),
            segments=16,
            rings=8,
        )
        cap_key = _shape_batch_key(shape_type, np.asarray([radius, radius, radius]), None, "capsule_cap", color)
        cap0_tf = _offset_local_transform_xyzw(local_tf, np.asarray([0.0, 0.0, -half_height]))
        cap1_tf = _offset_local_transform_xyzw(local_tf, np.asarray([0.0, 0.0, half_height]))
        return [
            (cylinder_key, _BatchPrototype(cylinder_vertices, cylinder_faces, color, True, "double"), local_tf),
            (cap_key, _BatchPrototype(cap_vertices, cap_faces, color, False, "double"), cap0_tf),
            (cap_key, _BatchPrototype(cap_vertices, cap_faces, color, False, "double"), cap1_tf),
        ]

    return []


def _add_shape_handle(
    server: viser.ViserServer,
    model: Any,
    shape_id: int,
    parent_path: str,
    local_tf: np.ndarray,
    color_override: Any = None,
) -> bool:
    shape_type = int(_as_numpy(model.shape_type).reshape(-1)[shape_id])
    scale = _as_numpy(model.shape_scale).reshape(-1, 3)[shape_id].astype(np.float64)
    labels = tuple(getattr(model, "shape_label", ()) or ())
    label = labels[shape_id] if shape_id < len(labels) else f"shape_{shape_id}"
    name = _shape_node_name(parent_path, shape_id, str(label))
    color = _viewer_color(color_override, _shape_color(shape_type))
    position = local_tf[:3]
    wxyz = _quat_xyzw_to_wxyz(local_tf[3:7])

    if shape_type in (int(SapGeoType.MESH), int(SapGeoType.CONVEX_MESH)):
        mesh = _shape_source(model, shape_id)
        if mesh is None or not hasattr(mesh, "vertices") or not hasattr(mesh, "indices"):
            return False
        vertices = np.asarray(mesh.vertices, dtype=np.float32).reshape(-1, 3) * scale.reshape(1, 3).astype(np.float32)
        faces = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
        server.scene.add_mesh_simple(
            name,
            vertices=vertices,
            faces=faces,
            color=_mesh_color(mesh, color),
            side="double",
            position=position,
            wxyz=wxyz,
        )
        return True

    if shape_type == int(SapGeoType.PLANE):
        width = float(scale[0]) if abs(float(scale[0])) > 0.0 else 8.0
        height = float(scale[1]) if abs(float(scale[1])) > 0.0 else 8.0
        width *= 50.0
        height *= 50.0
        server.scene.add_grid(
            name,
            width=width,
            height=height,
            plane="xy",
            cell_size=0.25,
            section_size=1.0,
            plane_opacity=0.08,
            position=position,
            wxyz=wxyz,
        )
        return True

    if shape_type == int(SapGeoType.SPHERE):
        server.scene.add_icosphere(
            name,
            radius=max(float(scale[0]), 1.0e-4),
            subdivisions=2,
            color=color,
            position=position,
            wxyz=wxyz,
        )
        return True

    if shape_type == int(SapGeoType.BOX):
        vertices, faces = _box_mesh(scale)
    elif shape_type == int(SapGeoType.CYLINDER):
        vertices, faces = _cylinder_mesh(scale[0], scale[1])
    elif shape_type == int(SapGeoType.CONE):
        vertices, faces = _cone_mesh(scale[0], scale[1])
    elif shape_type == int(SapGeoType.CAPSULE):
        vertices, faces = _cylinder_mesh(scale[0], scale[1])
    elif shape_type == int(SapGeoType.ELLIPSOID):
        vertices, faces = _ellipsoid_mesh(scale)
    else:
        return False

    server.scene.add_mesh_simple(
        name,
        vertices=vertices,
        faces=faces,
        color=color,
        flat_shading=True,
        side="double",
        position=position,
        wxyz=wxyz,
    )

    if shape_type != int(SapGeoType.CAPSULE):
        return True

    radius = max(float(scale[0]), 1.0e-4)
    cap_offsets = (np.asarray([0.0, 0.0, -float(scale[1])]), np.asarray([0.0, 0.0, float(scale[1])]))
    cap0_position, cap_wxyz = _offset_transform_xyzw(local_tf, cap_offsets[0])
    cap1_position, _ = _offset_transform_xyzw(local_tf, cap_offsets[1])
    server.scene.add_icosphere(
        f"{name}_cap_0",
        radius=radius,
        subdivisions=2,
        color=color,
        position=cap0_position,
        wxyz=cap_wxyz,
    )
    server.scene.add_icosphere(
        f"{name}_cap_1",
        radius=radius,
        subdivisions=2,
        color=color,
        position=cap1_position,
        wxyz=cap_wxyz,
    )
    return True


def _build_viewer(
    server: viser.ViserServer,
    model: Any,
    state: Any,
    shape_colors: Sequence[Any] = (),
) -> ViewerRenderState:
    server.scene.set_up_direction("+z")
    server.scene.add_light_ambient("/lights/ambient", intensity=0.45)
    server.scene.add_light_directional("/lights/key", intensity=1.8, position=(2.0, -3.0, 5.0))
    server.initial_camera.position = (3.5, -4.0, 2.5)
    server.initial_camera.look_at = (0.0, 0.0, 0.8)

    body_tfs = _body_transforms(state)
    world_path = "/sap/world"
    server.scene.add_frame(world_path, show_axes=False)

    shape_body = _as_numpy(model.shape_body).reshape(-1).astype(np.int32)
    shape_transform = _as_numpy(model.shape_transform).reshape(-1, 7).astype(np.float64)
    shape_type = _as_numpy(model.shape_type).reshape(-1).astype(np.int32)
    shape_scale = _as_numpy(model.shape_scale).reshape(-1, 3).astype(np.float64)
    shape_labels = tuple(getattr(model, "shape_label", ()) or ())
    groups: dict[tuple, dict[str, Any]] = {}
    visible_shape_count = 0
    for shape_id in _visible_shape_ids(model):
        body_id = int(shape_body[shape_id])
        visuals = _batched_shape_visuals(
            model,
            shape_id,
            shape_transform[shape_id],
            int(shape_type[shape_id]),
            shape_scale[shape_id],
            shape_colors[shape_id] if shape_id < len(shape_colors) else None,
        )
        if not visuals:
            label = shape_labels[shape_id] if shape_id < len(shape_labels) else f"shape_{shape_id}"
            if _add_shape_handle(
                server,
                model,
                shape_id,
                world_path,
                shape_transform[shape_id],
                shape_colors[shape_id] if shape_id < len(shape_colors) else None,
            ):
                visible_shape_count += 1
            else:
                print(f"Skipping unsupported visible shape {shape_id}: {label}", flush=True)
            continue
        for key, prototype, local_tf in visuals:
            group = groups.setdefault(
                (key, body_id >= 0),
                {
                    "prototype": prototype,
                    "body_ids": [],
                    "local_transforms": [],
                },
            )
            group["body_ids"].append(body_id)
            group["local_transforms"].append(local_tf)
        if visuals:
            visible_shape_count += 1

    shape_batches: list[ViewerShapeBatch] = []
    visual_instance_count = 0
    for group_index, group in enumerate(groups.values()):
        prototype = group["prototype"]
        body_ids = np.asarray(group["body_ids"], dtype=np.int32)
        local_transforms = np.asarray(group["local_transforms"], dtype=np.float64).reshape(-1, 7)
        batch = _make_viewer_shape_batch(None, body_ids, local_transforms, body_tfs)
        handle = server.scene.add_batched_meshes_simple(
            f"/sap/batches/{group_index:04d}",
            vertices=prototype.vertices,
            faces=prototype.faces,
            batched_wxyzs=batch.wxyzs,
            batched_positions=batch.positions,
            batched_colors=prototype.color,
            flat_shading=prototype.flat_shading,
            side=prototype.side,
        )
        batch.handle = handle
        shape_batches.append(batch)
        visual_instance_count += int(body_ids.shape[0])
    return ViewerRenderState(
        shape_batches=tuple(shape_batches),
        visible_shape_count=visible_shape_count,
        visual_instance_count=visual_instance_count,
    )


def _update_viewer_from_body_transforms(
    server: viser.ViserServer,
    body_tfs: np.ndarray,
    render_state: ViewerRenderState,
) -> None:
    with server.atomic():
        for batch in render_state.shape_batches:
            if _update_shape_batch_buffers(body_tfs, batch):
                _queue_shape_batch_pose_update(batch)


def _update_viewer(server: viser.ViserServer, state: Any, render_state: ViewerRenderState) -> None:
    _update_viewer_from_body_transforms(server, _body_transforms(state), render_state)


def _build_controls(server: viser.ViserServer, settings: dict[str, Any]) -> ViewerControls:
    server.gui.add_markdown("### Sap Warp Viser Viewer")
    running = {"value": True}
    reset_requested = {"value": False}
    controls_button = server.gui.add_button_group("Controls", ("Run", "Pause", "Reset"))
    status_text = server.gui.add_text("State", "starting\nframe 0  t=0.000s", multiline=True, disabled=True)
    perf_text = server.gui.add_text(
        "Performance",
        "sim -- FPS   RTR --x\nviewer -- FPS",
        multiline=True,
        disabled=True,
    )
    with server.gui.add_folder("Scene Info", expand_by_default=False):
        server.gui.add_text("Device", str(settings["device"]), disabled=True)
        server.gui.add_text("Scene", str(settings["scene"]), disabled=True)
        server.gui.add_text("Scene path", str(settings["scene_path"]), disabled=True)
        server.gui.add_number("num_worlds", int(settings["num_worlds"]), disabled=True)
        server.gui.add_number("viewer_fps", float(settings["viewer_fps"]), disabled=True)
        server.gui.add_text("cuda_graph", str(settings["cuda_graph"]), disabled=True)
    with server.gui.add_folder("Simulation Settings", expand_by_default=False):
        server.gui.add_number("dt", float(settings["dt"]), disabled=True)
        server.gui.add_number("max_rigid_contact", int(settings["max_rigid_contact"]), disabled=True)
        server.gui.add_number("rigid_contact_capacity", int(settings["rigid_contact_capacity"]), disabled=True)
        server.gui.add_number("substeps_per_frame", int(settings["substeps_per_frame"]), disabled=True)
        with server.gui.add_folder("solver", expand_by_default=True):
            server.gui.add_text("contact_preset_variant", str(settings["contact_preset_variant"]), disabled=True)
            server.gui.add_text("line_search_variant", str(settings["line_search_variant"]), disabled=True)
        if settings["gravity"] is not None or settings["rigid_gap"] is not None:
            with server.gui.add_folder("builder", expand_by_default=True):
                if settings["gravity"] is not None:
                    server.gui.add_number("gravity", float(settings["gravity"]), disabled=True)
                if settings["rigid_gap"] is not None:
                    server.gui.add_number("rigid_gap", float(settings["rigid_gap"]), disabled=True)
        if settings["shape_ke"] is not None or settings["shape_tau"] is not None or settings["shape_mu"] is not None:
            with server.gui.add_folder("shape", expand_by_default=True):
                if settings["shape_ke"] is not None:
                    server.gui.add_number("ke", float(settings["shape_ke"]), disabled=True)
                if settings["shape_tau"] is not None:
                    server.gui.add_number("tau", float(settings["shape_tau"]), disabled=True)
                if settings["shape_mu"] is not None:
                    server.gui.add_number("mu", float(settings["shape_mu"]), disabled=True)
    updating_controls = {"value": False}

    def _sync_controls_button() -> None:
        updating_controls["value"] = True
        controls_button.value = "Run" if running["value"] else "Pause"
        updating_controls["value"] = False

    @controls_button.on_click
    def _(_) -> None:
        if updating_controls["value"]:
            return
        if controls_button.value == "Reset":
            reset_requested["value"] = True
        elif controls_button.value == "Run":
            running["value"] = True
        else:
            running["value"] = False
        _sync_controls_button()

    _sync_controls_button()

    return ViewerControls(
        running=running,
        status_text=status_text,
        perf_text=perf_text,
        reset_requested=reset_requested,
    )


def run_viewer(args: argparse.Namespace) -> None:
    """Launch the Viser SAP scene viewer, load the selected scene, and drive interactive simulation updates."""
    scene_path = _resolve_path(args.scene)
    scene_config = load_sap_scene_config(scene_path)
    simulation = scene_config.get("simulation", {}) or {}
    if not isinstance(simulation, dict):
        simulation = {}
    dt = _effective_dt(args, simulation)
    frames = _frame_limit(args, dt)
    substeps_per_frame = _effective_substeps_per_frame(args, dt)
    num_worlds = _scene_num_worlds(args, scene_path, simulation)
    max_rigid_contact_per_env = _scene_max_rigid_contact(scene_path, simulation)
    rigid_contact_capacity = _total_rigid_contact_capacity(max_rigid_contact_per_env, num_worlds)
    solver_kwargs = _scene_solver_kwargs(scene_path, simulation)
    shape_tau = _nested_config_value(scene_config, ("builder", "defaults", "shape", "tau"))
    contact_tau_d = float(shape_tau) if shape_tau is not None else None

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
        contact_tau_d=contact_tau_d,
        **solver_kwargs,
    )
    collision_pipeline = SapCollisionPipeline(loaded.collision_model, rigid_contact_max=rigid_contact_capacity)
    contacts = collision_pipeline.contacts()
    state_0 = loaded.sap_state
    state_1 = model.state()
    control = loaded.sap_control
    reset_control = model.control(requires_grad=False)
    control_sequence = load_sap_control_sequence(scene_path, model, state_0, control, device=device)
    reset_state = model.state()
    if control_sequence is not None:
        control_sequence.apply_initial_state(model, reset_state)

    server = viser.ViserServer(host=args.host, port=args.port, label="SAP Warp")
    render_state = _build_viewer(server, model, state_0, loaded.shape_colors)
    settings = {
        "scene": scene_config.get("name", scene_path.name),
        "scene_path": scene_path,
        "device": str(device),
        "dt": dt,
        "num_worlds": num_worlds,
        "max_rigid_contact": max_rigid_contact_per_env,
        "rigid_contact_capacity": rigid_contact_capacity,
        "viewer_fps": max(float(args.viewer_fps), 1.0),
        "substeps_per_frame": substeps_per_frame,
        "cuda_graph": "disabled" if args.disable_cuda_graph else "requested",
        "contact_preset_variant": solver_kwargs.get("contact_preset_variant", "default"),
        "line_search_variant": solver_kwargs.get("line_search_variant", "default"),
        "gravity": _nested_config_value(scene_config, ("builder", "gravity")),
        "rigid_gap": _nested_config_value(scene_config, ("builder", "rigid_gap")),
        "shape_ke": _nested_config_value(scene_config, ("builder", "defaults", "shape", "ke")),
        "shape_tau": shape_tau,
        "shape_mu": _nested_config_value(scene_config, ("builder", "defaults", "shape", "mu")),
    }
    controls = _build_controls(server, settings)
    mouse_control = ViewerMouseControl(server, model, state_0, control, render_state) if args.mouse_control else None

    graph = None
    if not args.disable_cuda_graph:
        controls.status_text.value = "capturing CUDA graph"
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
        controls.status_text.value = "cuda graph ready" if graph is not None else "direct stepping"
    else:
        controls.status_text.value = "direct stepping"

    url_host = "localhost" if args.host in {"0.0.0.0", "::"} else args.host
    viewer_url = f"http://{url_host}:{server.get_port()}"
    print(
        f"Viser viewer: {viewer_url} "
        f"scene={scene_path} device={device} dt={dt:.6f} shapes={render_state.visible_shape_count} "
        f"visual_instances={render_state.visual_instance_count} shape_batches={len(render_state.shape_batches)} "
        f"num_worlds={num_worlds} max_rigid_contact_per_env={max_rigid_contact_per_env} "
        f"rigid_contact_capacity={rigid_contact_capacity} "
        f"viewer_fps={max(float(args.viewer_fps), 1.0):.1f} "
        f"substeps_per_frame={substeps_per_frame} mouse_control={mouse_control is not None} "
        f"cuda_graph={graph is not None}",
        flush=True,
    )
    print("Simulation loop starting. The first frame may compile Warp kernels.", flush=True)

    recorder: _ChromeMp4Recorder | None = None
    if args.record_mp4 is not None:
        record_fps = float(args.record_fps if args.record_fps is not None else args.viewer_fps)
        recorder = _ChromeMp4Recorder(
            url=viewer_url,
            output=_resolve_path(args.record_mp4),
            fps=record_fps,
            width=int(args.record_width),
            height=int(args.record_height),
            webgl_backend=str(args.record_webgl),
        )
        recorder.start()
        print(
            f"Recording MP4 to {recorder.output} at {record_fps:.1f} FPS "
            f"({recorder.width}x{recorder.height}) using {recorder.webgl_renderer}.",
            flush=True,
        )

    frame_id = 0
    sim_time = 0.0
    frame_interval = 1.0 / max(float(args.viewer_fps), 1.0)
    next_display_time = time.perf_counter() + frame_interval
    curr_body_tfs = _body_transforms(state_0)
    state_update_interval = 0.2
    last_status_time = time.perf_counter()
    last_state_update_time = 0.0
    last_state_status = ""
    perf_window_start = last_status_time
    perf_window_frames = 0
    perf_window_sim_time = 0.0
    perf_window_step_wall = 0.0
    perf_window_view_wall = 0.0
    perf_window_updates = 0

    def _update_state_text(status: str, *, force: bool = False) -> None:
        nonlocal last_state_status, last_state_update_time
        now = time.perf_counter()
        if not force and status == last_state_status and now - last_state_update_time < state_update_interval:
            return
        controls.status_text.value = f"{status}\nframe {frame_id}  t={sim_time:.3f}s"
        last_state_status = status
        last_state_update_time = now

    try:
        try:
            while frames is None or frame_id < frames:
                if controls.reset_requested["value"]:
                    controls.reset_requested["value"] = False
                    state_0.assign(reset_state)
                    state_1.assign(reset_state)
                    _assign_control(control, reset_control)
                    frame_id = 0
                    sim_time = 0.0
                    solver.sim_time = 0.0
                    solver.frame_id = 0
                    if hasattr(solver, "_has_contact_solve_v_guess"):
                        solver._has_contact_solve_v_guess = False
                    if control_sequence is not None:
                        control_sequence.apply(control, frame_id, dt)
                    curr_body_tfs = _body_transforms(state_0)
                    if mouse_control is not None:
                        mouse_control.reset(state_0)
                    _update_viewer_from_body_transforms(server, curr_body_tfs, render_state)
                    perf_window_start = time.perf_counter()
                    perf_window_frames = 0
                    perf_window_sim_time = 0.0
                    perf_window_step_wall = 0.0
                    perf_window_view_wall = 0.0
                    perf_window_updates = 0
                    next_display_time = time.perf_counter() + frame_interval
                    _update_state_text("running" if controls.running["value"] else "paused", force=True)

                if controls.running["value"]:
                    status = "running"
                    step_start = time.perf_counter()
                    stepped = False
                    while frames is None or frame_id < frames:
                        steps_this_batch = 0
                        for _ in range(substeps_per_frame):
                            if mouse_control is not None:
                                mouse_control.apply_targets(state_0, control)
                            if control_sequence is not None:
                                control_sequence.apply(control, frame_id, dt)
                            if graph is not None:
                                _launch_step_graph(graph)
                            else:
                                state_0, state_1 = _step_native(
                                    solver=solver,
                                    collision_pipeline=collision_pipeline,
                                    state_0=state_0,
                                    state_1=state_1,
                                    control=control,
                                    contacts=contacts,
                                    dt=dt,
                                )
                            frame_id += 1
                            sim_time += dt
                            perf_window_frames += 1
                            perf_window_sim_time += dt
                            steps_this_batch += 1
                            if frames is not None and frame_id >= frames:
                                break
                        if steps_this_batch == 0:
                            break
                        stepped = True
                        wp.synchronize_device(device)
                        if time.perf_counter() >= next_display_time:
                            break
                    if stepped:
                        curr_body_tfs = _body_transforms(state_0)
                        perf_window_step_wall += time.perf_counter() - step_start

                    now = time.perf_counter()
                    should_render = now >= next_display_time or (frames is not None and frame_id >= frames)
                    if should_render:
                        view_start = time.perf_counter()
                        _update_viewer_from_body_transforms(server, curr_body_tfs, render_state)
                        view_elapsed = time.perf_counter() - view_start
                        perf_window_view_wall += view_elapsed
                        perf_window_updates += 1
                        while next_display_time <= now:
                            next_display_time += frame_interval
                else:
                    status = "paused"
                    next_display_time = time.perf_counter() + frame_interval
                    time.sleep(frame_interval)

                _update_state_text(status)
                perf_elapsed = time.perf_counter() - perf_window_start
                if perf_elapsed >= 0.5:
                    sim_steps_per_second = perf_window_frames / perf_elapsed
                    viewer_updates_per_second = perf_window_updates / perf_elapsed
                    rtr = perf_window_sim_time / perf_elapsed
                    sim_step_ms = (
                        1000.0 * perf_window_step_wall / perf_window_frames if perf_window_frames > 0 else 0.0
                    )
                    viewer_update_ms = (
                        1000.0 * perf_window_view_wall / perf_window_updates if perf_window_updates > 0 else 0.0
                    )
                    controls.perf_text.value = (
                        f"sim {sim_steps_per_second:.1f} FPS   RTR {rtr:.2f}x   {sim_step_ms:.2f} ms\n"
                        f"viewer {viewer_updates_per_second:.1f} FPS   {viewer_update_ms:.2f} ms"
                    )
                    perf_window_start = time.perf_counter()
                    perf_window_frames = 0
                    perf_window_sim_time = 0.0
                    perf_window_step_wall = 0.0
                    perf_window_view_wall = 0.0
                    perf_window_updates = 0
                if time.perf_counter() - last_status_time >= 1.0:
                    state_status = controls.status_text.value.replace("\n", "  ")
                    perf_status = controls.perf_text.value.replace("\n", "  ")
                    print(
                        f"viewer frame={frame_id} sim_time={sim_time:.3f}s "
                        f"status={state_status} perf={perf_status}",
                        flush=True,
                    )
                    last_status_time = time.perf_counter()
        except KeyboardInterrupt:
            return
    finally:
        if recorder is not None:
            recorder.stop()
            print(f"Saved MP4 recording to {recorder.output}", flush=True)

    print("Simulation finished.", flush=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the Viser SAP viewer."""
    parser = argparse.ArgumentParser(description="Visualize SAP Warp simulation with viser.")
    parser.add_argument("--scene", type=str, default=str(DEFAULT_SCENE), help="YAML scene file.")
    parser.add_argument("--duration", type=float, default=math.inf, help="Simulation duration in seconds.")
    parser.add_argument("--frames", type=int, default=None, help="Number of simulation frames. Overrides --duration.")
    parser.add_argument("--dt", type=float, default=None, help="Simulation timestep. Defaults to simulation.dt.")
    parser.add_argument("--num-worlds", type=int, default=None, help="Number of replicated worlds.")
    parser.add_argument("--device", type=str, default=None, help="Warp device, for example cuda:0 or cpu.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Viser server host.")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port.")
    parser.add_argument("--viewer-fps", type=float, default=60.0, help="Smooth viewer display update rate.")
    parser.add_argument("--record-mp4", type=str, default=None, help="Record the Viser browser output to an MP4 file.")
    parser.add_argument("--record-fps", type=float, default=None, help="MP4 recording FPS. Defaults to --viewer-fps.")
    parser.add_argument("--record-width", type=int, default=1280, help="MP4 recording browser width in pixels.")
    parser.add_argument("--record-height", type=int, default=720, help="MP4 recording browser height in pixels.")
    parser.add_argument(
        "--record-webgl",
        choices=("gpu", "swiftshader"),
        default="gpu",
        help="WebGL backend for MP4 recording. Use swiftshader only as a software fallback.",
    )
    parser.add_argument(
        "--substeps-per-frame",
        type=int,
        default=None,
        help="Simulation launch batch size between viewer updates. Defaults to 1.",
    )
    parser.add_argument("--disable-cuda-graph", action="store_true", help="Disable CUDA graph replay for simulation steps.")
    parser.add_argument(
        "--mouse-control",
        action="store_true",
        help="Enable Viser EE pose mouse controls for supported robot arms. Body force dragging is disabled.",
    )
    return parser


def main() -> None:
    """Parse viewer command-line arguments and start the Viser application."""
    args = build_parser().parse_args()
    run_viewer(args)

if __name__ == "__main__":
    main()
