"""Mesh resources used by SAP scene loading and collision generation.

Source note: the SAP modifications in this module are based on Newton's mesh
resource code and adapted for compatibility with imported assets and collision
data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp


def _normalize_texture_input(texture: str | np.ndarray | None) -> str | np.ndarray | None:
    if texture is None:
        return None
    if isinstance(texture, str):
        return texture
    return np.ascontiguousarray(np.asarray(texture))


class SapMesh:
    MAX_HULL_VERTICES = 64

    def __init__(
        self,
        vertices: Sequence[Sequence[float]] | np.ndarray,
        indices: Sequence[int] | np.ndarray,
        normals: Sequence[Sequence[float]] | np.ndarray | None = None,
        uvs: Sequence[Sequence[float]] | np.ndarray | None = None,
        compute_inertia: bool = False,
        is_solid: bool = True,
        maxhullvert: int | None = None,
        color: Sequence[float] | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
        texture: str | np.ndarray | None = None,
        *,
        sdf: Any | None = None,
    ) -> None:
        self._vertices = np.array(vertices, dtype=np.float32).reshape(-1, 3)
        self._indices = np.array(indices, dtype=np.int32).flatten()
        self._normals = np.array(normals, dtype=np.float32).reshape(-1, 3) if normals is not None else None
        self._uvs = np.array(uvs, dtype=np.float32).reshape(-1, 2) if uvs is not None else None
        self.is_solid = is_solid
        self.has_inertia = compute_inertia
        self.mesh: wp.Mesh | None = None
        self.maxhullvert = SapMesh.MAX_HULL_VERTICES if maxhullvert is None else maxhullvert
        self._cached_hash: int | None = None
        self._texture = _normalize_texture_input(texture)
        self._roughness = roughness
        self._metallic = metallic
        self.color = color
        self.sdf = sdf

        self.inertia = wp.mat33(np.eye(3))
        self.mass = 1.0
        self.com = wp.vec3()

    def copy(
        self,
        vertices: Sequence[Sequence[float]] | np.ndarray | None = None,
        indices: Sequence[int] | np.ndarray | None = None,
        recompute_inertia: bool = False,
    ) -> "SapMesh":
        if vertices is None:
            vertices = self.vertices.copy()
        if indices is None:
            indices = self.indices.copy()
        mesh = SapMesh(
            vertices,
            indices,
            normals=self.normals.copy() if self.normals is not None else None,
            uvs=self.uvs.copy() if self.uvs is not None else None,
            compute_inertia=recompute_inertia,
            is_solid=self.is_solid,
            maxhullvert=self.maxhullvert,
            color=self.color,
            roughness=self._roughness,
            metallic=self._metallic,
            texture=self._texture.copy() if isinstance(self._texture, np.ndarray) else self._texture,
            sdf=self.sdf,
        )
        if not recompute_inertia:
            mesh.inertia = self.inertia
            mesh.mass = self.mass
            mesh.com = self.com
            mesh.has_inertia = self.has_inertia
        return mesh

    def build_sdf(
        self,
        *,
        device: object | None = None,
        narrow_band_range: tuple[float, float] | None = None,
        target_voxel_size: float | None = None,
        max_resolution: int | None = None,
        margin: float | None = None,
        shape_margin: float = 0.0,
        scale: tuple[float, float, float] | None = None,
        texture_format: str = "uint16",
    ):
        if self.sdf is not None:
            raise RuntimeError("SapMesh already has an SDF. Call clear_sdf() before rebuilding.")

        from .sdf import SapSDF

        self.sdf = SapSDF.create_from_mesh(
            self,
            device=device,
            narrow_band_range=narrow_band_range if narrow_band_range is not None else (-0.1, 0.1),
            target_voxel_size=target_voxel_size,
            max_resolution=max_resolution,
            margin=margin if margin is not None else 0.05,
            shape_margin=shape_margin,
            scale=scale,
            texture_format=texture_format,
        )
        return self.sdf

    def clear_sdf(self) -> None:
        self.sdf = None

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @vertices.setter
    def vertices(self, value: Sequence[Sequence[float]] | np.ndarray) -> None:
        self._vertices = np.array(value, dtype=np.float32).reshape(-1, 3)
        self._cached_hash = None

    @property
    def indices(self) -> np.ndarray:
        return self._indices

    @indices.setter
    def indices(self, value: Sequence[int] | np.ndarray) -> None:
        self._indices = np.array(value, dtype=np.int32).flatten()
        self._cached_hash = None

    @property
    def normals(self) -> np.ndarray | None:
        return self._normals

    @property
    def uvs(self) -> np.ndarray | None:
        return self._uvs

    @property
    def texture(self) -> str | np.ndarray | None:
        return self._texture

    @texture.setter
    def texture(self, value: str | np.ndarray | None) -> None:
        self._texture = _normalize_texture_input(value)
        self._cached_hash = None

    @property
    def roughness(self) -> float | None:
        return self._roughness

    @roughness.setter
    def roughness(self, value: float | None) -> None:
        self._roughness = value
        self._cached_hash = None

    @property
    def metallic(self) -> float | None:
        return self._metallic

    @metallic.setter
    def metallic(self, value: float | None) -> None:
        self._metallic = value
        self._cached_hash = None

    def finalize(self, device: object | None = None, requires_grad: bool = False) -> wp.uint64:
        with wp.ScopedDevice(device):
            pos = wp.array(self.vertices, requires_grad=requires_grad, dtype=wp.vec3)
            vel = wp.zeros_like(pos)
            indices = wp.array(self.indices, dtype=wp.int32)
            self.mesh = wp.Mesh(points=pos, velocities=vel, indices=indices)
            return self.mesh.id

    def __hash__(self) -> int:
        if self._cached_hash is None:
            texture_hash = 0
            if isinstance(self._texture, str):
                texture_hash = hash(self._texture)
            elif self._texture is not None:
                texture_hash = hash(self._texture.tobytes())
            self._cached_hash = hash(
                (
                    tuple(np.array(self.vertices).flatten()),
                    tuple(np.array(self.indices).flatten()),
                    self.is_solid,
                    texture_hash,
                    self._roughness,
                    self._metallic,
                )
            )
        return self._cached_hash


__all__ = ["SapMesh"]
