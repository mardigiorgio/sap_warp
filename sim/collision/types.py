"""Shape type identifiers for the Newton-compatible SAP collision path.

Source note: the SAP modifications in this module are based on Newton's
collision type/metadata code and adapted for SAP Warp compatibility.
"""

from __future__ import annotations

import enum

import warp as wp


class SapGeoType(enum.IntEnum):
    """Geometric shape type identifiers used by SAP collision generation."""

    NONE = 0
    PLANE = 1
    HFIELD = 2
    SPHERE = 3
    CAPSULE = 4
    ELLIPSOID = 5
    CYLINDER = 6
    BOX = 7
    MESH = 8
    CONE = 9
    CONVEX_MESH = 10
    GAUSSIAN = 11


SapVec5 = wp.types.vector(length=5, dtype=wp.float32)


__all__ = ["SapGeoType", "SapVec5"]
