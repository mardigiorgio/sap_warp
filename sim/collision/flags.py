"""Shape flags consumed by the Newton-compatible SAP collision path.

Source note: the SAP modifications in this module are based on Newton's
collision metadata code and adapted for SAP Warp compatibility.
"""

from __future__ import annotations

from enum import IntEnum


class SapShapeFlags(IntEnum):
    """Flags for shape properties consumed by SAP collision generation."""

    VISIBLE = 1 << 0
    COLLIDE_SHAPES = 1 << 1
    COLLIDE_PARTICLES = 1 << 2
    SITE = 1 << 3
    HYDROELASTIC = 1 << 4


__all__ = ["SapShapeFlags"]
