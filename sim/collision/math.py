"""Math helpers used by SAP collision kernels.

Source note: the SAP modifications in this module are based on Newton's
collision math/support code and adapted for SAP Warp's broad/narrow-phase
stages.
"""

from __future__ import annotations

from typing import Any

import warp as wp


SAP_MAXVAL = 1e10
SAP_MATH_EPSILON = 1e-15


@wp.func
def sap_orthonormal_basis(n: wp.vec3):
    b1 = wp.vec3()
    b2 = wp.vec3()
    if n[2] < 0.0:
        a = 1.0 / (1.0 - n[2])
        b = n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = -b
        b1[2] = n[0]

        b2[0] = b
        b2[1] = n[1] * n[1] * a - 1.0
        b2[2] = -n[1]
    else:
        a = 1.0 / (1.0 + n[2])
        b = -n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = b
        b1[2] = -n[0]

        b2[0] = b
        b2[1] = 1.0 - n[1] * n[1] * a
        b2[2] = -n[1]

    return b1, b2


@wp.func
def sap_safe_div(x: Any, y: Any, eps: float = SAP_MATH_EPSILON) -> Any:
    return x / wp.where(y != 0.0, y, eps)


@wp.func
def sap_normalize_with_norm(x: Any):
    norm = wp.length(x)
    if norm == 0.0:
        return x, 0.0
    return x / norm, norm


__all__ = [
    "SAP_MATH_EPSILON",
    "SAP_MAXVAL",
    "sap_normalize_with_norm",
    "sap_orthonormal_basis",
    "sap_safe_div",
]
