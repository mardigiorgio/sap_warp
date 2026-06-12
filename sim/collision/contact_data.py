"""Contact records shared by SAP collision kernels.

Source note: the SAP modifications in this module are based on Newton's
collision contact code and adapted for SAP Warp contact buffers.
"""

from __future__ import annotations

import warp as wp


SAP_SHAPE_PAIR_HFIELD_BIT = wp.int32(1 << 30)
SAP_SHAPE_PAIR_INDEX_MASK = wp.int32((1 << 30) - 1)


@wp.struct
class SapContactData:
    """Internal contact representation passed through SAP collision kernels."""

    contact_point_center: wp.vec3
    contact_normal_a_to_b: wp.vec3
    contact_distance: float
    radius_eff_a: float
    radius_eff_b: float
    margin_a: float
    margin_b: float
    shape_a: int
    shape_b: int
    gap_sum: float
    contact_stiffness: float
    contact_damping: float
    contact_friction_scale: float


@wp.func
def sap_contact_passes_gap_check(
    contact_data: SapContactData,
) -> bool:
    total_separation_needed = (
        contact_data.radius_eff_a + contact_data.radius_eff_b + contact_data.margin_a + contact_data.margin_b
    )

    contact_normal_a_to_b = wp.normalize(contact_data.contact_normal_a_to_b)

    a_contact_world = contact_data.contact_point_center - contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_a
    )
    b_contact_world = contact_data.contact_point_center + contact_normal_a_to_b * (
        0.5 * contact_data.contact_distance + contact_data.radius_eff_b
    )

    diff = b_contact_world - a_contact_world
    distance = wp.dot(diff, contact_normal_a_to_b)
    d = distance - total_separation_needed

    return d <= contact_data.gap_sum


__all__ = [
    "SAP_SHAPE_PAIR_HFIELD_BIT",
    "SAP_SHAPE_PAIR_INDEX_MASK",
    "SapContactData",
    "sap_contact_passes_gap_check",
]
