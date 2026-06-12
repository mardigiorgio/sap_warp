"""SAP-owned rigid contact buffers for the Newton-compatible collision path.

Source note: the SAP modifications in this module are based on Newton's
contact storage code and adapted so SAP Warp can interoperate with
Newton-authored scenes and runtime objects.
"""

from __future__ import annotations

import warp as wp
from warp import DeviceLike as SapDeviceLike


class SapContacts:
    """
    Rigid contact storage consumed by SAP-owned collision generation.

    The buffer layout mirrors the rigid-contact portion of the source collision
    storage: one active contact counter, per-contact shape indices, witness
    points, offsets, normal, margins, thread ids, contact force, and optional
    per-contact material properties.
    """

    EXTENDED_ATTRIBUTES: frozenset[str] = frozenset(("force",))

    @classmethod
    def validate_extended_attributes(cls, attributes: tuple[str, ...]) -> None:
        """Validate that requested extended contact attributes are supported by the contact buffer
        implementation.
        """
        if not attributes:
            return

        invalid = sorted(set(attributes).difference(cls.EXTENDED_ATTRIBUTES))
        if invalid:
            allowed = ", ".join(sorted(cls.EXTENDED_ATTRIBUTES))
            bad = ", ".join(invalid)
            raise ValueError(f"Unknown extended contact attribute(s): {bad}. Allowed: {allowed}.")

    def __init__(
        self,
        rigid_contact_max: int,
        *,
        requires_grad: bool = False,
        device: SapDeviceLike = None,
        per_contact_shape_properties: bool = False,
        clear_buffers: bool = False,
        requested_attributes: set[str] | None = None,
    ) -> None:
        self.per_contact_shape_properties = per_contact_shape_properties
        self.clear_buffers = clear_buffers
        requested_attributes = requested_attributes or set()
        self.validate_extended_attributes(tuple(requested_attributes))

        with wp.ScopedDevice(device):
            self._counter_array = wp.zeros(1, dtype=wp.int32)
            self.rigid_contact_count = self._counter_array[0:1]

            self.rigid_contact_point_id = wp.zeros(rigid_contact_max, dtype=wp.int32)
            self.rigid_contact_shape0 = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            self.rigid_contact_shape1 = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            self.rigid_contact_point0 = wp.zeros(rigid_contact_max, dtype=wp.vec3)
            self.rigid_contact_point1 = wp.zeros(rigid_contact_max, dtype=wp.vec3)
            self.rigid_contact_offset0 = wp.zeros(rigid_contact_max, dtype=wp.vec3)
            self.rigid_contact_offset1 = wp.zeros(rigid_contact_max, dtype=wp.vec3)
            self.rigid_contact_normal = wp.zeros(rigid_contact_max, dtype=wp.vec3)
            self.rigid_contact_margin0 = wp.zeros(rigid_contact_max, dtype=wp.float32)
            self.rigid_contact_margin1 = wp.zeros(rigid_contact_max, dtype=wp.float32)
            self.rigid_contact_tids = wp.full(rigid_contact_max, -1, dtype=wp.int32)
            self.rigid_contact_force = wp.zeros(rigid_contact_max, dtype=wp.vec3)

            if self.per_contact_shape_properties:
                self.rigid_contact_stiffness = wp.zeros(rigid_contact_max, dtype=wp.float32)
                self.rigid_contact_damping = wp.zeros(rigid_contact_max, dtype=wp.float32)
                self.rigid_contact_friction = wp.zeros(rigid_contact_max, dtype=wp.float32)
            else:
                self.rigid_contact_stiffness = None
                self.rigid_contact_damping = None
                self.rigid_contact_friction = None

            self.force: wp.array | None = None
            if "force" in requested_attributes:
                self.force = wp.zeros(rigid_contact_max, dtype=wp.spatial_vector, requires_grad=requires_grad)

        self.requires_grad = requires_grad
        self.rigid_contact_max = rigid_contact_max

    def clear(self) -> None:
        """Reset the active contact count and, when configured, clear stale contact payload buffers."""
        self._counter_array.zero_()

        if self.clear_buffers:
            self.rigid_contact_shape0.fill_(-1)
            self.rigid_contact_shape1.fill_(-1)
            self.rigid_contact_tids.fill_(-1)
            self.rigid_contact_force.zero_()

            if self.force is not None:
                self.force.zero_()

            if self.per_contact_shape_properties:
                self.rigid_contact_stiffness.zero_()
                self.rigid_contact_damping.zero_()
                self.rigid_contact_friction.zero_()

    @property
    def device(self):
        """Return the Warp device that owns the contact counter and payload arrays."""
        return self.rigid_contact_count.device

    @property
    def has_rigid_contacts(self) -> bool:
        """Return True because this buffer always owns rigid-contact storage."""
        return self.rigid_contact_count is not None


__all__ = ["SapContacts"]
