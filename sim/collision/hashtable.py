"""Hash table helpers used by SAP collision reduction.

Source note: the SAP modifications in this module are based on Newton's
collision support code and adapted for SAP Warp's broad/narrow-phase stages.
"""

from __future__ import annotations

import warp as wp


_SAP_HASHTABLE_EMPTY_KEY_VALUE = 0xFFFFFFFFFFFFFFFF
SAP_HASHTABLE_EMPTY_KEY = wp.constant(wp.uint64(_SAP_HASHTABLE_EMPTY_KEY_VALUE))
SAP_HASH_MIX_MULTIPLIER = wp.constant(wp.uint64(0xFF51AFD7ED558CCD))


def _sap_next_power_of_two(n: int) -> int:
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


@wp.func
def _sap_hashtable_hash(key: wp.uint64, capacity_mask: int) -> int:
    h = key
    h = h ^ (h >> wp.uint64(33))
    h = h * SAP_HASH_MIX_MULTIPLIER
    h = h ^ (h >> wp.uint64(33))
    return int(h) & capacity_mask


@wp.func
def sap_hashtable_find(
    key: wp.uint64,
    keys: wp.array(dtype=wp.uint64),
) -> int:
    capacity = keys.shape[0]
    capacity_mask = capacity - 1
    idx = _sap_hashtable_hash(key, capacity_mask)

    for _i in range(capacity):
        stored_key = keys[idx]

        if stored_key == key:
            return idx

        if stored_key == SAP_HASHTABLE_EMPTY_KEY:
            return -1

        idx = (idx + 1) & capacity_mask

    return -1


@wp.func
def sap_hashtable_find_or_insert(
    key: wp.uint64,
    keys: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
) -> int:
    capacity = keys.shape[0]
    capacity_mask = capacity - 1
    idx = _sap_hashtable_hash(key, capacity_mask)

    for _i in range(capacity):
        stored_key = keys[idx]

        if stored_key == key:
            return idx

        if stored_key == SAP_HASHTABLE_EMPTY_KEY:
            old_key = wp.atomic_cas(keys, idx, SAP_HASHTABLE_EMPTY_KEY, key)

            if old_key == SAP_HASHTABLE_EMPTY_KEY:
                active_idx = wp.atomic_add(active_slots, capacity, 1)
                if active_idx < capacity:
                    active_slots[active_idx] = idx
                return idx
            elif old_key == key:
                return idx

        idx = (idx + 1) & capacity_mask

    return -1


@wp.kernel
def _sap_hashtable_clear_keys_kernel(
    keys: wp.array(dtype=wp.uint64),
    active_slots: wp.array(dtype=wp.int32),
    capacity: int,
    num_threads: int,
):
    tid = wp.tid()
    count = active_slots[capacity]

    i = tid
    while i < count:
        entry_idx = active_slots[i]
        keys[entry_idx] = SAP_HASHTABLE_EMPTY_KEY
        i += num_threads


@wp.kernel
def _sap_zero_count_kernel(
    active_slots: wp.array(dtype=wp.int32),
    capacity: int,
):
    active_slots[capacity] = 0


class SapHashTable:
    """Concurrent key-to-index map used by SAP collision reduction kernels."""

    def __init__(self, capacity: int, device: str | None = None):
        self.capacity = _sap_next_power_of_two(capacity)
        self.device = device
        self.keys = wp.zeros(self.capacity, dtype=wp.uint64, device=device)
        self.active_slots = wp.zeros(self.capacity + 1, dtype=wp.int32, device=device)
        self.clear()

    def clear(self):
        self.keys.fill_(_SAP_HASHTABLE_EMPTY_KEY_VALUE)
        self.active_slots.zero_()

    def clear_active(self):
        num_threads = min(65536, self.capacity)
        wp.launch(
            _sap_hashtable_clear_keys_kernel,
            dim=num_threads,
            inputs=[self.keys, self.active_slots, self.capacity, num_threads],
            device=self.device,
        )
        wp.launch(
            _sap_zero_count_kernel,
            dim=1,
            inputs=[self.active_slots, self.capacity],
            device=self.device,
        )


__all__ = [
    "SAP_HASHTABLE_EMPTY_KEY",
    "SAP_HASH_MIX_MULTIPLIER",
    "SapHashTable",
    "sap_hashtable_find",
    "sap_hashtable_find_or_insert",
]
