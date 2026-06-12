"""MPR helpers for convex collision detection.

Source note: the SAP modifications in this module are based on Newton's
convex collision code and adapted for SAP Warp's narrow-phase stages.
"""

from __future__ import annotations

from typing import Any

import warp as wp


@wp.struct
class SapMprVert:
    """Vertex structure for MPR containing points on both shapes."""

    B: wp.vec3
    BtoA: wp.vec3


@wp.func
def sap_mpr_vert_a(vert: SapMprVert) -> wp.vec3:
    return vert.B + vert.BtoA


def sap_create_support_map_function(support_func: Any):
    @wp.func
    def sap_support_map_b(
        geom_b: Any,
        direction: wp.vec3,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        data_provider: Any,
    ) -> wp.vec3:
        tmp = wp.quat_rotate_inv(orientation_b, direction)
        result = support_func(geom_b, tmp, data_provider)
        result = wp.quat_rotate(orientation_b, result)
        result = result + position_b
        return result

    @wp.func
    def sap_minkowski_support(
        geom_a: Any,
        geom_b: Any,
        direction: wp.vec3,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        extend: float,
        data_provider: Any,
    ) -> SapMprVert:
        v = SapMprVert()

        point_a = support_func(geom_a, direction, data_provider)

        tmp_direction = -direction
        v.B = sap_support_map_b(geom_b, tmp_direction, orientation_b, position_b, data_provider)

        if extend != 0.0:
            d = wp.normalize(direction) * extend * 0.5
            point_a = point_a + d
            v.B = v.B - d

        v.BtoA = point_a - v.B

        return v

    @wp.func
    def sap_geometric_center(
        geom_a: Any,
        geom_b: Any,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        data_provider: Any,
    ) -> SapMprVert:
        center = SapMprVert()
        center.B = position_b
        center.BtoA = -position_b
        return center

    return sap_support_map_b, sap_minkowski_support, sap_geometric_center


def sap_create_solve_mpr(support_func: Any, _support_funcs: Any = None):
    if _support_funcs is not None:
        _sap_support_map_b, sap_minkowski_support, sap_geometric_center = _support_funcs
    else:
        _sap_support_map_b, sap_minkowski_support, sap_geometric_center = sap_create_support_map_function(support_func)

    @wp.func
    def sap_solve_mpr_core(
        geom_a: Any,
        geom_b: Any,
        orientation_b: wp.quat,
        position_b: wp.vec3,
        extend: float,
        data_provider: Any,
        MAX_ITER: int = 30,
        COLLIDE_EPSILON: float = 1e-5,
    ) -> tuple[bool, wp.vec3, wp.vec3, wp.vec3, float]:
        NUMERIC_EPSILON = 1e-16

        penetration = float(0.0)
        point_a = wp.vec3(0.0, 0.0, 0.0)
        point_b = wp.vec3(0.0, 0.0, 0.0)
        normal = wp.vec3(0.0, 0.0, 0.0)

        v0 = sap_geometric_center(geom_a, geom_b, orientation_b, position_b, data_provider)

        normal = v0.BtoA
        if (
            wp.abs(normal[0]) < NUMERIC_EPSILON
            and wp.abs(normal[1]) < NUMERIC_EPSILON
            and wp.abs(normal[2]) < NUMERIC_EPSILON
        ):
            v0.BtoA = wp.vec3(1e-05, 0.0, 0.0)

        normal = -v0.BtoA

        v1 = sap_minkowski_support(geom_a, geom_b, normal, orientation_b, position_b, extend, data_provider)

        point_a = sap_mpr_vert_a(v1)
        point_b = v1.B

        if wp.dot(v1.BtoA, normal) <= 0.0:
            return False, point_a, point_b, normal, penetration

        normal = wp.cross(v1.BtoA, v0.BtoA)

        if wp.length_sq(normal) < NUMERIC_EPSILON * NUMERIC_EPSILON:
            normal = v1.BtoA - v0.BtoA
            normal = wp.normalize(normal)

            temp1 = v1.BtoA
            penetration = wp.dot(temp1, normal)

            return True, point_a, point_b, normal, penetration

        v2 = sap_minkowski_support(geom_a, geom_b, normal, orientation_b, position_b, extend, data_provider)

        if wp.dot(v2.BtoA, normal) <= 0.0:
            return False, point_a, point_b, normal, penetration

        temp1 = v1.BtoA - v0.BtoA
        temp2 = v2.BtoA - v0.BtoA
        normal = wp.cross(temp1, temp2)

        dist = wp.dot(normal, v0.BtoA)

        if dist > 0.0:
            tmp_b = v1.B
            tmp_btoa = v1.BtoA
            v1.B = v2.B
            v1.BtoA = v2.BtoA
            v2.B = tmp_b
            v2.BtoA = tmp_btoa
            normal = -normal

        phase1 = int(0)
        phase2 = int(0)
        hit = bool(False)

        v3 = SapMprVert()
        while True:
            if phase1 > MAX_ITER:
                return False, point_a, point_b, normal, penetration

            phase1 += 1

            v3 = sap_minkowski_support(geom_a, geom_b, normal, orientation_b, position_b, extend, data_provider)

            if wp.dot(v3.BtoA, normal) <= 0.0:
                return False, point_a, point_b, normal, penetration

            temp1 = wp.cross(v1.BtoA, v3.BtoA)
            if wp.dot(temp1, v0.BtoA) < 0.0:
                v2 = v3
                temp1 = v1.BtoA - v0.BtoA
                temp2 = v3.BtoA - v0.BtoA
                normal = wp.cross(temp1, temp2)
                continue

            temp1 = wp.cross(v3.BtoA, v2.BtoA)
            if wp.dot(temp1, v0.BtoA) < 0.0:
                v1 = v3
                temp1 = v3.BtoA - v0.BtoA
                temp2 = v2.BtoA - v0.BtoA
                normal = wp.cross(temp1, temp2)
                continue

            break

        v4 = SapMprVert()
        while True:
            phase2 += 1

            temp1 = v2.BtoA - v1.BtoA
            temp2 = v3.BtoA - v1.BtoA
            normal = wp.cross(temp1, temp2)

            normal_sq = wp.length_sq(normal)

            if normal_sq < NUMERIC_EPSILON * NUMERIC_EPSILON:
                return False, point_a, point_b, normal, penetration

            if not hit:
                d = wp.dot(normal, v1.BtoA)
                hit = d >= 0.0

            v4 = sap_minkowski_support(geom_a, geom_b, normal, orientation_b, position_b, extend, data_provider)

            temp3 = v4.BtoA - v3.BtoA
            delta = wp.dot(temp3, normal)
            penetration = wp.dot(v4.BtoA, normal)

            if (
                delta * delta <= COLLIDE_EPSILON * COLLIDE_EPSILON * normal_sq
                or penetration <= 0.0
                or phase2 > MAX_ITER
            ):
                if hit:
                    inv_normal = 1.0 / wp.sqrt(normal_sq)
                    penetration *= inv_normal
                    normal = normal * inv_normal

                    temp3 = wp.cross(v1.BtoA, temp1)
                    gamma = wp.dot(temp3, normal) * inv_normal
                    temp3 = wp.cross(temp2, v1.BtoA)
                    beta = wp.dot(temp3, normal) * inv_normal
                    alpha = 1.0 - gamma - beta

                    point_a = alpha * sap_mpr_vert_a(v1) + beta * sap_mpr_vert_a(v2) + gamma * sap_mpr_vert_a(v3)
                    point_b = alpha * v1.B + beta * v2.B + gamma * v3.B

                return hit, point_a, point_b, normal, penetration

            temp1 = wp.cross(v4.BtoA, v0.BtoA)
            dot = wp.dot(temp1, v1.BtoA)

            if dot >= 0.0:
                dot = wp.dot(temp1, v2.BtoA)
                if dot >= 0.0:
                    v1 = v4
                else:
                    v3 = v4
            else:
                dot = wp.dot(temp1, v3.BtoA)
                if dot >= 0.0:
                    v2 = v4
                else:
                    v1 = v4

    @wp.func
    def sap_solve_mpr(
        geom_a: Any,
        geom_b: Any,
        orientation_a: wp.quat,
        orientation_b: wp.quat,
        position_a: wp.vec3,
        position_b: wp.vec3,
        sum_of_contact_offsets: float,
        data_provider: Any,
        MAX_ITER: int = 30,
        COLLIDE_EPSILON: float = 1e-5,
    ) -> tuple[bool, float, wp.vec3, wp.vec3]:
        relative_orientation_b = wp.quat_inverse(orientation_a) * orientation_b
        relative_position_b = wp.quat_rotate_inv(orientation_a, position_b - position_a)

        result = sap_solve_mpr_core(
            geom_a,
            geom_b,
            relative_orientation_b,
            relative_position_b,
            sum_of_contact_offsets,
            data_provider,
            MAX_ITER,
            COLLIDE_EPSILON,
        )

        collision, point_a, point_b, normal, penetration = result

        point = 0.5 * (point_a + point_b)

        point = wp.quat_rotate(orientation_a, point) + position_a
        normal = wp.quat_rotate(orientation_a, normal)

        signed_distance = -penetration

        return collision, signed_distance, point, normal

    sap_solve_mpr.core = sap_solve_mpr_core
    return sap_solve_mpr


__all__ = [
    "SapMprVert",
    "sap_create_solve_mpr",
    "sap_create_support_map_function",
    "sap_mpr_vert_a",
]
