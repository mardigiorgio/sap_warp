"""Blocked Cholesky kernels used by SAP contact solves.

Source note: the SAP modifications in this module are based on NVIDIA Warp's
tile blocked-Cholesky example code and adapted here for batched, masked, and
multi-right-hand-side SAP solve paths:
https://github.com/NVIDIA/warp/blob/main/warp/examples/tile/example_tile_block_cholesky.py
"""

import warp as wp
from functools import cache

_RHS_COL_BLOCK = 4


@cache
def create_blocked_cholesky_kernel(block_size: int, dtype=wp.float64):
    @wp.kernel(module="unique")
    def blocked_cholesky_kernel(
        A: wp.array(dtype=dtype, ndim=2),
        L: wp.array(dtype=dtype, ndim=2),
        active_matrix_size_arr: wp.array(dtype=int, ndim=1),
    ):
        _tid, tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()
        active_matrix_size = active_matrix_size_arr[0]

        n = ((active_matrix_size + block_size - 1) // block_size) * block_size

        for k in range(0, n, block_size):
            end = k + block_size

            A_kk_tile = wp.tile_load(
                A,
                shape=(block_size, block_size),
                offset=(k, k),
                storage="shared",
            )

            if k + block_size > active_matrix_size:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                for it in range(num_iterations):
                    linear_index = tid_block + it * num_threads_per_block
                    linear_index = linear_index % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = A_kk_tile[row, col]
                    if k + row >= active_matrix_size or k + col >= active_matrix_size:
                        value = wp.where(row == col, dtype(1.0), dtype(0.0))
                    A_kk_tile[row, col] = value

            if k > 0:
                for j in range(0, k, block_size):
                    L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(k, j))
                    L_block_T = wp.tile_transpose(L_block)
                    wp.tile_matmul(L_block, L_block_T, A_kk_tile, alpha=-1.0)

            L_kk_tile = wp.tile_cholesky(A_kk_tile)
            wp.tile_store(L, L_kk_tile, offset=(k, k))

            for i in range(end, n, block_size):
                A_ik_tile = wp.tile_load(
                    A,
                    shape=(block_size, block_size),
                    offset=(i, k),
                    storage="shared",
                )

                if i + block_size > active_matrix_size or k + block_size > active_matrix_size:
                    num_tile_elements = block_size * block_size
                    num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                    for it in range(num_iterations):
                        linear_index = tid_block + it * num_threads_per_block
                        linear_index = linear_index % num_tile_elements
                        row = linear_index // block_size
                        col = linear_index % block_size
                        value = A_ik_tile[row, col]
                        if i + row >= active_matrix_size or k + col >= active_matrix_size:
                            value = wp.where(i + row == k + col, dtype(1.0), dtype(0.0))
                        A_ik_tile[row, col] = value

                if k > 0:
                    for j in range(0, k, block_size):
                        L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j))
                        L_2_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(k, j))
                        L_T_tile = wp.tile_transpose(L_2_tile)
                        wp.tile_matmul(L_tile, L_T_tile, A_ik_tile, alpha=-1.0)

                t = wp.tile_transpose(A_ik_tile)
                wp.tile_lower_solve_inplace(L_kk_tile, t)
                sol_tile = wp.tile_transpose(t)
                wp.tile_store(L, sol_tile, offset=(i, k))

    return blocked_cholesky_kernel


@cache
def create_blocked_cholesky_solve_kernel(block_size: int, dtype=wp.float64):
    @wp.kernel(module="unique")
    def blocked_cholesky_solve_kernel(
        L: wp.array(dtype=dtype, ndim=2),
        b: wp.array(dtype=dtype, ndim=2),
        x: wp.array(dtype=dtype, ndim=2),
        y: wp.array(dtype=dtype, ndim=2),
        active_matrix_size_arr: wp.array(dtype=int, ndim=1),
    ):
        active_matrix_size = active_matrix_size_arr[0]

        n = ((active_matrix_size + block_size - 1) // block_size) * block_size

        # Forward substitution: solve L y = b
        for i in range(0, n, block_size):
            rhs_tile = wp.tile_load(b, shape=(block_size, 1), offset=(i, 0))

            if i > 0:
                for j in range(0, i, block_size):
                    L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j))
                    y_block = wp.tile_load(y, shape=(block_size, 1), offset=(j, 0))
                    wp.tile_matmul(L_block, y_block, rhs_tile, alpha=-1.0)

            L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, i))
            wp.tile_lower_solve_inplace(L_tile, rhs_tile)
            wp.tile_store(y, rhs_tile, offset=(i, 0))

        # Backward substitution: solve L^T x = y
        for i in range(n - block_size, -1, -block_size):
            rhs_tile = wp.tile_load(y, shape=(block_size, 1), offset=(i, 0))

            if i + block_size < n:
                for j in range(i + block_size, n, block_size):
                    L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(j, i))
                    L_T_tile = wp.tile_transpose(L_tile)
                    x_tile = wp.tile_load(x, shape=(block_size, 1), offset=(j, 0))
                    wp.tile_matmul(L_T_tile, x_tile, rhs_tile, alpha=-1.0)

            L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, i))
            wp.tile_upper_solve_inplace(wp.tile_transpose(L_tile), rhs_tile)
            wp.tile_store(x, rhs_tile, offset=(i, 0))

    return blocked_cholesky_solve_kernel


def create_blocked_cholesky_forward_solve_kernel_multi_rhs(
    block_size: int,
    max_rhs: int | None = None,
    dtype=wp.float64,
):
    return _create_blocked_cholesky_forward_solve_kernel_multi_rhs(block_size, dtype)


@cache
def _create_blocked_cholesky_forward_solve_kernel_multi_rhs(
    block_size: int,
    dtype=wp.float64,
):
    @wp.kernel(module="unique")
    def blocked_cholesky_forward_solve_kernel_multi_rhs(
        L: wp.array(dtype=dtype, ndim=2),
        b: wp.array(dtype=dtype, ndim=2),
        y: wp.array(dtype=dtype, ndim=2),
        active_matrix_size_arr: wp.array(dtype=int, ndim=1),
        rhs_cols: int,
        max_rhs: int,
    ):
        col_block, _tid_block = wp.tid()
        active_matrix_size = active_matrix_size_arr[0]
        col = col_block * _RHS_COL_BLOCK

        n = ((active_matrix_size + block_size - 1) // block_size) * block_size
        if col >= rhs_cols or col >= max_rhs:
            return

        # Solve L * y = b for many right-hand sides without paying for the
        # backward substitution used by the full A^{-1} solve.
        for i in range(0, n, block_size):
            rhs_tile = wp.tile_load(b, shape=(block_size, _RHS_COL_BLOCK), offset=(i, col))

            if i > 0:
                for j in range(0, i, block_size):
                    L_block = wp.tile_load(L, shape=(block_size, block_size), offset=(i, j))
                    y_block = wp.tile_load(y, shape=(block_size, _RHS_COL_BLOCK), offset=(j, col))
                    wp.tile_matmul(L_block, y_block, rhs_tile, alpha=-1.0)

            L_tile = wp.tile_load(L, shape=(block_size, block_size), offset=(i, i))
            wp.tile_lower_solve_inplace(L_tile, rhs_tile)
            wp.tile_store(y, rhs_tile, offset=(i, col))

    return blocked_cholesky_forward_solve_kernel_multi_rhs


class BlockCholeskySolver:
    def __init__(self, max_num_equations: int, block_size: int = 16, device=None, dtype=wp.float64):
        max_num_equations = ((max_num_equations + block_size - 1) // block_size) * block_size

        self.max_num_equations = max_num_equations
        self.block_size = block_size
        self.device = device
        self.dtype = dtype

        self.num_threads_per_block_factorize = 128
        self.num_threads_per_block_solve = 128 if block_size >= 32 else 64

        self.active_matrix_size_int = -1
        self.active_matrix_size_external = None

        self.cholesky_kernel = create_blocked_cholesky_kernel(block_size, dtype)
        self.solve_kernel = create_blocked_cholesky_solve_kernel(block_size, dtype)
        self.forward_solve_kernel_multi_rhs = create_blocked_cholesky_forward_solve_kernel_multi_rhs(
            block_size,
            self.max_num_equations,
            dtype,
        )

        self.L = wp.zeros(
            shape=(self.max_num_equations, self.max_num_equations),
            dtype=self.dtype,
            device=self.device,
        )
        self.y = wp.zeros(
            shape=(self.max_num_equations, 1),
            dtype=self.dtype,
            device=self.device,
        )
        self.y_multi = wp.zeros(
            shape=(self.max_num_equations, self.max_num_equations),
            dtype=self.dtype,
            device=self.device,
        )

        self.active_matrix_size = wp.zeros(shape=(1,), dtype=int, device=self.device)

    def factorize(self, A, num_active_equations: int):
        assert num_active_equations <= self.max_num_equations

        padded_n = ((num_active_equations + self.block_size - 1) // self.block_size) * self.block_size
        assert A.shape[0] == A.shape[1]
        assert A.shape[0] >= padded_n

        self.active_matrix_size.fill_(int(num_active_equations))
        self.factorize_dynamic(A, self.active_matrix_size)

        self.active_matrix_size_external = None
        self.active_matrix_size_int = num_active_equations

    def factorize_dynamic(
        self,
        A,
        num_active_equations,
    ):
        self.active_matrix_size_external = num_active_equations
        self.active_matrix_size_int = -1

        wp.launch_tiled(
            self.cholesky_kernel,
            dim=1,
            inputs=[A, self.L, num_active_equations],
            block_dim=self.num_threads_per_block_factorize,
            device=self.device,
        )

    def solve(self, rhs, result):
        if self.active_matrix_size_external is not None:
            matrix_size = self.active_matrix_size_external
        else:
            matrix_size = self.active_matrix_size

        wp.launch_tiled(
            self.solve_kernel,
            dim=1,
            inputs=[self.L, rhs, result, self.y, matrix_size],
            block_dim=self.num_threads_per_block_solve,
            device=self.device,
        )

    def solve_lower_multi_rhs(
        self,
        rhs,
        result,
        rhs_cols: int,
    ):
        if rhs_cols <= 0:
            return

        if self.active_matrix_size_external is not None:
            matrix_size = self.active_matrix_size_external
        else:
            matrix_size = self.active_matrix_size

        wp.launch_tiled(
            self.forward_solve_kernel_multi_rhs,
            dim=(rhs_cols + _RHS_COL_BLOCK - 1) // _RHS_COL_BLOCK,
            inputs=[self.L, rhs, result, matrix_size, rhs_cols, self.max_num_equations],
            block_dim=self.num_threads_per_block_solve,
            device=self.device,
        )


@cache
def create_copy_dense_to_padded(dtype=wp.float64):
    @wp.kernel(module="unique")
    def _copy_dense_to_padded_kernel(
        src: wp.array(dtype=dtype, ndim=2),
        dst: wp.array(dtype=dtype, ndim=2),
    ):
        i, j = wp.tid()
        dst[i, j] = src[i, j]

    return _copy_dense_to_padded_kernel


_copy_dense_to_padded = create_copy_dense_to_padded()


@cache
def _get_block_cholesky_workspace(dim: int, block_size: int = 16, device=None, dtype=wp.float64):
    solver = BlockCholeskySolver(dim, block_size=block_size, device=device, dtype=dtype)

    A = wp.zeros(
        shape=(solver.max_num_equations, solver.max_num_equations),
        dtype=dtype,
        device=solver.device,
    )
    b = wp.zeros(shape=(solver.max_num_equations, 1), dtype=dtype, device=solver.device)
    x = wp.zeros(shape=(solver.max_num_equations, 1), dtype=dtype, device=solver.device)

    return solver, A, b, x


@cache
def create_blocked_cholesky_kernel_batched(block_size: int, dtype=wp.float64):
    @wp.kernel(module="unique")
    def blocked_cholesky_kernel_batched(
        A: wp.array(dtype=dtype, ndim=3),
        L: wp.array(dtype=dtype, ndim=3),
        active_matrix_size_arr: wp.array(dtype=int, ndim=1),
    ):
        env, tid_block = wp.tid()
        num_threads_per_block = wp.block_dim()
        active_matrix_size = active_matrix_size_arr[0]

        A_env = A[env]
        L_env = L[env]

        n = ((active_matrix_size + block_size - 1) // block_size) * block_size

        for k in range(0, n, block_size):
            end = k + block_size

            A_kk_tile = wp.tile_load(
                A_env,
                shape=(block_size, block_size),
                offset=(k, k),
                storage="shared",
            )

            if k + block_size > active_matrix_size:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                for it in range(num_iterations):
                    linear_index = tid_block + it * num_threads_per_block
                    linear_index = linear_index % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = A_kk_tile[row, col]
                    if k + row >= active_matrix_size or k + col >= active_matrix_size:
                        value = wp.where(row == col, dtype(1.0), dtype(0.0))
                    A_kk_tile[row, col] = value

            if k > 0:
                for j in range(0, k, block_size):
                    L_block = wp.tile_load(L_env, shape=(block_size, block_size), offset=(k, j))
                    L_block_T = wp.tile_transpose(L_block)
                    wp.tile_matmul(L_block, L_block_T, A_kk_tile, alpha=-1.0)

            L_kk_tile = wp.tile_cholesky(A_kk_tile)
            wp.tile_store(L_env, L_kk_tile, offset=(k, k))

            for i in range(end, n, block_size):
                A_ik_tile = wp.tile_load(
                    A_env,
                    shape=(block_size, block_size),
                    offset=(i, k),
                    storage="shared",
                )

                if i + block_size > active_matrix_size or k + block_size > active_matrix_size:
                    num_tile_elements = block_size * block_size
                    num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                    for it in range(num_iterations):
                        linear_index = tid_block + it * num_threads_per_block
                        linear_index = linear_index % num_tile_elements
                        row = linear_index // block_size
                        col = linear_index % block_size
                        value = A_ik_tile[row, col]
                        if i + row >= active_matrix_size or k + col >= active_matrix_size:
                            value = wp.where(i + row == k + col, dtype(1.0), dtype(0.0))
                        A_ik_tile[row, col] = value

                if k > 0:
                    for j in range(0, k, block_size):
                        L_tile = wp.tile_load(L_env, shape=(block_size, block_size), offset=(i, j))
                        L_2_tile = wp.tile_load(L_env, shape=(block_size, block_size), offset=(k, j))
                        L_T_tile = wp.tile_transpose(L_2_tile)
                        wp.tile_matmul(L_tile, L_T_tile, A_ik_tile, alpha=-1.0)

                t = wp.tile_transpose(A_ik_tile)
                wp.tile_lower_solve_inplace(L_kk_tile, t)
                sol_tile = wp.tile_transpose(t)
                wp.tile_store(L_env, sol_tile, offset=(i, k))

    return blocked_cholesky_kernel_batched


@cache
def create_blocked_cholesky_kernel_batched_masked(block_size: int, dtype=wp.float64):
    @wp.kernel(module="unique")
    def blocked_cholesky_kernel_batched_masked(
        A: wp.array(dtype=dtype, ndim=3),
        L: wp.array(dtype=dtype, ndim=3),
        active_matrix_size_arr: wp.array(dtype=int, ndim=1),
        env_active: wp.array(dtype=int),
    ):
        env, tid_block = wp.tid()
        if env_active[env] == 0:
            return

        num_threads_per_block = wp.block_dim()
        active_matrix_size = active_matrix_size_arr[0]

        A_env = A[env]
        L_env = L[env]

        n = ((active_matrix_size + block_size - 1) // block_size) * block_size

        for k in range(0, n, block_size):
            end = k + block_size

            A_kk_tile = wp.tile_load(
                A_env,
                shape=(block_size, block_size),
                offset=(k, k),
                storage="shared",
            )

            if k + block_size > active_matrix_size:
                num_tile_elements = block_size * block_size
                num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                for it in range(num_iterations):
                    linear_index = tid_block + it * num_threads_per_block
                    linear_index = linear_index % num_tile_elements
                    row = linear_index // block_size
                    col = linear_index % block_size
                    value = A_kk_tile[row, col]
                    if k + row >= active_matrix_size or k + col >= active_matrix_size:
                        value = wp.where(row == col, dtype(1.0), dtype(0.0))
                    A_kk_tile[row, col] = value

            if k > 0:
                for j in range(0, k, block_size):
                    L_block = wp.tile_load(L_env, shape=(block_size, block_size), offset=(k, j))
                    L_block_T = wp.tile_transpose(L_block)
                    wp.tile_matmul(L_block, L_block_T, A_kk_tile, alpha=-1.0)

            L_kk_tile = wp.tile_cholesky(A_kk_tile)
            wp.tile_store(L_env, L_kk_tile, offset=(k, k))

            for i in range(end, n, block_size):
                A_ik_tile = wp.tile_load(
                    A_env,
                    shape=(block_size, block_size),
                    offset=(i, k),
                    storage="shared",
                )

                if i + block_size > active_matrix_size or k + block_size > active_matrix_size:
                    num_tile_elements = block_size * block_size
                    num_iterations = (num_tile_elements + num_threads_per_block - 1) // num_threads_per_block
                    for it in range(num_iterations):
                        linear_index = tid_block + it * num_threads_per_block
                        linear_index = linear_index % num_tile_elements
                        row = linear_index // block_size
                        col = linear_index % block_size
                        value = A_ik_tile[row, col]
                        if i + row >= active_matrix_size or k + col >= active_matrix_size:
                            value = wp.where(i + row == k + col, dtype(1.0), dtype(0.0))
                        A_ik_tile[row, col] = value

                if k > 0:
                    for j in range(0, k, block_size):
                        L_tile = wp.tile_load(L_env, shape=(block_size, block_size), offset=(i, j))
                        L_2_tile = wp.tile_load(L_env, shape=(block_size, block_size), offset=(k, j))
                        L_T_tile = wp.tile_transpose(L_2_tile)
                        wp.tile_matmul(L_tile, L_T_tile, A_ik_tile, alpha=-1.0)

                t = wp.tile_transpose(A_ik_tile)
                wp.tile_lower_solve_inplace(L_kk_tile, t)
                sol_tile = wp.tile_transpose(t)
                wp.tile_store(L_env, sol_tile, offset=(i, k))

    return blocked_cholesky_kernel_batched_masked


@cache
def create_blocked_cholesky_solve_kernel_batched(block_size: int, dtype=wp.float64):
    @wp.kernel(module="unique")
    def blocked_cholesky_solve_kernel_batched(
        L: wp.array(dtype=dtype, ndim=3),
        b: wp.array(dtype=dtype, ndim=3),
        x: wp.array(dtype=dtype, ndim=3),
        y: wp.array(dtype=dtype, ndim=3),
        active_matrix_size_arr: wp.array(dtype=int, ndim=1),
    ):
        env, _tid_block = wp.tid()
        active_matrix_size = active_matrix_size_arr[0]

        L_env = L[env]
        b_env = b[env]
        x_env = x[env]
        y_env = y[env]

        n = ((active_matrix_size + block_size - 1) // block_size) * block_size

        for i in range(0, n, block_size):
            rhs_tile = wp.tile_load(b_env, shape=(block_size, 1), offset=(i, 0))

            if i > 0:
                for j in range(0, i, block_size):
                    L_block = wp.tile_load(L_env, shape=(block_size, block_size), offset=(i, j))
                    y_block = wp.tile_load(y_env, shape=(block_size, 1), offset=(j, 0))
                    wp.tile_matmul(L_block, y_block, rhs_tile, alpha=-1.0)

            L_tile = wp.tile_load(L_env, shape=(block_size, block_size), offset=(i, i))
            wp.tile_lower_solve_inplace(L_tile, rhs_tile)
            wp.tile_store(y_env, rhs_tile, offset=(i, 0))

        for i in range(n - block_size, -1, -block_size):
            rhs_tile = wp.tile_load(y_env, shape=(block_size, 1), offset=(i, 0))

            if i + block_size < n:
                for j in range(i + block_size, n, block_size):
                    L_tile = wp.tile_load(L_env, shape=(block_size, block_size), offset=(j, i))
                    L_T_tile = wp.tile_transpose(L_tile)
                    x_tile = wp.tile_load(x_env, shape=(block_size, 1), offset=(j, 0))
                    wp.tile_matmul(L_T_tile, x_tile, rhs_tile, alpha=-1.0)

            L_tile = wp.tile_load(L_env, shape=(block_size, block_size), offset=(i, i))
            wp.tile_upper_solve_inplace(wp.tile_transpose(L_tile), rhs_tile)
            wp.tile_store(x_env, rhs_tile, offset=(i, 0))

    return blocked_cholesky_solve_kernel_batched


@cache
def create_blocked_cholesky_solve_kernel_batched_masked(block_size: int, dtype=wp.float64):
    @wp.kernel(module="unique")
    def blocked_cholesky_solve_kernel_batched_masked(
        L: wp.array(dtype=dtype, ndim=3),
        b: wp.array(dtype=dtype, ndim=3),
        x: wp.array(dtype=dtype, ndim=3),
        y: wp.array(dtype=dtype, ndim=3),
        active_matrix_size_arr: wp.array(dtype=int, ndim=1),
        env_active: wp.array(dtype=int),
    ):
        env, _tid_block = wp.tid()
        if env_active[env] == 0:
            return

        active_matrix_size = active_matrix_size_arr[0]

        L_env = L[env]
        b_env = b[env]
        x_env = x[env]
        y_env = y[env]

        n = ((active_matrix_size + block_size - 1) // block_size) * block_size

        for i in range(0, n, block_size):
            rhs_tile = wp.tile_load(b_env, shape=(block_size, 1), offset=(i, 0))

            if i > 0:
                for j in range(0, i, block_size):
                    L_block = wp.tile_load(L_env, shape=(block_size, block_size), offset=(i, j))
                    y_block = wp.tile_load(y_env, shape=(block_size, 1), offset=(j, 0))
                    wp.tile_matmul(L_block, y_block, rhs_tile, alpha=-1.0)

            L_tile = wp.tile_load(L_env, shape=(block_size, block_size), offset=(i, i))
            wp.tile_lower_solve_inplace(L_tile, rhs_tile)
            wp.tile_store(y_env, rhs_tile, offset=(i, 0))

        for i in range(n - block_size, -1, -block_size):
            rhs_tile = wp.tile_load(y_env, shape=(block_size, 1), offset=(i, 0))

            if i + block_size < n:
                for j in range(i + block_size, n, block_size):
                    L_tile = wp.tile_load(L_env, shape=(block_size, block_size), offset=(j, i))
                    L_T_tile = wp.tile_transpose(L_tile)
                    x_tile = wp.tile_load(x_env, shape=(block_size, 1), offset=(j, 0))
                    wp.tile_matmul(L_T_tile, x_tile, rhs_tile, alpha=-1.0)

            L_tile = wp.tile_load(L_env, shape=(block_size, block_size), offset=(i, i))
            wp.tile_upper_solve_inplace(wp.tile_transpose(L_tile), rhs_tile)
            wp.tile_store(x_env, rhs_tile, offset=(i, 0))

    return blocked_cholesky_solve_kernel_batched_masked


def create_blocked_cholesky_solve_kernel_batched_multi_rhs(
    block_size: int,
    max_rhs: int | None = None,
    dtype=wp.float64,
):
    return _create_blocked_cholesky_solve_kernel_batched_multi_rhs(block_size, dtype)


@cache
def _create_blocked_cholesky_solve_kernel_batched_multi_rhs(
    block_size: int,
    dtype=wp.float64,
):
    @wp.kernel(module="unique")
    def blocked_cholesky_solve_kernel_batched_multi_rhs(
        L: wp.array(dtype=dtype, ndim=3),
        b: wp.array(dtype=dtype, ndim=3),
        x: wp.array(dtype=dtype, ndim=3),
        y: wp.array(dtype=dtype, ndim=3),
        active_matrix_size_arr: wp.array(dtype=int, ndim=1),
        rhs_cols: int,
        max_rhs: int,
    ):
        env, col_block, _tid_block = wp.tid()
        active_matrix_size = active_matrix_size_arr[0]
        col = col_block * _RHS_COL_BLOCK

        L_env = L[env]
        b_env = b[env]
        x_env = x[env]
        y_env = y[env]

        n = ((active_matrix_size + block_size - 1) // block_size) * block_size
        if col >= rhs_cols or col >= max_rhs:
            return

        for i in range(0, n, block_size):
            rhs_tile = wp.tile_load(b_env, shape=(block_size, _RHS_COL_BLOCK), offset=(i, col))

            if i > 0:
                for j in range(0, i, block_size):
                    L_block = wp.tile_load(L_env, shape=(block_size, block_size), offset=(i, j))
                    y_block = wp.tile_load(y_env, shape=(block_size, _RHS_COL_BLOCK), offset=(j, col))
                    wp.tile_matmul(L_block, y_block, rhs_tile, alpha=-1.0)

            L_tile = wp.tile_load(L_env, shape=(block_size, block_size), offset=(i, i))
            wp.tile_lower_solve_inplace(L_tile, rhs_tile)
            wp.tile_store(y_env, rhs_tile, offset=(i, col))

        for i in range(n - block_size, -1, -block_size):
            rhs_tile = wp.tile_load(y_env, shape=(block_size, _RHS_COL_BLOCK), offset=(i, col))

            if i + block_size < n:
                for j in range(i + block_size, n, block_size):
                    L_tile = wp.tile_load(L_env, shape=(block_size, block_size), offset=(j, i))
                    L_T_tile = wp.tile_transpose(L_tile)
                    x_tile = wp.tile_load(x_env, shape=(block_size, _RHS_COL_BLOCK), offset=(j, col))
                    wp.tile_matmul(L_T_tile, x_tile, rhs_tile, alpha=-1.0)

            L_tile = wp.tile_load(L_env, shape=(block_size, block_size), offset=(i, i))
            wp.tile_upper_solve_inplace(wp.tile_transpose(L_tile), rhs_tile)
            wp.tile_store(x_env, rhs_tile, offset=(i, col))

    return blocked_cholesky_solve_kernel_batched_multi_rhs


def create_blocked_cholesky_forward_solve_kernel_batched_multi_rhs(
    block_size: int,
    max_rhs: int | None = None,
    dtype=wp.float64,
):
    return _create_blocked_cholesky_forward_solve_kernel_batched_multi_rhs(block_size, dtype)


@cache
def _create_blocked_cholesky_forward_solve_kernel_batched_multi_rhs(
    block_size: int,
    dtype=wp.float64,
):
    @wp.kernel(module="unique")
    def blocked_cholesky_forward_solve_kernel_batched_multi_rhs(
        L: wp.array(dtype=dtype, ndim=3),
        b: wp.array(dtype=dtype, ndim=3),
        y: wp.array(dtype=dtype, ndim=3),
        active_matrix_size_arr: wp.array(dtype=int, ndim=1),
        rhs_cols: int,
        max_rhs: int,
    ):
        env, col_block, _tid_block = wp.tid()
        active_matrix_size = active_matrix_size_arr[0]
        col = col_block * _RHS_COL_BLOCK

        L_env = L[env]
        b_env = b[env]
        y_env = y[env]

        n = ((active_matrix_size + block_size - 1) // block_size) * block_size
        if col >= rhs_cols or col >= max_rhs:
            return

        # Solve L * y = b for many right-hand sides and keep the lower-triangular
        # inverse factors needed to recover diag(A^{-1}) exactly.
        for i in range(0, n, block_size):
            rhs_tile = wp.tile_load(b_env, shape=(block_size, _RHS_COL_BLOCK), offset=(i, col))

            if i > 0:
                for j in range(0, i, block_size):
                    L_block = wp.tile_load(L_env, shape=(block_size, block_size), offset=(i, j))
                    y_block = wp.tile_load(y_env, shape=(block_size, _RHS_COL_BLOCK), offset=(j, col))
                    wp.tile_matmul(L_block, y_block, rhs_tile, alpha=-1.0)

            L_tile = wp.tile_load(L_env, shape=(block_size, block_size), offset=(i, i))
            wp.tile_lower_solve_inplace(L_tile, rhs_tile)
            wp.tile_store(y_env, rhs_tile, offset=(i, col))

    return blocked_cholesky_forward_solve_kernel_batched_multi_rhs


class BlockCholeskySolverBatched:
    def __init__(
        self,
        max_num_equations: int,
        batch_size: int,
        block_size: int = 16,
        device=None,
        dtype=wp.float64,
    ):
        max_num_equations = ((max_num_equations + block_size - 1) // block_size) * block_size

        self.max_num_equations = max_num_equations
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype

        self.num_threads_per_block_factorize = 128
        self.num_threads_per_block_solve = 128 if block_size >= 32 else 64

        self.active_matrix_size_int = -1
        self.active_matrix_size_external = None

        self.cholesky_kernel = create_blocked_cholesky_kernel_batched(block_size, dtype)
        self.cholesky_kernel_masked = create_blocked_cholesky_kernel_batched_masked(block_size, dtype)
        self.solve_kernel = create_blocked_cholesky_solve_kernel_batched(block_size, dtype)
        self.solve_kernel_masked = create_blocked_cholesky_solve_kernel_batched_masked(block_size, dtype)
        self.solve_kernel_multi_rhs = create_blocked_cholesky_solve_kernel_batched_multi_rhs(
            block_size,
            self.max_num_equations,
            dtype,
        )
        self.forward_solve_kernel_multi_rhs = create_blocked_cholesky_forward_solve_kernel_batched_multi_rhs(
            block_size,
            self.max_num_equations,
            dtype,
        )

        self.L = wp.zeros(
            shape=(self.batch_size, self.max_num_equations, self.max_num_equations),
            dtype=self.dtype,
            device=self.device,
        )
        self.y = wp.zeros(
            shape=(self.batch_size, self.max_num_equations, 1),
            dtype=self.dtype,
            device=self.device,
        )
        self.y_multi = wp.zeros(
            shape=(self.batch_size, self.max_num_equations, self.max_num_equations),
            dtype=self.dtype,
            device=self.device,
        )

        self.active_matrix_size = wp.zeros(shape=(1,), dtype=int, device=self.device)

    def factorize(self, A, num_active_equations: int):
        assert num_active_equations <= self.max_num_equations

        padded_n = ((num_active_equations + self.block_size - 1) // self.block_size) * self.block_size
        assert A.shape[0] == self.batch_size
        assert A.shape[1] == A.shape[2]
        assert A.shape[1] >= padded_n

        self.active_matrix_size.fill_(int(num_active_equations))
        self.factorize_dynamic(A, self.active_matrix_size)

        self.active_matrix_size_external = None
        self.active_matrix_size_int = num_active_equations

    def factorize_dynamic(
        self,
        A,
        num_active_equations,
    ):
        self.active_matrix_size_external = num_active_equations
        self.active_matrix_size_int = -1

        wp.launch_tiled(
            self.cholesky_kernel,
            dim=self.batch_size,
            inputs=[A, self.L, num_active_equations],
            block_dim=self.num_threads_per_block_factorize,
            device=self.device,
        )

    def solve(self, rhs, result):
        if self.active_matrix_size_external is not None:
            matrix_size = self.active_matrix_size_external
        else:
            matrix_size = self.active_matrix_size

        wp.launch_tiled(
            self.solve_kernel,
            dim=self.batch_size,
            inputs=[self.L, rhs, result, self.y, matrix_size],
            block_dim=self.num_threads_per_block_solve,
            device=self.device,
        )

    def factorize_masked(self, A, num_active_equations: int, env_active):
        assert num_active_equations <= self.max_num_equations

        padded_n = ((num_active_equations + self.block_size - 1) // self.block_size) * self.block_size
        assert A.shape[0] == self.batch_size
        assert A.shape[1] == A.shape[2]
        assert A.shape[1] >= padded_n

        self.active_matrix_size.fill_(int(num_active_equations))
        self.active_matrix_size_external = None
        self.active_matrix_size_int = num_active_equations

        wp.launch_tiled(
            self.cholesky_kernel_masked,
            dim=self.batch_size,
            inputs=[A, self.L, self.active_matrix_size, env_active],
            block_dim=self.num_threads_per_block_factorize,
            device=self.device,
        )

    def solve_masked(self, rhs, result, env_active):
        if self.active_matrix_size_external is not None:
            matrix_size = self.active_matrix_size_external
        else:
            matrix_size = self.active_matrix_size

        wp.launch_tiled(
            self.solve_kernel_masked,
            dim=self.batch_size,
            inputs=[self.L, rhs, result, self.y, matrix_size, env_active],
            block_dim=self.num_threads_per_block_solve,
            device=self.device,
        )

    def solve_multi_rhs(
        self,
        rhs,
        result,
        rhs_cols: int,
    ):
        if rhs_cols <= 0:
            return

        if self.active_matrix_size_external is not None:
            matrix_size = self.active_matrix_size_external
        else:
            matrix_size = self.active_matrix_size

        wp.launch_tiled(
            self.solve_kernel_multi_rhs,
            dim=(self.batch_size, (rhs_cols + _RHS_COL_BLOCK - 1) // _RHS_COL_BLOCK),
            inputs=[self.L, rhs, result, self.y_multi, matrix_size, rhs_cols, self.max_num_equations],
            block_dim=self.num_threads_per_block_solve,
            device=self.device,
        )

    def solve_lower_multi_rhs(
        self,
        rhs,
        result,
        rhs_cols: int,
    ):
        if rhs_cols <= 0:
            return

        if self.active_matrix_size_external is not None:
            matrix_size = self.active_matrix_size_external
        else:
            matrix_size = self.active_matrix_size

        wp.launch_tiled(
            self.forward_solve_kernel_multi_rhs,
            dim=(self.batch_size, (rhs_cols + _RHS_COL_BLOCK - 1) // _RHS_COL_BLOCK),
            inputs=[self.L, rhs, result, matrix_size, rhs_cols, self.max_num_equations],
            block_dim=self.num_threads_per_block_solve,
            device=self.device,
        )


@cache
def _get_block_cholesky_workspace_batched(
    dim: int,
    num_envs: int,
    block_size: int = 16,
    device=None,
    dtype=wp.float64,
):
    solver = BlockCholeskySolverBatched(
        dim,
        batch_size=num_envs,
        block_size=block_size,
        device=device,
        dtype=dtype,
    )

    A = wp.zeros(
        shape=(num_envs, solver.max_num_equations, solver.max_num_equations),
        dtype=dtype,
        device=solver.device,
    )
    b = wp.zeros(
        shape=(num_envs, solver.max_num_equations, 1),
        dtype=dtype,
        device=solver.device,
    )
    x = wp.zeros(
        shape=(num_envs, solver.max_num_equations, 1),
        dtype=dtype,
        device=solver.device,
    )

    return solver, A, b, x
