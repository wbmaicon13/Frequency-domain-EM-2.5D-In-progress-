"""
FEM 연립방정식 풀이

Fortran 대응:
  Fem25Dfwd.f90의 FWD2 + Make_BC + apply_bc + Intel MKL ZGBTRF/ZGBTRS

풀이 절차 (Fortran과 동일):
  1. 전역 강성행렬 K, 힘벡터 f 조립 (fem_assembly.py)
  2. Robin(임피던스) 경계 조건 적용: K += K_robin (경계 적분)
  3. 전체 시스템 K·x = f 풀이 (Dirichlet 제거 없음!)
  4. 해벡터에서 Ey_s, Hy_s 분리

풀이 방법:
  - ILU+GMRES (기본): 불완전 LU 전처리 + GMRES 반복법, ~0.8초/풀이
  - PARDISO (실수분리): complex→real 2×2 블록 확장 후 MKL PARDISO, ~1.5초/풀이
  - GPU: CuPy sparse solver (fallback)

Fortran에서는 n_bc=0 (Dirichlet 없음), n_vc>0 (Robin BC만 사용).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ..mesh.boundary import apply_robin_boundary
from ..constants import MU_0, EPSILON_0


def solve_fem_system(
    K_global,
    f_global: np.ndarray,
    grid,
    element_resistivity: np.ndarray,
    omega: float,
    ky: float,
    use_gpu: bool = False,
    solver: str = "direct",
    tol: float = 1e-10,
    max_iter: int = 500,
    mu: float = MU_0,
    epsilon: float = EPSILON_0,
) -> np.ndarray:
    """
    Robin BC 적용 후 FEM 시스템 풀이

    Fortran 대응: Make_BC → apply_bc → ZGBTRF/ZGBTRS

    DOF 구조:
      전역 DOF = 2 × n_nodes
      짝수 인덱스(0,2,4,...): Ey 성분 DOF
      홀수 인덱스(1,3,5,...): Hy 성분 DOF
    """
    # Robin BC 적용
    f_bc = f_global.copy()
    K_bc, f_bc = apply_robin_boundary(
        K_global, f_bc, grid, element_resistivity, omega, ky, mu, epsilon)

    # 풀이
    if use_gpu:
        solution = _solve_gpu(K_bc, f_bc)
    elif solver == "direct":
        # ILU 전처리 + GMRES (가장 빠름)
        solution = _solve_ilu_gmres(K_bc, f_bc, tol, max_iter, ky=ky)
    elif solver == "pardiso":
        solution = _solve_pardiso_real_split(K_bc, f_bc)
    elif solver == "gmres":
        solution, info = spla.gmres(K_bc, f_bc, rtol=tol, maxiter=max_iter)
        if info != 0:
            raise RuntimeError(f"GMRES 수렴 실패 (info={info})")
    elif solver == "bicgstab":
        solution, info = spla.bicgstab(K_bc, f_bc, rtol=tol, maxiter=max_iter)
        if info != 0:
            raise RuntimeError(f"BiCGSTAB 수렴 실패 (info={info})")
    else:
        raise ValueError(f"알 수 없는 solver: {solver!r}")

    return solution


def build_robin_stiffness(
    K_global,
    grid,
    element_resistivity: np.ndarray,
    omega: float,
    ky: float,
    mu: float = MU_0,
    epsilon: float = EPSILON_0,
):
    """
    K 행렬에 Robin BC를 적용한 K_bc 반환 (f 벡터 불필요)

    Robin BC의 fe_bc는 항상 0 (homogeneous) 이므로
    K만 수정하면 됨. 송신기(tx)에 독립적.
    """
    f_dummy = np.zeros(K_global.shape[0], dtype=complex)
    K_bc, _ = apply_robin_boundary(
        K_global, f_dummy, grid, element_resistivity, omega, ky, mu, epsilon)
    return K_bc


def factorize_system(
    K_bc,
    ky: float = None,
    solver: str = "direct",
    use_gpu: bool = False,
):
    """
    K_bc를 사전 분해하여 여러 RHS에 재사용 가능한 solver 객체 반환

    K-reuse 최적화: (freq, ky)당 1회 분해, tx 수만큼 solve 호출

    반환:
      solve_fn : callable(f) → solution
    """
    if use_gpu:
        return _factorize_gpu(K_bc)

    if solver == "pardiso":
        return _factorize_pardiso(K_bc)

    K_csc = K_bc.tocsc()

    if solver == "band":
        # Banded LU 분해 (LAPACK zgbtrf) — Fortran 동일, 최적 성능
        return _factorize_banded(K_bc)

    if solver == "splu":
        # 직접 LU 분해 (SuperLU) — 느림, 비권장
        lu = spla.splu(K_csc)
        return lu.solve

    # ILU + GMRES 방식: ILU 전처리기를 미리 계산
    if ky is not None and abs(ky) < 1e-4:
        drop_tol, fill_factor = 1e-8, 100
        gmres_tol = 1e-8
    else:
        drop_tol, fill_factor = 1e-4, 20
        gmres_tol = 1e-10

    ilu = spla.spilu(K_csc, drop_tol=drop_tol, fill_factor=fill_factor)
    M = spla.LinearOperator(K_csc.shape, ilu.solve)

    def solve_fn(f):
        solution, info = spla.gmres(
            K_csc, f, M=M, rtol=gmres_tol, maxiter=500, restart=50)
        if info != 0:
            ilu2 = spla.spilu(K_csc, drop_tol=1e-10, fill_factor=200)
            M2 = spla.LinearOperator(K_csc.shape, ilu2.solve)
            solution, info = spla.gmres(
                K_csc, f, M=M2, rtol=gmres_tol, maxiter=2000, restart=100)
            if info != 0:
                raise RuntimeError(f"GMRES 수렴 실패 (info={info})")
        return solution

    return solve_fn


def _factorize_gpu(K_bc):
    """GPU LU 분해 + solve (PyTorch 우선, CuPy fallback)"""
    # PyTorch GPU LU (K를 미리 분해)
    torch_solver = _factorize_gpu_torch(K_bc)
    if torch_solver is not None:
        return torch_solver

    # CuPy fallback: 매번 풀이
    def solve_fn(f):
        return _solve_gpu(K_bc, f)
    return solve_fn


def _factorize_banded(K_bc):
    """
    Banded LU 분해 (LAPACK zgbtrf/zgbtrs) — Fortran zGBTRF 동일

    직교 격자 FEM 행렬은 대역폭이 제한적이므로 banded solver가 최적.
    splu 대비 ~10배 빠름.
    """
    import scipy.linalg.lapack as lapack

    n = K_bc.shape[0]
    K_csr = K_bc.tocsr()

    # 대역폭 계산
    rows, cols = K_bc.nonzero()
    max_bw = int(np.max(np.abs(rows - cols)))
    kl = ku = max_bw

    # CSR → banded format 변환 (벡터화)
    ab = np.zeros((2 * kl + ku + 1, n), dtype=complex)
    for i in range(n):
        start, end = K_csr.indptr[i], K_csr.indptr[i + 1]
        js = K_csr.indices[start:end]
        vs = K_csr.data[start:end]
        ab[kl + ku + i - js, js] = vs

    # LU 분해
    lub, piv, info = lapack.zgbtrf(ab, kl, ku)
    if info != 0:
        raise RuntimeError(f"zgbtrf 실패 (info={info})")

    def solve_fn(f):
        sol, info2 = lapack.zgbtrs(lub, kl, ku, f, piv)
        if info2 != 0:
            raise RuntimeError(f"zgbtrs 실패 (info={info2})")
        return sol

    return solve_fn


def _factorize_pardiso(K_bc):
    """PARDISO: complex→real 확장 후 분해"""
    try:
        from pypardiso import PyPardisoSolver
        K_csc = K_bc.tocsc()
        Kr = K_csc.real.copy()
        Ki = K_csc.imag.copy()
        K_real = sp.bmat([[Kr, -Ki], [Ki, Kr]], format='csr')
        n = K_csc.shape[0]
        solver = PyPardisoSolver()
        solver.factorize(K_real)

        def solve_fn(f):
            f_real = np.concatenate([f.real, f.imag])
            sol_real = solver.solve(K_real, f_real)
            return sol_real[:n] + 1j * sol_real[n:]
        return solve_fn
    except (ImportError, AttributeError):
        return factorize_system(K_bc, solver="direct")


def _solve_ilu_gmres(K, f, tol=1e-10, max_iter=500, ky=None):
    """ILU 전처리 + GMRES (복소 행렬 직접 지원)

    ky≈0 (DC 근사)에서는 행렬 조건수가 극단적이므로
    ILU 파라미터를 강화하여 수렴 보장.
    """
    K_csc = K.tocsc()

    # ky≈0일 때 더 정밀한 ILU (fill-in↑, drop_tol↓)
    if ky is not None and abs(ky) < 1e-4:
        drop_tol, fill_factor = 1e-8, 100
    else:
        drop_tol, fill_factor = 1e-4, 20

    ilu = spla.spilu(K_csc, drop_tol=drop_tol, fill_factor=fill_factor)
    M = spla.LinearOperator(K_csc.shape, ilu.solve)
    gmres_tol = max(tol, 1e-8) if (ky is not None and abs(ky) < 1e-4) else tol
    solution, info = spla.gmres(
        K_csc, f, M=M, rtol=gmres_tol, maxiter=max_iter, restart=50)
    if info != 0:
        # fallback: 더 강한 ILU로 재시도
        ilu2 = spla.spilu(K_csc, drop_tol=1e-10, fill_factor=200)
        M2 = spla.LinearOperator(K_csc.shape, ilu2.solve)
        solution, info = spla.gmres(
            K_csc, f, M=M2, rtol=gmres_tol, maxiter=max_iter*4, restart=100)
        if info != 0:
            raise RuntimeError(f"GMRES 수렴 실패 (info={info})")
    return solution


def _solve_pardiso_real_split(K, f):
    """
    Complex → Real 2×2 블록 확장 후 PARDISO 풀이

    K*(xr+i*xi) = (fr+i*fi)
    → [Kr -Ki] [xr]   [fr]
      [Ki  Kr] [xi] = [fi]
    """
    try:
        from pypardiso import spsolve as pardiso_solve
        K_csc = K.tocsc()
        Kr = K_csc.real.copy()
        Ki = K_csc.imag.copy()
        K_real = sp.bmat([[Kr, -Ki], [Ki, Kr]], format='csr')
        f_real = np.concatenate([f.real, f.imag])
        sol_real = pardiso_solve(K_real, f_real)
        n = K_csc.shape[0]
        return sol_real[:n] + 1j * sol_real[n:]
    except ImportError:
        return _solve_ilu_gmres(K, f)


def _solve_gpu(K, f: np.ndarray) -> np.ndarray:
    """GPU 풀이 (CuPy 우선, PyTorch fallback)"""
    # CuPy 시도
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
        import cupyx.scipy.sparse.linalg as cpspla

        K_csr = K.tocsr()
        K_gpu = cpsp.csr_matrix(
            (cp.asarray(K_csr.data),
             cp.asarray(K_csr.indices),
             cp.asarray(K_csr.indptr)),
            shape=K_csr.shape,
        )
        f_gpu = cp.asarray(f)

        try:
            sol_gpu = cpspla.spsolve(K_gpu, f_gpu)
        except Exception:
            sol_gpu, info = cpspla.gmres(K_gpu, f_gpu, tol=1e-10, maxiter=2000)
            if info != 0:
                return _solve_ilu_gmres(K, f)

        return cp.asnumpy(sol_gpu)
    except ImportError:
        pass

    # PyTorch CUDA fallback
    try:
        import torch
        if torch.cuda.is_available():
            K_dense = K.toarray()
            Kd = torch.tensor(K_dense, dtype=torch.complex128, device='cuda')
            fd = torch.tensor(f, dtype=torch.complex128, device='cuda')
            sol = torch.linalg.solve(Kd, fd)
            return sol.cpu().numpy()
    except ImportError:
        pass

    import warnings
    warnings.warn("GPU 미사용 — ILU+GMRES 전환", stacklevel=2)
    return _solve_ilu_gmres(K, f)


def _factorize_gpu_torch(K_bc):
    """PyTorch GPU: K를 GPU dense 행렬로 미리 전송하여 재사용"""
    try:
        import torch
        if torch.cuda.is_available():
            K_dense = K_bc.toarray()
            Kd = torch.tensor(K_dense, dtype=torch.complex128, device='cuda')
            # LU 분해 미리 계산
            LU, pivots = torch.linalg.lu_factor(Kd)

            def solve_fn(f):
                fd = torch.tensor(f, dtype=torch.complex128, device='cuda')
                sol = torch.linalg.lu_solve(LU, pivots, fd.unsqueeze(-1))
                return sol.squeeze(-1).cpu().numpy()

            return solve_fn
    except (ImportError, RuntimeError):
        pass
    return None


def extract_secondary_fields(
    solution: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    해벡터에서 Ey_s, Hy_s 분리

    DOF 매핑 (0-based):
      solution[2i]   = Ey(node i)  ← Fortran 2i-1 (홀수)
      solution[2i+1] = Hy(node i)  ← Fortran 2i   (짝수)
    """
    Ey_s = solution[0::2]
    Hy_s = solution[1::2]
    return Ey_s, Hy_s


def compute_total_fields(
    Ey_secondary: np.ndarray,
    Hy_secondary: np.ndarray,
    Ey_primary: np.ndarray,
    Hy_primary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """전체장 = 2차장 + 1차장"""
    return Ey_secondary + Ey_primary, Hy_secondary + Hy_primary
