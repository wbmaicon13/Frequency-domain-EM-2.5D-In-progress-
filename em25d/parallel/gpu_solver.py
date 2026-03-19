"""
GPU 가속 FEM 선형 시스템 풀이

Fortran 대응: 없음 (Python 신규 기능)

CuPy 기반 GPU sparse solver.
CuPy 미설치 시 SciPy CPU solver 로 자동 fallback.

지원 backend:
  - CuPy  : NVIDIA GPU (CUDA)
  - SciPy : CPU (fallback)

사용 예시:
    from em25d.parallel.gpu_solver import GPUSolver
    solver = GPUSolver()
    print("GPU 사용 가능:", solver.available)
    x = solver.solve(K, f)          # GPU or CPU 자동 선택
    x = solver.solve(K, f, force_cpu=True)   # CPU 강제
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional


class GPUSolver:
    """
    희소 선형 시스템 K x = f 풀이 (GPU 우선, CPU fallback)

    Parameters
    ----------
    tol     : iterative solver 수렴 허용 오차
    maxiter : iterative solver 최대 반복 수
    verbose : 풀이 정보 출력 여부
    """

    def __init__(
        self,
        tol: float = 1e-10,
        maxiter: int = 2000,
        verbose: bool = False,
    ):
        self.tol     = tol
        self.maxiter = maxiter
        self.verbose = verbose

        # CuPy 가용성 검사
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cpsp
            import cupyx.scipy.sparse.linalg as cpspla
            self._cp      = cp
            self._cpsp    = cpsp
            self._cpspla  = cpspla
            self.available = True
            self._device_name = cp.cuda.Device().attributes.get(
                "DeviceName", b"Unknown GPU").decode(errors="replace")
        except ImportError:
            self._cp = self._cpsp = self._cpspla = None
            self.available = False
            self._device_name = "N/A"

    def solve(
        self,
        K: sp.spmatrix,
        f: np.ndarray,
        force_cpu: bool = False,
    ) -> np.ndarray:
        """
        희소 선형 시스템 K x = f 풀이

        Parameters
        ----------
        K         : (n, n) scipy sparse 강성행렬 (complex)
        f         : (n,) 우변 벡터
        force_cpu : True 면 GPU 미사용 (디버깅/검증용)

        Returns
        -------
        x : (n,) 해 벡터
        """
        if self.available and not force_cpu:
            return self._solve_gpu(K, f)
        return self._solve_cpu(K, f)

    def solve_batched(
        self,
        K: sp.spmatrix,
        F: np.ndarray,
        force_cpu: bool = False,
    ) -> np.ndarray:
        """
        여러 우변 벡터에 대해 K x_i = f_i 일괄 풀이

        Parameters
        ----------
        K : (n, n) 강성행렬 (모든 우변에서 공유)
        F : (n, m) 우변 행렬 (m 개의 우변)

        Returns
        -------
        X : (n, m) 해 행렬
        """
        n, m = F.shape
        X = np.zeros_like(F)
        for j in range(m):
            X[:, j] = self.solve(K, F[:, j], force_cpu=force_cpu)
        return X

    @property
    def backend(self) -> str:
        """현재 사용 중인 backend 이름"""
        return f"CuPy ({self._device_name})" if self.available else "SciPy (CPU)"

    def __repr__(self) -> str:
        return f"GPUSolver(backend={self.backend!r}, tol={self.tol})"

    # ── 내부 구현 ─────────────────────────────────────────────────────────────

    def _solve_gpu(self, K: sp.spmatrix, f: np.ndarray) -> np.ndarray:
        """CuPy GMRES → 실패 시 직접법(spsolve) fallback"""
        cp     = self._cp
        cpsp   = self._cpsp
        cpspla = self._cpspla

        # CPU sparse → GPU sparse (CSR)
        K_csr = K.tocsr()
        K_gpu = cpsp.csr_matrix(
            (cp.asarray(K_csr.data),
             cp.asarray(K_csr.indices),
             cp.asarray(K_csr.indptr)),
            shape=K_csr.shape,
        )
        f_gpu = cp.asarray(f)

        # 1차 시도: GMRES (반복법)
        x_gpu, info = cpspla.gmres(
            K_gpu, f_gpu,
            tol=self.tol,
            maxiter=self.maxiter,
        )

        if info != 0:
            if self.verbose:
                print(f"[GPUSolver] GMRES 미수렴 (info={info}), spsolve fallback")
            x_gpu = cpspla.spsolve(K_gpu, f_gpu)

        return cp.asnumpy(x_gpu)

    def _solve_cpu(self, K: sp.spmatrix, f: np.ndarray) -> np.ndarray:
        """
        SciPy 직접법 (LU 분해)

        복소 시스템에 대해 splu 이 가장 안정적.
        """
        K_csc = K.tocsc()
        try:
            lu = spla.splu(K_csc)
            x  = lu.solve(f)
        except Exception:
            # 마지막 수단: lsqr
            x, *_ = spla.lsqr(K_csc, f, atol=self.tol, btol=self.tol,
                               iter_lim=self.maxiter)
        return x


# ── 전역 기본 solver 인스턴스 (싱글턴) ──────────────────────────────────────

_default_solver: Optional[GPUSolver] = None


def get_default_solver() -> GPUSolver:
    """모듈 전역 GPUSolver 인스턴스 반환 (싱글턴)"""
    global _default_solver
    if _default_solver is None:
        _default_solver = GPUSolver()
    return _default_solver


def solve_sparse_system(
    K: sp.spmatrix,
    f: np.ndarray,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    편의 함수: GPU 또는 CPU 로 K x = f 풀기

    Parameters
    ----------
    K       : (n, n) sparse 강성행렬
    f       : (n,) 우변 벡터
    use_gpu : False 면 CPU 직접법 사용

    Returns
    -------
    x : (n,) 해 벡터
    """
    solver = get_default_solver()
    return solver.solve(K, f, force_cpu=not use_gpu)
