"""
MPI 병렬화 — ky 공간주파수 분배

Fortran 대응: Fem25Dfwd.f90 내 MPI_PARAMETERS 모듈 + MPI_REDUCE 호출

병렬 전략:
  - ky 배열을 MPI 프로세스에 균등 분배
  - 각 프로세스가 자신의 ky 부분집합에 대해 FEM 계산
  - MPI_REDUCE 로 결과 합산 (rank=0 수집)
  - rank=0 에서 역 Fourier 변환

단일 프로세스 fallback:
  mpi4py 없을 경우 자동으로 직렬 실행

사용 예시:
    from em25d.parallel.mpi_manager import MPIContext
    from em25d.forward.forward_loop import ForwardModeling, ForwardConfig

    mpi = MPIContext()
    config = ForwardConfig(n_wavenumbers=20, use_gpu=True)
    fwd = ForwardModeling(grid, model, survey, profile, config)
    data = fwd.run(mpi=mpi)  # rank=0만 결과 반환
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# MPI optional import
try:
    from mpi4py import MPI
    _HAS_MPI = True
except ImportError:
    _HAS_MPI = False


class MPIContext:
    """
    MPI 실행 환경 (단일 프로세스 fallback 포함)
    """

    def __init__(self):
        if _HAS_MPI:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.rank = 0
            self.size = 1

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if _HAS_MPI and self.comm is not None:
            self.comm.Barrier()

    def print_root(self, msg: str) -> None:
        if self.is_root:
            print(msg)

    def reduce_sum(self, local: np.ndarray) -> Optional[np.ndarray]:
        """모든 프로세스의 배열을 rank=0 에서 합산"""
        if not _HAS_MPI or self.comm is None:
            return local

        global_result = np.zeros_like(local) if self.is_root else None
        self.comm.Reduce(local, global_result, op=MPI.SUM, root=0)
        return global_result

    def broadcast(self, data: np.ndarray) -> np.ndarray:
        if not _HAS_MPI or self.comm is None:
            return data
        return self.comm.bcast(data, root=0)


def distribute_ky(
    wavenumbers: np.ndarray,
    mpi: MPIContext,
) -> tuple[np.ndarray, np.ndarray]:
    """
    ky 배열을 프로세스별로 stride 분배 (Fortran 동일)

    Fortran 대응: do iky = 1+rank_mpi, n_ky, nsize_mpi

    n_procs >= n_ky: 각 프로세스가 최대 1개 ky 담당 (나머지 유휴)
    n_procs < n_ky: round-robin stride (rank, rank+nproc, rank+2*nproc, ...)

    Returns
    -------
    local_ky     : 이 프로세스가 담당하는 ky 값 배열
    local_indices: 전체 ky 배열에서의 인덱스
    """
    n_ky = len(wavenumbers)
    local_indices = np.arange(mpi.rank, n_ky, mpi.size)
    local_ky = wavenumbers[local_indices]
    return local_ky, local_indices


# ── 전역 MPI 컨텍스트 (싱글턴) ───────────────────────────────────────────────

_mpi_ctx: Optional[MPIContext] = None


def get_mpi_context() -> MPIContext:
    global _mpi_ctx
    if _mpi_ctx is None:
        _mpi_ctx = MPIContext()
    return _mpi_ctx
