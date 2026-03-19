"""
parallel — MPI/GPU 병렬화 모듈
"""

from .mpi_manager import (
    MPIContext,
    distribute_ky,
    get_mpi_context,
)
from .gpu_solver import (
    GPUSolver,
    get_default_solver,
    solve_sparse_system,
)

__all__ = [
    "MPIContext",
    "distribute_ky",
    "get_mpi_context",
    "GPUSolver",
    "get_default_solver",
    "solve_sparse_system",
]
