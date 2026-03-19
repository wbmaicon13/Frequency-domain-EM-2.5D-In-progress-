"""
forward — 2.5D EM 순방향 모델링 패키지
"""

from .primary_field import (
    PrimaryFieldParams,
    compute_background_wavenumber,
    compute_wavenumber_sampling,
    primary_field_ky_domain,
    primary_field_space_domain,
    modified_bessel_K0_K1,
)
from .fem_assembly import (
    assemble_element_matrix,
    assemble_global_system,
    assemble_force_vector,
)
from .fem_solver import (
    solve_fem_system,
    extract_secondary_fields,
    compute_total_fields,
    build_robin_stiffness,
    factorize_system,
)
from .postprocess import (
    compute_secondary_field_components,
    extract_profile_fields,
)
from .forward_loop import (
    ForwardConfig,
    ForwardModeling,
    run_forward,
)

__all__ = [
    "PrimaryFieldParams",
    "compute_background_wavenumber",
    "compute_wavenumber_sampling",
    "primary_field_ky_domain",
    "primary_field_space_domain",
    "modified_bessel_K0_K1",
    "assemble_element_matrix",
    "assemble_global_system",
    "assemble_force_vector",
    "solve_fem_system",
    "extract_secondary_fields",
    "compute_total_fields",
    "build_robin_stiffness",
    "factorize_system",
    "compute_secondary_field_components",
    "extract_profile_fields",
    "ForwardConfig",
    "ForwardModeling",
    "run_forward",
]
