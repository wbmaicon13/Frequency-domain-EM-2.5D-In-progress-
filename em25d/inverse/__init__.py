"""
inverse — 역산 모듈

Fortran 대응: Fem25Dacb.f90, Fem25DjacReci.f90,
              Fem25DSequence.f90, Fem25D_Measures.f90, Fem25Dinv.f90
"""

from .measures import (
    ekblom_norm,
    huber_norm,
    support_norm,
    compute_norm,
    ekblom_weights,
    huber_weights,
    support_weights,
    compute_irls_weights,
)
from .jacobian import (
    element_surface_integral,
    compute_field_components_at_nodes,
    jacobian_inverse_fourier,
    apply_resistivity_transform,
    compute_jacobian,
    JacobianResult,
)
from .regularization import (
    build_roughening_matrix,
    build_roughening_matrix_sparse,
    build_identity_regularization,
    compute_model_structure,
    model_roughness_objective,
    scale_roughening_matrix,
)
from .acb import (
    to_inversion_param,
    from_inversion_param,
    compute_acb_lagrangian,
    solve_normal_equations,
    line_search_step_size,
    inversion_step,
    InversionStepResult,
    NormalEquationResult,
)
from .sequence import (
    build_sequence_matrix,
    compute_sequence_contribution,
)
from .inversion_loop import (
    InversionConfig,
    InversionResult,
    InversionModeling,
    run_inversion,
    select_data_components,
    compute_residual,
    compute_rms,
)

__all__ = [
    # measures
    "ekblom_norm", "huber_norm", "support_norm", "compute_norm",
    "ekblom_weights", "huber_weights", "support_weights", "compute_irls_weights",
    # jacobian
    "element_surface_integral", "compute_field_components_at_nodes",
    "jacobian_inverse_fourier", "apply_resistivity_transform",
    "compute_jacobian", "JacobianResult",
    # regularization
    "build_roughening_matrix", "build_roughening_matrix_sparse",
    "build_identity_regularization", "compute_model_structure",
    "model_roughness_objective", "scale_roughening_matrix",
    # acb
    "to_inversion_param", "from_inversion_param",
    "compute_acb_lagrangian", "solve_normal_equations",
    "line_search_step_size", "inversion_step",
    "InversionStepResult", "NormalEquationResult",
    # sequence
    "build_sequence_matrix", "compute_sequence_contribution",
    # inversion_loop
    "InversionConfig", "InversionResult", "InversionModeling", "run_inversion",
    "select_data_components", "compute_residual", "compute_rms",
]
