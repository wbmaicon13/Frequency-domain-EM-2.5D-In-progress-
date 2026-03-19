"""격자(Mesh) 생성 모듈 — model_setup 대체."""

from .grid import Grid, GridConfig
from .boundary import get_boundary_node_indices, apply_dirichlet_boundary
from .topography import TopographyData, apply_topography
from .block import BlockPartition, BlockConfig
from .profile import ProfileNodes
