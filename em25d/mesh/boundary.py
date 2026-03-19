"""
경계 노드/요소 처리 — Robin(임피던스) 경계 조건

Fortran 대응:
  Make_BC (Fem25Dfwd.f90, 라인 682-795): 경계 엣지별 임피던스 계수 p 계산
  apply_bc (Fem25Dsub.f, 라인 534-577): 전역 강성행렬에 BC 적분 추가
  bc_int (Fem25Dsub.f, 라인 1075-1123): 1D Gauss 구적법 BC 적분

물리적 배경:
  2.5D EM 문제의 외부 경계에서 파동 임피던스 조건:
    ∂E/∂n + p·E = 0,  ∂H/∂n + p·H = 0
  여기서 p = -sqrt(ky² - k²), k² = ω²με - iωμσ

  이 조건은 FEM 약형식에서 경계 적분으로 자연스럽게 처리:
    ∫_Γ p·N_i·N_j dΓ  →  강성행렬에 추가

  Dirichlet(E=0) 대신 Robin BC를 사용하면:
  - 파동이 경계에서 반사 없이 흡수됨
  - 유한 격자에서도 무한 매질 조건을 근사
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from .grid import Grid
from ..constants import MU_0, EPSILON_0


# ── 1D Gauss 구적법 (order 3) ────────────────────────────────────────────────
# Fortran gaussian_1d / shape_1d 서브루틴 대응 (nl=3)
_GAUSS_1D_PTS_3 = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
_GAUSS_1D_WTS_3 = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])


def _shape_1d_linear(xi: float):
    """
    1D 선형 형상함수 (2절점 경계 엣지)

    Fortran shape_1d (n=2, 라인 1037-1047)
    ψ₁ = (1-ξ)/2,  ψ₂ = (1+ξ)/2  (ξ ∈ [-1, 1])
    """
    return np.array([(1.0 - xi) / 2.0, (1.0 + xi) / 2.0])


def get_boundary_node_indices(grid: Grid) -> np.ndarray:
    """
    전체 격자의 외부 경계 노드 전역 인덱스 반환

    외부 경계 = 격자의 4개 변 위에 있는 모든 노드
    """
    nx = grid.n_nodes_x
    nz = grid.n_nodes_z

    top_row    = np.arange(nx)                          # iz = 0
    bottom_row = np.arange((nz - 1) * nx, nz * nx)     # iz = nz-1
    left_col   = np.arange(0, nz * nx, nx)              # ix = 0
    right_col  = np.arange(nx - 1, nz * nx, nx)         # ix = nx-1

    all_boundary = np.concatenate([top_row, bottom_row, left_col, right_col])
    return np.unique(all_boundary)


def get_boundary_edges(grid: Grid):
    """
    경계 엣지(2절점 쌍) 및 인접 요소 비저항 추출

    Fortran 대응: Make_BC의 b_node.dat/b_elem.dat 읽기 부분

    반환:
      edges : list of (node1_global, node2_global, side)
              side: 'left'(0), 'right'(1), 'top'(2), 'bottom'(3)
    """
    nx = grid.n_nodes_x
    nz = grid.n_nodes_z
    edges = []

    # 좌측 경계 (ix=0, iz 순서 증가)
    for iz in range(nz - 1):
        n1 = grid.node_index(0, iz)
        n2 = grid.node_index(0, iz + 1)
        edges.append((n1, n2, 'left'))

    # 우측 경계 (ix=nx-1)
    for iz in range(nz - 1):
        n1 = grid.node_index(nx - 1, iz)
        n2 = grid.node_index(nx - 1, iz + 1)
        edges.append((n1, n2, 'right'))

    # 상단 경계 (iz=0, ix 순서 증가)
    for ix in range(nx - 1):
        n1 = grid.node_index(ix, 0)
        n2 = grid.node_index(ix + 1, 0)
        edges.append((n1, n2, 'top'))

    # 하단 경계 (iz=nz-1)
    for ix in range(nx - 1):
        n1 = grid.node_index(ix, nz - 1)
        n2 = grid.node_index(ix + 1, nz - 1)
        edges.append((n1, n2, 'bottom'))

    return edges


def compute_robin_impedance(
    sigma: float,
    omega: float,
    ky: float,
    mu: float = MU_0,
    epsilon: float = EPSILON_0,
) -> complex:
    """
    Robin BC 임피던스 계수 p 계산

    Fortran 대응: Make_BC (라인 739-740)
      ke2 = ky² - (ω²με - iωμσ)
      p = -sqrt(ke2)

    Parameters
    ----------
    sigma   : 경계 인접 요소의 전기전도도 [S/m]
    omega   : 각주파수 [rad/s]
    ky      : 공간주파수 [1/m]
    """
    k2 = complex(omega**2 * mu * epsilon, -mu * sigma * omega)
    ke2 = ky**2 - k2
    return -np.sqrt(ke2)


def robin_boundary_integral(
    x1: float,
    x2: float,
    p: complex,
) -> tuple[np.ndarray, np.ndarray]:
    """
    경계 엣지에서 Robin BC 적분 계산

    Fortran 대응: bc_int (Fem25Dsub.f, 라인 1075-1123)

    2절점 선형 요소에 대해:
      Ke_bc[ii,jj] = ∫ p·ψ_i·ψ_j dΓ  (1D Gauss 구적, 3점)
      fe_bc[ii]    = ∫ s·ψ_i dΓ       (s=0 for homogeneous Robin)

    Ey(odd DOF)와 Hy(even DOF) 독립적으로 적용.
    Fortran: ii = 2*i-1 (E), ii = 2*i (H)
    Python:  iE = 2*i   (E), iH = 2*i+1 (H)

    Parameters
    ----------
    x1, x2 : 경계 엣지 양단 좌표 (x 또는 z)
    p      : Robin 임피던스 계수

    Returns
    -------
    Ke_bc : (4, 4) 경계 요소 행렬 (2노드 × 2DOF = 4 DOF)
    fe_bc : (4,) 경계 힘벡터
    """
    dx = (x2 - x1) / 2.0
    n_dof = 4   # 2 nodes × 2 DOF (Ey, Hy)
    Ke_bc = np.zeros((n_dof, n_dof), dtype=complex)
    fe_bc = np.zeros(n_dof, dtype=complex)

    # E 성분 (DOF 인덱스 0, 2 → node 0, 1의 Ey)
    for l in range(3):
        xi = _GAUSS_1D_PTS_3[l]
        w = _GAUSS_1D_WTS_3[l]
        psi = _shape_1d_linear(xi)

        for i in range(2):
            iE = 2 * i   # Ey DOF
            for j in range(2):
                jE = 2 * j
                Ke_bc[iE, jE] += p * psi[i] * psi[j] * w * dx

    # H 성분 (DOF 인덱스 1, 3 → node 0, 1의 Hy)
    for l in range(3):
        xi = _GAUSS_1D_PTS_3[l]
        w = _GAUSS_1D_WTS_3[l]
        psi = _shape_1d_linear(xi)

        for i in range(2):
            iH = 2 * i + 1   # Hy DOF
            for j in range(2):
                jH = 2 * j + 1
                Ke_bc[iH, jH] += p * psi[i] * psi[j] * w * dx

    return Ke_bc, fe_bc


def apply_robin_boundary(
    K_global,
    f_global: np.ndarray,
    grid: Grid,
    element_resistivity: np.ndarray,
    omega: float,
    ky: float,
    mu: float = MU_0,
    epsilon: float = EPSILON_0,
):
    """
    Robin(임피던스) 경계 조건을 전역 강성행렬에 적용

    Fortran 대응:
      Make_BC: 각 경계 엣지의 임피던스 p 계산
      apply_bc: bc_int로 적분 후 assmbl_b3로 전역 행렬에 조립

    Dirichlet(E=0)와 달리, 행/열을 제거하지 않고
    경계 적분을 강성행렬에 직접 더함.
    전체 시스템 크기가 유지되므로 spsolve로 풀면 됨.

    Parameters
    ----------
    K_global : (2*n_nodes, 2*n_nodes) sparse 강성행렬 (수정됨)
    f_global : (2*n_nodes,) 힘벡터 (수정됨)
    grid : Grid 객체
    element_resistivity : (n_ex, n_ez) 요소별 비저항
    omega : 각주파수
    ky : 공간주파수
    """
    edges = get_boundary_edges(grid)
    nx = grid.n_nodes_x

    x_coords = grid.node_x[:, 0]   # (n_nodes_x,) x좌표
    z_coords = grid.node_z[0, :]   # (n_nodes_z,) z좌표

    rows, cols, vals = [], [], []

    for n1, n2, side in edges:
        # 엣지의 인접 요소 비저항에서 전도도 결정
        # Fortran: mat_prop(iBelemNo(i, iside))
        if side in ('left', 'right'):
            # 좌/우 경계: z 방향 엣지
            iz1 = n1 // nx
            if side == 'left':
                ex = 0
            else:
                ex = grid.n_elements_x - 1
            ez = min(iz1, grid.n_elements_z - 1)
            coord1 = z_coords[iz1]
            coord2 = z_coords[iz1 + 1]
        else:
            # 상/하 경계: x 방향 엣지
            ix1 = n1 % nx
            if side == 'top':
                ez = 0
            else:
                ez = grid.n_elements_z - 1
            ex = min(ix1, grid.n_elements_x - 1)
            coord1 = x_coords[ix1]
            coord2 = x_coords[ix1 + 1]

        rho = element_resistivity[ex, ez]
        sigma = 1e-8 if rho == 0 else 1.0 / rho

        # Robin 임피던스 계수
        p = compute_robin_impedance(sigma, omega, ky, mu, epsilon)

        # 경계 엣지 적분
        Ke_bc, fe_bc = robin_boundary_integral(coord1, coord2, p)

        # 전역 DOF 매핑: node → (2*node, 2*node+1)
        node_ids = [n1, n2]
        global_dofs = []
        for nid in node_ids:
            global_dofs.extend([2 * nid, 2 * nid + 1])

        # sparse 행렬에 누적
        for li, gi in enumerate(global_dofs):
            f_global[gi] += fe_bc[li]
            for lj, gj in enumerate(global_dofs):
                if Ke_bc[li, lj] != 0:
                    rows.append(gi)
                    cols.append(gj)
                    vals.append(Ke_bc[li, lj])

    # 기존 sparse 행렬에 Robin BC 행렬 추가
    if vals:
        n_dof = K_global.shape[0]
        K_robin = sp.csr_matrix(
            (vals, (rows, cols)), shape=(n_dof, n_dof), dtype=complex)
        K_global = K_global + K_robin

    return K_global, f_global


# ── 하위 호환용 (이전 Dirichlet 코드 — 사용 중단) ─────────────────────────────

def get_interior_node_indices(grid: Grid) -> np.ndarray:
    """경계를 제외한 내부 노드 전역 인덱스 (Robin BC에서는 사용 안 함)"""
    all_nodes = np.arange(grid.n_nodes)
    boundary = get_boundary_node_indices(grid)
    mask = np.ones(grid.n_nodes, dtype=bool)
    mask[boundary] = False
    return all_nodes[mask]


def apply_dirichlet_boundary(stiffness_matrix, force_vector, boundary_indices):
    """Dirichlet BC (더 이상 사용하지 않음 — Robin BC로 대체)"""
    n_dof = stiffness_matrix.shape[0]
    interior = np.setdiff1d(np.arange(n_dof), boundary_indices)
    K_int = stiffness_matrix[interior, :][:, interior]
    f_int = force_vector[interior]
    return K_int, f_int, interior


def expand_solution(solution_interior, interior_indices, n_total, dtype=complex):
    """Dirichlet 확장 (더 이상 사용하지 않음 — Robin BC로 대체)"""
    solution_full = np.zeros(n_total, dtype=dtype)
    solution_full[interior_indices] = solution_interior
    return solution_full
