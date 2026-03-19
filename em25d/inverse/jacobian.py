"""
야코비안(감도 행렬) 계산 — 상반정리(Reciprocity) 기반

Fortran 대응: Fem25DjacReci.f90
  - JacobianReciprocity
  - FemJacobianReciprocity
  - Calculate_ExyzTx / Calculate_Exyz
  - deltah (요소 면적분)

물리:
  데이터 d_i 에 대한 블록 로그-비저항 log(ρ_b) 의 감도:
    J[i, b] = ∂d_i / ∂log(ρ_b)
             = -σ_b ∫∫_Ω_b  E_fwd(r) · E_adj(r)  dA

  여기서:
    E_fwd  = 송신기 소스에서 계산한 순방향 장
    E_adj  = 가상 소스(수신기 위치)에서 계산한 역산 장 (상반정리)

ky 영역 처리:
  1. 각 ky 에 대해 J_ky[ky, idata, iblock] 를 요소적분으로 계산
  2. ky → 공간 역 Fourier 변환으로 J[idata, iblock] 획득
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..constants import MU_0, EPSILON_0, PI, SourceType
from ..mesh.grid import Grid
from ..model.resistivity import ResistivityModel


# ── 요소 면적분 (deltah) ──────────────────────────────────────────────────────

def element_surface_integral(
    green: np.ndarray,   # (4,) 복소수  — 순방향 장 (Green 함수)
    e_val: np.ndarray,   # (4,) 복소수  — 역산 장
    x1: float, x2: float,  # 요소 x 좌표 (좌, 우)
    z_nodes: np.ndarray,    # (4,) z 좌표 [z1, z2, z3, z4]
) -> complex:
    """
    쌍일차(bilinear) 2D 수치 적분

    Fortran 대응: complex*16 function deltah(green, e_value, x1,x2,z1,z2,z3,z4)

    수학:
      factor = (x2-x1)*(z3-z2) / 36
      result = factor * Σ_i e_i * (4δ_{ii} + 2δ_{ij_adj} + 1δ_{ij_opp}) * g_j

    노드 순서 (Fortran 전통):
        1 ── 2
        |    |
        4 ── 3

    Parameters
    ----------
    green  : (4,) — Green 함수 (순방향 장)
    e_val  : (4,) — 역산 장
    x1, x2 : 요소 x 범위 [m]
    z_nodes : (4,) z 좌표 [z1, z2, z3, z4]
    """
    z1, z2, z3, z4 = z_nodes
    factor = (x2 - x1) * (z3 - z2) / 36.0

    # 4×4 가중치 행렬 (bilinear)
    W = np.array([
        [4., 2., 1., 2.],
        [2., 4., 2., 1.],
        [1., 2., 4., 2.],
        [2., 1., 2., 4.],
    ])
    return complex(factor * (e_val @ W @ green))


# ── 전기장 성분 계산 ─────────────────────────────────────────────────────────

def compute_field_components_at_nodes(
    Ey_nodes: np.ndarray,   # (n_nodes,) 복소수 — 총 Ey (2차 + 1차)
    Hy_nodes: np.ndarray,   # (n_nodes,) 복소수 — 총 Hy
    E_primary: np.ndarray,  # (3, n_nodes) 복소수 — 1차장 [Ex, Ey, Ez]
    grid: Grid,
    omega: float,
    ky: float,
    element_sigma: np.ndarray,   # (n_elements,) 도전율 [S/m]
    layer_sigma: np.ndarray,     # (n_elements,) 배경 도전율 [S/m]
) -> dict[str, np.ndarray]:
    """
    Ey, Hy 로부터 Ex, Ez 성분을 커얼 방정식으로 계산

    Fortran 대응: Calculate_ExyzTx / Calculate_Exyz

    ky 영역 Maxwell 방정식:
      ke² = ky² - k²       k² = ω²με - iωμσ
      Ex  = (-i·ky·∂Ey/∂x - i·ω·μ·∂Hy/∂z + ka²·Ex_primary) / ke² + Ex_primary
      Ez  = (-i·ky·∂Hy/∂x + i·ω·μ·∂Ey/∂z + ka²·Ez_primary) / ke² + Ez_primary

    where ka² = -i·ω·μ·Δσ (conductivity contrast correction)

    Returns
    -------
    dict with keys "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"  shape (n_nodes,)
    """
    n_nz, n_nx = grid.n_nodes_z, grid.n_nodes_x
    dx = np.diff(grid.cell_x)   # (n_nx-1,) 간격
    dz = np.diff(grid.cell_z)   # (n_nz-1,) 간격

    # 2D 배열로 재배치  [iz, ix]
    def to_2d(v):
        return v.reshape(n_nz, n_nx)

    Ey2d = to_2d(Ey_nodes)
    Hy2d = to_2d(Hy_nodes)

    # 수치 미분 (중심 차분, 경계는 전진/후진)
    dEy_dx = np.gradient(Ey2d, axis=1) / np.gradient(
        grid.node_x[:, 0] if grid.node_x.ndim == 2 else np.arange(n_nx), axis=0
    )  # 단순화: 균등 격자 근사
    dEy_dz = np.gradient(Ey2d, axis=0)
    dHy_dx = np.gradient(Hy2d, axis=1)
    dHy_dz = np.gradient(Hy2d, axis=0)

    # 노드 좌표 배열 (실제 격자 간격 적용)
    x_1d = grid.node_x[:, 0] if grid.node_x.ndim == 2 else np.linspace(0, 1, n_nx)
    z_1d = grid.node_z[0, :] if grid.node_z.ndim == 2 else np.linspace(0, 1, n_nz)
    dEy_dx = np.gradient(Ey2d, x_1d, axis=1)
    dEy_dz = np.gradient(Ey2d, z_1d, axis=0)
    dHy_dx = np.gradient(Hy2d, x_1d, axis=1)
    dHy_dz = np.gradient(Hy2d, z_1d, axis=0)

    # 노드별 물성 (요소 평균으로부터 추정)
    # 단순화: 배경값으로 k², ke² 계산
    # 보다 정확한 구현은 요소별 sigma를 노드로 보간해야 함
    sigma_bg = np.mean(layer_sigma)
    k2 = omega**2 * MU_0 * EPSILON_0 - 1j * MU_0 * sigma_bg * omega
    sigma_mean = np.mean(element_sigma)
    ka2 = -1j * MU_0 * (sigma_mean - sigma_bg) * omega
    ke2 = ky**2 - k2

    Ex_p = E_primary[0]  # (n_nodes,)
    Ez_p = E_primary[2]

    Ex_2d = (
        ((-1j * ky) * dEy_dx + (-1j * omega * MU_0) * dHy_dz
         + ka2 * to_2d(Ex_p)) / ke2
        + to_2d(Ex_p)
    )
    Ez_2d = (
        ((-1j * ky) * dHy_dx + (1j * omega * MU_0) * dEy_dz
         + ka2 * to_2d(Ez_p)) / ke2
        + to_2d(Ez_p)
    )

    Ex_nodes = Ex_2d.ravel()
    Ez_nodes = Ez_2d.ravel()
    Ey_total = Ey_nodes + E_primary[1]

    # Hx, Hz (커얼 E = -iωμH 로부터)
    Hx_nodes = ((-1j * ky) * to_2d(Ey_total) - dEy_dz).ravel() / (-1j * omega * MU_0)
    Hz_nodes = (dEy_dx - (-1j * ky) * to_2d(Ey_total)).ravel() / (-1j * omega * MU_0)

    return {
        "Ex": Ex_nodes,
        "Ey": Ey_total,
        "Ez": Ez_nodes,
        "Hx": Hx_nodes,
        "Hy": Hy_nodes + E_primary[1] * 0,  # Hy는 직접 FEM 해
        "Hz": Hz_nodes,
    }


# ── ky 영역 야코비안 ─────────────────────────────────────────────────────────

def compute_jacobian_ky(
    grid: Grid,
    model: ResistivityModel,
    forward_fields: dict,    # {(itx, iky): {"Ex":..., "Ey":..., "Ez":...}}
    adjoint_fields: dict,    # {(irx, iky): {"Ex":..., "Ey":..., "Ez":...}}
    wavenumbers: np.ndarray,  # (n_ky,) ky 값
    data_map: np.ndarray,    # (n_data, 3) — [itx, irx, icomp]
    omega: float,
    iky_start: int = 0,
) -> np.ndarray:
    """
    ky 영역 야코비안 J_ky[n_ky, n_data, n_blocks]

    Fortran 대응: JacobianReciprocity 내부 루프

    Parameters
    ----------
    forward_fields : 송신기별, ky별 전기장 성분 딕셔너리
    adjoint_fields : 수신기별, ky별 전기장 성분 딕셔너리 (가상 소스)
    data_map       : (n_data, 3) — 각 데이터 포인트의 (itx, irx, icomp)
    """
    n_ky    = len(wavenumbers)
    n_data  = len(data_map)
    n_blocks = model.n_blocks

    J_ky = np.zeros((n_ky, n_data, n_blocks), dtype=complex)

    elem_block = model.element_block_index  # (n_elem,) 블록 인덱스 (-1: 비역산)

    for iky, ky in enumerate(wavenumbers):
        for i_data, (itx, irx, icomp) in enumerate(data_map):
            key_fwd = (int(itx), iky)
            key_adj = (int(irx), iky)
            if key_fwd not in forward_fields or key_adj not in adjoint_fields:
                continue

            F = forward_fields[key_fwd]   # 순방향 Ex, Ey, Ez
            A = adjoint_fields[key_adj]   # 역산 Ex, Ey, Ez

            comp_key = ["Ex", "Ey", "Ez"][icomp % 3]

            # 자기장 성분인 경우 스케일 인자
            if icomp >= 3:
                scale = -1.0 / (1j * omega * MU_0)
            else:
                scale = 1.0

            # 요소별 적분
            for ie, iblck in enumerate(elem_block):
                if iblck < 0:
                    continue

                # 요소 노드 인덱스 (0-based)
                nodes_ie = grid.element_nodes[ie]  # (4,) [n0,n1,n2,n3]

                # 요소 좌표
                x_e = np.array([grid.all_node_x[n] for n in nodes_ie])
                z_e = np.array([grid.all_node_z[n] for n in nodes_ie])

                # 순방향 / 역산 장 4노드 값
                if iky == 0:  # ky=0: Ey만 사용 (TE 모드)
                    # y 성분 기여만
                    g_vals = np.array([F["Ey"][n] for n in nodes_ie])
                    e_vals = np.array([A["Ey"][n] for n in nodes_ie])
                    total  = element_surface_integral(
                        g_vals, e_vals, x_e[0], x_e[1], z_e)
                else:
                    # 세 성분 합산 (TM 기여)
                    total = 0.0
                    signs = {"Ex": -1., "Ey": 1., "Ez": -1.}
                    for comp in ["Ex", "Ey", "Ez"]:
                        g_vals = np.array([F[comp][n] for n in nodes_ie])
                        e_vals = np.array([A[comp][n] for n in nodes_ie])
                        sgn = signs[comp]
                        total += sgn * element_surface_integral(
                            g_vals, e_vals, x_e[0], x_e[1], z_e)

                J_ky[iky, i_data, iblck] += total * scale

    return J_ky


# ── 역 Fourier 변환 ───────────────────────────────────────────────────────────

def jacobian_inverse_fourier(
    J_ky: np.ndarray,        # (n_ky, n_data, n_blocks) complex
    wavenumbers: np.ndarray,  # (n_ky,)
    symmetry: Optional[np.ndarray] = None,  # (n_data,) 대칭성 정보 (0=even,1=odd)
) -> np.ndarray:
    """
    J_ky → J (공간 영역) 역 Fourier 변환

    Fortran 대응: FemJacobianReciprocity 내 fourint 호출

    even 대칭 (cos 변환): J = (2/π) ∫ J_ky(ky) dky  → 허수부 사용
    odd  대칭 (sin 변환): J = (2/π) ∫ J_ky(ky) dky  → 실수부 사용

    대부분의 EM 데이터는 even 대칭.

    Parameters
    ----------
    symmetry : 0=허수부(even), 1=실수부(odd)

    Returns
    -------
    J : (n_data, n_blocks) 실수 야코비안
    """
    n_ky, n_data, n_blocks = J_ky.shape
    J = np.zeros((n_data, n_blocks), dtype=float)

    if symmetry is None:
        symmetry = np.zeros(n_data, dtype=int)

    # 사다리꼴 적분 (trapezoid)
    for i_data in range(n_data):
        for i_blck in range(n_blocks):
            buffer = J_ky[:, i_data, i_blck]
            rlt = np.trapz(buffer, wavenumbers)
            # y=0 단면 역 Fourier: factor = 2/π
            rlt *= 2.0 / PI
            if symmetry[i_data] == 0:
                J[i_data, i_blck] = rlt.imag
            else:
                J[i_data, i_blck] = rlt.real

    return J


# ── 비저항 감도 보정 ─────────────────────────────────────────────────────────

def apply_resistivity_transform(
    J: np.ndarray,                   # (n_data, n_blocks)
    block_rho: np.ndarray,           # (n_blocks,)
    norm_factor: np.ndarray,         # (n_data,)
    rho_min: float = 0.1,
    rho_max: float = 1e5,
    log_transform: bool = True,
) -> np.ndarray:
    """
    비저항 파라미터화에 맞춘 야코비안 감도 보정

    Fortran 대응: FemJacobianReciprocity 내 fac 계산

    log 변환 (IK_LOG):
      fac_b = σ_b · log(σ_max/σ_b) · log(σ_b/σ_min) / log(σ_max/σ_min)

    Parameters
    ----------
    norm_factor : 데이터 정규화 인자 (분모)
    """
    sigma = 1.0 / block_rho
    sigma_min = 1.0 / rho_max
    sigma_max = 1.0 / rho_min

    if log_transform:
        log_ba = np.log(sigma_max / sigma_min)
        log_bm = np.log(sigma_max / sigma)
        log_ma = np.log(sigma / sigma_min)
        fac = sigma * log_bm * log_ma / log_ba  # (n_blocks,)
    else:
        fac = sigma * (sigma - sigma_min) * (sigma_max - sigma) / (sigma_max - sigma_min)

    # J[i, b] /= norm_factor[i]  ·  fac[b]
    J_scaled = J / norm_factor[:, np.newaxis] * fac[np.newaxis, :]
    return J_scaled


# ── 통합 인터페이스 ───────────────────────────────────────────────────────────

@dataclass
class JacobianResult:
    """야코비안 계산 결과"""
    J: np.ndarray            # (n_data, n_blocks) 실수
    J_ky: np.ndarray         # (n_ky, n_data, n_blocks) 복소수 — 선택적 저장
    norm_factor: np.ndarray  # (n_data,) 정규화 인자


def compute_jacobian(
    J_ky: np.ndarray,
    wavenumbers: np.ndarray,
    block_rho: np.ndarray,
    norm_factor: np.ndarray,
    symmetry: Optional[np.ndarray] = None,
    rho_min: float = 0.1,
    rho_max: float = 1e5,
    log_transform: bool = True,
) -> JacobianResult:
    """
    ky 영역 야코비안 → 공간 야코비안 (역 Fourier + 감도 보정)

    Fortran 대응: FemJacobianReciprocity

    Parameters
    ----------
    J_ky       : (n_ky, n_data, n_blocks) ky-domain 야코비안
    wavenumbers: (n_ky,) ky 값
    block_rho  : (n_blocks,) 비저항
    norm_factor: (n_data,) 데이터 정규화 인자
    symmetry   : (n_data,) 0=허수부, 1=실수부
    """
    J = jacobian_inverse_fourier(J_ky, wavenumbers, symmetry)
    J = apply_resistivity_transform(
        J, block_rho, norm_factor, rho_min, rho_max, log_transform)
    return JacobianResult(J=J, J_ky=J_ky, norm_factor=norm_factor)
