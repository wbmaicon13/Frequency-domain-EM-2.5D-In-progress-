"""
FEM 강성행렬(Stiffness Matrix) 및 힘벡터(Force Vector) 조립

Fortran 대응: Fem25Dsub.f — cal_elem, get_mat, source_e, source_h

수학적 배경:
  2.5D EM 문제는 y 방향 Fourier 변환 후 (x,z) 평면의 2D FEM으로 풀림.
  결합 TE/TM Maxwell 방정식:

    E 방정식: ∇·[(σ+iωε)/(ky²-k²) ∇Ey_s] - (σ+iωε)·Ey_s = f_E
    H 방정식: ∇·[-iωμ/(ky²-k²) ∇Hy_s]  + iωμ·Hy_s = f_H

  여기서 하첨자 s = 2차장(secondary), k² = -iωμ(σ+iωε)
  2차장 = 전체장 - 1차장 (차이 공식으로 격자 내 특이점 제거)

  DOF 구성 (노드당 2개, Fortran 1-based → Python 0-based 변환):
    짝수 인덱스 (2i):   Ey 성분  ← Fortran: ii = 2*i-1 (홀수)
    홀수 인덱스 (2i+1): Hy 성분  ← Fortran: ii = 2*i   (짝수)
    (Fortran의 1-based 홀수=1,3,5... → 0-based 짝수=0,2,4...)

요소: 4절점 사각형(bilinear quadrilateral), 2×2 Gauss 구적법
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from ..constants import MU_0, EPSILON_0, PI

# ── Gauss 구적법 파라미터 (2×2) ──────────────────────────────────────────────
_GAUSS_PTS = np.array([-1.0/np.sqrt(3), 1.0/np.sqrt(3)])
_GAUSS_WTS = np.array([1.0, 1.0])

# 공기 전기전도도
_SIG_AIR = 1e-8   # [S/m]


def _shape_functions(xi: float, eta: float):
    """
    4절점 사각형 요소의 쌍선형 형상함수 및 편미분

    Fortran 대응: shape2d 서브루틴

    ψ₁ = (1-ξ)(1-η)/4,  ψ₂ = (1+ξ)(1-η)/4
    ψ₃ = (1+ξ)(1+η)/4,  ψ₄ = (1-ξ)(1+η)/4

    반환:
      N    : (4,)  형상함수 값
      dN   : (2,4) 편미분 [∂N/∂ξ ; ∂N/∂η]
    """
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta),
    ])
    dN = 0.25 * np.array([
        [-(1-eta),  (1-eta), (1+eta), -(1+eta)],   # ∂N/∂ξ
        [-(1-xi),  -(1+xi),  (1+xi),   (1-xi)],    # ∂N/∂η
    ])
    return N, dN


def _material_coefficients(
    resistivity: float,
    layer_resistivity: float,
    omega: float,
    ky: float,
    mu: float = MU_0,
    epsilon: float = EPSILON_0,
) -> dict:
    """
    요소 재질 계수 계산

    Fortran 대응: get_mat 서브루틴 (forward mode)

    반환:
      xk_E: E 방정식 확산 계수  (σ+iωε)/(ky²-k²)·i
      xb_E: E 방정식 질량 계수  (σ+iωε)·i
      xk_H: H 방정식 확산 계수  -iωμ/(ky²-k²)
      xb_H: H 방정식 질량 계수  -iωμ
      delta_sig: 배경 대비 전도도 대비 (2차장 소스항)
    """
    sig   = _SIG_AIR if resistivity == 0 else 1.0 / resistivity
    sig_b = _SIG_AIR if layer_resistivity == 0 else 1.0 / layer_resistivity

    k2   = mu * epsilon * omega**2 - 1j * mu * sig * omega
    denom = ky**2 - k2

    xk_E = 1j * (sig + 1j * omega * epsilon) / denom
    xb_E = 1j * (sig + 1j * omega * epsilon)
    xk_H = -omega * mu / denom
    xb_H = -omega * mu

    delta_sig = sig - sig_b
    if abs(delta_sig) < 1e-5:
        delta_sig = 0.0

    return dict(xk_E=xk_E, xb_E=xb_E, xk_H=xk_H, xb_H=xb_H,
                delta_sig=delta_sig, k2=k2, sig=sig)


def _coupling_coefficients(
    resistivity: float,
    layer_resistivity: float,
    omega: float,
    ky: float,
    mu: float = MU_0,
    epsilon: float = EPSILON_0,
    jacobian_mode: bool = False,
) -> tuple[complex, complex, complex, complex]:
    """
    결합 계수 (E↔H 교차 항)

    Fortran 대응: Cal_dsig 서브루틴 (Fem25Dsub.f, 라인 410-456)

    2.5D Maxwell 방정식의 ky 방향 교차 미분 결합:
      E 방정식: xc1_E = -ky / (ky²-k²),  xc2_E = +ky / (ky²-k²)
      H 방정식: xc1_H = +ky / (ky²-k²),  xc2_H = -ky / (ky²-k²)

    Fortran 원본:
      E모드: xc1 = -ky,  xc2 = +ky  (실수값)
      H모드: xc1 = +ky,  xc2 = -ky  (부호 반전)
      공통:  const = 1/(ky²-k²)  (forward) 또는 -iωμ/(ky²-k²)² (jacobian)
      최종:  xc1 *= const,  xc2 *= const
    """
    sig = _SIG_AIR if resistivity == 0 else 1.0 / resistivity
    k2 = mu * epsilon * omega**2 - 1j * mu * sig * omega
    ke2 = ky**2 - k2

    # E 모드 기본 부호: xc1 = -ky, xc2 = +ky
    xc1_E = complex(-ky, 0.0)
    xc2_E = complex(+ky, 0.0)

    # H 모드 기본 부호: 반전 → xc1 = +ky, xc2 = -ky
    xc1_H = complex(+ky, 0.0)
    xc2_H = complex(-ky, 0.0)

    # 공통 스케일링 팩터
    if jacobian_mode:
        const = complex(0.0, -omega * mu) / ke2**2
    else:
        const = 1.0 / ke2

    xc1_E *= const
    xc2_E *= const
    xc1_H *= const
    xc2_H *= const

    return xc1_E, xc2_E, xc1_H, xc2_H


def assemble_element_matrix(
    element_coords: np.ndarray,    # (4, 2) 요소 4노드의 (x, z) 좌표
    resistivity: float,            # 요소 비저항
    layer_resistivity: float,      # 배경 층 비저항
    E_primary_nodes: np.ndarray,   # (3, 4) 요소 4노드의 1차장 [Ex,Ey,Ez]
    omega: float,
    ky: float,
    mu: float = MU_0,
    epsilon: float = EPSILON_0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    단일 요소 강성행렬 및 힘벡터 조립

    Fortran 대응: cal_elem 서브루틴

    반환:
      Ke : (8, 8) 요소 강성행렬 (DOF=2×4노드)
      fe : (8,)   요소 힘벡터
           홀수 행/열(0,2,4,6): Ey DOF
           짝수 행/열(1,3,5,7): Hy DOF
    """
    n_nodes = 4
    n_dof   = 2 * n_nodes   # 8 DOF

    Ke = np.zeros((n_dof, n_dof), dtype=complex)
    fe = np.zeros(n_dof, dtype=complex)

    mat = _material_coefficients(resistivity, layer_resistivity,
                                  omega, ky, mu, epsilon)
    xc1_E, xc2_E, xc1_H, xc2_H = _coupling_coefficients(
        resistivity, layer_resistivity, omega, ky, mu, epsilon)

    # 1차장 노드 평균 (요소 내 균일 가정)
    Exp_avg = np.mean(E_primary_nodes[0])
    Eyp_avg = np.mean(E_primary_nodes[1])
    Ezp_avg = np.mean(E_primary_nodes[2])

    delta_sig = mat['delta_sig']
    k2        = mat['k2']
    d         = ky**2 - k2

    # 2×2 Gauss 구적
    for gxi in _GAUSS_PTS:
        for geta in _GAUSS_PTS:
            N, dN_nat = _shape_functions(gxi, geta)
            w = 1.0 * 1.0   # 가중치 = 1×1

            # Jacobian 행렬 (자연좌표 → 물리좌표)
            J = dN_nat @ element_coords   # (2,4)@(4,2) = (2,2)
            detJ = J[0,0]*J[1,1] - J[0,1]*J[1,0]
            if detJ <= 0:
                raise ValueError(f"요소 Jacobian 음수: detJ={detJ:.4g}")

            Jinv = np.array([[ J[1,1], -J[0,1]],
                             [-J[1,0],  J[0,0]]]) / detJ

            # 형상함수 물리 좌표 편미분 dN/dx, dN/dz
            dN_phys = Jinv @ dN_nat   # (2,4): [dN/dx ; dN/dz]
            dNdx = dN_phys[0]         # (4,)
            dNdz = dN_phys[1]         # (4,)

            fac = detJ * w

            # ── 강성행렬 조립 ───────────────────────────────────────────
            for i in range(n_nodes):
                iE = 2 * i      # Ey DOF
                iH = 2 * i + 1  # Hy DOF

                for j in range(n_nodes):
                    jE = 2 * j
                    jH = 2 * j + 1

                    grad_grad = dNdx[i]*dNdx[j] + dNdz[i]*dNdz[j]
                    mass      = N[i] * N[j]

                    # E 방정식 (대각)
                    Ke[iE, jE] += fac * (mat['xk_E'] * grad_grad
                                         + mat['xb_E'] * mass)
                    # H 방정식 (대각)
                    Ke[iH, jH] += fac * (mat['xk_H'] * grad_grad
                                         + mat['xb_H'] * mass)
                    # E←H 결합 (Fortran cal_elem 라인 114-116)
                    # ek(ii, jj1) = ek(ii,jj1) + fac*(xc1*dpsix(i)*dpsiz(j) + xc2*dpsiz(i)*dpsix(j))
                    Ke[iE, jH] += fac * (xc1_E * dNdx[i]*dNdz[j]
                                         + xc2_E * dNdz[i]*dNdx[j])
                    # H←E 결합 (Fortran cal_elem 라인 143-145) — 동일한 dNdx/dNdz 패턴
                    # ek(ii, jj1) = ek(ii,jj1) + fac*(xc1*dpsix(i)*dpsiz(j) + xc2*dpsiz(i)*dpsix(j))
                    Ke[iH, jE] += fac * (xc1_H * dNdx[i]*dNdz[j]
                                         + xc2_H * dNdz[i]*dNdx[j])

                # ── 힘벡터 (2차장 소스항) ────────────────────────────────
                if delta_sig != 0.0:
                    # E 방정식 소스: Fortran source_e
                    fe[iE] += fac * (
                        -1j * delta_sig * Eyp_avg * N[i]
                        + ky * delta_sig / d
                        * (Exp_avg * dNdx[i] + Ezp_avg * dNdz[i])
                    )
                    # H 방정식 소스: Fortran source_h
                    fe[iH] += fac * (
                        delta_sig / d * omega * mu
                        * (Exp_avg * dNdz[i] - Ezp_avg * dNdx[i])
                    )

    return Ke, fe


# ══════════════════════════════════════════════════════════════════════════════
# Precomputed reference integrals for rectangular elements
# ══════════════════════════════════════════════════════════════════════════════
#
# For a rectangular element mapped to [-1,1]x[-1,1], the Jacobian is diagonal:
#   J = diag(dx/2, dz/2),  detJ = dx*dz/4
#   dN/dx = (2/dx) * dN/dxi,  dN/dz = (2/dz) * dN/deta
#
# We precompute reference integrals over the master element [-1,1]x[-1,1]
# using 2x2 Gauss quadrature:
#   M_ref[i,j]  = sum_gp { N_i * N_j }                       (mass)
#   Kxx_ref[i,j] = sum_gp { dN_i/dxi * dN_j/dxi }            (stiffness xx)
#   Kzz_ref[i,j] = sum_gp { dN_i/deta * dN_j/deta }          (stiffness zz)
#   Cxz_ref[i,j] = sum_gp { dN_i/dxi * dN_j/deta }           (coupling xz)
#   Czx_ref[i,j] = sum_gp { dN_i/deta * dN_j/dxi }           (coupling zx)
#   Nx_ref[i]    = sum_gp { dN_i/dxi }                        (force grad x)
#   Nz_ref[i]    = sum_gp { dN_i/deta }                       (force grad z)
#   N_ref[i]     = sum_gp { N_i }                             (force mass)
#
# Physical integrals are then:
#   grad_grad integral = (dz/dx) * Kxx_ref + (dx/dz) * Kzz_ref
#   mass integral      = (dx*dz/4) * M_ref
#   coupling integral  = Cxz_ref (no dx,dz factor since dN/dx*dN/dz * detJ
#                        = (2/dx)*(2/dz)*(dx*dz/4) = 1)
#   force N integral   = (dx*dz/4) * N_ref
#   force dNdx integral= (dz/2) * Nx_ref
#   force dNdz integral= (dx/2) * Nz_ref

def _precompute_reference_integrals():
    """
    Precompute reference element integrals on [-1,1]^2 with 2x2 Gauss quadrature.
    These are constant matrices independent of element geometry.
    """
    M_ref   = np.zeros((4, 4))
    Kxx_ref = np.zeros((4, 4))
    Kzz_ref = np.zeros((4, 4))
    Cxz_ref = np.zeros((4, 4))
    Czx_ref = np.zeros((4, 4))
    N_ref   = np.zeros(4)
    Nx_ref  = np.zeros(4)
    Nz_ref  = np.zeros(4)

    for gxi in _GAUSS_PTS:
        for geta in _GAUSS_PTS:
            N, dN = _shape_functions(gxi, geta)
            dNdxi  = dN[0]   # (4,)
            dNdeta = dN[1]   # (4,)

            # Gauss weight = 1.0 * 1.0 = 1.0
            M_ref   += np.outer(N, N)
            Kxx_ref += np.outer(dNdxi, dNdxi)
            Kzz_ref += np.outer(dNdeta, dNdeta)
            Cxz_ref += np.outer(dNdxi, dNdeta)
            Czx_ref += np.outer(dNdeta, dNdxi)
            N_ref   += N
            Nx_ref  += dNdxi
            Nz_ref  += dNdeta

    return M_ref, Kxx_ref, Kzz_ref, Cxz_ref, Czx_ref, N_ref, Nx_ref, Nz_ref

# Compute once at module load
_M_REF, _KXX_REF, _KZZ_REF, _CXZ_REF, _CZX_REF, _N_REF, _NX_REF, _NZ_REF = \
    _precompute_reference_integrals()


def assemble_global_system(
    grid,              # Grid 객체
    element_resistivity: np.ndarray,    # (n_ex, n_ez)
    layer_resistivity: np.ndarray,      # (n_ex, n_ez)  배경 층
    E_primary: np.ndarray,              # (3, n_nodes) 노드별 1차장
    omega: float,
    ky: float,
    mu: float = MU_0,
    epsilon: float = EPSILON_0,
) -> tuple:
    """
    전체 격자 전역 강성행렬 및 힘벡터 조립 (sparse) — 벡터화 버전

    직교(rectangular) 격자 전용 최적화:
      - 모든 요소의 재질/결합 계수를 한 번에 벡터 연산
      - 사전 계산된 참조 적분행렬 사용 (dx, dz 스케일링만 적용)
      - Python 루프 제거, COO 형식 일괄 조립

    Fortran 대응: FWD2 서브루틴 내 요소 루프

    반환:
      K_global : (2*n_nodes, 2*n_nodes) CSR sparse 복소 행렬
      f_global : (2*n_nodes,) 복소 벡터
    """
    n_ex = grid.n_elements_x
    n_ez = grid.n_elements_z
    n_nodes_x = grid.n_nodes_x
    n_nodes = grid.n_nodes
    n_dof = 2 * n_nodes
    n_elem = n_ex * n_ez

    x_nodes = grid.node_x[:, 0]   # (n_nodes_x,)
    z_nodes = grid.node_z[0, :]   # (n_nodes_z,)

    # ── 1. Element sizes ─────────────────────────────────────────────────────
    dx_vec = np.diff(x_nodes)  # (n_ex,)
    dz_vec = np.diff(z_nodes)  # (n_ez,)

    # Broadcast to (n_ex, n_ez)
    dx_all = dx_vec[:, np.newaxis] * np.ones(n_ez)  # (n_ex, n_ez)
    dz_all = np.ones(n_ex)[:, np.newaxis] * dz_vec  # (n_ex, n_ez)

    # Flatten to (n_elem,) in row-major order (ex varies slowest)
    dx = dx_all.ravel()  # (n_elem,)
    dz = dz_all.ravel()  # (n_elem,)

    # ── 2. Material coefficients (vectorized over all elements) ──────────────
    rho   = element_resistivity.ravel()   # (n_elem,)
    rho_b = layer_resistivity.ravel()     # (n_elem,)

    sig   = np.where(rho == 0, _SIG_AIR, 1.0 / rho)
    sig_b = np.where(rho_b == 0, _SIG_AIR, 1.0 / rho_b)

    k2    = mu * epsilon * omega**2 - 1j * mu * sig * omega   # (n_elem,)
    denom = ky**2 - k2                                         # (n_elem,)

    sig_iwe = sig + 1j * omega * epsilon   # σ + iωε  (n_elem,)

    xk_E = 1j * sig_iwe / denom
    xb_E = 1j * sig_iwe
    xk_H = -omega * mu / denom       # scalar / array = array
    xb_H = -omega * mu               # scalar (same for all elements)

    # Coupling coefficients
    const = 1.0 / denom               # (n_elem,)
    xc1_E = -ky * const               # (n_elem,)
    xc2_E =  ky * const
    xc1_H =  ky * const
    xc2_H = -ky * const

    # Delta sigma for source terms
    delta_sig = sig - sig_b            # (n_elem,)
    delta_sig = np.where(np.abs(delta_sig) < 1e-5, 0.0, delta_sig)

    # ── 3. Node indices for all elements ─────────────────────────────────────
    # Element (ex, ez) has corners at grid positions (ex, ez), (ex+1, ez),
    # (ex+1, ez+1), (ex, ez+1)
    # node_index(ix, iz) = iz * n_nodes_x + ix

    ex_idx = np.arange(n_ex)[:, np.newaxis] * np.ones(n_ez, dtype=int)  # (n_ex, n_ez)
    ez_idx = np.ones(n_ex, dtype=int)[:, np.newaxis] * np.arange(n_ez)

    ex_flat = ex_idx.ravel()   # (n_elem,)
    ez_flat = ez_idx.ravel()   # (n_elem,)

    # 4 corner node indices: n0(ex,ez), n1(ex+1,ez), n2(ex+1,ez+1), n3(ex,ez+1)
    n0 = ez_flat * n_nodes_x + ex_flat             # (n_elem,)
    n1 = ez_flat * n_nodes_x + (ex_flat + 1)
    n2 = (ez_flat + 1) * n_nodes_x + (ex_flat + 1)
    n3 = (ez_flat + 1) * n_nodes_x + ex_flat

    # Stack: (n_elem, 4) — node indices for each element
    node_ids = np.stack([n0, n1, n2, n3], axis=1)   # (n_elem, 4)

    # ── 4. Primary field averages per element ────────────────────────────────
    # E_primary: (3, n_nodes), node_ids: (n_elem, 4)
    Ep_elem = E_primary[:, node_ids]   # (3, n_elem, 4)
    Exp_avg = Ep_elem[0].mean(axis=1)  # (n_elem,)
    Eyp_avg = Ep_elem[1].mean(axis=1)
    Ezp_avg = Ep_elem[2].mean(axis=1)

    # ── 5. Build element 8x8 matrices using reference integrals ──────────────
    #
    # For rectangular element with width dx, height dz:
    #   detJ = dx*dz/4 (constant over element)
    #   dN/dx = (2/dx) * dN/dxi,  dN/dz = (2/dz) * dN/deta
    #
    # Physical integrals (summed over 4 Gauss points, weight=1 each):
    #   grad_grad[i,j] = sum_gp detJ * (dNdx_i*dNdx_j + dNdz_i*dNdz_j)
    #                  = (dz/dx) * Kxx_ref[i,j] + (dx/dz) * Kzz_ref[i,j]
    #   mass[i,j]      = sum_gp detJ * N_i * N_j
    #                  = (dx*dz/4) * M_ref[i,j]
    #   coupling_xz[i,j] = sum_gp detJ * dNdx_i * dNdz_j
    #                     = Cxz_ref[i,j]   (factors cancel!)
    #   coupling_zx[i,j] = sum_gp detJ * dNdz_i * dNdx_j
    #                     = Czx_ref[i,j]

    # Scaling factors per element: (n_elem,)
    dz_over_dx = dz / dx
    dx_over_dz = dx / dz
    dxdz_4     = dx * dz / 4.0

    # Reference matrices are (4, 4), scale by element-specific factors
    # Result: (n_elem, 4, 4) for each integral type

    # grad_grad(i,j) per element
    GG = (dz_over_dx[:, None, None] * _KXX_REF[None, :, :]
          + dx_over_dz[:, None, None] * _KZZ_REF[None, :, :])   # (n_elem, 4, 4)

    MM = dxdz_4[:, None, None] * _M_REF[None, :, :]              # (n_elem, 4, 4)

    # Coupling matrices are the same for all elements (geometry factors cancel)
    CXZ = _CXZ_REF  # (4, 4) — broadcast as needed
    CZX = _CZX_REF  # (4, 4)

    # ── 6. Assemble 8x8 element matrices ─────────────────────────────────────
    # Ke is 8x8 with interleaved DOFs: [Ey0, Hy0, Ey1, Hy1, Ey2, Hy2, Ey3, Hy3]
    # We build 4 blocks (4x4 each in node space), then scatter to 8x8.

    # EE block: Ke[2i, 2j] = xk_E * GG[i,j] + xb_E * MM[i,j]
    KE_EE = xk_E[:, None, None] * GG + xb_E[:, None, None] * MM   # (n_elem, 4, 4)

    # HH block: Ke[2i+1, 2j+1] = xk_H * GG[i,j] + xb_H * MM[i,j]
    KE_HH = xk_H[:, None, None] * GG + xb_H * MM   # (n_elem, 4, 4)

    # EH block: Ke[2i, 2j+1] = xc1_E * Cxz[i,j] + xc2_E * Czx[i,j]
    KE_EH = xc1_E[:, None, None] * CXZ[None, :, :] + xc2_E[:, None, None] * CZX[None, :, :]

    # HE block: Ke[2i+1, 2j] = xc1_H * Cxz[i,j] + xc2_H * Czx[i,j]
    KE_HE = xc1_H[:, None, None] * CXZ[None, :, :] + xc2_H[:, None, None] * CZX[None, :, :]

    # ── 7. Force vector per element ──────────────────────────────────────────
    #
    # Physical integrals for force vector (summed over Gauss pts):
    #   sum_gp detJ * N_i        = (dx*dz/4) * N_ref[i]
    #   sum_gp detJ * dNdx_i     = (dz/2) * Nx_ref[i]
    #   sum_gp detJ * dNdz_i     = (dx/2) * Nz_ref[i]

    # Force vector entries (only where delta_sig != 0)
    has_source = delta_sig != 0.0
    d = denom   # ky^2 - k^2, (n_elem,)

    # Preallocate force in (n_elem, 4) for E and H DOFs
    fe_E = np.zeros((n_elem, 4), dtype=complex)
    fe_H = np.zeros((n_elem, 4), dtype=complex)

    if np.any(has_source):
        # Scale factors for force integrals: (n_elem,)
        fN  = dxdz_4          # sum_gp detJ * N_i  scale
        fdx = dz / 2.0        # sum_gp detJ * dNdx_i scale
        fdz = dx / 2.0        # sum_gp detJ * dNdz_i scale

        # fe_E[i] = sum_gp fac * (-1j * delta_sig * Eyp_avg * N[i]
        #           + ky * delta_sig / d * (Exp_avg * dNdx[i] + Ezp_avg * dNdz[i]))
        # = -1j * delta_sig * Eyp_avg * fN * N_ref[i]
        #   + ky * delta_sig / d * (Exp_avg * fdx * Nx_ref[i] + Ezp_avg * fdz * Nz_ref[i])

        ds = delta_sig          # (n_elem,)
        ds_d = ds / d           # (n_elem,)

        # E force: (n_elem, 4)
        fe_E = ((-1j * ds * Eyp_avg * fN)[:, None] * _N_REF[None, :]
                + (ky * ds_d * Exp_avg * fdx)[:, None] * _NX_REF[None, :]
                + (ky * ds_d * Ezp_avg * fdz)[:, None] * _NZ_REF[None, :])

        # H force: (n_elem, 4)
        # fe_H[i] = sum_gp fac * delta_sig / d * omega * mu
        #           * (Exp_avg * dNdz[i] - Ezp_avg * dNdx[i])
        omu_ds_d = omega * mu * ds_d   # (n_elem,)
        fe_H = ((omu_ds_d * Exp_avg * fdz)[:, None] * _NZ_REF[None, :]
                - (omu_ds_d * Ezp_avg * fdx)[:, None] * _NX_REF[None, :])

        # Zero out elements without source
        no_source = ~has_source
        fe_E[no_source] = 0.0
        fe_H[no_source] = 0.0

    # ── 8. Build global DOF indices and COO data ─────────────────────────────
    # For each element, we have 8 DOFs: [2*n0, 2*n0+1, 2*n1, 2*n1+1, ...]
    # Global DOF indices: (n_elem, 8)
    gdof = np.empty((n_elem, 8), dtype=np.int64)
    gdof[:, 0] = 2 * node_ids[:, 0]       # Ey of node 0
    gdof[:, 1] = 2 * node_ids[:, 0] + 1   # Hy of node 0
    gdof[:, 2] = 2 * node_ids[:, 1]       # Ey of node 1
    gdof[:, 3] = 2 * node_ids[:, 1] + 1   # Hy of node 1
    gdof[:, 4] = 2 * node_ids[:, 2]       # Ey of node 2
    gdof[:, 5] = 2 * node_ids[:, 2] + 1   # Hy of node 2
    gdof[:, 6] = 2 * node_ids[:, 3]       # Ey of node 3
    gdof[:, 7] = 2 * node_ids[:, 3] + 1   # Hy of node 3

    # Build 8x8 element stiffness matrices from the 4 blocks
    # Layout in 8x8:
    #   rows 0,2,4,6 (E) x cols 0,2,4,6 (E) -> KE_EE
    #   rows 1,3,5,7 (H) x cols 1,3,5,7 (H) -> KE_HH
    #   rows 0,2,4,6 (E) x cols 1,3,5,7 (H) -> KE_EH
    #   rows 1,3,5,7 (H) x cols 0,2,4,6 (E) -> KE_HE

    # Instead of building full 8x8 and scattering, build COO directly from blocks.
    # Each 4x4 block contributes 16 entries per element, 4 blocks = 64 entries.
    # Total COO entries: n_elem * 64

    # Node-pair local indices for 4x4 blocks
    li, lj = np.meshgrid(np.arange(4), np.arange(4), indexing='ij')
    li_flat = li.ravel()  # (16,) : 0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3
    lj_flat = lj.ravel()  # (16,) : 0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3

    # Global DOF for E and H of each node
    gdof_E = gdof[:, 0::2]  # (n_elem, 4) — columns 0,2,4,6
    gdof_H = gdof[:, 1::2]  # (n_elem, 4) — columns 1,3,5,7

    # Row/col indices for each block: (n_elem, 16)
    row_EE = gdof_E[:, li_flat]   # (n_elem, 16)
    col_EE = gdof_E[:, lj_flat]
    row_HH = gdof_H[:, li_flat]
    col_HH = gdof_H[:, lj_flat]
    row_EH = gdof_E[:, li_flat]
    col_EH = gdof_H[:, lj_flat]
    row_HE = gdof_H[:, li_flat]
    col_HE = gdof_E[:, lj_flat]

    # Values for each block: (n_elem, 4, 4) -> (n_elem, 16)
    val_EE = KE_EE.reshape(n_elem, 16)
    val_HH = KE_HH.reshape(n_elem, 16)
    val_EH = KE_EH.reshape(n_elem, 16)
    val_HE = KE_HE.reshape(n_elem, 16)

    # Concatenate all 4 blocks
    all_rows = np.concatenate([row_EE, row_HH, row_EH, row_HE], axis=1).ravel()  # (n_elem*64,)
    all_cols = np.concatenate([col_EE, col_HH, col_EH, col_HE], axis=1).ravel()
    all_vals = np.concatenate([val_EE, val_HH, val_EH, val_HE], axis=1).ravel()

    K_global = sp.coo_matrix(
        (all_vals, (all_rows, all_cols)), shape=(n_dof, n_dof)).tocsr()

    # ── 9. Assemble force vector ─────────────────────────────────────────────
    # Interleave fe_E and fe_H into 8-entry vectors, then scatter
    # fe[2*node_k] += fe_E[elem, k],  fe[2*node_k+1] += fe_H[elem, k]
    f_global = np.zeros(n_dof, dtype=complex)

    if np.any(has_source):
        # Scatter using np.add.at for proper accumulation of duplicate indices
        for k in range(4):
            np.add.at(f_global, gdof_E[:, k], fe_E[:, k])
            np.add.at(f_global, gdof_H[:, k], fe_H[:, k])

    return K_global, f_global


def assemble_force_vector(
    grid,
    element_resistivity: np.ndarray,    # (n_ex, n_ez)
    layer_resistivity: np.ndarray,      # (n_ex, n_ez)
    E_primary: np.ndarray,              # (3, n_nodes)
    omega: float,
    ky: float,
    mu: float = MU_0,
    epsilon: float = EPSILON_0,
) -> np.ndarray:
    """
    힘벡터만 조립 (K-reuse 최적화용)

    K 행렬은 E_primary에 의존하지 않으므로 (freq, ky)마다 한 번만 조립.
    f 벡터만 각 송신기별로 재조립.

    반환:
      f_global : (2*n_nodes,) 복소 벡터
    """
    n_ex = grid.n_elements_x
    n_ez = grid.n_elements_z
    n_nodes_x = grid.n_nodes_x
    n_nodes = grid.n_nodes
    n_dof = 2 * n_nodes
    n_elem = n_ex * n_ez

    x_nodes = grid.node_x[:, 0]
    z_nodes = grid.node_z[0, :]

    dx_vec = np.diff(x_nodes)
    dz_vec = np.diff(z_nodes)
    dx_all = dx_vec[:, np.newaxis] * np.ones(n_ez)
    dz_all = np.ones(n_ex)[:, np.newaxis] * dz_vec
    dx = dx_all.ravel()
    dz = dz_all.ravel()

    rho   = element_resistivity.ravel()
    rho_b = layer_resistivity.ravel()
    sig   = np.where(rho == 0, _SIG_AIR, 1.0 / rho)
    sig_b = np.where(rho_b == 0, _SIG_AIR, 1.0 / rho_b)

    k2    = mu * epsilon * omega**2 - 1j * mu * sig * omega
    denom = ky**2 - k2

    delta_sig = sig - sig_b
    delta_sig = np.where(np.abs(delta_sig) < 1e-5, 0.0, delta_sig)

    # Node indices
    ex_flat = (np.arange(n_ex)[:, np.newaxis] * np.ones(n_ez, dtype=int)).ravel()
    ez_flat = (np.ones(n_ex, dtype=int)[:, np.newaxis] * np.arange(n_ez)).ravel()
    n0 = ez_flat * n_nodes_x + ex_flat
    n1 = ez_flat * n_nodes_x + (ex_flat + 1)
    n2 = (ez_flat + 1) * n_nodes_x + (ex_flat + 1)
    n3 = (ez_flat + 1) * n_nodes_x + ex_flat
    node_ids = np.stack([n0, n1, n2, n3], axis=1)

    # Primary field averages
    Ep_elem = E_primary[:, node_ids]
    Exp_avg = Ep_elem[0].mean(axis=1)
    Eyp_avg = Ep_elem[1].mean(axis=1)
    Ezp_avg = Ep_elem[2].mean(axis=1)

    # Geometry scaling
    dxdz_4 = dx * dz / 4.0

    has_source = delta_sig != 0.0
    d = denom

    fe_E = np.zeros((n_elem, 4), dtype=complex)
    fe_H = np.zeros((n_elem, 4), dtype=complex)

    if np.any(has_source):
        fN  = dxdz_4
        fdx = dz / 2.0
        fdz = dx / 2.0
        ds = delta_sig
        ds_d = ds / d

        fe_E = ((-1j * ds * Eyp_avg * fN)[:, None] * _N_REF[None, :]
                + (ky * ds_d * Exp_avg * fdx)[:, None] * _NX_REF[None, :]
                + (ky * ds_d * Ezp_avg * fdz)[:, None] * _NZ_REF[None, :])

        omu_ds_d = omega * mu * ds_d
        fe_H = ((omu_ds_d * Exp_avg * fdz)[:, None] * _NZ_REF[None, :]
                - (omu_ds_d * Ezp_avg * fdx)[:, None] * _NX_REF[None, :])

        no_source = ~has_source
        fe_E[no_source] = 0.0
        fe_H[no_source] = 0.0

    # Global DOF indices
    gdof_E = 2 * node_ids        # (n_elem, 4)
    gdof_H = 2 * node_ids + 1    # (n_elem, 4)

    f_global = np.zeros(n_dof, dtype=complex)
    if np.any(has_source):
        for k in range(4):
            np.add.at(f_global, gdof_E[:, k], fe_E[:, k])
            np.add.at(f_global, gdof_H[:, k], fe_H[:, k])

    return f_global
