"""
후처리(Post-processing): 보조 장 성분 계산

Fortran 대응:
  FemPost (Fem25DPost.f90, 라인 6-306): 프로파일 노드에서 Ey, Hy 및 미분 추출
  FemIntegral (Fem25DPost.f90, 라인 314-460): ky 영역 보조 장 계산

물리적 배경:
  FEM에서 직접 구한 해: Ey_s(2차장), Hy_s(2차장)
  나머지 4성분(Ex, Ez, Hx, Hz)은 Maxwell 방정식 ky 영역 관계식으로 유도.

  Fortran FemIntegral 변수 대응 (주의: Fortran 이름이 물리 성분과 다름):
    Fortran Ex_field  → Ey_secondary (FEM Ey DOF)
    Fortran Hx_field  → Hy_secondary (FEM Hy DOF)
    Fortran Ey_Field  → ∂Ey_s/∂x (형상함수 x미분)
    Fortran Ez_Field  → ∂Ey_s/∂z (형상함수 z미분)
    Fortran Hy_Field  → ∂Hy_s/∂x (형상함수 x미분)
    Fortran Hz_Field  → ∂Hy_s/∂z (형상함수 z미분)

  보조 장 공식 (Fortran 라인 412-428):
    ke2 = ky² - k²  (k² = ω²με - iωμσ)
    ka2 = -iωμΔσ    (Δσ = σ - σ_layer, σ_layer=0 free-space)

    Hx = [(σ+iωε)·∂Ey_s/∂z - iky·∂Hy_s/∂x + iky·Δσ·Ez_primary] / ke2
    Hz = [-(σ+iωε)·∂Ey_s/∂x - iky·∂Hy_s/∂z - iky·Δσ·Ex_primary] / ke2
    Ex = [-iky·∂Ey_s/∂x - iωμ·∂Hy_s/∂z + ka2·Ex_primary] / ke2
    Ez = [-iky·∂Ey_s/∂z + iωμ·∂Hy_s/∂x + ka2·Ez_primary] / ke2
    Ey = Ey_secondary (FEM 해 그대로)
    Hy = Hy_secondary (FEM 해 그대로)

    ★ 모든 미분은 2차장(secondary)에 대해 계산 ★
    ★ 1차장 보정은 delta_sig, ka2 항으로만 포함 ★

  이후 forward_loop에서:
    1. IFT: 6성분 모두 ky→공간 변환
    2. 공간 영역 1차장 추가 (total field 모드 시)

참고:
  Fortran은 프로파일 노드에서 형상함수 미분을 사용하지만,
  Python은 전체 격자에서 np.gradient 유한차분으로 근사 (2차 중앙차분).
  수신기가 격자 노드 위에 있으면 정확도 차이 무시 가능.
"""

from __future__ import annotations

import numpy as np
from ..constants import MU_0, EPSILON_0


def compute_secondary_field_components(
    Ey_secondary: np.ndarray,    # (n_nodes,) 2차장
    Hy_secondary: np.ndarray,    # (n_nodes,) 2차장
    E_primary: np.ndarray,       # (3, n_nodes): [Ex_p, Ey_p, Ez_p] ky영역 1차장
    grid,
    omega: float,
    ky: float,
    resistivity_elem: np.ndarray,    # (n_ex, n_ez) 요소 비저항
    background_resistivity: float,   # 배경 비저항 (0 = free-space)
    mu: float = MU_0,
    epsilon: float = EPSILON_0,
) -> dict:
    """
    FEM 해(Ey_s, Hy_s)에서 6성분 ky영역 2차장 계산

    Fortran 대응: FemIntegral (Fem25DPost.f90, 라인 395-434)

    ★ 입출력은 모두 2차장(secondary) ★
    ★ 1차장 보정은 delta_sig 항으로 포함 ★

    Returns
    -------
    fields : dict with keys "Ex","Ey","Ez","Hx","Hy","Hz"
             각 값: (n_nodes,) 복소 배열 (ky 영역 2차장)
    """
    n_nodes_x = grid.n_nodes_x
    n_nodes_z = grid.n_nodes_z

    # 2D reshape: node_index(ix,iz) = iz*n_nodes_x + ix → arr2d[iz, ix]
    def to_2d(arr):
        return arr.reshape(n_nodes_z, n_nodes_x, order='C')

    def to_1d(arr):
        return arr.reshape(-1, order='C')

    x_nodes = grid.node_x[:, 0]   # (n_nodes_x,)
    z_nodes = grid.node_z[0, :]   # (n_nodes_z,)

    # ── 2차장 Ey_s, Hy_s (FEM 해 그대로, 1차장 미포함) ─────────────────────
    Ey_s_2d = to_2d(Ey_secondary)
    Hy_s_2d = to_2d(Hy_secondary)

    # ── 2차장 수치 미분 (∂/∂x, ∂/∂z) ──────────────────────────────────────
    # arr2d[iz, ix] → axis=0: z 방향, axis=1: x 방향
    dEy_s_dx = np.gradient(Ey_s_2d, x_nodes, axis=1)
    dEy_s_dz = np.gradient(Ey_s_2d, z_nodes, axis=0)
    dHy_s_dx = np.gradient(Hy_s_2d, x_nodes, axis=1)
    dHy_s_dz = np.gradient(Hy_s_2d, z_nodes, axis=0)

    # ── 요소 전도도 → 노드 보간 ────────────────────────────────────────────
    sig_elem = 1.0 / np.where(resistivity_elem == 0, 1e8, resistivity_elem)
    sig_2d = _element_to_node_avg(sig_elem, n_nodes_x, n_nodes_z)

    # 배경 전도도 (free-space: σ_layer = 0)
    if background_resistivity > 0:
        sig_bg = 1.0 / background_resistivity
    else:
        sig_bg = 0.0  # free-space

    # 전도도 대비 Δσ = σ - σ_layer
    delta_sig_2d = sig_2d - sig_bg

    # ── 파수 관련 상수 ─────────────────────────────────────────────────────
    k2 = mu * epsilon * omega**2 - 1j * mu * sig_2d * omega
    ke2 = ky**2 - k2                # (n_nodes_z, n_nodes_x)
    ka2 = complex(0.0, -omega * mu) * delta_sig_2d   # -iωμΔσ

    sig_iweps = sig_2d + 1j * omega * epsilon   # σ + iωε
    iwmu = 1j * omega * mu                       # iωμ
    iky_val = 1j * ky                            # iky

    # ── 1차장 2D 배열 (ky 영역) ──────────────────────────────────────────
    Ex_p_2d = to_2d(E_primary[0])
    Ez_p_2d = to_2d(E_primary[2])

    # ── FemIntegral 공식 (Fortran 라인 412-428) ────────────────────────────
    # ★ 미분은 모두 2차장에 대해 계산 ★

    # Hx = [(σ+iωε)·∂Ey_s/∂z - iky·∂Hy_s/∂x + iky·Δσ·Ez_p] / ke2
    Hx_2d = (sig_iweps * dEy_s_dz
             - iky_val * dHy_s_dx
             + iky_val * delta_sig_2d * Ez_p_2d) / ke2

    # Hz = [-(σ+iωε)·∂Ey_s/∂x - iky·∂Hy_s/∂z - iky·Δσ·Ex_p] / ke2
    Hz_2d = (-sig_iweps * dEy_s_dx
             - iky_val * dHy_s_dz
             - iky_val * delta_sig_2d * Ex_p_2d) / ke2

    # Ex = [-iky·∂Ey_s/∂x - iωμ·∂Hy_s/∂z + ka2·Ex_p] / ke2
    Ex_2d = (-iky_val * dEy_s_dx
             - iwmu * dHy_s_dz
             + ka2 * Ex_p_2d) / ke2

    # Ez = [-iky·∂Ey_s/∂z + iωμ·∂Hy_s/∂x + ka2·Ez_p] / ke2
    Ez_2d = (-iky_val * dEy_s_dz
             + iwmu * dHy_s_dx
             + ka2 * Ez_p_2d) / ke2

    return {
        "Ex": to_1d(Ex_2d),
        "Ey": to_1d(Ey_secondary),   # 2차장 그대로
        "Ez": to_1d(Ez_2d),
        "Hx": to_1d(Hx_2d),
        "Hy": to_1d(Hy_secondary),   # 2차장 그대로
        "Hz": to_1d(Hz_2d),
    }


def extract_profile_fields(
    fields: dict,
    profile_node_indices: np.ndarray,
) -> dict:
    """프로파일 수신기 위치의 장 값 추출"""
    return {comp: arr[profile_node_indices]
            for comp, arr in fields.items()}


def compute_fields_at_profile(
    Ey_secondary: np.ndarray,    # (n_nodes,)
    Hy_secondary: np.ndarray,    # (n_nodes,)
    E_primary: np.ndarray,       # (3, n_nodes)
    grid,
    omega: float,
    ky: float,
    resistivity_elem: np.ndarray,
    background_resistivity: float,
    profile_node_indices: np.ndarray,  # (n_rx,) Python 전역 노드 인덱스
    mu: float = MU_0,
    epsilon: float = EPSILON_0,
) -> np.ndarray:
    """
    프로파일 노드에서만 6성분 ky영역 2차장 계산 — 최적화 버전

    전체 격자 np.gradient 대신 프로파일 노드 주변의 유한차분만 계산.
    4575 노드 전체 대신 22 프로파일 노드만 처리하여 ~40배 빠름.

    Returns
    -------
    fields_rx : (n_rx, 6) complex — [Ex, Ey, Ez, Hx, Hy, Hz]
    """
    n_rx = len(profile_node_indices)
    n_nodes_x = grid.n_nodes_x
    n_nodes_z = grid.n_nodes_z

    x_nodes = grid.node_x[:, 0]
    z_nodes = grid.node_z[0, :]

    # 프로파일 노드의 (ix, iz) 인덱스
    ix_arr = profile_node_indices % n_nodes_x
    iz_arr = profile_node_indices // n_nodes_x

    # 요소 전도도 → 노드 전도도 (프로파일 노드 + 이웃만)
    sig_elem = 1.0 / np.where(resistivity_elem == 0, 1e8, resistivity_elem)

    # 배경 전도도
    sig_bg = 1.0 / background_resistivity if background_resistivity > 0 else 0.0

    result = np.zeros((n_rx, 6), dtype=complex)

    for irx in range(n_rx):
        ix = ix_arr[irx]
        iz = iz_arr[irx]
        p = profile_node_indices[irx]

        # 이웃 노드 인덱스 (중앙차분)
        ix_m = max(ix - 1, 0)
        ix_p = min(ix + 1, n_nodes_x - 1)
        iz_m = max(iz - 1, 0)
        iz_p = min(iz + 1, n_nodes_z - 1)

        p_xm = iz * n_nodes_x + ix_m
        p_xp = iz * n_nodes_x + ix_p
        p_zm = iz_m * n_nodes_x + ix
        p_zp = iz_p * n_nodes_x + ix

        # 유한차분 미분
        dx_val = x_nodes[ix_p] - x_nodes[ix_m]
        dz_val = z_nodes[iz_p] - z_nodes[iz_m]

        dEy_dx = (Ey_secondary[p_xp] - Ey_secondary[p_xm]) / dx_val
        dEy_dz = (Ey_secondary[p_zp] - Ey_secondary[p_zm]) / dz_val
        dHy_dx = (Hy_secondary[p_xp] - Hy_secondary[p_xm]) / dx_val
        dHy_dz = (Hy_secondary[p_zp] - Hy_secondary[p_zm]) / dz_val

        # 노드 전도도 (인접 요소 평균)
        ex_l = max(ix - 1, 0)
        ex_r = min(ix, resistivity_elem.shape[0] - 1)
        ez_l = max(iz - 1, 0)
        ez_r = min(iz, resistivity_elem.shape[1] - 1)
        sig_node = np.mean(sig_elem[ex_l:ex_r+1, ez_l:ez_r+1])

        delta_sig = sig_node - sig_bg
        k2 = mu * epsilon * omega**2 - 1j * mu * sig_node * omega
        ke2 = ky**2 - k2
        ka2 = complex(0, -omega * mu) * delta_sig
        sig_iweps = sig_node + 1j * omega * epsilon
        iwmu = 1j * omega * mu
        iky_val = 1j * ky

        Ex_p = E_primary[0, p]
        Ez_p = E_primary[2, p]

        Hx = (sig_iweps * dEy_dz - iky_val * dHy_dx + iky_val * delta_sig * Ez_p) / ke2
        Hz = (-sig_iweps * dEy_dx - iky_val * dHy_dz - iky_val * delta_sig * Ex_p) / ke2
        Ex = (-iky_val * dEy_dx - iwmu * dHy_dz + ka2 * Ex_p) / ke2
        Ez = (-iky_val * dEy_dz + iwmu * dHy_dx + ka2 * Ez_p) / ke2

        result[irx, 0] = Ex
        result[irx, 1] = Ey_secondary[p]
        result[irx, 2] = Ez
        result[irx, 3] = Hx
        result[irx, 4] = Hy_secondary[p]
        result[irx, 5] = Hz

    return result


def _element_to_node_avg(
    elem_values: np.ndarray,    # (n_ex, n_ez)
    n_nodes_x: int,
    n_nodes_z: int,
) -> np.ndarray:
    """
    요소 중심값 → 노드 값 보간 (인접 요소 평균)

    반환 shape: (n_nodes_z, n_nodes_x)
    """
    n_ex, n_ez = elem_values.shape
    node_val = np.zeros((n_nodes_z, n_nodes_x), dtype=elem_values.dtype)
    count = np.zeros((n_nodes_z, n_nodes_x), dtype=int)

    for ex in range(n_ex):
        for ez in range(n_ez):
            val = elem_values[ex, ez]
            for dix, diz in [(0, 0), (1, 0), (1, 1), (0, 1)]:
                node_val[ez + diz, ex + dix] += val
                count[ez + diz, ex + dix] += 1

    count = np.where(count == 0, 1, count)
    return node_val / count
