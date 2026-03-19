"""
순방향 모델링 루프 — MPI + GPU 통합

Fortran 대응:
  Fem25Dfwd.f90 — Forward_Loop + FWD2
  Fem25DPost.f90 — FemPost + FemIntegral + ytran_splint
  Fem25Dsub.f — fourint (스플라인 역 Fourier 변환)

순방향 모델링 절차:
  1. ky 샘플링 (로그간격, Fortran 방식)
  2. [MPI 분배] 각 프로세스에 ky 부분집합 할당
  3. 각 ky에 대해:
     a. 1차장 계산 (층상 배경 Hankel 변환)
     b. FEM 강성행렬 조립 + Robin BC 추가
     c. FEM 풀이 (GPU 또는 CPU) → Ey_s, Hy_s
     d. 보조 장 6성분 계산 (FemIntegral 공식)
     e. 프로파일 노드 추출
  4. [MPI gather] 전체 ky 결과 수집
  5. 역 Fourier 변환 (스플라인 보간 + 2차 적분, even/odd 대칭)
  6. 1차장(공간 영역) 추가

역 Fourier 변환 (Fortran fourint):
  - 입력: F̃(ky) 복소 배열 (n_ky 점)
  - ky[0]=1e-8은 DC 근사, ky[1:]은 로그간격
  - 스플라인: th(j) = F(j+1)·ky(j+1) 로그 공간에서 cubic spline
  - 적분: 균일 ky 간격으로 보간 후 2차 다항식 적분
  - 짝함수(isym=0): F(y=0) = (1/π) ∫ F̃(ky) dky
  - 홀함수(isym=1): F(y=0) = 0  (y=0에서 항상 0)

Jy 소스(EL_source_y) 기준 even/odd:
  Hx, Hz, Ey → even (isym=0) → 비영값
  Hy, Ex, Ez → odd  (isym=1) → y=0에서 0
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.interpolate import CubicSpline

from ..constants import MU_0, EPSILON_0, PI, SourceType
from ..mesh.grid import Grid
from ..mesh.profile import ProfileNodes
from ..model.resistivity import ResistivityModel
from ..survey.survey import Survey
from .primary_field import (
    PrimaryFieldParams,
    compute_wavenumber_sampling,
    primary_field_ky_domain,
    primary_field_space_domain,
)
from .fem_assembly import assemble_global_system, assemble_force_vector
from .fem_solver import (
    solve_fem_system, extract_secondary_fields,
    build_robin_stiffness, factorize_system,
)
from .postprocess import (
    compute_secondary_field_components, extract_profile_fields,
    compute_fields_at_profile,
)


# ── even/odd 대칭 분류 (Fortran FemIntegral + ytran_splint) ────────────────
# Jy 소스(EL_source_y, itype=2) 기준: Fortran 라인 446-451
# even (isym=0): Hx, Hz, Ey  → 비영 Fourier 적분
# odd  (isym=1): Hy, Ex, Ez  → y=0에서 0
_SYMMETRY_JY = {
    "Ex": "odd",   "Ey": "even",  "Ez": "odd",
    "Hx": "even",  "Hy": "odd",   "Hz": "even",
}
# Jx/Jz/Mx/Mz 소스: Fortran 라인 452-458
_SYMMETRY_JX = {
    "Ex": "even",  "Ey": "odd",   "Ez": "even",
    "Hx": "odd",   "Hy": "even",  "Hz": "odd",
}


def _get_symmetry_map(source_type: int) -> dict:
    """소스 타입에 따른 성분별 even/odd 분류"""
    if source_type in (SourceType.Jy, SourceType.Mx, SourceType.Mz):
        return _SYMMETRY_JY
    else:
        return _SYMMETRY_JX


@dataclass
class ForwardConfig:
    """순방향 모델링 설정"""
    n_wavenumbers: int = 20          # 공간주파수(ky) 개수 (Fortran 기본 20)
    use_gpu: bool = False
    solver: str = "direct"           # "direct" | "gmres" | "bicgstab"
    primary_dir: str = "./output_primary"
    output_dir: str = "./output_data"
    save_primary: bool = True
    load_primary: bool = False
    field_type: str = "total"        # "secondary" | "total"
    log_interval: int = 1
    ky_resistivity: Optional[float] = None  # ky 샘플링용 비저항 (Fortran: res0=anomalous_resistivity)


class ForwardModeling:
    """
    2.5D EM 순방향 모델링 실행기 (MPI + GPU 지원)

    사용 예시 (직렬):
      fwd = ForwardModeling(grid, model, survey, profile, config)
      data = fwd.run()

    사용 예시 (MPI):
      from em25d.parallel.mpi_manager import MPIContext
      mpi = MPIContext()
      data = fwd.run(mpi=mpi)  # rank=0만 결과 반환
    """

    def __init__(
        self,
        grid: Grid,
        resistivity_model: ResistivityModel,
        survey: Survey,
        profile: ProfileNodes,
        config: ForwardConfig = ForwardConfig(),
    ):
        self.grid = grid
        self.model = resistivity_model
        self.survey = survey
        self.profile = profile
        self.config = config

        self._primary_params = PrimaryFieldParams(
            background_resistivity=resistivity_model.background_resistivity)

    def _build_layer_resistivity(self) -> np.ndarray:
        """
        배경 layer 비저항 배열 생성

        Fortran 대응: mat_prop_layer 배열
          Fortran (Fem25Dpar.f90 라인 440-442):
            do i = 1, n_elem
              mat_prop_layer(i) = 0.
            enddo
          → 모든 요소의 layer 비저항을 0 (자유공간)으로 강제.
          → delta_sigma = sigma_element - 0 = sigma_element (전체 전도도)
        """
        n_ex = self.model.element_resistivity.shape[0]
        n_ez = self.model.element_resistivity.shape[1]
        return np.zeros((n_ex, n_ez))

    def run(self, mpi=None) -> Optional[np.ndarray]:
        """
        순방향 모델링 실행

        Parameters
        ----------
        mpi : MPIContext or None
            MPI 환경 (None이면 직렬 실행)

        Returns
        -------
        synthetic_data : (n_freq, n_tx, n_rx, 6) 복소 배열 (rank=0 또는 직렬)
                         축 3: [Ex, Ey, Ez, Hx, Hy, Hz]
                         rank>0: None
        """
        cfg = self.config
        survey = self.survey
        grid = self.grid

        n_freq = survey.frequencies.n_frequencies
        n_tx = survey.sources.n_sources
        n_rx = self.profile.n_receivers

        # ky 샘플링
        # Fortran: res0 = anomalous_resistivity (survey.dat에서 읽음)
        ky_rho = cfg.ky_resistivity if cfg.ky_resistivity is not None \
            else self.model.background_resistivity
        wavenumbers_ky = compute_wavenumber_sampling(
            ky_rho,
            float(survey.frequencies.frequencies.min()),
            grid.minimum_cell_size,
            cfg.n_wavenumbers,
        )
        n_ky = len(wavenumbers_ky)

        # MPI ky 분배
        if mpi is not None and mpi.size > 1:
            from ..parallel.mpi_manager import distribute_ky
            local_ky, local_indices = distribute_ky(wavenumbers_ky, mpi)
            mpi.print_root(
                f"[Forward] MPI {mpi.size}프로세스, n_ky={n_ky}, "
                f"ky_min={wavenumbers_ky[1]:.3e}, ky_max={wavenumbers_ky[-1]:.3e}")
        else:
            local_ky = wavenumbers_ky
            local_indices = np.arange(n_ky)
            print(f"[Forward] n_ky={n_ky}, "
                  f"ky_min={wavenumbers_ky[1]:.3e}, ky_max={wavenumbers_ky[-1]:.3e}")

        n_local = len(local_ky)

        # ky 영역 결과 배열: 전체 n_ky 크기 (MPI reduce용)
        E_ky = np.zeros((n_freq, n_tx, n_ky, n_rx, 6), dtype=complex)

        x_nodes = grid.node_x[:, 0]
        z_nodes = grid.node_z[0, :]
        n_nx, n_nz = len(x_nodes), len(z_nodes)
        rx_nodes = self.profile.global_node_indices()

        # ── 주파수 루프 ──────────────────────────────────────────────────
        for ifreq, freq in enumerate(survey.frequencies.frequencies):
            omega = 2 * PI * freq
            if mpi is None or mpi.is_root:
                print(f"[Forward] 주파수 {ifreq+1}/{n_freq}: {freq:.3g} Hz")

            # Fortran: mat_prop_layer=0 (자유공간), primary=자유공간
            # → background_resistivity=0 for postprocess delta_sigma
            bg_rho = 0.0  # 자유공간 배경 (Fortran 동일)
            layer_resistivity = self._build_layer_resistivity()

            # ── ky 루프 (로컬 부분) ──────────────────────────────────────
            for iloc, iky_global in enumerate(local_indices):
                ky = wavenumbers_ky[iky_global]

                if (mpi is None or mpi.is_root) and iloc % cfg.log_interval == 0:
                    print(f"  ky [{iloc+1}/{n_local}] = {ky:.4e}")

                # ── K 행렬 조립 + Robin BC (tx 독립, 1회만) ──────────────
                # E_primary=0으로 K만 구함 (f는 무시)
                E_primary_zero = np.zeros((3, n_nz * n_nx), dtype=complex)
                K_global, _ = assemble_global_system(
                    grid=grid,
                    element_resistivity=self.model.element_resistivity,
                    layer_resistivity=layer_resistivity,
                    E_primary=E_primary_zero,
                    omega=omega,
                    ky=ky,
                )
                K_bc = build_robin_stiffness(
                    K_global, grid,
                    element_resistivity=self.model.element_resistivity,
                    omega=omega, ky=ky)

                # ILU 전처리 1회 계산 (모든 tx에 재사용)
                solve_fn = factorize_system(
                    K_bc, ky=ky, solver=cfg.solver, use_gpu=cfg.use_gpu)

                # ── tx 루프: f 벡터만 재조립 + solve ─────────────────────
                prev_sol = None  # GMRES 초기추정 재사용
                for itx, source in enumerate(survey.sources.sources):
                    # 1) 1차장 계산 (ky 영역, 모든 격자 노드)
                    E_primary_3xz = primary_field_ky_domain(
                        source_x=source.x,
                        source_z=source.z,
                        node_x=x_nodes,
                        node_z=z_nodes,
                        wavenumber_ky=ky,
                        omega=omega,
                        source_type=source.source_type,
                        params=self._primary_params,
                        source_length=source.length,
                        source_strength=source.strength,
                    )
                    E_primary_nodes = (E_primary_3xz
                                       .transpose(0, 2, 1)
                                       .reshape(3, n_nz * n_nx))

                    # 2) f 벡터만 조립 (K는 재사용)
                    f_global = assemble_force_vector(
                        grid=grid,
                        element_resistivity=self.model.element_resistivity,
                        layer_resistivity=layer_resistivity,
                        E_primary=E_primary_nodes,
                        omega=omega,
                        ky=ky,
                    )

                    # 3) 사전 분해된 K로 풀이
                    solution = solve_fn(f_global)
                    prev_sol = solution
                    Ey_s, Hy_s = extract_secondary_fields(solution, grid.n_nodes)

                    # 4) 프로파일 노드에서만 6성분 계산 (최적화)
                    fields_rx = compute_fields_at_profile(
                        Ey_s, Hy_s, E_primary_nodes, grid, omega, ky,
                        self.model.element_resistivity,
                        background_resistivity=bg_rho,
                        profile_node_indices=rx_nodes)

                    # 5) 결과 저장 (n_rx, 6) → E_ky
                    E_ky[ifreq, itx, iky_global, :, :] = fields_rx

        # ── MPI reduce: 모든 프로세스의 E_ky 합산 ────────────────────────
        if mpi is not None and mpi.size > 1:
            E_ky_global = mpi.reduce_sum(E_ky)
            if not mpi.is_root:
                return None
            E_ky = E_ky_global

        # ── 역 Fourier 변환 ──────────────────────────────────────────────
        if mpi is None or mpi.is_root:
            print("[Forward] 역 Fourier 변환 (스플라인 IFT)...")

        # 소스 타입에 따른 even/odd 분류
        source_type = survey.sources.sources[0].source_type
        sym_map = _get_symmetry_map(source_type)

        synthetic = self._inverse_fourier_transform(
            E_ky, wavenumbers_ky, sym_map)

        # ── 공간 영역 1차장 추가 ─────────────────────────────────────────
        if cfg.field_type == "total":
            if mpi is None or mpi.is_root:
                print("[Forward] 공간 영역 1차장 추가...")
            for ifreq, freq in enumerate(survey.frequencies.frequencies):
                omega = 2 * PI * freq
                for itx, source in enumerate(survey.sources.sources):
                    rx_x = self.profile.x_positions
                    rx_z = self.profile.receiver_z

                    # E 1차장 (공간 영역)
                    E_p = primary_field_space_domain(
                        source.x, source.z, rx_x, rx_z, omega,
                        source.source_type, params=self._primary_params,
                        field_type="E",
                        source_length=source.length,
                        source_strength=source.strength)
                    # H 1차장
                    H_p = primary_field_space_domain(
                        source.x, source.z, rx_x, rx_z, omega,
                        source.source_type, params=self._primary_params,
                        field_type="H",
                        source_length=source.length,
                        source_strength=source.strength)

                    for ic, comp in enumerate(["Ex", "Ey", "Ez"]):
                        synthetic[ifreq, itx, :, ic] += E_p[ic]
                    for ic, comp in enumerate(["Hx", "Hy", "Hz"]):
                        synthetic[ifreq, itx, :, ic + 3] += H_p[ic]

        return synthetic

    def _inverse_fourier_transform(
        self,
        E_ky: np.ndarray,          # (n_freq, n_tx, n_ky, n_rx, 6)
        wavenumbers_ky: np.ndarray,  # (n_ky,)
        sym_map: dict,
    ) -> np.ndarray:
        """
        ky 영역 → 실공간 역 Fourier 변환 (y=0)

        Fortran 대응: fourint (Fem25Dsub.f, 라인 2529-2622)

        방법:
          1. th(j) = F(j+1) * ky(j+1)  (j=1..nky-1, 로그 공간)
          2. cubic spline 보간 (로그 ky 좌표)
          3. 균일 ky 간격으로 2차 다항식 적분 (Simpson-like)
          4. even(isym=0): 결과 / π
             odd(isym=1):  0 (y=0에서 항상 0)

        Parameters
        ----------
        sym_map : dict  성분별 "even" / "odd" 분류
        """
        n_freq, n_tx, n_ky, n_rx, n_comp = E_ky.shape
        result = np.zeros((n_freq, n_tx, n_rx, n_comp), dtype=complex)

        comp_names = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

        # 적분 그리드 사전 계산 (모든 신호에 공통)
        aky = wavenumbers_ky
        ky_log = aky[1:]
        log_ky = np.log(ky_log)
        yfund = 2.0 * aky[1]
        ntky2 = 1000 // 2 - 3
        ks_all = np.arange(1, ntky2 + 1, 2)
        n_panels = len(ks_all)
        aa2 = yfund * ks_all
        aa3 = yfund * (ks_all + 1)
        aa1 = np.empty(n_panels)
        aa1[0] = aky[0]
        aa1[1:] = yfund * (ks_all[:-1] + 1)  # aa3 of previous panel
        rl2 = np.log(aa2)
        rl3 = np.log(aa3)

        for ic, comp in enumerate(comp_names):
            if sym_map.get(comp, "even") == "odd":
                continue

            # 배치 처리: (n_freq, n_tx, n_rx)를 1D로 평탄화
            batch = E_ky[:, :, :, :, ic]  # (n_freq, n_tx, n_ky, n_rx)
            batch_flat = batch.reshape(-1, n_ky, n_rx)
            # → (n_freq*n_tx, n_ky, n_rx) → 전치해서 각 신호를 처리
            for ib in range(batch_flat.shape[0]):
                fields_all_rx = batch_flat[ib, :, :]  # (n_ky, n_rx)
                vals = _spline_fourier_integral_batch(
                    fields_all_rx, aky, ky_log, log_ky,
                    aa1, aa2, aa3, rl2, rl3, n_panels)
                ifreq = ib // n_tx
                itx = ib % n_tx
                result[ifreq, itx, :, ic] = vals

        return result


def _spline_fourier_integral_batch(
    fields: np.ndarray,        # (n_ky, n_rx) 복소값
    aky: np.ndarray,           # (n_ky,)
    ky_log: np.ndarray,        # (n_ky-1,) = aky[1:]
    log_ky: np.ndarray,        # (n_ky-1,) = log(aky[1:])
    aa1: np.ndarray,           # (n_panels,)
    aa2: np.ndarray,           # (n_panels,)
    aa3: np.ndarray,           # (n_panels,)
    rl2: np.ndarray,           # (n_panels,) = log(aa2)
    rl3: np.ndarray,           # (n_panels,) = log(aa3)
    n_panels: int,
) -> np.ndarray:
    """
    Fortran fourint의 벡터화 Python 구현 (n_rx 신호 동시 처리)

    Returns: (n_rx,) 복소 배열
    """
    n_ky, n_rx = fields.shape

    # 1) 스케일링: 1/π
    fields_scaled = fields / PI   # (n_ky, n_rx)

    # 2) 스플라인 입력: th(j) = F(j+1) * ky(j+1)
    th = fields_scaled[1:, :] * ky_log[:, None]  # (n_ky-1, n_rx)

    # 3) cubic spline (CubicSpline은 다차원 y 지원)
    try:
        cs_real = CubicSpline(log_ky, th.real, bc_type='natural')
        cs_imag = CubicSpline(log_ky, th.imag, bc_type='natural')
    except Exception:
        # fallback: 사다리꼴
        result = np.zeros(n_rx, dtype=complex)
        for i in range(n_ky - 1):
            dky = aky[i + 1] - aky[i]
            result += 0.5 * (fields[i] + fields[i + 1]) * dky
        return result / PI

    # 4) 벡터화 적분: 모든 패널을 한꺼번에 계산

    # 스플라인 평가 (모든 점 한꺼번에)
    def _eval_batch(rl_arr, aa_arr):
        """rl_arr: (n_panels,), aa_arr: (n_panels,) → (n_panels, n_rx)"""
        in_range = (rl_arr >= log_ky[0]) & (rl_arr <= log_ky[-1])
        ff = np.zeros((len(rl_arr), n_rx), dtype=complex)
        if in_range.any():
            idx = np.where(in_range)[0]
            pts = rl_arr[idx]
            ff_r = cs_real(pts)    # (n_in, n_rx)
            ff_i = cs_imag(pts)    # (n_in, n_rx)
            ff[idx] = ff_r + 1j * ff_i
        if (~in_range).any():
            idx = np.where(~in_range)[0]
            pts = rl_arr[idx]
            for j in idx:
                ff[j].real = np.interp(rl_arr[j], log_ky, th[:, 0].real)
                ff[j].imag = np.interp(rl_arr[j], log_ky, th[:, 0].imag)
                if n_rx > 1:
                    for irx in range(1, n_rx):
                        ff[j, irx] = complex(
                            np.interp(rl_arr[j], log_ky, th[:, irx].real),
                            np.interp(rl_arr[j], log_ky, th[:, irx].imag))
        return ff / aa_arr[:, None]  # (n_panels, n_rx)

    f2 = _eval_batch(rl2, aa2)   # (n_panels, n_rx)
    f3 = _eval_batch(rl3, aa3)   # (n_panels, n_rx)

    # f1: f1[0] = fields_scaled[0], f1[k] = f3[k-1]
    f1 = np.empty((n_panels, n_rx), dtype=complex)
    f1[0] = fields_scaled[0]     # DC 점, (n_rx,)
    f1[1:] = f3[:-1]

    # Lagrange 계수 (스칼라, 모든 rx에 공통)
    e1 = aa2 * aa3 * (aa3 - aa2)         # (n_panels,)
    e2 = aa3 * aa1 * (aa1 - aa3)
    e3 = aa1 * aa2 * (aa2 - aa1)
    dt = e1 + e2 + e3

    valid = np.abs(dt) >= 1e-30
    e1 = e1[valid, None]   # (n_valid, 1)
    e2 = e2[valid, None]
    e3 = e3[valid, None]
    dt_v = dt[valid, None]
    f1_v = f1[valid]       # (n_valid, n_rx)
    f2_v = f2[valid]
    f3_v = f3[valid]
    aa1_v = aa1[valid, None]
    aa2_v = aa2[valid, None]
    aa3_v = aa3[valid, None]

    a = (e1 * f1_v + e2 * f2_v + e3 * f3_v) / dt_v
    b = ((aa2_v**2 - aa3_v**2) * f1_v
         + (aa3_v**2 - aa1_v**2) * f2_v
         + (aa1_v**2 - aa2_v**2) * f3_v) / dt_v
    c = ((aa3_v - aa2_v) * f1_v
         + (aa1_v - aa3_v) * f2_v
         + (aa2_v - aa1_v) * f3_v) / dt_v

    sy = (a * (aa3_v - aa1_v)
          + b * (aa3_v**2 - aa1_v**2) / 2.0
          + c * (aa3_v**3 - aa1_v**3) / 3.0)

    return sy.sum(axis=0)   # (n_rx,)


def run_forward(
    grid: Grid,
    model: ResistivityModel,
    survey: Survey,
    profile: ProfileNodes,
    config: ForwardConfig = ForwardConfig(),
    mpi=None,
) -> Optional[np.ndarray]:
    """순방향 모델링 간편 실행 함수"""
    fwd = ForwardModeling(grid, model, survey, profile, config)
    return fwd.run(mpi=mpi)
