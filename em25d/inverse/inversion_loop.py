"""
역산 반복 루프 제어

Fortran 대응: Fem25Dinv.f90 — FEM_MAIN_INVERSION

역산 절차:
  1. 초기 비저항 모델 m₀ (균질 배경)
  2. 반복 loop (iter = 1 .. max_iter):
     a. 순방향 모델링 → 예측 데이터 d_pred
     b. 야코비안 계산 J[n_data, n_para]
     c. 잔차 r = d_obs - d_pred
     d. ACB + IRLS 역산 스텝 → Δm
     e. 라인 서치 → m_{iter+1}
     f. RMS 수렴 판정
  3. 결과 저장

재시작(Restart):
  이전 역산 결과 blck_res.dat 에서 모델 읽어 계속

로그 저장:
  inversion_log/run_YYMMDD_HHMMSS/
    model_iter_NNN.dat
    misfit_log.csv
    jacobian_iter_NNN.npz  (선택)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

from ..constants import NormType
from ..mesh.grid import Grid
from ..mesh.profile import ProfileNodes
from ..model.resistivity import ResistivityModel
from ..survey.survey import Survey
from ..forward.forward_loop import ForwardModeling, ForwardConfig
from ..io.legacy_io import write_model_iteration, write_jacobian_npz
from .acb import inversion_step, InversionStepResult, to_inversion_param
from .measures import compute_norm


# ── 역산 설정 ────────────────────────────────────────────────────────────────

@dataclass
class InversionConfig:
    """역산 제어 파라미터"""
    max_iterations: int = 10
    iteration_type: str = "jumping"    # "jumping" | "creeping"

    # IRLS
    norm_data: NormType = NormType.L2
    norm_model: NormType = NormType.L2
    irls_start: int = 1

    # 비저항 범위
    rho_min: float = 0.1
    rho_max: float = 1e5

    # 사용 성분 (0: 미사용, 1: 허수부, 2: 실수+허수)
    use_Ex: int = 0
    use_Ey: int = 0
    use_Ez: int = 0
    use_Hx: int = 0
    use_Hy: int = 0
    use_Hz: int = 1

    # ACB / Occam
    use_acb: bool = True
    use_occam: bool = True
    smoothness_v: float = 0.5
    smoothness_h: float = 0.5

    # 데이터 가중치
    gamma: float = 1.0
    whitening_factor: float = 1e-9

    # 라그랑지안 스케일
    lambda_scale: float = -5.0e-15

    # 수렴 판정
    target_rms: float = 1.0
    min_delta_rms: float = 1e-4

    # 로그
    log_dir: str | None = "./inversion_log"
    save_jacobian: bool = False

    # 재시작
    restart: bool = False
    restart_model_path: Optional[str] = None


# ── 데이터 선택 및 정규화 ────────────────────────────────────────────────────

def select_data_components(
    synthetic: np.ndarray,    # (n_freq, n_tx, n_rx, 6) 복소
    observed: np.ndarray,     # (n_freq, n_tx, n_rx, 6) 복소
    config: InversionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    설정에 따라 사용할 데이터 성분 선택

    Fortran 대응: iuse(1..6) 플래그에 따른 데이터 선택

    Returns
    -------
    d_pred     : (n_data,) 예측 데이터 (실수화)
    d_obs      : (n_data,) 관측 데이터
    norm_factor: (n_data,) 정규화 인자
    """
    use_flags = [
        config.use_Ex, config.use_Ey, config.use_Ez,
        config.use_Hx, config.use_Hy, config.use_Hz,
    ]
    n_freq, n_tx, n_rx, _ = synthetic.shape
    d_pred_list = []
    d_obs_list  = []

    for icomp, use in enumerate(use_flags):
        if use == 0:
            continue
        syn_c = synthetic[:, :, :, icomp]   # (n_freq, n_tx, n_rx)
        obs_c = observed[:, :, :, icomp]

        if use == 1:  # 허수부만
            d_pred_list.append(syn_c.imag.ravel())
            d_obs_list.append(obs_c.imag.ravel())
        elif use == 2:  # 실수 + 허수
            d_pred_list.append(syn_c.real.ravel())
            d_obs_list.append(obs_c.real.ravel())
            d_pred_list.append(syn_c.imag.ravel())
            d_obs_list.append(obs_c.imag.ravel())

    d_pred = np.concatenate(d_pred_list) if d_pred_list else np.array([])
    d_obs  = np.concatenate(d_obs_list)  if d_obs_list  else np.array([])

    # 정규화 인자 (관측 데이터 RMS + 잡음 플로어)
    amp_obs = np.abs(d_obs)
    noise_floor = config.whitening_factor * np.max(amp_obs)
    norm_factor = np.maximum(amp_obs * config.gamma, noise_floor)
    norm_factor = np.where(norm_factor > 0, norm_factor, 1.0)

    return d_pred, d_obs, norm_factor


def compute_residual(
    d_pred: np.ndarray,
    d_obs: np.ndarray,
    norm_factor: np.ndarray,
) -> np.ndarray:
    """정규화 잔차 (d_obs - d_pred) / norm_factor"""
    return (d_obs - d_pred) / norm_factor


def compute_rms(residual: np.ndarray) -> float:
    """RMS 오차"""
    return float(np.sqrt(np.mean(residual**2)))


# ── 역산 로그 ────────────────────────────────────────────────────────────────

class InversionLogger:
    """역산 진행 상황 로그"""

    def __init__(self, log_dir: str | Path | None, save_jacobian: bool = False):
        self.save_jacobian = save_jacobian
        if log_dir is None:
            self.log_dir  = None
            self.run_dir  = None
            self._csv_path = None
            return
        self.log_dir = Path(log_dir)
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        self.run_dir = self.log_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # CSV 헤더
        csv = self.run_dir / "misfit_log.csv"
        csv.write_text("iteration,rms_data,rms_model,step_size\n")
        self._csv_path = csv

    def log_iteration(
        self,
        iteration: int,
        block_rho: np.ndarray,
        step_result: InversionStepResult,
    ) -> None:
        if self.run_dir is not None:
            write_model_iteration(block_rho, iteration, step_result.rms_data, self.run_dir)
            with open(self._csv_path, "a") as f:
                f.write(f"{iteration},{step_result.rms_data:.6E},"
                        f"{step_result.rms_model:.6E},{step_result.step_size:.6E}\n")
        print(f"[Inversion] iter {iteration:3d} | "
              f"RMS_data={step_result.rms_data:.4f} | "
              f"RMS_model={step_result.rms_model:.4f} | "
              f"step={step_result.step_size:.3f}")

    def log_jacobian(
        self, J: np.ndarray, iteration: int, frequency: float = None
    ) -> None:
        if self.save_jacobian and self.run_dir is not None:
            path = self.run_dir / f"jacobian_iter_{iteration:03d}.npz"
            write_jacobian_npz(J, path, iteration, frequency)


# ── 역산 실행기 ───────────────────────────────────────────────────────────────

@dataclass
class InversionResult:
    """역산 최종 결과"""
    block_rho: np.ndarray         # (n_para,) 최종 비저항
    rms_history: list             # 반복별 RMS 오차
    iterations: int               # 수행한 반복 수
    converged: bool
    log_dir: str | None


class InversionModeling:
    """
    EM 2.5D 역산 실행기

    사용 예시
    ---------
    >>> inv = InversionModeling(grid, model, survey, profile, observed, config)
    >>> result = inv.run()
    """

    def __init__(
        self,
        grid: Grid,
        model: ResistivityModel,
        survey: Survey,
        profile: ProfileNodes,
        observed: np.ndarray,       # (n_freq, n_tx, n_rx, 6) 복소 관측 데이터
        inv_config: InversionConfig = InversionConfig(),
        fwd_config: ForwardConfig = ForwardConfig(),
        jacobian_fn: Optional[Callable] = None,  # 외부 야코비안 함수 (테스트용)
    ):
        self.grid     = grid
        self.model    = model
        self.survey   = survey
        self.profile  = profile
        self.observed = observed
        self.inv_cfg  = inv_config
        self.fwd_cfg  = fwd_config
        self._jacobian_fn = jacobian_fn

        self._logger = InversionLogger(inv_config.log_dir, inv_config.save_jacobian)
        self._fwd = ForwardModeling(grid, model, survey, profile, fwd_config)

    def run(self) -> InversionResult:
        """
        역산 반복 루프 실행

        Returns
        -------
        InversionResult
        """
        cfg = self.inv_cfg
        block_rho = self.model.block_resistivity.copy()

        # 재시작 처리
        if cfg.restart and cfg.restart_model_path:
            from ..io.legacy_io import read_block_resistivity
            block_rho = read_block_resistivity(cfg.restart_model_path)
            print(f"[Inversion] 재시작: {cfg.restart_model_path}")

        rms_history = []
        converged = False
        prev_rms = np.inf

        n_x = self.model.block_partition.config.n_blocks_x
        n_z = self.model.block_partition.config.n_blocks_z

        for iteration in range(1, cfg.max_iterations + 1):
            print(f"\n[Inversion] ========== 반복 {iteration}/{cfg.max_iterations} ==========")

            # 1) 현재 모델로 순방향 모델링
            self.model.update_block_resistivity(block_rho)
            self._fwd.model = self.model
            synthetic = self._fwd.run()

            # 2) 데이터 선택 및 잔차
            d_pred, d_obs, norm_factor = select_data_components(
                synthetic, self.observed, cfg)

            if d_pred.size == 0:
                print("[Inversion] 경고: 사용 가능한 데이터 성분이 없습니다.")
                break

            residual = compute_residual(d_pred, d_obs, norm_factor)
            rms = compute_rms(residual)
            rms_history.append(rms)
            print(f"[Inversion] 순방향 완료. RMS = {rms:.4f}")

            # 3) 야코비안 계산
            J = self._compute_jacobian(
                synthetic, d_pred, norm_factor, block_rho, iteration)

            if J is None or J.shape[0] == 0:
                print("[Inversion] 경고: 야코비안 계산 실패.")
                break

            # 4) 역산 스텝 (ACB + IRLS)
            step_result = inversion_step(
                J=J,
                residual=residual,
                block_rho=block_rho,
                n_x=n_x,
                n_z=n_z,
                iteration=iteration,
                norm_data=cfg.norm_data,
                norm_model=cfg.norm_model,
                irls_start=cfg.irls_start,
                smoothness_v=cfg.smoothness_v,
                smoothness_h=cfg.smoothness_h,
                use_acb=cfg.use_acb,
                use_occam=cfg.use_occam,
                lambda_scale=cfg.lambda_scale,
                rho_min=cfg.rho_min,
                rho_max=cfg.rho_max,
            )

            # 5) 로그 저장
            self._logger.log_iteration(iteration, step_result.new_rho, step_result)
            if cfg.save_jacobian:
                self._logger.log_jacobian(J, iteration)

            # 6) 모델 업데이트
            if cfg.iteration_type == "jumping":
                block_rho = step_result.new_rho
            else:  # creeping
                block_rho = 0.5 * block_rho + 0.5 * step_result.new_rho

            # 7) 수렴 판정
            delta_rms = abs(prev_rms - step_result.rms_data)
            if step_result.rms_data <= cfg.target_rms:
                print(f"[Inversion] 수렴 달성: RMS={step_result.rms_data:.4f} ≤ {cfg.target_rms}")
                converged = True
                break
            if delta_rms < cfg.min_delta_rms and iteration > 2:
                print(f"[Inversion] RMS 변화 미미 (ΔRMS={delta_rms:.2e}). 조기 종료.")
                break

            prev_rms = step_result.rms_data

        # 최종 모델 저장
        if self._logger.run_dir is not None:
            from ..io.legacy_io import write_block_resistivity
            final_path = Path(self._logger.run_dir) / "final_model.dat"
            write_block_resistivity(block_rho, final_path)
            print(f"\n[Inversion] 완료. 최종 모델: {final_path}")

        return InversionResult(
            block_rho=block_rho,
            rms_history=rms_history,
            iterations=len(rms_history),
            converged=converged,
            log_dir=str(self._logger.run_dir) if self._logger.run_dir else None,
        )

    def _compute_jacobian(
        self,
        synthetic: np.ndarray,
        d_pred: np.ndarray,
        norm_factor: np.ndarray,
        block_rho: np.ndarray,
        iteration: int,
    ) -> Optional[np.ndarray]:
        """
        야코비안 계산 (외부 함수 또는 수치 미분 fallback)

        수치 미분 야코비안 (검증용):
          J[i, b] ≈ (d_pred(ρ + Δρ) - d_pred(ρ)) / Δlog(ρ)
        """
        if self._jacobian_fn is not None:
            return self._jacobian_fn(
                synthetic, block_rho, norm_factor, self.model, self.grid,
                self.survey, self.profile, self.fwd_cfg)

        # 수치 미분 야코비안 (느리지만 항상 동작)
        print("[Inversion] 수치 미분 야코비안 계산 중... (검증용 — 느림)")
        n_data  = len(d_pred)
        n_para  = len(block_rho)
        J = np.zeros((n_data, n_para), dtype=float)

        delta_log = 0.01  # log(ρ) 에서 1% 섭동

        for ib in range(n_para):
            rho_pert = block_rho.copy()
            rho_pert[ib] *= np.exp(delta_log)

            model_pert = self.model.copy()
            model_pert.update_block_resistivity(rho_pert)
            fwd_pert = ForwardModeling(
                self.grid, model_pert, self.survey, self.profile, self.fwd_cfg)
            syn_pert = fwd_pert.run()

            d_pert, _, _ = select_data_components(syn_pert, self.observed, self.inv_cfg)
            J[:, ib] = (d_pert - d_pred) / delta_log / norm_factor

        return J


def run_inversion(
    grid: Grid,
    model: ResistivityModel,
    survey: Survey,
    profile: ProfileNodes,
    observed: np.ndarray,
    inv_config: InversionConfig = InversionConfig(),
    fwd_config: ForwardConfig = ForwardConfig(),
) -> InversionResult:
    """
    역산 간편 실행 함수

    Returns
    -------
    InversionResult
    """
    inv = InversionModeling(grid, model, survey, profile, observed, inv_config, fwd_config)
    return inv.run()
