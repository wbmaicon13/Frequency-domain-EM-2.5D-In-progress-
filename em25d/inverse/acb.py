"""
ACB (Active Constraint Balancing) + 정규방정식 풀이

Fortran 대응: Fem25Dacb.f90 — DoInversionACB

ACB 알고리즘:
  1. 야코비안 J 에서 헤시안 H = JᵀJ 계산
  2. 모델 해상도 행렬 R_res = (H + λI)⁻¹ H
  3. Spread function (분산도) spr_i = Σ_j d²(i,j) · (R_res[i,j]/rowsum)²
  4. ACB 라그랑지안: λ_i ∝ spr_i^slope
  5. λ_i 로 스케일된 거칠기 행렬 R̃ 구성
  6. 정규방정식: (JᵀWᵈJ + R̃ᵀR̃) Δm = Jᵀ Wᵈ r

불평등 제약 (Inequality Constraints):
  log-transform 파라미터화로 비저항 범위 [ρ_min, ρ_max] 를 자동 보장:
    m_i = log((σ_i - σ_min) / (σ_max - σ_i))  (IK_LOG)

수렴 판정:
  RMS 오차 = sqrt(φ_d / n_data) ≤ threshold
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from dataclasses import dataclass
from typing import Optional

from ..constants import MU_0
from .measures import compute_irls_weights, compute_norm
from .regularization import (
    build_roughening_matrix,
    build_identity_regularization,
    scale_roughening_matrix,
    compute_model_structure,
)
from ..constants import NormType


# ── 파라미터화 변환 ───────────────────────────────────────────────────────────

def to_inversion_param(
    block_rho: np.ndarray,
    rho_min: float,
    rho_max: float,
    log_transform: bool = True,
) -> np.ndarray:
    """
    비저항 → 역산 파라미터 (불평등 제약 변환)

    Fortran 대응: IK_LOG 분기
      m_i = log((σ_i - σ_min) / (σ_max - σ_i))
    """
    sigma = 1.0 / block_rho
    sigma_min = 1.0 / rho_max
    sigma_max = 1.0 / rho_min

    if log_transform:
        log_sigma = np.log(sigma)
        log_min   = np.log(sigma_min)
        log_max   = np.log(sigma_max)
        ratio = (np.exp(log_sigma) - np.exp(log_min)) / (np.exp(log_max) - np.exp(log_sigma))
        ratio = np.clip(ratio, 1e-30, 1e30)
        return np.log(ratio)
    else:
        sigma_c = np.clip(sigma, sigma_min + 1e-15, sigma_max - 1e-15)
        return np.log((sigma_c - sigma_min) / (sigma_max - sigma_c))


def from_inversion_param(
    m: np.ndarray,
    rho_min: float,
    rho_max: float,
    log_transform: bool = True,
) -> np.ndarray:
    """
    역산 파라미터 → 비저항

    Fortran 대응: 역변환
    """
    sigma_min = 1.0 / rho_max
    sigma_max = 1.0 / rho_min
    exp_m = np.exp(m)

    if log_transform:
        log_min = np.log(sigma_min)
        log_max = np.log(sigma_max)
        log_sigma = (log_min + log_max * exp_m) / (1.0 + exp_m)
        sigma = np.exp(log_sigma)
    else:
        sigma = (sigma_min + sigma_max * exp_m) / (1.0 + exp_m)

    return np.clip(1.0 / sigma, rho_min, rho_max)


# ── 해상도 행렬 / ACB 라그랑지안 ────────────────────────────────────────────

def compute_acb_lagrangian(
    H: np.ndarray,          # (n_para, n_para) 헤시안 JᵀJ
    n_x: int,
    n_z: int,
    damping_frac: float = 0.001,
    lambda_min: float = 1e-2,
    lambda_max: float = 1.0,
) -> np.ndarray:
    """
    ACB 라그랑지안 승수 벡터 λ_ACB : (n_para,)

    Fortran 대응: DoInversionACB 내 spr, rLagrangian_ACB 계산

    알고리즘:
      1. H_damp = H + damping * max(diag(H)) * I
      2. H_inv = H_damp⁻¹
      3. R_res = H_inv · H  (모델 해상도 행렬)
      4. spr_i = Σ_j d²(i,j) * (R_res[i,j] * rowsum_inv)²
                 + (1 - R_res[i,i] * rowsum_inv)²
      5. λ_i = 10^(slope * (log10(spr_i) - log10(spr_min)) + log10(λ_min))

    Parameters
    ----------
    H           : (n_para, n_para) 헤시안
    n_x, n_z    : 블록 격자 크기
    damping_frac: 대각 감쇠 비율
    lambda_min  : ACB λ 최솟값
    lambda_max  : ACB λ 최댓값

    Returns
    -------
    lagrangian : (n_para,) ACB 라그랑지안 승수
    """
    n_para = H.shape[0]

    # 초기 감쇠
    damping = np.max(np.abs(np.diag(H))) * damping_frac
    H_damp = H.copy()
    H_damp[np.arange(n_para), np.arange(n_para)] += damping

    # 역행렬
    try:
        H_inv = linalg.inv(H_damp)
    except linalg.LinAlgError:
        H_inv = np.eye(n_para) / damping

    # 해상도 행렬
    R_res = H_inv @ H   # (n_para, n_para)

    # Spread function
    spr = np.zeros(n_para)
    for i in range(n_para):
        ji = i // n_z   # x 인덱스
        ii = i % n_z    # z 인덱스
        rowsum = np.sum(np.abs(R_res[i, :])) + 1e-30
        rowsum_inv = 1.0 / rowsum
        val = 0.0
        for j in range(n_para):
            jj = j // n_z
            ij = j % n_z
            dist2 = (jj - ji)**2 + (ij - ii)**2
            val += dist2 * (R_res[i, j] * rowsum_inv * (1.0 - 0.0))**2  # (1 - RoughenMatrix)
        val += (1.0 - R_res[i, i] * rowsum_inv)**2
        spr[i] = val

    # ACB 라그랑지안
    spr_max = max(np.max(spr), 1e-30)
    spr_min = max(np.min(spr), 1e-30)

    if spr_max > spr_min:
        slope = (np.log10(lambda_max) - np.log10(lambda_min)) / \
                (np.log10(spr_max)   - np.log10(spr_min))
    else:
        slope = 0.0

    lagrangian = 10.0 ** (
        slope * (np.log10(np.maximum(spr, 1e-30)) - np.log10(spr_min))
        + np.log10(lambda_min)
    )
    return lagrangian


# ── 정규방정식 구성 및 풀이 ───────────────────────────────────────────────────

@dataclass
class NormalEquationResult:
    """정규방정식 풀이 결과"""
    delta_m: np.ndarray         # (n_para,) 모델 업데이트 벡터
    rms_data: float             # 데이터 RMS 오차
    rms_model: float            # 모델 정규화 항
    lagrangian: np.ndarray      # (n_para,) ACB λ_ACB
    resolution: np.ndarray      # (n_para,) 대각 해상도


def solve_normal_equations(
    J: np.ndarray,               # (n_data, n_para)
    residual: np.ndarray,        # (n_data,) 데이터 잔차
    block_rho: np.ndarray,       # (n_para,)
    R: np.ndarray,               # (n_para, n_para) 거칠기 행렬
    irls_data_weights: np.ndarray,   # (n_data,) IRLS 데이터 가중치
    irls_model_weights: np.ndarray,  # (n_para,) IRLS 모델 가중치
    lagrangian: np.ndarray,      # (n_para,) ACB λ
    lambda_scale: float,         # 전체 λ 스케일
    iteration: int,
    irls_start: int = 1,
    damping_frac: float = 0.005,
    use_acb: bool = True,
) -> NormalEquationResult:
    """
    정규방정식 풀이: Δm = (JᵀWᵈJ + R̃ᵀW_m R̃)⁻¹ Jᵀ Wᵈ r

    Fortran 대응: DoInversionACB 내 DGEMM/DGEMV/DGETRF/DGETRI 블록

    Parameters
    ----------
    J               : (n_data, n_para) 야코비안
    residual        : (n_data,) 데이터 잔차 (d_obs - d_pred)
    block_rho       : (n_para,) 현재 비저항 모델
    R               : (n_para, n_para) 거칠기 행렬
    irls_data_weights : (n_data,) sqrt 적용 전 가중치
    irls_model_weights: (n_para,) sqrt 적용 전 가중치
    lagrangian      : (n_para,) ACB λ_ACB
    lambda_scale    : 전체 라그랑지안 스케일 인자
    iteration       : 현재 반복 번호
    irls_start      : IRLS 시작 반복 번호

    Returns
    -------
    NormalEquationResult
    """
    n_data, n_para = J.shape

    # IRLS 가중 야코비안 / 잔차
    sqrt_wd = np.sqrt(np.abs(irls_data_weights))
    J_w = sqrt_wd[:, np.newaxis] * J
    r_w = sqrt_wd * residual

    # 헤시안 H = JᵀWᵈJ
    H = J_w.T @ J_w              # (n_para, n_para)
    g = J_w.T @ r_w              # (n_para,) 그래디언트

    # 모델 RMS (데이터)
    rms_data = float(np.sqrt(np.dot(r_w, r_w) / max(n_data, 1)))

    # ACB 라그랑지안 스케일
    if lambda_scale > 0:
        lam = lambda_scale
    else:
        lam = abs(lambda_scale) / max(iteration, 1)

    lagrangian_scaled = lam * lagrangian

    # 거칠기 행렬 스케일
    R_scaled = scale_roughening_matrix(R, lagrangian_scaled, irls_model_weights)

    # R̃ᵀ R̃ 항 추가
    RTR = R_scaled.T @ R_scaled

    # 모델 정규화 항 목적함수
    model_struct = R @ np.log(block_rho)
    rms_model = float(np.sqrt(np.dot(model_struct, model_struct) / max(n_para, 1)))

    # 정규화된 헤시안
    H_reg = H + RTR

    # 안정화 감쇠
    damping = np.max(np.abs(np.diag(H_reg))) * damping_frac
    H_reg[np.arange(n_para), np.arange(n_para)] += damping

    # 풀이
    try:
        delta_m = linalg.solve(H_reg, g, assume_a="gen")
    except linalg.LinAlgError:
        delta_m = np.zeros(n_para)

    # 대각 해상도
    try:
        H_inv = linalg.inv(H_reg)
        resolution_diag = np.diag(H_inv @ H)
    except linalg.LinAlgError:
        resolution_diag = np.ones(n_para)

    return NormalEquationResult(
        delta_m=delta_m,
        rms_data=rms_data,
        rms_model=rms_model,
        lagrangian=lagrangian_scaled,
        resolution=resolution_diag,
    )


# ── 라인 서치 ────────────────────────────────────────────────────────────────

def line_search_step_size(
    delta_m: np.ndarray,       # (n_para,) 업데이트 방향
    block_rho: np.ndarray,     # (n_para,) 현재 비저항
    rho_min: float,
    rho_max: float,
    step_scale: float = 1.0,
    max_step: float = 2.0,
    log_transform: bool = True,
) -> tuple[np.ndarray, float]:
    """
    라인 서치: 비저항 범위 제약을 만족하는 최대 스텝 크기 결정

    Fortran 대응: DoInversionACB 내 적용 후 범위 클리핑

    Returns
    -------
    new_rho    : (n_para,) 업데이트된 비저항
    actual_step: 적용된 스텝 크기
    """
    step = min(step_scale, max_step)

    if log_transform:
        log_rho = np.log(block_rho)
        new_log_rho = log_rho + step * delta_m
        new_rho = np.exp(new_log_rho)
    else:
        new_rho = block_rho + step * delta_m

    # 범위 클리핑
    new_rho = np.clip(new_rho, rho_min, rho_max)
    return new_rho, step


# ── 전체 역산 스텝 (단일 반복) ───────────────────────────────────────────────

@dataclass
class InversionStepResult:
    """단일 역산 반복 결과"""
    new_rho: np.ndarray       # (n_para,) 업데이트된 비저항
    rms_data: float
    rms_model: float
    lagrangian: np.ndarray    # (n_para,)
    delta_m: np.ndarray       # (n_para,)
    step_size: float


def inversion_step(
    J: np.ndarray,
    residual: np.ndarray,
    block_rho: np.ndarray,
    n_x: int,
    n_z: int,
    iteration: int,
    norm_data: NormType = NormType.L2,
    norm_model: NormType = NormType.L2,
    irls_start: int = 1,
    smoothness_v: float = 0.5,
    smoothness_h: float = 0.5,
    use_acb: bool = True,
    use_occam: bool = True,
    lambda_scale: float = 1.0,
    rho_min: float = 0.1,
    rho_max: float = 1e5,
    log_transform: bool = True,
) -> InversionStepResult:
    """
    ACB + IRLS 단일 역산 반복

    Fortran 대응: DoInversionACB(iter)

    Parameters
    ----------
    J         : (n_data, n_para) 야코비안
    residual  : (n_data,) 잔차
    block_rho : (n_para,) 현재 비저항
    n_x, n_z  : 블록 격자 크기
    iteration : 반복 번호 (1-based)
    """
    n_data, n_para = J.shape

    # IRLS 데이터 가중치
    irls_wd = compute_irls_weights(-residual, norm_data)

    # 거칠기 행렬
    if use_occam:
        R = build_roughening_matrix(n_x, n_z, smoothness_v, smoothness_h, use_acb)
    else:
        R = build_identity_regularization(n_para)

    # 모델 구조 벡터 (IRLS 모델 가중치)
    model_struct = compute_model_structure(block_rho, R)

    if iteration < irls_start:
        irls_wm = compute_irls_weights(model_struct, NormType.L2)
    else:
        irls_wm = compute_irls_weights(model_struct, norm_model)

    # ACB 라그랑지안
    sqrt_wd = np.sqrt(np.abs(irls_wd))
    J_w = sqrt_wd[:, np.newaxis] * J
    H = J_w.T @ J_w

    if use_acb:
        lagrangian = compute_acb_lagrangian(H, n_x, n_z)
    else:
        lagrangian = np.ones(n_para)

    # 정규방정식 풀이
    result = solve_normal_equations(
        J=J,
        residual=residual,
        block_rho=block_rho,
        R=R,
        irls_data_weights=irls_wd,
        irls_model_weights=irls_wm,
        lagrangian=lagrangian,
        lambda_scale=lambda_scale,
        iteration=iteration,
        irls_start=irls_start,
    )

    # 모델 업데이트 (라인 서치)
    new_rho, step_size = line_search_step_size(
        result.delta_m, block_rho, rho_min, rho_max,
        log_transform=log_transform,
    )

    return InversionStepResult(
        new_rho=new_rho,
        rms_data=result.rms_data,
        rms_model=result.rms_model,
        lagrangian=result.lagrangian,
        delta_m=result.delta_m,
        step_size=step_size,
    )
