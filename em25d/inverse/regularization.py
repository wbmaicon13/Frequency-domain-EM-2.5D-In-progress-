"""
정규화 행렬 (평활화 / 거칠기 행렬)

Fortran 대응: Fem25Dinv.f90 — Roughening_Occam
              Fem25Dacb.f90 — RoughenMatrix 설정 로직

블록 인덱싱 규칙 (Fortran 1-based → Python 0-based):
  블록 k(0-based) = j * n_z + i
  여기서 j = x-열 인덱스 (0..n_x-1), i = z-행 인덱스 (0..n_z-1)
  (Fortran: k = (j-1)*nz_blck + i, 1-based)

거칠기 행렬 R 의 구조:
  - 대각 원소: 1
  - 이웃 블록(위/아래/좌/우): 음수 가중치
  - 내부 블록 (이웃 4개): Rk,k±1 = -1/4, Rk,k±nz = -1/4
  - 경계 블록 (이웃 3개): -1/3
  - 모서리 블록 (이웃 2개): -1/2

ACB 미사용 시 (non-ACB):
  수직 가중치 Sm_V, 수평 가중치 Sm_H 를 곱해 이방성 조절
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


# ── 거칠기(Roughening) 행렬 ─────────────────────────────────────────────────

def build_roughening_matrix(
    n_x: int,
    n_z: int,
    smoothness_v: float = 0.5,
    smoothness_h: float = 0.5,
    use_acb: bool = True,
) -> np.ndarray:
    """
    Occam 평활화 거칠기 행렬 R : (n_para, n_para)

    Fortran 대응: Roughening_Occam

    블록 k = j*n_z + i  (j: x-열, i: z-행, 0-based)

    ACB 사용 시: Sm_V = Sm_H = 1 (이방성 없음)
    ACB 미사용:  Sm_V, Sm_H 로 수직/수평 평활화 비율 조절

    Parameters
    ----------
    n_x          : x 방향 블록 수
    n_z          : z 방향 블록 수
    smoothness_v : 수직 평활화 가중치 (ACB 미사용 시)
    smoothness_h : 수평 평활화 가중치 (ACB 미사용 시)
    use_acb      : True → 이방성 없음 (Sm_V = Sm_H = 1)

    Returns
    -------
    R : (n_para, n_para) 밀집 행렬
    """
    n_para = n_x * n_z
    R = np.zeros((n_para, n_para), dtype=float)

    Sm_V = 1.0 if use_acb else smoothness_v
    Sm_H = 1.0 if use_acb else smoothness_h

    for j in range(n_x):
        for i in range(n_z):
            k = j * n_z + i

            # 이웃 블록 결정
            has_up    = (i > 0)
            has_down  = (i < n_z - 1)
            has_left  = (j > 0)
            has_right = (j < n_x - 1)

            n_neighbors = sum([has_up, has_down, has_left, has_right])
            w = 1.0 / max(n_neighbors, 1)

            R[k, k] = 1.0

            if has_down:   # z 방향 +1 (아래)
                R[k, k + 1]      = -w * Sm_V * (2.0 if not use_acb else 1.0)
            if has_up:     # z 방향 -1 (위)
                R[k, k - 1]      = -w * Sm_V * (2.0 if not use_acb else 1.0)
            if has_right:  # x 방향 +1 (우)
                R[k, k + n_z]    = -w * Sm_H * (2.0 if not use_acb else 1.0)
            if has_left:   # x 방향 -1 (좌)
                R[k, k - n_z]    = -w * Sm_H * (2.0 if not use_acb else 1.0)

    return R


def build_roughening_matrix_sparse(
    n_x: int,
    n_z: int,
    smoothness_v: float = 0.5,
    smoothness_h: float = 0.5,
    use_acb: bool = True,
) -> sp.csr_matrix:
    """
    희소 행렬(CSR) 형태의 거칠기 행렬

    대규모 역산에서 메모리 절약.

    Parameters
    ----------
    n_x, n_z     : 블록 격자 크기
    smoothness_v : 수직 평활화 가중치
    smoothness_h : 수평 평활화 가중치
    use_acb      : True → 이방성 없음

    Returns
    -------
    R_sparse : (n_para, n_para) CSR 희소 행렬
    """
    n_para = n_x * n_z
    Sm_V = 1.0 if use_acb else smoothness_v
    Sm_H = 1.0 if use_acb else smoothness_h

    rows, cols, vals = [], [], []

    for j in range(n_x):
        for i in range(n_z):
            k = j * n_z + i

            has_up    = (i > 0)
            has_down  = (i < n_z - 1)
            has_left  = (j > 0)
            has_right = (j < n_x - 1)

            n_nb = sum([has_up, has_down, has_left, has_right])
            w = 1.0 / max(n_nb, 1)
            scale = 2.0 if not use_acb else 1.0

            rows.append(k); cols.append(k); vals.append(1.0)

            if has_down:
                rows.append(k); cols.append(k + 1);    vals.append(-w * Sm_V * scale)
            if has_up:
                rows.append(k); cols.append(k - 1);    vals.append(-w * Sm_V * scale)
            if has_right:
                rows.append(k); cols.append(k + n_z);  vals.append(-w * Sm_H * scale)
            if has_left:
                rows.append(k); cols.append(k - n_z);  vals.append(-w * Sm_H * scale)

    return sp.csr_matrix(
        (np.array(vals), (np.array(rows), np.array(cols))),
        shape=(n_para, n_para),
    )


# ── DLS 대각 정규화 ───────────────────────────────────────────────────────────

def build_identity_regularization(n_para: int) -> np.ndarray:
    """
    단위 행렬 정규화 (DLS: Damped Least Squares)

    Fortran 대응: iOccam == DLS_USED 분기
    """
    return np.eye(n_para, dtype=float)


# ── 모델 구조 벡터 ────────────────────────────────────────────────────────────

def compute_model_structure(
    block_rho: np.ndarray,   # (n_blocks,)
    R: np.ndarray,           # (n_para, n_para) 거칠기 행렬
) -> np.ndarray:
    """
    모델 구조 벡터 Rm (IRLS 가중치 계산에 사용)

    Fortran 대응: rModelStructure = R · model_params
    """
    log_rho = np.log(block_rho)
    return R @ log_rho


# ── 정규화 항 목적함수 ────────────────────────────────────────────────────────

def model_roughness_objective(
    model_structure: np.ndarray,  # (n_para,) — Rm
    irls_weights: np.ndarray,     # (n_para,) — IRLS 가중치
) -> float:
    """
    정규화 목적함수 φ_m = ||W_m · R · m||²

    Fortran 대응: getMeasure(iIRLS_model, BufferVector, n_para, robj_model_zero)

    Parameters
    ----------
    model_structure : R · log(ρ)
    irls_weights    : IRLS 재가중 벡터 (sqrt 적용 전)
    """
    return float(np.sum(irls_weights * model_structure**2))


# ── 정규화 행렬 스케일링 ──────────────────────────────────────────────────────

def scale_roughening_matrix(
    R: np.ndarray,
    lagrangian: np.ndarray,   # (n_para,) ACB 라그랑지안 승수
    irls_weights: np.ndarray, # (n_para,) IRLS 모델 가중치
) -> np.ndarray:
    """
    가중치를 적용한 거칠기 행렬 R̃

    Fortran 대응:
      R(i,:) = sqrt(rLagrangian_ACB(i)) * R(i,:)
      R(i,:) = sqrt(rRWStructure(i)) * R(i,:)

    Parameters
    ----------
    R           : (n_para, n_para) 기본 거칠기 행렬
    lagrangian  : (n_para,) ACB 라그랑지안 승수
    irls_weights: (n_para,) IRLS 재가중치

    Returns
    -------
    R_scaled : (n_para, n_para) 스케일된 거칠기 행렬
    """
    # sqrt(λ_i * w_i) * R[i, :]
    combined = np.sqrt(lagrangian * irls_weights)
    return combined[:, np.newaxis] * R
