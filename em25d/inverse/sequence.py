"""
주파수 시퀀스 제약 조건

Fortran 대응: Fem25DSequence.f90 — FrequencySequence

물리적 배경:
  인접 주파수 간 데이터의 연속성을 강제하는 제약조건.
  낮은 주파수에서 높은 주파수로의 전이가 매끄러워야 한다는 사전 정보.

수학:
  시퀀스 행렬 S:
    S[ifreq, ircv, k+(ifreq-1)*n_station] = -w  (이전 주파수)
    S[ifreq, ircv, k+ ifreq   *n_station] = +w  (현재 주파수)

  where w = 1  (iSequenceNorm=0)
        w = 1 / |d_pred(f) - d_pred(f+1)|  (정규화 모드)

  시퀀스 잔차:
    r_seq = S_pred · d_pred - S_obs · d_obs

  정규방정식에 추가:
    NormMatrix += Jᵀ Sᵀ W_seq S J
    MapVector  += Jᵀ Sᵀ W_seq r_seq
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .measures import compute_irls_weights
from ..constants import NormType


def build_sequence_matrix(
    n_freq: int,
    n_station: int,
    n_data: int,
    d_pred: Optional[np.ndarray] = None,   # (n_data,) 예측 데이터 (정규화 모드)
    d_obs: Optional[np.ndarray] = None,    # (n_data,) 관측 데이터 (정규화 모드)
    normalize: bool = False,
    epsilon_frac: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    시퀀스 행렬 S_pred, S_obs 생성

    Fortran 대응: FrequencySequence 내 SequenceMatrixPre, SequenceMatrixObs

    Parameters
    ----------
    n_freq    : 주파수 수
    n_station : 수신기(측점) 수
    n_data    : 전체 데이터 수 (n_freq * n_station)
    d_pred    : 예측 데이터 (정규화 모드 시 필요)
    d_obs     : 관측 데이터 (정규화 모드 시 필요)
    normalize : True → 주파수 차이로 정규화

    Returns
    -------
    S_pred : (n_seq, n_data) 예측 데이터용 시퀀스 행렬
    S_obs  : (n_seq, n_data) 관측 데이터용 시퀀스 행렬
      n_seq = (n_freq - 1) * n_station
    """
    n_seq = (n_freq - 1) * n_station
    S_pred = np.zeros((n_seq, n_data), dtype=float)
    S_obs  = np.zeros((n_seq, n_data), dtype=float)

    if normalize and d_pred is not None:
        eps = abs(np.max(d_pred)) * epsilon_frac
    else:
        eps = 1.0

    for ifreq in range(n_freq - 1):
        for ircv in range(n_station):
            i_seq = ifreq * n_station + ircv
            k_prev = ircv + ifreq * n_station       # 이전 주파수 데이터 인덱스
            k_curr = ircv + (ifreq + 1) * n_station  # 현재 주파수 데이터 인덱스

            if k_prev >= n_data or k_curr >= n_data:
                continue

            if normalize and d_pred is not None and d_obs is not None:
                denom_pred = abs(d_pred[k_prev] - d_pred[k_curr]) + eps
                denom_obs  = abs(d_obs[k_prev]  - d_obs[k_curr])  + eps
                w_pred = 1.0 / denom_pred
                w_obs  = 1.0 / denom_obs
            else:
                w_pred = 1.0
                w_obs  = 1.0

            S_pred[i_seq, k_prev] = -w_pred
            S_pred[i_seq, k_curr] = +w_pred
            S_obs[i_seq, k_prev]  = -w_obs
            S_obs[i_seq, k_curr]  = +w_obs

    return S_pred, S_obs


def compute_sequence_contribution(
    J: np.ndarray,             # (n_data, n_para) 야코비안
    d_pred: np.ndarray,        # (n_data,) 예측 데이터
    d_obs: np.ndarray,         # (n_data,) 관측 데이터
    n_freq: int,
    n_station: int,
    sequence_weights: np.ndarray,  # (n_seq,) 시퀀스 가중치
    norm_type: NormType = NormType.L2,
    normalize: bool = False,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    시퀀스 제약이 정규방정식에 기여하는 항 계산

    Fortran 대응: FrequencySequence → MapVector, NormMatrix

    수학:
      S_weighted = diag(sqrt(w_irls) * seq_weight) · S
      NormMatrix += Jᵀ Sᵀ S J
      MapVector  += Jᵀ Sᵀ r_seq

    Parameters
    ----------
    sequence_weights : (n_seq,) 사용자 정의 시퀀스 가중치

    Returns
    -------
    norm_matrix  : (n_para, n_para) 정규방정식에 추가할 행렬
    map_vector   : (n_para,) 정규방정식에 추가할 벡터
    """
    n_data, n_para = J.shape
    n_seq = (n_freq - 1) * n_station

    S_pred, S_obs = build_sequence_matrix(
        n_freq, n_station, n_data, d_pred, d_obs, normalize)

    # 시퀀스 잔차
    r_seq = S_pred @ d_pred - S_obs @ d_obs   # (n_seq,)

    # IRLS 시퀀스 가중치
    irls_w = compute_irls_weights(r_seq, norm_type)  # (n_seq,)

    # 결합 가중치: sqrt(irls_w * seq_weight)
    combined = np.sqrt(np.abs(irls_w) * np.abs(sequence_weights) * scale)

    S_w = combined[:, np.newaxis] * S_pred   # (n_seq, n_data)
    r_w = combined * r_seq                    # (n_seq,)

    # JᵀSᵀSJ
    SJ = S_w @ J   # (n_seq, n_para)
    norm_matrix = SJ.T @ SJ   # (n_para, n_para)

    # JᵀSᵀr_seq
    map_vector = SJ.T @ r_w   # (n_para,)

    return norm_matrix, map_vector
