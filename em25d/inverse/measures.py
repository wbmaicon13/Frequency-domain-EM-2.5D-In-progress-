"""
목적함수(Measure) 및 IRLS 가중치

Fortran 대응: Fem25D_Measures.f90
  - getMeasure, dEkblomNorm, dHuberNorm, dSupportNorm
  - getEkblomWeight, getHuberWeight, getSupportWeight

IRLS (Iteratively Reweighted Least Squares) 는 L1/L2/Huber/Ekblom 노름을
최소자승 형태로 반복 근사. 가중치 행렬 W 를 갱신하며 수렴.

수학:
  Ekblom 노름 (p=1, L1 근사):
    ‖x‖_Ekblom = Σ sqrt(xi² + ε²)    ε = mean(|x|)/1000
  Ekblom 노름 (p=2, L2):
    ‖x‖_L2 = Σ xi²
  Huber 노름:
    ρ(xi) = xi²/(2ε)  if |xi| ≤ ε
            |xi| - ε/2  otherwise
    ε = 1.345 · MAD / 0.6745  (Holland & Welsch 1977)
"""

from __future__ import annotations

import numpy as np
from ..constants import NormType


# ── 노름 값 계산 ─────────────────────────────────────────────────────────────

def ekblom_norm(x: np.ndarray, p: int = 1) -> float:
    """
    Ekblom 노름 계산

    Fortran 대응: dEkblomNorm(x, n, p)

    Parameters
    ----------
    x : 잔차 또는 모델 구조 벡터
    p : 1 (L1 근사) 또는 2 (L2)
    """
    eps = np.mean(np.abs(x)) / 1000.0
    if p == 1:
        return float(np.sum(np.sqrt(x**2 + eps**2)))
    elif p == 2:
        return float(np.sum(x**2))
    else:
        raise ValueError(f"p must be 1 or 2, got {p}")


def huber_norm(x: np.ndarray) -> float:
    """
    Huber 노름 계산

    Fortran 대응: dHuberNorm(x, n)

    ε = 1.345 · MAD(x) / 0.6745
    """
    eps = _huber_threshold(x)
    if eps == 0.0:
        return ekblom_norm(x, p=1)
    mask = np.abs(x) <= eps
    return float(
        np.sum(x[mask]**2 / (2 * eps)) +
        np.sum(np.abs(x[~mask]) - eps / 2)
    )


def support_norm(x: np.ndarray) -> float:
    """
    Support (Minimum Support) 노름

    Fortran 대응: dSupportNorm(x, n)

    ρ(xi) = xi² / (xi² + ε²)
    """
    eps = np.mean(np.abs(x)) / 1000.0 + 1e-30
    return float(np.sum(x**2 / (x**2 + eps**2)))


def compute_norm(x: np.ndarray, norm_type: NormType) -> float:
    """
    노름 유형에 따른 목적함수 값 계산

    Fortran 대응: getMeasure(opt, xx, n, norm)
    """
    if norm_type == NormType.L2:
        return ekblom_norm(x, p=2)
    elif norm_type == NormType.L1:
        return ekblom_norm(x, p=1)
    elif norm_type == NormType.HUBER:
        return huber_norm(x)
    elif norm_type == NormType.EKBLOM:
        return support_norm(x)
    else:
        raise ValueError(f"알 수 없는 노름 유형: {norm_type}")


# ── IRLS 가중치 계산 ─────────────────────────────────────────────────────────

def ekblom_weights(x: np.ndarray, p: int = 1) -> np.ndarray:
    """
    Ekblom IRLS 가중치

    Fortran 대응: getEkblomWeight(x, n, p, w)

    p=2 (L2): w = 1  (가중치 없음)
    p=1 (L1): wi = 1 / sqrt(xi² + ε²)
    """
    if p == 2:
        return np.ones_like(x, dtype=float)
    elif p == 1:
        eps = np.mean(np.abs(x)) / 1000.0 + 1e-30
        return 1.0 / np.sqrt(x**2 + eps**2)
    else:
        raise ValueError(f"p must be 1 or 2, got {p}")


def huber_weights(x: np.ndarray) -> np.ndarray:
    """
    Huber IRLS 가중치

    Fortran 대응: getHuberWeight(x, n, w)

    wi = 1/(2ε)  if |xi| ≤ ε
         1/(2|xi|) otherwise
    """
    eps = _huber_threshold(x)
    if eps == 0.0:
        return ekblom_weights(x, p=1)
    w = np.empty_like(x, dtype=float)
    mask = np.abs(x) <= eps
    w[mask]  = 1.0 / (2 * eps)
    w[~mask] = 1.0 / (2 * np.abs(x[~mask]))
    return w


def support_weights(x: np.ndarray) -> np.ndarray:
    """
    Support 노름 IRLS 가중치

    Fortran 대응: getSupportWeight(x, n, w)

    wi = 1 / (xi² + ε²)
    """
    eps = np.mean(np.abs(x)) / 1000.0 + 1e-30
    return 1.0 / (x**2 + eps**2)


def compute_irls_weights(x: np.ndarray, norm_type: NormType) -> np.ndarray:
    """
    IRLS 가중치 벡터 계산

    Fortran 대응: getEkblomWeight / getHuberWeight / getSupportWeight 선택

    Parameters
    ----------
    x         : 잔차(데이터 오차) 또는 모델 구조 벡터
    norm_type : 노름 유형

    Returns
    -------
    w : (n,) float 가중치 벡터 (대각 가중치 행렬의 대각 원소)
    """
    if norm_type == NormType.L2:
        return ekblom_weights(x, p=2)
    elif norm_type == NormType.L1:
        return ekblom_weights(x, p=1)
    elif norm_type == NormType.HUBER:
        return huber_weights(x)
    elif norm_type == NormType.EKBLOM:
        return support_weights(x)
    else:
        raise ValueError(f"알 수 없는 노름 유형: {norm_type}")


# ── 내부 헬퍼 ────────────────────────────────────────────────────────────────

def _mad(x: np.ndarray) -> float:
    """Median Absolute Deviation"""
    return float(np.median(np.abs(x - np.median(x))))


def _huber_threshold(x: np.ndarray) -> float:
    """Huber 임계값 ε = 1.345 · MAD / 0.6745"""
    return 1.345 * _mad(x) / 0.6745
