"""
물리 상수 및 열거형 정의

Fortran 대응: Fem25Dmod.f90 의 `constant` 모듈
"""

import numpy as np
from enum import IntEnum


# ── 기본 물리 상수 ──────────────────────────────────────────────────────────
PI = np.pi
MU_0 = 4.0 * PI * 1.0e-7          # 진공 자기 투자율 [H/m]
EPSILON_0 = 8.854187817e-12        # 진공 유전율 [F/m]
SPEED_OF_LIGHT = 1.0 / np.sqrt(MU_0 * EPSILON_0)  # 빛의 속도 [m/s]


# ── 송신기(Source) 유형 ──────────────────────────────────────────────────────
class SourceType(IntEnum):
    """
    송신기 유형 (Fortran i_source_type 에 대응)

    전기 쌍극자(Electric Dipole):
      Jx = 1: x 방향 전기 선전류
      Jy = 2: y 방향 전기 선전류  ← 2.5D 기본 (주향 방향)
      Jz = 3: z 방향 전기 선전류

    자기 쌍극자(Magnetic Dipole):
      Mx = 4: x 방향 자기 쌍극자
      My = 5: y 방향 자기 쌍극자
      Mz = 6: z 방향 자기 쌍극자 (수직 루프)
    """
    Jx = 1
    Jy = 2
    Jz = 3
    Mx = 4
    My = 5
    Mz = 6


# ── 장(Field) 성분 유형 ─────────────────────────────────────────────────────
class FieldComponent(IntEnum):
    """관측/계산 장 성분"""
    Ex = 1
    Ey = 2
    Ez = 3
    Hx = 4
    Hy = 5
    Hz = 6


# ── 노름(Norm) 유형 ─────────────────────────────────────────────────────────
class NormType(IntEnum):
    """
    역산 목적함수 노름 유형

    Fortran 대응: Fem25D_Measures.f90 의 i_norm
    """
    L2     = 0   # 최소자승법 (Least Squares)
    L1     = 1   # L1 노름
    HUBER  = 2   # Huber 노름 (L1/L2 혼합)
    EKBLOM = 3   # Ekblom 노름 (부드러운 L1 근사)


# ── 전장(Total/Secondary) 선택 ──────────────────────────────────────────────
class FieldType(IntEnum):
    """계산할 장의 유형"""
    SECONDARY = 0  # 2차장 (= 전체장 - 1차장)
    TOTAL     = 1  # 전체장


# ── 파일 포맷 ────────────────────────────────────────────────────────────────
class FileFormat(IntEnum):
    """출력 파일 포맷 (Fortran i_file_type 대응)"""
    BINARY = 0
    ASCII  = 1


# ── 수치 안전값 ──────────────────────────────────────────────────────────────
FLOAT_EPS = np.finfo(float).eps    # 부동소수점 머신 엡실론
MIN_RESISTIVITY = 1e-4             # 역산 비저항 하한 (Ω·m)
MAX_RESISTIVITY = 1e6              # 역산 비저항 상한 (Ω·m)
