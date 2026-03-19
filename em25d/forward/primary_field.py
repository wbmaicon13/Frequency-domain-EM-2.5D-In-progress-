"""
1차장(Primary Field) 계산

Fortran 대응: Fem25Dprimary.f90 — Primary_Driver, E_primary_M, primary_space_w

2.5D 전자탐사에서 1차장은 균질 전공간(homogeneous wholespace) 해석해.
공간주파수(ky) 영역에서 계산한 뒤, 역 Fourier 변환으로 실공간 취득.

수학적 배경:
  Maxwell 방정식을 y 방향으로 Fourier 변환하면 2D 문제로 환원.
  배경 전공간의 전기장은 수정 Bessel 함수 K0, K1 으로 표현됨.

  파수 관계:  r0 = sqrt(ky² - kk_h²)
  배경 파수:  kk_h = sqrt(-iωμ · iωε) = sqrt(ω²με)  (도전율 무시)
  K0, K1:    2차 수정 Bessel 함수 (scipy.special.kv 사용)
"""

from __future__ import annotations

import numpy as np
from scipy.special import kv
from dataclasses import dataclass
from typing import Optional

from ..constants import MU_0, EPSILON_0, PI, SourceType


# ── 내부 상수 ────────────────────────────────────────────────────────────────
_CONE = 1j   # 허수 단위 (Fortran: cone = dcmplx(0., 1.))
_SIG_AIR = 1e-8   # 공기 전기전도도 [S/m] (Fortran: sig_air)


@dataclass
class PrimaryFieldParams:
    """
    1차장 계산 파라미터

    Fortran 대응: YzK 모듈 + Constant 모듈의 전역 변수
    """
    background_resistivity: float = 100.0  # 배경 비저항 [Ω·m] (anomalous_resistivity)
    mu: float = MU_0                        # 자기 투자율 [H/m]
    epsilon: float = EPSILON_0              # 유전율 [F/m]


def compute_background_wavenumber(
    omega: float,
    params: PrimaryFieldParams,
) -> tuple:
    """
    배경 매질의 복소 파수 kk_h 계산

    kk_h = sqrt(-z0 · y_h)
    z0   = iωμ              (임피던스)
    y_h  = iωε              (어드미턴스, 자유공간)

    Fortran 대응 (Fem25Dprimary.f90):
      z0   = cone * w * mu
      y_h  = cone * w * eps          ← 도전율 항 주석 처리됨
      !y_h = cone * w * eps + 1/res0 ← 비활성

    주의: Fortran은 자유공간(whole-space) 1차장 사용.
      mat_prop_layer=0 (Fem25Dpar.f90 라인 440-442)으로 강제하여
      delta_sigma = sigma (전체 전도도)가 2차장 소스항.

    Returns
    -------
    (kk_h, z0, y_h) : 모두 complex
    """
    z0  = _CONE * omega * params.mu
    # Fortran: y_h = cone * w * eps (자유공간, 도전율 항 없음)
    y_h = _CONE * omega * params.epsilon
    kk_h = np.sqrt(-z0 * y_h)
    return kk_h, z0, y_h


def modified_bessel_K0_K1(arg: complex) -> tuple[complex, complex]:
    """
    2차 수정 Bessel 함수 K0(arg), K1(arg)

    Fortran 대응: mbessel(2, arg, cm) 서브루틴

    scipy.special.kv 사용:
      K0(z) = kv(0, z)
      K1(z) = kv(1, z)
    """
    K0 = kv(0, arg)
    K1 = kv(1, arg)
    return K0, K1


def primary_field_ky_domain(
    source_x: float,
    source_z: float,
    node_x: np.ndarray,    # (n_x,) 격자 x 좌표
    node_z: np.ndarray,    # (n_z,) 격자 z 좌표
    wavenumber_ky: float,
    omega: float,
    source_type: SourceType,
    params: PrimaryFieldParams,
    source_length: float = 0.0,
    source_strength: float = 1.0,
) -> np.ndarray:
    """
    공간주파수(ky) 영역에서 1차장 E 계산 (numpy 벡터화)

    Fortran 대응: E_primary_M 서브루틴

    Fortran 인덱싱: ix=1..nx, iz=1..nzt (1-based)
    Python 인덱싱: ix=0..n_x-1, iz=0..n_z-1 (0-based) — 값은 동일

    반환:
      E_p : (3, n_x, n_z) 복소 배열
            E_p[0] = Ex, E_p[1] = Ey, E_p[2] = Ez
    """
    kk_h, z0, y_h = compute_background_wavenumber(omega, params)
    ky = wavenumber_ky

    rr0 = ky**2 - kk_h**2
    r0  = np.sqrt(rr0 + 0j)

    # 브로드캐스트: (n_x, 1) × (1, n_z) → (n_x, n_z) — Fortran 이중 루프 대체
    Xr = (node_x - source_x)[:, np.newaxis]   # (n_x, 1)
    dz = (node_z - source_z)[np.newaxis, :]   # (1, n_z)

    rho = np.sqrt(Xr**2 + dz**2)   # (n_x, n_z)

    # 특이점 마스크 (소스 위치 = Xr=0, dz=0)
    # Fortran 대응 (Fem25Dprimary.f90 라인 336-355):
    #   singular: rho=1e-3, K0=0, K1=0
    #   itype=2(Jy dipole): e_p = dcmplx(-1,-1) 마커
    singular = rho < 1e-10
    rho_safe = np.where(singular, 1e-3, rho)   # Fortran: rho=1.d-3

    arg = r0 * rho_safe                        # (n_x, n_z)
    K0 = kv(0, arg)                            # (n_x, n_z)
    K1 = kv(1, arg)                            # (n_x, n_z)

    # Fortran: singular 점에서 K0=K1=0
    K0 = np.where(singular, 0.0, K0)
    K1 = np.where(singular, 0.0, K1)

    # 송신기 유형별 계산 (벡터화)
    E_p = _source_field(
        source_type, ky, r0, rr0, kk_h, z0, y_h,
        Xr, dz, rho_safe, K0, K1,
        source_length, source_strength,
    )   # (3, n_x, n_z)

    # Fortran: itype=2(Jy dipole)에서 singular 점은 (-1,-1) 마커
    # Primary() 함수가 4노드 평균 시 마커도 포함 (필터링 없음)
    if source_type == SourceType.Jy and source_length == 0.0:
        E_p[:, singular] = complex(-1.0, -1.0)
    else:
        E_p[:, singular] = 0.0

    return E_p


def _source_field(
    source_type: SourceType,
    ky: float,
    r0: complex, rr0: complex,
    kk_h: complex, z0: complex, y_h: complex,
    Xr: np.ndarray, dz: np.ndarray, rho: np.ndarray,
    K0: np.ndarray, K1: np.ndarray,
    source_length: float,
    source_strength: float,
) -> np.ndarray:
    """
    송신기 유형별 1차장 공식 (numpy 배열 입력 지원)

    Fortran 대응: E_primary_M 내 itype 분기 (line 347~386)

    입력 배열 shape: (n_x, n_z) (브로드캐스트 결과)
    반환: (3, n_x, n_z)

    수학 공식 (전공간, ky 영역):
      Jy 쌍극자(source_length=0):
        Ex = -i·ky·r0·Xr / (2π·y_h·ρ) · K1
        Ey = -r0² / (2π·y_h) · K0
        Ez = -i·ky·r0·dz / (2π·y_h·ρ) · K1

    주의: Fortran 1-based 인덱스와 Python 0-based 인덱스의 차이는
          좌표값 자체에 영향을 주지 않으므로 수식은 동일함.
    """
    shape = np.broadcast_shapes(np.shape(Xr), np.shape(dz))
    Ex = np.zeros(shape, dtype=complex)
    Ey = np.zeros(shape, dtype=complex)
    Ez = np.zeros(shape, dtype=complex)
    i0 = 1j

    if source_type == SourceType.Jx:
        Ex = (-z0 / (2*PI) * ((1 + rr0*Xr**2 / ((kk_h*rho)**2)) * K0
              - r0 / (kk_h**2 * rho) * (1 - 2*Xr**2/rho**2) * K1))
        Ey = (-i0 * ky * r0 * Xr / (2*PI * y_h * rho) * K1)
        Ez = (r0 * Xr * dz / (2*PI * y_h * rho**2) * (r0*K0 + 2/rho*K1))

    elif source_type == SourceType.Jy:
        if source_length == 0.0:   # 쌍극자(dipole)
            Ex = -i0 * ky * r0 * Xr / (2*PI * y_h * rho) * K1
            Ey = -rr0 / (2*PI * y_h) * K0
            Ez = -i0 * ky * r0 * dz / (2*PI * y_h * rho) * K1
        else:                       # 유한 길이 선 전류(finite wire)
            sinc_val = np.sin(ky * source_length / 2)
            # Fortran: z0*(rr0)/(pi*kk0**2)*k0*sin(aky*L/2)/aky
            # kk0 = kk_h (공기 배경 파수)
            Ex = (2*i0 * source_strength * z0 / (2*PI)
                  * r0 * Xr / rho * K1 * sinc_val)
            if ky != 0:
                Ey = (source_strength * z0 * rr0 / (PI * kk_h**2)
                      * K0 * sinc_val / ky)
            else:
                Ey = np.zeros(shape, dtype=complex)
            Ez = (2*i0 * source_strength * z0 / (2*PI)
                  * r0 * dz / rho * K1 * sinc_val)

    elif source_type == SourceType.Jz:
        Ex = r0 * Xr * dz / (2*PI * y_h * rho**2) * (r0*K0 + 2/rho*K1)
        Ey = -i0 * ky * r0 * dz / (2*PI * y_h * rho) * K1
        Ez = (-z0 / (2*PI) * ((1 + rr0*dz**2 / ((kk_h*rho)**2)) * K0
              + r0 / (kk_h**2 * rho) * (1 - 2*Xr**2/rho**2) * K1))

    elif source_type == SourceType.Mx:
        # Ex = 0
        Ey = z0 * r0 * dz / (2*PI * rho) * K1
        Ez = i0 * ky * z0 / (2*PI) * K0

    elif source_type == SourceType.My:
        Ex = -z0 * r0 * dz / (2*PI * rho) * K1
        # Ey = 0
        Ez = z0 * r0 * Xr / (2*PI * rho) * K1

    elif source_type == SourceType.Mz:
        Ex = -i0 * ky * z0 / (2*PI) * K0
        Ey = -z0 * r0 * Xr / (2*PI * rho) * K1
        # Ez = 0

    return np.array([Ex, Ey, Ez])   # (3, n_x, n_z)


def primary_field_space_domain(
    source_x: float,
    source_z: float,
    receiver_x: np.ndarray,   # (n_receivers,)
    receiver_z: np.ndarray,
    omega: float,
    source_type: SourceType,
    params: PrimaryFieldParams,
    field_type: str = "E",    # "E" or "H"
    source_length: float = 0.0,
    source_strength: float = 1.0,
) -> np.ndarray:
    """
    실공간에서 1차장 해석해 계산 (전공간 균질 매질)

    Fortran 대응: primary_space_w 서브루틴

    실공간 전공간 해 (Jy 쌍극자, field_type="E"):
      Ey = -z0 · exp(-kk_h·r) / (4π·r) · K0(kk_h·r)  [선전류]
           (여기서 r = sqrt(dx²+dz²))

    반환:
      field : (3, n_receivers) 복소 배열 [Ex, Ey, Ez]
    """
    kk_h, z0, y_h = compute_background_wavenumber(omega, params)

    dx = receiver_x - source_x
    dz = receiver_z - source_z
    rr = np.sqrt(dx**2 + dz**2)
    rr = np.where(rr < 1e-10, 1e-10, rr)   # 특이점 방지

    n_rec = len(receiver_x)
    field = np.zeros((3, n_rec), dtype=complex)

    if field_type == "E":
        ker1 = np.exp(-_CONE * kk_h * rr) / (4*PI * rr**2) * (_CONE*kk_h*rr + 1)
        ker2 = np.exp(-_CONE * kk_h * rr) / (4*PI * rr)
        ker3 = -kk_h**2 * rr**2 + 3*_CONE*kk_h*rr + 3

        if source_type == SourceType.Jy:
            if source_length == 0.0:   # 쌍극자
                field[1] = -ker2 * z0 - ker1 / (y_h * rr)
            else:                       # 선 전류
                arg = _CONE * kk_h * rr
                K0, _ = modified_bessel_K0_K1(arg)
                field[1] = (-z0 / (2*PI) * K0
                            * source_length * source_strength)
        elif source_type == SourceType.Jx:
            field[0] = (-z0*ker2 - ker1/(y_h*rr)
                        + ker2*ker3*dx**2 / (rr**4*y_h))
            field[2] = ker2 / (rr**2*y_h) * ker3 * dx*dz / rr**2
        elif source_type == SourceType.Jz:
            field[0] = ker2 / (rr**2*y_h) * ker3 * dx*dz / rr**2
            field[2] = (-z0*ker2 - ker1/(y_h*rr)
                        + ker2*ker3*dz**2 / (rr**4*y_h))
        elif source_type == SourceType.Mx:
            field[1] = z0 * ker1 * dz / rr
        elif source_type == SourceType.My:
            field[0] = -z0 * ker1 * dz / rr
            field[2] =  z0 * ker1 * dx / rr
        elif source_type == SourceType.Mz:
            field[1] = -z0 * ker1 * dx / rr

    else:   # field_type == "H"
        ker1 = np.exp(-_CONE * kk_h * rr) / (4*PI * rr)
        ker2 = _CONE * kk_h * rr + 1
        ker3 = -kk_h**2 * rr**2 + 3*_CONE*kk_h*rr + 3

        if source_type == SourceType.Jy:
            if source_length == 0.0:
                field[0] =  ker1 / rr * ker2 * dz / rr
                field[2] = -ker1 / rr * ker2 * dx / rr
            else:
                arg = _CONE * kk_h * rr
                _, K1 = modified_bessel_K0_K1(arg)
                field[0] = (_CONE*kk_h / (2*PI) * K1 * dz/rr
                            * source_length * source_strength)
                field[2] = (-_CONE*kk_h / (2*PI) * K1 * dx/rr
                            * source_length * source_strength)
        elif source_type == SourceType.Jx:
            field[1] = -ker1 / rr * ker2 * dz / rr
        elif source_type == SourceType.Jz:
            field[1] =  ker1 / rr * ker2 * dx / rr
        elif source_type == SourceType.Mx:
            field[0] = ker1 * (kk_h**2 + 1/rr**2*(-ker2 + dx**2/rr**2*ker3))
            field[2] = ker1 / rr**2 * ker3 * dx*dz / rr**2
        elif source_type == SourceType.My:
            field[1] = ker1 * (kk_h**2 - 1/rr**2 * ker2)
        elif source_type == SourceType.Mz:
            field[0] = ker1 / rr**2 * ker3 * dx*dz / rr**2
            field[2] = ker1 * (kk_h**2 + 1/rr**2*(-ker2 + dz**2/rr**2*ker3))

    return field


def compute_wavenumber_sampling(
    background_resistivity: float,
    min_frequency: float,
    min_cell_size: float,
    n_wavenumbers: int,
) -> np.ndarray:
    """
    공간주파수(ky) 샘플링 결정

    Fortran 대응: Primary_Driver 의 ky 배열 설정 (line 99~120)

    수학:
      스킨 깊이: δ = 500·sqrt(ρ/f) [m]
      ky_max = π / Δx_min   (Nyquist)
      ky_min = 2π / (30·δ)  (저주파 한계)
      ky 배열: 로그 균등 + ky[0] = 1e-8 (직류 근사)
    """
    skin_depth = 500.0 * np.sqrt(background_resistivity / min_frequency)
    ky_max = PI / min_cell_size
    ky_min = 2 * PI / (30.0 * skin_depth)

    # 로그 균등 샘플링 (2번째부터)
    log_min = np.log10(ky_min)
    log_max = np.log10(ky_max)
    delf = (log_max - log_min) / (n_wavenumbers - 2)

    ky = np.zeros(n_wavenumbers)
    ky[0] = 1.0e-8   # 직류 근사
    for i in range(1, n_wavenumbers):
        ky[i] = 10.0 ** (log_min + (i - 1) * delf)

    return ky
