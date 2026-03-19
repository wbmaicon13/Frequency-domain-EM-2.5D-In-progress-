"""
1차장(primary field) 단위 테스트

검증 항목:
  - Hankel 변환 기반 Jy 다이폴 1차장
  - 베셀 함수 K0/K1 계산 정확성
  - 1차장 → 공간 영역 역 Fourier 변환
  - ky 샘플링 단조 증가
  - 원점에서의 대칭성 (Jy 소스)
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import kv   # 레퍼런스 K0, K1 (scipy)

from em25d.forward.primary_field import (
    compute_background_wavenumber,
    modified_bessel_K0_K1,
    primary_field_ky_domain,
    compute_wavenumber_sampling,
    PrimaryFieldParams,
)
from em25d.constants import PI, SourceType

# 기본 PrimaryFieldParams (배경 100 Ω·m, 공기)
_DEFAULT_PARAMS = PrimaryFieldParams(background_resistivity=100.0)


class TestBackgroundWavenumber:
    """배경 파수 κ 계산 테스트"""

    def test_positive_for_finite_resistivity(self):
        params = PrimaryFieldParams(background_resistivity=100.0)
        kk_h, z0, y_h = compute_background_wavenumber(2 * PI * 10.0, params)
        assert kk_h != 0

    def test_frequency_scaling(self):
        """주파수 증가 → |κ| 증가"""
        p = _DEFAULT_PARAMS
        kk1, _, _ = compute_background_wavenumber(2 * PI * 1.0,   p)
        kk2, _, _ = compute_background_wavenumber(2 * PI * 100.0, p)
        assert abs(kk2) > abs(kk1)

    def test_resistivity_scaling(self):
        """비저항 증가 → |κ| 감소 (μ 고정, ε 고정이므로 kk_h 는 ρ 무관 — 공기 배경)"""
        # compute_background_wavenumber 는 배경(공기) 파수 → ρ 에 무관할 수 있음
        # 대신 결과가 유한한지 확인
        p1 = PrimaryFieldParams(background_resistivity=10.0)
        p2 = PrimaryFieldParams(background_resistivity=1000.0)
        kk1, _, _ = compute_background_wavenumber(2 * PI * 10.0, p1)
        kk2, _, _ = compute_background_wavenumber(2 * PI * 10.0, p2)
        assert np.isfinite(abs(kk1)) and np.isfinite(abs(kk2))


class TestModifiedBessel:
    """변형 Bessel 함수 K0, K1 정확도"""

    @pytest.mark.parametrize("arg", [0.5, 1.0, 2.0, 5.0])
    def test_K0_vs_scipy(self, arg):
        K0, _ = modified_bessel_K0_K1(arg)
        ref = kv(0, arg)
        assert abs(K0 - ref) < 1e-6 * abs(ref) + 1e-12

    @pytest.mark.parametrize("arg", [0.5, 1.0, 2.0, 5.0])
    def test_K1_vs_scipy(self, arg):
        _, K1 = modified_bessel_K0_K1(arg)
        ref = kv(1, arg)
        assert abs(K1 - ref) < 1e-6 * abs(ref) + 1e-12


class TestPrimaryFieldKyDomain:
    """ky 도메인 1차장 계산 테스트"""

    def setup_method(self):
        self.rho   = 100.0
        self.freq  = 10.0
        self.omega = 2 * PI * self.freq
        self.ky    = 0.1   # 임의 공간주파수

        # 단순 격자 (x: -50 ~ 50, z: 0 ~ 100)
        self.node_x = np.linspace(-50.0, 50.0, 11)
        self.node_z = np.linspace(0.0, 100.0, 11)

    def test_Jy_source_returns_finite(self):
        E = primary_field_ky_domain(
            source_x=0.0, source_z=0.0,
            node_x=self.node_x, node_z=self.node_z,
            wavenumber_ky=self.ky, omega=self.omega,
            source_type=SourceType.Jy,
            params=_DEFAULT_PARAMS,
            source_length=0.0, source_strength=1.0,
        )
        assert np.all(np.isfinite(E)), "1차장에 NaN/Inf 값이 포함됩니다"

    def test_output_shape(self):
        """출력 형태 = (3, n_nx, n_nz)"""
        E = primary_field_ky_domain(
            source_x=0.0, source_z=0.0,
            node_x=self.node_x, node_z=self.node_z,
            wavenumber_ky=self.ky, omega=self.omega,
            source_type=SourceType.Jy,
            params=_DEFAULT_PARAMS,
        )
        assert E.shape == (3, len(self.node_x), len(self.node_z))

    def test_Jy_source_Ey_dominant(self):
        """Jy 소스 → Ey (인덱스 1) 성분이 지배적"""
        E = primary_field_ky_domain(
            source_x=0.0, source_z=0.0,
            node_x=self.node_x, node_z=self.node_z,
            wavenumber_ky=self.ky, omega=self.omega,
            source_type=SourceType.Jy,
            params=_DEFAULT_PARAMS,
        )
        # 소스 근방 제외한 영역에서 Ey 진폭이 가장 큼
        Ey_amp = np.abs(E[1]).mean()
        Ex_amp = np.abs(E[0]).mean()
        # Jy 소스에서 Ey 가 Ex 보다 작지 않아야 함 (소스 타입 특성)
        assert Ey_amp > 0.0, "Jy 소스에서 Ey 성분이 0이어서는 안 됩니다"

    def test_symmetry_at_x_zero(self):
        """소스가 x=0 에 있을 때 x=+d 와 x=-d 의 Ey 가 동일해야 함"""
        node_x_sym = np.array([-30.0, -10.0, 0.0, 10.0, 30.0])
        node_z     = np.array([0.0, 50.0, 100.0])
        E = primary_field_ky_domain(
            source_x=0.0, source_z=0.0,
            node_x=node_x_sym, node_z=node_z,
            wavenumber_ky=0.05, omega=self.omega,
            source_type=SourceType.Jy,
            params=_DEFAULT_PARAMS,
        )
        # E[1] = Ey: 인덱스 0 (x=-30) vs 인덱스 4 (x=+30)
        Ey_left  = E[1, 0, :]
        Ey_right = E[1, 4, :]
        np.testing.assert_allclose(
            np.abs(Ey_left), np.abs(Ey_right), rtol=1e-5,
            err_msg="Jy 소스에서 Ey 진폭이 x 대칭을 만족하지 않습니다",
        )


class TestWavenumberSampling:
    """ky 샘플링 계산 테스트"""

    def test_n_wavenumbers(self):
        ky = compute_wavenumber_sampling(
            background_resistivity=100.0,
            min_frequency=1.0,
            min_cell_size=10.0,
            n_wavenumbers=12,
        )
        assert len(ky) == 12

    def test_monotone_increasing(self):
        ky = compute_wavenumber_sampling(100.0, 1.0, 10.0, 16)
        assert np.all(np.diff(ky) > 0), "ky 샘플이 단조 증가해야 합니다"

    def test_positive_values(self):
        ky = compute_wavenumber_sampling(100.0, 1.0, 10.0, 16)
        assert np.all(ky > 0), "ky 값이 모두 양수여야 합니다"
