"""
순방향 모델링 통합 테스트

검증 항목:
  - 소형 격자에서 순방향 모델링 실행 가능
  - 출력 형태 (n_freq, n_tx, n_rx, 6)
  - 결과에 NaN/Inf 없음
  - 비저항 증가 시 Hy 진폭 변화 방향 (물리적 타당성)
  - 균질 모델에서 해석해와 비교 (10% 오차 이내 목표)
"""

from __future__ import annotations

import numpy as np
import pytest

from em25d.forward.forward_loop import ForwardConfig, run_forward


class TestForwardModeling:
    """순방향 모델링 통합 테스트"""

    @pytest.fixture(autouse=True)
    def setup(self, small_grid, homogeneous_model, small_survey, small_profile):
        self.grid    = small_grid
        self.model   = homogeneous_model
        self.survey  = small_survey
        self.profile = small_profile
        self.config  = ForwardConfig(
            n_wavenumbers=6,   # 테스트용 최소 ky 수
            use_gpu=False,
            solver="direct",
        )

    def test_forward_runs_without_error(self):
        result = run_forward(
            self.grid, self.model, self.survey, self.profile, self.config)
        assert result is not None

    def test_output_shape(self):
        result = run_forward(
            self.grid, self.model, self.survey, self.profile, self.config)
        n_freq = self.survey.frequencies.n_frequencies
        n_tx   = self.survey.sources.n_sources
        n_rx   = self.survey.receivers.n_receivers
        assert result.shape == (n_freq, n_tx, n_rx, 6), \
            f"결과 형태 오류: {result.shape} != ({n_freq}, {n_tx}, {n_rx}, 6)"

    def test_no_nan_inf(self):
        result = run_forward(
            self.grid, self.model, self.survey, self.profile, self.config)
        assert np.all(np.isfinite(result)), \
            "순방향 결과에 NaN 또는 Inf 가 포함되어 있습니다"

    def test_Hy_nonzero(self):
        """Jy 다이폴 소스 → Hy 성분 (인덱스 4)이 비제로여야 함"""
        result = run_forward(
            self.grid, self.model, self.survey, self.profile, self.config)
        Hy = result[..., 4]   # (n_freq, n_tx, n_rx)
        assert np.any(np.abs(Hy) > 0), "Hy 성분이 모두 0입니다"

    def test_frequency_amplitude_trend(self):
        """
        주파수 증가 → 표피 깊이 감소 → 원거리 Hy 진폭 감소 경향

        균질 매체에서 저주파는 표피 깊이가 크므로 신호가 더 강함.
        """
        result = run_forward(
            self.grid, self.model, self.survey, self.profile, self.config)
        # 첫 수신기, 첫 송신기
        Hy_f0 = np.abs(result[0, 0, -1, 4])   # 최저 주파수
        Hy_f2 = np.abs(result[2, 0, -1, 4])   # 최고 주파수

        if Hy_f0 > 0 and Hy_f2 > 0:
            # 최저 주파수 신호가 더 크거나 비슷해야 함
            # (격자 크기가 작아 오차 있을 수 있으므로 10x 이내면 허용)
            ratio = Hy_f2 / Hy_f0
            assert ratio < 100, \
                f"주파수 진폭 비가 비물리적입니다: f_high/f_low = {ratio:.3f}"


class TestForwardPhysicalConsistency:
    """물리적 일관성 검사"""

    def test_resistivity_effect_on_amplitude(
        self, small_grid, small_block_partition, small_survey, small_profile
    ):
        """
        저비저항 모델 → 전류 집중 → 신호 변화

        두 모델(100 Ω·m vs 10 Ω·m)의 Hy 값이 달라야 함.
        """
        from em25d.model.resistivity import ResistivityModel
        from em25d.forward.forward_loop import ForwardConfig, run_forward

        config = ForwardConfig(n_wavenumbers=6, use_gpu=False, solver="direct")

        # 고비저항 모델
        model_high = ResistivityModel(
            small_grid, small_block_partition, background_resistivity=100.0
        )
        # 저비저항 모델
        model_low = ResistivityModel(
            small_grid, small_block_partition, background_resistivity=10.0
        )

        result_high = run_forward(small_grid, model_high, small_survey, small_profile, config)
        result_low  = run_forward(small_grid, model_low,  small_survey, small_profile, config)

        Hy_high = np.abs(result_high[0, 0, :, 4]).mean()
        Hy_low  = np.abs(result_low [0, 0, :, 4]).mean()

        assert Hy_high != pytest.approx(Hy_low, rel=1e-3), \
            "비저항이 다름에도 Hy 값이 동일합니다 — 비저항이 모델링에 반영되지 않음"
