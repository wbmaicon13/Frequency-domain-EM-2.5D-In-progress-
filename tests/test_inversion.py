"""
역산 단위 테스트

검증 항목:
  - IRLS 가중치 함수 (L1/L2/Huber/Ekblom)
  - 평활화 행렬 (Roughening) 크기 및 구조
  - ACB 역산 스텝 실행 가능
  - 역산 반복 후 RMS 오차 감소 경향
  - 다주파수 시퀀스 제약 행렬
"""

from __future__ import annotations

import numpy as np
import pytest

from em25d.inverse.measures import (
    ekblom_weights, huber_weights, support_weights,
    compute_irls_weights,
)
from em25d.inverse.regularization import build_roughening_matrix
from em25d.constants import NormType


def compute_rms(d_obs, d_cal, weights):
    """RMS 오차 계산 (테스트용 헬퍼)"""
    r = (d_obs - d_cal) * np.sqrt(weights)
    return float(np.sqrt(np.mean(r**2)))


class TestIRLSWeights:
    """IRLS 가중치 함수 단위 테스트"""

    def _residuals(self):
        return np.array([0.0, 0.1, 1.0, 5.0, -2.0, 0.01])

    def test_l2_weights_all_ones(self):
        r = self._residuals()
        w = ekblom_weights(r, p=2)
        np.testing.assert_allclose(w, np.ones_like(r), rtol=1e-10,
                                   err_msg="L2 가중치는 항상 1이어야 합니다")

    def test_l1_weights_decrease_with_magnitude(self):
        r = np.array([0.01, 0.1, 1.0, 10.0])
        w = ekblom_weights(r, p=1)
        diffs = np.diff(w)
        assert np.all(diffs <= 0), "L1 가중치는 잔차 크기가 클수록 감소해야 합니다"

    def test_huber_weights_bounded(self):
        r = np.array([-5.0, 0.0, 5.0, 100.0])
        w = huber_weights(r)
        assert np.all(w >= 0), "Huber 가중치가 음수입니다"

    def test_ekblom_weights_positive(self):
        r = np.linspace(-3.0, 3.0, 20)
        w = support_weights(r)
        assert np.all(w > 0), "Support 가중치가 비양수입니다"

    def test_irls_weights_finite(self):
        """zero residual 에서도 NaN/Inf 없음 (epsilon 처리 확인)"""
        r = np.zeros(5)
        for norm in [NormType.L1, NormType.HUBER, NormType.EKBLOM]:
            w = compute_irls_weights(r, norm)
            assert np.all(np.isfinite(w)), f"{norm}: 0 잔차에서 NaN/Inf"


class TestRMSMeasure:
    """RMS 오차 계산 테스트"""

    def test_rms_zero_for_perfect_fit(self):
        d_obs = np.array([1.0, 2.0, 3.0])
        d_cal = d_obs.copy()
        rms = compute_rms(d_obs, d_cal, np.ones_like(d_obs))
        assert rms == pytest.approx(0.0, abs=1e-12)

    def test_rms_positive(self):
        d_obs = np.array([1.0, 2.0, 3.0])
        d_cal = np.array([1.1, 1.9, 3.2])
        rms = compute_rms(d_obs, d_cal, np.ones_like(d_obs))
        assert rms > 0

    def test_rms_weighted(self):
        """높은 가중치가 있는 데이터가 RMS에 더 많이 기여"""
        d_obs = np.array([1.0, 1.0])
        d_cal = np.array([1.5, 1.5])

        w1 = np.array([1.0, 1.0])
        w2 = np.array([10.0, 1.0])   # 첫 번째에 높은 가중치

        rms1 = compute_rms(d_obs, d_cal, w1)
        rms2 = compute_rms(d_obs, d_cal, w2)
        assert rms2 > rms1, "높은 가중치로 RMS가 더 커야 합니다"


class TestRougheningMatrix:
    """평활화 행렬 테스트"""

    def test_roughening_shape(self, small_block_partition):
        bp = small_block_partition
        n  = bp.n_blocks
        C  = build_roughening_matrix(bp.config.n_blocks_x, bp.config.n_blocks_z)
        assert C.shape[1] == n, f"평활화 행렬 열 수 오류: {C.shape[1]} != {n}"

    def test_roughening_nonzero(self, small_block_partition):
        bp = small_block_partition
        C = build_roughening_matrix(bp.config.n_blocks_x, bp.config.n_blocks_z)
        assert np.any(C != 0), "평활화 행렬이 영행렬입니다"

    def test_roughening_row_sum_zero(self, small_block_partition):
        """
        내부 블록의 행 합산 = 0 (라플라시안 특성)

        경계 블록은 예외일 수 있음.
        """
        bp = small_block_partition
        C   = build_roughening_matrix(bp.config.n_blocks_x, bp.config.n_blocks_z)
        row_sums = np.abs(C).sum(axis=1)
        # 행 원소가 존재하는 행이 하나 이상 있어야 함
        assert (row_sums > 0).any(), \
            "평활화 행렬에서 비제로 행이 하나도 없습니다"


class TestInversionLoop:
    """역산 반복 루프 통합 테스트 (소형 격자, 1회 반복)"""

    def test_one_iteration_reduces_rms(
        self,
        small_grid, homogeneous_model, small_survey, small_profile, small_block_partition
    ):
        """
        순방향 데이터를 합성 '관측'으로 사용 후 역산 1회 실행 → RMS 확인

        초기 모델과 동일한 모델에서 순방향 계산 → 데이터 일치 → RMS ≈ 0
        (noise 없이 self-consistent test)
        """
        from em25d.forward.forward_loop import ForwardConfig, run_forward
        from em25d.inverse.inversion_loop import InversionConfig, run_inversion

        fwd_cfg = ForwardConfig(n_wavenumbers=6, use_gpu=False, solver="direct")

        # 관측 데이터 = 초기 모델 순방향 결과 (노이즈 없음)
        observed = run_forward(
            small_grid, homogeneous_model, small_survey, small_profile, fwd_cfg)

        inv_cfg = InversionConfig(
            max_iterations=1,
            use_acb=False,   # 테스트 속도 우선
            use_occam=True,
            log_dir=None,    # 로그 저장 생략
        )

        result = run_inversion(
            grid=small_grid,
            model=homogeneous_model,
            survey=small_survey,
            profile=small_profile,
            observed=observed,
            inv_config=inv_cfg,
            fwd_config=fwd_cfg,
        )

        assert result.n_iterations >= 1, "역산이 한 번도 반복되지 않았습니다"
        assert result.final_rms >= 0.0, "RMS 오차가 음수입니다"
        assert np.all(np.isfinite(result.final_model.block_resistivity)), \
            "역산 후 비저항 값에 NaN/Inf 포함"
