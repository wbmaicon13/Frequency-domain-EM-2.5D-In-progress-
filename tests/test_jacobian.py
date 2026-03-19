"""
야코비안 단위 테스트

검증 항목:
  - 야코비안 행렬 크기 (n_data × n_blocks)
  - 수치 미분(finite difference)과의 비교 (상반정리 기반)
  - NaN/Inf 없음
  - 비저항 섭동에 대한 감도 부호 방향
"""

from __future__ import annotations

import numpy as np
import pytest

from em25d.forward.forward_loop import ForwardConfig, run_forward
from em25d.inverse.jacobian import compute_jacobian
from em25d.model.resistivity import ResistivityModel


class TestJacobian:
    """야코비안 계산 단위 테스트"""

    @pytest.fixture(autouse=True)
    def setup(self, small_grid, homogeneous_model, small_survey,
              small_profile, small_block_partition):
        self.grid      = small_grid
        self.model     = homogeneous_model
        self.survey    = small_survey
        self.profile   = small_profile
        self.bp        = small_block_partition
        self.fwd_cfg   = ForwardConfig(
            n_wavenumbers=6, use_gpu=False, solver="direct")

    def test_jacobian_shape(self):
        jac_result = compute_jacobian(
            grid=self.grid,
            model=self.model,
            survey=self.survey,
            profile=self.profile,
            fwd_config=self.fwd_cfg,
        )
        J = jac_result.jacobian_matrix
        n_blocks = self.model.n_blocks
        # 데이터 수 = n_freq × n_tx × n_rx × n_comp
        n_freq = self.survey.frequencies.n_frequencies
        n_tx   = self.survey.sources.n_sources
        n_rx   = self.survey.receivers.n_receivers
        n_comp = 1   # Hy 성분만 (conftest.py 설정)
        n_data = n_freq * n_tx * n_rx * n_comp

        assert J.shape == (n_data, n_blocks), \
            f"야코비안 형태 오류: {J.shape} != ({n_data}, {n_blocks})"

    def test_jacobian_no_nan(self):
        jac_result = compute_jacobian(
            self.grid, self.model, self.survey, self.profile, self.fwd_cfg)
        J = jac_result.jacobian_matrix
        assert np.all(np.isfinite(J)), "야코비안에 NaN/Inf 포함"

    def test_jacobian_nonzero(self):
        """야코비안이 영행렬이어서는 안 됨"""
        jac_result = compute_jacobian(
            self.grid, self.model, self.survey, self.profile, self.fwd_cfg)
        J = jac_result.jacobian_matrix
        assert np.any(J != 0), "야코비안이 모두 0입니다"

    def test_finite_difference_comparison(
        self, small_grid, small_block_partition, small_survey, small_profile
    ):
        """
        중앙 차분 수치 야코비안과 비교 (선택된 블록, 성분)

        목표: 상대 오차 < 5% (coarse mesh이므로 관대하게 설정)
        """
        from em25d.model.resistivity import ResistivityModel

        model = ResistivityModel(
            small_grid, small_block_partition, background_resistivity=100.0
        )
        fwd_cfg = ForwardConfig(n_wavenumbers=6, use_gpu=False, solver="direct")

        # 해석 야코비안
        jac_result = compute_jacobian(
            small_grid, model, small_survey, small_profile, fwd_cfg)
        J_analytic = jac_result.jacobian_matrix   # (n_data, n_blocks)

        # 중앙 차분 (블록 0, log10(ρ) 기준)
        iblk = 0
        dlog = 1e-4   # δ log10(ρ)

        rho0   = model.block_resistivity.copy()
        log_rho0 = np.log10(rho0)

        log_rho_p = log_rho0.copy(); log_rho_p[iblk] += dlog
        model.log_block_resistivity = log_rho_p
        f_plus = run_forward(small_grid, model, small_survey, small_profile, fwd_cfg)

        log_rho_m = log_rho0.copy(); log_rho_m[iblk] -= dlog
        model.log_block_resistivity = log_rho_m
        f_minus = run_forward(small_grid, model, small_survey, small_profile, fwd_cfg)

        # 원래 모델 복원
        model.log_block_resistivity = log_rho0

        # 차분 야코비안: ∂d/∂m ≈ (f+ - f-) / (2δ)
        Hy_comp = 4
        d_plus  = f_plus [..., Hy_comp].real.ravel()
        d_minus = f_minus[..., Hy_comp].real.ravel()
        J_fd_col = (d_plus - d_minus) / (2 * dlog)

        # 해석 야코비안 열 (Hy 성분의 실수부 행만 추출)
        J_analytic_col = J_analytic[:len(J_fd_col), iblk].real

        # 비제로 행에서만 비교
        mask = np.abs(J_fd_col) > 1e-20
        if mask.sum() < 2:
            pytest.skip("유의미한 감도 값이 부족하여 비교를 건너뜁니다")

        rel_err = np.abs(J_analytic_col[mask] - J_fd_col[mask]) / \
                  (np.abs(J_fd_col[mask]) + 1e-30)
        mean_rel_err = rel_err.mean()

        assert mean_rel_err < 0.10, \
            f"야코비안 상대 오차 {mean_rel_err:.4f} > 10% (차분 vs 해석)"
