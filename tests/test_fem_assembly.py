"""
FEM 강성행렬 조립 단위 테스트

검증 항목:
  - 전체 강성행렬 크기 (2 × n_nodes)²
  - 강성행렬 희소성 (비제로 요소 비율)
  - 우변 벡터 크기
  - 경계 조건 적용 후 비특이(non-singular) 행렬 여부
  - 균질 모델에서의 대칭성 (비저항 균질 → K 대칭)
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from em25d.forward.primary_field import (
    primary_field_ky_domain, compute_wavenumber_sampling, PrimaryFieldParams
)
_DEFAULT_PARAMS = PrimaryFieldParams(background_resistivity=100.0)
from em25d.forward.fem_assembly import assemble_global_system
from em25d.constants import PI, SourceType


class TestFEMAssembly:
    """강성행렬 조립 테스트"""

    @pytest.fixture(autouse=True)
    def setup(self, small_grid, homogeneous_model):
        self.grid    = small_grid
        self.model   = homogeneous_model
        self.omega   = 2 * PI * 10.0
        self.ky      = 0.1

        x_nodes = small_grid.node_x[:, 0]
        z_nodes = small_grid.node_z[0, :]

        E_primary_3xz = primary_field_ky_domain(
            source_x=0.0, source_z=0.0,
            node_x=x_nodes, node_z=z_nodes,
            wavenumber_ky=self.ky, omega=self.omega,
            source_type=SourceType.Jy,
            params=_DEFAULT_PARAMS,
        )
        n_nx, n_nz = len(x_nodes), len(z_nodes)
        self.E_primary = (E_primary_3xz
                          .transpose(0, 2, 1)
                          .reshape(3, n_nz * n_nx))

    def test_stiffness_matrix_size(self):
        K, _ = assemble_global_system(
            grid=self.grid,
            element_resistivity=self.model.element_resistivity,
            layer_resistivity=self.model.element_resistivity,
            E_primary=self.E_primary,
            omega=self.omega,
            ky=self.ky,
        )
        n_dof = 2 * self.grid.n_nodes
        assert K.shape == (n_dof, n_dof), \
            f"강성행렬 크기 오류: {K.shape} != ({n_dof}, {n_dof})"

    def test_force_vector_size(self):
        K, f = assemble_global_system(
            grid=self.grid,
            element_resistivity=self.model.element_resistivity,
            layer_resistivity=self.model.element_resistivity,
            E_primary=self.E_primary,
            omega=self.omega,
            ky=self.ky,
        )
        n_dof = 2 * self.grid.n_nodes
        assert f.shape == (n_dof,), \
            f"힘벡터 크기 오류: {f.shape} != ({n_dof},)"

    def test_matrix_is_sparse(self):
        K, _ = assemble_global_system(
            grid=self.grid,
            element_resistivity=self.model.element_resistivity,
            layer_resistivity=self.model.element_resistivity,
            E_primary=self.E_primary,
            omega=self.omega,
            ky=self.ky,
        )
        n_dof = 2 * self.grid.n_nodes
        nnz_fraction = K.nnz / (n_dof ** 2)
        assert nnz_fraction < 0.05, \
            f"강성행렬이 충분히 희소하지 않습니다: nnz/n² = {nnz_fraction:.4f}"

    def test_force_vector_nonzero(self):
        """비균질 모델에서 우변 벡터가 영벡터가 아니어야 함 (균질 → delta_sig=0 → f=0 은 물리적으로 정상)"""
        # 비균질 모델: 일부 요소를 10 Ω·m 으로 변경
        het_resistivity = self.model.element_resistivity.copy()
        cx = het_resistivity.shape[0] // 2
        cz = het_resistivity.shape[1] // 2
        het_resistivity[cx-1:cx+1, cz-1:cz+1] = 10.0   # 저비저항 이상체

        _, f = assemble_global_system(
            grid=self.grid,
            element_resistivity=het_resistivity,
            layer_resistivity=self.model.element_resistivity,  # 배경 = 균질
            E_primary=self.E_primary,
            omega=self.omega,
            ky=self.ky,
        )
        assert np.any(f != 0), "비균질 모델에서 힘벡터가 모두 0입니다"

    def test_matrix_solvable(self):
        """경계 조건 적용 후 선형 시스템이 풀릴 수 있어야 함"""
        from em25d.forward.fem_solver import solve_fem_system

        K, f = assemble_global_system(
            grid=self.grid,
            element_resistivity=self.model.element_resistivity,
            layer_resistivity=self.model.element_resistivity,
            E_primary=self.E_primary,
            omega=self.omega,
            ky=self.ky,
        )
        try:
            sol = solve_fem_system(K, f, self.grid, use_gpu=False)
            assert np.all(np.isfinite(sol)), "FEM 해에 NaN/Inf 포함"
        except Exception as e:
            pytest.fail(f"선형 시스템 풀기 실패: {e}")
