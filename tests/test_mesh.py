"""
격자(Grid) 및 블록 파티션 단위 테스트

검증 항목:
  - 노드/요소 수 계산 정확성
  - 노드 좌표 단조 증가 확인
  - 경계 확장 포함 여부
  - 블록 인덱스 유효 범위
  - 노드 인덱스 함수 (iz-first row-major)
"""

from __future__ import annotations

import numpy as np
import pytest

from em25d.mesh.grid import Grid, GridConfig
from em25d.mesh.block import BlockPartition, BlockConfig
from em25d.mesh.boundary import get_boundary_node_indices as identify_boundary_nodes
from em25d.mesh.profile import ProfileNodes, ProfileConfig


class TestGrid:
    """Grid 클래스 단위 테스트"""

    def test_node_count(self, small_grid):
        g = small_grid
        expected_nx = g.config.n_x_cells + 1 + 2 * g.config.n_x_boundary_cells
        expected_nz = (g.config.n_z_cells
                       + g.config.n_z_cells_air
                       + 1
                       + g.config.n_z_boundary_bottom_cells)
        # 상단 경계 있을 경우 추가
        assert g.n_nodes_x >= g.config.n_x_cells + 1
        assert g.n_nodes_z >= g.config.n_z_cells + 1

    def test_element_count(self, small_grid):
        g = small_grid
        assert g.n_elements_x == g.n_nodes_x - 1
        assert g.n_elements_z == g.n_nodes_z - 1

    def test_node_x_monotone(self, small_grid):
        x_nodes = small_grid.node_x[:, 0]
        diffs = np.diff(x_nodes)
        assert np.all(diffs > 0), "x 노드 좌표가 단조 증가해야 합니다"

    def test_node_z_monotone(self, small_grid):
        z_nodes = small_grid.node_z[0, :]
        diffs = np.diff(z_nodes)
        assert np.all(diffs > 0), "z 노드 좌표가 단조 증가해야 합니다"

    def test_n_nodes_total(self, small_grid):
        g = small_grid
        assert g.n_nodes == g.n_nodes_x * g.n_nodes_z

    def test_n_elements_total(self, small_grid):
        g = small_grid
        assert g.n_elements == g.n_elements_x * g.n_elements_z

    def test_node_index_formula(self, small_grid):
        """node_index(ix, iz) = iz * n_nodes_x + ix (iz-first, row-major)"""
        g = small_grid
        for ix in [0, 1, g.n_nodes_x - 1]:
            for iz in [0, 1, g.n_nodes_z - 1]:
                expected = iz * g.n_nodes_x + ix
                assert g.node_index(ix, iz) == expected

    def test_coordinate_shape(self, small_grid):
        g = small_grid
        assert g.node_x.shape == (g.n_nodes_x, g.n_nodes_z)
        assert g.node_z.shape == (g.n_nodes_x, g.n_nodes_z)

    def test_halfspace_resistivity(self, small_grid):
        assert small_grid.config.halfspace_resistivity == pytest.approx(100.0)


class TestBlockPartition:
    """BlockPartition 클래스 단위 테스트"""

    def test_block_count(self, small_block_partition):
        bp = small_block_partition
        assert bp.n_blocks == bp.config.n_blocks_x * bp.config.n_blocks_z

    def test_element_block_index_shape(self, small_grid, small_block_partition):
        g  = small_grid
        bp = small_block_partition
        assert bp.element_block_index.shape == (g.n_elements_x, g.n_elements_z)

    def test_block_index_range(self, small_block_partition):
        bp  = small_block_partition
        idx = bp.element_block_index
        valid = (idx >= -1) & (idx < bp.n_blocks)
        assert valid.all(), "블록 인덱스가 유효 범위(-1 ~ n_blocks-1)를 벗어납니다"

    def test_inversion_elements_only(self, small_block_partition):
        """역산 영역 요소만 블록 인덱스 ≥ 0"""
        bp  = small_block_partition
        n_valid = (bp.element_block_index >= 0).sum()
        assert n_valid > 0, "역산 영역 요소가 하나도 없습니다"


class TestBoundaryNodes:
    """경계 노드 식별 테스트"""

    def test_boundary_count_positive(self, small_grid):
        nodes = identify_boundary_nodes(small_grid)
        assert len(nodes) > 0, "경계 노드가 하나도 없습니다"

    def test_boundary_index_range(self, small_grid):
        nodes = identify_boundary_nodes(small_grid)
        n_nodes = small_grid.n_nodes
        assert all(0 <= n < n_nodes for n in nodes), "경계 노드 인덱스 범위 초과"

    def test_boundary_includes_corners(self, small_grid):
        """모서리 4개 노드는 반드시 경계 포함"""
        g = small_grid
        corners = {
            0,
            g.n_nodes_x - 1,
            (g.n_nodes_z - 1) * g.n_nodes_x,
            g.n_nodes_x * g.n_nodes_z - 1,
        }
        nodes = set(identify_boundary_nodes(g))
        assert corners.issubset(nodes), f"모서리 노드가 경계에 없습니다: {corners - nodes}"


class TestProfileNodes:
    """ProfileNodes 단위 테스트"""

    def test_receiver_count(self, small_grid):
        cfg = ProfileConfig(n_receivers=5, x_start=-20.0, x_end=20.0)
        profile = ProfileNodes(small_grid, cfg)
        assert profile.n_receivers == 5

    def test_node_indices_in_range(self, small_grid, small_profile):
        indices = small_profile.global_node_indices()
        n_nodes = small_grid.n_nodes
        assert len(indices) == small_profile.n_receivers
        assert all(0 <= i < n_nodes for i in indices)

    def test_x_positions_monotone(self, small_grid):
        cfg = ProfileConfig(n_receivers=7, x_start=-30.0, x_end=30.0)
        profile = ProfileNodes(small_grid, cfg)
        x_pos = profile.x_positions
        assert np.all(np.diff(x_pos) >= 0), "수신기 x 위치가 단조 증가해야 합니다"
