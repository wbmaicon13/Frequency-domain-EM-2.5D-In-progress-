"""
역산 블록(Block) 구성

Fortran 대응: GetModel.for 의 GenerateBlock 서브루틴 +
             GetModel.par 의 블록 설정

역산 블록 = 동일한 전기비저항을 갖는 요소들의 집합 (역산의 기본 단위).
블록 수가 요소 수보다 훨씬 적어야 역산이 안정적으로 수행됨.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from .grid import Grid


@dataclass
class BlockConfig:
    """
    역산 블록 분할 설정

    Fortran 대응: GetModel.par 의 nxtblck, nztblck + idblckx, idblckz
    """
    n_blocks_x: int = 10     # x 방향 블록 수
    n_blocks_z: int = 10     # z 방향 블록 수 (지하 영역만)
    include_air_in_inversion: bool = False  # 공기층 역산 포함 여부


class BlockPartition:
    """
    격자 요소를 역산 블록으로 분할

    격자의 모델 영역(경계 제외)을 균일하게 블록으로 나누고
    각 요소에 블록 인덱스를 할당.

    Attributes
    ----------
    element_block_index : ndarray (n_elements_x, n_elements_z)
        각 요소의 블록 인덱스 (0-based)
    n_blocks : int
        전체 블록 수
    block_area : ndarray (n_blocks,)
        각 블록의 면적 [m²]
    """

    def __init__(self, grid: Grid, config: BlockConfig):
        self.grid   = grid
        self.config = config
        self._build()

    @property
    def n_blocks(self) -> int:
        return int(self.element_block_index.max()) + 1

    def _build(self):
        cfg  = self.config
        grid = self.grid

        n_ex = grid.n_elements_x
        n_ez = grid.n_elements_z

        # 모델 영역 요소 범위 (경계 요소 제외)
        ix_start = grid.ix_model_start
        ix_end   = grid.ix_model_end
        iz_start = grid.iz_model_start
        iz_end   = grid.iz_model_end

        n_model_x = ix_end - ix_start   # 모델 영역 x 방향 요소 수
        n_model_z = iz_end - iz_start   # 모델 영역 z 방향 요소 수

        # 블록 수가 모델 요소 수를 초과하지 않도록 조정
        nbx = min(cfg.n_blocks_x, n_model_x)
        nbz = min(cfg.n_blocks_z, n_model_z)

        # 요소 → 블록 인덱스 매핑 (모델 영역 내부)
        block_index_x = np.floor(
            np.arange(n_model_x) * nbx / n_model_x
        ).astype(np.int32)
        block_index_z = np.floor(
            np.arange(n_model_z) * nbz / n_model_z
        ).astype(np.int32)

        # 전체 요소 배열 초기화 (경계 요소 = -1, 역산 제외)
        element_block_index = np.full((n_ex, n_ez), -1, dtype=np.int32)

        # 모델 영역 블록 인덱스 할당
        for local_iz, global_iz in enumerate(range(iz_start, iz_end)):
            for local_ix, global_ix in enumerate(range(ix_start, ix_end)):
                bx = block_index_x[local_ix]
                bz = block_index_z[local_iz]
                element_block_index[global_ix, global_iz] = bz * nbx + bx

        self.element_block_index = element_block_index
        self.n_blocks_x = nbx
        self.n_blocks_z = nbz

        # 블록별 면적 계산
        self.block_area = self._compute_block_areas()

    def _compute_block_areas(self) -> np.ndarray:
        """각 블록의 합산 면적 [m²] 계산"""
        grid = self.grid
        areas = np.zeros(self.n_blocks, dtype=float)

        dx = np.diff(grid.node_x[:, 0])   # 요소 x 방향 크기
        dz = np.diff(grid.node_z[0, :])   # 요소 z 방향 크기

        for ex in range(grid.n_elements_x):
            for ez in range(grid.n_elements_z):
                blk = self.element_block_index[ex, ez]
                if blk >= 0:
                    areas[blk] += abs(dx[ex]) * abs(dz[ez])

        return areas

    def summary(self) -> str:
        cfg = self.config
        lines = [
            "=== Block Partition Summary ===",
            f"  블록 분할: {self.n_blocks_x} x {self.n_blocks_z} = {self.n_blocks}",
            f"  블록 면적: min={self.block_area.min():.1f}, max={self.block_area.max():.1f} m²",
        ]
        return "\n".join(lines)
