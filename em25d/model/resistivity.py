"""
전기비저항 모델

Fortran 대응: model_res/topo_001/Model_00001.dat + mproprty.dat / blck_res.dat

전기비저항 모델은 두 가지 표현을 지원:
  1. 블록 기반 (block-based): 역산 블록별 비저항 (역산 변수)
  2. 요소 기반 (element-based): FEM 각 요소별 비저항 (순방향 계산용)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional
from ..mesh.grid import Grid
from ..mesh.block import BlockPartition


class ResistivityModel:
    """
    전기비저항 모델

    Attributes
    ----------
    block_resistivity : ndarray (n_blocks,)
        블록별 전기비저항 [Ω·m] (역산 변수)
    element_resistivity : ndarray (n_elements_x, n_elements_z)
        요소별 전기비저항 [Ω·m] (FEM 계산용)
    """

    def __init__(
        self,
        grid: Grid,
        block_partition: BlockPartition,
        background_resistivity: float = 100.0,
    ):
        self.grid             = grid
        self.block_partition  = block_partition
        self.background_resistivity = background_resistivity

        n_blocks = block_partition.n_blocks
        self.block_resistivity = np.full(n_blocks, background_resistivity, dtype=float)
        self._sync_elements()

    # ── 공개 메서드 ─────────────────────────────────────────────────────────

    @property
    def n_blocks(self) -> int:
        return len(self.block_resistivity)

    @property
    def log_block_resistivity(self) -> np.ndarray:
        """역산 변수: log10(ρ) — 역산 시 대수 변환 사용"""
        return np.log10(self.block_resistivity)

    @log_block_resistivity.setter
    def log_block_resistivity(self, log_rho: np.ndarray):
        self.block_resistivity = 10.0 ** np.asarray(log_rho)
        self._sync_elements()

    def set_block_resistivity(self, values: np.ndarray):
        """블록 비저항 설정 후 요소 배열 동기화"""
        self.block_resistivity = np.asarray(values, dtype=float)
        self._sync_elements()

    def element_conductivity(self) -> np.ndarray:
        """요소별 전기전도도 σ = 1/ρ [S/m]"""
        return 1.0 / self.element_resistivity

    def to_file(self, filepath: str):
        """
        비저항 모델 파일 저장

        형식: Fortran Model_NNNNN.dat 와 호환
        헤더 포함, 블록 인덱스 + 비저항 값
        """
        with open(filepath, "w") as f:
            f.write(f"# ResistivityModel n_blocks={self.n_blocks}\n")
            f.write(f"{'Block':>8s}  {'Rho(Ohm.m)':>14s}\n")
            for i, rho in enumerate(self.block_resistivity):
                f.write(f"{i + 1:8d}  {rho:14.4f}\n")

    @classmethod
    def from_file(
        cls,
        filepath: str,
        grid: Grid,
        block_partition: BlockPartition,
    ) -> "ResistivityModel":
        """비저항 모델 파일에서 읽기"""
        data = np.loadtxt(filepath, comments="#")
        rho  = data[:, 1] if data.ndim == 2 else data
        model = cls(grid, block_partition)
        model.set_block_resistivity(rho)
        return model

    # ── 내부 메서드 ─────────────────────────────────────────────────────────

    def _sync_elements(self):
        """블록 비저항 → 요소 비저항 배열 동기화"""
        bp  = self.block_partition
        idx = bp.element_block_index   # (n_ex, n_ez)

        n_ex = self.grid.n_elements_x
        n_ez = self.grid.n_elements_z
        self.element_resistivity = np.full(
            (n_ex, n_ez), self.background_resistivity, dtype=float)

        mask = idx >= 0
        self.element_resistivity[mask] = self.block_resistivity[idx[mask]]
