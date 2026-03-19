"""
프로파일 노드 생성

Fortran 대응: GetModel.for 의 GenerateProfileNodes 서브루틴

프로파일 = 수신기(Rx) 위치에 대응하는 격자 노드.
FEM 해로부터 관측점 위치의 전자기장을 추출할 때 사용.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from .grid import Grid


@dataclass
class ProfileConfig:
    """
    프로파일 노드 설정

    Parameters
    ----------
    n_receivers : 수신기 개수
    x_start     : 프로파일 시작 x 좌표 [m]
    x_end       : 프로파일 끝 x 좌표 [m]
    surface_z   : 수신기 z 좌표 (기본 0 = 지표)
    """
    n_receivers: int = 21
    x_start: float = -100.0
    x_end: float   = 100.0
    surface_z: float = 0.0


class ProfileNodes:
    """
    수신기 위치에 가장 가까운 격자 노드 탐색

    Attributes
    ----------
    node_indices_x : ndarray (n_receivers,)
        각 수신기에 대응하는 x 방향 노드 인덱스
    node_indices_z : ndarray (n_receivers,)
        각 수신기에 대응하는 z 방향 노드 인덱스
    receiver_x : ndarray (n_receivers,)
        수신기 x 좌표
    receiver_z : ndarray (n_receivers,)
        수신기 z 좌표 (지표 = 0, 지하 = 양수)
    """

    def __init__(
        self,
        grid: Grid,
        receiver_x_or_config,
        receiver_z: np.ndarray = None,
    ):
        """
        두 가지 초기화 방식 지원:

        1. ProfileNodes(grid, receiver_x_array, receiver_z_array)
        2. ProfileNodes(grid, ProfileConfig)  — 등간격 지표 프로파일 자동 생성
        """
        self.grid = grid

        if isinstance(receiver_x_or_config, ProfileConfig):
            cfg = receiver_x_or_config
            rx  = np.linspace(cfg.x_start, cfg.x_end, cfg.n_receivers)
            rz  = np.full(cfg.n_receivers, cfg.surface_z)
        else:
            rx = np.asarray(receiver_x_or_config, dtype=float)
            rz = np.asarray(receiver_z, dtype=float)

        self.receiver_x = rx
        self.receiver_z = rz
        self._find_nearest_nodes()

    @property
    def n_receivers(self) -> int:
        return len(self.receiver_x)

    @property
    def x_positions(self) -> np.ndarray:
        """수신기 x 좌표 배열"""
        return self.receiver_x

    def global_node_indices(self) -> np.ndarray:
        """각 수신기에 대응하는 전역 노드 번호 (1D)"""
        return self.grid.node_index(self.node_indices_x, self.node_indices_z)

    def _find_nearest_nodes(self):
        """수신기 좌표에 가장 가까운 격자 노드 탐색 (최근접 보간)"""
        grid = self.grid

        # 1D 좌표 배열 (직교 격자이므로 첫 행/열 사용)
        x_nodes = grid.node_x[:, 0]
        z_nodes = grid.node_z[0, :]

        ix = np.array([np.argmin(np.abs(x_nodes - rx)) for rx in self.receiver_x])
        iz = np.array([np.argmin(np.abs(z_nodes - rz)) for rz in self.receiver_z])

        self.node_indices_x = ix
        self.node_indices_z = iz

    @classmethod
    def surface_profile(
        cls,
        grid: Grid,
        x_start: float,
        x_end: float,
        n_receivers: int,
    ) -> "ProfileNodes":
        """
        지표 수신기 프로파일 생성 (등간격)

        Parameters
        ----------
        x_start, x_end : 프로파일 x 범위 [m]
        n_receivers     : 수신기 개수
        """
        rx = np.linspace(x_start, x_end, n_receivers)
        rz = np.zeros(n_receivers)   # 지표 (z = 0)
        return cls(grid, rx, rz)

    @classmethod
    def borehole_profile(
        cls,
        grid: Grid,
        x_position: float,
        z_start: float,
        z_end: float,
        n_receivers: int,
    ) -> "ProfileNodes":
        """
        시추공(Borehole) 수신기 프로파일 생성 (등간격, 수직)

        Parameters
        ----------
        x_position      : 시추공 x 위치 [m]
        z_start, z_end  : 시추공 심도 범위 [m] (양수 = 지하)
        n_receivers     : 수신기 개수
        """
        rx = np.full(n_receivers, x_position)
        rz = np.linspace(z_start, z_end, n_receivers)
        return cls(grid, rx, rz)
