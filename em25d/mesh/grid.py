"""
2.5D FEM 격자 생성

Fortran 대응: GetModel.for 의 l_GetModel2D20001 + mkz 서브루틴

주요 개선:
  - .2DF 바이너리 의존 제거 → Python 파라미터로 직접 설정
  - 경계 확장(공기/바닥/좌우)을 별도 함수로 명확히 분리
  - 지형 처리(topography.py)와 완전히 분리
  - 노드/요소 인덱스 0-based (Fortran 1-based → 변환)

격자 좌표 규약:
  x: 수평 방향 (오른쪽 양수)
  z: 깊이 방향 (아래 양수)  ← Fortran과 동일 부호 규약
  공기층: z < 0 (지표 위)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GridConfig:
    """
    격자 설정 파라미터

    Fortran 대응: start_settings_module + model_grid_module 변수들
    """
    # ── 모델 영역 ──────────────────────────────────────────────────────────
    n_x_cells: int = 50             # 모델 영역 x 방향 셀 수
    n_z_cells: int = 30             # 모델 영역 z 방향 셀 수 (지하)
    n_z_cells_air: int = 5          # 공기층 셀 수 (지표 위)

    base_x_cell_size: float = 10.0  # 기본 x 셀 크기 [m]
    base_z_cell_size: float = 5.0   # 기본 z 셀 크기 [m] (지하)
    base_z_cell_size_air: float = 5.0  # 기본 z 셀 크기 [m] (공기)

    # ── 경계 확장 ──────────────────────────────────────────────────────────
    include_air_boundary: bool = True   # 공기층 위쪽 경계 확장 여부
    boundary_stretch_factor: float = 1.3   # 경계 셀 팽창 계수 (Fortran zfact)
    n_x_boundary_cells: int = 0     # x 경계 셀 수 (0 = 자동 계산)
    n_z_boundary_top_cells: int = 0 # 상단 경계 셀 수 (0 = 자동 계산)
    n_z_boundary_bottom_cells: int = 0  # 하단 경계 셀 수 (0 = 자동 계산)

    # ── 배경 물성 ──────────────────────────────────────────────────────────
    halfspace_resistivity: float = 100.0   # 배경 전기비저항 [Ω·m]
    halfspace_permittivity: float = 1.0    # 비유전율 (상대)
    halfspace_ip_effect: float = 0.0       # IP 효과 (미사용시 0)

    # ── 경계 셀 자동 계산용 주파수 ──────────────────────────────────────────
    reference_frequency: float = 1.0      # 경계 크기 계산 기준 주파수 [Hz]


class Grid:
    """
    2.5D FEM 직교 격자

    전체 격자 = 모델 영역 + 경계 확장 (좌/우/상/하)

    Attributes
    ----------
    node_x : ndarray (n_nodes_x, n_nodes_z)
        각 노드의 x 좌표 [m]
    node_z : ndarray (n_nodes_x, n_nodes_z)
        각 노드의 z 좌표 [m] (아래 방향 양수)
    element_block_index : ndarray (n_elements_x, n_elements_z)
        각 요소가 속하는 블록 인덱스 (역산 단위)
    """

    def __init__(self, config: GridConfig = None, *, _skip_build: bool = False):
        self.config = config
        if config is not None and not _skip_build:
            self._build()

    @classmethod
    def from_coordinates(
        cls,
        x_nodes_1d: np.ndarray,
        z_nodes_1d: np.ndarray,
    ) -> "Grid":
        """
        1D 노드 좌표 배열로부터 직교 격자 생성 (레거시 파일 호환)

        Parameters
        ----------
        x_nodes_1d : (n_nodes_x,) 정렬된 x 좌표
        z_nodes_1d : (n_nodes_z,) 정렬된 z 좌표
        """
        grid = cls(_skip_build=True)
        nx = len(x_nodes_1d)
        nz = len(z_nodes_1d)
        grid.node_x = np.broadcast_to(
            x_nodes_1d[:, np.newaxis], (nx, nz)).copy()
        grid.node_z = np.broadcast_to(
            z_nodes_1d[np.newaxis, :], (nx, nz)).copy()
        grid.element_block_index = np.zeros(
            (nx - 1, nz - 1), dtype=np.int32)
        grid._minimum_cell_size = float(min(
            np.diff(x_nodes_1d).min(),
            np.diff(z_nodes_1d).min(),
        ))
        # 경계 인덱스 (레거시: 전체가 모델 영역)
        grid.n_x_boundary_cells = 0
        grid.n_z_boundary_top_cells = 0
        grid.n_z_boundary_bottom_cells = 0
        grid.ix_model_start = 0
        grid.ix_model_end = nx - 1
        grid.iz_model_start = 0
        grid.iz_model_end = nz - 1
        return grid

    # ── 공개 메서드 ─────────────────────────────────────────────────────────

    @property
    def n_nodes_x(self) -> int:
        return self.node_x.shape[0]

    @property
    def n_nodes_z(self) -> int:
        return self.node_x.shape[1]

    @property
    def n_nodes(self) -> int:
        return self.n_nodes_x * self.n_nodes_z

    @property
    def n_elements_x(self) -> int:
        return self.n_nodes_x - 1

    @property
    def n_elements_z(self) -> int:
        return self.n_nodes_z - 1

    @property
    def n_elements(self) -> int:
        return self.n_elements_x * self.n_elements_z

    @property
    def minimum_cell_size(self) -> float:
        """모델 내부 영역의 최소 셀 크기 [m]"""
        return self._minimum_cell_size

    def node_index(self, ix: int, iz: int) -> int:
        """2D 인덱스 → 1D 전역 노드 번호 (0-based, 행 우선)"""
        return iz * self.n_nodes_x + ix

    def element_nodes(self, ex: int, ez: int) -> tuple[int, int, int, int]:
        """
        요소 (ex, ez)의 4개 노드 전역 번호 반환

        반환 순서: 좌하, 우하, 우상, 좌상 (반시계)
        Fortran 대응: cal_elem 의 노드 순서와 동일
        """
        n0 = self.node_index(ex,     ez)
        n1 = self.node_index(ex + 1, ez)
        n2 = self.node_index(ex + 1, ez + 1)
        n3 = self.node_index(ex,     ez + 1)
        return n0, n1, n2, n3

    def summary(self) -> str:
        cfg = self.config
        lines = [
            "=== Grid Summary ===",
            f"  모델 영역   : {cfg.n_x_cells} x {cfg.n_z_cells} cells",
            f"  공기층      : {cfg.n_z_cells_air} cells",
            f"  경계 확장   : x±{self.n_x_boundary_cells}, z상:{self.n_z_boundary_top_cells}, z하:{self.n_z_boundary_bottom_cells}",
            f"  전체 노드   : {self.n_nodes_x} x {self.n_nodes_z} = {self.n_nodes}",
            f"  전체 요소   : {self.n_elements_x} x {self.n_elements_z} = {self.n_elements}",
            f"  최소 셀 크기: {self.minimum_cell_size:.2f} m",
            f"  x 범위      : [{self.node_x.min():.1f}, {self.node_x.max():.1f}] m",
            f"  z 범위      : [{self.node_z.min():.1f}, {self.node_z.max():.1f}] m",
        ]
        return "\n".join(lines)

    # ── 내부 구현 ───────────────────────────────────────────────────────────

    def _build(self):
        """격자 좌표 배열 생성 (경계 확장 포함)"""
        cfg = self.config

        # 1) 스킨 깊이 기반 경계 크기 계산
        skin_depth = _skin_depth(cfg.halfspace_resistivity, cfg.reference_frequency)

        # 2) 모델 영역 x 좌표 (노드 위치)
        x_model = np.arange(cfg.n_x_cells + 1) * cfg.base_x_cell_size

        # 3) 모델 영역 z 좌표 (공기 + 지하)
        z_air = -np.arange(cfg.n_z_cells_air, -1, -1) * cfg.base_z_cell_size_air
        z_sub = np.arange(1, cfg.n_z_cells + 1) * cfg.base_z_cell_size
        z_model = np.concatenate([z_air, z_sub])   # 음수(공기) ~ 양수(지하)

        # 4) 경계 셀 크기 배열 (기하급수 팽창)
        dx_left  = cfg.base_x_cell_size
        dx_right = cfg.base_x_cell_size
        dz_top   = cfg.base_z_cell_size_air
        dz_bot   = cfg.base_z_cell_size

        n_xb = cfg.n_x_boundary_cells or _auto_boundary_count(
            dx_left, cfg.boundary_stretch_factor, skin_depth * 3.0)
        n_ztop = (cfg.n_z_boundary_top_cells or _auto_boundary_count(
            dz_top, cfg.boundary_stretch_factor, skin_depth * 4.0)
            if cfg.include_air_boundary else 0)
        n_zbot = cfg.n_z_boundary_bottom_cells or _auto_boundary_count(
            dz_bot, cfg.boundary_stretch_factor, skin_depth * 2.0)

        self.n_x_boundary_cells = n_xb
        self.n_z_boundary_top_cells = n_ztop
        self.n_z_boundary_bottom_cells = n_zbot

        x_bound_left  = _boundary_coordinates(dx_left,  cfg.boundary_stretch_factor, n_xb,  side="left")
        x_bound_right = _boundary_coordinates(dx_right, cfg.boundary_stretch_factor, n_xb,  side="right", offset=x_model[-1])
        z_bound_top   = _boundary_coordinates(dz_top,   cfg.boundary_stretch_factor, n_ztop, side="left", offset=z_model[0])
        z_bound_bot   = _boundary_coordinates(dz_bot,   cfg.boundary_stretch_factor, n_zbot, side="right", offset=z_model[-1])

        # 5) 전체 좌표 배열 조합
        x_all = np.concatenate([x_bound_left, x_model, x_bound_right])
        z_all = np.concatenate([z_bound_top,  z_model, z_bound_bot])

        # 6) 2D 좌표 격자 (브로드캐스트)
        self.node_x = np.broadcast_to(x_all[:, np.newaxis], (len(x_all), len(z_all))).copy()
        self.node_z = np.broadcast_to(z_all[np.newaxis, :], (len(x_all), len(z_all))).copy()

        # 7) 요소 블록 인덱스 (기본: 단일 배경 블록 = 0)
        self.element_block_index = np.zeros(
            (self.n_elements_x, self.n_elements_z), dtype=np.int32)

        # 8) 최소 셀 크기
        self._minimum_cell_size = float(np.min(np.diff(x_all)))

        # 9) 경계 범위 인덱스 저장 (다른 모듈에서 참조)
        self.ix_model_start = n_xb
        self.ix_model_end   = n_xb + cfg.n_x_cells
        self.iz_model_start = n_ztop
        self.iz_model_end   = n_ztop + cfg.n_z_cells_air + cfg.n_z_cells


# ── 헬퍼 함수 ───────────────────────────────────────────────────────────────

def _skin_depth(resistivity: float, frequency: float) -> float:
    """전자기 스킨 깊이 δ = sqrt(ρ / (π·f·μ₀)) [m]"""
    from em25d.constants import MU_0, PI
    return np.sqrt(resistivity / (PI * frequency * MU_0))


def _auto_boundary_count(
    first_cell_size: float,
    stretch_factor: float,
    max_extent: float,
) -> int:
    """
    경계 확장 셀 수 자동 결정

    기하급수적으로 팽창하는 셀이 max_extent 를 넘을 때까지의 개수

    Fortran 대응: mkz 서브루틴
    """
    total = 0.0
    size  = first_cell_size
    count = 0
    while total < max_extent:
        total += size
        size  *= stretch_factor
        count += 1
        if count > 200:     # 무한 루프 안전장치
            break
    return count


def _boundary_coordinates(
    first_cell_size: float,
    stretch_factor: float,
    n_cells: int,
    side: str,
    offset: float = 0.0,
) -> np.ndarray:
    """
    경계 확장 노드 좌표 배열 생성

    Parameters
    ----------
    first_cell_size : 경계에 가장 가까운 셀 크기
    stretch_factor  : 기하급수 팽창 비율
    n_cells         : 경계 셀 수
    side            : "left" (음의 방향) 또는 "right" (양의 방향)
    offset          : 시작 좌표 기준값
    """
    if n_cells == 0:
        return np.array([])

    sizes = first_cell_size * stretch_factor ** np.arange(n_cells)
    cumul = np.cumsum(sizes)

    if side == "left":
        # 모델 영역 왼쪽/위쪽: 음의 방향으로 누적
        coords = offset - cumul[::-1]
    else:
        # 모델 영역 오른쪽/아래쪽: 양의 방향으로 누적
        coords = offset + cumul

    return coords
