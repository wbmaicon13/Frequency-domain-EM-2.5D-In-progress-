"""
전기비저항 이상대(Anomaly) 정의

요구사항 3-1 대응:
  - 원형(Circle), 사각형(Rectangle), 다각형(Polygon) 이상대 지원
  - ResistivityModel 에 직접 삽입

좌표 규약:
  x: 수평 [m], z: 깊이 [m] (양수 = 지하)
  이상대 중심 기준으로 정의
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from .resistivity import ResistivityModel


class Anomaly(ABC):
    """이상대 추상 기반 클래스"""

    def __init__(self, resistivity: float):
        self.resistivity = resistivity

    @abstractmethod
    def contains(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        격자 좌표 (x, z) 가 이상대 내부인지 판별

        Parameters
        ----------
        x, z : 브로드캐스트 가능한 좌표 배열

        Returns
        -------
        mask : bool 배열, True = 이상대 내부
        """

    def apply(self, model: ResistivityModel):
        """
        이상대를 비저항 모델에 적용 (블록 단위)

        이상대가 포함하는 블록의 중심 좌표를 기준으로 판별.
        """
        grid = model.grid
        bp   = model.block_partition

        # 블록 중심 좌표 계산 (요소 중심 = 노드 평균)
        x_nodes = grid.node_x[:, 0]
        z_nodes = grid.node_z[0, :]
        x_elem_centers = 0.5 * (x_nodes[:-1] + x_nodes[1:])
        z_elem_centers = 0.5 * (z_nodes[:-1] + z_nodes[1:])

        n_ex = grid.n_elements_x
        n_ez = grid.n_elements_z

        new_block_rho = model.block_resistivity.copy()

        for ex in range(n_ex):
            for ez in range(n_ez):
                blk = bp.element_block_index[ex, ez]
                if blk < 0:
                    continue
                xc = x_elem_centers[ex]
                zc = z_elem_centers[ez]
                if self.contains(np.array([xc]), np.array([zc]))[0]:
                    new_block_rho[blk] = self.resistivity

        model.set_block_resistivity(new_block_rho)


@dataclass
class CircleAnomaly(Anomaly):
    """
    원형 이상대

    Parameters
    ----------
    center_x, center_z : 중심 좌표 [m]
    radius             : 반지름 [m]
    resistivity        : 이상대 전기비저항 [Ω·m]
    """
    center_x: float
    center_z: float
    radius: float
    resistivity: float = 10.0

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("반지름은 양수여야 합니다.")

    def contains(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (x - self.center_x) ** 2 + (z - self.center_z) ** 2 <= self.radius ** 2


@dataclass
class RectangleAnomaly(Anomaly):
    """
    사각형 이상대

    Parameters
    ----------
    x_min, x_max : x 방향 범위 [m]
    z_min, z_max : z 방향 범위 [m] (깊이)
    resistivity  : 이상대 전기비저항 [Ω·m]
    """
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    resistivity: float = 10.0

    def __post_init__(self):
        if self.x_min >= self.x_max:
            raise ValueError("x_min < x_max 이어야 합니다.")
        if self.z_min >= self.z_max:
            raise ValueError("z_min < z_max 이어야 합니다.")

    def contains(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (
            (x >= self.x_min) & (x <= self.x_max) &
            (z >= self.z_min) & (z <= self.z_max)
        )


@dataclass
class PolygonAnomaly(Anomaly):
    """
    다각형 이상대 (임의 볼록/오목 다각형)

    Parameters
    ----------
    vertices_x : 꼭짓점 x 좌표 배열 [m] (순서 유지)
    vertices_z : 꼭짓점 z 좌표 배열 [m]
    resistivity : 이상대 전기비저항 [Ω·m]
    """
    vertices_x: np.ndarray
    vertices_z: np.ndarray
    resistivity: float = 10.0

    def __post_init__(self):
        self.vertices_x = np.asarray(self.vertices_x, dtype=float)
        self.vertices_z = np.asarray(self.vertices_z, dtype=float)
        if len(self.vertices_x) < 3:
            raise ValueError("다각형은 최소 3개 꼭짓점이 필요합니다.")
        if len(self.vertices_x) != len(self.vertices_z):
            raise ValueError("꼭짓점 x, z 배열 길이가 다릅니다.")

    def contains(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Ray-casting 알고리즘으로 점이 다각형 내부인지 판별
        """
        x = np.asarray(x, dtype=float).ravel()
        z = np.asarray(z, dtype=float).ravel()
        n_pts = len(x)
        inside = np.zeros(n_pts, dtype=bool)

        vx = self.vertices_x
        vz = self.vertices_z
        n_vert = len(vx)

        for k in range(n_pts):
            px, pz = x[k], z[k]
            j = n_vert - 1
            for i in range(n_vert):
                xi, zi = vx[i], vz[i]
                xj, zj = vx[j], vz[j]
                if ((zi > pz) != (zj > pz)) and (
                    px < (xj - xi) * (pz - zi) / (zj - zi) + xi
                ):
                    inside[k] = not inside[k]
                j = i

        return inside.reshape(np.broadcast_shapes(x.shape, z.shape))


def apply_anomalies(model: ResistivityModel, anomalies: list[Anomaly]):
    """여러 이상대를 순서대로 모델에 적용 (나중 이상대가 우선)"""
    for anomaly in anomalies:
        anomaly.apply(model)
