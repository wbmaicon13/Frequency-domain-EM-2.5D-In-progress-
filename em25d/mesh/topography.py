"""
지형(Topography) 처리

Fortran 대응: GetModel.for 의 지형 처리 블록 (line 490~553)

Fortran 문제점:
  - 지형 보간 루프에서 변수 ip, it 가 외부에서 초기화 없이 사용됨
  - 경계 영역의 지형 처리가 주석처리되어 의도가 불명확
  - 지형 좌표 z 의 부호 규약이 혼재

개선 사항:
  - 지형 보간을 명확한 함수로 분리
  - 모든 노드에 일관된 선형 보간 적용
  - 경계 영역은 좌우 끝단 지형값으로 일정하게 연장
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TopographyData:
    """
    지형 데이터

    Parameters
    ----------
    x : 지형 측점 x 좌표 [m], 단조 증가
    z : 각 측점의 지표 고도 [m] (양수 = 지표 위, 음수 = 해수면 아래)
    """
    x: np.ndarray
    z: np.ndarray

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.z = np.asarray(self.z, dtype=float)
        if len(self.x) != len(self.z):
            raise ValueError("지형 x, z 배열 길이가 다릅니다.")
        if len(self.x) < 2:
            raise ValueError("지형 데이터는 최소 2개 측점이 필요합니다.")
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("지형 x 좌표가 단조 증가하지 않습니다.")

    @classmethod
    def from_file(cls, filepath: str) -> "TopographyData":
        """
        지형 데이터 파일 읽기

        파일 형식 (Fortran topo_001/topography_001.dat):
            # 주석 줄
            NoOfTopo
            # 주석 줄
            x1  z1
            x2  z2
            ...
        """
        with open(filepath) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("!")]
        n = int(lines[0])
        coords = [list(map(float, lines[i + 1].split())) for i in range(n)]
        data = np.array(coords)
        return cls(x=data[:, 0], z=data[:, 1])

    def elevation_at(self, x_query: np.ndarray) -> np.ndarray:
        """
        임의 x 위치에서의 지표 고도 선형 보간

        모델 범위 밖은 양 끝단 값으로 연장 (extrapolation)
        """
        return np.interp(x_query, self.x, self.z,
                         left=self.z[0], right=self.z[-1])


def apply_topography(
    node_x: np.ndarray,
    node_z: np.ndarray,
    topography: TopographyData,
    ix_model_start: int,
    ix_model_end: int,
) -> np.ndarray:
    """
    격자 노드 z 좌표에 지형 보정 적용

    각 x 열(column)의 지표 고도만큼 z 좌표 전체를 평행 이동.
    지형이 있는 경우 지표는 항상 z=0 이 아닌 실제 고도에 위치.

    Fortran 대응: GetModel.for line 508~553 의 지형 처리
    (Fortran의 혼재된 부호 규약과 루프 변수 문제를 수정)

    Parameters
    ----------
    node_x : (n_nodes_x, n_nodes_z) — 노드 x 좌표
    node_z : (n_nodes_x, n_nodes_z) — 노드 z 좌표 (수정 대상)
    topography : 지형 데이터
    ix_model_start : 모델 영역 시작 x 인덱스 (경계 제외)
    ix_model_end   : 모델 영역 끝 x 인덱스

    Returns
    -------
    node_z_topo : (n_nodes_x, n_nodes_z) — 지형 보정된 z 좌표
    """
    node_z_topo = node_z.copy()
    n_nodes_x = node_x.shape[0]

    # 각 x 열의 대표 x 좌표 (z 방향으로 같은 x 값이므로 첫 행 사용)
    x_col = node_x[:, 0]

    # 전체 x 범위에 대해 지표 고도 보간
    elevation = topography.elevation_at(x_col)

    # 모든 열에 대해 z 좌표 이동
    # Fortran 규약: z > 0 이 지하. 지형 고도만큼 전체 열 shift
    for ix in range(n_nodes_x):
        node_z_topo[ix, :] += elevation[ix]

    return node_z_topo


def load_topography_or_flat(
    filepath: Optional[str],
    n_nodes_x: int,
    x_range: tuple[float, float],
) -> Optional[TopographyData]:
    """
    지형 파일이 있으면 읽고, 없으면 None 반환 (평탄 지형)

    Parameters
    ----------
    filepath   : 지형 파일 경로 (None 이면 평탄 지형)
    n_nodes_x  : x 방향 노드 수 (평탄 지형 생성 시 사용)
    x_range    : (x_min, x_max) 모델 x 범위
    """
    if filepath is None:
        return None
    return TopographyData.from_file(filepath)
