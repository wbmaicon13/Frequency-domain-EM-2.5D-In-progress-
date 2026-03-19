"""
송신기(Source/Transmitter) 설정

Fortran 대응: survey_setup_module 의 Tx 관련 변수 +
             Fem25Dinv.par 의 Source 파라미터

6종 송신기 유형:
  전기 쌍극자: Jx(1), Jy(2), Jz(3)
  자기 쌍극자: Mx(4), My(5), Mz(6)

2.5D 가정: 송신기는 y=0 평면에 위치, y 방향으로 무한 연장 또는 쌍극자
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from ..constants import SourceType


@dataclass
class Source:
    """
    단일 송신기

    Parameters
    ----------
    x, z        : 송신기 위치 [m] (y=0 고정)
    source_type : SourceType 열거형 (Jx~Mz)
    strength    : 전류 강도 [A] 또는 자기 모멘트 [A·m²]
    length      : 쌍극자 길이 [m] (전기 쌍극자의 경우)
    """
    x: float
    z: float
    source_type: SourceType = SourceType.Jy
    strength: float = 1.0
    length: float = 1.0

    @property
    def is_electric_dipole(self) -> bool:
        return self.source_type in (SourceType.Jx, SourceType.Jy, SourceType.Jz)

    @property
    def is_magnetic_dipole(self) -> bool:
        return self.source_type in (SourceType.Mx, SourceType.My, SourceType.Mz)

    def moment(self) -> float:
        """
        쌍극자 모멘트 크기

        전기 쌍극자: I·L [A·m]
        자기 쌍극자: m [A·m²]
        """
        if self.is_electric_dipole:
            return self.strength * self.length
        return self.strength


class SourceArray:
    """
    송신기 배열 (여러 송신기 위치)

    Fortran 대응: transmitter_node_x, transmitter_node_z +
                 num_transmitters
    """

    def __init__(self, sources: list[Source]):
        if len(sources) == 0:
            raise ValueError("최소 1개의 송신기가 필요합니다.")
        self.sources = sources

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    @property
    def x(self) -> np.ndarray:
        return np.array([s.x for s in self.sources])

    @property
    def z(self) -> np.ndarray:
        return np.array([s.z for s in self.sources])

    @property
    def source_type(self) -> SourceType:
        """모든 송신기가 동일한 유형이어야 함 (2.5D 가정)"""
        types = set(s.source_type for s in self.sources)
        if len(types) > 1:
            raise ValueError("2.5D 순방향 모델링은 단일 송신기 유형만 지원합니다.")
        return types.pop()

    @classmethod
    def surface_line(
        cls,
        x_start: float,
        x_end: float,
        n_sources: int,
        z: float = 0.0,
        source_type: SourceType = SourceType.Jy,
        strength: float = 1.0,
        length: float = 1.0,
    ) -> "SourceArray":
        """지표 등간격 송신기 배열 생성"""
        x_positions = np.linspace(x_start, x_end, n_sources)
        sources = [
            Source(x=xp, z=z, source_type=source_type,
                   strength=strength, length=length)
            for xp in x_positions
        ]
        return cls(sources)
