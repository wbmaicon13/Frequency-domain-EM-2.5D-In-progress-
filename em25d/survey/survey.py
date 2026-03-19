"""
탐사 배열(Survey) 통합 클래스

Fortran 대응: survey_setup_module 전체 +
             Fem25Dinv.par 의 Forward 블록

데이터 포인트 = (송신기 인덱스, 수신기 인덱스) 조합.
기본값: 모든 (Tx, Rx) 조합 (full gather)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from .source import SourceArray
from .receiver import ReceiverArray
from .frequency import FrequencySet


@dataclass
class DataPoint:
    """단일 데이터 포인트 (Tx-Rx 쌍)"""
    tx_index: int     # 송신기 인덱스
    rx_index: int     # 수신기 인덱스


class Survey:
    """
    탐사 배열 전체 설정

    Attributes
    ----------
    sources   : 송신기 배열
    receivers : 수신기 배열
    frequencies : 주파수 집합
    data_points : (Tx, Rx) 쌍 목록
    """

    def __init__(
        self,
        sources: SourceArray,
        receivers: ReceiverArray,
        frequencies: FrequencySet,
        data_points: list[DataPoint] | None = None,
    ):
        self.sources     = sources
        self.receivers   = receivers
        self.frequencies = frequencies

        if data_points is None:
            # 기본: 모든 (Tx, Rx) 조합
            self.data_points = [
                DataPoint(tx, rx)
                for tx in range(sources.n_sources)
                for rx in range(receivers.n_receivers)
            ]
        else:
            self.data_points = data_points

    @property
    def n_data_points(self) -> int:
        return len(self.data_points)

    @property
    def n_total_data(self) -> int:
        """전체 데이터 수 = (Tx×Rx 쌍) × 주파수 × 성분 수"""
        n_comp = len(self.receivers.receivers[0].measured_components)
        return self.n_data_points * self.frequencies.n_frequencies * n_comp

    def tx_rx_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """데이터 포인트의 (tx, rx) 인덱스 배열"""
        tx = np.array([dp.tx_index for dp in self.data_points])
        rx = np.array([dp.rx_index for dp in self.data_points])
        return tx, rx

    def summary(self) -> str:
        lines = [
            "=== Survey Summary ===",
            f"  송신기  : {self.sources.n_sources}개  "
            f"(유형: {self.sources.source_type.name})",
            f"  수신기  : {self.receivers.n_receivers}개",
            f"  주파수  : {self.frequencies.n_frequencies}개  "
            f"({self.frequencies.frequencies[0]:.3g}~"
            f"{self.frequencies.frequencies[-1]:.3g} Hz)",
            f"  데이터  : {self.n_data_points} (Tx×Rx 쌍) "
            f"× {self.frequencies.n_frequencies} freq = {self.n_total_data}",
        ]
        return "\n".join(lines)
