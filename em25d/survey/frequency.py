"""
주파수 설정

Fortran 대응: survey_setup_module 의 frequency_list +
             Fem25Dinv.par 의 주파수 목록

주파수 샘플링 방식:
  - 직접 지정 (list)
  - 로그 균등 (log-spaced)
  - 선형 균등 (lin-spaced)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class FrequencySet:
    """
    탐사 주파수 집합

    Attributes
    ----------
    frequencies : 주파수 배열 [Hz], 오름차순 정렬
    """
    frequencies: np.ndarray

    def __post_init__(self):
        self.frequencies = np.sort(np.asarray(self.frequencies, dtype=float))
        if np.any(self.frequencies <= 0):
            raise ValueError("모든 주파수는 양수여야 합니다.")

    @property
    def n_frequencies(self) -> int:
        return len(self.frequencies)

    @property
    def angular_frequencies(self) -> np.ndarray:
        """각주파수 ω = 2πf [rad/s]"""
        return 2.0 * np.pi * self.frequencies

    @classmethod
    def log_spaced(
        cls,
        f_min: float,
        f_max: float,
        n_frequencies: int,
    ) -> "FrequencySet":
        """로그 균등 주파수 생성"""
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_frequencies)
        return cls(freqs)

    @classmethod
    def lin_spaced(
        cls,
        f_min: float,
        f_max: float,
        n_frequencies: int,
    ) -> "FrequencySet":
        """선형 균등 주파수 생성"""
        freqs = np.linspace(f_min, f_max, n_frequencies)
        return cls(freqs)

    @classmethod
    def from_list(cls, frequency_list: list) -> "FrequencySet":
        """목록에서 직접 생성"""
        return cls(np.array(frequency_list, dtype=float))

    def __repr__(self) -> str:
        return (f"FrequencySet(n={self.n_frequencies}, "
                f"range=[{self.frequencies[0]:.3g}, {self.frequencies[-1]:.3g}] Hz)")
