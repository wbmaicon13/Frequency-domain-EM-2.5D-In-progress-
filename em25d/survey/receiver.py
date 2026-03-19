"""
수신기(Receiver) 설정

Fortran 대응: survey_setup_module 의 Rx 관련 변수 +
             station_setup_module (지표/시추공 수신기)

수신기 유형:
  - 지표(surface): has_surface_receivers
  - 시추공1(borehole1): has_bh1_receivers
  - 시추공2(borehole2): has_bh2_receivers
  - EM Tx 위치: has_em_tx_receivers
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from ..constants import FieldComponent


class ReceiverType(IntEnum):
    SURFACE   = 0
    BOREHOLE1 = 1
    BOREHOLE2 = 2
    EM_TX     = 3


@dataclass
class Receiver:
    """단일 수신기"""
    x: float
    z: float
    receiver_type: ReceiverType = ReceiverType.SURFACE
    measured_components: tuple = (FieldComponent.Ey,)


class ReceiverArray:
    """
    수신기 배열

    Fortran 대응: receiver_node_x, receiver_node_z + num_receivers
    """

    def __init__(self, receivers: list[Receiver]):
        if len(receivers) == 0:
            raise ValueError("최소 1개의 수신기가 필요합니다.")
        self.receivers = receivers

    @property
    def n_receivers(self) -> int:
        return len(self.receivers)

    @property
    def x(self) -> np.ndarray:
        return np.array([r.x for r in self.receivers])

    @property
    def z(self) -> np.ndarray:
        return np.array([r.z for r in self.receivers])

    @classmethod
    def surface_line(
        cls,
        x_start: float,
        x_end: float,
        n_receivers: int,
        z: float = 0.0,
        measured_components: tuple = (FieldComponent.Ey,),
    ) -> "ReceiverArray":
        """지표 등간격 수신기 배열 생성"""
        x_positions = np.linspace(x_start, x_end, n_receivers)
        receivers = [
            Receiver(x=xp, z=z, receiver_type=ReceiverType.SURFACE,
                     measured_components=measured_components)
            for xp in x_positions
        ]
        return cls(receivers)

    @classmethod
    def from_profile(
        cls,
        profile,   # ProfileNodes
        measured_components: list | tuple = ("Ey",),
    ) -> "ReceiverArray":
        """
        ProfileNodes 로부터 수신기 배열 생성

        Parameters
        ----------
        profile            : ProfileNodes 객체
        measured_components: 측정할 장 성분 이름 목록 (e.g. ["Hy", "Hz"])
        """
        comp_map = {
            "Ex": FieldComponent.Ex, "Ey": FieldComponent.Ey,
            "Ez": FieldComponent.Ez, "Hx": FieldComponent.Hx,
            "Hy": FieldComponent.Hy, "Hz": FieldComponent.Hz,
        }
        comps = tuple(
            comp_map.get(c, FieldComponent.Ey) for c in measured_components
        )
        receivers = [
            Receiver(x=float(profile.receiver_x[i]),
                     z=float(profile.receiver_z[i]),
                     receiver_type=ReceiverType.SURFACE,
                     measured_components=comps)
            for i in range(profile.n_receivers)
        ]
        return cls(receivers)

    @classmethod
    def borehole(
        cls,
        x_position: float,
        z_start: float,
        z_end: float,
        n_receivers: int,
        hole_id: int = 1,
        measured_components: tuple = (FieldComponent.Ey,),
    ) -> "ReceiverArray":
        """시추공 수신기 배열 생성"""
        z_positions = np.linspace(z_start, z_end, n_receivers)
        rtype = ReceiverType.BOREHOLE1 if hole_id == 1 else ReceiverType.BOREHOLE2
        receivers = [
            Receiver(x=x_position, z=zp, receiver_type=rtype,
                     measured_components=measured_components)
            for zp in z_positions
        ]
        return cls(receivers)
