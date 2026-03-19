"""
딥러닝 데이터셋용 전기비저항 모델 대량 생성

요구사항 3-3 대응:
  - GUI에서 설정한 기본 모델로부터 위치/크기/비저항을 랜덤 섭동
  - 자유도(다양성) 파라미터로 변형 범위 제어
  - 최대 200개 생성
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy

from .resistivity import ResistivityModel
from .anomaly import Anomaly, CircleAnomaly, RectangleAnomaly, apply_anomalies
from ..mesh.grid import Grid
from ..mesh.block import BlockPartition


@dataclass
class GeneratorConfig:
    """
    모델 생성기 설정

    Parameters
    ----------
    n_models          : 생성할 모델 수 (최대 200)
    freedom           : 자유도 (0~1). 1에 가까울수록 원본 모델과 달라짐
    seed              : 랜덤 시드 (재현성)
    resistivity_range : (min, max) 이상대 비저항 범위 [Ω·m]
    position_range    : 이상대 중심 이동 최대 거리 [m] (freedom × range)
    size_range        : 이상대 크기 변화 최대 비율 (0~1, freedom 기반)
    """
    n_models: int = 10
    freedom: float = 0.3           # 0 = 원본 유지, 1 = 최대 변형
    seed: Optional[int] = 42
    resistivity_log_range: tuple = (0.5, 4.5)   # log10(Ω·m) 범위
    allow_new_anomalies: bool = True             # 새 이상대 추가 여부
    max_anomalies_per_model: int = 3             # 모델당 최대 이상대 수

    def __post_init__(self):
        self.n_models = min(self.n_models, 200)
        self.freedom  = np.clip(self.freedom, 0.0, 1.0)


class ModelGenerator:
    """
    기본 모델로부터 변형 모델들을 생성

    사용 예시
    ---------
    >>> config = GeneratorConfig(n_models=50, freedom=0.4)
    >>> gen = ModelGenerator(base_model, base_anomalies, config)
    >>> models = gen.generate()
    """

    def __init__(
        self,
        base_model: ResistivityModel,
        base_anomalies: list[Anomaly],
        config: GeneratorConfig,
    ):
        self.base_model     = base_model
        self.base_anomalies = base_anomalies
        self.config         = config
        self._rng = np.random.default_rng(config.seed)

    def generate(self) -> list[ResistivityModel]:
        """
        변형 모델 리스트 생성

        Returns
        -------
        models : ResistivityModel 리스트 (n_models 개)
        """
        cfg    = self.config
        models = []

        for _ in range(cfg.n_models):
            model = self._create_variant()
            models.append(model)

        return models

    def _create_variant(self) -> ResistivityModel:
        """단일 변형 모델 생성"""
        cfg   = self.config
        grid  = self.base_model.grid
        bp    = self.base_model.block_partition

        # 배경 비저항 섭동
        bg_log = np.log10(self.base_model.background_resistivity)
        noise  = cfg.freedom * self._rng.uniform(-0.3, 0.3)
        new_bg = 10.0 ** np.clip(bg_log + noise, *cfg.resistivity_log_range)

        model = ResistivityModel(grid, bp, background_resistivity=new_bg)

        # 기존 이상대 변형
        perturbed = [self._perturb_anomaly(a) for a in self.base_anomalies]

        # 새 이상대 추가 (allow_new_anomalies 옵션)
        if cfg.allow_new_anomalies:
            n_new = self._rng.integers(0, cfg.max_anomalies_per_model)
            for _ in range(n_new):
                perturbed.append(self._random_anomaly(grid))

        apply_anomalies(model, perturbed)
        return model

    def _perturb_anomaly(self, anomaly: Anomaly) -> Anomaly:
        """기존 이상대의 위치/크기/비저항 랜덤 섭동"""
        cfg = self.config
        f   = cfg.freedom

        # 비저항 섭동
        log_rho = np.log10(anomaly.resistivity)
        log_rho += f * self._rng.uniform(-0.5, 0.5)
        new_rho = 10.0 ** np.clip(log_rho, *cfg.resistivity_log_range)

        if isinstance(anomaly, CircleAnomaly):
            x_range = self.base_model.grid.node_x.max() - self.base_model.grid.node_x.min()
            dx = f * x_range * self._rng.uniform(-0.1, 0.1)
            dz = f * x_range * self._rng.uniform(-0.1, 0.1)
            dr = f * anomaly.radius * self._rng.uniform(-0.3, 0.3)
            return CircleAnomaly(
                center_x    = anomaly.center_x + dx,
                center_z    = max(0.0, anomaly.center_z + dz),
                radius      = max(1.0, anomaly.radius + dr),
                resistivity = new_rho,
            )
        elif isinstance(anomaly, RectangleAnomaly):
            w  = anomaly.x_max - anomaly.x_min
            h  = anomaly.z_max - anomaly.z_min
            dx = f * w * self._rng.uniform(-0.2, 0.2)
            dz = f * h * self._rng.uniform(-0.2, 0.2)
            dw = f * w * self._rng.uniform(-0.3, 0.3)
            dh = f * h * self._rng.uniform(-0.3, 0.3)
            new_w = max(1.0, w + dw)
            new_h = max(1.0, h + dh)
            cx = 0.5 * (anomaly.x_min + anomaly.x_max) + dx
            cz = 0.5 * (anomaly.z_min + anomaly.z_max) + dz
            return RectangleAnomaly(
                x_min       = cx - new_w / 2,
                x_max       = cx + new_w / 2,
                z_min       = max(0.0, cz - new_h / 2),
                z_max       = cz + new_h / 2,
                resistivity = new_rho,
            )
        else:
            # 다각형: 비저항만 변경
            perturbed = deepcopy(anomaly)
            perturbed.resistivity = new_rho
            return perturbed

    def _random_anomaly(self, grid: Grid) -> Anomaly:
        """격자 범위 내 랜덤 이상대 생성"""
        x_min_g = grid.node_x[grid.ix_model_start, 0]
        x_max_g = grid.node_x[grid.ix_model_end,   0]
        z_min_g = 0.0
        z_max_g = grid.node_z[0, grid.iz_model_end]

        log_rho = self._rng.uniform(*self.config.resistivity_log_range)
        rho     = 10.0 ** log_rho

        # 50% 확률로 원형 또는 사각형
        if self._rng.random() < 0.5:
            cx = self._rng.uniform(x_min_g, x_max_g)
            cz = self._rng.uniform(z_min_g, z_max_g)
            r  = self._rng.uniform(
                (x_max_g - x_min_g) * 0.03,
                (x_max_g - x_min_g) * 0.15,
            )
            return CircleAnomaly(cx, cz, r, resistivity=rho)
        else:
            w  = self._rng.uniform(
                (x_max_g - x_min_g) * 0.05,
                (x_max_g - x_min_g) * 0.25,
            )
            h  = self._rng.uniform(
                (z_max_g - z_min_g) * 0.05,
                (z_max_g - z_min_g) * 0.20,
            )
            cx = self._rng.uniform(x_min_g + w / 2, x_max_g - w / 2)
            cz = self._rng.uniform(z_min_g + h / 2, z_max_g - h / 2)
            return RectangleAnomaly(
                cx - w / 2, cx + w / 2,
                cz - h / 2, cz + h / 2,
                resistivity=rho,
            )
