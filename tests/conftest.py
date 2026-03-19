"""
pytest 공용 픽스처 (fixture)

모든 테스트 모듈에서 공유하는 최소 크기 격자/모델/탐사배열 제공.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# 패키지 경로 (em25d_python/)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from em25d.mesh.grid import Grid, GridConfig
from em25d.mesh.block import BlockPartition, BlockConfig
from em25d.mesh.profile import ProfileNodes, ProfileConfig
from em25d.model.resistivity import ResistivityModel
from em25d.survey.source import SourceArray, Source
from em25d.survey.receiver import ReceiverArray
from em25d.survey.frequency import FrequencySet
from em25d.survey.survey import Survey
from em25d.constants import SourceType


# ── 최소 크기 격자 픽스처 ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def small_grid():
    """테스트용 소형 격자 (10×10 모델 + 3 공기층)"""
    cfg = GridConfig(
        n_x_cells=10,
        n_z_cells=10,
        n_z_cells_air=3,
        base_x_cell_size=10.0,
        base_z_cell_size=10.0,
        n_x_boundary_cells=5,
        n_z_boundary_bottom_cells=5,
        halfspace_resistivity=100.0,
    )
    return Grid(cfg)


@pytest.fixture(scope="session")
def small_block_partition(small_grid):
    cfg = BlockConfig(n_blocks_x=5, n_blocks_z=5)
    return BlockPartition(small_grid, cfg)


@pytest.fixture(scope="session")
def homogeneous_model(small_grid, small_block_partition):
    """균질 100 Ω·m 모델"""
    return ResistivityModel(
        small_grid, small_block_partition, background_resistivity=100.0
    )


@pytest.fixture(scope="session")
def small_profile(small_grid):
    cfg = ProfileConfig(
        n_receivers=5,
        x_start=-20.0,
        x_end=20.0,
        surface_z=0.0,
    )
    return ProfileNodes(small_grid, cfg)


@pytest.fixture(scope="session")
def small_survey(small_profile):
    """Jy 단일 송신기, 3개 주파수"""
    source = Source(x=0.0, z=0.0, source_type=SourceType.Jy, strength=1.0)
    sources = SourceArray([source])
    receivers = ReceiverArray.from_profile(small_profile, measured_components=["Hy"])
    freqs = FrequencySet([1.0, 10.0, 100.0])
    return Survey(sources, receivers, freqs)
