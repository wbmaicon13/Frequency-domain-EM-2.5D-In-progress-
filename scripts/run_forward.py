"""
순방향 모델링 실행 스크립트

Fortran 대응: Fem25Dfwd.f90 + Fem25Dinv.par (Forward 블록)

사용법:
    # 단일 프로세스
    python scripts/run_forward.py --config config/default_params.yaml

    # MPI 병렬 (ky 분배)
    mpirun -n 4 python scripts/run_forward.py --config config/default_params.yaml

    # GPU 가속
    python scripts/run_forward.py --config config/default_params.yaml --gpu

파라미터 파일 없이 기본값 사용:
    python scripts/run_forward.py --demo
"""

from __future__ import annotations

import sys
import argparse
import time
from pathlib import Path

# 패키지 루트 경로 추가 (scripts/ 에서 실행하는 경우)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from em25d.mesh.grid import Grid, GridConfig
from em25d.mesh.block import BlockPartition, BlockConfig
from em25d.mesh.profile import ProfileNodes, ProfileConfig
from em25d.model.resistivity import ResistivityModel
from em25d.model.anomaly import RectangleAnomaly, apply_anomalies
from em25d.survey.source import SourceArray, Source
from em25d.survey.receiver import ReceiverArray, Receiver
from em25d.survey.frequency import FrequencySet
from em25d.survey.survey import Survey
from em25d.forward.forward_loop import ForwardModeling, ForwardConfig, run_forward
from em25d.io.params import load_params
from em25d.io.data_io import save_synthetic_data
from em25d.constants import SourceType


def parse_args():
    parser = argparse.ArgumentParser(
        description="EM 2.5D 순방향 모델링",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="YAML 파라미터 파일 경로 (Fem25Dinv.par 대체)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="데모 모드: 기본값으로 간단한 순방향 계산 실행",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="GPU 가속 사용 (CuPy 필요)",
    )
    parser.add_argument(
        "--n-ky", "-k",
        type=int,
        default=None,
        help="공간주파수(ky) 개수 (YAML 설정 override)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="결과 저장 디렉토리 (YAML 설정 override)",
    )
    parser.add_argument(
        "--mpi",
        action="store_true",
        help="MPI 병렬 모드 활성화",
    )
    return parser.parse_args()


def build_demo_model():
    """데모용 단순 모델 구성"""
    print("[Demo] 격자 및 모델 구성 중...")

    # 격자 설정
    grid_cfg = GridConfig(
        n_x_cells=40,
        n_z_cells=20,
        n_z_cells_air=4,
        base_x_cell_size=10.0,
        base_z_cell_size=10.0,
        n_x_boundary_cells=8,
        n_z_boundary_bottom_cells=8,
        halfspace_resistivity=100.0,
    )
    grid = Grid(grid_cfg)

    # 역산 블록 파티션
    block_cfg = BlockConfig(n_blocks_x=20, n_blocks_z=10)
    block_partition = BlockPartition(grid, block_cfg)

    # 비저항 모델 (균질 100 Ω·m + 저비저항 이상대)
    model = ResistivityModel(
        grid=grid,
        block_partition=block_partition,
        background_resistivity=100.0,
    )
    anomaly = RectangleAnomaly(
        x_min=-30.0, x_max=30.0,   # 모델 중심 ±30 m
        z_min=30.0,  z_max=70.0,   # 깊이 30~70 m
        resistivity=10.0,           # 10 Ω·m (저비저항체)
    )
    apply_anomalies(model, [anomaly])

    # 프로파일 노드 (수신기 배열)
    profile_cfg = ProfileConfig(
        n_receivers=21,
        x_start=-100.0,
        x_end=100.0,
        surface_z=0.0,
    )
    profile = ProfileNodes(grid, profile_cfg)

    # 송신기 (중심 위치, Jy 다이폴)
    source = Source(
        x=0.0, z=0.0,
        source_type=SourceType.Jy,
        strength=1.0,
        length=0.0,   # 다이폴
    )
    sources = SourceArray([source])

    # 수신기 배열 (프로파일과 동일 위치)
    receivers = ReceiverArray.from_profile(profile, measured_components=["Hy"])

    # 주파수
    freqs = FrequencySet([1.0, 10.0, 100.0])

    survey = Survey(sources, receivers, freqs)

    return grid, model, survey, profile


def build_model_from_config(cfg: dict):
    """YAML 설정으로 모델 구성"""
    mesh_cfg_dict = cfg.get("mesh", {})
    grid_cfg = GridConfig(**{
        k: v for k, v in mesh_cfg_dict.items()
        if k in GridConfig.__dataclass_fields__
    })
    grid = Grid(grid_cfg)

    block_cfg = BlockConfig(
        n_blocks_x=cfg.get("inversion", {}).get("n_x_blocks", grid_cfg.n_x_cells // 2),
        n_blocks_z=cfg.get("inversion", {}).get("n_z_blocks", grid_cfg.n_z_cells // 2),
    )
    block_partition = BlockPartition(grid, block_cfg)

    bg_rho = cfg.get("forward", {}).get("background_resistivity", 100.0)
    model = ResistivityModel(
        grid=grid,
        block_partition=block_partition,
        background_resistivity=bg_rho,
    )

    # 프로파일 (수신기 위치)
    profile_cfg = ProfileConfig(
        n_receivers=cfg.get("receivers", {}).get("n_receivers", 21),
        x_start=cfg.get("receivers", {}).get("x_start", -100.0),
        x_end=cfg.get("receivers", {}).get("x_end", 100.0),
        surface_z=0.0,
    )
    profile = ProfileNodes(grid, profile_cfg)

    # 송신기
    fwd_cfg = cfg.get("forward", {})
    src_type_map = {
        1: SourceType.Jx, 2: SourceType.Jy, 3: SourceType.Jz,
        4: SourceType.Mx, 5: SourceType.My, 6: SourceType.Mz,
    }
    src_type = src_type_map.get(fwd_cfg.get("source_type", 2), SourceType.Jy)
    source = Source(
        x=fwd_cfg.get("source_x", 0.0),
        z=fwd_cfg.get("source_z", 0.0),
        source_type=src_type,
        strength=fwd_cfg.get("source_strength", 1.0),
        length=fwd_cfg.get("source_length", 0.0),
    )
    sources = SourceArray([source])

    # 수신기
    components = cfg.get("inversion", {}).get("field_components", ["Hy"])
    receivers = ReceiverArray.from_profile(profile, measured_components=components)

    # 주파수
    freqs = FrequencySet(cfg.get("frequencies", [1.0, 10.0, 100.0]))

    survey = Survey(sources, receivers, freqs)

    return grid, model, survey, profile


def main():
    args = parse_args()

    # ── 설정 로드 ──────────────────────────────────────────────────────────
    if args.demo or args.config is None:
        print("=" * 60)
        print("  EM 2.5D 순방향 모델링  [데모 모드]")
        print("=" * 60)
        grid, model, survey, profile = build_demo_model()
        n_ky = args.n_ky or 12
        output_dir = args.output or "./output_data"
        primary_dir = "./output_primary"
    else:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"[오류] 파라미터 파일을 찾을 수 없습니다: {config_path}")
            sys.exit(1)

        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        print("=" * 60)
        print(f"  EM 2.5D 순방향 모델링  [{config_path.name}]")
        print("=" * 60)

        grid, model, survey, profile = build_model_from_config(cfg)
        fwd_cfg = cfg.get("forward", {})
        n_ky       = args.n_ky or fwd_cfg.get("n_wavenumbers", 16)
        output_dir = args.output or fwd_cfg.get("output_dir", "./output_data")
        primary_dir = fwd_cfg.get("primary_dir", "./output_primary")

    # ── 설정 요약 출력 ─────────────────────────────────────────────────────
    print(f"  격자      : {grid.n_nodes_x} × {grid.n_nodes_z} 노드 "
          f"({grid.n_elements_x} × {grid.n_elements_z} 요소)")
    print(f"  주파수    : {survey.frequencies.frequencies} Hz")
    print(f"  송신기    : {survey.sources.n_sources}개")
    print(f"  수신기    : {survey.receivers.n_receivers}개")
    print(f"  ky 개수   : {n_ky}")
    print(f"  GPU 사용  : {args.gpu}")
    print(f"  출력 경로 : {output_dir}")
    print()

    # ── MPI 모드 ─────────────────────────────────────────────────────────
    if args.mpi:
        from em25d.parallel.mpi_manager import run_forward_mpi, get_mpi_context
        mpi = get_mpi_context()

        fwd_config = ForwardConfig(
            n_wavenumbers=n_ky,
            use_gpu=args.gpu,
            output_dir=output_dir,
            primary_dir=primary_dir,
        )

        t0 = time.time()
        mpi.print_root("순방향 모델링 시작 (MPI 모드)...")
        result = run_forward_mpi(grid, model, survey, profile, fwd_config, mpi)

        if mpi.is_root:
            elapsed = time.time() - t0
            print(f"\n완료 (경과: {elapsed:.1f}초)")
            _save_and_report(result, output_dir, survey)
        return

    # ── 단일 프로세스 모드 ────────────────────────────────────────────────
    fwd_config = ForwardConfig(
        n_wavenumbers=n_ky,
        use_gpu=args.gpu,
        output_dir=output_dir,
        primary_dir=primary_dir,
    )

    t0 = time.time()
    print("순방향 모델링 시작...")
    result = run_forward(grid, model, survey, profile, fwd_config)
    elapsed = time.time() - t0

    print(f"\n완료 (경과: {elapsed:.1f}초)")
    _save_and_report(result, output_dir, survey)


def _save_and_report(result: np.ndarray, output_dir: str, survey: "Survey"):
    """결과 저장 및 요약 출력"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "synthetic_data.npz"
    np.savez_compressed(
        out_path,
        data=result,
        frequencies=survey.frequencies.frequencies,
    )
    print(f"결과 저장: {out_path}")
    print(f"  결과 형태: {result.shape}  (n_freq, n_tx, n_rx, 6성분)")
    print(f"  |Hy| 최대값: {np.abs(result[..., 4]).max():.4e}")


if __name__ == "__main__":
    main()
