"""
역산 실행 스크립트

Fortran 대응: Fem25Dinv.f90 + Fem25Dinv.par (Inversion 블록)

사용법:
    python scripts/run_inversion.py --config config/default_params.yaml \\
           --observed output_data/synthetic_data.npz

    # MPI 병렬
    mpirun -n 4 python scripts/run_inversion.py \\
           --config config/default_params.yaml \\
           --observed output_data/field_data.npz

역산 결과는 inversion_log/ 디렉토리에 저장됩니다.
"""

from __future__ import annotations

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from em25d.mesh.grid import Grid, GridConfig
from em25d.mesh.block import BlockPartition, BlockConfig
from em25d.mesh.profile import ProfileNodes, ProfileConfig
from em25d.model.resistivity import ResistivityModel
from em25d.survey.source import SourceArray, Source
from em25d.survey.receiver import ReceiverArray
from em25d.survey.frequency import FrequencySet
from em25d.survey.survey import Survey
from em25d.forward.forward_loop import ForwardConfig
from em25d.inverse.inversion_loop import (
    InversionConfig, InversionModeling, run_inversion
)
from em25d.constants import SourceType, NormType


def parse_args():
    parser = argparse.ArgumentParser(
        description="EM 2.5D 역산",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="YAML 파라미터 파일")
    parser.add_argument("--observed", "-d", type=str, required=True,
                        help="관측 데이터 파일 (.npz 또는 텍스트)")
    parser.add_argument("--restart", action="store_true",
                        help="이전 역산에서 재시작 (last_model.dat 사용)")
    parser.add_argument("--max-iter", type=int, default=None,
                        help="최대 반복 횟수 (YAML 설정 override)")
    parser.add_argument("--gpu", action="store_true",
                        help="GPU 가속 사용")
    parser.add_argument("--log-dir", type=str, default="./inversion_log",
                        help="역산 로그 저장 디렉토리")
    return parser.parse_args()


def load_observed(path: str) -> np.ndarray:
    """관측 데이터 로드 (.npz 또는 Fortran 텍스트 포맷)"""
    p = Path(path)
    if not p.exists():
        print(f"[오류] 관측 데이터 파일 없음: {path}")
        sys.exit(1)

    if p.suffix == ".npz":
        d = np.load(p, allow_pickle=False)
        return d["data"]
    else:
        # Fortran 텍스트 포맷 (레거시)
        from em25d.io.legacy_io import load_fortran_data
        return load_fortran_data(str(p))


def build_model_from_config(cfg: dict):
    """YAML에서 격자/모델/탐사 배열 구성"""
    mesh_cfg_dict = cfg.get("mesh", {})
    grid_cfg = GridConfig(**{
        k: v for k, v in mesh_cfg_dict.items()
        if k in GridConfig.__dataclass_fields__
    })
    grid = Grid(grid_cfg)

    inv_cfg_dict = cfg.get("inversion", {})
    block_cfg = BlockConfig(
        n_blocks_x=inv_cfg_dict.get("n_x_blocks", grid_cfg.n_x_cells // 2),
        n_blocks_z=inv_cfg_dict.get("n_z_blocks", grid_cfg.n_z_cells // 2),
    )
    block_partition = BlockPartition(grid, block_cfg)

    bg_rho = cfg.get("forward", {}).get("background_resistivity", 100.0)
    model = ResistivityModel(
        grid=grid,
        block_partition=block_partition,
        background_resistivity=bg_rho,
    )

    profile_cfg = ProfileConfig(
        n_receivers=cfg.get("receivers", {}).get("n_receivers", 21),
        x_start=cfg.get("receivers", {}).get("x_start", -100.0),
        x_end=cfg.get("receivers", {}).get("x_end", 100.0),
        surface_z=0.0,
    )
    profile = ProfileNodes(grid, profile_cfg)

    fwd_cfg_dict = cfg.get("forward", {})
    src_type_map = {
        1: SourceType.Jx, 2: SourceType.Jy, 3: SourceType.Jz,
        4: SourceType.Mx, 5: SourceType.My, 6: SourceType.Mz,
    }
    src_type = src_type_map.get(fwd_cfg_dict.get("source_type", 2), SourceType.Jy)
    source = Source(
        x=fwd_cfg_dict.get("source_x", 0.0),
        z=fwd_cfg_dict.get("source_z", 0.0),
        source_type=src_type,
        strength=fwd_cfg_dict.get("source_strength", 1.0),
        length=fwd_cfg_dict.get("source_length", 0.0),
    )
    sources = SourceArray([source])

    components = inv_cfg_dict.get("field_components", ["Hy"])
    receivers = ReceiverArray.from_profile(profile, measured_components=components)

    freqs = FrequencySet(cfg.get("frequencies", [1.0, 10.0, 100.0]))
    survey = Survey(sources, receivers, freqs)

    return grid, model, survey, profile


def build_inversion_config(cfg: dict, args) -> InversionConfig:
    """YAML + 커맨드라인 args → InversionConfig"""
    inv_dict = cfg.get("inversion", {})

    norm_map = {
        "l1": NormType.L1, "l2": NormType.L2,
        "huber": NormType.Huber, "ekblom": NormType.Ekblom,
    }
    norm_str = inv_dict.get("norm_type", "l2").lower()
    norm = norm_map.get(norm_str, NormType.L2)

    return InversionConfig(
        max_iterations=args.max_iter or inv_dict.get("max_iterations", 10),
        damping_factor=inv_dict.get("damping_factor", 1.0),
        resistivity_min=inv_dict.get("resistivity_min", 0.1),
        resistivity_max=inv_dict.get("resistivity_max", 1e5),
        irls_data_norm=norm,
        irls_model_norm=norm,
        use_acb=inv_dict.get("use_acb", True),
        use_occam=inv_dict.get("use_occam", True),
        log_dir=args.log_dir,
    )


def main():
    args = parse_args()

    # ── 설정 로드 ──────────────────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[오류] 파라미터 파일 없음: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"run_{run_ts}"

    print("=" * 60)
    print(f"  EM 2.5D 역산  [{config_path.name}]")
    print(f"  시작: {run_ts}")
    print("=" * 60)

    # ── 모델 구성 ──────────────────────────────────────────────────────────
    grid, model, survey, profile = build_model_from_config(cfg)

    # ── 재시작 모드: 이전 역산 모델 로드 ──────────────────────────────────
    if args.restart:
        last_model_path = Path(args.log_dir) / "last_model.dat"
        if last_model_path.exists():
            model.from_file(str(last_model_path))
            print(f"[재시작] 모델 로드: {last_model_path}")
        else:
            print(f"[경고] 재시작 모델 파일 없음: {last_model_path}, 초기 모델 사용")

    # ── 관측 데이터 로드 ──────────────────────────────────────────────────
    observed = load_observed(args.observed)
    print(f"관측 데이터: {observed.shape}  (n_freq, n_tx, n_rx, 6성분)")

    # ── 역산 설정 ─────────────────────────────────────────────────────────
    inv_config = build_inversion_config(cfg, args)
    inv_config.log_dir = str(log_dir)

    fwd_config = ForwardConfig(
        n_wavenumbers=cfg.get("forward", {}).get("n_wavenumbers", 16),
        use_gpu=args.gpu,
        output_dir=cfg.get("forward", {}).get("output_dir", "./output_data"),
        primary_dir=cfg.get("forward", {}).get("primary_dir", "./output_primary"),
    )

    # 설정 요약
    print(f"  격자      : {grid.n_nodes_x} × {grid.n_nodes_z} 노드")
    print(f"  블록      : {model.n_blocks}개")
    print(f"  주파수    : {survey.frequencies.frequencies} Hz")
    print(f"  최대 반복 : {inv_config.max_iterations}회")
    print(f"  노름      : {inv_config.irls_data_norm.name}")
    print(f"  ACB       : {inv_config.use_acb}")
    print(f"  로그 경로 : {log_dir}")
    print()

    # ── 역산 실행 ─────────────────────────────────────────────────────────
    t0 = time.time()
    print("역산 시작...")

    result = run_inversion(
        grid=grid,
        model=model,
        survey=survey,
        profile=profile,
        observed=observed,
        inv_config=inv_config,
        fwd_config=fwd_config,
    )

    elapsed = time.time() - t0
    print(f"\n역산 완료 (경과: {elapsed:.1f}초)")
    print(f"  최종 반복 수  : {result.n_iterations}")
    print(f"  최종 RMS 오차 : {result.final_rms:.4f}")
    print(f"  로그 저장     : {log_dir}")

    # 최종 모델 저장 (재시작용)
    last_path = Path(args.log_dir) / "last_model.dat"
    result.final_model.to_file(str(last_path))
    print(f"  최종 모델     : {last_path}")


if __name__ == "__main__":
    main()
