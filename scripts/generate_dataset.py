"""
딥러닝 학습용 EM 2.5D 데이터셋 생성 스크립트

요구사항 3-3 대응:
  - 기본 모델을 GUI 또는 코드로 설정한 뒤 무작위 변형 모델 생성
  - 각 모델에 대해 순방향 모델링 수행 → (model, data) 쌍 저장
  - 최대 200개 모델 생성
  - 진행률 표시 및 그림 저장 옵션

사용법:
    # 데모 모드 (기본 설정, 10개 모델)
    python scripts/generate_dataset.py --demo --n-models 10

    # 설정 파일 기반, 50개 생성, 자유도 0.5
    python scripts/generate_dataset.py \\
        --config config/default_params.yaml \\
        --n-models 50 \\
        --freedom 0.5 \\
        --output-dir datasets/run_001 \\
        --save-figures

    # GUI 모드로 기본 모델 편집 후 생성
    python scripts/generate_dataset.py --gui --n-models 30

출력 구조:
    output_dir/
    ├── models/
    │   ├── model_000001.npz    # 비저항 모델
    │   ├── model_000002.npz
    │   └── ...
    ├── data/
    │   ├── data_000001.npz     # 순방향 모델링 결과
    │   └── ...
    ├── figures/                # 모델 그림 (--save-figures 옵션)
    │   ├── model_000001.png
    │   └── ...
    ├── metadata.yaml           # 생성 설정 정보
    └── dataset.npz             # 전체 데이터셋 (numpy 배열)
"""

from __future__ import annotations

import sys
import argparse
import time
import yaml
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from em25d.mesh.grid import Grid, GridConfig
from em25d.mesh.block import BlockPartition, BlockConfig
from em25d.mesh.profile import ProfileNodes, ProfileConfig
from em25d.model.resistivity import ResistivityModel
from em25d.model.anomaly import (
    RectangleAnomaly, CircleAnomaly, apply_anomalies
)
from em25d.model.generator import ModelGenerator, GeneratorConfig
from em25d.model.visualize import plot_resistivity_model
from em25d.survey.source import SourceArray, Source
from em25d.survey.receiver import ReceiverArray
from em25d.survey.frequency import FrequencySet
from em25d.survey.survey import Survey
from em25d.forward.forward_loop import ForwardConfig, run_forward
from em25d.constants import SourceType


def parse_args():
    parser = argparse.ArgumentParser(
        description="딥러닝 학습용 EM 2.5D 데이터셋 생성",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="YAML 파라미터 파일")
    parser.add_argument("--demo", action="store_true",
                        help="데모 모드 (기본 모델 자동 구성)")
    parser.add_argument("--gui", action="store_true",
                        help="GUI 모드로 기본 모델 편집")
    parser.add_argument("--n-models", "-n", type=int, default=10,
                        help="생성할 모델 수 (최대 200)")
    parser.add_argument("--freedom", "-f", type=float, default=0.3,
                        help="모델 변형 자유도 (0=원본 유지, 1=최대 변형)")
    parser.add_argument("--seed", type=int, default=42,
                        help="랜덤 시드 (재현성)")
    parser.add_argument("--n-ky", type=int, default=12,
                        help="공간주파수(ky) 개수")
    parser.add_argument("--gpu", action="store_true",
                        help="GPU 가속 사용")
    parser.add_argument("--output-dir", "-o", type=str,
                        default="./datasets",
                        help="결과 저장 디렉토리")
    parser.add_argument("--save-figures", action="store_true",
                        help="모델 그림 저장")
    parser.add_argument("--no-save-dataset", action="store_true",
                        help="통합 dataset.npz 저장 생략 (개별 파일만)")
    return parser.parse_args()


def build_demo_setup():
    """데모용 기본 격자/탐사 배열 반환"""
    grid_cfg = GridConfig(
        n_x_cells=40, n_z_cells=20, n_z_cells_air=4,
        base_x_cell_size=10.0, base_z_cell_size=10.0,
        n_x_boundary_cells=8, n_z_boundary_bottom_cells=8,
        halfspace_resistivity=100.0,
    )
    grid = Grid(grid_cfg)
    block_cfg = BlockConfig(n_blocks_x=20, n_blocks_z=10)
    block_partition = BlockPartition(grid, block_cfg)

    base_model = ResistivityModel(grid, block_partition, background_resistivity=100.0)

    # 기본 이상대: 저비저항 사각형
    base_anomalies = [
        RectangleAnomaly(
            center_x=0.0, center_z=50.0,
            half_width=30.0, half_depth=20.0,
            resistivity=10.0,
        )
    ]

    # 탐사 배열
    profile_cfg = ProfileConfig(n_receivers=21, x_start=-100.0, x_end=100.0)
    profile = ProfileNodes(grid, profile_cfg)

    source = Source(x=0.0, z=0.0, source_type=SourceType.Jy, strength=1.0)
    sources = SourceArray([source])
    receivers = ReceiverArray.from_profile(profile, measured_components=["Hy"])
    freqs = FrequencySet([1.0, 10.0, 100.0])
    survey = Survey(sources, receivers, freqs)

    return grid, base_model, base_anomalies, survey, profile, block_partition


def launch_gui(grid, base_model, base_anomalies, survey, profile):
    """GUI 편집기 실행 후 수정된 모델/이상대 반환"""
    try:
        from em25d.gui.model_editor import ModelEditor
        editor = ModelEditor(grid, base_model, base_anomalies, survey, profile)
        updated_model, updated_anomalies = editor.run()
        return updated_model, updated_anomalies
    except ImportError as e:
        print(f"[경고] GUI 모듈 로드 실패: {e}")
        print("  → 코드 기본 모델을 사용합니다.")
        return base_model, base_anomalies


def main():
    args = parse_args()
    n_models = min(args.n_models, 200)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"dataset_{run_ts}"

    print("=" * 60)
    print("  EM 2.5D 딥러닝 데이터셋 생성")
    print(f"  생성 수  : {n_models} (최대 200)")
    print(f"  자유도   : {args.freedom}")
    print(f"  시드     : {args.seed}")
    print(f"  출력     : {output_dir}")
    print("=" * 60)

    # ── 기본 모델 구성 ─────────────────────────────────────────────────────
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        # 설정 파일 기반 구성은 run_forward.py와 동일한 로직 사용
        # (간결성을 위해 demo 모드로 대체 가능)
        print("[정보] 설정 파일 기반 모델 구성 (현재 버전은 demo 격자 사용)")
        grid, base_model, base_anomalies, survey, profile, block_partition = \
            build_demo_setup()
    else:
        grid, base_model, base_anomalies, survey, profile, block_partition = \
            build_demo_setup()

    # ── GUI 모드 ──────────────────────────────────────────────────────────
    if args.gui:
        print("\n[GUI] 모델 편집기를 시작합니다...")
        base_model, base_anomalies = launch_gui(
            grid, base_model, base_anomalies, survey, profile)
        print("[GUI] 모델 편집 완료.")

    # ── 기본 이상대 적용 ──────────────────────────────────────────────────
    apply_anomalies(base_model, base_anomalies)

    # ── 모델 생성 ─────────────────────────────────────────────────────────
    gen_config = GeneratorConfig(
        n_models=n_models,
        freedom=args.freedom,
        seed=args.seed,
    )
    generator = ModelGenerator(base_model, base_anomalies, gen_config)
    models = generator.generate()   # List[ResistivityModel]

    print(f"\n{len(models)}개 모델 생성 완료. 순방향 계산 시작...\n")

    # ── 출력 디렉토리 생성 ────────────────────────────────────────────────
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    if args.save_figures:
        (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    # ── 순방향 모델링 ─────────────────────────────────────────────────────
    fwd_config = ForwardConfig(
        n_wavenumbers=args.n_ky,
        use_gpu=args.gpu,
        output_dir=str(output_dir / "data"),
        primary_dir=str(output_dir / "primary"),
    )

    all_models_rho = []   # (n_models, n_elements_x, n_elements_z)
    all_data = []         # (n_models, n_freq, n_tx, n_rx, 6)

    t_total = time.time()
    for i, mdl in enumerate(models):
        idx = i + 1
        t0 = time.time()

        # 순방향 계산
        synthetic = run_forward(grid, mdl, survey, profile, fwd_config)
        elapsed = time.time() - t0

        # 결과 저장
        model_path = output_dir / "models" / f"model_{idx:06d}.npz"
        data_path  = output_dir / "data"   / f"data_{idx:06d}.npz"

        np.savez_compressed(model_path, resistivity=mdl.element_resistivity)
        np.savez_compressed(
            data_path,
            data=synthetic,
            frequencies=survey.frequencies.frequencies,
        )

        all_models_rho.append(mdl.element_resistivity.copy())
        all_data.append(synthetic)

        # 그림 저장
        if args.save_figures:
            fig_path = output_dir / "figures" / f"model_{idx:06d}.png"
            plot_resistivity_model(
                grid, mdl,
                title=f"Model {idx:06d}",
                save_path=str(fig_path),
                show=False,
            )

        # 진행률
        pct = idx / n_models * 100
        bar_len = 30
        filled = int(bar_len * idx / n_models)
        bar = "█" * filled + "░" * (bar_len - filled)
        eta = (time.time() - t_total) / idx * (n_models - idx)
        print(f"\r  [{bar}] {pct:5.1f}%  "
              f"({idx}/{n_models})  {elapsed:.1f}s/모델  ETA {eta:.0f}s",
              end="", flush=True)

    print()  # 줄바꿈

    # ── 통합 데이터셋 저장 ────────────────────────────────────────────────
    if not args.no_save_dataset:
        dataset_path = output_dir / "dataset.npz"
        np.savez_compressed(
            dataset_path,
            resistivity=np.stack(all_models_rho, axis=0),     # (N, nx, nz)
            data=np.stack(all_data, axis=0),                   # (N, nf, nt, nr, 6)
            frequencies=survey.frequencies.frequencies,
        )
        print(f"\n통합 데이터셋 저장: {dataset_path}")
        print(f"  resistivity shape : {np.stack(all_models_rho, axis=0).shape}")
        print(f"  data shape        : {np.stack(all_data, axis=0).shape}")

    # ── 메타데이터 저장 ──────────────────────────────────────────────────
    metadata = {
        "generated_at": run_ts,
        "n_models": n_models,
        "freedom": args.freedom,
        "seed": args.seed,
        "n_ky": args.n_ky,
        "frequencies": survey.frequencies.frequencies.tolist(),
        "n_receivers": survey.receivers.n_receivers,
        "grid": {
            "n_nodes_x": grid.n_nodes_x,
            "n_nodes_z": grid.n_nodes_z,
            "n_elements_x": grid.n_elements_x,
            "n_elements_z": grid.n_elements_z,
        },
    }
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, allow_unicode=True)

    total_time = time.time() - t_total
    print(f"\n완료! 총 소요: {total_time:.1f}초 "
          f"(평균 {total_time / n_models:.1f}초/모델)")
    print(f"저장 경로: {output_dir}")


if __name__ == "__main__":
    main()
