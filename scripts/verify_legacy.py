#!/usr/bin/env python
"""
Fortran 레거시 데이터를 이용한 Python 수치모델링 검증

model_setup/topo_001 의 격자/탐사/모델 파일을 읽어
Python ForwardModeling 으로 계산 후 output_data/topo_001/Data_001_00001.dat 과 비교.

실행:
  cd em25d_python
  python scripts/verify_legacy.py                       # 직렬, splu solver
  python scripts/verify_legacy.py --gpu                  # GPU (PyTorch CUDA)
  mpirun -np 20 python scripts/verify_legacy.py          # MPI 20 프로세스
  mpirun -np 20 python scripts/verify_legacy.py --gpu    # MPI + GPU
"""
from __future__ import annotations

import sys, time, argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from em25d.constants import PI, SourceType, MU_0, EPSILON_0
from em25d.mesh.grid import Grid
from em25d.mesh.profile import ProfileNodes
from em25d.survey.source import Source, SourceArray
from em25d.survey.receiver import Receiver, ReceiverArray
from em25d.survey.frequency import FrequencySet
from em25d.survey.survey import Survey
from em25d.forward.forward_loop import ForwardModeling, ForwardConfig
from em25d.io.legacy_io import (
    build_legacy_mesh, read_survey_dat, load_legacy_resistivity,
    read_profile_c, get_profile_global_nodes, read_fortran_output_data,
)
from em25d.io.data_io import save_synthetic_inp


class SimpleResistivityModel:
    """ForwardModeling 호환 간이 비저항 모델 (BlockPartition 불필요)"""

    def __init__(self, element_resistivity: np.ndarray, background_resistivity: float):
        self.element_resistivity = element_resistivity
        self.background_resistivity = background_resistivity


def load_legacy_setup(topo_dir: Path):
    """레거시 파일에서 Grid, Survey, Profile, Model 로딩"""

    # 1) 메시
    mesh = build_legacy_mesh(topo_dir)
    grid = Grid.from_coordinates(mesh.node_x_1d, mesh.node_z_1d)
    print(f"[Grid] {grid.n_nodes_x}x{grid.n_nodes_z} = {grid.n_nodes} nodes, "
          f"{grid.n_elements} elements")

    # 2) survey
    survey_dat = read_survey_dat(topo_dir / "survey.dat")
    itype_map = {1: SourceType.Jx, 2: SourceType.Jy, 3: SourceType.Jz,
                 4: SourceType.Mx, 5: SourceType.My, 6: SourceType.Mz}
    stype = itype_map[survey_dat.source_type]
    print(f"[Survey] {survey_dat.n_freq} freqs, {survey_dat.n_transmitters} TX, "
          f"type={stype.name}, homo_rho={survey_dat.homogeneous_resistivity}")

    # 송신기 배열
    sources = SourceArray([
        Source(x=survey_dat.source_x[i], z=survey_dat.source_z[i],
               source_type=stype, strength=1.0, length=1.0)
        for i in range(survey_dat.n_transmitters)
    ])

    # 프로파일 (수신기 위치)
    profile_x, profile_z = read_profile_c(topo_dir / "profile_c.dat")
    profile = ProfileNodes(grid, profile_x, profile_z)
    print(f"[Profile] {profile.n_receivers} receivers, "
          f"x=[{profile_x[0]:.1f}, {profile_x[-1]:.1f}], z={profile_z[0]:.1f}")

    # 수신기 배열 (프로파일 = 수신기)
    receivers = ReceiverArray([
        Receiver(x=profile_x[i], z=profile_z[i])
        for i in range(len(profile_x))
    ])

    # 주파수
    freqs = FrequencySet(survey_dat.frequencies)

    survey = Survey(sources, receivers, freqs)

    # 3) 비저항 모델
    rho = load_legacy_resistivity(topo_dir / "mproprty.dat", mesh)
    model = SimpleResistivityModel(
        element_resistivity=rho,
        background_resistivity=survey_dat.homogeneous_resistivity,
    )
    print(f"[Model] air={int((rho == 0).sum())} elems, "
          f"ground={int((rho > 0).sum())} elems, "
          f"unique_rho={sorted(set(rho.ravel()))[:5]}")

    return grid, survey, profile, model, survey_dat, mesh


def load_fortran_reference(data_path: Path, n_freq: int, n_tx: int):
    """Fortran Data_001_00001.dat 로딩 → (n_freq, n_tx, 3) complex"""
    raw, _, _ = read_fortran_output_data(data_path)
    # Columns: ifreq(1-based) itx(1-based) irx(1-based) Ey_r Ey_i Hx_r Hx_i Hz_r Hz_i
    result = {}
    result["Ey"] = np.zeros((n_freq, n_tx), dtype=complex)
    result["Hx"] = np.zeros((n_freq, n_tx), dtype=complex)
    result["Hz"] = np.zeros((n_freq, n_tx), dtype=complex)

    for row in raw:
        ifreq = int(row[0]) - 1
        itx = int(row[1]) - 1
        result["Ey"][ifreq, itx] = row[3] + 1j * row[4]
        result["Hx"][ifreq, itx] = row[5] + 1j * row[6]
        result["Hz"][ifreq, itx] = row[7] + 1j * row[8]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="GPU 가속 (PyTorch CUDA)")
    parser.add_argument("--mpi", action="store_true", help="MPI 병렬화 (자동감지)")
    parser.add_argument("--solver", default="splu",
                        choices=["splu", "direct", "gmres"],
                        help="solver 종류 (default: splu)")
    args = parser.parse_args()

    # MPI 초기화
    mpi = None
    try:
        from em25d.parallel.mpi_manager import MPIContext
        mpi_ctx = MPIContext()
        if mpi_ctx.size > 1:
            mpi = mpi_ctx
    except ImportError:
        pass

    is_root = mpi is None or mpi.is_root
    def rprint(*a, **k):
        if is_root: print(*a, **k, flush=True)

    topo_dir = DATA_ROOT / "model_setup" / "topo_001"
    model_res_dir = DATA_ROOT / "model_res"
    output_dir = DATA_ROOT / "output_data" / "topo_001"

    rprint("=" * 60)
    rprint("EM 2.5D Python vs Fortran 검증")
    if mpi: rprint(f"  MPI: {mpi.size} 프로세스")
    if args.gpu: rprint(f"  GPU: 활성화")
    rprint(f"  Solver: {args.solver}")
    rprint("=" * 60)

    # ── 레거시 데이터 로딩 (모든 프로세스가 독립적으로 로드) ─────────
    grid, survey, profile, model, survey_dat, mesh = load_legacy_setup(topo_dir)
    model_res_rho = load_legacy_resistivity(
        model_res_dir / "Model_00001.dat", mesh)
    model.element_resistivity = model_res_rho
    if is_root:
        print(f"[Model updated] from model_res/Model_00001.dat")

    # ── Python 순방향 모델링 ──────────────────────────────────────────
    config = ForwardConfig(
        n_wavenumbers=20,
        use_gpu=args.gpu,
        solver=args.solver,
        field_type="total",
        ky_resistivity=survey_dat.anomalous_resistivity,
    )
    rprint(f"\n[Config] n_ky={config.n_wavenumbers}, solver={config.solver}, "
           f"gpu={config.use_gpu}, ky_rho={config.ky_resistivity}")

    fwd = ForwardModeling(grid, model, survey, profile, config)

    rprint("\n[Forward] 계산 시작...")
    t0 = time.time()
    synthetic = fwd.run(mpi=mpi)  # (n_freq, n_tx, n_rx, 6) on root
    elapsed = time.time() - t0
    rprint(f"[Forward] 완료: {elapsed:.1f}초")

    if not is_root:
        return

    rprint(f"[Forward] 결과 shape: {synthetic.shape}")

    # ── Fortran 기준 데이터 로딩 ──────────────────────────────────────
    ref_path = output_dir / "Data_001_00001.dat"
    if not ref_path.exists():
        print(f"\n[ERROR] Fortran 출력 파일 없음: {ref_path}")
        return

    ref = load_fortran_reference(
        ref_path, survey_dat.n_freq, survey_dat.n_transmitters)
    print(f"\n[Reference] Fortran 데이터 로드 완료")

    # ── 비교 ─────────────────────────────────────────────────────────
    # survey.dat: 각 TX는 1개 RX → ReceiverNode는 프로파일 인덱스
    # TX i → RX = profile point (receiver_nodes_fortran[i][0] - 1)
    print("\n" + "=" * 60)
    print("결과 비교 (Python vs Fortran)")
    print("=" * 60)

    # synthetic: (n_freq, n_tx, n_rx, 6) [Ex, Ey, Ez, Hx, Hy, Hz]
    comp_idx = {"Ey": 1, "Hx": 3, "Hz": 5}

    for comp_name in ["Ey", "Hx", "Hz"]:
        ic = comp_idx[comp_name]
        print(f"\n--- {comp_name} ---")
        print(f"{'Freq':>8s}  {'TX':>4s}  {'Python':>22s}  {'Fortran':>22s}  {'RelErr':>10s}")

        total_err = 0
        n_pts = 0
        for ifreq in range(min(survey_dat.n_freq, 3)):  # 처음 3개 주파수만
            for itx in range(min(survey_dat.n_transmitters, 5)):  # 처음 5개 TX
                # RX 인덱스 (프로파일 내)
                irx = survey_dat.receiver_nodes_fortran[itx][0] - 1
                py_val = synthetic[ifreq, itx, irx, ic]
                ft_val = ref[comp_name][ifreq, itx]

                if abs(ft_val) > 1e-30:
                    rel_err = abs(py_val - ft_val) / abs(ft_val) * 100
                else:
                    rel_err = float('nan')

                total_err += rel_err if not np.isnan(rel_err) else 0
                n_pts += 1

                print(f"{survey_dat.frequencies[ifreq]:8.0f}  "
                      f"{itx+1:4d}  "
                      f"{py_val.real:+10.4e}{py_val.imag:+10.4e}j  "
                      f"{ft_val.real:+10.4e}{ft_val.imag:+10.4e}j  "
                      f"{rel_err:8.2f}%")

        avg_err = total_err / n_pts if n_pts > 0 else 0
        print(f"  평균 상대오차: {avg_err:.2f}%")

    # 전체 오차 요약
    print("\n" + "=" * 60)
    print("전체 오차 요약")
    print("=" * 60)
    for comp_name in ["Ey", "Hx", "Hz"]:
        ic = comp_idx[comp_name]
        errs = []
        for ifreq in range(survey_dat.n_freq):
            for itx in range(survey_dat.n_transmitters):
                irx = survey_dat.receiver_nodes_fortran[itx][0] - 1
                py_val = synthetic[ifreq, itx, irx, ic]
                ft_val = ref[comp_name][ifreq, itx]
                if abs(ft_val) > 1e-30:
                    errs.append(abs(py_val - ft_val) / abs(ft_val) * 100)
        errs = np.array(errs)
        print(f"  {comp_name}: mean={errs.mean():.2f}%, "
              f"max={errs.max():.2f}%, median={np.median(errs):.2f}%")

    # 결과 저장 (NPZ)
    out_path = DATA_ROOT / "output_data" / "python" / "verify_result.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        synthetic=synthetic,
        frequencies=survey_dat.frequencies,
    )
    print(f"\n[Save] NPZ 저장: {out_path}")

    # .inp 파일 생성 (역산 검증용)
    inp_path = DATA_ROOT / "output_data" / "python" / "Fem25Dinv_verify.inp"
    use_comp = [False, True, False, True, False, True]  # Ey, Hx, Hz
    prof_x, prof_z = read_profile_c(topo_dir / "profile_c.dat")
    save_synthetic_inp(
        synthetic=synthetic,
        frequencies=survey_dat.frequencies,
        source_x=survey_dat.source_x,
        source_z=survey_dat.source_z,
        receiver_x=prof_x,
        receiver_z=prof_z,
        path=inp_path,
        use_components=use_comp,
    )
    print(f"[Save] .inp 저장: {inp_path}")


if __name__ == "__main__":
    main()
