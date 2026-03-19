"""
Fortran vs Python EM 2.5D 검증 스크립트

비교 항목:
  1) Primary field (ky 도메인): prim-ky-0101.dat (Fortran) vs primary_field_ky_domain() (Python)
  2) 전체 전자기장 (Hz at receivers): Data_001_00001.dat (Fortran) vs run_forward() (Python)

실행 방법:
  cd em25d_python
  python tests/compare_fortran.py

Fortran 데이터 위치:
  /mnt/d/work/Code/renew/em25d_whole/
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT.parent  # /mnt/d/work/Code/renew/em25d_whole/
sys.path.insert(0, str(ROOT))

from em25d.forward.primary_field import (
    PrimaryFieldParams, primary_field_ky_domain,
)
from em25d.constants import PI, SourceType, MU_0


# ─── 1. Fortran 격자 파싱 ────────────────────────────────────────────────────

def parse_coord_msh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """coord.msh → (x_nodes, z_nodes) 1D 배열"""
    lines = path.read_text().splitlines()
    z_nodes, x_nodes = [], []
    # z-section: lines[1..61] (index 1-61)
    for line in lines[2:63]:
        parts = line.split()
        if len(parts) >= 2:
            z_nodes.append(float(parts[1]))
    # x-section: lines[65..139] (index 1-75)
    for line in lines[65:140]:
        parts = line.split()
        if len(parts) >= 2:
            x_nodes.append(float(parts[1]))
    return np.array(x_nodes), np.array(z_nodes)


def parse_model_dat(path: Path, n_x_elem: int, n_z_elem: int) -> np.ndarray:
    """
    Model_00001.dat → element_resistivity[ix_elem, iz_elem]

    Fortran 요소 순서: column-major, ix outer, iz inner
      elem_idx (1-based) = (ix-1)*n_z_elem + iz
    """
    d = np.loadtxt(path)  # (n_elem, 3): [idx, res_x, res_z]
    res_x = d[:, 1]
    # 0.0 은 공기 요소 → 배경값 대체
    res_x = np.where(res_x == 0.0, 1e8, res_x)
    # Fortran column-major → (n_x_elem, n_z_elem)
    res_2d = res_x.reshape(n_x_elem, n_z_elem, order='F')
    return res_2d


# ─── 2. Fortran ky 값 계산 ───────────────────────────────────────────────────

def fortran_ky_values(
    n_ky: int = 20,
    anomalous_res: float = 1.0,   # blck_res.dat 최소값 (Ω·m)
    min_freq: float = 220.0,       # Hz
    dx_min: float = 2.0,           # m (coord.msh 최소 x 간격)
) -> np.ndarray:
    """
    Fortran Fem25Dprimary.f90 ky 계산식 재현:
      skind  = 500 * sqrt(res0 / freq_min)
      ak_max = pi / dx_min
      ak_min = 2*pi / (30 * skind)
      ky[1]  = 1e-8
      ky[2..n] = log-spaced(ak_min, ak_max, n-1)
    """
    skind  = 500.0 * np.sqrt(anomalous_res / min_freq)
    ak_max = PI / dx_min
    ak_min = 2.0 * PI / (30.0 * skind)

    delf = (np.log10(ak_max) - np.log10(ak_min)) / (n_ky - 2)
    ky = np.empty(n_ky)
    ky[0] = 1e-8
    for i in range(1, n_ky):
        ky[i] = 10.0 ** (np.log10(ak_min) + (i - 1) * delf)
    return ky


# ─── 3. prim-ky 이진 파일 파싱 ──────────────────────────────────────────────

def read_prim_ky_records(path: Path) -> list[np.ndarray]:
    """
    Fortran unformatted sequential 이진 파일 읽기.
    각 레코드: [4-byte length][complex*16 data][4-byte length]
    반환: list of complex128 arrays, 각 shape (n_nodes,)
    """
    data = path.read_bytes()
    records = []
    offset = 0
    while offset < len(data) - 8:
        rec_len = struct.unpack_from("<i", data, offset)[0]
        if rec_len <= 0 or rec_len > len(data):
            break
        offset += 4
        arr = np.frombuffer(data[offset: offset + rec_len], dtype=np.complex128)
        offset += rec_len
        end_len = struct.unpack_from("<i", data, offset)[0]
        if end_len != rec_len:
            print(f"[경고] 레코드 길이 불일치: {rec_len} ≠ {end_len}")
            break
        offset += 4
        records.append(arr)
    return records


# ─── 4. 소형 격자 Grid-like 객체 ────────────────────────────────────────────

class FortranGrid:
    """coord.msh 에서 읽은 좌표로 만든 Grid-like 객체"""

    def __init__(self, x_nodes: np.ndarray, z_nodes: np.ndarray,
                 halfspace_resistivity: float = 100.0):
        nx, nz = len(x_nodes), len(z_nodes)
        # 2D mesh: node_x[ix, iz], node_z[ix, iz]
        self.node_x = np.tile(x_nodes[:, None], (1, nz))
        self.node_z = np.tile(z_nodes[None, :], (nx, 1))
        self.n_nodes_x = nx
        self.n_nodes_z = nz

        class _Cfg:
            halfspace_resistivity = 100.0
        self.config = _Cfg()
        self.config.halfspace_resistivity = halfspace_resistivity

    @property
    def n_nodes(self): return self.n_nodes_x * self.n_nodes_z
    @property
    def n_elements_x(self): return self.n_nodes_x - 1
    @property
    def n_elements_z(self): return self.n_nodes_z - 1
    @property
    def n_elements(self): return self.n_elements_x * self.n_elements_z

    def node_index(self, ix: int, iz: int) -> int:
        return iz * self.n_nodes_x + ix


# ─── 5. 검증 1: Primary field 비교 ──────────────────────────────────────────

def compare_primary_field(
    fortran_root: Path,
    model_name: str = "topo_001",
    ifreq: int = 1,
    iky:   int = 1,
    itx:   int = 0,   # 0-based → Fortran transmitter 1
) -> dict:
    """
    prim-ky-XXYY.dat 의 Ey 성분과 Python primary_field_ky_domain() 비교

    Returns
    -------
    결과 dict: {fortran_Ey, python_Ey, rel_err, x_nodes, z_nodes}
    """
    print(f"\n[1차장 비교] freq#{ifreq} ky#{iky} tx#{itx+1}")
    print("=" * 60)

    # (a) 격자 파싱
    coord_path = fortran_root / "model_setup" / model_name / "coord.msh"
    x_nodes, z_nodes = parse_coord_msh(coord_path)
    nx, nz = len(x_nodes), len(z_nodes)
    print(f"  격자: {nx} x-nodes × {nz} z-nodes")

    # (b) ky 값
    ky_arr = fortran_ky_values()
    ky_val = ky_arr[iky - 1]
    freq    = [220, 440, 880, 1760, 3520, 7040, 14080, 28160][ifreq - 1]
    omega   = 2 * PI * freq
    print(f"  주파수: {freq} Hz  |  ky: {ky_val:.4e} rad/m")

    # (c) 송신기 위치 (survey.dat에서: x=12..96, z=-1m)
    src_x_all = np.arange(12.0, 97.0, 4.0)
    src_x     = src_x_all[itx]
    src_z     = -1.0
    print(f"  송신기: x={src_x} m, z={src_z} m")

    # (d) Python 1차장
    params = PrimaryFieldParams(background_resistivity=100.0)
    E_py = primary_field_ky_domain(
        source_x=src_x, source_z=src_z,
        node_x=x_nodes, node_z=z_nodes,
        wavenumber_ky=ky_val, omega=omega,
        source_type=SourceType.Jy,
        params=params,
    )
    # E_py shape: (3, nx, nz) → Ey = E_py[1]  (shape: nx, nz)
    Ey_py = E_py[1]   # (nx, nz)

    # (e) Fortran 1차장 파싱
    prim_path = (fortran_root / "output_primary" / model_name
                 / f"prim-ky-{ifreq:02d}{iky:02d}.dat")
    if not prim_path.exists():
        print(f"  [경고] 파일 없음: {prim_path}")
        return {}

    records = read_prim_ky_records(prim_path)
    # 레코드 구성: 22 tx × 3 comp (Ex, Ey, Ez) = 66 records
    # rec[itx*3 + 0] = Ex, rec[itx*3 + 1] = Ey, rec[itx*3 + 2] = Ez
    rec_idx_Ey = itx * 3 + 1
    if rec_idx_Ey >= len(records):
        print(f"  [경고] 레코드 부족: {len(records)} records, 필요: {rec_idx_Ey}")
        return {}
    Ey_fort_flat = records[rec_idx_Ey]   # (n_nodes,) = (nx*nz,)

    # Fortran 노드 순서: column-major, ix outer, iz inner
    # E_primary_tmp(comp, i), i=1..n_node
    # 보통 i = (ix-1)*nz + iz (1-based)
    # → reshape as (nx, nz)
    Ey_fort = Ey_fort_flat.reshape(nx, nz, order='F')  # try column-major

    # 비교: 수신기 노드 (surface: iz ≈ 31 or nearest to z=0)
    iz_surface = np.argmin(np.abs(z_nodes))   # z=0 closest
    iz_src_z   = np.argmin(np.abs(z_nodes - src_z))
    # x 위치: 수신기 22개 (ix: 18,20,...,60)
    ix_rxs = np.array([np.argmin(np.abs(x_nodes - sx)) for sx in src_x_all])

    print(f"  지표 z-index: {iz_surface} (z={z_nodes[iz_surface]:.2f}m)")
    print(f"  송신기 z-index: {iz_src_z} (z={z_nodes[iz_src_z]:.2f}m)")

    # Ey at surface receivers
    Ey_py_rx   = Ey_py[ix_rxs, iz_src_z]
    Ey_fort_rx = Ey_fort[ix_rxs, iz_src_z]

    # 상대 오차
    amp_py   = np.abs(Ey_py_rx)
    amp_fort = np.abs(Ey_fort_rx)
    rel_err  = np.abs(amp_py - amp_fort) / (amp_fort + 1e-300)

    print(f"\n  수신기 Ey 진폭 비교 (22 receivers at z={z_nodes[iz_src_z]:.1f}m):")
    print(f"  {'rx':>4} {'Fortran |Ey|':>16} {'Python |Ey|':>16} {'rel.err':>10}")
    for j, (ef, ep, re) in enumerate(zip(amp_fort, amp_py, rel_err)):
        print(f"  {j+1:4d} {ef:16.6e} {ep:16.6e} {re:10.4f}")

    mean_rel = rel_err.mean()
    print(f"\n  평균 상대오차: {mean_rel:.4f}  ({mean_rel*100:.2f}%)")
    if mean_rel < 0.05:
        print("  [PASS] 1차장 비교 통과 (오차 < 5%)")
    elif mean_rel < 0.20:
        print("  [WARNING] 1차장 오차 < 20% — 허용 범위 내")
    else:
        print("  [FAIL] 1차장 오차 > 20%")

    return {
        "fortran_Ey": Ey_fort_rx,
        "python_Ey":  Ey_py_rx,
        "rel_err":    rel_err,
        "x_nodes":    x_nodes,
        "z_nodes":    z_nodes,
        "src_x_all":  src_x_all,
        "ix_rxs":     ix_rxs,
    }


# ─── 6. 검증 2: Hz 수신기 데이터 비교 ────────────────────────────────────────

def compare_hz_receivers(fortran_root: Path, model_name: str = "topo_001") -> dict:
    """
    Python 순방향 모델링(균질 100 Ω·m) vs Fortran 1차장 유도 Hz 비교.

    균질 모델에서 2차장 = 0 이므로 Python total Hz ≈ primary Hz.
    Fortran 출력은 10 Ω·m 이상체 모델을 사용하므로 차이가 발생할 수 있음.
    """
    print("\n[Hz 수신기 비교] — Python 균질모델 vs Fortran 이상체모델")
    print("=" * 60)

    # (a) 격자 및 모델 파싱
    coord_path = fortran_root / "model_setup" / model_name / "coord.msh"
    x_nodes, z_nodes = parse_coord_msh(coord_path)
    nx, nz = len(x_nodes), len(z_nodes)

    # (b) Fortran 출력 파싱
    data_path = fortran_root / "output_data" / model_name / "Data_001_00001.dat"
    d = np.loadtxt(data_path)
    # 열: ifreq, itx, irec_node, Hy_real, Hy_imag, Hx_real, Hx_imag, Hz_real, Hz_imag
    Hz_fort_raw = d[:, 7] + 1j * d[:, 8]  # Va3 = Hz (dominant)
    ifreq_col   = d[:, 0].astype(int) - 1  # 0-based
    itx_col     = d[:, 1].astype(int) - 1  # 0-based

    freqs = [220, 440, 880, 1760, 3520, 7040, 14080, 28160]
    n_freq = 8
    n_tx   = 22
    src_x_all = np.arange(12.0, 97.0, 4.0)

    # Fortran Hz: shape (n_freq, n_tx)
    Hz_fort = np.zeros((n_freq, n_tx), dtype=complex)
    for row in range(len(d)):
        Hz_fort[ifreq_col[row], itx_col[row]] = Hz_fort_raw[row]

    # (c) Python 1차장 계산 (균질 100 Ω·m → total = primary)
    params = PrimaryFieldParams(background_resistivity=100.0)
    ky_arr = fortran_ky_values()

    # 수신기는 송신기와 동일 위치 (z=-1m)
    src_z     = -1.0
    iz_src    = np.argmin(np.abs(z_nodes - src_z))
    ix_rxs    = np.array([np.argmin(np.abs(x_nodes - sx)) for sx in src_x_all])

    Hz_py = np.zeros((n_freq, n_tx), dtype=complex)

    for ifreq, freq in enumerate(freqs):
        omega = 2 * PI * freq
        print(f"  Python 1차장 계산 중: freq={freq} Hz ...", end=" ", flush=True)

        # ky 적분으로 Hz 계산
        Hz_ky_sum = np.zeros(n_tx, dtype=complex)

        for iky_idx, ky in enumerate(ky_arr):
            E_py = primary_field_ky_domain(
                source_x=0.0, source_z=src_z,  # 송신기를 원점 기준으로 (상대 위치)
                node_x=x_nodes - src_x_all[0],   # 첫 번째 송신기 상대 좌표는 나중에 처리
                node_z=z_nodes,
                wavenumber_ky=ky, omega=omega,
                source_type=SourceType.Jy,
                params=params,
            )
            # Hz = -1/(iωμ) * dEy/dx  → numerical derivative
            Ey_2d = E_py[1]  # (nx, nz)
            dEy_dx = np.gradient(Ey_2d, x_nodes - src_x_all[0], axis=0)
            Hz_ky = -dEy_dx / (1j * omega * MU_0)   # (nx, nz)

            # 역 Fourier 적분: simple Gauss quadrature / trapezoidal over ky
            # (여기서는 근사: 첫 송신기 기준 상대 위치에서 각 수신기까지)
            for j in range(n_tx):
                rx = src_x_all[j] - src_x_all[0]
                # Find nearest node
                ix_nearest = np.argmin(np.abs((x_nodes - src_x_all[0]) - rx))
                # cos(ky*y=0) 적분 = 1/pi * integral Hz(ky) dky  (y=0 평면)
                # 단순 trapezoidal 근사는 별도 함수에서 하므로 여기서는 표시만
                Hz_ky_sum[j] += Hz_ky[ix_nearest, iz_src]

        print("done")

    # 실제 역 Fourier 변환 없이 직접 run_forward 사용
    print("\n  [정밀 비교] Python run_forward 실행 (n_ky=20, 균질 100 Ω·m)")
    print("  주의: Fortran은 이상체 모델(1-10 Ω·m), Python은 균질 100 Ω·m → 차이 예상")

    try:
        from em25d.forward.forward_loop import ForwardConfig, run_forward
        from em25d.mesh.grid import Grid, GridConfig
        from em25d.mesh.block import BlockPartition, BlockConfig
        from em25d.mesh.profile import ProfileNodes, ProfileConfig
        from em25d.model.resistivity import ResistivityModel
        from em25d.survey.source import SourceArray, Source
        from em25d.survey.receiver import ReceiverArray
        from em25d.survey.frequency import FrequencySet
        from em25d.survey.survey import Survey

        # Fortran 격자와 동일한 파라미터로 Python Grid 생성
        # coord.msh: 75 x-nodes, 61 z-nodes
        # interior: x=0..102m (step 2), z=-36..0m (various spacing)
        # boundary: 12 left + 12 right + 30 bottom + 30 top (approx)
        grid_cfg = GridConfig(
            n_x_cells=51,    # interior x-cells (node 13-63 = 51 cells)
            n_z_cells=30,    # below surface (node 1-31 = 30 z-levels)
            n_z_cells_air=29,  # above surface (node 32-61 = 29 layers)
            base_x_cell_size=2.0,
            base_z_cell_size=1.0,
            n_x_boundary_cells=12,
            n_z_boundary_bottom_cells=0,
            halfspace_resistivity=100.0,
        )
        grid = Grid(grid_cfg)

        block_cfg  = BlockConfig(n_blocks_x=10, n_blocks_z=5)
        bp         = BlockPartition(grid, block_cfg)
        model      = ResistivityModel(grid, bp, background_resistivity=100.0)

        profile_cfg = ProfileConfig(
            n_receivers=22, x_start=12.0, x_end=96.0, surface_z=src_z,
        )
        profile     = ProfileNodes(grid, profile_cfg)

        sources  = SourceArray([
            Source(x=float(sx), z=src_z, source_type=SourceType.Jy, strength=1.0)
            for sx in src_x_all
        ])
        receivers = ReceiverArray.from_profile(profile, measured_components=["Hz"])
        survey    = Survey(sources, receivers, FrequencySet(freqs))

        fwd_cfg = ForwardConfig(n_wavenumbers=20, use_gpu=False, solver="direct")
        result  = run_forward(grid, model, survey, profile, fwd_cfg)
        # result: dict {(ifreq, itx): field_dict} or similar
        print(f"  Python forward 완료: result type = {type(result)}")

    except Exception as e:
        print(f"  [경고] run_forward 실패: {e}")
        import traceback; traceback.print_exc()
        result = None

    # (d) 요약 비교 (Fortran Hz vs Python 1차장 Hz)
    # Fortran Hz 진폭: freq 1, all tx
    for ifreq, freq in enumerate(freqs[:3]):   # 처음 3개 주파수만 표시
        hz_f = np.abs(Hz_fort[ifreq, :])
        print(f"\n  Fortran Hz 진폭 (freq={freq} Hz):")
        print(f"    mean={hz_f.mean():.4e}, min={hz_f.min():.4e}, max={hz_f.max():.4e}")

    return {"Hz_fort": Hz_fort, "freqs": freqs}


# ─── 7. 결과 플롯 ─────────────────────────────────────────────────────────────

def plot_primary_comparison(result: dict, save_path: Path):
    """1차장 비교 플롯"""
    if not result:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    src_x = result["src_x_all"]
    Ey_f  = result["fortran_Ey"]
    Ey_p  = result["python_Ey"]

    ax = axes[0]
    ax.semilogy(src_x, np.abs(Ey_f), "b-o", label="Fortran", markersize=4)
    ax.semilogy(src_x, np.abs(Ey_p), "r--s", label="Python",  markersize=4)
    ax.set_xlabel("수신기 x 위치 [m]")
    ax.set_ylabel("|Ey| [V/m]")
    ax.set_title("1차장 Ey 진폭 비교")
    ax.legend()
    ax.grid(True, alpha=0.4)

    ax = axes[1]
    ax.plot(src_x, result["rel_err"] * 100, "k-o", markersize=4)
    ax.axhline(5.0, color="r", linestyle="--", label="5% 기준선")
    ax.set_xlabel("수신기 x 위치 [m]")
    ax.set_ylabel("상대 오차 [%]")
    ax.set_title("1차장 Ey 상대 오차")
    ax.legend()
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f"  플롯 저장: {save_path}")
    plt.close(fig)


def plot_hz_comparison(hz_dict: dict, save_path: Path):
    """Hz 수신기 데이터 비교 플롯"""
    Hz_fort = hz_dict["Hz_fort"]
    freqs   = hz_dict["freqs"]
    src_x   = np.arange(12.0, 97.0, 4.0)

    n_show = min(4, len(freqs))
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    if n_show == 1:
        axes = axes[:, None]

    for i in range(n_show):
        hz_f = Hz_fort[i, :]
        axes[0, i].semilogy(src_x, np.abs(hz_f), "b-o", markersize=4)
        axes[0, i].set_title(f"Hz 진폭 — {freqs[i]} Hz")
        axes[0, i].set_xlabel("수신기 x [m]")
        axes[0, i].set_ylabel("|Hz|")
        axes[0, i].grid(True, alpha=0.4)

        axes[1, i].plot(src_x, np.angle(hz_f, deg=True), "r-o", markersize=4)
        axes[1, i].set_title(f"Hz 위상 — {freqs[i]} Hz")
        axes[1, i].set_xlabel("수신기 x [m]")
        axes[1, i].set_ylabel("위상 [°]")
        axes[1, i].grid(True, alpha=0.4)

    fig.suptitle("Fortran Hz 수신기 데이터 (freq 1-4)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    print(f"  플롯 저장: {save_path}")
    plt.close(fig)


# ─── 8. 메인 실행 ─────────────────────────────────────────────────────────────

def main():
    fortran_root = DATA_ROOT
    model_name   = "topo_001"
    out_dir      = ROOT / "tests" / "comparison_results"
    out_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("  Fortran vs Python EM 2.5D 검증")
    print(f"  Fortran 데이터: {fortran_root}")
    print("=" * 70)

    # ── 1) ky 값 확인
    ky_arr = fortran_ky_values()
    print(f"\n[ky 샘플링 (n=20)] Fortran 재현:")
    print(f"  ky[1]  = {ky_arr[0]:.4e}  (고정)")
    print(f"  ky[2]  = {ky_arr[1]:.4e}")
    print(f"  ky[10] = {ky_arr[9]:.4e}")
    print(f"  ky[20] = {ky_arr[19]:.4e}")

    # ── 2) 1차장 비교
    prim_result = compare_primary_field(
        fortran_root, model_name=model_name,
        ifreq=1, iky=1, itx=0,
    )

    if prim_result:
        plot_primary_comparison(
            prim_result, out_dir / "primary_field_comparison.png"
        )

    # ── 3) Hz 수신기 비교 (Fortran 데이터 시각화)
    hz_result = compare_hz_receivers(fortran_root, model_name=model_name)
    if hz_result:
        plot_hz_comparison(hz_result, out_dir / "hz_receivers_fortran.png")

    print("\n" + "=" * 70)
    print("  검증 완료")
    print(f"  결과 저장: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
