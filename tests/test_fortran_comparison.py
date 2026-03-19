"""
Fortran 결과와 Python 결과 비교 검증 테스트

목표: 동일 모델/탐사조건에서 Fortran과 Python이 일치하는지 확인
      - 상대 오차 목표: 1% 이내 (Hy 성분)
      - 필요 파일: output_data/topo_001/Data_001_00001.dat (Fortran 실행 결과)

파일 없으면 자동으로 skip.

검증 시나리오:
  - 격자: coord.msh (30z × 75x 노드)
  - 모델: blck_res.dat (배경 100 Ω·m)
  - 탐사: survey.dat / profile.dat (22 수신기, 8 주파수)
  - 비교 성분: Hz (Fortran .par 에서 Use_Hz=1)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Fortran 기준 데이터 경로 ─────────────────────────────────────────────────

FORTRAN_ROOT = ROOT.parent   # em25d_whole/
DATA_FILE    = FORTRAN_ROOT / "output_data" / "topo_001" / "Data_001_00001.dat"
COORD_FILE   = FORTRAN_ROOT / "model_setup" / "topo_001" / "coord.msh"
PROFILE_FILE = FORTRAN_ROOT / "model_setup" / "topo_001" / "profile.dat"
BLCK_RES_FILE = FORTRAN_ROOT / "model_setup" / "topo_001" / "blck_res.dat"
SURVEY_FILE  = FORTRAN_ROOT / "model_setup" / "topo_001" / "survey.dat"


def _fortran_data_available() -> bool:
    return DATA_FILE.exists() and COORD_FILE.exists()


# ── Fortran 텍스트 출력 파서 ─────────────────────────────────────────────────

def _load_fortran_output(path: Path) -> np.ndarray:
    """
    Data_001_00001.dat 파싱

    형식: ifreq  itx  irx  Ex  Ey  Ez  Hx  Hy  Hz  (실수부)
    반환: (n_freq, n_tx, n_rx, 6) float64
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                ifreq = int(parts[0]) - 1
                itx   = int(parts[1]) - 1
                irx   = int(parts[2]) - 1
                vals  = [float(v) for v in parts[3:9]]
                rows.append((ifreq, itx, irx, vals))
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"파싱 실패: {path}")

    n_freq = max(r[0] for r in rows) + 1
    n_tx   = max(r[1] for r in rows) + 1
    n_rx   = max(r[2] for r in rows) + 1

    data = np.zeros((n_freq, n_tx, n_rx, 6), dtype=float)
    for ifreq, itx, irx, vals in rows:
        data[ifreq, itx, irx, :] = vals

    return data


def _load_fortran_coord(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    coord.msh 파싱

    형식:
      z-section: 'n_z  n_z_total  n_topo  n_x' + z 좌표
      x-section: 'n_x  n_left  n_model  n_right' + x 좌표

    반환: (z_nodes, x_nodes) 1D 배열 (단위: m)
    """
    lines = path.read_text().splitlines()
    z_nodes, x_nodes = [], []
    reading_z = reading_x = False

    for line in lines:
        stripped = line.strip()
        if "z coordinate" in stripped.lower():
            reading_z, reading_x = True, False
            continue
        if "x coordinate" in stripped.lower():
            reading_z, reading_x = False, True
            continue

        parts = stripped.split()
        if len(parts) < 2:
            continue

        # 앞 두 토큰이 모두 정수면 헤더 행 → 무시
        try:
            int(parts[0]); int(parts[1])
            # 헤더: n_z/n_x 등
            continue
        except ValueError:
            pass

        try:
            _idx = int(parts[0])
            val  = float(parts[1].replace("D", "e").replace("d", "e"))
        except (ValueError, IndexError):
            continue

        if reading_z:
            z_nodes.append(val)
        elif reading_x:
            x_nodes.append(val)

    return np.array(z_nodes), np.array(x_nodes)


def _load_fortran_frequencies() -> np.ndarray:
    """
    Fortran 실행에 사용된 주파수 추출 (survey.dat 또는 하드코딩)

    survey.dat 첫 줄: 'n_nodes_x  n_nodes_z'
    주파수 정보는 survey.dat 에 없으므로 Data 파일의 주파수 인덱스와
    mproprty.dat 에서 읽거나, 알려진 .par 설정 사용.
    """
    mproprty = FORTRAN_ROOT / "model_setup" / "topo_001" / "mproprty.dat"
    if not mproprty.exists():
        # fallback: .par 기본값
        return np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])

    lines = mproprty.read_text().splitlines()
    freqs = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                # 두 번째 열이 주파수 (Hz)
                f = float(parts[1].replace("D", "e").replace("d", "e"))
                freqs.append(f)
            except ValueError:
                continue
    return np.array(freqs) if freqs else np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])


# ── Fortran 격자로 Python 모델 재구성 ────────────────────────────────────────

def _build_python_model_from_fortran_grid():
    """
    Fortran coord.msh → Python Grid, ResistivityModel, Survey, ProfileNodes

    배경 비저항: 100 Ω·m (Fortran .par 의 Background Resistivity = 100.0)
    """
    from em25d.mesh.grid import Grid, GridConfig
    from em25d.mesh.block import BlockPartition, BlockConfig
    from em25d.mesh.profile import ProfileNodes, ProfileConfig
    from em25d.model.resistivity import ResistivityModel
    from em25d.survey.source import SourceArray, Source
    from em25d.survey.receiver import ReceiverArray
    from em25d.survey.frequency import FrequencySet
    from em25d.survey.survey import Survey
    from em25d.constants import SourceType

    z_nodes, x_nodes = _load_fortran_coord(COORD_FILE)
    freqs = _load_fortran_frequencies()[:8]   # 최대 8개

    # Fortran 격자: z 방향이 위→아래 (음수=위, 양수=아래)
    # Python은 z=0 이 지표, z>0 이 지하
    # 지표 노드 인덱스 (z=0 에 해당하는 위치)
    iz_surface = np.argmin(np.abs(z_nodes))

    # GridConfig 로 동일 격자 재현은 복잡 → 직접 노드 배열로 Grid 구성
    # Grid 클래스가 node_x, node_z 직접 주입을 지원해야 함
    # 지원 안 할 경우 근사적 GridConfig 사용
    n_nz = len(z_nodes)
    n_nx = len(x_nodes)

    # 근사 격자 설정 (비균등 격자이므로 평균 셀 크기 사용)
    dz_interior = np.diff(z_nodes[iz_surface:iz_surface + 11]).mean()
    dx_interior = np.diff(x_nodes[n_nx // 2 - 5:n_nx // 2 + 5]).mean()

    n_z_model = n_nz - iz_surface - 1
    n_z_air   = iz_surface
    n_x_model = n_nx - 24   # 좌우 경계(각 12개) 제외 근사

    cfg = GridConfig(
        n_x_cells=max(n_x_model, 10),
        n_z_cells=max(n_z_model, 10),
        n_z_cells_air=max(n_z_air, 3),
        base_x_cell_size=max(dx_interior, 1.0),
        base_z_cell_size=max(dz_interior, 1.0),
        n_x_boundary_cells=12,
        n_z_boundary_bottom_cells=10,
        halfspace_resistivity=100.0,
    )
    grid = Grid(cfg)

    # 블록 파티션 (Fortran blck_res.dat 구조에 근사)
    block_cfg = BlockConfig(n_blocks_x=52, n_blocks_z=11)
    bp = BlockPartition(grid, block_cfg)
    model = ResistivityModel(grid, bp, background_resistivity=100.0)

    # 수신기 프로파일 (Data 파일에서 22개 Rx)
    n_rx = 22
    profile_cfg = ProfileConfig(
        n_receivers=n_rx,
        x_start=x_nodes[12],     # 모델 영역 시작 x
        x_end=x_nodes[12 + n_rx - 1],
        surface_z=0.0,
    )
    profile = ProfileNodes(grid, profile_cfg)

    # 송신기 (Jy 다이폴, x=0)
    source = Source(x=0.0, z=0.0, source_type=SourceType.Jy, strength=1.0)
    sources = SourceArray([source])

    receivers = ReceiverArray.from_profile(profile, measured_components=["Hz"])
    freq_set  = FrequencySet(freqs)
    survey    = Survey(sources, receivers, freq_set)

    return grid, model, survey, profile


# ── 검증 테스트 ───────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not _fortran_data_available(),
    reason="Fortran 기준 데이터 없음 (output_data/topo_001/Data_001_00001.dat)"
)
class TestFortranComparison:
    """Fortran vs Python 비교 검증"""

    @pytest.fixture(scope="class")
    def fortran_data(self):
        return _load_fortran_output(DATA_FILE)

    @pytest.fixture(scope="class")
    def python_result(self):
        from em25d.forward.forward_loop import ForwardConfig, run_forward

        grid, model, survey, profile = _build_python_model_from_fortran_grid()
        cfg = ForwardConfig(
            n_wavenumbers=20,   # Fortran .par 의 n_ky=20
            use_gpu=False,
            solver="direct",
        )
        return run_forward(grid, model, survey, profile, cfg), survey

    def test_shape_matches(self, fortran_data):
        n_freq, n_tx, n_rx, n_comp = fortran_data.shape
        assert n_comp == 6, "Fortran 데이터 성분 수 오류"

    def test_hz_component_relative_error(self, fortran_data, python_result):
        """
        Hz 성분 (인덱스 5) 실수부 상대 오차 < 5%

        Fortran .par: Use_Hz=1, Total field (field_type=0)
        Python: total field 모드로 Hz 추출
        """
        py_data, survey = python_result
        fort_hz = fortran_data[:, :, :, 5]   # (n_freq, n_tx, n_rx)

        # Python Hz = 인덱스 5 의 실수부
        n_freq_f = fortran_data.shape[0]
        n_tx_f   = fortran_data.shape[1]
        n_rx_f   = fortran_data.shape[2]

        n_freq_p = min(py_data.shape[0], n_freq_f)
        n_rx_p   = min(py_data.shape[2], n_rx_f)

        py_hz = py_data[:n_freq_p, :n_tx_f, :n_rx_p, 5].real

        # 비제로 요소만 상대 오차 계산
        ref = np.abs(fort_hz[:n_freq_p, :n_tx_f, :n_rx_p])
        mask = ref > 1e-30

        if mask.sum() == 0:
            pytest.skip("비교 가능한 비제로 데이터 없음")

        rel_err = np.abs(py_hz[mask] - fort_hz[:n_freq_p, :n_tx_f, :n_rx_p][mask]) / ref[mask]
        mean_err = rel_err.mean()
        max_err  = rel_err.max()

        print(f"\nHz 상대 오차:  평균={mean_err*100:.2f}%  최대={max_err*100:.2f}%")

        assert mean_err < 0.05, \
            f"Hz 성분 평균 상대 오차 {mean_err*100:.2f}% > 5%"

    def test_hy_component_relative_error(self, fortran_data, python_result):
        """Hy 성분 (인덱스 4) 실수부 상대 오차 < 5%"""
        py_data, _ = python_result
        n_freq_f = fortran_data.shape[0]
        n_tx_f   = fortran_data.shape[1]
        n_rx_f   = fortran_data.shape[2]
        n_freq_p = min(py_data.shape[0], n_freq_f)
        n_rx_p   = min(py_data.shape[2], n_rx_f)

        fort_hy = fortran_data[:n_freq_p, :n_tx_f, :n_rx_p, 4]
        py_hy   = py_data[:n_freq_p, :n_tx_f, :n_rx_p, 4].real

        ref = np.abs(fort_hy)
        mask = ref > 1e-30
        if mask.sum() == 0:
            pytest.skip("비교 가능한 비제로 데이터 없음")

        rel_err  = np.abs(py_hy[mask] - fort_hy[mask]) / ref[mask]
        mean_err = rel_err.mean()
        print(f"\nHy 상대 오차:  평균={mean_err*100:.2f}%")
        assert mean_err < 0.05, f"Hy 평균 상대 오차 {mean_err*100:.2f}% > 5%"


# ── 독립 실행 비교 도구 ──────────────────────────────────────────────────────

def compare_and_report(save_path: str = None):
    """
    Fortran vs Python 상세 비교 리포트 출력 (pytest 외부 실행용)

    사용법:
        python tests/test_fortran_comparison.py
    """
    if not _fortran_data_available():
        print("[경고] Fortran 기준 데이터 없음. 검증 건너뜀.")
        return

    import matplotlib.pyplot as plt

    print("Fortran 데이터 로드 중...")
    fort_data = _load_fortran_output(DATA_FILE)
    print(f"  형태: {fort_data.shape}  (n_freq, n_tx, n_rx, 6)")

    print("Python 순방향 계산 중...")
    from em25d.forward.forward_loop import ForwardConfig, run_forward

    grid, model, survey, profile = _build_python_model_from_fortran_grid()
    cfg = ForwardConfig(n_wavenumbers=20, use_gpu=False, solver="direct")
    py_data = run_forward(grid, model, survey, profile, cfg)
    print(f"  형태: {py_data.shape}")

    n_freq = min(fort_data.shape[0], py_data.shape[0])
    n_rx   = min(fort_data.shape[2], py_data.shape[2])
    n_comp = 6

    comp_names = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    rx_x = np.arange(n_rx)

    fig, axes = plt.subplots(n_freq, 2, figsize=(14, 3 * n_freq))
    if n_freq == 1:
        axes = axes[np.newaxis, :]

    freqs = survey.frequencies.frequencies

    for ifreq in range(n_freq):
        for icol, (part_fn, part_label) in enumerate(
                [(np.real, "실수부"), (np.imag, "허수부")]):
            ax = axes[ifreq, icol]
            for ic, cname in enumerate(comp_names):
                fort_v = part_fn(fort_data[ifreq, 0, :n_rx, ic])
                py_v   = part_fn(py_data [ifreq, 0, :n_rx, ic].astype(complex))
                ax.plot(rx_x, fort_v, label=f"Fort-{cname}", lw=1.5, ls="-")
                ax.plot(rx_x, py_v,   label=f"Py-{cname}",  lw=1.0, ls="--")

            ax.set_title(f"f={freqs[ifreq]:.1f} Hz  {part_label}")
            ax.set_xlabel("수신기 인덱스")
            ax.legend(fontsize=6, ncol=2)
            ax.grid(True, alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"그림 저장: {save_path}")
    else:
        plt.show()

    # 수치 요약
    print("\n=== 상대 오차 요약 (Hy 성분, 실수부) ===")
    hy_fort = fort_data[:n_freq, 0, :n_rx, 4]
    hy_py   = py_data  [:n_freq, 0, :n_rx, 4].real
    ref = np.abs(hy_fort)
    mask = ref > 1e-30
    if mask.sum() > 0:
        rel_err = np.abs(hy_py[mask] - hy_fort[mask]) / ref[mask]
        print(f"  평균 상대 오차: {rel_err.mean()*100:.3f}%")
        print(f"  최대 상대 오차: {rel_err.max() *100:.3f}%")
        print(f"  중간 상대 오차: {np.median(rel_err)*100:.3f}%")


if __name__ == "__main__":
    import sys
    save = sys.argv[1] if len(sys.argv) > 1 else None
    compare_and_report(save_path=save)
