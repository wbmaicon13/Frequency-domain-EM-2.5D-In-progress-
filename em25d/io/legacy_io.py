"""
Fortran 바이너리/텍스트 포맷 호환

Fortran 대응:
  - output_primary/prim-ky-*.dat  (1차장 바이너리)
  - output_primary/prim-e.dat     (실공간 전기장)
  - output_primary/prim-h.dat     (실공간 자기장)
  - output_data/ 결과 파일
  - model_setup/topo_NNN/ 격자 파일 (nodetest.dat, elemtest.dat 등)

Fortran unformatted binary 는 레코드 마커(4바이트 정수)가 앞뒤로 붙음.
Python 에서는 scipy.io FortranFile 로 읽음.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.io import FortranFile
from dataclasses import dataclass
from typing import Iterator, Optional


# ── 1차장 바이너리 ────────────────────────────────────────────────────────────

def read_primary_ky_file(
    path: str | Path,
    n_nodes: int,
    n_transmitters: int,
) -> np.ndarray:
    """
    Fortran unformatted 1차장(ky 영역) 파일 읽기

    Fortran 대응: prim-ky-FQFK.dat (Primary_Driver 에서 write)

    파일 구조 (송신기당):
      write(3) (E_primary_tmp(1, i), i=1,n_node)  ! Ex
      write(3) (E_primary_tmp(2, i), i=1,n_node)  ! Ey
      write(3) (E_primary_tmp(3, i), i=1,n_node)  ! Ez

    반환:
      E_primary : (3, n_nodes, n_transmitters) complex128
    """
    path = Path(path)
    E_primary = np.zeros((3, n_nodes, n_transmitters), dtype=complex)

    with FortranFile(path, "r") as f:
        for itx in range(n_transmitters):
            for ic in range(3):
                rec = f.read_record(dtype=np.complex128)
                E_primary[ic, :, itx] = rec

    return E_primary


def read_primary_space_file(
    path: str | Path,
    n_profile_nodes: int,
    n_transmitters: int,
    n_frequencies: int,
) -> np.ndarray:
    """
    실공간 1차장 파일 읽기 (prim-e.dat 또는 prim-h.dat)

    Fortran 대응: primary_space_w 에서 write(ipe) (exyz(l,ir), l=1,3)

    파일 구조 (주파수 × 송신기 × 수신기):
      write(ipe) (exyz(l, ir), l=1,3)   ! ir = 수신기, l = 성분

    반환:
      field : (n_frequencies, n_transmitters, n_profile_nodes, 3) complex128
    """
    path = Path(path)
    field = np.zeros(
        (n_frequencies, n_transmitters, n_profile_nodes, 3), dtype=complex)

    with FortranFile(path, "r") as f:
        for ifreq in range(n_frequencies):
            for itx in range(n_transmitters):
                for irx in range(n_profile_nodes):
                    rec = f.read_record(dtype=np.complex128)
                    field[ifreq, itx, irx, :] = rec[:3]

    return field


# ── 순방향 결과 바이너리 ─────────────────────────────────────────────────────

def read_forward_result_binary(
    path: str | Path,
    n_profile_nodes: int,
    n_transmitters: int,
    n_frequencies: int,
    n_components: int = 6,
) -> np.ndarray:
    """
    Fortran 순방향 결과 바이너리 읽기

    반환:
      data : (n_frequencies, n_transmitters, n_profile_nodes, n_components)
    """
    path = Path(path)
    data = np.zeros(
        (n_frequencies, n_transmitters, n_profile_nodes, n_components),
        dtype=complex)

    with FortranFile(path, "r") as f:
        for ifreq in range(n_frequencies):
            for itx in range(n_transmitters):
                for irx in range(n_profile_nodes):
                    rec = f.read_record(dtype=np.complex128)
                    data[ifreq, itx, irx, :] = rec[:n_components]

    return data


# ── 블록 비저항 파일 ─────────────────────────────────────────────────────────

def read_block_resistivity(path: str | Path) -> np.ndarray:
    """
    역산 블록 비저항 파일 읽기

    Fortran 대응: blck_res.dat
    파일 형식: 블록 번호(1-based)  비저항[Ω·m]  (한 줄씩)

    반환:
      block_rho : (n_blocks,) float64
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    values = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(("!", "#")):
            toks = stripped.split()
            if len(toks) >= 2:
                values.append(float(toks[1].replace("D", "e").replace("d", "e")))
            elif len(toks) == 1:
                values.append(float(toks[0].replace("D", "e").replace("d", "e")))
    return np.array(values)


def write_block_resistivity(
    block_rho: np.ndarray,
    path: str | Path,
) -> None:
    """
    역산 블록 비저항 파일 저장 (Fortran 호환 텍스트)

    파일 형식: 블록 번호(1-based)  비저항[Ω·m]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, rho in enumerate(block_rho, start=1):
            f.write(f"{i:6d}  {rho:.6E}\n")


# ── 야코비안 파일 ────────────────────────────────────────────────────────────

def read_jacobian_binary(
    path: str | Path,
    n_data: int,
    n_blocks: int,
) -> np.ndarray:
    """
    야코비안 행렬 바이너리 읽기

    Fortran 대응: JacobianMatrixKy 저장 파일

    반환:
      J : (n_data, n_blocks) complex128
    """
    path = Path(path)
    J = np.zeros((n_data, n_blocks), dtype=complex)

    with FortranFile(path, "r") as f:
        for i in range(n_data):
            rec = f.read_record(dtype=np.complex128)
            J[i, :] = rec[:n_blocks]

    return J


def write_jacobian_npz(
    J: np.ndarray,
    path: str | Path,
    iteration: int = 0,
    frequency: float = None,
) -> None:
    """
    야코비안 행렬을 NPZ로 저장 (역산 로그)

    inversion_log/run_*/jacobian_iter_NNN.npz
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {}
    if frequency is not None:
        meta["frequency_hz"] = frequency
    np.savez_compressed(
        path,
        jacobian=J,
        iteration=iteration,
        **meta,
    )


# ── 반복 모델 저장 ────────────────────────────────────────────────────────────

def write_model_iteration(
    block_rho: np.ndarray,
    iteration: int,
    rms_error: float,
    log_dir: str | Path,
) -> None:
    """
    역산 반복별 비저항 모델 저장

    저장 위치: log_dir/model_iter_NNN.dat
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"model_iter_{iteration:03d}.dat"
    write_block_resistivity(block_rho, path)

    # 로그 파일에 RMS 추가 (append 모드)
    log_path = log_dir / "misfit_log.csv"
    if not log_path.exists():
        log_path.write_text("iteration,rms_error\n")
    with open(log_path, "a") as f:
        f.write(f"{iteration},{rms_error:.6E}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 레거시 격자/모델/탐사 파일 읽기 (model_setup/topo_NNN/)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LegacyMesh:
    """Fortran 메시 데이터"""
    node_x_1d: np.ndarray      # (n_nodes_x,) 유니크 x 좌표 (정렬)
    node_z_1d: np.ndarray      # (n_nodes_z,) 유니크 z 좌표 (정렬)
    n_nodes_x: int
    n_nodes_z: int
    n_nodes: int
    n_elements: int
    n_elements_x: int
    n_elements_z: int
    # Fortran→Python 노드 인덱스 매핑: fortran_to_python[f_idx] = p_idx
    fortran_to_python: np.ndarray   # (n_nodes,) int


@dataclass
class LegacySurvey:
    """Fortran survey.dat 탐사 파라미터"""
    n_freq: int
    frequencies: np.ndarray         # (n_freq,)
    n_transmitters: int
    n_receiver_max: int
    source_x: np.ndarray            # (n_tx,)
    source_z: np.ndarray            # (n_tx,)
    source_type: int                # itype: 1=Jx,2=Jy,3=Jz,4=Mx,5=My,6=Mz
    homogeneous_resistivity: float
    anomalous_resistivity: float
    add_sea_layer: bool
    # 수신기 노드 인덱스 (Fortran 1-based → Python 0-based 변환 전)
    receiver_nodes_fortran: list     # [tx][rx] Fortran 1-based
    n_receivers_per_tx: np.ndarray   # (n_tx,) 각 TX의 RX 수


def read_nodetest(path: str | Path) -> tuple[np.ndarray, int]:
    """
    nodetest.dat 읽기

    Fortran 대응: Fem25Dpar.f90 read_data 서브루틴 (line 355-377)

    파일 형식 (ASCII):
      n_node                          (총 노드 수)
      node_id  x  z                   (n_node 줄)

    Fortran 노드 넘버링: f = ix * n_nz + iz (z 먼저 순회)

    Returns
    -------
    coords : (n_node, 2) — [x, z] 좌표
    n_node : int
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    n_node = int(lines[0].strip())
    coords = np.zeros((n_node, 2))
    for i in range(n_node):
        parts = lines[i + 1].split()
        coords[i, 0] = float(parts[1])   # x
        coords[i, 1] = float(parts[2])   # z
    return coords, n_node


def read_elemtest(path: str | Path) -> tuple[np.ndarray, int]:
    """
    elemtest.dat 읽기

    파일 형식 (ASCII):
      n_elem
      elem_id  n1  n2  n3  n4        (4 corner node IDs, 1-based)

    Returns
    -------
    connectivity : (n_elem, 4) — Fortran 1-based 노드 인덱스
    n_elem       : int
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    n_elem = int(lines[0].strip())
    connectivity = np.zeros((n_elem, 4), dtype=int)
    for i in range(n_elem):
        parts = lines[i + 1].split()
        for j in range(4):
            connectivity[i, j] = int(parts[j + 1])
    return connectivity, n_elem


def read_mproprty(path: str | Path, n_elem: int = None) -> np.ndarray:
    """
    mproprty.dat 또는 Model_NNNNN.dat 읽기

    파일 형식:
      [n_elem 헤더 (선택)]
      elem_id  resistivity  [conductivity]

    Returns
    -------
    resistivity : (n_elem,) float — 요소별 비저항 [Ω·m]
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    values = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) == 1:
            continue  # 헤더 (n_elem)
        values.append(float(parts[1].replace("D", "e").replace("d", "e")))
    arr = np.array(values)
    if n_elem is not None and len(arr) != n_elem:
        raise ValueError(
            f"요소 수 불일치: 파일={len(arr)}, 기대={n_elem}")
    return arr


def read_survey_dat(path: str | Path) -> LegacySurvey:
    """
    survey.dat 읽기

    Fortran 대응: Fem25Dpar.f90 (line 139-183)

    파일 형식:
      F2DF (파일명)
      NoOfXNode  NoOfZNode   (출력용 차원, 계산격자 아님)
      n_freq
      freq1  freq2  ...
      n_transmitter  n_receiver_max
      tx_x  tx_z    (n_transmitter 줄)
      Addsealayer (T/F)
      homo_rho  anom_rho  (or 3 values if sea layer)
      itype
      For each TX: n_rx / rx_node_indices
    """
    path = Path(path)
    text = path.read_text()
    # 전체 토큰화 (첫 줄은 파일명이므로 별도 처리)
    lines = text.splitlines()

    # Line 1: F2DF filename (문자열, 스킵)
    line_idx = 0
    line_idx += 1  # skip filename line

    # 나머지를 토큰으로 파싱
    remaining = "\n".join(lines[line_idx:])
    tokens = remaining.split()
    it = iter(tokens)

    def nxt_int():
        return int(next(it))

    def nxt_float():
        return float(next(it).replace("D", "e").replace("d", "e"))

    def nxt_str():
        return next(it)

    # NoOfXNode, NoOfZNode (출력용 — 무시)
    _nox = nxt_int()
    _noz = nxt_int()

    n_freq = nxt_int()
    frequencies = np.array([nxt_float() for _ in range(n_freq)])

    n_transmitter = nxt_int()
    n_receiver_max = nxt_int()

    source_x = np.zeros(n_transmitter)
    source_z = np.zeros(n_transmitter)
    for itx in range(n_transmitter):
        source_x[itx] = nxt_float()
        source_z[itx] = nxt_float()

    # Addsealayer
    add_sea_str = nxt_str().upper()
    add_sea_layer = add_sea_str in ("T", ".TRUE.", "TRUE")

    if add_sea_layer:
        _waterdepth = nxt_float()
        _seawater_rho = nxt_float()
        homogeneous_resistivity = nxt_float()
        anomalous_resistivity = homogeneous_resistivity
    else:
        homogeneous_resistivity = nxt_float()
        anomalous_resistivity = nxt_float()

    itype = nxt_int()

    # 수신기 노드 인덱스 (Fortran 1-based)
    receiver_nodes = []
    n_receivers_per_tx = np.zeros(n_transmitter, dtype=int)
    for itx in range(n_transmitter):
        n_rx = nxt_int()
        n_receivers_per_tx[itx] = n_rx
        nodes = [nxt_int() for _ in range(n_rx)]
        receiver_nodes.append(nodes)

    return LegacySurvey(
        n_freq=n_freq,
        frequencies=frequencies,
        n_transmitters=n_transmitter,
        n_receiver_max=n_receiver_max,
        source_x=source_x,
        source_z=source_z,
        source_type=itype,
        homogeneous_resistivity=homogeneous_resistivity,
        anomalous_resistivity=anomalous_resistivity,
        add_sea_layer=add_sea_layer,
        receiver_nodes_fortran=receiver_nodes,
        n_receivers_per_tx=n_receivers_per_tx,
    )


def build_legacy_mesh(topo_dir: str | Path) -> LegacyMesh:
    """
    Fortran 레거시 메시 파일에서 메시 정보 구성

    Parameters
    ----------
    topo_dir : model_setup/topo_NNN/ 디렉토리 경로

    Returns
    -------
    LegacyMesh
    """
    topo_dir = Path(topo_dir)
    coords, n_node = read_nodetest(topo_dir / "nodetest.dat")

    # 유니크 좌표 추출 (직교격자이므로 x, z 각각 유니크)
    x_unique = np.sort(np.unique(coords[:, 0]))
    z_unique = np.sort(np.unique(coords[:, 1]))
    n_nx = len(x_unique)
    n_nz = len(z_unique)

    if n_nx * n_nz != n_node:
        raise ValueError(
            f"직교격자 검증 실패: {n_nx}×{n_nz}={n_nx*n_nz} ≠ {n_node}")

    # Fortran→Python 노드 인덱스 매핑
    # Fortran: f = ix * n_nz + iz (z 먼저)
    # Python:  p = iz * n_nx + ix (x 먼저)
    x_to_ix = {x: i for i, x in enumerate(x_unique)}
    z_to_iz = {z: i for i, z in enumerate(z_unique)}

    f2p = np.zeros(n_node, dtype=int)
    for f_idx in range(n_node):
        x, z = coords[f_idx]
        ix = x_to_ix[x]
        iz = z_to_iz[z]
        p_idx = iz * n_nx + ix
        f2p[f_idx] = p_idx

    n_ex = n_nx - 1
    n_ez = n_nz - 1

    _, n_elem = read_elemtest(topo_dir / "elemtest.dat")
    if n_ex * n_ez != n_elem:
        raise ValueError(
            f"요소 수 검증 실패: {n_ex}×{n_ez}={n_ex*n_ez} ≠ {n_elem}")

    return LegacyMesh(
        node_x_1d=x_unique,
        node_z_1d=z_unique,
        n_nodes_x=n_nx,
        n_nodes_z=n_nz,
        n_nodes=n_node,
        n_elements=n_elem,
        n_elements_x=n_ex,
        n_elements_z=n_ez,
        fortran_to_python=f2p,
    )


def load_legacy_resistivity(
    path: str | Path,
    mesh: LegacyMesh,
) -> np.ndarray:
    """
    레거시 비저항 모델 로딩 → (n_ex, n_ez) Python 배열

    Fortran 요소 넘버링: k = ix * n_ez + iz (z 먼저 순회)
    → reshape(n_ex, n_ez) C-order로 바로 사용 가능

    Returns
    -------
    element_resistivity : (n_ex, n_ez) float — 0 = 공기
    """
    rho_flat = read_mproprty(path, mesh.n_elements)
    return rho_flat.reshape(mesh.n_elements_x, mesh.n_elements_z)


def read_fortran_output_data(
    path: str | Path,
) -> tuple[np.ndarray, int, int]:
    """
    Fortran 순방향 결과 ASCII 파일 읽기 (Data_NNN_NNNNN.dat)

    파일 형식 (176줄 = 8 freq × 22 rx):
      ifreq  itx  irx  Ey_r  Ey_i  Hx_r  Hx_i  Hz_r  Hz_i

    Returns
    -------
    data     : (n_data, 9) — [ifreq, itx, irx, Ey_r, Ey_i, Hx_r, Hx_i, Hz_r, Hz_i]
    n_freq   : int
    n_data   : int
    """
    raw = np.loadtxt(path)
    n_data = raw.shape[0]
    n_freq = int(raw[:, 0].max())
    return raw, n_freq, n_data


def fortran_node_to_python(
    fortran_node_1based: int,
    mesh: LegacyMesh,
) -> int:
    """Fortran 1-based 노드 인덱스 → Python 0-based 인덱스"""
    return int(mesh.fortran_to_python[fortran_node_1based - 1])


def read_profile_c(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    profile_c.dat 읽기 — 프로파일(수신기) 좌표

    파일 형식:
      flag  profile_idx  x  z  (한 줄씩)

    Returns
    -------
    profile_x : (n_profile,) 프로파일 x 좌표
    profile_z : (n_profile,) 프로파일 z 좌표
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    xs, zs = [], []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 4:
            xs.append(float(parts[2].replace("D", "e").replace("d", "e")))
            zs.append(float(parts[3].replace("D", "e").replace("d", "e")))
    return np.array(xs), np.array(zs)


def get_profile_global_nodes(
    profile_x: np.ndarray,
    profile_z: np.ndarray,
    mesh: LegacyMesh,
) -> np.ndarray:
    """
    프로파일 좌표를 Python 전역 노드 인덱스로 변환

    프로파일 점이 격자 노드 위에 있다고 가정.

    Returns
    -------
    global_nodes : (n_profile,) int — Python 0-based 전역 노드 인덱스
    """
    nodes = np.zeros(len(profile_x), dtype=int)
    for i in range(len(profile_x)):
        ix = np.argmin(np.abs(mesh.node_x_1d - profile_x[i]))
        iz = np.argmin(np.abs(mesh.node_z_1d - profile_z[i]))
        nodes[i] = iz * mesh.n_nodes_x + ix
    return nodes


def get_tx_rx_pairs(
    survey: LegacySurvey,
    profile_global_nodes: np.ndarray,
) -> list[tuple[int, int, int]]:
    """
    TX-RX 쌍 목록 생성

    survey.dat의 ReceiverNode는 프로파일 포인트 인덱스(1-based).
    profile_global_nodes는 Python 전역 노드 인덱스(0-based).

    Returns
    -------
    pairs : list of (itx, irx_profile_0based, global_node_idx)
    """
    pairs = []
    for itx in range(survey.n_transmitters):
        for f_rx_node in survey.receiver_nodes_fortran[itx]:
            # Fortran 1-based profile index → 0-based
            irx = f_rx_node - 1
            global_node = int(profile_global_nodes[irx])
            pairs.append((itx, irx, global_node))
    return pairs
