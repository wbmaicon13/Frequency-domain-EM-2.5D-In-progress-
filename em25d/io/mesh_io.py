"""
격자(Grid) 저장/읽기

Fortran 대응: GetModel.for 의 격자 출력 + mproprty.dat / blck_res.dat 파일

지원 포맷:
  - NPZ  (numpy 압축, 권장)
  - CSV  (사람이 읽을 수 있는 좌표 덤프)
  - legacy Fortran 텍스트 (좌표 목록)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

from ..mesh.grid import Grid, GridConfig


# ── NPZ (권장) ───────────────────────────────────────────────────────────────

def save_grid_npz(grid: Grid, path: str | Path) -> None:
    """
    Grid 객체를 NumPy 압축 파일(.npz)로 저장

    저장 항목:
      node_x, node_z              : 노드 좌표 (n_nodes_x, n_nodes_z)
      element_block_index         : 요소 블록 인덱스
      config_*                    : GridConfig 스칼라 필드들
    """
    path = Path(path).with_suffix(".npz")
    cfg = grid.config
    np.savez_compressed(
        path,
        node_x=grid.node_x,
        node_z=grid.node_z,
        element_block_index=grid.element_block_index,
        # GridConfig 스칼라
        n_x_cells=cfg.n_x_cells,
        n_z_cells=cfg.n_z_cells,
        n_z_cells_air=cfg.n_z_cells_air,
        base_x_cell_size=cfg.base_x_cell_size,
        base_z_cell_size=cfg.base_z_cell_size,
        base_z_cell_size_air=cfg.base_z_cell_size_air,
        boundary_stretch_factor=cfg.boundary_stretch_factor,
        halfspace_resistivity=cfg.halfspace_resistivity,
        reference_frequency=cfg.reference_frequency,
        # 경계 메타
        ix_model_start=grid.ix_model_start,
        ix_model_end=grid.ix_model_end,
        iz_model_start=grid.iz_model_start,
        iz_model_end=grid.iz_model_end,
    )


def load_grid_npz(path: str | Path) -> Grid:
    """NPZ 파일에서 Grid 복원"""
    path = Path(path).with_suffix(".npz")
    d = np.load(path, allow_pickle=False)

    cfg = GridConfig(
        n_x_cells=int(d["n_x_cells"]),
        n_z_cells=int(d["n_z_cells"]),
        n_z_cells_air=int(d["n_z_cells_air"]),
        base_x_cell_size=float(d["base_x_cell_size"]),
        base_z_cell_size=float(d["base_z_cell_size"]),
        base_z_cell_size_air=float(d["base_z_cell_size_air"]),
        boundary_stretch_factor=float(d["boundary_stretch_factor"]),
        halfspace_resistivity=float(d["halfspace_resistivity"]),
        reference_frequency=float(d["reference_frequency"]),
    )
    grid = Grid(cfg)

    # 저장된 좌표 배열로 덮어쓰기 (경계 복원)
    grid.node_x = d["node_x"]
    grid.node_z = d["node_z"]
    grid.element_block_index = d["element_block_index"]
    grid.ix_model_start = int(d["ix_model_start"])
    grid.ix_model_end   = int(d["ix_model_end"])
    grid.iz_model_start = int(d["iz_model_start"])
    grid.iz_model_end   = int(d["iz_model_end"])

    return grid


# ── CSV (사람이 읽을 수 있는) ────────────────────────────────────────────────

def save_grid_csv(grid: Grid, path: str | Path) -> None:
    """
    격자 노드 좌표를 CSV로 저장

    파일 형식:
      ix, iz, x[m], z[m]
    """
    path = Path(path).with_suffix(".csv")
    rows = []
    for ix in range(grid.n_nodes_x):
        for iz in range(grid.n_nodes_z):
            rows.append([ix, iz,
                         grid.node_x[ix, 0],
                         grid.node_z[0, iz]])
    arr = np.array(rows)
    header = "ix,iz,x_m,z_m"
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


def save_element_resistivity_csv(
    element_resistivity: np.ndarray,   # (n_ex, n_ez)
    grid: Grid,
    path: str | Path,
) -> None:
    """
    요소 비저항을 CSV로 저장

    파일 형식:
      ex, ez, x_center[m], z_center[m], resistivity[Ohm·m]
    """
    path = Path(path).with_suffix(".csv")
    x_nodes = grid.node_x[:, 0]
    z_nodes = grid.node_z[0, :]
    rows = []
    for ex in range(grid.n_elements_x):
        for ez in range(grid.n_elements_z):
            xc = 0.5 * (x_nodes[ex] + x_nodes[ex + 1])
            zc = 0.5 * (z_nodes[ez] + z_nodes[ez + 1])
            rows.append([ex, ez, xc, zc, element_resistivity[ex, ez]])
    arr = np.array(rows)
    header = "ex,ez,x_center_m,z_center_m,resistivity_ohm_m"
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


# ── Fortran 텍스트 레거시 ────────────────────────────────────────────────────

def load_fortran_coordinate_file(path: str | Path) -> np.ndarray:
    """
    Fortran 좌표 파일 읽기 (x_coord, z_coord 형식)

    Fortran 대응: GetModel.for 의 x_coord / z_coord 배열 출력
    파일 형식: 한 줄에 하나의 좌표값 (m 단위)
    """
    data = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("!"):
                data.append(float(stripped.replace("D", "e").replace("d", "e")))
    return np.array(data)


def load_fortran_resistivity_model(
    path: str | Path,
    n_elements_x: int,
    n_elements_z: int,
) -> np.ndarray:
    """
    Fortran 비저항 모델 파일 읽기

    Fortran 대응: model_res/topo_*/Model_*.dat 파일
    파일 형식: 요소별 비저항 값 (1-based 순서 → 0-based 배열로 변환)

    반환:
      resistivity : (n_elements_x, n_elements_z)
    """
    values = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith(("!", "#")):
                for tok in stripped.split():
                    values.append(float(tok.replace("D", "e").replace("d", "e")))

    arr = np.array(values)
    if len(arr) != n_elements_x * n_elements_z:
        raise ValueError(
            f"비저항 모델 크기 불일치: 파일={len(arr)}, "
            f"격자={n_elements_x}×{n_elements_z}={n_elements_x*n_elements_z}"
        )
    # Fortran 저장 순서 확인 (열 우선 또는 행 우선)
    # GetModel.for 에서 요소는 (ix=1..nx, iz=1..nz) 순으로 기록됨
    return arr.reshape(n_elements_x, n_elements_z, order="C")
