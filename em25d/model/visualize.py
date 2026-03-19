"""
전기비저항 모델 시각화 및 저장

요구사항 3-4 대응:
  - 생성된 모델 + 송수신기 배열 그림 저장
  - 단일 모델 및 다중 모델 일괄 저장 지원
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")   # GUI 없는 환경에서도 동작
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Optional, Sequence

from .resistivity import ResistivityModel
from ..mesh.grid import Grid


def plot_resistivity_model(
    model: ResistivityModel,
    ax: Optional[plt.Axes] = None,
    vmin: float = 1.0,
    vmax: float = 10000.0,
    cmap: str = "jet_r",
    title: str = "Resistivity Model",
    show_grid: bool = False,
    source_x: Optional[np.ndarray] = None,
    source_z: Optional[np.ndarray] = None,
    receiver_x: Optional[np.ndarray] = None,
    receiver_z: Optional[np.ndarray] = None,
) -> plt.Axes:
    """
    전기비저항 모델 단면도 그리기

    Parameters
    ----------
    model       : ResistivityModel
    ax          : 기존 Axes (None 이면 새로 생성)
    vmin, vmax  : 색상 범위 [Ω·m]
    cmap        : 컬러맵
    show_grid   : 격자선 표시 여부
    source_x/z  : 송신기 좌표 (★ 마커)
    receiver_x/z: 수신기 좌표 (▽ 마커)
    """
    grid = model.grid
    x_nodes = grid.node_x[:, 0]
    z_nodes = grid.node_z[0, :]

    # 모델 영역만 표시 (경계 제외)
    ix_s, ix_e = grid.ix_model_start, grid.ix_model_end
    iz_s, iz_e = grid.iz_model_start, grid.iz_model_end

    x_plot = x_nodes[ix_s:ix_e + 1]
    z_plot = z_nodes[iz_s:iz_e + 1]
    rho    = model.element_resistivity[ix_s:ix_e, iz_s:iz_e].T  # (z, x)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    pcm  = ax.pcolormesh(
        x_plot, z_plot, rho,
        norm=norm, cmap=cmap, shading="flat",
    )

    plt.colorbar(pcm, ax=ax, label="Resistivity (Ω·m)")

    if show_grid:
        ax.set_xticks(x_plot, minor=True)
        ax.set_yticks(z_plot, minor=True)
        ax.grid(True, which="minor", color="gray", linewidth=0.3, alpha=0.4)

    # 송신기/수신기 표시
    if source_x is not None and source_z is not None:
        ax.plot(source_x, source_z, "r*", markersize=10,
                label="Tx (Source)", zorder=5)
    if receiver_x is not None and receiver_z is not None:
        ax.plot(receiver_x, receiver_z, "bv", markersize=6,
                label="Rx (Receiver)", zorder=5)
    if source_x is not None or receiver_x is not None:
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    ax.invert_yaxis()     # 깊이 아래 방향
    ax.set_aspect("equal", adjustable="box")

    return ax


def save_model_figure(
    model: ResistivityModel,
    filepath: str,
    dpi: int = 150,
    **kwargs,
):
    """단일 모델 그림 파일 저장"""
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_resistivity_model(model, ax=ax, **kwargs)
    fig.tight_layout()
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_model_batch(
    models: Sequence[ResistivityModel],
    output_dir: str,
    prefix: str = "model",
    dpi: int = 100,
    n_cols: int = 4,
    **kwargs,
):
    """
    다중 모델 일괄 그림 저장

    요구사항 3-4: 생성된 모델들을 그림으로 저장

    Parameters
    ----------
    models     : ResistivityModel 리스트
    output_dir : 저장 디렉토리
    prefix     : 파일명 접두사 (prefix_0001.png 형식)
    n_cols     : 개별 저장 외 격자 요약 그림의 열 수
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 개별 파일 저장
    for i, model in enumerate(models):
        fname = out / f"{prefix}_{i + 1:04d}.png"
        save_model_figure(
            model, str(fname), dpi=dpi,
            title=f"{prefix} {i + 1:04d}",
            **kwargs,
        )

    # 요약 격자 그림 (최대 20개)
    n_summary = min(len(models), 20)
    n_rows = (n_summary + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4, n_rows * 3),
    )
    axes_flat = np.array(axes).ravel()

    for i in range(n_summary):
        plot_resistivity_model(models[i], ax=axes_flat[i],
                               title=f"#{i + 1}", **kwargs)

    for j in range(n_summary, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Model Summary (1–{n_summary} of {len(models)})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_summary.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[visualize] {len(models)}개 모델 저장 완료: {out}/")
