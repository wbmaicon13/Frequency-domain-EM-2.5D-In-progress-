"""
역산/순방향 결과 뷰어

요구사항 3-4 대응:
  - 비저항 모델 단면도
  - 관측/계산 데이터 비교 (주파수별, 성분별)
  - 역산 오차 수렴 곡선
  - 그림 저장 옵션

사용법:
    viewer = ResultViewer(grid, result_model, observed, synthetic)
    viewer.show()
    viewer.save("results/summary.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatter

try:
    from em25d.mesh.grid import Grid
    from em25d.model.resistivity import ResistivityModel
    from em25d.survey.survey import Survey
    from em25d.inverse.inversion_loop import InversionResult
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from em25d.mesh.grid import Grid
    from em25d.model.resistivity import ResistivityModel
    from em25d.survey.survey import Survey
    from em25d.inverse.inversion_loop import InversionResult


# 성분 이름 매핑
_COMPONENT_NAMES = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]


class ResultViewer:
    """
    순방향 / 역산 결과 시각화기

    Parameters
    ----------
    grid          : 2.5D FEM 격자
    model         : 결과 비저항 모델 (역산 최종 또는 입력 모델)
    survey        : 탐사 배열 (주파수 및 성분 정보용)
    observed      : 관측 데이터 (n_freq, n_tx, n_rx, 6)  — None 가능
    synthetic     : 계산 데이터 (n_freq, n_tx, n_rx, 6)  — None 가능
    inv_result    : 역산 결과 객체 (수렴 곡선 포함)        — None 가능
    true_model    : 참조/참값 모델 (검증용)                — None 가능
    """

    def __init__(
        self,
        grid: Grid,
        model: ResistivityModel,
        survey: Optional[Survey] = None,
        observed: Optional[np.ndarray] = None,
        synthetic: Optional[np.ndarray] = None,
        inv_result: Optional["InversionResult"] = None,
        true_model: Optional[ResistivityModel] = None,
    ):
        self.grid        = grid
        self.model       = model
        self.survey      = survey
        self.observed    = observed
        self.synthetic   = synthetic
        self.inv_result  = inv_result
        self.true_model  = true_model

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def show(self):
        """모든 패널을 포함하는 요약 그림 표시"""
        fig = self._build_summary_figure()
        plt.show()
        return fig

    def save(self, path: str, dpi: int = 150):
        """요약 그림을 파일로 저장"""
        fig = self._build_summary_figure()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[ResultViewer] 그림 저장: {path}")

    def plot_model(
        self,
        ax: Optional[plt.Axes] = None,
        title: str = "비저항 모델",
        show_survey: bool = True,
    ) -> plt.Axes:
        """비저항 모델 단면도"""
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 5))

        x_nodes = self.grid.node_x[:, 0]
        z_nodes = self.grid.node_z[0, :]
        rho     = self.model.element_resistivity

        im = ax.pcolormesh(
            x_nodes, z_nodes, rho.T,
            norm=LogNorm(vmin=max(rho.min(), 0.01), vmax=min(rho.max(), 1e5)),
            cmap="RdYlBu_r",
            shading="flat",
        )
        plt.colorbar(im, ax=ax, label="ρ (Ω·m)")

        if show_survey and self.survey is not None:
            for src in self.survey.sources.sources:
                ax.plot(src.x, src.z, "r*", ms=14, label="Tx", zorder=5)

        ax.invert_yaxis()
        ax.set_xlabel("수평거리 (m)")
        ax.set_ylabel("깊이 (m)")
        ax.set_title(title)
        return ax

    def plot_data_comparison(
        self,
        freq_index: int = 0,
        tx_index: int = 0,
        component_index: int = 4,   # 기본: Hy
        ax_re: Optional[plt.Axes] = None,
        ax_im: Optional[plt.Axes] = None,
    ) -> tuple[plt.Axes, plt.Axes]:
        """관측 vs 계산 데이터 비교 (실수부 + 허수부)"""
        if self.observed is None and self.synthetic is None:
            raise ValueError("observed 또는 synthetic 데이터가 없습니다.")

        comp = _COMPONENT_NAMES[component_index]
        n_rx = (self.observed if self.observed is not None else self.synthetic).shape[2]
        rx_x = np.linspace(-1, 1, n_rx)   # 임시 수평축 (실제 좌표 필요 시 profile 사용)

        if ax_re is None or ax_im is None:
            _, (ax_re, ax_im) = plt.subplots(1, 2, figsize=(12, 4))

        freq = (self.survey.frequencies.frequencies[freq_index]
                if self.survey else f"freq{freq_index}")

        for ax, part, label in [(ax_re, np.real, "실수부"), (ax_im, np.imag, "허수부")]:
            if self.observed is not None:
                d_obs = part(self.observed[freq_index, tx_index, :, component_index])
                ax.plot(rx_x, d_obs, "ko-", ms=4, lw=1.2, label="관측")
            if self.synthetic is not None:
                d_syn = part(self.synthetic[freq_index, tx_index, :, component_index])
                ax.plot(rx_x, d_syn, "r--", lw=1.5, label="계산")
            ax.set_xlabel("수신기 위치")
            ax.set_ylabel(f"{comp} {label}")
            ax.set_title(f"{comp} {label}  (f={freq} Hz)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.4)

        return ax_re, ax_im

    def plot_convergence(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """역산 오차 수렴 곡선"""
        if self.inv_result is None:
            raise ValueError("역산 결과(InversionResult) 가 없습니다.")

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        rms_log = self.inv_result.rms_history
        iters   = np.arange(1, len(rms_log) + 1)
        ax.semilogy(iters, rms_log, "b-o", ms=5, lw=1.5)
        ax.set_xlabel("반복 횟수")
        ax.set_ylabel("RMS 오차")
        ax.set_title("역산 수렴 곡선")
        ax.grid(True, which="both", alpha=0.4)
        ax.set_xticks(iters)
        return ax

    def plot_model_comparison(
        self,
        ax_inv: Optional[plt.Axes] = None,
        ax_true: Optional[plt.Axes] = None,
    ) -> tuple[plt.Axes, plt.Axes]:
        """역산 모델 vs 참값 모델 나란히 비교"""
        if self.true_model is None:
            raise ValueError("참값 모델(true_model) 이 없습니다.")

        if ax_inv is None or ax_true is None:
            _, (ax_inv, ax_true) = plt.subplots(1, 2, figsize=(14, 5))

        self.plot_model(ax_inv, title="역산 모델")

        # 임시로 true_model 단면도 그리기
        rho_t = self.true_model.element_resistivity
        x_nodes = self.grid.node_x[:, 0]
        z_nodes = self.grid.node_z[0, :]
        im = ax_true.pcolormesh(
            x_nodes, z_nodes, rho_t.T,
            norm=LogNorm(vmin=max(rho_t.min(), 0.01), vmax=min(rho_t.max(), 1e5)),
            cmap="RdYlBu_r", shading="flat",
        )
        plt.colorbar(im, ax=ax_true, label="ρ (Ω·m)")
        ax_true.invert_yaxis()
        ax_true.set_xlabel("수평거리 (m)")
        ax_true.set_ylabel("깊이 (m)")
        ax_true.set_title("참값 모델")

        return ax_inv, ax_true

    # ── 요약 그림 구성 ────────────────────────────────────────────────────────

    def _build_summary_figure(self) -> plt.Figure:
        """모든 패널을 포함하는 요약 Figure 구성"""
        n_rows = 2 + (1 if self.inv_result is not None else 0)
        n_cols = 2 + (1 if self.true_model is not None else 0)

        fig = plt.figure(figsize=(6 * n_cols, 5 * n_rows))
        gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.35)

        row = 0

        # 비저항 모델
        ax_model = fig.add_subplot(gs[row, 0])
        self.plot_model(ax_model, title="비저항 모델 (역산 결과)")

        # 참값 모델
        if self.true_model is not None:
            ax_true = fig.add_subplot(gs[row, 1])
            rho_t   = self.true_model.element_resistivity
            x_nodes = self.grid.node_x[:, 0]
            z_nodes = self.grid.node_z[0, :]
            im = ax_true.pcolormesh(
                x_nodes, z_nodes, rho_t.T,
                norm=LogNorm(vmin=max(rho_t.min(), 0.01), vmax=min(rho_t.max(), 1e5)),
                cmap="RdYlBu_r", shading="flat",
            )
            plt.colorbar(im, ax=ax_true, label="ρ (Ω·m)")
            ax_true.invert_yaxis()
            ax_true.set_xlabel("수평거리 (m)")
            ax_true.set_ylabel("깊이 (m)")
            ax_true.set_title("참값 모델")

        row += 1

        # 데이터 비교 (첫 번째 주파수, Hy 성분)
        if self.observed is not None or self.synthetic is not None:
            ax_re = fig.add_subplot(gs[row, 0])
            ax_im = fig.add_subplot(gs[row, 1])
            try:
                self.plot_data_comparison(
                    freq_index=0, component_index=4,
                    ax_re=ax_re, ax_im=ax_im,
                )
            except Exception:
                ax_re.text(0.5, 0.5, "데이터 없음", ha="center", va="center",
                           transform=ax_re.transAxes)
            row += 1

        # 수렴 곡선
        if self.inv_result is not None:
            ax_conv = fig.add_subplot(gs[row, 0])
            try:
                self.plot_convergence(ax_conv)
            except Exception:
                ax_conv.text(0.5, 0.5, "수렴 곡선 없음", ha="center", va="center",
                             transform=ax_conv.transAxes)

        fig.suptitle("EM 2.5D 역산 결과 요약", fontsize=13, fontweight="bold")
        return fig
