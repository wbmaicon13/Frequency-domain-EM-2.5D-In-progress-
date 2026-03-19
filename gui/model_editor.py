"""
비저항 모델 GUI 편집기

요구사항 3-1, 3-2 대응:
  - 마우스 드래그로 이상대 추가/이동/크기 조정
  - 원형 / 사각형 / 다각형 이상대 지원
  - 이상대별 비저항 값 직접 입력
  - 설정 완료 후 변형 모델 생성(3-3) 연계

사용법:
    editor = ModelEditor(grid, base_model, base_anomalies, survey, profile)
    final_model, final_anomalies = editor.run()
"""

from __future__ import annotations

import sys
import copy
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
from matplotlib.colors import LogNorm

# 패키지 임포트 (scripts/ 또는 직접 실행 모두 지원)
try:
    from em25d.mesh.grid import Grid
    from em25d.mesh.profile import ProfileNodes
    from em25d.model.resistivity import ResistivityModel
    from em25d.model.anomaly import (
        Anomaly, CircleAnomaly, RectangleAnomaly, PolygonAnomaly, apply_anomalies
    )
    from em25d.survey.survey import Survey
except ImportError:
    # 모듈 경로 조정
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from em25d.mesh.grid import Grid
    from em25d.mesh.profile import ProfileNodes
    from em25d.model.resistivity import ResistivityModel
    from em25d.model.anomaly import (
        Anomaly, CircleAnomaly, RectangleAnomaly, PolygonAnomaly, apply_anomalies
    )
    from em25d.survey.survey import Survey


# ── 이상대 타입 상수 ─────────────────────────────────────────────────────────

ANOMALY_CIRCLE    = "circle"
ANOMALY_RECTANGLE = "rectangle"
ANOMALY_POLYGON   = "polygon"


class _DraftAnomaly:
    """
    GUI 에서 그리기 중인 이상대의 중간 상태

    드래그 완료 후 Anomaly 객체로 변환.
    """
    def __init__(self, kind: str, resistivity: float):
        self.kind        = kind
        self.resistivity = resistivity

        # 사각형/원형 공통
        self.x0: Optional[float] = None    # 드래그 시작 x
        self.z0: Optional[float] = None    # 드래그 시작 z
        self.x1: Optional[float] = None    # 드래그 끝 x
        self.z1: Optional[float] = None    # 드래그 끝 z

        # 다각형 전용
        self.polygon_pts: list[tuple[float, float]] = []

        # 현재 matplotlib patch
        self.patch: Optional[plt.Artist] = None

    def to_anomaly(self) -> Optional[Anomaly]:
        """드래프트 → Anomaly 변환"""
        if self.kind == ANOMALY_RECTANGLE:
            if None in (self.x0, self.z0, self.x1, self.z1):
                return None
            cx = 0.5 * (self.x0 + self.x1)
            cz = 0.5 * (self.z0 + self.z1)
            hw = 0.5 * abs(self.x1 - self.x0)
            hd = 0.5 * abs(self.z1 - self.z0)
            if hw < 1e-3 or hd < 1e-3:
                return None
            return RectangleAnomaly(
                center_x=cx, center_z=cz,
                half_width=hw, half_depth=hd,
                resistivity=self.resistivity,
            )

        if self.kind == ANOMALY_CIRCLE:
            if None in (self.x0, self.z0, self.x1, self.z1):
                return None
            cx = 0.5 * (self.x0 + self.x1)
            cz = 0.5 * (self.z0 + self.z1)
            r  = 0.5 * max(abs(self.x1 - self.x0), abs(self.z1 - self.z0))
            if r < 1e-3:
                return None
            return CircleAnomaly(
                center_x=cx, center_z=cz,
                radius=r,
                resistivity=self.resistivity,
            )

        if self.kind == ANOMALY_POLYGON:
            if len(self.polygon_pts) < 3:
                return None
            pts = np.array(self.polygon_pts)
            return PolygonAnomaly(
                vertices=pts,
                resistivity=self.resistivity,
            )

        return None


class ModelEditor:
    """
    비저항 모델 GUI 편집기

    Parameters
    ----------
    grid          : 2.5D FEM 격자
    base_model    : 초기 비저항 모델 (배경 + 기존 이상대)
    base_anomalies: 초기 이상대 목록
    survey        : 탐사 배열 (송수신기 위치 표시용)
    profile       : 수신기 프로파일 노드
    """

    def __init__(
        self,
        grid: Grid,
        base_model: ResistivityModel,
        base_anomalies: list[Anomaly],
        survey: Optional[Survey] = None,
        profile: Optional[ProfileNodes] = None,
    ):
        self.grid    = grid
        self.model   = copy.deepcopy(base_model)
        self.anomalies: list[Anomaly] = list(base_anomalies)
        self.survey  = survey
        self.profile = profile

        # 편집 상태
        self._anomaly_kind  = ANOMALY_RECTANGLE
        self._resistivity   = 10.0   # 새 이상대 기본 비저항 (Ω·m)
        self._drawing: Optional[_DraftAnomaly] = None
        self._press_xy: Optional[tuple] = None
        self._polygon_line = None    # 다각형 선 누적

        # 완료 플래그
        self._done = False
        self._cancelled = False

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def run(self) -> tuple[ResistivityModel, list[Anomaly]]:
        """
        GUI 편집기 실행

        Returns
        -------
        (final_model, final_anomalies)
        편집이 취소되면 원본 모델을 반환.
        """
        self._build_ui()
        plt.show()

        if self._cancelled:
            return self.model, self.anomalies

        # 최종 이상대를 모델에 적용
        apply_anomalies(self.model, self.anomalies)
        return self.model, self.anomalies

    # ── UI 구성 ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        """matplotlib figure 및 위젯 구성"""
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle("비저항 모델 편집기  —  Drag to add anomaly | Right-click: undo",
                          fontsize=11)

        # 좌측: 모델 단면도
        self.ax = self.fig.add_axes([0.06, 0.12, 0.60, 0.80])
        self._draw_model()

        # 우측 컨트롤 패널
        self._build_controls()

        # 이벤트 연결
        self.fig.canvas.mpl_connect("button_press_event",   self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    def _build_controls(self):
        """우측 컨트롤 위젯 배치"""
        # 이상대 형태 선택
        ax_radio = self.fig.add_axes([0.70, 0.70, 0.26, 0.18])
        self.radio = RadioButtons(
            ax_radio,
            labels=["사각형", "원형", "다각형"],
            active=0,
        )
        self.radio.on_clicked(self._on_radio)
        ax_radio.set_title("이상대 형태", fontsize=9)

        # 비저항 입력 (log10 슬라이더)
        ax_slider = self.fig.add_axes([0.70, 0.55, 0.26, 0.03])
        self.slider = Slider(
            ax_slider,
            label="log₁₀ρ",
            valmin=0.0,
            valmax=5.0,
            valinit=np.log10(self._resistivity),
            valstep=0.05,
        )
        self.slider.on_changed(self._on_slider)

        # 비저항 값 표시 (TextBox)
        ax_tbox = self.fig.add_axes([0.70, 0.48, 0.26, 0.04])
        init_rho_str = f"{self._resistivity:.1f}"
        self.tbox = TextBox(ax_tbox, "ρ (Ω·m)", initial=init_rho_str)
        self.tbox.on_submit(self._on_textbox)

        # 이상대 목록 레이블
        ax_list = self.fig.add_axes([0.70, 0.25, 0.26, 0.20])
        ax_list.axis("off")
        self._ax_list = ax_list
        self._update_anomaly_list()

        # 버튼: undo / clear / done / cancel
        ax_undo = self.fig.add_axes([0.70, 0.16, 0.12, 0.05])
        self.btn_undo = Button(ax_undo, "실행취소")
        self.btn_undo.on_clicked(self._on_undo)

        ax_clear = self.fig.add_axes([0.84, 0.16, 0.12, 0.05])
        self.btn_clear = Button(ax_clear, "전체삭제")
        self.btn_clear.on_clicked(self._on_clear)

        ax_done = self.fig.add_axes([0.70, 0.08, 0.12, 0.05])
        self.btn_done = Button(ax_done, "완료")
        self.btn_done.on_clicked(self._on_done)

        ax_cancel = self.fig.add_axes([0.84, 0.08, 0.12, 0.05])
        self.btn_cancel = Button(ax_cancel, "취소")
        self.btn_cancel.on_clicked(self._on_cancel)

    # ── 모델 단면도 그리기 ────────────────────────────────────────────────────

    def _draw_model(self):
        """비저항 모델 단면도 갱신"""
        self.ax.cla()

        # 요소 중심 좌표 및 비저항
        x_nodes = self.grid.node_x[:, 0]
        z_nodes = self.grid.node_z[0, :]
        xc = 0.5 * (x_nodes[:-1] + x_nodes[1:])
        zc = 0.5 * (z_nodes[:-1] + z_nodes[1:])
        XX, ZZ = np.meshgrid(xc, zc, indexing="ij")
        rho = self.model.element_resistivity   # (n_ex, n_ez)

        im = self.ax.pcolormesh(
            x_nodes, z_nodes, rho.T,
            norm=LogNorm(vmin=1.0, vmax=1e4),
            cmap="RdYlBu_r",
            shading="flat",
        )
        if not hasattr(self, "_colorbar"):
            self._colorbar = self.fig.colorbar(im, ax=self.ax, label="ρ (Ω·m)")

        # 탐사 배열 표시
        if self.survey is not None:
            for src in self.survey.sources.sources:
                self.ax.plot(src.x, src.z, "r*", ms=12, label="Tx", zorder=5)
        if self.profile is not None:
            x_rx = self.grid.node_x[self.profile.global_node_indices(), 0] \
                   if hasattr(self.profile, "global_node_indices") else []
            if len(x_rx):
                self.ax.plot(x_rx, np.zeros_like(x_rx), "bv", ms=5, label="Rx")

        self.ax.invert_yaxis()
        self.ax.set_xlabel("수평거리 (m)")
        self.ax.set_ylabel("깊이 (m)")
        self.ax.set_title("비저항 모델 단면도")
        self.ax.legend(loc="lower right", fontsize=8)
        self.fig.canvas.draw_idle()

    def _update_anomaly_list(self):
        """우측 이상대 목록 텍스트 갱신"""
        self._ax_list.cla()
        self._ax_list.axis("off")
        lines = [f"이상대 목록 ({len(self.anomalies)}개)\n"]
        for i, a in enumerate(self.anomalies):
            lines.append(f"  {i+1}. {a.__class__.__name__}  {a.resistivity:.1f} Ω·m")
        self._ax_list.text(0.02, 0.95, "\n".join(lines),
                           va="top", ha="left", fontsize=8,
                           transform=self._ax_list.transAxes)
        self.fig.canvas.draw_idle()

    # ── 이벤트 핸들러 ────────────────────────────────────────────────────────

    def _on_radio(self, label: str):
        label_map = {"사각형": ANOMALY_RECTANGLE, "원형": ANOMALY_CIRCLE, "다각형": ANOMALY_POLYGON}
        self._anomaly_kind = label_map.get(label, ANOMALY_RECTANGLE)

    def _on_slider(self, val: float):
        self._resistivity = 10.0 ** val
        self.tbox.set_val(f"{self._resistivity:.2f}")

    def _on_textbox(self, text: str):
        try:
            rho = float(text)
            if rho > 0:
                self._resistivity = rho
                self.slider.set_val(np.log10(rho))
        except ValueError:
            pass

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 3:   # 우클릭 → undo
            self._on_undo(None)
            return

        x, z = event.xdata, event.ydata
        if x is None or z is None:
            return

        if self._anomaly_kind == ANOMALY_POLYGON:
            if self._drawing is None:
                self._drawing = _DraftAnomaly(ANOMALY_POLYGON, self._resistivity)
            self._drawing.polygon_pts.append((x, z))
            # 더블클릭으로 완료
            if event.dblclick and len(self._drawing.polygon_pts) >= 3:
                self._finalize_anomaly()
            else:
                self._update_polygon_preview()
        else:
            self._press_xy = (x, z)
            self._drawing = _DraftAnomaly(self._anomaly_kind, self._resistivity)
            self._drawing.x0, self._drawing.z0 = x, z

    def _on_motion(self, event):
        if event.inaxes != self.ax or self._drawing is None:
            return
        if self._anomaly_kind == ANOMALY_POLYGON:
            return
        if self._press_xy is None:
            return

        x, z = event.xdata, event.ydata
        if x is None or z is None:
            return

        self._drawing.x1, self._drawing.z1 = x, z
        self._update_draft_patch()

    def _on_release(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return
        if self._anomaly_kind == ANOMALY_POLYGON:
            return
        if self._drawing is None:
            return

        x, z = event.xdata, event.ydata
        if x is None or z is None:
            self._drawing = None
            return

        self._drawing.x1, self._drawing.z1 = x, z
        self._finalize_anomaly()

    def _update_draft_patch(self):
        """드래그 중 점선 미리보기 patch 갱신"""
        if self._drawing is None:
            return

        # 기존 patch 제거
        if self._drawing.patch is not None:
            self._drawing.patch.remove()
            self._drawing.patch = None

        d = self._drawing
        if None in (d.x0, d.z0, d.x1, d.z1):
            return

        if d.kind == ANOMALY_RECTANGLE:
            x_lo = min(d.x0, d.x1)
            z_lo = min(d.z0, d.z1)
            w    = abs(d.x1 - d.x0)
            h    = abs(d.z1 - d.z0)
            patch = patches.Rectangle(
                (x_lo, z_lo), w, h,
                linewidth=1.5, edgecolor="yellow",
                facecolor="yellow", alpha=0.3, linestyle="--",
            )
        elif d.kind == ANOMALY_CIRCLE:
            cx = 0.5 * (d.x0 + d.x1)
            cz = 0.5 * (d.z0 + d.z1)
            r  = 0.5 * max(abs(d.x1 - d.x0), abs(d.z1 - d.z0))
            patch = patches.Ellipse(
                (cx, cz), width=2 * r, height=2 * r,
                linewidth=1.5, edgecolor="yellow",
                facecolor="yellow", alpha=0.3, linestyle="--",
            )
        else:
            return

        self._drawing.patch = self.ax.add_patch(patch)
        self.fig.canvas.draw_idle()

    def _update_polygon_preview(self):
        """다각형 꼭짓점 미리보기 선 갱신"""
        if self._drawing is None:
            return
        pts = self._drawing.polygon_pts
        if len(pts) < 2:
            return

        if self._polygon_line is not None:
            self._polygon_line.remove()

        xs = [p[0] for p in pts] + [pts[0][0]]
        zs = [p[1] for p in pts] + [pts[0][1]]
        self._polygon_line, = self.ax.plot(
            xs, zs, "y--", linewidth=1.5)
        self.fig.canvas.draw_idle()

    def _finalize_anomaly(self):
        """드래프트 이상대를 확정하고 모델에 적용"""
        if self._drawing is None:
            return

        anomaly = self._drawing.to_anomaly()
        if anomaly is not None:
            self.anomalies.append(anomaly)
            anomaly.apply(self.model)
            self._draw_model()
            self._update_anomaly_list()

        # 정리
        if self._drawing.patch is not None:
            try:
                self._drawing.patch.remove()
            except Exception:
                pass
        if self._polygon_line is not None:
            try:
                self._polygon_line.remove()
            except Exception:
                pass
            self._polygon_line = None

        self._drawing = None
        self._press_xy = None

    def _on_undo(self, _event):
        """마지막 이상대 제거"""
        if not self.anomalies:
            return
        self.anomalies.pop()

        # 모델 재구성 (배경 → 이상대 순서로 재적용)
        self.model.set_block_resistivity(
            np.full(self.model.n_blocks,
                    self.model.background_resistivity, dtype=float)
        )
        apply_anomalies(self.model, self.anomalies)
        self._draw_model()
        self._update_anomaly_list()

    def _on_clear(self, _event):
        """모든 이상대 제거"""
        self.anomalies.clear()
        self.model.set_block_resistivity(
            np.full(self.model.n_blocks,
                    self.model.background_resistivity, dtype=float)
        )
        self._draw_model()
        self._update_anomaly_list()

    def _on_done(self, _event):
        """편집 완료"""
        self._done = True
        plt.close(self.fig)

    def _on_cancel(self, _event):
        """편집 취소"""
        self._cancelled = True
        plt.close(self.fig)
