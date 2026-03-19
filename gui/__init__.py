"""
GUI 모듈 — 비저항 모델 편집기 + 결과 뷰어

요구사항 3-2, 3-4 대응.
Matplotlib 기반 interactive 편집기 (tkinter 백엔드 권장).
"""

from .model_editor import ModelEditor
from .result_viewer import ResultViewer

__all__ = ["ModelEditor", "ResultViewer"]
