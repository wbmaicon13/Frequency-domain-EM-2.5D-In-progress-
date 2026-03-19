"""
파라미터 파일 읽기/쓰기

Fortran 대응: Fem25Dinv.par + Read_Par_Inversion 서브루틴

Fortran 의 고정 포맷 텍스트 파라미터 파일을 YAML 로 대체.
기존 .par 파일도 읽을 수 있도록 레거시 파서 제공.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
from ..constants import NormType


# ── 역산 설정 ────────────────────────────────────────────────────────────────

@dataclass
class InversionParams:
    """
    역산 제어 파라미터

    Fortran 대응: Fem25Dinv.par 의 'Parameter block for inversion control'
    """
    run_inversion: bool = False           # 역산 실행 여부 (Fortran: Inversion)
    restart: bool = False                 # 이전 역산 재시작 여부
    max_iterations: int = 10             # 최대 반복 횟수 (iter_max)
    iteration_type: int = 1              # 0: creeping, 1: jumping

    scaling_factor: float = -5.0e-15    # 감쇠 인자 스케일

    # IRLS (Iteratively Reweighted Least Squares)
    irls_data: NormType = NormType.L2    # 데이터 노름 (1:L1, 2:L2, 3:HB, 4:BW)
    irls_model: NormType = NormType.L2  # 모델 노름
    irls_start: int = 1                  # IRLS 시작 반복 번호

    # 비저항 범위
    resistivity_min: float = 0.1        # [Ω·m]
    resistivity_max: float = 1.0e5      # [Ω·m]

    # 사용할 장 성분 (0: 미사용, 1: 허수부만, 2: 실수+허수)
    use_Ex: int = 0
    use_Ey: int = 0
    use_Ez: int = 0
    use_Hx: int = 0
    use_Hy: int = 0
    use_Hz: int = 1

    # ACB / Occam
    use_acb: bool = True                # ACB (Active Constraint Balancing)
    use_occam: bool = True              # Occam 역산 (최평활)

    # ACB 미사용 시 평활화 비율
    smoothness_vertical: float = 0.5
    smoothness_horizontal: float = 0.5

    # 데이터 가중치
    gamma: float = 1.0                  # 데이터 오차 가중
    whitening_factor: float = 1.0e-9   # 배경 잡음 비율

    # 제약조건 시퀀스 (반복별 on/off)
    sequence_constraint: list = field(default_factory=lambda: [1]*14)

    # 시퀀스 가중치
    sequence_norm: NormType = NormType.L2
    sequence_scale: float = 1.0
    sequence_normalize: bool = False

    # 데이터 수 및 가중치 (주파수별)
    n_data_per_freq: int = 3
    data_weights: list = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # 기준 모델 사용 여부
    use_reference_model: bool = False

    # 모델 선호도 (뮤, 반복별)
    model_preference: list = field(default_factory=lambda: [0.1, 0.1])

    # 역산 로그 디렉토리
    log_dir: str = "./inversion_log"


# ── 순방향 설정 ──────────────────────────────────────────────────────────────

@dataclass
class ForwardParams:
    """
    순방향 모델링 파라미터

    Fortran 대응: Fem25Dinv.par 의 'Parameter block for forward control'
    """
    source_index_start: int = 1         # 송신기 시작 인덱스 (Fortran 1-based)
    source_index_end: int = 1           # 송신기 끝 인덱스

    n_wavenumbers: int = 20             # 공간주파수(ky) 개수

    file_type: int = 1                  # 0: binary, 1: ascii
    field_type: int = 0                 # 0: total field, 1: secondary field

    # 송신기 강도·길이 (송신기별)
    # [(index, strength, length), ...]
    sources: list = field(default_factory=list)

    # 송신기 활성화 여부 (주파수 × 송신기)
    source_active: list = field(default_factory=list)

    # 송수신기 위치 지정 방법 (0: Modeler, 1: 파일)
    src_rec_from_file: bool = False

    # 1차장 계산 방법
    calculate_primary: bool = True      # False: 저장된 파일 사용

    # 입력 디렉토리
    input_dir: str = "./"
    primary_dir: str = "./output_primary"
    output_dir: str = "./output_data"


# ── 데이터 설정 ──────────────────────────────────────────────────────────────

@dataclass
class DataParams:
    """
    관측 데이터 파라미터

    Fortran 대응: Fem25Dinv.par 의 'Parameter block for field data'
    """
    data_file: str = "Fem25Dinv_inv.inp"
    data_unit_gamma: bool = False       # True: gamma 단위 사용
    background_resistivity: float = 100.0  # 초기 배경 비저항 [Ω·m]


# ── 전체 설정 ────────────────────────────────────────────────────────────────

@dataclass
class Em25dConfig:
    """
    EM 2.5D 전체 설정 (Fem25Dinv.par 전체 대체)

    사용 예시
    ---------
    >>> cfg = Em25dConfig.from_yaml("config/default_params.yaml")
    >>> cfg.inversion.max_iterations = 20
    >>> cfg.to_yaml("config/my_run.yaml")
    """
    inversion: InversionParams = field(default_factory=InversionParams)
    forward: ForwardParams = field(default_factory=ForwardParams)
    data: DataParams = field(default_factory=DataParams)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Em25dConfig":
        """YAML 파일에서 설정 읽기"""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        inv_raw = raw.get("inversion", {})
        fwd_raw = raw.get("forward", {})
        dat_raw = raw.get("data", {})

        # NormType 문자열 → enum 변환
        for key in ("irls_data", "irls_model", "sequence_norm"):
            if key in inv_raw and isinstance(inv_raw[key], str):
                inv_raw[key] = NormType[inv_raw[key].upper()]

        return cls(
            inversion=InversionParams(**inv_raw),
            forward=ForwardParams(**fwd_raw),
            data=DataParams(**dat_raw),
        )

    def to_yaml(self, path: str | Path) -> None:
        """YAML 파일로 설정 저장"""
        def _convert(obj):
            if isinstance(obj, NormType):
                return obj.name
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(i) for i in obj]
            return obj

        raw = _convert(asdict(self))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(raw, f, allow_unicode=True, sort_keys=False)

    @classmethod
    def from_fortran_par(cls, path: str | Path) -> "Em25dConfig":
        """
        기존 Fortran .par 파일 파싱 (레거시 호환)

        Fortran 대응: Read_Par_Inversion 서브루틴
        """
        return _parse_fortran_par(path)

    def summary(self) -> str:
        inv = self.inversion
        fwd = self.forward
        dat = self.data
        lines = [
            "=== EM 2.5D 설정 요약 ===",
            f"  모드          : {'역산' if inv.run_inversion else '순방향'}",
            f"  최대 반복     : {inv.max_iterations}",
            f"  공간주파수 수 : {fwd.n_wavenumbers}",
            f"  데이터 파일   : {dat.data_file}",
            f"  배경 비저항   : {dat.background_resistivity} Ω·m",
            f"  비저항 범위   : [{inv.resistivity_min}, {inv.resistivity_max}] Ω·m",
            f"  사용 성분     : Ex={inv.use_Ex} Ey={inv.use_Ey} Ez={inv.use_Ez}"
            f" Hx={inv.use_Hx} Hy={inv.use_Hy} Hz={inv.use_Hz}",
        ]
        return "\n".join(lines)


# ── Fortran .par 레거시 파서 ─────────────────────────────────────────────────

def _parse_fortran_par(path: str | Path) -> Em25dConfig:
    """
    Fortran 고정 포맷 .par 파일 파싱

    주석 행(>> 로 시작하거나 -- 로 시작)은 건너뜀.
    데이터는 공백/탭으로 구분된 토큰으로 읽음.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    tokens = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("-", ">")):
            continue
        tokens.extend(stripped.split())

    it = iter(tokens)

    def nxt():
        return next(it)

    def read_bool():
        v = nxt().lower()
        return v in (".true.", "true", "1", "t")

    def read_int():
        return int(nxt())

    def read_float():
        return float(nxt().replace("D", "e").replace("d", "e"))

    cfg = Em25dConfig()
    inv = cfg.inversion
    fwd = cfg.forward
    dat = cfg.data

    # ── 역산 블록 ──────────────────────────────────────────────────────────
    # 입력 디렉토리
    fwd.input_dir = nxt()

    # Inversion, Restart
    inv.run_inversion = read_bool()
    inv.restart = read_bool()

    # 최대 반복
    inv.max_iterations = read_int()

    # 반복 유형
    inv.iteration_type = read_int()

    # 스케일링 인자
    inv.scaling_factor = read_float()

    # IRLS
    irls_d = read_int()
    irls_m = read_int()
    inv.irls_data = NormType(irls_d - 1)   # .par: 1=L1,2=LS → enum: 0=L2,1=L1
    inv.irls_model = NormType(irls_m - 1)
    inv.irls_start = read_int()

    # 비저항 범위
    inv.resistivity_min = read_float()
    inv.resistivity_max = read_float()

    # 사용 성분
    inv.use_Ex = read_int()
    inv.use_Ey = read_int()
    inv.use_Ez = read_int()
    inv.use_Hx = read_int()
    inv.use_Hy = read_int()
    inv.use_Hz = read_int()

    # ACB, Occam
    inv.use_acb = bool(read_int())
    inv.use_occam = bool(read_int())

    # 평활화 비율
    inv.smoothness_vertical = read_float()
    inv.smoothness_horizontal = read_float()

    # 데이터 가중치
    inv.gamma = read_float()
    inv.whitening_factor = read_float()

    # 시퀀스 제약 (14개)
    inv.sequence_constraint = [read_int() for _ in range(14)]

    # 시퀀스 가중
    seq_norm = read_int()
    inv.sequence_norm = NormType(seq_norm - 1)
    inv.sequence_scale = read_float()
    inv.sequence_normalize = bool(read_int())

    # 데이터 수 및 가중치
    inv.n_data_per_freq = read_int()
    inv.data_weights = [read_float() for _ in range(inv.n_data_per_freq)]

    # 기준 모델
    inv.use_reference_model = bool(read_int())

    # 모델 선호도 (2개)
    inv.model_preference = [read_float(), read_float()]

    # ── 순방향 블록 ────────────────────────────────────────────────────────
    fwd.source_index_start = read_int()
    fwd.source_index_end = read_int()
    fwd.n_wavenumbers = read_int()
    fwd.file_type = read_int()
    fwd.field_type = read_int()

    # 송신기 정보 (source_index_end 개)
    n_src = fwd.source_index_end
    fwd.sources = []
    for _ in range(n_src):
        idx = read_int()
        strength = read_float()
        length = read_float()
        fwd.sources.append({"index": idx, "strength": strength, "length": length})

    # 송신기 활성화 (n_src × n_freq — 여기서는 간단히 플랫 리스트)
    # 실제 주파수 수는 데이터 읽기 후 결정되므로 일단 남은 정수 토큰 수집
    active = []
    try:
        while True:
            v = nxt()
            if not v.lstrip("-").isdigit():
                # 더 이상 정수가 아니면 다음 필드
                # 먼저 이 토큰을 src_rec_from_file 로 사용
                fwd.src_rec_from_file = bool(int(v))
                break
            active.append(int(v))
    except StopIteration:
        pass
    fwd.source_active = active

    # 1차장 계산 여부
    fwd.calculate_primary = not bool(read_int())

    # ── 데이터 블록 ────────────────────────────────────────────────────────
    dat.data_file = nxt()
    dat.data_unit_gamma = bool(read_int())
    dat.background_resistivity = read_float()

    return cfg
