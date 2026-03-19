"""
관측/계산 데이터 I/O

Fortran 대응:
  - Read_data 서브루틴 (Fem25Dinv_inv.inp 읽기)
  - output_data/ 디렉토리 출력 파일

관측 데이터 파일 형식 (Fem25Dinv_inv.inp):
  # 주파수 수
  n_freq
  # 각 주파수별
  frequency [Hz]
  n_transmitters
  # 각 송신기별
  tx_x  tx_z  n_receivers
  rx_x  rx_z  [Ex_real Ex_imag] [Ey_real ...] ...  (사용 성분만)

계산 결과 저장:
  - NPZ (권장): 모든 성분 한 번에
  - CSV: 사람이 읽을 수 있는 형식
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ObservedData:
    """
    관측 전자기장 데이터

    Fortran 대응: femarr 모듈의 관측 데이터 배열
    """
    frequencies: np.ndarray          # (n_freq,) [Hz]
    source_x: np.ndarray             # (n_freq, n_tx) 송신기 x [m]
    source_z: np.ndarray             # (n_freq, n_tx) 송신기 z [m]
    receiver_x: np.ndarray           # (n_freq, n_tx, n_rx) 수신기 x [m]
    receiver_z: np.ndarray           # (n_freq, n_tx, n_rx) 수신기 z [m]

    # 관측 장 성분 (n_freq, n_tx, n_rx, 6) — [Ex,Ey,Ez,Hx,Hy,Hz]
    observed: np.ndarray
    # 데이터 오차 (동일 shape, 없으면 None)
    error: Optional[np.ndarray] = None

    # 사용 성분 플래그 (6,) bool
    use_components: np.ndarray = field(
        default_factory=lambda: np.array([False,False,False,False,False,True]))

    @property
    def n_freq(self) -> int:
        return len(self.frequencies)

    @property
    def n_transmitters(self) -> int:
        return self.source_x.shape[1]

    @property
    def n_receivers(self) -> int:
        return self.receiver_x.shape[2]


def load_observed_data(
    path: str | Path,
    use_components: list[bool] = None,
) -> ObservedData:
    """
    Fortran 관측 데이터 파일 읽기

    Fortran 대응: Read_data 서브루틴 (Fem25Dinv_inv.inp)

    Parameters
    ----------
    path           : 데이터 파일 경로
    use_components : [Ex,Ey,Ez,Hx,Hy,Hz] 사용 여부 (None = Hz만 사용)
    """
    if use_components is None:
        use_components = [False, False, False, False, False, True]
    use_arr = np.array(use_components, dtype=bool)
    n_comp_used = use_arr.sum()

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    tokens = _tokenize(lines)
    it = iter(tokens)

    def nxt_float():
        return float(next(it).replace("D", "e").replace("d", "e"))

    def nxt_int():
        return int(next(it))

    n_freq = nxt_int()
    freq_list, sx_list, sz_list = [], [], []
    rx_list, rz_list, obs_list, err_list = [], [], [], []

    for _ in range(n_freq):
        freq = nxt_float()
        freq_list.append(freq)
        n_tx = nxt_int()

        sx_freq, sz_freq = [], []
        rx_freq, rz_freq, obs_freq, err_freq = [], [], [], []

        for _ in range(n_tx):
            tx_x = nxt_float()
            tx_z = nxt_float()
            n_rx = nxt_int()
            sx_freq.append(tx_x)
            sz_freq.append(tx_z)

            rx_tx, rz_tx, obs_tx, err_tx = [], [], [], []
            for _ in range(n_rx):
                rx_x = nxt_float()
                rx_z = nxt_float()
                rx_tx.append(rx_x)
                rz_tx.append(rx_z)
                # 사용 성분별 실수부 + 허수부
                obs_rx = np.zeros(6, dtype=complex)
                err_rx = np.ones(6, dtype=float)
                for ic, used in enumerate(use_arr):
                    if used:
                        re = nxt_float()
                        im = nxt_float()
                        obs_rx[ic] = complex(re, im)
                        # 오차가 있으면 읽기 (선택적)
                        try:
                            err_rx[ic] = nxt_float()
                        except StopIteration:
                            pass
                obs_tx.append(obs_rx)
                err_tx.append(err_rx)

            rx_freq.append(rx_tx)
            rz_freq.append(rz_tx)
            obs_freq.append(obs_tx)
            err_freq.append(err_tx)

        sx_list.append(sx_freq)
        sz_list.append(sz_freq)
        rx_list.append(rx_freq)
        rz_list.append(rz_freq)
        obs_list.append(obs_freq)
        err_list.append(err_freq)

    n_tx_max = max(len(s) for s in sx_list)
    n_rx_max = max(len(r) for freq in rx_list for r in freq)

    def pad_3d(lst, n_f, n_t, n_r, fill=0.0, dtype=float):
        arr = np.full((n_f, n_t, n_r), fill, dtype=dtype)
        for i, freq in enumerate(lst):
            for j, tx in enumerate(freq):
                arr[i, j, :len(tx)] = tx
        return arr

    return ObservedData(
        frequencies=np.array(freq_list),
        source_x=np.array([[x for x in row] for row in sx_list], dtype=float),
        source_z=np.array([[z for z in row] for row in sz_list], dtype=float),
        receiver_x=pad_3d(rx_list, n_freq, n_tx_max, n_rx_max, dtype=float),
        receiver_z=pad_3d(rz_list, n_freq, n_tx_max, n_rx_max, dtype=float),
        observed=pad_3d(obs_list, n_freq, n_tx_max, n_rx_max,
                        fill=0j, dtype=complex),
        error=pad_3d(err_list, n_freq, n_tx_max, n_rx_max,
                     fill=1.0, dtype=float),
        use_components=use_arr,
    )


def save_synthetic_data(
    synthetic: np.ndarray,    # (n_freq, n_tx, n_rx, 6)
    observed_data: ObservedData,
    path: str | Path,
    fmt: str = "npz",
) -> None:
    """
    순방향 계산 결과 저장

    Parameters
    ----------
    synthetic     : 계산된 전자기장 (n_freq, n_tx, n_rx, 6)
    observed_data : 관측 데이터 (격자/주파수 정보 참조)
    path          : 저장 경로
    fmt           : "npz" | "csv"
    """
    path = Path(path)

    if fmt == "npz":
        np.savez_compressed(
            path.with_suffix(".npz"),
            synthetic=synthetic,
            frequencies=observed_data.frequencies,
            source_x=observed_data.source_x,
            source_z=observed_data.source_z,
            receiver_x=observed_data.receiver_x,
            receiver_z=observed_data.receiver_z,
        )
    elif fmt == "csv":
        _save_synthetic_csv(synthetic, observed_data, path.with_suffix(".csv"))
    else:
        raise ValueError(f"지원하지 않는 포맷: {fmt!r}")


def _save_synthetic_csv(
    synthetic: np.ndarray,
    obs: ObservedData,
    path: Path,
) -> None:
    comp_names = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    header = "freq_hz,tx_x_m,tx_z_m,rx_x_m,rx_z_m"
    for cn in comp_names:
        header += f",{cn}_real,{cn}_imag"

    rows = []
    n_freq, n_tx, n_rx, _ = synthetic.shape
    for ifreq in range(n_freq):
        for itx in range(n_tx):
            for irx in range(n_rx):
                row = [
                    obs.frequencies[ifreq],
                    obs.source_x[ifreq, itx],
                    obs.source_z[ifreq, itx],
                    obs.receiver_x[ifreq, itx, irx],
                    obs.receiver_z[ifreq, itx, irx],
                ]
                for ic in range(6):
                    v = synthetic[ifreq, itx, irx, ic]
                    row += [v.real, v.imag]
                rows.append(row)

    np.savetxt(path, rows, delimiter=",", header=header, comments="")


def save_synthetic_inp(
    synthetic: np.ndarray,       # (n_freq, n_tx, n_rx, 6)
    frequencies: np.ndarray,     # (n_freq,) [Hz]
    source_x: np.ndarray,        # (n_tx,) or (n_freq, n_tx)
    source_z: np.ndarray,        # (n_tx,) or (n_freq, n_tx)
    receiver_x: np.ndarray,      # (n_rx,) or (n_freq, n_tx, n_rx)
    receiver_z: np.ndarray,      # (n_rx,) or (n_freq, n_tx, n_rx)
    path: str | Path,
    use_components: list[bool] = None,
) -> None:
    """
    순방향 계산 결과를 Fortran .inp 형식으로 저장

    Fortran 대응: Fem25Dinv_inv.inp (Read_data가 읽는 형식)

    형식:
      n_freq
      frequency [Hz]
      n_tx
      tx_x  tx_z  n_rx
      rx_x  rx_z  comp1_re comp1_im [comp2_re comp2_im ...]
      ...

    Parameters
    ----------
    synthetic      : (n_freq, n_tx, n_rx, 6) 계산된 전자기장 [Ex,Ey,Ez,Hx,Hy,Hz]
    frequencies    : (n_freq,) 주파수 배열 [Hz]
    source_x/z     : 송신기 좌표 (1D: 모든 주파수 동일, 2D: 주파수별)
    receiver_x/z   : 수신기 좌표 (1D: 모든 tx/freq 동일, 3D: 주파수/tx별)
    path           : 저장 경로
    use_components : [Ex,Ey,Ez,Hx,Hy,Hz] 출력 성분 (None = 전체)
    """
    if use_components is None:
        use_components = [True, True, True, True, True, True]
    use_arr = np.array(use_components, dtype=bool)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_freq, n_tx, n_rx, _ = synthetic.shape

    # 좌표 배열 차원 통일
    sx = np.atleast_2d(source_x)     # (n_freq, n_tx) or (1, n_tx)
    sz = np.atleast_2d(source_z)
    if sx.shape[0] == 1:
        sx = np.broadcast_to(sx, (n_freq, n_tx))
        sz = np.broadcast_to(sz, (n_freq, n_tx))

    rx = np.array(receiver_x)
    rz = np.array(receiver_z)
    if rx.ndim == 1:
        rx = np.broadcast_to(rx[np.newaxis, np.newaxis, :], (n_freq, n_tx, n_rx))
        rz = np.broadcast_to(rz[np.newaxis, np.newaxis, :], (n_freq, n_tx, n_rx))
    elif rx.ndim == 2:
        rx = np.broadcast_to(rx[np.newaxis, :, :], (n_freq, n_tx, n_rx))
        rz = np.broadcast_to(rz[np.newaxis, :, :], (n_freq, n_tx, n_rx))

    with open(path, 'w') as f:
        f.write(f"{n_freq}\n")
        for ifreq in range(n_freq):
            f.write(f"  {frequencies[ifreq]:.6f}\n")
            f.write(f"  {n_tx}\n")
            for itx in range(n_tx):
                f.write(f"  {sx[ifreq, itx]:.4f}  {sz[ifreq, itx]:.4f}  {n_rx}\n")
                for irx in range(n_rx):
                    parts = [f"  {rx[ifreq, itx, irx]:.4f}",
                             f"  {rz[ifreq, itx, irx]:.4f}"]
                    for ic in range(6):
                        if use_arr[ic]:
                            v = synthetic[ifreq, itx, irx, ic]
                            parts.append(f"  {v.real:20.14E}")
                            parts.append(f"  {v.imag:20.14E}")
                    f.write("".join(parts) + "\n")


def load_synthetic_npz(path: str | Path) -> dict:
    """저장된 순방향 결과 NPZ 읽기"""
    return dict(np.load(Path(path).with_suffix(".npz"), allow_pickle=False))


def _tokenize(lines: list[str]) -> list[str]:
    """주석 제거 + 토큰화"""
    tokens = []
    for line in lines:
        # ! 또는 # 이후는 주석
        for ch in ("!", "#"):
            idx = line.find(ch)
            if idx >= 0:
                line = line[:idx]
        tokens.extend(line.split())
    return tokens
