#!/usr/bin/env python
"""
Fortran vs Python EM 2.5D 수치모델링 검증 — MPI 병렬 + GPU 가속

실행:
  python scripts/verify_fortran_match.py                          # 직렬
  mpirun -n 20 python scripts/verify_fortran_match.py             # MPI 20 프로세스
  mpirun -n 20 python scripts/verify_fortran_match.py --gpu       # MPI + GPU
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from em25d.constants import PI, SourceType, MU_0, EPSILON_0
from em25d.forward.primary_field import PrimaryFieldParams, primary_field_ky_domain

# ── MPI ───────────────────────────────────────────────────────────────────────
try:
    from mpi4py import MPI
    COMM = MPI.COMM_WORLD; RANK = COMM.Get_rank(); SIZE = COMM.Get_size()
except ImportError:
    COMM = None; RANK = 0; SIZE = 1

def mpi_print(*a, **k):
    if RANK == 0: print(*a, **k, flush=True)

# ── GPU solver ────────────────────────────────────────────────────────────────
def _get_gpu_solver():
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsp
        import cupyx.scipy.sparse.linalg as cpspla
        def solve(K, f):
            Kg = cpsp.csc_matrix(K); fg = cp.array(f)
            return cp.asnumpy(cpspla.spsolve(Kg, fg))
        return solve, f"CuPy ({cp.cuda.Device().attributes.get('DeviceName',b'GPU').decode(errors='replace')})"
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0)
            def solve(K, f):
                Kd = torch.tensor(K.toarray(), dtype=torch.complex128, device='cuda')
                fd = torch.tensor(f, dtype=torch.complex128, device='cuda')
                return torch.linalg.solve(Kd, fd).cpu().numpy()
            return solve, f"PyTorch ({dev})"
    except Exception:
        pass
    return None, "N/A"

# ── Fortran 파싱 ──────────────────────────────────────────────────────────────
def parse_coord(path):
    lines = path.read_text().splitlines()
    z, x = [], []
    for l in lines[2:63]:
        p = l.split()
        if len(p) >= 2: z.append(float(p[1]))
    for j, l in enumerate(lines):
        if j > 2 and ">> X" in l:
            for l2 in lines[j+2:j+77]:
                p2 = l2.split()
                if len(p2) >= 2: x.append(float(p2[1]))
            break
    return np.array(x), np.array(z)

def parse_model(path, nex, nez):
    d = np.loadtxt(path)
    r = np.where(d[:,1]==0, 1e8, d[:,1])
    return r.reshape(nex, nez, order='F')

def parse_output(path):
    d = np.loadtxt(path)
    nf, nt = int(d[:,0].max()), int(d[:,1].max())
    Hy = np.zeros((nf,nt), dtype=complex)
    Hz = np.zeros((nf,nt), dtype=complex)
    for r in range(len(d)):
        f, t = int(d[r,0])-1, int(d[r,1])-1
        Hy[f,t] = d[r,3]+1j*d[r,4]
        Hz[f,t] = d[r,7]+1j*d[r,8]
    return {"Hy": Hy, "Hz": Hz, "nf": nf, "nt": nt}

# ── 참조 행렬 사전 계산 (핵심 최적화) ─────────────────────────────────────────
# 직교 격자: Jacobian = diag(dx/2, dz/2), detJ = dx*dz/4
# dN/dx = (2/dx)*dN/dξ, dN/dz = (2/dz)*dN/dη
# 적분 = detJ * Σ_g f(ξ_g, η_g)  (weight=1 for 2×2 Gauss)
#
# Ke[iE,jE] = xk_E*(dz/dx*Aξ[i,j] + dx/dz*Aη[i,j]) + xb_E*(dx*dz/4*M[i,j])
# 여기서 Aξ, Aη, M 은 (4,4) 참조 행렬 = 상수 (dx,dz 무관)

_GP = 1.0/np.sqrt(3.0)
_GXI  = [-_GP, _GP, _GP, -_GP]
_GETA = [-_GP, -_GP, _GP, _GP]

def _precompute_ref():
    """4×4 참조 행렬 사전 계산 (프로그램 시작 시 1회)"""
    Axi  = np.zeros((4,4))  # Σ_g dNdξ_i · dNdξ_j
    Aeta = np.zeros((4,4))  # Σ_g dNdη_i · dNdη_j
    M    = np.zeros((4,4))  # Σ_g N_i · N_j
    Cxz  = np.zeros((4,4))  # Σ_g dNdξ_i · dNdη_j
    Czx  = np.zeros((4,4))  # Σ_g dNdη_i · dNdξ_j
    Nvec = np.zeros(4)      # Σ_g N_i
    DNxi = np.zeros(4)      # Σ_g dNdξ_i
    DNeta= np.zeros(4)      # Σ_g dNdη_i

    for xi, eta in zip(_GXI, _GETA):
        N = 0.25*np.array([(1-xi)*(1-eta),(1+xi)*(1-eta),(1+xi)*(1+eta),(1-xi)*(1+eta)])
        dxi = 0.25*np.array([-(1-eta),(1-eta),(1+eta),-(1+eta)])
        det = 0.25*np.array([-(1-xi),-(1+xi),(1+xi),(1-xi)])
        Axi  += np.outer(dxi, dxi)
        Aeta += np.outer(det, det)
        M    += np.outer(N, N)
        Cxz  += np.outer(dxi, det)
        Czx  += np.outer(det, dxi)
        Nvec += N
        DNxi += dxi
        DNeta+= det
    return Axi, Aeta, M, Cxz, Czx, Nvec, DNxi, DNeta

_Axi, _Aeta, _M, _Cxz, _Czx, _Nvec, _DNxi, _DNeta = _precompute_ref()
_SIG_AIR = 1e-8

def assemble_vectorized(x_nodes, z_nodes, elem_res, layer_res, E_primary, omega, ky):
    """
    완전 벡터화 FEM 조립 — Python 요소 루프 최소화

    참조 행렬과 broadcasting 활용, 순수 Python 루프 대비 ~20x 빠름.
    """
    nx, nz = len(x_nodes), len(z_nodes)
    nex, nez = nx-1, nz-1
    ndof = 2*nx*nz

    dx = np.diff(x_nodes)  # (nex,)
    dz = np.diff(z_nodes)  # (nez,)

    # 물성 (nex, nez)
    sig = np.where(elem_res==0, _SIG_AIR, 1.0/elem_res)
    sig_b = np.where(layer_res==0, _SIG_AIR, 1.0/layer_res)
    k2 = MU_0*EPSILON_0*omega**2 - 1j*MU_0*sig*omega
    dd = ky**2 - k2
    ysig = sig + 1j*omega*EPSILON_0

    xkE = 1j*ysig/dd;       xbE = 1j*ysig
    xkH = -omega*MU_0/dd;   xbH = np.full_like(dd, -omega*MU_0)
    dsig = sig - sig_b
    dsig = np.where(np.abs(dsig)<1e-5, 0.0, dsig)
    c1E = 1j*ky/dd*ysig;    c2E = -c1E
    c1H = 1j*ky*omega*MU_0/dd; c2H = -c1H

    # 기하: dx/dz ratio for each element
    dzodx = dz[np.newaxis,:] / dx[:,np.newaxis]  # (nex,nez) WRONG shape
    # Actually: element(ex,ez) has dx[ex], dz[ez]
    # dzodx[ex,ez] = dz[ez]/dx[ex]
    DX = dx[:,np.newaxis] * np.ones(nez)[np.newaxis,:]  # (nex,nez)
    DZ = np.ones(nex)[:,np.newaxis] * dz[np.newaxis,:]  # (nex,nez)
    DZoDX = DZ/DX  # (nex,nez)
    DXoDZ = DX/DZ
    DXxDZ4 = DX*DZ/4.0  # detJ

    # COO 축적: 각 요소별 8×8 = 64 entries, 총 nex*nez*64 entries
    n_elem = nex*nez
    total_entries = n_elem * 64

    all_rows = np.zeros(total_entries, dtype=np.int64)
    all_cols = np.zeros(total_entries, dtype=np.int64)
    all_vals = np.zeros(total_entries, dtype=complex)
    f_global = np.zeros(ndof, dtype=complex)

    # 전역 DOF 인덱스 매핑 사전 계산
    # node_index(ix,iz) = iz*nx + ix
    EX, EZ = np.meshgrid(np.arange(nex), np.arange(nez), indexing='ij')  # (nex,nez)
    N0 = EZ*nx + EX          # 좌하 (ex, ez)
    N1 = EZ*nx + (EX+1)      # 우하
    N2 = (EZ+1)*nx + (EX+1)  # 우상
    N3 = (EZ+1)*nx + EX      # 좌상

    idx = 0
    for i in range(4):
        for j in range(4):
            # 각 (i,j) 쌍에 대해 모든 요소 한꺼번에 계산
            nids_i = [N0, N1, N2, N3][i]  # (nex, nez)
            nids_j = [N0, N1, N2, N3][j]

            # E-E block: Ke[2i, 2j]
            val_EE = (xkE * (DZoDX*_Axi[i,j] + DXoDZ*_Aeta[i,j])
                     + xbE * DXxDZ4*_M[i,j])
            # H-H block: Ke[2i+1, 2j+1]
            val_HH = (xkH * (DZoDX*_Axi[i,j] + DXoDZ*_Aeta[i,j])
                     + xbH * DXxDZ4*_M[i,j])
            # E-H coupling: Ke[2i, 2j+1]
            val_EH = c1E*_Cxz[i,j] + c2E*_Czx[i,j]
            # H-E coupling: Ke[2i+1, 2j]
            val_HE = c1H*_Czx[i,j] + c2H*_Cxz[i,j]

            flat = np.arange(n_elem)

            # E-E
            all_rows[idx*n_elem:(idx+1)*n_elem] = 2*nids_i.ravel()
            all_cols[idx*n_elem:(idx+1)*n_elem] = 2*nids_j.ravel()
            all_vals[idx*n_elem:(idx+1)*n_elem] = val_EE.ravel()
            idx += 1

            # H-H
            all_rows[idx*n_elem:(idx+1)*n_elem] = 2*nids_i.ravel()+1
            all_cols[idx*n_elem:(idx+1)*n_elem] = 2*nids_j.ravel()+1
            all_vals[idx*n_elem:(idx+1)*n_elem] = val_HH.ravel()
            idx += 1

            # E-H
            all_rows[idx*n_elem:(idx+1)*n_elem] = 2*nids_i.ravel()
            all_cols[idx*n_elem:(idx+1)*n_elem] = 2*nids_j.ravel()+1
            all_vals[idx*n_elem:(idx+1)*n_elem] = val_EH.ravel()
            idx += 1

            # H-E
            all_rows[idx*n_elem:(idx+1)*n_elem] = 2*nids_i.ravel()+1
            all_cols[idx*n_elem:(idx+1)*n_elem] = 2*nids_j.ravel()
            all_vals[idx*n_elem:(idx+1)*n_elem] = val_HE.ravel()
            idx += 1

    K = sp.csr_matrix((all_vals, (all_rows, all_cols)), shape=(ndof, ndof))

    # 힘벡터 (2차장 소스): dsig != 0 인 요소만
    mask_ds = np.abs(dsig) > 1e-5
    if mask_ds.any():
        for i in range(4):
            nid = [N0, N1, N2, N3][i]  # (nex, nez)
            # 요소 4노드의 1차장 평균
            Exp = 0.25*(E_primary[0, N0.ravel()] + E_primary[0, N1.ravel()]
                       + E_primary[0, N2.ravel()] + E_primary[0, N3.ravel()])
            Eyp = 0.25*(E_primary[1, N0.ravel()] + E_primary[1, N1.ravel()]
                       + E_primary[1, N2.ravel()] + E_primary[1, N3.ravel()])
            Ezp = 0.25*(E_primary[2, N0.ravel()] + E_primary[2, N1.ravel()]
                       + E_primary[2, N2.ravel()] + E_primary[2, N3.ravel()])
            Exp = Exp.reshape(nex, nez)
            Eyp = Eyp.reshape(nex, nez)
            Ezp = Ezp.reshape(nex, nez)

            # fe[iE] : E 소스
            feE = (-1j*dsig*Eyp*DXxDZ4*_Nvec[i]
                  + ky*dsig/dd*(Exp*DZ/2*_DNxi[i] + Ezp*DX/2*_DNeta[i]))
            # fe[iH] : H 소스
            feH = dsig/dd*omega*MU_0*(Exp*DZ/2*_DNeta[i] - Ezp*DX/2*_DNxi[i])

            np.add.at(f_global, 2*nid.ravel(), feE.ravel())
            np.add.at(f_global, 2*nid.ravel()+1, feH.ravel())

    return K, f_global


def solve_system(K, f, nx, nz, use_gpu=False, gpu_fn=None):
    ndof = K.shape[0]
    bnodes = set()
    for ix in range(nx):
        bnodes.add(ix); bnodes.add((nz-1)*nx+ix)
    for iz in range(nz):
        bnodes.add(iz*nx); bnodes.add(iz*nx+nx-1)
    bn = np.array(sorted(bnodes))
    bdofs = np.concatenate([2*bn, 2*bn+1])
    interior = np.setdiff1d(np.arange(ndof), bdofs)
    Ki = K[interior][:, interior]; fi = f[interior]

    if use_gpu and gpu_fn is not None:
        xi = gpu_fn(Ki.tocsc(), fi)
    else:
        xi = spla.spsolve(Ki.tocsc(), fi)

    x = np.zeros(ndof, dtype=complex)
    x[interior] = xi
    return x


def postprocess_rx(sol, Ep, x_n, z_n, omega, ky, eres, ix_rx, iz_rx):
    """수신기 위치에서만 장 성분 추출 (전체 후처리 대신)"""
    nx, nz = len(x_n), len(z_n)
    Eys = sol[0::2]; Hys = sol[1::2]

    def to2d(a): return a.reshape(nz, nx)

    Ey_tot = Eys + Ep[1]
    Exp2 = to2d(Ep[0]); Ezp2 = to2d(Ep[2])
    dExdz = np.gradient(Exp2, z_n, axis=0)
    dEzdx = np.gradient(Ezp2, x_n, axis=1)
    Hyp2 = (dEzdx - dExdz)/(1j*omega*MU_0)
    Hy_tot = Hys + Hyp2.ravel()

    Ey2 = to2d(Ey_tot); Hy2 = to2d(Hy_tot)
    dEydx = np.gradient(Ey2, x_n, axis=1)
    dEydz = np.gradient(Ey2, z_n, axis=0)
    dHydx = np.gradient(Hy2, x_n, axis=1)

    # 수신기 위치
    Hy_rx = Hy2[iz_rx, ix_rx]
    Hz_rx = -dEydx[iz_rx, ix_rx]/(1j*omega*MU_0)

    return Hy_rx, Hz_rx


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--n-ky", type=int, default=20)
    parser.add_argument("--max-freq", type=int, default=2)
    args = parser.parse_args()

    mpi_print("="*70)
    mpi_print(f"  EM 2.5D 검증 [MPI={SIZE} procs, GPU={args.gpu}]")
    mpi_print("="*70)

    gpu_fn, gpu_name = (None, "N/A")
    if args.gpu:
        gpu_fn, gpu_name = _get_gpu_solver()
        mpi_print(f"  GPU: {gpu_name}")

    # 데이터
    xn, zn = parse_coord(DATA_ROOT/"model_setup"/"topo_001"/"coord.msh")
    nx, nz = len(xn), len(zn)
    nex, nez = nx-1, nz-1
    mpi_print(f"  격자: {nx}×{nz} nodes")

    eres = parse_model(DATA_ROOT/"model_res"/"Model_00001.dat", nex, nez)
    lres = np.full_like(eres, 100.0)
    for ez in range(nez):
        if 0.5*(zn[ez]+zn[ez+1]) < 0:
            lres[:, ez] = 1e8

    fort = parse_output(DATA_ROOT/"output_data"/"topo_001"/"Data_001_00001.dat")
    freqs_all = [220,440,880,1760,3520,7040,14080,28160]
    nf = min(args.max_freq, len(freqs_all))
    freqs = freqs_all[:nf]
    src_x = np.arange(12.0, 97.0, 4.0); ntx = len(src_x)
    src_z = -1.0

    ix_rx = np.array([np.argmin(np.abs(xn-sx)) for sx in src_x])
    iz_rx = np.argmin(np.abs(zn-src_z))

    # ky
    nky = args.n_ky
    skind = 500*np.sqrt(100/freqs[0])
    akmax = PI/np.min(np.diff(xn)); akmin = 2*PI/(30*skind)
    dl = (np.log10(akmax)-np.log10(akmin))/(nky-2)
    ky_arr = np.zeros(nky); ky_arr[0] = 1e-8
    for i in range(1, nky): ky_arr[i] = 10**(np.log10(akmin)+(i-1)*dl)

    mpi_print(f"  ky: n={nky}, freqs={freqs}")
    params = PrimaryFieldParams(background_resistivity=100.0)

    # MPI 분배
    my_ky = list(range(RANK, nky, SIZE))
    mpi_print(f"  프로세스당 ky: ~{len(my_ky)}개")

    # 결과
    Hy_ky = np.zeros((nf, nky, ntx), dtype=complex)
    Hz_ky = np.zeros((nf, nky, ntx), dtype=complex)

    t0 = time.time()
    for ifreq, freq in enumerate(freqs):
        omega = 2*PI*freq
        mpi_print(f"\n  freq {ifreq+1}/{nf}: {freq} Hz")

        for cnt, iky in enumerate(my_ky):
            ky = ky_arr[iky]
            if RANK == 0:
                print(f"    ky {cnt+1}/{len(my_ky)} ({iky+1}/{nky})", flush=True)

            for itx in range(ntx):
                Ep = primary_field_ky_domain(
                    source_x=src_x[itx], source_z=src_z,
                    node_x=xn, node_z=zn,
                    wavenumber_ky=ky, omega=omega,
                    source_type=SourceType.Jy, params=params)
                Epn = Ep.transpose(0,2,1).reshape(3, nz*nx)

                K, f = assemble_vectorized(xn, zn, eres, lres, Epn, omega, ky)
                sol = solve_system(K, f, nx, nz, args.gpu, gpu_fn)
                hy, hz = postprocess_rx(sol, Epn, xn, zn, omega, ky, eres,
                                        ix_rx[itx], iz_rx)
                Hy_ky[ifreq, iky, itx] = hy
                Hz_ky[ifreq, iky, itx] = hz

    # MPI reduce
    if COMM is not None and SIZE > 1:
        Hy_g = np.zeros_like(Hy_ky); Hz_g = np.zeros_like(Hz_ky)
        COMM.Allreduce(Hy_ky, Hy_g, op=MPI.SUM)
        COMM.Allreduce(Hz_ky, Hz_g, op=MPI.SUM)
    else:
        Hy_g, Hz_g = Hy_ky, Hz_ky

    elapsed = time.time() - t0

    if RANK != 0:
        return

    # 역 Fourier
    Hy_py = np.zeros((nf, ntx), dtype=complex)
    Hz_py = np.zeros((nf, ntx), dtype=complex)
    for i in range(nky-1):
        dk = ky_arr[i+1]-ky_arr[i]
        Hy_py += 0.5*(Hy_g[:,i,:]+Hy_g[:,i+1,:])*dk
        Hz_py += 0.5*(Hz_g[:,i,:]+Hz_g[:,i+1,:])*dk
    Hy_py *= 2/PI; Hz_py *= 2/PI

    # 비교
    print(f"\n  완료: {elapsed:.1f}초 ({SIZE} procs)")
    print("\n"+"="*70)
    print("  Fortran vs Python 비교")
    print("="*70)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = ROOT/"tests"/"comparison_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cn, py_c, fk in [("Hy", Hy_py, "Hy"), ("Hz", Hz_py, "Hz")]:
        fc = fort[fk][:nf,:]
        print(f"\n  ── {cn} ──")
        print(f"  {'freq':>8} {'Fort':>14} {'Python':>14} {'err%':>8}")
        for fi, freq in enumerate(freqs):
            fa, pa = np.abs(fc[fi]), np.abs(py_c[fi])
            m = fa > 1e-30
            e = np.mean(np.abs(fa[m]-pa[m])/(fa[m]+1e-300))*100 if m.any() else 0
            print(f"  {freq:8.0f} {fa.mean():14.4e} {pa.mean():14.4e} {e:8.1f}")

    # 플롯
    fig, axes = plt.subplots(2, nf, figsize=(6*nf, 8), squeeze=False)
    for i in range(nf):
        for ri, (cn, py, fk) in enumerate([("Hy", Hy_py, "Hy"), ("Hz", Hz_py, "Hz")]):
            ax = axes[ri, i]
            ax.semilogy(src_x, np.abs(fort[fk][i,:]), "b-o", ms=3, label="Fortran")
            ax.semilogy(src_x, np.abs(py[i,:]), "r--s", ms=3, label="Python")
            ax.set_title(f"|{cn}| — {freqs[i]} Hz")
            ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir/"fortran_vs_python.png", dpi=150)
    plt.close(fig)
    print(f"\n  플롯: {out_dir/'fortran_vs_python.png'}")

    np.savez_compressed(out_dir/"verification.npz",
                        Hy_py=Hy_py, Hz_py=Hz_py, freqs=freqs, src_x=src_x)
    print(f"  데이터: {out_dir/'verification.npz'}")
    print(f"\n  소요: {elapsed:.1f}초, procs={SIZE}, gpu={gpu_name}")

if __name__ == "__main__":
    main()
