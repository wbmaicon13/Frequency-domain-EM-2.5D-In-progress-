# EM 2.5D Python 패키지 구조 가이드(2026.03.19)

> 지구물리 전자탐사(EM) 2.5D 수치모델링/역산 코드의 Python 패키지 구조

---

## 전체 구조 한눈에 보기

```
em25d_python/
│
├── em25d/                      ← 메인 패키지 (핵심 알고리즘)
│   ├── constants.py            ← 물리 상수 & 열거형
│   ├── mesh/                   ← 격자 (지하 모델의 뼈대)
│   ├── model/                  ← 비저항 모델 (지하 매질)
│   ├── survey/                 ← 탐사 배열 (송·수신기)
│   ├── forward/                ← 순방향 모델링 (핵심 계산)
│   ├── inverse/                ← 역산 (관측→모델 추정)
│   ├── parallel/               ← 병렬화 (MPI, GPU)
│   └── io/                     ← 파일 입출력
│
├── scripts/                    ← 실행 스크립트
├── gui/                        ← GUI (모델 편집기)
└── tests/                      ← 테스트 코드
```

---

## 1. constants.py — 물리 상수와 타입 정의

```python
MU_0     = 4π × 10⁻⁷  # 진공 투자율 [H/m]
EPSILON_0 = 8.854e-12   # 진공 유전율 [F/m]
PI       = 3.14159...

# 소스 타입: 어떤 종류의 송신기를 쓸 것인가?
SourceType.Jx  # 수평 전기 쌍극자 (x방향)
SourceType.Jy  # 수평 전기 쌍극자 (y방향)
SourceType.Mz  # 수직 자기 쌍극자 (VMD) ← 이번 검증에서 사용
```

---

## 2. mesh/ — 격자 (지하 공간을 나누는 그물)

```
┌──────────────────────────────┐  ← 공기 (z < 0)
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
├──────────────────────────────┤  ← 지표 (z = 0)
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■│  ← 지하 (z > 0)
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■│     각 ■ = 하나의 "요소"
│ ■ ■ ■ ■ ★ ★ ★ ■ ■ ■ ■ ■ ■ │        ★ = 이상대 (다른 비저항)
│ ■ ■ ■ ■ ★ ★ ★ ■ ■ ■ ■ ■ ■ │
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■│
└──────────────────────────────┘  ← 경계 (충분히 깊은 곳)
```

| 파일 | 역할 | 설명 |
|------|------|----------|
| **grid.py** | 격자 생성 | 바둑판의 가로·세로 칸 만들기. 노드(꼭짓점)와 요소(칸) 정의 |
| **boundary.py** | 경계 조건 | 격자 가장자리에서 파동이 반사 없이 빠져나가도록 처리 (Robin BC) |
| **profile.py** | 수신기 위치 | 지표면에 수신기를 놓을 위치 결정 |
| **block.py** | 역산 블록 | 여러 개의 작은 요소를 묶어서 역산할 때 쓰는 큰 블록 |
| **topography.py** | 지형 처리 | 산이나 골짜기 같은 지표 기복 반영 |

**핵심 개념 — 노드 인덱싱:**
```
노드 번호 = iz × n_nodes_x + ix
(iz: 깊이 인덱스, ix: 수평 인덱스)

예) 75×61 격자 → 4,575개 노드, 4,440개 요소
```

---

## 3. model/ — 비저항 모델 (지하에 뭐가 있는가)

| 파일 | 역할 | 설명 |
|------|------|----------|
| **resistivity.py** | 비저항 모델 | 각 격자 칸의 비저항 값 (Ω·m). 0 = 공기, 10 = 보통 땅, 1 = 광체 |
| **anomaly.py** | 이상대 생성 | 원형/사각형/다각형 이상대를 모델에 삽입 |
| **generator.py** | 모델 대량 생성 | 딥러닝 학습용 랜덤 모델 최대 200개 생성 |
| **visualize.py** | 시각화 | 비저항 모델을 컬러맵으로 그리기 |

---

## 4. survey/ — 탐사 배열 (어디서 측정하는가)


| 파일 | 역할 | 설명 |
|------|------|----------|
| **source.py** | 송신기 | 전자기파를 보내는 장치 (위치, 종류, 세기) |
| **receiver.py** | 수신기 | 전자기장을 측정하는 장치 (위치) |
| **frequency.py** | 주파수 | 측정할 주파수 목록 (220~28160 Hz) |
| **survey.py** | 통합 클래스 | 송신기 + 수신기 + 주파수를 묶어서 관리 |

---

## 5. forward/ — 순방향 모델링 (핵심 계산 엔진)

### 전체 계산 흐름

```
           ① 1차장 계산
송신기 ──→ (균질 매질에서의 해석해)
           ↓
           ② FEM 행렬 조립        ← 지하 구조 반영
           K·x = f 형태의 연립방정식 생성
           ↓
           ③ 연립방정식 풀이       ← 가장 오래 걸리는 단계
           ILU+GMRES 또는 LU 분해
           ↓
           ④ 후처리               ← Ey, Hy → Ex, Ez, Hx, Hz 유도
           Maxwell 방정식 관계식
           ↓
           ⑤ 역 Fourier 변환      ← ky 영역 → 실공간
           스플라인 보간 + 수치 적분
           ↓
           ⑥ 전체장 = 2차장 + 1차장
```

### 각 파일의 역할

| 파일 | 단계 | 설명 |
|------|------|----------|
| **primary_field.py** | ① | 균질 반무한 매질에서 Whole space solution 기반 전자기장 계산. "이상대가 없을 때의 기본 신호" |
| **fem_assembly.py** | ② | 지하 구조를 반영한 행렬(K) 조립. 각 격자 칸의 비저항으로 가중된 적분 계산. **벡터화**로 모든 요소를 한번에 처리 |
| **fem_solver.py** | ③ | K·x = f 풀기. ILU 전처리 + GMRES 반복법 사용. **가장 시간이 오래 걸리는 단계** (전체의 62%) |
| **postprocess.py** | ④ | FEM으로 구한 Ey, Hy로부터 나머지 4성분 유도. Maxwell 방정식의 ky 영역 관계식 활용 |
| **forward_loop.py** | 전체 | 위 ①~⑥을 연결하는 메인 루프. 주파수별 × ky별 × 송신기별 반복 |

### 2.5D란?

```
3D 문제(3차원 전자기 송신신호)를 y 방향으로 Fourier 변환하여 여러 개의 2D 문제로 분해:

  3D: (x, y, z) → 매우 큰 행렬
  2.5D: (x, z) × 20개 ky → 작은 행렬 20번 풀기 (병렬화 가능)

각 ky에서 2D FEM을 풀고, 결과를 역 Fourier 변환으로 합침
→ 3D 결과를 2D 계산만으로 얻는 효율적 방법
```

## 6. inverse/ — 역산 (측정 데이터 → 지하 구조 추정)

### 역산이란?

순방향 모델링이 "지하 구조가 주어졌을 때 측정값을 계산"하는 것이라면,
역산은 반대로 **"측정값으로부터 지하 구조를 추정"**하는 과정입니다.

```
순방향 (Forward):    지하 모델 ρ  ──→  계산 데이터 d_pred
역산 (Inverse):     관측 데이터 d_obs ──→  추정 모델 ρ*
```

역산은 본질적으로 **비선형 최적화 문제**입니다.
관측값과 계산값의 차이(잔차)를 최소화하되,
해가 물리적으로 의미 있도록 정규화(Regularization)를 적용합니다.

### 역산 전체 흐름

```
┌────────────────────────────────────────────────────────────────────┐
│  초기 모델 ρ₀ (균질 배경 또는 이전 결과)                             │
└──────────────────────┬─────────────────────────────────────────────┘
                       ▼
       ┌──────────── 반복 루프 (iter = 1, 2, ..., N) ────────────┐
       │                                                          │
       │  ① 순방향 모델링                                          │
       │     ρ_iter → ForwardModeling → d_pred                    │
       │                                                          │
       │  ② 잔차 계산                                              │
       │     r = (d_obs − d_pred) / norm_factor                   │
       │     RMS = √(mean(r²))                                    │
       │                                                          │
       │  ③ 자코비안(감도 행렬) 계산                                │
       │     J[i,b] = ∂d_i / ∂log(ρ_b)  ← 상반정리 활용           │
       │                                                          │
       │  ④ IRLS 가중치 계산                                       │
       │     W_d = f(r)       ← 데이터 가중치 (잡음 강건)           │
       │     W_m = f(R·m)     ← 모델 가중치 (구조 강건)             │
       │                                                          │
       │  ⑤ ACB 라그랑주 승수 계산                                  │
       │     H = J^T W_d J    ← 헤시안 (근사)                      │
       │     λ_i ∝ spread_i   ← 해상도 낮은 블록은 더 강하게 제약    │
       │                                                          │
       │  ⑥ 정규화 정규방정식 풀이                                   │
       │     (H + R̃^T R̃) Δm = J^T W_d r                         │
       │                                                          │
       │  ⑦ 모델 갱신                                              │
       │     ρ_new = clip(ρ + Δm, [ρ_min, ρ_max])                 │
       │                                                          │
       │  ⑧ 수렴 검사                                              │
       │     RMS ≤ 목표? 또는 ΔRMS < 최소 변화?                     │
       │     → 예: 종료                                            │
       │     → 아니오: iter += 1, ①로 복귀                          │
       └──────────────────────────────────────────────────────────┘
                       ▼
       ┌─────────────────────────────────────────────────────────┐
       │  결과: 최종 비저항 모델 ρ* + 수렴 이력 + 로그 파일        │
       └─────────────────────────────────────────────────────────┘
```

### 각 파일의 역할

| 파일 | 단계 | Fortran 대응 | 역할 |
|------|------|-------------|------|
| **measures.py** | ④ | Fem25D_Measures.f90 | 목적함수(Norm) 및 IRLS 가중치 |
| **jacobian.py** | ③ | Fem25DjacReci.f90 | 상반정리 기반 자코비안 행렬 |
| **regularization.py** | ⑤⑥ | Fem25Dinv.f90 | Occam 평활화 행렬 (정규화) |
| **acb.py** | ④⑤⑥⑦ | Fem25Dacb.f90 | ACB 역산 스텝 (정규방정식 풀이) |
| **sequence.py** | ⑥ | Fem25DSequence.f90 | 다중 주파수 시퀀스 제약 |
| **inversion_loop.py** | 전체 | Fem25Dinv.f90 | 반복 루프 제어, 로깅, 수렴 판정 |

---

### 6.1 measures.py — 목적함수와 IRLS 가중치

역산의 목적함수(misfit)를 정의하고, 잡음에 강건한 **IRLS(Iteratively Reweighted Least Squares)** 가중치를 계산합니다.

```
목적함수 선택: "잔차를 어떤 기준으로 측정할 것인가?"

L2 (최소제곱):    φ = Σ rᵢ²          ← 가장 기본, 잡음에 민감
L1 (Ekblom):     φ = Σ √(rᵢ²+ε²)   ← 이상치(outlier)에 강건
Huber:           φ = { rᵢ²/2ε      (|rᵢ| ≤ ε)     ← L2/L1 절충
                     { |rᵢ| - ε/2   (|rᵢ| > ε)
Minimum Support: φ = Σ rᵢ²/(rᵢ²+ε²) ← 희소(sparse) 해 유도
```

IRLS는 비선형 Norm을 **가중 최소제곱**으로 변환하는 기법입니다:
```
L1 가중치:    wᵢ = 1 / √(rᵢ² + ε²)     ← 큰 잔차에 낮은 가중치
Huber 가중치: wᵢ = 1/(2ε) 또는 1/(2|rᵢ|) ← 임계값 기준 전환
```

데이터 잔차와 모델 구조 양쪽 모두에 IRLS를 적용할 수 있습니다.

---

### 6.2 jacobian.py — 감도 행렬 (자코비안)

**"블록 b의 비저항이 변하면 데이터 i가 얼마나 변하는가?"**

```
J[i, b] = ∂dᵢ / ∂log(ρ_b) = -σ_b ∫∫_Ωb E_fwd(r) · E_adj(r) dA
```

- `E_fwd`: 송신기에 의한 전기장 (순방향 풀이 결과)
- `E_adj`: 수신기를 가상 소스로 놓았을 때의 전기장 (**상반정리**)

상반정리를 쓰면 **TX×RX 조합마다 별도 풀이 없이** 감도를 구할 수 있어,
수치 유한차분(perturbation)보다 훨씬 효율적입니다.

```
상반정리 (Reciprocity):
  송신기→수신기 경로의 감도 = 수신기→송신기 경로의 감도
  → 수신기에 가상 소스를 놓고 한 번 더 순방향 풀이
  → 모든 TX-RX 조합의 감도를 동시에 계산
```

**계산 절차:**
```
1. ky 영역에서 각 요소에 대해 E_fwd · E_adj 면적분 (bilinear 구적법)
2. 모든 ky에서의 적분 결과를 역 Fourier 변환 (사다리꼴 공식)
3. 비저항 파라미터 변환 적용 (log 스케일링)
```

**비저항 파라미터 변환 (Jacobian scaling):**
```
fac_b = σ_b · log(σ_max / σ_b) · log(σ_b / σ_min) / log(σ_max / σ_min)
```
이 변환은 비저항의 상·하한 제약(ρ_min, ρ_max)을 자동으로 만족시킵니다.

---

### 6.3 regularization.py — 정규화 (Occam 평활화)

역산 문제는 **비유일성(non-uniqueness)**이 심합니다 —
같은 관측 데이터를 설명하는 지하 모델이 무수히 많습니다.
정규화는 이 중에서 **가장 매끄러운(smooth) 해**를 선택하도록 강제합니다.

**Roughening(거칠기) 행렬 R:**
```
                 인접 블록과의 차이를 벌점으로 부과

  블록 배치 (n_x × n_z):           Roughening 행렬 R:
  ┌───┬───┬───┐                    R[k, k]     = 1 (자기 자신)
  │ 0 │ 1 │ 2 │ ← z행 0           R[k, 위/아래] = -w · Sm_V
  ├───┼───┼───┤                    R[k, 좌/우]   = -w · Sm_H
  │ 3 │ 4 │ 5 │ ← z행 1           w = 1 / 이웃 수 (꼭짓점 2, 변 3, 내부 4)
  ├───┼───┼───┤
  │ 6 │ 7 │ 8 │ ← z행 2           Sm_V, Sm_H: 수직/수평 평활 강도
  └───┴───┴───┘
   x열0 x열1 x열2

  블록 인덱스: k = j·n_z + i (j=x열, i=z행)
```

**모델 구조 벡터:**
```
Rm = R · log(ρ)    ← "인접 블록 간 비저항 차이"의 가중합
```

Rm이 작을수록 모델이 매끄럽습니다. 목적함수에 `||W_m · Rm||²` 항을 추가하여
데이터 적합과 모델 부드러움 사이의 균형을 맞춥니다.

---

### 6.4 acb.py — ACB 역산 스텝

**ACB(Active Constraint Balancing)**는 블록마다 개별적인 정규화 강도를 부여합니다.

```
기존 Occam:                      ACB:
  모든 블록에 동일한 λ 적용         각 블록 i에 λ_i 개별 적용
  → 해상도 높은 곳도 과도한 평활    → 해상도 기반 적응형 제약

해상도가 높은 블록 (데이터 감도 큼):   λ_i 작음 → 자유롭게 변화 허용
해상도가 낮은 블록 (데이터 감도 작음): λ_i 큼  → 강하게 평활화
```

**ACB 라그랑주 승수 계산 절차:**
```
1. 헤시안:     H = J^T · W_d · J
2. 감쇠 역행렬: H_inv = (H + δ·I)^(-1)
3. 해상도 행렬: R_res = H_inv · H          ← 대각이 1에 가까우면 해상도 높음
4. 확산 함수:   spr_i = Σ_j d²(i,j) · R_res[i,j]²  +  (1 - R_res[i,i])²
               (d(i,j) = 격자 상 블록 간 거리)
5. 라그랑주:   λ_i = 10^(slope · (log(spr_i) - log(spr_min)) + log(λ_min))
```

**정규방정식:**
```
(J^T W_d J  +  R̃^T R̃) · Δm  =  J^T W_d r

여기서:
  R̃[i, :] = √(λ_i · w_m_i) · R[i, :]    ← ACB + IRLS로 행별 스케일링
  Δm = 모델 갱신 벡터
  r = 정규화된 잔차
```

**비저항 갱신:**
```
log-변환 파라미터: m_i = log((σ_i - σ_min) / (σ_max - σ_i))
  → ρ_min ≤ ρ ≤ ρ_max 자동 보장 (별도 클리핑 불필요)

갱신 모드:
  Jumping:  ρ_new = 변환(m + Δm)          ← 공격적, 빠른 수렴
  Creeping: ρ_new = 0.5·ρ_old + 0.5·변환   ← 안정적, 느린 수렴
```

---

### 6.5 sequence.py — 다중 주파수 시퀀스 제약

인접 주파수 간 데이터의 **연속성**을 강제하는 추가 제약입니다.

```
주파수 f₁, f₂, f₃ 에서의 데이터:

  d(f₁) ≈ d(f₂) ≈ d(f₃)  ← 급격한 변화 억제

시퀀스 잔차:
  r_seq = S_pred · d_pred - S_obs · d_obs

  S는 차분 연산자: 인접 주파수 간 차이를 계산
  각 수신기별로 (n_freq - 1)개의 제약 생성
```

이 제약은 정규방정식에 추가 항으로 들어갑니다:
```
H_total = H + H_seq     (H_seq = J^T S_w^T S_w J)
g_total = g + g_seq     (g_seq = J^T S_w^T (w · r_seq))
```

---

### 6.6 inversion_loop.py — 역산 반복 루프

전체 역산 프로세스를 제어하는 **오케스트레이터**입니다.

**InversionConfig — 주요 설정:**
```python
InversionConfig(
    max_iterations = 10,        # 최대 반복 횟수
    iteration_type = "jumping",  # "jumping" (공격적) / "creeping" (안정적)

    # 사용할 데이터 성분 (0=미사용, 1=허수부만, 2=실수+허수)
    use_Ey = 0, use_Hx = 0, use_Hz = 1,  # Hz만 사용하는 예

    # IRLS Norm
    norm_data  = NormType.L2,   # 데이터 잔차 Norm
    norm_model = NormType.L2,   # 모델 구조 Norm
    irls_start = 1,             # IRLS 적용 시작 반복

    # 비저항 범위 제약
    rho_min = 0.1,  rho_max = 1e5,  # [Ω·m]

    # 정규화
    use_acb = True,             # ACB 사용 여부
    smoothness_v = 0.5,         # 수직 평활 강도
    smoothness_h = 0.5,         # 수평 평활 강도

    # 수렴 조건
    target_rms = 1.0,           # 목표 RMS (1.0 = 잡음 수준)
    min_delta_rms = 1e-4,       # 최소 RMS 변화 (정체 판정)
)
```

**로깅:**
```
inversion_log/
└── run_260319_143022/
    ├── misfit_log.csv         ← (반복, rms_data, rms_model, step_size)
    ├── model_iter_001.dat     ← 각 반복의 비저항 모델
    ├── model_iter_002.dat
    ├── ...
    └── model_final.dat        ← 최종 결과
```

---

### 6.7 모듈 간 데이터 흐름

```
inversion_loop.py (오케스트레이터)
│
├── forward/ → 순방향 모델링 → d_pred
│
├── select_data_components()
│   → (d_pred, d_obs, norm_factor) 추출
│
├── compute_residual() → 정규화된 잔차 r
│
├── jacobian.py
│   ├── element_surface_integral()  ← ky 영역 면적분
│   ├── compute_field_components()  ← Maxwell 관계식
│   ├── jacobian_inverse_fourier()  ← ky → 공간 영역
│   └── apply_resistivity_transform() ← log(ρ) 스케일링
│   → 자코비안 J (n_data × n_blocks)
│
├── acb.py:inversion_step()
│   ├── measures.py:compute_irls_weights()  ← W_d, W_m
│   ├── regularization.py
│   │   ├── build_roughening_matrix()       ← R
│   │   ├── compute_model_structure()       ← Rm = R·log(ρ)
│   │   └── scale_roughening_matrix()       ← R̃ = √(λ·w)·R
│   ├── compute_acb_lagrangian()            ← λ_ACB
│   ├── solve_normal_equations()            ← Δm
│   └── line_search_step_size()             ← ρ_new
│
├── [선택] sequence.py
│   └── compute_sequence_contribution()     ← (H_seq, g_seq)
│
└── InversionLogger → CSV + 모델 스냅샷
```

---


## 7. parallel/ — 병렬화 (빠르게 계산하기)

```
MPI 프로세스 분배 (n_ky = 20):

  Process 0:  ky[0]  → 8개 주파수 × 22개 TX 계산
  Process 1:  ky[1]  → 8개 주파수 × 22개 TX 계산
  ...
  Process 19: ky[19] → 8개 주파수 × 22개 TX 계산

  → MPI_REDUCE로 결과 합산 → 역 Fourier 변환
```

| 파일 | 역할 | 설명 |
|------|------|----------|
| **mpi_manager.py** | MPI 관리 | 20개 ky를 프로세스별로 나눠 주고, 결과 모으기 |
| **gpu_solver.py** | GPU 풀이 | NVIDIA GPU가 있으면 행렬 풀이를 GPU에서 수행 |

---

## 8. io/ — 파일 입출력

| 파일 | 역할 | 설명 |
|------|------|----------|
| **params.py** | 설정 파일 | YAML 형식의 파라미터 읽기/쓰기 |
| **mesh_io.py** | 격자 I/O | 격자 데이터 NPZ/CSV 저장/로드 |
| **data_io.py** | 데이터 I/O | 관측/계산 데이터 읽기, .inp 파일 생성 (역산 입력용) |
| **legacy_io.py** | 레거시 호환 | 기존 Fortran 포맷 (nodetest.dat, survey.dat 등) 읽기 |

---

## 9. scripts/ — 실행 스크립트

```bash
# 순방향 모델링 실행
python scripts/run_forward.py

# 역산 실행
python scripts/run_inversion.py --config config.yaml --observed data.npz

# Fortran 결과와 비교 검증 (MPI 20 프로세스)
OMP_NUM_THREADS=1 mpirun -np 20 python scripts/verify_legacy.py

# 딥러닝 학습 데이터 생성
python scripts/generate_dataset.py
```

---

## 10. 데이터 흐름 전체 그림

```
┌─────────────────────────────────────────────────────────┐
│                    입력 (Input)                          │
│  survey.dat    → 주파수, 송수신기 위치, 소스 타입        │
│  nodetest.dat  → 격자 노드 좌표 (75×61 = 4,575개)      │
│  elemtest.dat  → 요소 연결성 (4,440개)                  │
│  Model_*.dat   → 비저항 모델 (각 요소의 Ω·m 값)        │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│              순방향 모델링 (Forward)                    │
│                                                         │
│   for freq in [220, 440, ..., 28160]:                   │
│     for ky in [ky_1, ky_2, ..., ky_20]:  ← MPI 분배     │
│       K 행렬 조립 (1회)                                 │
│       Robin BC 적용                                     │
│       ILU 전처리 (1회)                                  │
│       for tx in [TX_1, ..., TX_22]:                     │
│         1차장 계산                                      │
│         Force 벡터 f 조립                               │
│         GMRES 풀이 → Ey_s, Hy_s                         │
│         후처리 → Ex, Ez, Hx, Hz (프로파일 노드만)       │
│     MPI REDUCE (ky 결과 합산)                           │
│     역 Fourier 변환 (ky → 공간)                         │
│     1차장 더하기 (total = secondary + primary)          │
└──────────────────────┬──────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    출력 (Output)                         │
│  verify_result.npz      → 전체 계산 결과 (NPZ)         │
│  Fem25Dinv_verify.inp   → 역산 입력용 (Fortran 호환)    │
│  verification_plot.png  → 비교 그래프                    │
└─────────────────────────────────────────────────────────┘
```
---


## 11. 주요 용어 정리

| 용어 | 의미 |
|------|------|
| **FEM** | 유한요소법 — 편미분방정식을 격자 위에서 수치적으로 푸는 방법 |
| **2.5D** | y 방향 Fourier 변환으로 3D를 2D로 표현(공간 상에서 전자기장 송신원은 3차원이므로) |
| **ky** | y 방향 공간주파수 — 2.5D에서 Fourier 변환 변수 |
| **1차장** | 균질 매질(이상대 없음)에서의 해석적 전자기장 |
| **2차장** | 이상대에 의한 산란장 (= 전체장 - 1차장) |
| **DOF** | Degree of Freedom — 노드당 미지수 (Ey, Hy = 2개) |
| **Robin BC** | 임피던스 경계 조건 — 파동이 격자 밖으로 빠져나가도록 처리 |
| **ILU** | Incomplete LU — 불완전 LU 분해 (GMRES 전처리용) |
| **GMRES** | 반복 선형대수 풀이법 — 큰 희소 행렬을 효율적으로 풀기 |
| **IFT** | Inverse Fourier Transform — ky 영역 → 실공간 변환 |
| **MPI** | Message Passing Interface — 다중 프로세스 병렬화 |
| **자코비안** | 감도 행렬 — 모델 변화 → 데이터 변화 관계 (역산에 필수) |
| **ACB** | Active Constraint Balancing — 역산 알고리즘 |
| **IRLS** | 반복 재가중 최소제곱법 — 잡음에 강건한 역산 |


## Special Thanks
Dr. Ki Ha Lee

Dr. Yoonho Song

Dr. Hyoung-Seok Kwon

Dr. Soon Jee Seol

Prof. Seogi Kang

Dr. Soocheol Jeong

Dr. Kyubo Noh

Dr. Seokmin Oh
