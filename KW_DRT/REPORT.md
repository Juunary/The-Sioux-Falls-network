# KW_DRT 프로젝트 분석 보고서

> 작성일: 2026-03-12

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [기술 스택](#2-기술-스택)
3. [시스템 아키텍처](#3-시스템-아키텍처)
4. [강화학습 설계](#4-강화학습-설계)
5. [시뮬레이션 환경](#5-시뮬레이션-환경)
6. [학습 파이프라인](#6-학습-파이프라인)
7. [결과 출력 구조](#7-결과-출력-구조)
8. [코드 품질 및 개선 사항](#8-코드-품질-및-개선-사항)

---

## 1. 프로젝트 개요

**KW_DRT**는 수요 응답형 교통(Demand-Responsive Transport, DRT) 시스템에서 차량 배차를 최적화하기 위한 강화학습 기반 연구 프로젝트입니다.

### 핵심 목표

- 실시간으로 발생하는 승객 요청을 여러 차량에 효율적으로 배정
- 승객 대기 시간과 차내 이동 시간 최소화
- 차량 이용률 극대화

### 접근 방식

| 항목 | 내용 |
|------|------|
| 알고리즘 | Double Deep Q-Network (DDQN) |
| 시뮬레이션 | 시간 이산형 (Time-discrete) 환경 |
| 네트워크 | 25개 노드, 실제 이동 시간 기반 |
| 학습 방식 | 경험 리플레이 + 타겟 네트워크 |

---

## 2. 기술 스택

### 언어 및 프레임워크

| 기술 | 버전 | 용도 |
|------|------|------|
| Python | 3.8 | 주 개발 언어 |
| TensorFlow | 2.10.0 | 신경망 학습 |
| Keras | (TF 내장) | 모델 구성 |
| NumPy | 1.24.4 | 수치 연산 |
| Pandas | 1.5.3 | 데이터 로딩 |

### GPU 가속 환경

| 기술 | 버전 |
|------|------|
| CUDA Toolkit | 11.2.2 |
| cuDNN | 8.1.0.77 |

### 개발 도구

- Conda 가상환경 (`environment.yml`)
- Jupyter / JupyterLab (탐색적 분석)
- IPython (인터랙티브 실행)

---

## 3. 시스템 아키텍처

### 디렉토리 구조

```
KW_DRT/
├── app/                   # 핵심 모듈
│   ├── action_type.py     # 행동 열거형 (REJECT, PICKUP, DROPOFF)
│   ├── agent.py           # DDQN 에이전트
│   ├── config.py          # 전역 설정 파라미터
│   ├── env.py             # 시뮬레이션 환경 (핵심 클래스)
│   ├── env_builder.py     # 환경 초기화 팩토리
│   ├── network.py         # 이동 네트워크 (이동 시간 계산)
│   ├── passenger.py       # 승객 클래스
│   ├── pending_buffer.py  # 지연 보상 버퍼
│   ├── replay_buffer.py   # 경험 리플레이 버퍼
│   ├── request.py         # 승차 요청 클래스
│   ├── request_status.py  # 요청 상태 열거형
│   ├── vehicle.py         # 차량 클래스
│   └── vehicle_status.py  # 차량 상태 열거형
├── data/
│   ├── od_matrix.csv          # 25노드 기점-종점 이동 시간 행렬
│   ├── requests_8.csv         # 소규모 테스트 데이터 (8건)
│   ├── requests_80.csv        # 메인 학습 데이터 (80건)
│   ├── travel_time.csv        # 이동 시간 참고
│   └── vehicle_positions.csv  # 초기 차량 위치
├── result/                # 학습 결과 저장 디렉토리
├── main.py                # 진입점 (학습/테스트)
├── test.py                # 테스트 파일 (미구현)
├── test.ipynb             # Jupyter 노트북
└── environment.yml        # Conda 환경 설정
```

### 계층 구조

```
┌─────────────────────────────────────────────────┐
│                  main.py (진입점)                 │
│         train_ddqn() / test_ddqn()               │
└────────────┬────────────────┬────────────────────┘
             │                │
    ┌─────────▼──────┐  ┌──────▼──────────┐
    │  DQNAgent      │  │ RideSharingEnv  │
    │  (agent.py)    │  │ (env.py)        │
    │                │  │                 │
    │  - 행동 선택    │  │  - 시뮬레이션   │
    │  - 모델 학습    │  │  - 상태 관리    │
    │  - 버퍼 관리    │  │  - 보상 계산    │
    └─────────┬──────┘  └──────┬──────────┘
              │                │
    ┌─────────▼──────────────────▼──────────────┐
    │           데이터 모델 계층                   │
    │  Vehicle / Request / DRTNetwork / Config   │
    └───────────────────────────────────────────┘
```

---

## 4. 강화학습 설계

### 상태 표현 (State Space)

총 3가지 입력으로 구성된 멀티인풋 신경망:

| 입력 | 형태 | 내용 |
|------|------|------|
| Vehicle Input | `(B, 2, 53)` | 차량 상태(4) + 위치 원핫(25) + 용량비율(1) + ... |
| Request Input | `(B, 8, 55)` | 요청 상태(3) + 출발/도착 원핫(25×2) + 대기시간 + 데드라인 |
| Relation Input | `(B, 2, 8, 2)` | 차량-요청 배정 상태, 정규화 이동 시간 |

### 행동 공간 (Action Space)

각 차량마다 **9가지 행동** 선택 가능:
- 행동 0: **REJECT** (요청 거절)
- 행동 1~8: **PICKUP/DROPOFF** (8개 요청 슬롯에 대해 승차/하차)

```python
# app/action_type.py
class ActionType(Enum):
    REJECT = 0
    PICKUP = 1
    DROPOFF = 2
```

### 보상 함수 (Reward Function)

| 이벤트 | 보상 | 설명 |
|--------|------|------|
| PICKUP 결정 | 0.5 ~ 1.5 | 이동 시간 및 대기 시간 기반 |
| DROPOFF 결정 | 0.5 ~ 1.5 | 이동 시간 및 차내 시간 기반 |
| 완료 보너스 | +0.5 | 요청 성공 완료 시 |
| 취소 페널티 | -1.0 | PICKUP 취소 시 |

**지연 보상 (Delayed Reward):** 승차/하차 결정 시점과 실제 완료 시점이 다르므로 `PendingBuffer`를 통해 나중에 보상을 확정합니다.

### 신경망 구조

```
Vehicle Input (B,2,53) ──► TimeDistributed Dense ──┐
                                                     ├──► Concat & Broadcast ──► Dense ──► Q-values (9)
Request Input (B,8,55) ──► TimeDistributed Dense ──┘

Relation Input (B,2,8,2) ────────────────────────────► (보조 입력)
```

### 하이퍼파라미터

| 파라미터 | 값 |
|----------|-----|
| Gamma (할인율) | 0.99 |
| Epsilon 초기값 | 1.0 |
| Epsilon 최솟값 | 0.05 |
| Epsilon 감소율 | 0.995 (에피소드당) |
| Hidden Dim | 256 |
| Batch Size | 32 |
| Learning Rate | 1e-5 |
| Replay Buffer 크기 | 10,000 |
| 타겟 네트워크 업데이트 | 500 steps |

---

## 5. 시뮬레이션 환경

### 네트워크 구성

- **노드 수:** 25개
- **데이터:** `data/od_matrix.csv` (기점-종점 이동 시간 행렬)
- **클래스:** `DRTNetwork` (`app/network.py`)
  - `get_duration(origin, dest)` 메서드로 이동 시간 조회

### 환경 설정 (`app/config.py`)

```python
MAX_NUM_VEHICLES = 2    # 차량 대수
MAX_NUM_REQUEST = 8     # 동시 처리 최대 요청 수
VEH_CAPACITY = 5        # 차량 1대 최대 탑승 인원
MAX_WAIT_TIME = 10      # 최대 대기 시간 (시간 단위)
MAX_INVEHICLE_TIME = 10 # 최대 차내 이동 시간
```

### 요청 생명주기

```
PENDING ──► ACCEPTED ──► PICKEDUP ──► SERVED
                │
                └──► CANCELLED (타임아웃 or 거절 후 취소)
```

### 차량 상태

```
IDLE ──► PICKUP (승차 이동 중) ──► DROPOFF (하차 이동 중) ──► IDLE
       └──► REJECT (현재 요청 거절)
```

### 에피소드 진행 흐름

```
env.reset()
    │
    ▼
while 처리할 요청 존재:
    │
    ├── 유휴 차량(IDLE) 확인
    ├── 유효 행동 마스크 생성
    ├── agent.act(state, mask) ──► 행동 선택
    ├── env.step(action) ──► 상태 전이 + 즉각 보상
    ├── PendingBuffer 확인 ──► 지연 보상 처리
    └── 10 steps마다 agent.train()

에피소드 종료 → 메트릭 로깅 → 모델 저장
```

---

## 6. 학습 파이프라인

### 학습 설정

| 항목 | 값 |
|------|-----|
| 총 에피소드 수 | 500 |
| 학습 데이터 | `data/requests_80.csv` (80건) |
| 차량 초기 위치 | `data/vehicle_positions.csv` |
| 그리드 서치 | hidden_dim × batch_size × learning_rate |

### 학습 루프 (main.py)

```python
# 하이퍼파라미터 그리드 서치
for hidden_dim, batch_size, lr in hyperparams:
    env = EnvBuilder.build(...)
    agent = DQNAgent(...)

    for episode in range(500):
        state = env.reset()

        while not done:
            mask = env.get_action_mask()
            action = agent.act(state, mask)   # ε-greedy
            next_state, reward, done = env.step(action)
            agent.remember(transition)

            if step % 10 == 0:
                agent.train()                 # 미니배치 학습

        epsilon *= 0.995                      # 탐색률 감소
        if best_reward: agent.save_model()
```

### 학습 vs 테스트 모드

| 모드 | 함수 | 특징 |
|------|------|------|
| 학습 | `train_ddqn()` | ε-greedy 탐색 + 가중치 업데이트 |
| 테스트 | `test_ddqn()` | ε=0 (완전 탐욕) + 저장된 가중치 로드 |

---

## 7. 결과 출력 구조

```
result/
└── hd{hidden_dim}_bs{batch_size}_lr{learning_rate}/
    ├── episode_001_vehicle.csv   # 차량별 성능 지표
    ├── episode_001_request.csv   # 요청별 처리 결과
    ├── episode_001_seq.txt       # 승차/하차 이벤트 시퀀스
    └── episodes.csv              # 에피소드별 집계 통계
└── hd256_bs32_lr0.00001.h5       # 최적 모델 가중치
```

### 로깅 지표

**차량 성능 (`*_vehicle.csv`):**
- 수락/거절/서빙한 요청 수
- 유휴 시간 vs 주행 시간 비율

**요청 결과 (`*_request.csv`):**
- 최종 상태 (SERVED / CANCELLED)
- 실제 대기 시간
- 실제 차내 이동 시간
- 우회율 (detour ratio)

---

## 8. 코드 품질 및 개선 사항

### 강점

| 항목 | 내용 |
|------|------|
| 모듈화 | 도메인별 명확한 클래스 분리 (차량, 요청, 네트워크, 에이전트) |
| 설정 집중화 | `config.py` 단일 파일로 모든 파라미터 관리 |
| 지연 보상 처리 | `PendingBuffer`로 비동기 보상 문제 우아하게 해결 |
| 그리드 서치 | 하이퍼파라미터 탐색 구조 내장 |
| 로깅 | 에피소드/차량/요청 단위 상세 CSV 로깅 |

### 개선 여지

| 항목 | 현황 | 개선 방향 |
|------|------|-----------|
| 테스트 코드 | `test.py` 완전 미구현 | 유닛 테스트 작성 (환경 리셋, 보상 계산 등) |
| README | 내용 거의 없음 | 설치/실행 방법, 알고리즘 설명 추가 |
| 하드코딩 경로 | 데이터 파일 경로가 코드에 직접 삽입 | config.py 또는 CLI 인자로 분리 |
| 확장성 | 차량 수/요청 수 변경 시 벡터 차원 재계산 필요 | 동적 차원 계산 로직 추가 |
| 시각화 | 학습 곡선 출력 없음 | TensorBoard 또는 matplotlib 연동 |
| 문서화 | docstring 미작성 | 핵심 클래스/함수에 문서 추가 |

---

## 요약

KW_DRT는 DDQN 기반의 수요 응답형 교통 배차 최적화 연구 프로젝트로, 강화학습 환경 구성부터 에이전트 학습, 결과 로깅까지 전체 파이프라인이 잘 구현되어 있습니다. 멀티인풋 신경망으로 차량-요청 관계를 효과적으로 표현하며, PendingBuffer를 통한 지연 보상 처리가 특징적입니다. 테스트 코드 보완과 문서화 개선이 주요 과제입니다.
