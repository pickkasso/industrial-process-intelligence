# Industrial Process Intelligence Platform

반도체, 배터리, 자동차 제조 데이터를 지원하는 범용 공정 이상탐지·진단·예측·의사결정 지원 플랫폼입니다.

사용자가 CSV, XLSX 또는 Parquet 파일을 업로드하면 데이터 이해부터 산업 판별, 데이터 품질 진단, 이상탐지, 원인 분석, 공정조건 추천, What-if Simulation 및 보고서 생성까지 수행하는 것을 목표로 합니다.

## Project Status

현재 상태: 설계 및 프로젝트 초기 구성 단계

아직 전체 애플리케이션 코드는 구현되지 않았습니다.  
구현은 `docs/ROADMAP.md`의 개발 단계에 따라 순차적으로 진행합니다.

## Core Workflow

```text
Data Understanding
→ Industry Identification
→ Schema Mapping
→ Data Validation
→ Preprocessing
→ Task Identification
→ Model Selection
→ Anomaly Detection or Quality Prediction
→ Root-Cause Diagnosis
→ Actionable Recommendation
→ What-if Verification
→ Report Generation
```

## Supported Industries

### Semiconductor

대표 분석 대상:

- RF source power
- RF bias power
- gas flow
- chamber pressure
- temperature
- OES intensity
- reflected power
- yield
- etch rate
- film thickness
- critical dimension
- uniformity
- defect count
- pass/fail

### Battery

대표 분석 대상:

- coating thickness
- slurry viscosity
- drying temperature
- calendaring pressure
- formation current
- formation voltage
- moisture
- electrode density
- capacity
- coulombic efficiency
- cycle life
- swelling
- defect rate
- pass/fail

### Automotive Manufacturing

대표 분석 대상:

- torque command
- rotation speed
- feed rate
- pressure
- temperature setpoint
- welding current
- welding voltage
- vibration
- dimensional error
- surface roughness
- welding quality
- strength
- defect count
- scrap rate
- pass/fail

## Main Features

- CSV, XLSX 및 Parquet 파일 업로드
- 산업 자동 추정과 사용자 확인
- 열 역할 자동 분류와 사용자 수정
- 데이터 품질 진단
- Data Quality Score 계산
- 회귀 및 분류
- 비지도 이상탐지
- 시계열 이상탐지
- Residual Anomaly 분석
- Drift Detection
- Root-Cause Diagnosis
- 조절 가능한 공정변수 기반 Recommendation
- What-if Simulation
- CSV 및 HTML 보고서 생성
- 합성 데이터 기반 전체 파이프라인 검증

## Documentation

프로젝트의 상세 요구사항과 개발 규칙은 다음 문서를 따릅니다.

- 제품 요구사항: [`docs/PRD.md`](docs/PRD.md)
- 기술 명세: [`docs/TECH_SPEC.md`](docs/TECH_SPEC.md)
- 개발 단계: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- Codex 및 Cursor 작업 규칙: [`AGENTS.md`](AGENTS.md)

## Requirements

- Python 3.11 이상
- Streamlit
- Polars
- Pandas
- PyArrow
- scikit-learn
- NumPy
- SciPy
- Plotly
- Pydantic
- PyYAML
- joblib
- pytest

## Windows Setup

프로젝트 폴더에서 다음 명령을 순서대로 실행합니다.

### 1. 가상환경 생성

```powershell
py -3.11 -m venv .venv
```

### 2. PowerShell 실행 정책 임시 설정

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 3. 가상환경 활성화

```powershell
.\.venv\Scripts\Activate.ps1
```

정상적으로 활성화되면 터미널 앞에 다음과 같이 표시됩니다.

```text
(.venv) PS C:\Users\<USER>\dev\industrial-process-intelligence>
```

### 4. pip 및 기본 도구 업그레이드

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 5. 필수 패키지 설치

```powershell
python -m pip install -r requirements.txt
```

가상환경 활성화가 되지 않아도 다음 명령으로 설치할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Linux or macOS Setup

### 1. 가상환경 생성

```bash
python -m venv .venv
```

### 2. 가상환경 활성화

```bash
source .venv/bin/activate
```

### 3. 패키지 설치

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## Editable Installation

`pyproject.toml`의 프로젝트 설정과 의존성이 완성된 이후 다음 명령을 사용할 수 있습니다.

```bash
pip install -e .
```

## Run

애플리케이션 구현 이후 프로젝트 최상단에서 다음 명령으로 실행합니다.

```powershell
streamlit run app.py
```

가상환경 실행 파일을 직접 사용할 경우:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

## Test

테스트 구현 이후 다음 명령을 실행합니다.

```powershell
python -m pytest
```

간단한 결과만 확인하려면 다음을 사용합니다.

```powershell
python -m pytest -q
```

## Project Structure

```text
industrial-process-intelligence/
├── docs/
│   ├── PRD.md
│   ├── TECH_SPEC.md
│   └── ROADMAP.md
├── src/
├── tests/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

세부 구현 디렉터리는 설계 단계가 완료된 이후 `docs/TECH_SPEC.md`의 권장 프로젝트 구조에 따라 생성합니다.

## Development Principles

- 전체 코드를 한 번에 생성하지 않습니다.
- 원본 데이터와 처리 데이터를 분리합니다.
- 데이터 누수 가능성을 모델 성능보다 먼저 검사합니다.
- 무작위 행 분할을 기본값으로 사용하지 않습니다.
- 공정 추천을 인과적 사실처럼 표현하지 않습니다.
- 추천 결과에 불확실성과 extrapolation 위험을 포함합니다.
- 선택 패키지가 없어도 MVP가 실행되도록 fallback을 제공합니다.
- 실행하지 않은 테스트를 통과했다고 기록하지 않습니다.
- UI와 분석 로직을 분리합니다.

## Phase 1 Scope

Phase 1 MVP에는 다음 기능을 구현합니다.

- 파일 업로드
- 데이터 preview와 프로파일링
- 산업 자동 추정
- schema mapping
- 사용자 수정 UI
- 정렬과 기본 전처리
- Data Quality Score
- 회귀
- 분류
- Isolation Forest
- 기본 설명 기능
- 단순 제약 기반 recommendation
- Streamlit 대시보드
- 합성 데이터와 테스트

이미지 기반 결함 분류는 MVP 범위에서 제외합니다.

## Important Notice

이 플랫폼의 분석 결과와 공정조건 추천은 관측 데이터 및 학습 모델을 기반으로 한 의사결정 지원 정보입니다.

추천 결과를 실제 공정에 적용하기 전에 다음 사항을 반드시 확인해야 합니다.

- 장비 운전 범위
- 안전 제약
- 공정 엔지니어 검토
- 추가 실험
- 현장 검증

모델이 제시한 연관성을 인과관계로 단정하지 않습니다.