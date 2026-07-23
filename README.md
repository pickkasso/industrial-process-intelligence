# Industrial Process Intelligence Platform

범용 제조 공정 데이터 분석 및 의사결정 지원 플랫폼입니다. 반도체·배터리·자동차 공정 CSV를 대상으로 supervised 품질 예측과 anomaly-only 이상 탐지를 지원합니다.

## Project overview

- CSV 업로드 후 열 역할 확인, 분석 실행, 보고서 검토, 설정 프리셋 재사용
- **SUPERVISED**: target 기반 예측·성능 수용·추천
- **ANOMALY_ONLY**: target 없이 비지도 이상 탐지·연관 변수 진단
- 결과는 모델 기반 의사결정 지원이며 운영 명령이 아닙니다

## Requirements

- Python 3.11
- Windows PowerShell 기준 명령 (아래 Installation / Run)
- 의존성은 `pyproject.toml` / `requirements.txt`에 정의됨 (Streamlit, Polars, scikit-learn, Pydantic 등)

## Installation

프로젝트 루트에서:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
```

가상환경을 활성화하지 않은 경우:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

## Run

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

## Quick start (UI)

1. Upload a CSV file.
2. Select supervised or anomaly-only analysis.
3. Review the suggested columns and safety-critical settings.
4. Run the analysis and inspect the report.
5. Export reusable configuration when needed.

자세한 운영 안내는 [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)를 참고하세요.

## Analysis modes

### SUPERVISED

- target column 필요
- regression 또는 지원되는 task 필요
- 독립 테스트 성능 기준 필요
- target이 constant/all-null이면 실행 불가
- recommendation은 실제 controllable·verified·constrained 변수만 사용

### ANOMALY_ONLY

- target 불필요
- 비지도 이상 탐지 및 관련 변수 진단
- 통계적 이상은 실제 결함을 의미하지 않음
- cohort filter는 선택 사항
- anomaly recommendation은 기본 비활성화
- 추천 활성화 시 실제 조절 가능한 변수만 허용

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest
```

간단 실행:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Documentation

- 운영 가이드: [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)
- 제품 요구사항: [`docs/PRD.md`](docs/PRD.md)
- 기술 명세: [`docs/TECH_SPEC.md`](docs/TECH_SPEC.md)
- 개발 단계: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- Agent 작업 규칙: [`AGENTS.md`](AGENTS.md)

## Important notice

분석 결과와 공정조건 추천은 관측 데이터 및 학습 모델 기반의 의사결정 지원 정보입니다. 실제 공정 적용 전 장비 운전 범위, 안전 제약, 공정 엔지니어 검토, 추가 실험, 현장 검증이 필요합니다. 모델이 제시한 연관성을 인과관계로 단정하지 않습니다.
