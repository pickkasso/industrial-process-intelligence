# Industrial Process Intelligence Platform
# Product Requirements Document

이 문서는 제품의 목표, 지원 산업, 사용자 기능, 분석 결과, 이상탐지, 원인진단, 추천, UI 및 최종 산출물을 정의한다.

아래 내용은 원본 구축 지시서에서 이동한 것이며,
문장과 요구사항을 임의로 축약하거나 변경하지 않는다.

## 2. 핵심 목표

프로그램은 사용자가 CSV, XLSX 또는 Parquet 파일을 업로드하면 다음 흐름을 수행해야 한다.

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

이 프로그램은 단순 정상·이상 분류기가 아니다.

다음 질문에 답할 수 있어야 한다.

1. 어떤 산업 데이터인가?
2. 각 열은 어떤 역할을 하는가?
3. 데이터 품질에 문제가 있는가?
4. 어떤 분석 경로가 적합한가?
5. 어떤 샘플과 구간이 이상인가?
6. 이상 유형은 무엇인가?
7. 어떤 변수가 이상에 기여했는가?
8. 사용자가 조정 가능한 변수는 무엇인가?
9. 최소한의 변경으로 품질을 개선할 수 있는가?
10. 변경 후 예상 결과와 불확실성은 무엇인가?
11. 현재 분석 결과를 얼마나 신뢰할 수 있는가?

## 5. 우선 지원 산업

### Semiconductor

대표 공정변수:

- RF source power
- RF bias power
- gas flow
- chamber pressure
- temperature
- process time
- matcher position

대표 센서변수:

- OES intensity
- reflected power
- measured pressure
- measured temperature

대표 품질변수:

- yield
- etch rate
- etched thickness
- film thickness
- critical dimension
- uniformity
- defect count
- pass/fail

### Battery

대표 공정변수:

- coating thickness
- slurry viscosity
- drying temperature
- drying speed
- calendaring pressure
- formation current
- formation voltage

대표 센서·상태변수:

- moisture
- electrode density
- process temperature
- internal resistance

대표 품질변수:

- capacity
- coulombic efficiency
- cycle life
- swelling
- defect rate
- pass/fail

### Automotive Manufacturing

대표 공정변수:

- torque command
- rotation speed
- feed rate
- pressure
- temperature setpoint
- welding current
- welding voltage
- machining time

대표 센서·상태변수:

- vibration
- measured torque
- measured pressure
- measured temperature

대표 품질변수:

- dimensional error
- surface roughness
- welding quality
- strength
- defect count
- scrap rate
- pass/fail

산업 판별은 단순 키워드 일치만으로 수행하지 않는다.

다음 신호를 함께 사용한다.

- 열 이름과 동의어
- 단위
- 값의 범위
- 시간축 존재 여부
- 열 간 관계
- 범주형 값
- 메타데이터
- 파일명
- 사용자가 입력한 설명

산업 추정 결과에는 다음을 제공한다.

- 추정 산업
- 산업별 점수
- confidence
- 판단 근거
- 불확실한 요소
- 사용자 확인 필요 여부

신뢰도가 설정된 임계값보다 낮으면 자동 확정하지 않는다.

## 7. 열 역할 자동 분류

업로드된 열을 다음 역할로 분류한다.

- IDENTIFIER
- TIME
- CONTROLLABLE_PROCESS
- STATE_SENSOR
- CONTEXT
- TARGET_QUALITY
- DERIVED_FEATURE
- UNKNOWN

### Identifier

예:

- sample ID
- wafer ID
- lot ID
- batch ID
- equipment ID
- serial number

기본 모델 feature에서는 제외하지만 그룹 분할과 추적에 사용한다.

### Time

예:

- timestamp
- process time
- cycle
- sequence
- step number

정렬, 시계열 분석, 변화점 탐지에 사용한다.

### Controllable Process

엔지니어가 조정할 수 있는 변수다.

추천 엔진은 원칙적으로 이 역할의 변수만 변경할 수 있다.

### State or Sensor

관찰 가능하지만 직접 변경하기 어려운 변수다.

### Context

설비, 자재, 작업자, 공급자, 생산일 등 공정 환경을 나타낸다.

### Target or Quality

예측하거나 규격을 판정할 결과 변수다.

### Derived Feature

프로그램이 계산한 통계량과 특징값이다.

자동 분류에는 다음 근거를 사용한다.

- 열 이름
- 데이터 타입
- cardinality
- 단위
- 범위
- 시간적 특성
- 다른 열과의 관계
- 산업 프로필
- 사용자 설명

각 열에 대해 다음 정보를 반환한다.

- 추정 역할
- confidence
- 판단 근거
- 대체 후보 역할
- 모델 사용 여부

Streamlit UI에서 사용자가 역할을 수정할 수 있어야 한다.

사용자 수정 결과는 이후 분석의 최우선 설정으로 사용한다.

## 9. 데이터 품질 진단

모델 학습 전에 다음을 검사한다.

- 결측치 비율
- 상수 및 준상수 열
- 중복 행과 중복 열
- 높은 상관관계
- 비정상 범위
- 물리적으로 불가능한 값
- 범주 불균형
- 클래스 불균형
- feature 수 대비 표본 수
- target leakage
- ID leakage
- 미래 정보 사용 가능성
- 동일 batch 또는 run의 train-test 혼입
- 시계열 자기상관
- 분포 변화
- 센서 고착
- spike
- scale 차이
- 혼합 단위
- 비정상적인 범주 증가

Data Quality Score 를 0~100으로 계산한다.

점수는 단일 임의값으로 생성하지 말고 다음을 제공한다.

- 전체 점수
- 항목별 점수
- 감점 근거
- 심각도
- 모델링 영향
- 권장 조치
- 계산식 또는 가중치

가중치는 설정 파일에서 변경할 수 있어야 한다.

## 10. 분석 경로 선택

프로그램은 가능한 분석 목적을 추천하고 사용자가 최종 선택하도록 한다.

지원 경로:

- REGRESSION
- CLASSIFICATION
- UNSUPERVISED_ANOMALY
- TIME_SERIES_ANOMALY
- RESIDUAL_ANOMALY
- DRIFT_DETECTION

추천 근거:

- target 존재 여부
- target dtype
- 고유값 수
- timestamp 존재 여부
- group 구조
- 라벨 존재 여부
- 데이터 크기
- 결측치
- 클래스 불균형
- 시계열 길이

각 경로에 대해 실행 가능 여부와 제한사항을 표시한다.

## 13. 이상 유형

이상 결과를 단일 anomaly score만으로 표현하지 않는다.

다음 유형을 지원한다.

- DATA_QUALITY
- PROCESS_INPUT
- STATE_SENSOR
- QUALITY_OUTPUT
- RELATIONSHIP
- DRIFT
- MULTIVARIATE_COMBINATION

각 이상 이벤트에는 다음 정보가 있어야 한다.

- anomaly ID
- anomaly type
- anomaly score
- severity
- sample ID 또는 시간
- group ID
- 정상 기준 대비 편차
- 주요 기여 변수
- 모델 confidence
- 사용한 detector
- 탐지 근거

severity 기준은 설정 가능하게 구현한다.

## 14. Root-Cause Diagnosis

가능한 방법:

- SHAP
- permutation importance
- 정상군과 이상군 분포 비교
- robust z-score contribution
- PCA contribution
- residual contribution
- 변화점 전후 변수 비교
- 상관관계 변화
- 유사 정상 샘플 비교
- 유사 과거 이상 사례 검색

결과 형식:

1. 영향력이 큰 변수
2. 증가 또는 감소 방향
3. 정상 기준 대비 편차
4. 변수 역할
5. 직접 조정 가능 여부
6. 분석 근거
7. confidence
8. 추가 확인이 필요한 항목

상관관계를 인과관계로 단정하지 않는다.

다음과 같은 표현을 사용한다.

> 해당 변수는 현재 모델과 관측 데이터에서 이상 점수 증가와 강하게 연관되어 있습니다.  
> 인과관계가 확정된 것은 아니므로 실제 공정 검증이 필요합니다.

## 15. Recommendation Engine

추천 대상은 사용자가 조정 가능하다고 확정한 CONTROLLABLE_PROCESS 변수로 제한한다.

각 변수에 대해 다음 제약을 입력할 수 있어야 한다.

- 조정 가능 여부
- 최소값
- 최대값
- 최대 변경 비율
- 변경 단위
- 변경 비용
- 안전 제약
- 장비 운전 범위
- 동시에 변경 가능한 변수 수
- 고정 여부

추천은 counterfactual optimization으로 수행한다.

목적:

```text
minimize:
 process_change_cost
 + quality_target_penalty
 + anomaly_score_penalty
 + extrapolation_penalty
 + uncertainty_penalty
 + safety_violation_penalty
```
 
 제약조건:

- controllable bounds
- user constraints
- equipment operating range
- safety constraints
- maximum number of changed variables
- maximum change ratio
- 학습 데이터 지원 범위

추천 결과:

- 현재 조건
- 추천 조건
- 절대 변경량
- 변경 비율
- 조정 우선순위
- 조정 후 예상 품질
- 조정 후 예상 anomaly score
- 예측 불확실성
- extrapolation 경고
- 추천 근거
- 실제 공정 검증 필요성

확정적 표현을 사용하지 않는다.

적절한 표현:

> 학습 데이터가 충분히 포함하는 범위에서 압력을 약 3% 낮추고 온도를 유지할 경우,  
> 목표 품질 범위에 진입할 가능성이 가장 높게 예측됩니다.

부적절한 표현:

> 압력을 3% 낮추면 반드시 불량이 해결됩니다.

## 16. 추천 안전장치

다음 조건에서는 강한 공정 변경 추천을 생성하지 않는다.

- 표본 수가 부족함
- target이 없음
- 조절 가능 변수가 없음
- 모델 성능이 기준 이하
- 추천값이 학습 범위를 벗어남
- 입력 제약조건이 없음
- 데이터 누수가 의심됨
- 불확실성이 높음
- 관계 안정성이 낮음
- 산업 판별 신뢰도가 낮음

이 경우 다음 메시지를 사용한다.

> 현재 데이터만으로는 공정조건 변경의 효과를 신뢰성 있게 추정하기 어렵습니다.  
> 추가 데이터, 공정 제약조건 또는 현장 검증이 필요합니다.

## 17. ±5% 처리 원칙

±5%를 프로그램의 고정 예측 정확도로 주장하지 않는다.

다음 의미로만 사용한다.

- 사용자 정의 품질 허용오차
- target specification
- prediction interval
- 검증 데이터에서 측정한 실제 오차
- 공정 허용 범위

예:

```text
Target = 100
Tolerance = ±5%
Specification range = [95, 105]
```

모델의 실제 예측 오차가 ±5% 이내인지는 별도 검증 결과로 표시한다.

## 18. What-if Simulation

사용자가 공정변수를 변경하면 다음을 다시 계산한다.

- 예측 품질
- 규격 진입 여부
- anomaly score
- 주요 feature contribution
- 추천안 대비 차이
- prediction interval
- extrapolation 위험
- 모델 confidence

현재 조건, 추천 조건, 사용자 변경 조건을 표와 그래프로 비교한다.

실시간 재계산이 어렵다면 캐시와 경량 추론을 사용하되, 실제 재학습이 필요한 작업과 혼동하지 않도록 구분한다.

## 20. Streamlit UI

MVP는 Streamlit으로 구현한다.

분석 로직은 UI와 분리해 추후 FastAPI 또는 배치 프로그램에서 재사용할 수 있어야 한다.

### Page 1: Data Upload

- CSV, XLSX, Parquet 업로드
- 데이터 preview
- 파일 정보
- 경량 프로파일
- Data Quality Score

### Page 2: Industry and Schema Mapping

- 산업 자동 추정
- confidence와 근거
- 사용자 산업 선택
- 열 역할 자동 분류
- 역할 수정
- 조절 가능 변수 지정

### Page 3: Data Preparation

- 정렬
- 필터
- 결측치 처리
- group 설정
- 전처리 preview
- preprocessing log

### Page 4: Analysis Setup

- 분석 목적 추천
- 사용자 최종 선택
- target 지정
- split 방식
- 모델 후보
- 분석 옵션

### Page 5: Results

- 모델 성능
- 이상 샘플
- 이상 유형
- severity
- 원인 변수
- 시각화
- 불확실성

### Page 6: Recommendation

- 변수별 제약조건
- 추천 변경안
- 예상 품질
- 예상 anomaly score
- extrapolation 경고
- what-if simulation

### Page 7: Report

- 분석 조건
- 데이터 품질
- 데이터 분할
- 모델
- 성능
- 이상 결과
- 원인 진단
- 추천
- 한계
- CSV 및 HTML export

PDF는 안정적인 HTML 보고서가 완성된 이후 확장 기능으로 구현한다.

## 21. 대시보드 핵심 지표

우선 표시할 정보:

- Industry
- Industry Confidence
- Analysis Task
- Data Quality Score
- Overall Process Health Score
- Predicted Quality
- Specification Status
- Anomaly Type
- Anomaly Severity
- Top Root-Cause Features
- Recommended Adjustment
- Expected Improvement
- Model Confidence
- Extrapolation Warning

Overall Process Health Score 는 다음 요소를 조합한다.

- data quality
- process anomaly
- sensor anomaly
- quality deviation
- residual anomaly
- drift
- prediction uncertainty

점수 계산식, 가중치, 누락된 요소를 UI에 공개한다.

## 28. 최종 산출물

다음을 제공한다.

1. 실행 가능한 Python 프로젝트
2. pyproject.toml 또는 requirements.txt
3. Streamlit 실행 명령
4. README
5. 시스템 아키텍처 설명
6. 산업별 합성 데이터 생성기
7. sample dataset
8. 단위 테스트
9. 통합 테스트
10. sample analysis report
11. 모델 한계 설명
12. 추천 결과 한계 설명
13. 향후 확장 계획

README에는 최소한 다음을 포함한다.

```text
python -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run app.py
pytest
```

Windows 명령이 필요한 부분은 별도로 제공한다.