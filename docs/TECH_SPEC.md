# Industrial Process Intelligence Platform
# Technical Specification

이 문서는 기술 스택, 산업 확장 아키텍처, 데이터 적재와 추적성, 모델 전략, 데이터 누수 방지, 성능 요구사항, 프로젝트 구조, 공통 인터페이스, 합성 데이터 및 테스트 원칙을 정의한다.

아래 내용은 원본 구축 지시서에서 이동한 것이며,
문장과 요구사항을 임의로 축약하거나 변경하지 않는다.

## 4. 기술 스택

### 필수

- Python 3.11 이상
- Streamlit
- Polars
- Pandas 호환 계층
- PyArrow
- scikit-learn
- NumPy
- SciPy
- Plotly
- Pydantic
- PyYAML
- joblib
- pytest

### 선택 의존성

설치되어 있거나 실제로 필요한 경우에만 사용한다.

- SHAP
- XGBoost
- LightGBM
- CatBoost
- Optuna
- ruptures
- FastAPI
- Uvicorn

선택 의존성이 없어도 MVP가 실행되도록 fallback을 구현한다.

## 6. 산업 확장 구조

산업별 처리를 거대한 if-elif 구조로 구현하지 않는다.

Adapter 또는 Plugin 패턴을 사용한다.

필수 인터페이스:

```python
class BaseIndustryProfile(ABC):
    industry_name: str

    @abstractmethod
    def score_industry(
        self,
        metadata: DatasetMetadata,
    ) -> IndustryScore:
        ...

    @abstractmethod
    def get_schema_hints(self) -> SchemaHints:
        ...

    @abstractmethod
    def get_preprocessing_rules(
        self,
    ) -> PreprocessingRules:
        ...

    @abstractmethod
    def get_default_model_candidates(
        self,
        task: AnalysisTask,
    ) -> list[ModelSpec]:
        ...

    @abstractmethod
    def validate_physical_ranges(
        self,
        frame: DataFrameLike,
    ) -> list[ValidationIssue]:
        ...

    @abstractmethod
    def get_recommendation_constraints(
        self,
    ) -> list[Constraint]:
        ...
```

구현 프로필:

- SemiconductorProfile
- BatteryProfile
- AutomotiveProfile

각 프로필은 다음 정보를 제공해야 한다.

- 키워드 및 동의어
- 예상 단위
- 대표 변수 역할
- 물리적 범위 힌트
- 기본 전처리 규칙
- 기본 분석 모델
- 산업별 시각화
- 변수 변경 시 주의사항
- 추천 문장 템플릿

새 산업은 기존 코어 로직을 수정하지 않고 프로필 등록만으로 추가할 수 있어야 한다.

## 8. 데이터 적재와 추적성

지원 형식:

- CSV
- XLSX
- Parquet

필수 기능:

- 원본 행 순서 보존
- `_original_row_id` 생성
- 원본 데이터 불변성 유지
- processed dataset 별도 생성
- 지정 열 정렬
- timestamp 정렬
- group 기반 정렬
- 정렬 전후 행 추적
- 변환 로그 기록
- dtype 자동 추정
- 숫자형 열에 포함된 문자열 탐지
- 중복 행 탐지
- 중복 timestamp 탐지
- 시간 순서 역전 탐지
- 결측 시간 구간 탐지
- 반복값과 센서 고착 탐지
- 혼합 단위 의심 탐지

모든 전처리 단계는 구조화된 로그로 기록한다.

```python
class PreprocessingEvent(BaseModel):
    step_name: str
    affected_columns: list[str]
    rows_before: int
    rows_after: int
    parameters: dict[str, Any]
    warnings: list[str]
    timestamp: datetime
```

## 11. 모델 전략

하나의 모델을 모든 데이터에 강제하지 않는다.

### Regression

기본 후보:

- DummyRegressor
- LinearRegression
- Ridge
- RandomForestRegressor
- HistGradientBoostingRegressor

선택 후보:

- XGBoost
- LightGBM
- CatBoost

평가 지표:

- MAE
- RMSE
- R²
- SMAPE 또는 MAPE
- prediction interval coverage

### Classification

기본 후보:

- DummyClassifier
- LogisticRegression
- RandomForestClassifier
- HistGradientBoostingClassifier

선택 후보:

- XGBoost
- LightGBM
- CatBoost

평가 지표:

- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- false alarm rate
- confusion matrix

Accuracy를 주요 단독 지표로 사용하지 않는다.

### Unsupervised Anomaly Detection

후보:

- robust z-score
- PCA reconstruction error
- PCA T²
- PCA SPE
- Isolation Forest
- Local Outlier Factor
- One-Class SVM

### Time-Series Anomaly Detection

후보:

- rolling mean
- rolling standard deviation
- EWMA
- rolling z-score
- trend slope
- coefficient of variation
- autocorrelation
- frequency-domain feature
- change point detection

Autoencoder와 sequence model은 충분한 데이터가 있을 때만 선택 후보로 제공한다.

### Residual Anomaly

```text
Residual = Actual Quality - Predicted Quality
```

입력 조건은 정상인데 residual이 큰 경우 다음 후보를 표시한다.

- 센서 또는 계측 이상
- 원재료 변화
- 장비 열화
- 미관측 외란
- 새로운 공정 상태
- 모델 미적합

## 12. 데이터 분할과 누수 방지

무작위 행 분할을 기본값으로 사용하지 않는다.

가능한 경우 다음 단위로 Group Split을 수행한다.

- wafer
- lot
- batch
- run
- equipment
- production date

시계열 데이터는 과거 학습·미래 검증 방식을 사용한다.

반드시 검사할 사항:

- 동일 group의 train-test 중복
- target에서 파생된 feature
- 미래 정보가 포함된 rolling feature
- 전처리를 전체 데이터에 먼저 fit한 경우
- ID가 target을 암시하는 경우
- timestamp가 결과를 직접 암시하는 경우

결과 화면에 다음을 표시한다.

- split 방식
- group 열
- train·validation·test 크기
- 시간 범위
- 누수 검사 결과

## 19. 성능 요구사항

다음 최적화를 우선 적용한다.

- Polars 우선
- PyArrow 기반 입출력
- Parquet 캐시
- lazy loading
- column pruning
- dtype downcasting
- chunk processing
- 병렬 처리
- 전처리 결과 캐시
- 모델 캐시
- preview sampling
- 모델별 time budget
- 불필요한 딥러닝 방지

파일 업로드 직후 전체 모델을 학습하지 않는다.

실행 순서:

1. 경량 프로파일링
2. 표본 기반 모델 후보 평가
3. 실행 가능한 모델 필터링
4. 적합한 후보만 전체 데이터에 적용

측정 시간:

- file loading
- profiling
- preprocessing
- training
- inference
- diagnosis
- recommendation
- report generation

고정 처리시간을 보장하지 않는다.

## 22. 권장 프로젝트 구조

```text
industrial-process-intelligence/
├── app.py
├── pyproject.toml
├── README.md
├── .env.example
├── config/
│   ├── application.yaml
│   ├── semiconductor.yaml
│   ├── battery.yaml
│   └── automotive.yaml
├── src/
│   └── process_intelligence/
│       ├── __init__.py
│       ├── core/
│       │   ├── schemas.py
│       │   ├── enums.py
│       │   ├── protocols.py
│       │   └── exceptions.py
│       ├── data/
│       │   ├── loader.py
│       │   ├── profiler.py
│       │   ├── validator.py
│       │   ├── sorter.py
│       │   ├── preprocessor.py
│       │   └── lineage.py
│       ├── industries/
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── semiconductor.py
│       │   ├── battery.py
│       │   └── automotive.py
│       ├── routing/
│       │   ├── industry_router.py
│       │   ├── schema_mapper.py
│       │   └── task_router.py
│       ├── features/
│       │   ├── tabular.py
│       │   ├── timeseries.py
│       │   └── residual.py
│       ├── models/
│       │   ├── base.py
│       │   ├── registry.py
│       │   ├── regression.py
│       │   ├── classification.py
│       │   ├── anomaly.py
│       │   └── timeseries.py
│       ├── evaluation/
│       │   ├── splitting.py
│       │   ├── metrics.py
│       │   └── leakage.py
│       ├── diagnosis/
│       │   ├── root_cause.py
│       │   ├── explainability.py
│       │   ├── residual.py
│       │   └── drift.py
│       ├── recommendation/
│       │   ├── counterfactual.py
│       │   ├── constraints.py
│       │   ├── optimizer.py
│       │   └── verifier.py
│       ├── reporting/
│       │   ├── report_builder.py
│       │   ├── templates.py
│       │   └── visualizer.py
│       ├── synthetic/
│       │   ├── common.py
│       │   ├── semiconductor.py
│       │   ├── battery.py
│       │   └── automotive.py
│       └── utils/
│           ├── logging.py
│           ├── timing.py
│           ├── cache.py
│           └── optional_dependencies.py
├── pages/
│   ├── 01_data_upload.py
│   ├── 02_schema_mapping.py
│   ├── 03_data_preparation.py
│   ├── 04_analysis_setup.py
│   ├── 05_results.py
│   ├── 06_recommendation.py
│   └── 07_report.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── sample_data/
│   ├── semiconductor/
│   ├── battery/
│   └── automotive/
└── artifacts/
    ├── models/
    ├── cache/
    └── reports/
```

UI 코드에서 직접 모델을 학습하거나 데이터를 전처리하지 않는다.

## 23. 공통 인터페이스

모든 모델은 공통 인터페이스를 사용한다.

```python
class BaseAnalysisModel(ABC):

    @abstractmethod
    def fit(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> "BaseAnalysisModel":
        ...

    @abstractmethod
    def predict(
        self,
        X: DataFrameLike,
    ) -> np.ndarray:
        ...

    @abstractmethod
    def evaluate(
        self,
        X: DataFrameLike,
        y: SeriesLike | None = None,
    ) -> ModelEvaluation:
        ...

    @abstractmethod
    def explain(
        self,
        X: DataFrameLike,
    ) -> ExplanationResult:
        ...

    @abstractmethod
    def get_metadata(
        self,
    ) -> ModelMetadata:
        ...
```

이상탐지 모델은 추가로 다음을 제공한다.

```python
def score_samples(
    self,
    X: DataFrameLike,
) -> np.ndarray:
    ...

def classify_anomalies(
    self,
    X: DataFrameLike,
) -> list[AnomalyEvent]:
    ...
```

함수와 클래스에는 type hint, docstring, 오류 처리를 적용한다.

## 24. 합성 데이터

실제 데이터 없이도 전체 파이프라인을 검증할 수 있도록 산업별 생성기를 구현한다.

### Semiconductor

포함 변수:

- RF source power
- RF bias power
- pressure
- gas flow
- temperature
- OES-like features
- reflected power
- etch rate
- yield

이상 유형:

- source-side anomaly
- bias-side anomaly
- chamber drift
- sensor stuck
- relationship breakdown

### Battery

포함 변수:

- coating thickness
- drying temperature
- calendaring pressure
- moisture
- capacity
- internal resistance

이상 유형:

- temperature drift
- coating anomaly
- moisture spike
- capacity degradation
- batch anomaly

### Automotive

포함 변수:

- torque
- vibration
- feed rate
- rotation speed
- temperature
- dimensional error

이상 유형:

- tool wear
- bearing fault
- vibration increase
- dimensional drift
- batch anomaly

### 공통 삽입 이상

- spike
- drift
- level shift
- increased noise
- sensor stuck
- target deviation
- relationship breakdown
- batch-specific anomaly

각 데이터에는 다음 ground truth를 포함한다.

- anomaly label
- anomaly type
- anomaly start and end
- affected variables
- root-cause variable
- batch 또는 equipment ID

## 26. 테스트 원칙

다음 테스트를 작성한다.

### Unit Tests

- 파일 로더
- dtype 추론
- 산업 라우터
- schema mapper
- 데이터 품질 검사
- group split
- leakage 검사
- anomaly scoring
- recommendation constraints
- extrapolation 검사

### Integration Tests

- 업로드부터 결과 생성까지
- 산업별 합성 데이터 전체 흐름
- 사용자 schema 수정 반영
- target이 없는 경우
- timestamp가 없는 경우
- 선택 의존성이 없는 경우
- 소량 데이터 안전장치
- 누수 의심 데이터 안전장치

### Acceptance Criteria

- 애플리케이션이 sample data로 실행된다.
- 세 산업 데이터가 산업 라우터에서 구분된다.
- 사용자가 산업 및 schema를 수정할 수 있다.
- 원본 데이터가 수정되지 않는다.
- 전처리 이력이 저장된다.
- 동일 group의 train-test 혼입을 검사한다.
- 적어도 하나의 기본 모델이 선택 의존성 없이 실행된다.
- 이상 결과에 유형, 점수, severity, 기여 변수가 포함된다.
- 추천은 controllable 변수만 변경한다.
- 범위 밖 추천에는 경고 또는 차단이 적용된다.
- 테스트가 재현 가능하다.
- README만으로 설치와 실행이 가능하다.