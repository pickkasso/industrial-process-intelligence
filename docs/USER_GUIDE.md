# Operator User Guide

Industrial Process Intelligence Platform을 혼자 실행하기 위한 짧은 운영 안내입니다.

## Start the app

프로젝트 루트에서:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

설치가 필요하면 `README.md`의 Installation을 먼저 수행합니다.

## Typical workflow

1. **Upload a CSV file** — UTF-8 CSV만 지원합니다. 업로드 내용은 세션에 영구 저장되지 않습니다.
2. **Select analysis mode**
   - `SUPERVISED`: target이 있고 품질·잔차·추천까지 필요할 때
   - `ANOMALY_ONLY`: label/target 없이 이상 탐지와 연관 변수 진단만 필요할 때
3. **Review suggested columns** — timestamp, identifier, feature, target 후보를 확인하고 필요하면 수정합니다. 자동 제안은 최종 확정이 아닙니다.
4. **Complete safety-critical settings**
   - SUPERVISED: target, analysis task, objective, 성능 수용 규칙
   - 추천을 쓸 경우: controllable / verified / constraint bounds
   - ANOMALY_ONLY: 선택적 cohort filter, 선택적 anomaly recommendation
5. **Check Current configuration summary and Run readiness** — False 항목을 해결한 뒤 Run analysis를 누릅니다.
6. **Inspect the report** — Overview, stage execution, anomaly/diagnosis, recommendation, warnings/disclaimers를 확인합니다.
7. **Export configuration** — 같은 설정으로 다시 돌릴 때 Download configuration JSON을 사용합니다.

## Mode differences

### SUPERVISED

- Target이 필요합니다.
- Regression 또는 현재 workflow가 지원하는 task만 실행됩니다.
- 독립 테스트 성능 규칙(metric / direction / finite threshold)이 필요합니다.
- Target이 constant이거나 전부 null이면 실행할 수 없습니다. anomaly-only로 전환하거나 다른 target를 선택하세요.
- Recommendation은 사용자가 확인한 controllable·verified·constrained 변수만 사용합니다.

### ANOMALY_ONLY

- Target이 필요하지 않습니다.
- 비지도 이상 탐지와 관련 변수 진단을 수행합니다.
- 통계적 이상(outlier)은 곧바로 결함이 아닙니다.
- Operating cohort filter는 선택 사항입니다.
- Anomaly recommendation은 기본 비활성화입니다.
- Recommendation을 켜면 실제로 조절 가능한 변수만 허용됩니다.

## Common fixes

| Situation | What to do |
| --- | --- |
| Target has no usable variation | Choose another target or switch to ANOMALY_ONLY |
| Unsupported analysis task | Select AUTO or REGRESSION |
| Incomplete performance rule | Complete metric name, direction, and finite threshold |
| Incomplete process-variable constraint | Complete or clear the selected constraint |
| Cohort too small after filter | Widen bounds or disable the cohort filter |
| Recommendation refused after anomaly success | Anomaly/diagnosis results remain usable; fix controllable/verified/constraint inputs if you need a recommendation |
| Configuration export blocked | Complete or clear incomplete rows; enabled-but-incomplete cohort filters are not silently dropped |

## Safety reminders

- Outputs are model-based decision support, not operational commands.
- Associations are not proven causes.
- Domain, process-safety, and experimental verification are required before applying any proposed change.
