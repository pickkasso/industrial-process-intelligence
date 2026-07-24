# Operator User Guide

Industrial Process Intelligence Platform을 혼자 실행하기 위한 짧은 운영 안내입니다.

## Start the app

프로젝트 루트에서:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

설치가 필요하면 `README.md`의 Installation을 먼저 수행합니다.

## Typical workflow

1. **Choose a data source** — Upload a UTF-8 CSV, or select **Built-in manufacturing demo** (no CSV file required). Uploaded content is not stored permanently.
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

## Built-in manufacturing demo

Use the in-memory synthetic dataset when you do not have a CSV ready:

1. Select **Built-in manufacturing demo**
2. Select a demo template (**Supervised quality prediction** or **Anomaly-only process monitoring**)
3. Click **Apply demo configuration**
4. Review the configuration
5. Click **Run analysis**

Notes:

- The data is **synthetic**. Results do **not** represent production accuracy.
- No analysis starts automatically when you select the demo source, change a template, or apply the template.
- Demo ground-truth columns (`injected_anomaly`, `anomaly_type`) are labeled as evaluation metadata and are excluded from model features by the demo templates.
- The **Supervised quality prediction** template configures four synthetic verified controllable setpoints so the public workflow can demonstrate prediction → residual diagnosis → recommendation → what-if verification:
  - `temperature_setpoint`
  - `pressure_setpoint`
  - `flow_rate_setpoint`
  - `cycle_time_setpoint`
- Approved setpoint bounds in the supervised demo are **demonstration-only**. Do **not** reuse them as real equipment limits or production decision constraints.
- The **Anomaly-only process monitoring** template remains recommendation-disabled and does not configure controllable constraints.

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

## Demo dataset

### Built-in UI demo

Prefer the Streamlit **Built-in manufacturing demo** when you want a ready-to-run synthetic dataset without writing a CSV file. Apply a demo template before running analysis.

### CLI CSV generator

Generate a reproducible manufacturing demo CSV (not committed to the repository):

```powershell
python scripts/generate_demo_dataset.py --output demo_manufacturing.csv
```

Optional knobs: `--rows`, `--seed`, `--anomaly-fraction`.

Recommended column roles for a quick demo:

| Mode | Suggested settings |
| --- | --- |
| SUPERVISED | target: `quality_score` (or `defect_rate`, not both as features); timestamp: `timestamp` |
| ANOMALY_ONLY | no target; process setpoints + measurements only; timestamp: `timestamp` |

`quality_score` and `defect_rate` are generated quality outputs. Select only one as a supervised target. Neither should be used as an anomaly-only input in the built-in acceptance test. Always exclude `injected_anomaly` and `anomaly_type` from analysis features. They remain evaluation metadata only.

### Demo validation

`scripts/validate_demo_workflow.py` runs the public analysis workflow against the default synthetic dataset and checks that:

- supervised regression selects a model with finite metrics that beat a naive mean-target baseline
- anomaly-only detections concentrate in injected anomaly rows above a documented enrichment threshold
- residual anomaly events, when exposed, show meaningful overlap with injected anomalies
- model features exclude quality outputs, ground-truth metadata, and identity/order columns (timestamp may still drive TIME splitting)

```powershell
python scripts/validate_demo_workflow.py
```

A successful validation only proves the synthetic demo path is useful and non-trivial. It does **not** represent real-world production accuracy.

## Safety reminders

- Outputs are model-based decision support, not operational commands.
- Associations are not proven causes.
- Domain, process-safety, and experimental verification are required before applying any proposed change.
