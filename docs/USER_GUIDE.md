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
6. **Inspect the report** — Overview, **Run provenance**, stage execution, anomaly/diagnosis, recommendation, warnings/disclaimers를 확인합니다.
7. **Export configuration** — 같은 설정으로 다시 돌릴 때 Download configuration JSON을 사용합니다.
8. **Export reproducibility bundle** — completed run의 declared-input provenance를 ZIP으로 받으려면 **Run provenance** 안의 **Download reproducibility bundle**을 사용합니다.
9. **Verify reproducibility bundle** — 이전에 보낸 ZIP을 **Verify reproducibility bundle** 섹션에 업로드하여 서명·레이아웃을 읽기 전용으로 검사합니다 (설정 복원/재실행 없음).

## Run provenance

After an analysis finishes, the report includes a collapsed **Run provenance** expander near the overview. It summarizes the immutable reproducibility manifest stored with that report, and includes a **Current setup comparison** against the currently loaded dataset and effective analysis configuration.

Distinguish these three identifiers:

| Identifier | Meaning |
| --- | --- |
| **Dataset fingerprint** | SHA-256 of the raw CSV bytes used for the run |
| **Configuration fingerprint** | SHA-256 of the effective workflow request/policy (features, target, constraints, cohort, seeds/policies that affect results) |
| **Run signature** | SHA-256 of dataset fingerprint + configuration fingerprint + application version + manifest schema version |

Signatures help compare whether two runs used the same data, effective configuration, software version, and schema. They do **not** prove model validity, causal correctness, or operational safety.

### Current setup comparison

| Overall status | Meaning |
| --- | --- |
| **EXACT_MATCH** | The declared dataset, effective configuration, application version, and manifest schema match this stored run |
| **CONFIGURATION_CHANGED** | Widgets that affect analysis differ from the run that produced this result — click **Run analysis** again before treating the result as current |
| **DATASET_CHANGED** | The active dataset content differs from the dataset used for this result |
| **CURRENT_CONFIGURATION_INCOMPLETE** | The current form cannot be fingerprinted until readiness issues (missing target, invalid rules, incomplete constraints, etc.) are resolved |
| **UNAVAILABLE** | Comparison cannot run (missing stored manifest or active dataset fingerprint) |

Notes:

- Provenance is taken from the completed report. Changing widgets without clicking **Run analysis** does not rewrite the stored manifest.
- Behavior-affecting configuration changes require rerunning analysis before the displayed result reflects the current setup.
- Presentation-only changes (expanders, table display, selected context rows, demo ground-truth evaluation) do **not** require rerunning and do not change the configuration match.
- Matching signatures identify matching declared inputs and software metadata; they do **not** guarantee model correctness, causality, or bit-for-bit deterministic execution.
- Incomplete configurations cannot be compared until readiness issues are resolved; the stored result and manifest remain unchanged.
- Switching data source clears stale reports, so the expander disappears until a new run completes.
- Demo ground-truth evaluation is presentation-only and does not change fingerprints, the run signature, or the comparison outcome.

### Reproducibility bundle export

After a completed run with a stored run manifest, **Run provenance** includes **Download reproducibility bundle**. The ZIP contains:

| File | Contents |
| --- | --- |
| `bundle_manifest.json` | Bundle schema version, bundle signature, included files, run/dataset/configuration digests; `generated_at` is present but **not** part of the signature |
| `run_manifest.json` | Exact serialized stored immutable run manifest |
| `effective_configuration.json` | Canonical effective workflow configuration used for that completed run |
| `feature_schema.json` | Target, final modeling features (pipeline order), roles/dtypes/excluded columns when available — no row values |
| `environment.json` | Python/platform summary, application version, manifest/bundle schema versions, relevant package versions |
| `README.txt` | How to interpret provenance vs model replay |

Intentionally excluded:

- Raw dataset rows / CSV bytes (privacy)
- Trained models and metric/duration values beyond what the immutable run manifest already stores
- Streamlit widget state, temporary paths, and presentation-only UI metadata

The **bundle signature** is a SHA-256 digest over the canonical contents of the signed files above. Identical provenance inputs produce the same signature and file contents. ZIP container metadata (entry timestamps) may differ and is not part of the signature.

This export proves **declared-input provenance** for offline inspection. It does **not** prove causal correctness or bit-for-bit deterministic model replay, and it does not restore configuration or re-run analysis. Use **Run provenance** / the Step 14B current-setup comparison to see whether the live UI still matches the stored run; use the bundle to take that stored provenance out of the session.

### Verify reproducibility bundle

The page includes a separate **Verify reproducibility bundle** section (not inside **Run provenance**). Upload one exported ZIP to check whether:

| Check | Meaning |
| --- | --- |
| Archive layout | Required Step 14C files are present under `reproducibility_bundle/` |
| Signature | Recomputed digest of signed file bytes matches `bundle_manifest.bundle_signature` |
| Manifest cross-check | Unsigned `bundle_manifest.json` metadata matches signed provenance (run signature, fingerprints, application version, included files) |
| Schema compatibility | Bundle schema and run-manifest schema versions are supported (currently version `1`) |
| Safety | Path traversal, symlinks, encrypted entries, duplicate names, oversized / ZIP-bomb-like archives, unexpected data/executable members are rejected |

Supported schema versions:

- `BUNDLE_SCHEMA_VERSION = 1`
- `MANIFEST_SCHEMA_VERSION = 1`

Future schema versions are rejected (not silently accepted).

What signature verification **proves**:

- The signed files in the ZIP match the digest recorded in `bundle_manifest.json`
- For a valid result, inspected provenance fields were derived from those signed files

What it does **not** prove:

- That replaying analysis would reproduce the same metrics or model outputs
- Causal correctness of diagnosis or recommendations
- That text fields contain no sensitive data (absence of raw CSV/Parquet members reduces risk but cannot mathematically prove it)

External bundle inspection vs active run provenance:

- **Run provenance** describes the completed run stored in the current session
- **Verify reproducibility bundle** inspects an uploaded ZIP as external, read-only provenance
- Uploading a bundle does **not** create an active workflow report, change the dataset source, update live configuration widgets, restore settings, or trigger analysis
- There is no restore or replay control in this step

Security restrictions for uploaded ZIPs (untrusted input):

- In-memory reading only (no extraction into the project directory)
- Rejects `../`, absolute paths, Windows drive paths, backslash traversal, symlinks, encrypted entries, duplicates, and non-regular members
- Enforces compressed / uncompressed size limits, member-count limits, and suspicious compression-ratio checks
- Accepts only the expected text/JSON provenance files; common raw-data and executable suffixes are rejected
- JSON and README are decoded as strict UTF-8 (invalid UTF-8 is rejected)

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
- After a successful built-in demo analysis, a **Demo ground-truth evaluation** panel can appear under the anomaly-event report. It is evaluation-only and does not change modeling, diagnosis, recommendation, or run readiness.
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

## Demo ground-truth evaluation

When the active source is **Built-in manufacturing demo** and a non-stale analysis report exists for that demo dataset, the Streamlit report shows a compact **Demo ground-truth evaluation** section after the detected anomaly events.

What it means:

- **Precision among displayed event representatives** — of the anomaly-event rows the workflow currently displays (often only a small top-N set), the share whose original row IDs land on injected anomaly rows.
- **Enrichment over demo prevalence** — that displayed-event precision divided by the full-demo injected-anomaly prevalence. Values above `1×` mean the displayed events are more concentrated in injected anomalies than a random row would be.

What it is not:

- Not full detector recall (many injected anomalies may never appear in the displayed top events).
- Not full anomaly-model precision across every scored row.
- Not production accuracy. Synthetic labels were not used for training or scoring.

Unmatched displayed events are **not** automatically proven false positives. The synthetic labels may omit unusual but real process states. Uploaded CSV files never show this panel.

## Safety reminders

- Outputs are model-based decision support, not operational commands.
- Associations are not proven causes.
- Domain, process-safety, and experimental verification are required before applying any proposed change.
