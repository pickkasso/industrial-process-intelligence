# Industrial Process Intelligence Platform
# Development Roadmap

이 문서는 필수 통합 시나리오, 단계별 개발 범위 및 첫 번째 작업을 정의한다.

아래 내용은 원본 구축 지시서에서 이동한 것이며,
문장과 요구사항을 임의로 축약하거나 변경하지 않는다.

## 25. 필수 통합 시나리오

반도체, 배터리, 자동차 데이터 각각에 대해 다음 흐름을 검증한다.

1. 파일 업로드
2. 산업 추정
3. 산업 사용자 확정
4. 열 역할 추정
5. 사용자 역할 수정
6. 데이터 품질 진단
7. tabular 또는 time-series 판별
8. 분석 경로 선택
9. 모델 실행
10. 이상 샘플 탐지
11. 이상 유형 분류
12. 주요 원인 변수 진단
13. controllable variable 및 범위 입력
14. 최소 변경안 계산
15. 변경 전후 예상 결과 비교
16. 보고서 저장

## 27. 개발 단계

### Phase 1: MVP

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

### Phase 2

- 시계열 feature
- PCA T² 및 SPE
- change point
- residual anomaly
- data drift
- prediction interval
- counterfactual optimization
- 향상된 보고서

### Phase 3

- 산업별 상세 ontology
- 유사 사례 검색
- FastAPI 분리
- 대용량 처리 최적화
- 추가 산업 plugin
- 모델 모니터링

이미지 기반 결함 분류는 MVP 범위에서 제외한다.

향후 ImageIndustryAdapter 를 추가할 수 있도록 인터페이스 확장 가능성만 고려한다.

## 30. 첫 번째 작업

아직 코드를 생성하지 말고 다음 내용을 제시하라.

1. 요구사항을 도메인, 데이터, 모델, 추천, UI, 운영 관점으로 재분류한 요약
2. 권장 시스템 아키텍처
3. 주요 데이터 객체와 Pydantic schema
4. 모듈별 책임
5. 산업 Plugin 등록 방식
6. 모델 Registry 구조
7. 데이터 누수 방지 전략
8. MVP 구현 순서
9. 테스트 전략
10. 가장 위험한 기술적 가정 10개
11. Phase 1에서 의도적으로 제외할 기능
12. 첫 구현 단계에서 생성할 파일 목록

코드 구현은 이 설계 결과를 기준으로 다음 단계부터 시작한다.