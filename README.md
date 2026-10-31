# 머신러닝 플랫폼 엔지니어링 스터디

*Machine Learning Platform Engineering* (Manning) 스터디 기록과 실습 코드.

| 주차 | 범위 | 글 |
|---|---|---|
| 1주차 | 1~3장: MLOps 생명주기, 컨테이너·K8s·CI/CD | [집에서 LLM 하나 바꾸는 데도 MLOps 생명주기가 다 들어 있었다](week01/README.md) — 도전 과제: Harbor에 hello-joker 올리기 |
| 2주차 | 4~5장: MLflow, Feast, Kubeflow Pipelines | [MLflow·Feast 최신 버전에 SeaweedFS·Valkey로 4장 실습하기](week02/README.md) — 도전 과제: 최신 버전, MinIO→SeaweedFS·Redis→Valkey, 동작 흐름, EDA fill vs stack |

## 실습 환경

- Windows 11 + WSL2 Ubuntu 26.04, RTX 5090
- Docker Engine 29.9 (WSL에 직접 설치, Docker Desktop 없이), NVIDIA Container Toolkit
- Harbor v2.15.4 (`localhost:8085`, HTTP)
- 2주차: SeaweedFS 4.51, Valkey 9.1.3, MLflow 3.17.2, Feast 0.67.0 (Docker Compose)

책 예제 코드는 [practical-mlops](https://github.com/practical-mlops)에 있습니다. 이 저장소에는 책 코드를 복사하지 않고, 바꾼 부분과 직접 쓴 스크립트만 둡니다.
