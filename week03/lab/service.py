"""책 6장 service.py를 BentoML 1.4 새 API로 옮긴 것.

바뀐 점
- bentoml.Service(runners=[...]) + @svc.api  →  @bentoml.service 클래스 + @bentoml.api 메서드
- Runner(to_runner) → 같은 프로세스에서 모델을 직접 불러 호출 (BentoModel 클래스 속성으로 Bento에 포함)
- @svc.on_startup + ctx.state  →  __init__ 과 인스턴스 속성
- bentoml.io.JSON(pydantic_model=...)  →  메서드 인자 타입 힌트 (요청 본문 {"user_id": "..."} 는 그대로)
"""
import os
import pickle

import bentoml
import pandas as pd
from bentoml.exceptions import NotFound
from dotenv import dotenv_values

from src.constants import ENV_FILE_NAME, FEATURE_LIST
from src.data_mapper import OutputMapper


def to_model_input(feature_df: pd.DataFrame, col_list: list) -> pd.DataFrame:
    """책 InputMapper는 get_dummies(drop_first=True)라 요청 1건(1행)이면 모든 더미가 사라져 입력이 전부 0이 된다.
    서빙에서는 drop_first=False로 만든 뒤 학습 컬럼 목록으로 맞춘다(학습 때 버린 기준 범주는 reindex에서 빠진다)."""
    dummies = pd.get_dummies(feature_df, drop_first=False, sparse=False, dtype=float)
    return dummies.reindex(columns=col_list, fill_value=0)


@bentoml.service(name="income_classifier_service", traffic={"timeout": 30})
class IncomeClassifier:
    # Bento를 빌드할 때 이 모델이 함께 들어간다 (책의 bentoml.mlflow.get(...).to_runner() 자리)
    model_ref = bentoml.models.BentoModel("random-forest-classifier:latest")

    def __init__(self) -> None:
        import mlflow
        from mlflow.tracking import MlflowClient
        from src.feature_store import DataStore

        config = dotenv_values(ENV_FILE_NAME)
        os.environ["FEAST_S3_ENDPOINT_URL"] = config["FEAST_S3_ENDPOINT_URL"]
        os.environ["AWS_ENDPOINT_URL"] = config["FEAST_S3_ENDPOINT_URL"]

        self.model = bentoml.mlflow.load_model(self.model_ref)

        # 학습 때 쓴 원-핫 컬럼 목록: Production 단계 모델의 run에서 받는다 (책과 같음)
        mlflow_client = MlflowClient(config["MLFLOW_HOST"])
        mlflow.set_tracking_uri(config["MLFLOW_HOST"])
        model_run_id = next(
            m.run_id
            for m in mlflow_client.search_model_versions(f"name='{config['MLFLOW_MODEL_NAME']}'")
            if m.current_stage == config["MLFLOW_MODEL_STAGE"]
        )
        mlflow.artifacts.download_artifacts(f"runs:/{model_run_id}/column_list.pkl", dst_path="column_list")
        with open("column_list/column_list.pkl", "rb") as f:
            self.col_list = pickle.load(f)

        self.store = DataStore(
            config["MINIO_HOST"],
            config["MINIO_ACCESS_KEY"],
            config["MINIO_SECRET_KEY"],
            config["FEATURE_REGISTRY_BUCKET_NAME"],
            config["FEATURE_REGSITRY_FILE_NAME"],
            config["FEAST_S3_ENDPOINT_URL"],
            config["FEAST_REDIS_HOST"],
            config["FEAST_REDIS_PASSWORD"],
        ).init_feature_store()

    @bentoml.api(route="/predict")
    def predict(self, user_id: str) -> dict:
        feature_df = self.store.get_online_features(
            features=FEATURE_LIST, entity_rows=[{"user_id": user_id}]
        ).to_df()
        feature_df.drop(columns=["user_id"], inplace=True)
        # 온라인 스토어에 없는 사용자는 피처가 전부 None → 책 코드는 그대로 예측해 200을 돌려준다
        if feature_df.isna().all(axis=None):
            raise NotFound(f"no online features for user_id={user_id}")
        input_df = to_model_input(feature_df, self.col_list)
        prediction = OutputMapper(int(self.model.predict(input_df)[0]))
        return {"income_category": prediction.map_prediction(), "user_id": user_id}
