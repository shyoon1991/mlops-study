"""6.2 데이터 드리프트 — 책 evidently_reports.py(0.5)를 Evidently 0.7 API로 옮겨 실행.

reference: MLflow Production 모델 run의 reference 데이터셋(SeaweedFS mlflow-datasets 버킷의 feature_df CSV) — 책 서비스와 같은 출처
current  : 같은 모집단에서 뽑은 윈도(크기 50/100/500/2000) + 일부러 분포를 옮긴 윈도
결과는 로컬 Evidently 워크스페이스에 저장해 `evidently ui`로 본다.
"""
import io
import json
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD"] = "false"

import boto3
import mlflow
import pandas as pd
from evidently import Report
from evidently.metrics import DatasetMissingValueCount, DriftedColumnsCount, ValueDrift
from evidently.ui.workspace import Workspace

COLS = ["Education", "Marital-Status", "Native_country", "Occupation", "Race", "Relationship", "Sex", "Workclass"]

# 책 data_drift_report와 같은 구성 (ColumnSummaryMetric은 0.7에 대응 지표가 없어 뺐다)
def make_report() -> Report:
    return Report([DriftedColumnsCount(), DatasetMissingValueCount(), *[ValueDrift(column=c) for c in COLS]])

# reference: Production 모델 run에 log_input(context="reference_features")로 남긴 데이터셋
mlflow.set_tracking_uri("http://localhost:5000")
client = mlflow.MlflowClient()
prod = client.get_model_version_by_alias("random-forest-classifier", "production")
run = client.get_run(prod.run_id)
src = next(json.loads(d.dataset.source)["uri"] for d in run.inputs.dataset_inputs
           if any(t.value == "reference_features" for t in d.tags))
bucket, key = src.split("/", 1)
s3 = boto3.client("s3", endpoint_url="http://localhost:8333", aws_access_key_id="seaweed", aws_secret_access_key="seaweed123")
reference = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key=key)["Body"].read()))[COLS]
print(f"reference: s3://{bucket}/{key}  rows={len(reference)}")

population = pd.read_csv(os.path.expanduser("~") + "/mlops-ch4/chapter-4/data/income_data.csv")[COLS]
windows = {f"same-dist n={n}": population.sample(n, random_state=n) for n in (50, 100, 500, 2000)}
shifted = population[population["Education"].str.strip().isin(["Bachelors", "Masters", "Doctorate", "Prof-school"])]
windows["shifted(대졸 이상만) n=100"] = shifted.sample(100, random_state=1)

ws = Workspace.create(os.path.expanduser("~") + "/mlops-ch6/evidently-ws")
project = next((p for p in ws.list_projects() if p.name == "income-classifier"), None) or ws.create_project("income-classifier")
project.description = "Week3: Evidently 0.7 drift windows"
project.save()

rows = []
for name, cur in windows.items():
    snap = make_report().run(current_data=cur, reference_data=reference, name=name)
    ws.add_run(project.id, snap, include_data=False)
    res = snap.dict()["metrics"]
    drifted = next(m for m in res if m["metric_name"].startswith("DriftedColumnsCount"))["value"]
    per_col = {m["metric_name"].split("column=")[1].split(",")[0].rstrip(")"): round(m["value"], 3)
               for m in res if m["metric_name"].startswith("ValueDrift")}
    method = next(m["metric_name"] for m in res if m["metric_name"].startswith("ValueDrift"))
    rows.append({"window": name, "drifted": f"{int(drifted['count'])}/8", "share": round(drifted["share"], 3), **per_col})
    if name.endswith("2000") or name.startswith("shifted"):
        snap.save_html(os.path.expanduser("~") + f"/mlops-ch6/report-{name.split()[0].replace('(', '_').replace(')', '')}.html")

print("method example:", method)
out = pd.DataFrame(rows).set_index("window")
print(out.to_string())
out.to_csv(os.path.expanduser("~") + "/mlops-ch6/drift_windows.csv")
print("project id:", project.id)
