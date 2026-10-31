import warnings; warnings.filterwarnings("ignore")
import boto3, mlflow, mlflow.sklearn, pandas as pd, pickle
mlflow.set_tracking_uri("http://localhost:5000")
c = mlflow.MlflowClient()
for v in c.search_model_versions("name='random-forest-classifier'", order_by=["version_number ASC"]):
    run = c.get_run(v.run_id)
    try:
        m = mlflow.sklearn.load_model(f"models:/random-forest-classifier/{v.version}"); ok = type(m).__name__
    except Exception as e:
        ok = "LOAD FAIL (" + str(e)[:40] + ")"
    print(f"v{v.version} run={run.info.run_name} roc_auc_test={run.data.metrics.get('roc_auc_score_test'):.4f} src={v.source} -> {ok}")
# MLflow 3: stage는 deprecated → alias 사용. 5장 파이프라인은 model_stage=Production을 쓰니 둘 다 설정해 본다
c.set_registered_model_alias("random-forest-classifier", "production", "2")
try:
    c.transition_model_version_stage("random-forest-classifier", "2", "Production"); print("stage Production: OK (deprecated API)")
except Exception as e:
    print("stage transition:", type(e).__name__, str(e)[:120])
m = mlflow.sklearn.load_model("models:/random-forest-classifier@production")
df = pd.read_csv("chapter-4/data/income_data.csv").head(5)
cols = pickle.load(open("chapter-4/mlflow/column_list.pkl", "rb"))
X = pd.get_dummies(df.drop("Target", axis=1), drop_first=True, dtype=float).reindex(columns=cols, fill_value=0)
print("predict @production:", list(m.predict_proba(X)[:, 1].round(2)), "truth:", list(df.Target))
try:
    print("load by stage:", type(mlflow.sklearn.load_model("models:/random-forest-classifier/Production")).__name__)
except Exception as e:
    print("load by stage FAIL:", str(e)[:120])
s3 = boto3.client("s3", endpoint_url="http://localhost:8333", aws_access_key_id="seaweed", aws_secret_access_key="seaweed123")
for b in ["mlflow-artifacts", "mlflow-datasets"]:
    objs = s3.list_objects_v2(Bucket=b).get("Contents", []); keys = [o["Key"] for o in objs]
    print(f"s3://{b}: {len(keys)} objects, e.g.", [k for k in keys if k.endswith(("MLmodel", ".csv"))][:3])
