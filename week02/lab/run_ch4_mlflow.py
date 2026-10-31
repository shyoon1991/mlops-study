"""4.1 실습: 책 노트북을 그대로 실행하되 접속 정보만 SeaweedFS로 바꾼다.

바꾸는 것: AWS_ENDPOINT_URL(9000→8333), 키(minio/minio123 → seaweed/seaweed123), Minio() 접속 대상.
코드 로직은 손대지 않는다 — 최신 MLflow에서 깨지는 곳이 있으면 그대로 드러나게.
"""
import sys
import boto3
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NB_IN, NB_OUT = sys.argv[1], sys.argv[2]

s3 = boto3.client("s3", endpoint_url="http://localhost:8333",
                  aws_access_key_id="seaweed", aws_secret_access_key="seaweed123")
have = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
for b in ["mlflow-artifacts", "mlflow-datasets", "feature-data-sets", "feature-registry"]:
    if b not in have:
        s3.create_bucket(Bucket=b)
print("buckets:", sorted(b["Name"] for b in s3.list_buckets()["Buckets"]))

nb = nbformat.read(NB_IN, as_version=4)
REPL = [
    ('"minio"', '"seaweed"'), ('"minio123"', '"seaweed123"'),
    ("http://localhost:9000", "http://localhost:8333"),
    ('Minio("localhost:9000"', 'Minio("localhost:8333"'),
    ('access_key="minio"', 'access_key="seaweed"'), ('secret_key="minio123"', 'secret_key="seaweed123"'),
]
if len(sys.argv) > 3 and sys.argv[3] == "--fix":
    # MLflow 3.17: sklearn 모델 저장이 skops 형식으로 바뀌어 신뢰 타입을 명시해야 한다. artifact_path → name
    TRUSTED = '["sklearn.tree._tree.Tree", "sklearn.utils._bunch.Bunch"]'
    REPL += [
        ('mlflow.sklearn.log_model(tree,"income-classifier")',
         f'mlflow.sklearn.log_model(tree, name="income-classifier", skops_trusted_types={TRUSTED})'),
        ('mlflow.sklearn.log_model(forest,"income-classifier")',
         f'mlflow.sklearn.log_model(forest, name="income-classifier", skops_trusted_types={TRUSTED})'),
    ]
changed = 0
for c in nb.cells:
    if c.cell_type != "code":
        continue
    src = c.source
    for a, b in REPL:
        src = src.replace(a, b)
    if src != c.source:
        changed += 1
        c.source = src
    # 그래프 창을 띄우지 않는다
    c.source = c.source.replace("print(plot)", "pass  # print(plot)")
print("patched cells:", changed)

ep = ExecutePreprocessor(timeout=1800, kernel_name="python3", allow_errors=True)
ep.preprocess(nb, {"metadata": {"path": "mlflow"}})
nbformat.write(nb, NB_OUT)
for i, c in enumerate(nb.cells):
    for o in c.get("outputs", []):
        if o.get("output_type") == "error":
            print(f"[cell {i}] {o['ename']}: {o['evalue'][:300]}")
        elif o.get("output_type") == "stream" and i >= 17:
            print(f"[cell {i}]", o["text"][-400:].strip())
