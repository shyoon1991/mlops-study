#!/bin/bash
# 4.2 Feast 0.67.0 + SeaweedFS(오프라인 parquet·레지스트리) + Valkey(온라인)
set -uo pipefail
cd ~/mlops-ch4/chapter-4
export PATH=$HOME/mlops-ch4/.venv/bin:$PATH
export AWS_ACCESS_KEY_ID=seaweed AWS_SECRET_ACCESS_KEY=seaweed123
export FEAST_S3_ENDPOINT_URL=http://localhost:8333 AWS_ENDPOINT_URL=http://localhost:8333
# 접속 정보만 교체
sed -i 's#http://localhost:9000#http://localhost:8333#' feast/features.py
cat > feast/feature_store.yaml <<'YAML'
project: my_feature_repo
registry: s3://feature-registry/registry_local.db
provider: local
offline_store:
  type: file
online_store:
  type: redis
  connection_string: "localhost:6379,password=valkeypass"
YAML
# 오프라인 데이터를 SeaweedFS 버킷에 올린다 (책: MinIO feature-data-sets)
python - <<'PY'
import boto3, glob, os
s3 = boto3.client("s3")
for f in glob.glob("feast/feature_data/*.parquet"):
    s3.upload_file(f, "feature-data-sets", os.path.basename(f))
print("feature-data-sets:", [o["Key"] for o in s3.list_objects_v2(Bucket="feature-data-sets")["Contents"]])
PY
cd feast
echo "== feast apply"; feast apply 2>&1 | grep -v -i warn | tail -12
echo "== feast materialize"; feast materialize 2022-09-16T00:00:00 2023-09-17T00:00:00 2>&1 | grep -v -i warn | tail -8
echo "== historical"; python ../feast_retrieval/retrieve_features.py 2>&1 | grep -v -i warn | tail -8
echo "== online"; python ../feast_retrieval/retrieve_online_features.py 2>&1 | grep -v -i warn | tail -8
