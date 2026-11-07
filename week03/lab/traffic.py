"""목업 요청: 2주차 Feast 엔티티(user_id)로 /predict 호출 (책 scripts/generate_mock_requests.py와 같은 역할)."""
import os
import random, sys, time
import pandas as pd, requests
ids = pd.read_parquet(os.path.expanduser("~") + "/mlops-ch4/chapter-4/feast/feature_data/demographic_features.parquet")["user_id"].tolist()
n, sleep = int(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
cnt = {}
for i in range(n):
    uid = random.choice(ids) if random.random() > 0.05 else "no-such-user"
    r = requests.post("http://localhost:3000/predict", json={"user_id": uid}, timeout=30)
    k = r.json().get("income_category") if r.ok else f"HTTP {r.status_code}"
    cnt[k] = cnt.get(k, 0) + 1
    time.sleep(sleep)
print(cnt)
