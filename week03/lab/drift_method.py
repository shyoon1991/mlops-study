"""같은 윈도에 검정 방법만 바꿔 본다: Jensen-Shannon(기본, 거리 0.1) vs 카이제곱(p<0.05)."""
import os
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from evidently import Report
from evidently.metrics import DriftedColumnsCount, ValueDrift
COLS = ["Education", "Marital-Status", "Native_country", "Occupation", "Race", "Relationship", "Sex", "Workclass"]
ref = pd.read_csv(os.path.expanduser("~") + "/mlops-ch4/chapter-4/data/income_data.csv")[COLS]
shifted = ref[ref["Education"].str.strip().isin(["Bachelors", "Masters", "Doctorate", "Prof-school"])]
rows = []
for method in ["jensenshannon", "chisquare"]:
    for n in (50, 100, 500):
        for label, pool in (("same", ref), ("shifted", shifted)):
            share = []
            for seed in range(20):   # 윈도 20개 평균 — 우연을 줄인다
                s = Report([DriftedColumnsCount(cat_method=method)]).run(
                    current_data=pool.sample(n, random_state=seed), reference_data=ref)
                share.append(s.dict()["metrics"][0]["value"]["share"])
            rows.append({"method": method, "n": n, "window": label,
                         "mean_share": round(sum(share) / len(share), 3),
                         "dataset_drift_rate": round(sum(x >= 0.5 for x in share) / len(share), 2)})
print(pd.DataFrame(rows).pivot_table(index=["method", "n"], columns="window", values=["mean_share", "dataset_drift_rate"]).to_string())
pd.DataFrame(rows).to_csv(os.path.expanduser("~") + "/mlops-ch6/drift_methods.csv", index=False)
