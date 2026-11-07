import os
import warnings; warnings.filterwarnings("ignore")
import pickle, pandas as pd
cols = pickle.load(open(os.path.expanduser("~") + "/mlops-ch4/chapter-4/mlflow/column_list.pkl", "rb"))
df = pd.read_csv(os.path.expanduser("~") + "/mlops-ch4/chapter-4/data/income_data.csv")
row = df.drop(columns="Target").iloc[[7]]          # 실제 >50K인 행 하나
print("원본 행:", row.iloc[0].str.strip().to_dict(), "| 정답:", df.Target.iloc[7].strip())
a = pd.get_dummies(row, drop_first=True, dtype=float)           # 책 InputMapper (요청 1건)
print("drop_first=True  → 더미 컬럼 수:", a.shape[1], "| column_list와 겹치는 1의 개수:", int(a.reindex(columns=cols, fill_value=0).sum(axis=1).iloc[0]))
b = pd.get_dummies(row, drop_first=False, dtype=float)
print("drop_first=False → 더미 컬럼 수:", b.shape[1], "| column_list와 겹치는 1의 개수:", int(b.reindex(columns=cols, fill_value=0).sum(axis=1).iloc[0]))
import bentoml
m = bentoml.mlflow.load_model("random-forest-classifier:latest")
print("예측(책 방식):", m.predict(a.reindex(columns=cols, fill_value=0))[0], "| 예측(수정):", m.predict(b.reindex(columns=cols, fill_value=0))[0])
