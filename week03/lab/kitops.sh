#!/bin/bash
# 도전과제: Yatai 레지스트리 대체 — KitOps ModelKit(모델+코드+데이터 메타를 OCI 아티팩트로) → Harbor
set -e
VER=v1.16.0
mkdir -p ~/.local/bin && cd /tmp
if ! ~/.local/bin/kit version >/dev/null 2>&1; then
  curl -fsSL -o kit.tgz https://github.com/kitops-ml/kitops/releases/download/$VER/kitops-linux-x86_64.tar.gz
  curl -fsSL -o kit.sha https://github.com/kitops-ml/kitops/releases/download/$VER/checksums.txt || true
  grep -q "kitops-linux-x86_64.tar.gz" kit.sha 2>/dev/null && (grep "kitops-linux-x86_64.tar.gz" kit.sha | sed 's#kitops-linux-x86_64.tar.gz#kit.tgz#' | sha256sum -c -)
  tar xzf kit.tgz kit && mv kit ~/.local/bin/kit
fi
~/.local/bin/kit version | head -2
# ModelKit 디렉터리: BentoML 모델 스토어에 있는 MLflow 모델 + 서비스 코드
W=~/mlops-ch6/modelkit && rm -rf $W && mkdir -p $W/model $W/code
M=$(ls -d ~/bentoml/models/random-forest-classifier/*/ | head -1)
cp -r $M/mlflow_model/* $W/model/
cp ~/mlops-ch6/chapter-6/bentoml/service_v14.py ~/mlops-ch6/chapter-6/bentoml/requirements_v14.txt $W/code/
cp -r ~/mlops-ch6/chapter-6/bentoml/src $W/code/src
cat > $W/Kitfile <<'K'
manifestVersion: "1.0"
package:
  name: income-classifier
  version: 2.0.0
  description: Census income binary classifier (RandomForest, MLflow registry v2 @production)
  authors: ["shyoon1991"]
model:
  name: random-forest-classifier
  path: model
  framework: scikit-learn
  version: "2"
  description: MLflow pyfunc model (skops), ROC AUC test 0.8578
code:
  - path: code
    description: BentoML 1.4 service + feature store client
K
cd $W && ~/.local/bin/kit pack . -t localhost:8085/mlops/income-classifier-modelkit:v2 2>&1 | tail -2
~/.local/bin/kit list | head -4
PW=$(cat ~/.config/harbor-admin-password)
echo "$PW" | ~/.local/bin/kit login localhost:8085 -u admin --password-stdin --plain-http 2>&1 | tail -1
~/.local/bin/kit push localhost:8085/mlops/income-classifier-modelkit:v2 --plain-http 2>&1 | tail -2
echo "== 다른 곳에서 모델만 받기 (kit unpack --model)"
rm -rf /tmp/unpack && ~/.local/bin/kit unpack localhost:8085/mlops/income-classifier-modelkit:v2 --model -d /tmp/unpack --plain-http 2>&1 | tail -1; find /tmp/unpack -maxdepth 2 | head
