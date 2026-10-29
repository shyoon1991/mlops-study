#!/bin/bash
# 책 chapter-3의 hello-joker를 빌드해 로컬 Harbor(localhost:8085/mlops)에 push한다.
# 사용: ./build-push.sh <chapter-3 클론 경로>
set -euo pipefail
SRC=${1:?chapter-3 경로}
HERE=$(cd "$(dirname "$0")" && pwd)
REG=localhost:8085; PRJ=mlops
PW=$(cat ~/.config/harbor-admin-password)
SHA=$(git -C "$SRC" rev-parse --short HEAD)

# 비공개 프로젝트 생성 (이미 있으면 409)
curl -s -o /dev/null -w "create project: %{http_code}\n" -u "admin:$PW" -H 'Content-Type: application/json' \
  -X POST http://$REG/api/v2.0/projects -d "{\"project_name\":\"$PRJ\",\"metadata\":{\"public\":\"false\"}}"
echo "$PW" | docker login $REG -u admin --password-stdin

docker build -f "$HERE/Dockerfile.bookworm" -t hello-joker:v1 "$SRC"
for t in v1 "$SHA"; do
  docker tag hello-joker:v1 $REG/$PRJ/hello-joker:$t
  docker push $REG/$PRJ/hello-joker:$t
done
