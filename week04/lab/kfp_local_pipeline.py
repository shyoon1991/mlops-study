"""도전과제: KFP v2 아티팩트 타입을 고루 쓰는 파이프라인 — Kubernetes 없이 kfp.local로 실행.

prepare  → Output[Dataset]  YOLO 데이터셋(영상 단위 분할)   + Output[Markdown] 분할 요약 + Output[Metrics] 장수
train    → Output[Model]    best.pt                         + Output[Metrics] 학습 시간
evaluate → Output[Metrics]  mAP                             + Output[ClassificationMetrics] 이미지별 검출/놓침 혼동행렬
                                                             + Output[HTML] 예측 샘플 이미지
"""
import sys

from kfp import dsl, local
from kfp.dsl import HTML, ClassificationMetrics, Dataset, Input, Markdown, Metrics, Model, Output


@dsl.component
def prepare(yolo_root: str, split: str, dataset: Output[Dataset], summary: Output[Markdown], counts: Output[Metrics]):
    import shutil
    from pathlib import Path
    src = Path(yolo_root) / split
    shutil.copytree(src, dataset.path)                 # 큰 데이터는 값이 아니라 아티팩트 경로로 넘긴다
    dataset.metadata.update({"split": split, "source": "MIDV-500 (smartengines FTP) 5 doc types"})
    lines = ["| split | images |", "|---|---|"]
    for s in ("train", "val", "test"):
        n = len(list((Path(dataset.path) / s / "images").glob("*.jpg")))
        counts.log_metric(f"{s}_images", n)
        lines.append(f"| {s} | {n} |")
    (Path(dataset.path) / "data.yaml").write_text(
        f"path: {dataset.path}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: id_card\n")
    Path(summary.path).write_text("# 데이터 분할 (" + split + ")\n\n" + "\n".join(lines) + "\n")


@dsl.component
def train(dataset: Input[Dataset], epochs: int, model: Output[Model], train_metrics: Output[Metrics]):
    import shutil, time
    from ultralytics import YOLO
    t0 = time.time()
    YOLO("yolov8n.pt").train(data=dataset.path + "/data.yaml", imgsz=640, epochs=epochs, batch=16, device=0,
                             project="/tmp/kfp-yolo", name="run", exist_ok=True, plots=False, verbose=False)
    # 아티팩트 경로는 확장자 없는 파일 이름이다(.../train/model). Ultralytics는 확장자로 형식을 판단하므로
    # 경로를 디렉터리로 쓰고 그 안에 best.pt를 둔다
    import os
    os.makedirs(model.path, exist_ok=True)
    shutil.copy("/tmp/kfp-yolo/run/weights/best.pt", os.path.join(model.path, "best.pt"))
    model.metadata.update({"framework": "ultralytics", "base": "yolov8n.pt", "epochs": epochs})
    train_metrics.log_metric("train_seconds", round(time.time() - t0, 1))
    train_metrics.log_metric("epochs", epochs)


@dsl.component
def evaluate(dataset: Input[Dataset], model: Input[Model], metrics: Output[Metrics],
             detection: Output[ClassificationMetrics], samples: Output[HTML]):
    import base64, io, random
    from pathlib import Path
    from PIL import Image
    from ultralytics import YOLO
    m = YOLO(model.path + "/best.pt")
    v = m.val(data=dataset.path + "/data.yaml", split="test", imgsz=640, device=0, plots=False, verbose=False)
    metrics.log_metric("map50", round(v.box.map50, 4))
    metrics.log_metric("map50_95", round(v.box.map, 4))
    # 이미지 단위: 신분증을 하나라도 찾았나(conf≥0.25) — 혼동행렬은 정답이 모두 '있음'이라 2×2 중 한 줄만 찬다
    imgs = sorted(Path(dataset.path, "test", "images").glob("*.jpg"))
    found = sum(len(r.boxes) > 0 for r in m.predict([str(i) for i in imgs], imgsz=640, device=0, verbose=False, stream=True))
    detection.log_confusion_matrix(["id_card", "none"], [[found, len(imgs) - found], [0, 0]])
    cells = []
    for i in random.Random(0).sample(imgs, 6):
        im = Image.fromarray(m.predict(str(i), imgsz=640, device=0, verbose=False)[0].plot()[..., ::-1])
        im.thumbnail((360, 360)); buf = io.BytesIO(); im.save(buf, "JPEG", quality=80)
        cells.append(f'<img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}" style="margin:4px">')
    Path(samples.path).write_text("<h3>test 예측 샘플</h3>" + "".join(cells))


@dsl.pipeline(name="id-card-yolo-local")
def pipeline(yolo_root: str, split: str = "clip", epochs: int = 5):
    p = prepare(yolo_root=yolo_root, split=split)
    t = train(dataset=p.outputs["dataset"], epochs=epochs)
    evaluate(dataset=p.outputs["dataset"], model=t.outputs["model"])


if __name__ == "__main__":
    local.init(runner=local.SubprocessRunner(use_venv=False), pipeline_root=sys.argv[2])
    run = pipeline(yolo_root=sys.argv[1], split="clip", epochs=int(sys.argv[3]) if len(sys.argv) > 3 else 5)
    print("pipeline finished")
