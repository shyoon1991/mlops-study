"""4주차 실험: 책 8장 train_model/validate_model과 같은 YOLO 호출(yolov8n.pt, imgsz 640)을 조건만 바꿔 돌린다.

사용: python experiments.py <yolo_root> <out_dir> <실험...>
  base        : 학습 전 yolov8n.pt(COCO)로 test 이미지 추론 — 무엇으로 보는지, 상자가 신분증과 얼마나 겹치는지
  epochs      : clip 분할로 에포크 1·5·20 학습 → test mAP
  device      : 1 에포크를 GPU / CPU로 — 학습 시간
  leak        : frame 분할 vs clip 분할, 같은 5 에포크 → 각자의 test mAP
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "results.json"
results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}


def save():
    RESULTS.write_text(json.dumps(results, indent=1, ensure_ascii=False))


def train(split, epochs, device, name, batch=16, **kw):
    model = YOLO("yolov8n.pt")
    t0 = time.time()
    model.train(data=str(ROOT / split / "data.yaml"), imgsz=640, epochs=epochs, batch=batch, device=device,
                project=str(OUT / "runs"), name=name, exist_ok=True, seed=0, deterministic=True,
                plots=False, verbose=False, **kw)
    return time.time() - t0, OUT / "runs" / name / "weights" / "best.pt"


def test_map(weights, split):
    m = YOLO(str(weights)).val(data=str(ROOT / split / "data.yaml"), split="test", imgsz=640, device=0,
                               plots=False, verbose=False, project=str(OUT / "val"), name="v", exist_ok=True)
    return {"map50": round(m.box.map50, 4), "map50_95": round(m.box.map, 4),
            "precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4)}


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


for exp in sys.argv[3:]:
    if exp == "base":
        # 학습 전 COCO 모델: id_card 클래스가 없으니 mAP 대신 '가장 큰 상자'가 신분증과 겹치는 정도와 붙인 이름을 본다
        imgs = sorted((ROOT / "clip" / "test" / "images").glob("*.jpg"))
        model = YOLO("yolov8n.pt")
        names, hits, n = Counter(), 0, 0
        for img in imgs:
            r = model.predict(str(img), imgsz=640, device=0, verbose=False)[0]
            h, w = r.orig_shape
            c, cy, bw, bh = map(float, (ROOT / "clip" / "test" / "labels" / (img.stem + ".txt")).read_text().split()[1:])
            gt = ((c - bw / 2) * w, (cy - bh / 2) * h, (c + bw / 2) * w, (cy + bh / 2) * h)
            n += 1
            if len(r.boxes) == 0:
                names["(없음)"] += 1
                continue
            areas = (r.boxes.xyxy[:, 2] - r.boxes.xyxy[:, 0]) * (r.boxes.xyxy[:, 3] - r.boxes.xyxy[:, 1])
            k = int(areas.argmax())
            names[r.names[int(r.boxes.cls[k])]] += 1
            hits += iou(r.boxes.xyxy[k].tolist(), gt) >= 0.5
        results["base"] = {"test_images": n, "iou50_rate": round(hits / n, 4), "top_labels": names.most_common(6)}
        save()
    elif exp == "epochs":
        for ep in (1, 3, 5, 10, 20):
            if f"clip-ep{ep}" in results:
                continue
            sec, w = train("clip", ep, 0, f"clip-ep{ep}")
            results[f"clip-ep{ep}"] = {"epochs": ep, "train_sec": round(sec, 1), **test_map(w, "clip")}
            save()
    elif exp == "warmup":
        # 1 에포크 mAP 0이 워밍업 탓인지: 같은 1 에포크에 warmup_epochs=0
        sec, w = train("clip", 1, 0, "clip-ep1-nowarmup", warmup_epochs=0)
        results["clip-ep1-nowarmup"] = {"epochs": 1, "warmup_epochs": 0, "train_sec": round(sec, 1), **test_map(w, "clip")}
        save()
    elif exp == "device":
        for dev, name in ((0, "gpu"), ("cpu", "cpu")):
            sec, _ = train("clip", 1, dev, f"device-{name}", workers=8)
            results[f"device-{name}"] = {"epochs": 1, "train_sec": round(sec, 1),
                                         "device": torch.cuda.get_device_name(0) if dev == 0 else f"CPU ({torch.get_num_threads()} threads)"}
            save()
    elif exp == "leak":
        for split in ("frame", "clip"):
            sec, w = train(split, 5, 0, f"leak-{split}-ep5")
            results[f"leak-{split}"] = {"split": split, "epochs": 5, **test_map(w, split)}
            save()
    print(exp, "done")
print(json.dumps(results, indent=1, ensure_ascii=False))
