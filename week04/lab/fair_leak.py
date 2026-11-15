"""같은 이미지로 누수 효과만 보기: clip 분할의 test 영상 5개에 속하면서 frame 분할의 test에도 든 프레임.
frame 모델은 그 영상의 다른 프레임으로 학습했고, clip 모델은 그 영상을 한 장도 본 적 없다."""
import json
from pathlib import Path
from ultralytics import YOLO
H = Path.home(); Y = H / "datasets/midv500/yolo"
clips = {p.stem.rsplit("_", 1)[0] for p in (Y / "clip/test/images").glob("*.jpg")}
same = sorted(str(p) for p in (Y / "frame/test/images").glob("*.jpg") if p.stem.rsplit("_", 1)[0] in clips)
lst = H / "mlops-ch8/fair_test.txt"; lst.write_text("\n".join(same) + "\n")
yaml = H / "mlops-ch8/fair.yaml"
yaml.write_text(f"path: {Y}\ntrain: {lst}\nval: {lst}\ntest: {lst}\nnames:\n  0: id_card\n")
out = {"images": len(same), "clips": sorted(clips)}
for name in ("leak-frame-ep5", "leak-clip-ep5"):
    m = YOLO(str(H / f"mlops-ch8/out/runs/{name}/weights/best.pt")).val(data=str(yaml), split="test", imgsz=640, device=0, plots=False, verbose=False, project=str(H / "mlops-ch8/out/val"), name="fair", exist_ok=True)
    out[name] = {"map50": round(m.box.map50, 4), "map50_95": round(m.box.map, 4), "recall": round(float(m.box.mr), 4)}
# 참고: 장면 종류별 이미지 수
out["frame_test_scenes"] = {}
for p in (Y / "frame/test/images").glob("*.jpg"):
    s = p.stem.split("__")[1][:2]; out["frame_test_scenes"][s] = out["frame_test_scenes"].get(s, 0) + 1
r = json.loads((H / "mlops-ch8/out/results.json").read_text()); r["fair-leak"] = out
(H / "mlops-ch8/out/results.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
print(json.dumps(out, ensure_ascii=False))
