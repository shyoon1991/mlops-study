"""같은 test 이미지 3장을 기본 yolov8n.pt / 3 에포크 / 20 에포크로 추론해 3×3 격자로."""
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
H = Path.home()
imgs = sorted((H / "datasets/midv500/yolo/clip/test/images").glob("*.jpg"))
pick = [next(i for i in imgs if i.stem.endswith(k)) for k in ("TS47_15", "HS09_15", "PS16_11")]   # 책상 위 / 손에 든 / 일부가 화면 밖
models = [("base yolov8n.pt (COCO)", H / "mlops-ch8/yolov8n.pt"),
          ("3 epochs", H / "mlops-ch8/out/runs/clip-ep3/weights/best.pt"),
          ("20 epochs", H / "mlops-ch8/out/runs/clip-ep20/weights/best.pt")]
W, CELL = 3, 380
font = ImageFont.load_default(24)
grid = Image.new("RGB", (W * CELL, 40 + len(pick) * CELL), "white")
d = ImageDraw.Draw(grid)
for c, (title, w) in enumerate(models):
    d.text((c * CELL + 10, 8), title, fill="black", font=font)
    m = YOLO(str(w))
    for r, img in enumerate(pick):
        res = m.predict(str(img), imgsz=640, device=0, verbose=False, conf=0.25)[0]
        im = Image.fromarray(res.plot(line_width=6, font_size=40)[..., ::-1])
        im.thumbnail((CELL - 8, CELL - 8))
        grid.paste(im, (c * CELL + (CELL - im.width) // 2, 40 + r * CELL + (CELL - im.height) // 2))
out = "/mnt/d/LOCAL-AI-SETUP/mlops-study/week4/shots/pred-grid.jpg"
grid.save(out, quality=88); print(out, [p.name for p in pick])
