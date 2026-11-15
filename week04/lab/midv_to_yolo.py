"""MIDV-500 원본(문서 네 꼭짓점 quad) → YOLO 검출 데이터셋 (클래스 1개: id_card).

두 가지로 나눈다.
- frame: 책 split_dataset처럼 프레임을 무작위로 75/15/10 (같은 영상의 이웃 프레임이 train·test에 함께 들어간다)
- clip : 영상(클립) 단위로 나눠 test의 영상은 학습에 한 장도 쓰지 않는다
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

RAW = Path(sys.argv[1])          # ~/datasets/midv500/raw
OUT = Path(sys.argv[2])          # ~/datasets/midv500/yolo
RATIOS = (0.75, 0.15, 0.10)
SEED = 42


def quad_to_yolo(quad, w, h):
    xs = [min(max(p[0], 0), w) for p in quad]
    ys = [min(max(p[1], 0), h) for p in quad]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if (x1 - x0) * (y1 - y0) < 0.01 * w * h:   # 화면 밖으로 거의 나간 프레임은 버린다
        return None
    return f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"


frames = []   # (doc_type, clip, image_path, label_line)
skipped = 0
for doc in sorted(p for p in RAW.iterdir() if p.is_dir()):
    for img in sorted((doc / "images").glob("*/*.tif")):
        clip = img.parent.name
        gt = doc / "ground_truth" / clip / (img.stem + ".json")
        quad = json.loads(gt.read_text())["quad"]
        with Image.open(img) as im:
            w, h = im.size
        line = quad_to_yolo(quad, w, h)
        if line is None:
            skipped += 1
            continue
        frames.append((doc.name, clip, img, line))

print(f"frames={len(frames)} skipped={skipped} docs={len({f[0] for f in frames})} clips={len({(f[0], f[1]) for f in frames})}")


def split_frame(items):
    items = items[:]
    random.Random(SEED).shuffle(items)
    n = len(items)
    a, b = int(n * RATIOS[0]), int(n * (RATIOS[0] + RATIOS[1]))
    return {"train": items[:a], "val": items[a:b], "test": items[b:]}


def split_clip(items):
    by_clip = defaultdict(list)
    for it in items:
        by_clip[(it[0], it[1])].append(it)
    out = {"train": [], "val": [], "test": []}
    # 문서 종류마다 영상 10개를 7.5/1.5/1 비율에 가깝게 → 종류별로 train 7, val 2, test 1
    per_doc = defaultdict(list)
    for key in by_clip:
        per_doc[key[0]].append(key)
    rng = random.Random(SEED)
    for doc, keys in per_doc.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        n_test, n_val = max(1, round(len(keys) * RATIOS[2])), max(1, round(len(keys) * RATIOS[1]))
        for i, k in enumerate(keys):
            split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
            out[split] += by_clip[k]
    return out


for name, splitter in (("frame", split_frame), ("clip", split_clip)):
    parts = splitter(frames)
    for split, items in parts.items():
        (OUT / name / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / name / split / "labels").mkdir(parents=True, exist_ok=True)
        for doc, clip, img, line in items:
            stem = f"{doc}__{img.stem}"
            dst = OUT / name / split / "images" / f"{stem}.jpg"
            if not dst.exists():
                with Image.open(img) as im:
                    im.convert("RGB").save(dst, quality=90)
            (OUT / name / split / "labels" / f"{stem}.txt").write_text(line + "\n")
    (OUT / name / "data.yaml").write_text(
        f"path: {OUT / name}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: id_card\n")
    clips = {s: len({(i[0], i[1]) for i in v}) for s, v in parts.items()}
    shared = {(i[0], i[1]) for i in parts["train"]} & {(i[0], i[1]) for i in parts["test"]}
    print(f"[{name}] " + " ".join(f"{s}={len(v)}장/{clips[s]}영상" for s, v in parts.items())
          + f" | train과 test가 공유하는 영상: {len(shared)}개")
