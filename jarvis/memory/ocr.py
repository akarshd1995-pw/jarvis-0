"""Read the text in screenshots and images with our own OCR models (phase 7 step 2), on the CPU via ONNX.

    ocr = OCR.load(checkpoints_dir)      # None if the models haven't been trained yet
    text = ocr.read("Screenshot 2026-11-20.png")

1. the line finder marks text-line cells (1/4 resolution); cells are grouped into line boxes
2. each box is cut out, scaled to 32 px high and read by the reader (CTC, greedy)
3. lines come back in reading order (top to bottom, left to right)
"""
from __future__ import annotations

from pathlib import Path

import re

import numpy as np

STRIDE = 4
MAX_SIDE = 2560
IMAGE_EXT = {"png", "jpg", "jpeg", "bmp", "webp", "gif", "tif", "tiff"}


def components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (y0, x0, y1, x1) of 4-connected regions in a small boolean map."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes = []
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        stack = [(y, x)]
        seen[y, x] = True
        y0 = y1 = y
        x0 = x1 = x
        while stack:
            cy, cx = stack.pop()
            y0, y1, x0, x1 = min(y0, cy), max(y1, cy), min(x0, cx), max(x1, cx)
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        boxes.append((y0, x0, y1 + 1, x1 + 1))
    return boxes


def line_boxes(prob: np.ndarray, img_h: int, img_w: int, thr: float = 0.5) -> list[tuple[int, int, int, int]]:
    """Probability map (H/4, W/4) -> line boxes in image pixels (x0, y0, x1, y1), reading order."""
    m = prob > thr
    # bridge one-cell gaps between words on the same line
    bridged = m.copy()
    bridged[:, 1:-1] |= m[:, :-2] & m[:, 2:]
    out = []
    for y0, x0, y1, x1 in components(bridged):
        hh, ww = y1 - y0, x1 - x0
        if hh < 1 or ww < 2:
            continue
        # the finder was trained on boxes shrunk by 1/5 of the height at top and bottom: grow back
        py0, py1 = y0 * STRIDE, y1 * STRIDE
        grow = int(round((py1 - py0) * 0.2))       # cell rounding already adds ~a cell
        bx0, bx1 = max(0, x0 * STRIDE - 2), min(img_w, x1 * STRIDE + 2)
        by0, by1 = max(0, py0 - grow), min(img_h, py1 + grow)
        if by1 - by0 >= 6 and bx1 - bx0 >= 6:
            out.append((bx0, by0, bx1, by1))
    # reading order: rows (boxes whose vertical centres are close), then left to right
    out.sort(key=lambda b: ((b[1] + b[3]) // 2 // 12, b[0]))
    return out


def looks_like_text(text: str, conf: float) -> bool:
    """Drop what the reader invents on photos, logos and charts: unsure readings, and strings that are mostly
    symbols or contain no real word."""
    if len(text) < 2 or conf < 0.75:
        return False
    chars = text.replace(" ", "")
    alnum = sum(c.isalnum() for c in chars) / max(1, len(chars))
    if alnum < 0.6:
        return False
    words = [w for w in re.findall(r"[A-Za-z]{2,}|\d+", text)]
    return bool(words) and (conf >= 0.9 or len(text) >= 4)


class OCR:
    def __init__(self, finder, reader, threads: int = 2):
        self.finder, self.reader = finder, reader

    @classmethod
    def load(cls, ck: Path, threads: int = 2) -> "OCR | None":
        f, r = ck / "ocr_finder.onnx", ck / "ocr_reader.onnx"
        if not (f.exists() and r.exists()):
            return None
        from ..onnxutil import session              # OCR stays full precision: CTC reading is sensitive to it
        return cls(session(f, threads=threads, int8=False, name="OCR text finder"),
                   session(r, threads=threads, int8=False, name="OCR reader"))

    def find(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = gray.shape
        ph, pw = (-h) % 16, (-w) % 16
        x = np.pad(gray, ((0, ph), (0, pw)), mode="edge")[None, None].astype(np.float32)
        logits = self.finder.run(["logits"], {"image": x})[0][0, 0]
        prob = 1 / (1 + np.exp(-logits))
        return line_boxes(prob[: (h + 3) // STRIDE, : (w + 3) // STRIDE], h, w)

    def read_lines(self, gray: np.ndarray, boxes: list) -> list[str]:
        from PIL import Image
        from training.ocr.charset import decode_greedy
        crops = []
        for x0, y0, x1, y1 in boxes:
            c = gray[y0:y1, x0:x1]
            if c.size == 0:
                continue
            new_w = max(8, int(round(c.shape[1] * 32 / c.shape[0] / 4)) * 4)
            new_w = min(new_w, 2048)
            img = Image.fromarray((c * 255).astype(np.uint8)).resize((new_w, 32), Image.BILINEAR)
            crops.append(np.asarray(img, np.float32) / 255.0)
        texts = [""] * len(crops)
        order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1])
        for i in range(0, len(order), 32):               # batches of similar width
            idx = order[i:i + 32]
            w = max(crops[k].shape[1] for k in idx)
            x = np.stack([np.pad(crops[k], ((0, 0), (0, w - crops[k].shape[1])), mode="edge") for k in idx])[:, None]
            lp = self.reader.run(["logprobs"], {"line": x.astype(np.float32)})[0]
            for j, k in enumerate(idx):
                t = crops[k].shape[1] // 4
                best = lp[j, :t].argmax(-1)
                text = decode_greedy(best.tolist()).strip()
                emitted = best != 0                              # steps that produced a character
                conf = float(np.exp(lp[j, :t].max(-1)[emitted]).mean()) if emitted.any() else 0.0
                texts[k] = text if looks_like_text(text, conf) else ""
        return [t for t in texts if t]

    def read(self, path: str | Path) -> str:
        from PIL import Image
        img = Image.open(path).convert("L")
        if max(img.size) > MAX_SIDE:
            s = MAX_SIDE / max(img.size)
            img = img.resize((int(img.width * s), int(img.height * s)), Image.BILINEAR)
        gray = np.asarray(img, np.float32) / 255.0
        return "\n".join(self.read_lines(gray, self.find(gray)))
