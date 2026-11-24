"""Training images, rendered on the fly with PIL.

render_line(): one text line, height 32, for the reader (with blur, noise, JPEG, colour and scale changes).
render_page(): a fake screenshot (panels, buttons, dividers, text in several sizes and colours, light or dark)
               plus a mask of where the text lines are, for the line finder.
"""
from __future__ import annotations

import io
import random

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import text as T

H = 32                     # reader input height


class FontBank:
    def __init__(self, paths: list[str]):
        self.paths = paths
        self.cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

    def get(self, r: random.Random, size: int):
        p = r.choice(self.paths)
        key = (p, size)
        if key not in self.cache:
            try:
                self.cache[key] = ImageFont.truetype(p, size)
            except OSError:
                self.cache[key] = ImageFont.load_default()
            if len(self.cache) > 400:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[key]


def _colors(r: random.Random) -> tuple[int, int]:
    """(foreground, background) grey levels with enough contrast; light or dark theme."""
    if r.random() < 0.6:
        bg = r.randint(200, 255)
        fg = r.randint(0, max(0, bg - 90))
    else:
        bg = r.randint(0, 70)
        fg = r.randint(min(255, bg + 90), 255)
    return fg, bg


def _jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("L")


def render_line(r: random.Random, fonts: FontBank, text: str | None = None) -> tuple[np.ndarray, str]:
    """Returns (float32 array H x W in 0..1, text)."""
    s = text if text is not None else T.line(r)
    size = r.randint(14, 44)
    font = fonts.get(r, size)
    fg, bg = _colors(r)
    try:
        x0, y0, x1, y1 = font.getbbox(s)
    except Exception:
        x0, y0, x1, y1 = 0, 0, size * len(s) // 2, size
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    padx, pady = r.randint(1, size // 2 + 2), r.randint(1, size // 3 + 2)
    canvas = Image.new("L", (w + 2 * padx, max(h, size) + 2 * pady), bg)
    d = ImageDraw.Draw(canvas)
    d.text((padx - x0, pady - y0 + r.randint(-1, 1)), s, fill=fg, font=font)
    if r.random() < 0.08:                                  # underline (links)
        yy = pady + h + 1
        d.line([(padx, yy), (padx + w, yy)], fill=fg, width=1)
    img = canvas
    if r.random() < 0.15:
        img = img.rotate(r.uniform(-1.5, 1.5), resample=Image.BILINEAR, expand=True, fillcolor=bg)
    # scale to height H with a random horizontal stretch (screen DPI / font width variety)
    scale = H / img.height
    new_w = max(8, int(img.width * scale * r.uniform(0.85, 1.15)))
    img = img.resize((new_w, H), Image.BILINEAR)
    if r.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.3, 0.9)))
    if r.random() < 0.25:
        img = _jpeg(img, r.randint(30, 90))
    a = np.asarray(img, dtype=np.float32) / 255.0
    if r.random() < 0.3:
        a = a + np.random.default_rng(r.randint(0, 2**31)).normal(0, r.uniform(0.01, 0.05), a.shape)
    if r.random() < 0.15:                                  # gradient background
        a = a * r.uniform(0.85, 1.0) + np.linspace(0, r.uniform(-0.1, 0.1), a.shape[1])[None, :]
    return np.clip(a, 0, 1).astype(np.float32), s


# ------------------------------------------------------------------ pages for the line finder
def render_page(r: random.Random, fonts: FontBank, W: int = 640, Hh: int = 384, stride: int = 4):
    """Returns (image float32 Hh x W, mask float32 Hh/stride x W/stride, boxes [(x0,y0,x1,y1,text)])."""
    fg_base, bg = _colors(r)
    img = Image.new("L", (W, Hh), bg)
    d = ImageDraw.Draw(img)
    mask = Image.new("L", (W // stride, Hh // stride), 0)
    md = ImageDraw.Draw(mask)
    # panels / cards / sidebars with their own background
    for _ in range(r.randint(0, 4)):
        x0, y0 = r.randint(0, W - 40), r.randint(0, Hh - 30)
        x1, y1 = min(W, x0 + r.randint(40, W)), min(Hh, y0 + r.randint(20, Hh))
        shade = int(np.clip(bg + r.randint(-40, 40), 0, 255))
        d.rectangle([x0, y0, x1, y1], fill=shade, outline=shade if r.random() < 0.5 else int(np.clip(shade - 40, 0, 255)))
    # non-text distractors: lines, icons, dots
    for _ in range(r.randint(0, 10)):
        c = r.randint(0, 255)
        k = r.random()
        x, y = r.randint(0, W), r.randint(0, Hh)
        if k < 0.4:
            d.line([(x, y), (x + r.randint(-200, 200), y + (0 if r.random() < 0.7 else r.randint(-100, 100)))],
                   fill=c, width=r.randint(1, 2))
        elif k < 0.7:
            s = r.randint(6, 22)
            d.rectangle([x, y, x + s, y + s], outline=c, fill=c if r.random() < 0.5 else None)
        else:
            s = r.randint(4, 14)
            d.ellipse([x, y, x + s, y + s], fill=c)
    boxes = []
    occupied = np.zeros((Hh, W), dtype=bool)
    for _ in range(r.randint(6, 45)):
        size = r.choice([10, 11, 12, 12, 13, 13, 14, 14, 15, 16, 18, 20, 22, 26, 30, 36])
        font = fonts.get(r, size)
        s = T.line(r, max_chars=r.choice([12, 24, 40, 60]))
        try:
            x0, y0, x1, y1 = font.getbbox(s)
        except Exception:
            continue
        tw, th = x1 - x0, y1 - y0
        if tw <= 2 or th <= 2 or tw >= W - 4 or th >= Hh - 4:
            continue
        for _try in range(6):
            px, py = r.randint(2, W - tw - 2), r.randint(2, Hh - th - 2)
            if not occupied[max(0, py - 3):py + th + 3, max(0, px - 6):px + tw + 6].any():
                break
        else:
            continue
        local_bg = int(np.asarray(img)[py + th // 2, px + tw // 2])
        fg = 255 - local_bg if abs((255 - local_bg) - local_bg) > 90 else (0 if local_bg > 127 else 255)
        fg = int(np.clip(fg + r.randint(-40, 40), 0, 255))
        if abs(fg - local_bg) < 70:
            fg = 0 if local_bg > 127 else 255
        if r.random() < 0.12:                             # a button behind the text
            pad = r.randint(3, 8)
            b = int(np.clip(local_bg + r.choice([-60, 60]), 0, 255))
            d.rectangle([px - pad, py - pad, px + tw + pad, py + th + pad], fill=b)
            fg = 0 if b > 127 else 255
        d.text((px - x0, py - y0), s, fill=fg, font=font)
        occupied[py:py + th, px:px + tw] = True
        boxes.append((px, py, px + tw, py + th, s))
        # mask: the line's box shrunk vertically (keeps neighbouring lines apart)
        sh = max(1, th // 5)
        md.rectangle([px // stride, (py + sh) // stride, (px + tw) // stride, max((py + sh) // stride,
                      (py + th - sh) // stride)], fill=255)
    if r.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(r.uniform(0.2, 0.6)))
    if r.random() < 0.25:
        img = _jpeg(img, r.randint(40, 92))
    a = np.asarray(img, dtype=np.float32) / 255.0
    if r.random() < 0.5:                                  # both polarities
        pass
    m = (np.asarray(mask, dtype=np.float32) / 255.0)
    return a.astype(np.float32), m.astype(np.float32), boxes


# ------------------------------------------------------------------ reader input helpers
MAX_W = 1024


def _need_steps(ids: list[int]) -> int:
    """CTC needs one step per character plus one between repeated characters."""
    return len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)


def prepare(a: np.ndarray, ids: list[int]) -> np.ndarray:
    """Make the width a multiple of 4, wide enough for CTC, at most MAX_W."""
    need = 4 * (_need_steps(ids) + 2)
    w = a.shape[1]
    target = min(MAX_W, max(w, need))
    target = int(math.ceil(target / 4) * 4)
    if target != w:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((target, a.shape[0]),
                                                                          Image.BILINEAR), np.float32) / 255.0
    return a


