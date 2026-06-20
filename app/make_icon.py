#!/usr/bin/env python
"""Generate app/AppIcon.icns — a simple mic glyph on a rounded blue tile."""
import os
import subprocess
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))


def draw_icon(size):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # rounded square background (macOS-ish), deep blue gradient feel
    r = int(size * 0.225)
    d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=(36, 99, 235, 255))
    # mic body (capsule)
    cx = size // 2
    bw = int(size * 0.20)
    top = int(size * 0.20)
    bot = int(size * 0.56)
    d.rounded_rectangle([cx - bw, top, cx + bw, bot], radius=bw, fill=(255, 255, 255, 255))
    # mic stand arc
    aw = int(size * 0.32)
    ay0 = int(size * 0.40)
    ay1 = int(size * 0.66)
    lw = max(2, int(size * 0.035))
    d.arc([cx - aw, ay0, cx + aw, ay1], start=20, end=160, fill=(255, 255, 255, 255), width=lw)
    # stem + base
    d.line([cx, int(size * 0.66), cx, int(size * 0.78)], fill=(255, 255, 255, 255), width=lw)
    d.line([cx - int(size * 0.13), int(size * 0.80), cx + int(size * 0.13), int(size * 0.80)],
           fill=(255, 255, 255, 255), width=lw)
    return img


def main():
    iconset = os.path.join(HERE, "AppIcon.iconset")
    os.makedirs(iconset, exist_ok=True)
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    base = draw_icon(1024)
    for s in sizes:
        im = base.resize((s, s), Image.LANCZOS)
        im.save(os.path.join(iconset, f"icon_{s}x{s}.png"))
        if s <= 512:
            im2 = base.resize((s * 2, s * 2), Image.LANCZOS)
            im2.save(os.path.join(iconset, f"icon_{s}x{s}@2x.png"))
    out = os.path.join(HERE, "AppIcon.icns")
    try:
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", out], check=True)
        print("wrote", out)
    except Exception as e:
        # fallback: save a single PNG py2app can still use
        base.save(os.path.join(HERE, "AppIcon.png"))
        print("iconutil failed, wrote AppIcon.png instead:", e)


if __name__ == "__main__":
    main()
