"""YC5 (cá nhân) – Tự động tạo slide .pptx từ file .md hoặc .txt.

Quy ước file đầu vào:
  # Tiêu đề bài        -> slide tiêu đề (dòng ngay sau là phụ đề, tuỳ chọn)
  ## Tiêu đề slide     -> slide nội dung mới
  - gạch đầu dòng      -> ý trong slide (thụt 2 dấu cách = ý con)
  đoạn văn thường      -> một ý
  ![chú thích](đường/dẫn/hình.png) -> chèn hình vào slide

Với .txt không có '#': mỗi đoạn (cách nhau bằng dòng trống) thành một slide,
dòng đầu của đoạn là tiêu đề.

Ví dụ:  python src/md_to_pptx.py docs/slides.md --out results/slides.pptx
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

BLUE = RGBColor(0x1F, 0x4E, 0x8C)
IMG_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")


def parse(text: str, is_md: bool) -> tuple[dict, list[dict]]:
    title, slides = {"title": "", "subtitle": ""}, []
    if not is_md or not re.search(r"^#", text, re.M):
        blocks = [b.strip().splitlines() for b in re.split(r"\n\s*\n", text) if b.strip()]
        title["title"] = blocks[0][0] if blocks else ""
        for b in blocks[1:]:
            slides.append({"title": b[0], "bullets": [(0, l) for l in b[1:]], "images": []})
        return title, slides

    cur = None
    for line in text.splitlines():
        if line.startswith("# "):
            title["title"] = line[2:].strip()
        elif line.startswith("## "):
            cur = {"title": line[3:].strip(), "bullets": [], "images": []}
            slides.append(cur)
        elif cur is None:
            if line.strip() and not title["subtitle"]:
                title["subtitle"] = line.strip()
        elif m := IMG_RE.search(line):
            cur["images"].append((m.group(1), m.group(2)))
        elif re.match(r"^\s*[-*+]\s+", line):
            level = (len(line) - len(line.lstrip())) // 2
            cur["bullets"].append((min(level, 3), re.sub(r"^\s*[-*+]\s+", "", line)))
        elif line.strip():
            cur["bullets"].append((0, line.strip()))
    return title, slides


def strip_md(s: str) -> str:
    return re.sub(r"(\*\*|__|\*|`)", "", s)


def build(title: dict, slides: list[dict], base: Path, out: Path) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)  # 16:9

    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = strip_md(title["title"])
    s.shapes.title.text_frame.paragraphs[0].font.color.rgb = BLUE
    s.placeholders[1].text = strip_md(title["subtitle"])

    for sl in slides:
        s = prs.slides.add_slide(prs.slide_layouts[1])
        s.shapes.title.text = strip_md(sl["title"])
        s.shapes.title.text_frame.paragraphs[0].font.color.rgb = BLUE
        s.shapes.title.text_frame.paragraphs[0].font.size = Pt(32)
        body = s.placeholders[1]
        has_img = bool(sl["images"])
        body.left, body.top = Inches(0.6), Inches(1.5)
        body.width, body.height = (Inches(6.2) if has_img else Inches(12)), Inches(5.5)
        tf = body.text_frame
        tf.clear()
        for i, (lvl, txt) in enumerate(sl["bullets"]):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.text, p.level = strip_md(txt), lvl
            p.font.size = Pt(20 if lvl == 0 else 17)
        if has_img:
            n = len(sl["images"])
            h = 5.4 / n
            for j, (cap, path) in enumerate(sl["images"]):
                p = (base / path).resolve()
                if p.exists():
                    s.shapes.add_picture(str(p), Inches(7.1), Inches(1.5 + j * h), height=Inches(h - 0.15))
        if not sl["bullets"]:
            body.text_frame.text = ""
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=".md/.txt -> .pptx")
    ap.add_argument("src")
    ap.add_argument("--out", default="results/slides.pptx")
    args = ap.parse_args()
    src = Path(args.src)
    title, slides = parse(src.read_text(encoding="utf-8"), src.suffix.lower() == ".md")
    build(title, slides, src.parent, Path(args.out))
    print(f"Đã tạo {args.out} ({len(slides) + 1} slide)")


if __name__ == "__main__":
    main()
