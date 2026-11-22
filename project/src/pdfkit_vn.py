"""Tiện ích dựng PDF tiếng Việt bằng reportlab (dùng chung cho csv_to_pdf.py và build_report.py)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# Font DejaVu đi kèm matplotlib, có đủ dấu tiếng Việt
_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(_FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(_FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")

_base = getSampleStyleSheet()
STYLES = {
    "title": ParagraphStyle("t", parent=_base["Title"], fontName="DejaVu-Bold", fontSize=17, leading=22),
    "subtitle": ParagraphStyle("st", parent=_base["Normal"], fontName="DejaVu", fontSize=9.5,
                               alignment=TA_CENTER, textColor=colors.grey),
    "h1": ParagraphStyle("h1", parent=_base["Heading2"], fontName="DejaVu-Bold", fontSize=13,
                         textColor=colors.HexColor("#1F4E8C"), spaceBefore=10),
    "body": ParagraphStyle("b", parent=_base["BodyText"], fontName="DejaVu", fontSize=10, leading=14.5),
    "caption": ParagraphStyle("c", parent=_base["Italic"], fontName="DejaVu", fontSize=8.5,
                              alignment=TA_CENTER, textColor=colors.HexColor("#444444")),
    "cell": ParagraphStyle("cell", parent=_base["BodyText"], fontName="DejaVu", fontSize=8, leading=10),
}


def para(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def figure(path: Path, caption: str, width_cm: float = 14) -> list:
    from reportlab.lib.utils import ImageReader

    w, h = ImageReader(str(path)).getSize()
    img = Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)
    return [img, para(caption, "caption"), Spacer(1, 8)]


def table(rows: list[list], header: bool = True, col_widths=None) -> Table:
    data = [[Paragraph(str(c), STYLES["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6FB")]),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D6E4F5")))
    t.setStyle(TableStyle(style))
    return t


def build(path: Path, story: list) -> None:
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("DejaVu", 8)
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Trang {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
