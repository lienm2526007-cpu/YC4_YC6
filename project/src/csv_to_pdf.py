"""YC5 (cá nhân) – Đọc một file CSV bất kỳ và sinh báo cáo PDF.

Báo cáo gồm: tiêu đề, mô tả, bảng dữ liệu (thống kê + mẫu), biểu đồ minh họa, kết luận ngắn.

Ví dụ:
  python src/csv_to_pdf.py data/raw/student-por.csv --sep ";" \
      --title "Báo cáo thống kê kết quả học tập" --target G3 --out results/csv_report.pdf
"""
from __future__ import annotations

import argparse
import tempfile
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from pdfkit_vn import build, figure, para, table


def make_charts(df: pd.DataFrame, target: str | None, tmp: Path) -> list[tuple[Path, str]]:
    charts = []
    num = df.select_dtypes("number")
    col = target if target in num.columns else (num.columns[-1] if len(num.columns) else None)

    if col is not None:
        p = tmp / "hist.png"
        plt.figure(figsize=(6, 3.2))
        plt.hist(num[col].dropna(), bins=20, color="#2E6FB7", edgecolor="white")
        plt.xlabel(col); plt.ylabel("Tần suất"); plt.tight_layout(); plt.savefig(p, dpi=180); plt.close()
        charts.append((p, f"Phân phối của biến {col}."))

        corr = num.corr()[col].drop(col).dropna().sort_values()
        if len(corr):
            top = pd.concat([corr.head(4), corr.tail(4)]).drop_duplicates()
            p = tmp / "corr.png"
            plt.figure(figsize=(6, 3.2))
            plt.barh(top.index, top.values, color=["#C0392B" if v < 0 else "#2E6FB7" for v in top.values])
            plt.xlabel(f"Hệ số tương quan với {col}"); plt.tight_layout(); plt.savefig(p, dpi=180); plt.close()
            charts.append((p, f"Các biến tương quan mạnh nhất với {col} (Pearson)."))

    cat = df.select_dtypes(exclude="number")
    if len(cat.columns):
        c = cat.columns[0]
        vc = cat[c].value_counts().head(8)
        p = tmp / "cat.png"
        plt.figure(figsize=(4.5, 3.2))
        plt.pie(vc.values, labels=vc.index, autopct="%1.0f%%", startangle=90)
        plt.tight_layout(); plt.savefig(p, dpi=180); plt.close()
        charts.append((p, f"Cơ cấu theo biến phân loại {c}."))
    return charts


def conclusion(df: pd.DataFrame, target: str | None) -> str:
    num = df.select_dtypes("number")
    parts = [f"Bộ dữ liệu có {len(df)} bản ghi và {df.shape[1]} cột "
             f"({num.shape[1]} cột số, {df.shape[1] - num.shape[1]} cột phân loại); "
             f"có {int(df.isna().sum().sum())} giá trị thiếu."]
    if target in num.columns:
        corr = num.corr()[target].drop(target).dropna()
        if len(corr):
            best = corr.abs().idxmax()
            parts.append(f"Biến {target} có trung bình {num[target].mean():.2f} "
                         f"(độ lệch chuẩn {num[target].std():.2f}); biến liên quan mạnh nhất là "
                         f"{best} (r = {corr[best]:.2f}).")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="CSV -> báo cáo PDF")
    ap.add_argument("csv")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--title", default="Báo cáo dữ liệu tự động")
    ap.add_argument("--desc", default="")
    ap.add_argument("--target", default=None, help="cột số chính cần phân tích")
    ap.add_argument("--out", default="results/csv_report.pdf")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, sep=args.sep)
    for c in df.columns:  # chuyển cột số bị đọc thành chuỗi (vd "5")
        if df[c].dtype == object:
            conv = pd.to_numeric(df[c], errors="coerce")
            if conv.notna().mean() > 0.95:
                df[c] = conv

    desc = args.desc or (f"Báo cáo được sinh tự động từ tệp <b>{Path(args.csv).name}</b> "
                         f"ngày {date.today():%d/%m/%Y} bằng Python (pandas, matplotlib, reportlab).")
    num = df.select_dtypes("number")
    stats = num.describe().T[["mean", "std", "min", "max"]].round(2).head(12)
    stats_rows = [["Biến", "Trung bình", "Độ lệch chuẩn", "Nhỏ nhất", "Lớn nhất"]] + \
                 [[i, *r] for i, r in zip(stats.index, stats.values.tolist())]
    sample = df.iloc[:8, :8]
    sample_rows = [list(sample.columns)] + sample.astype(str).values.tolist()

    story = [para(args.title, "title"), para(desc, "subtitle"),
             para("1. Mô tả dữ liệu", "h1"),
             para(f"Số bản ghi: <b>{len(df)}</b>; số cột: <b>{df.shape[1]}</b>. "
                  "Bảng 1 là một phần dữ liệu mẫu, Bảng 2 là thống kê mô tả các biến số."),
             para("Bảng 1. Dữ liệu mẫu (8 dòng, 8 cột đầu).", "caption"), table(sample_rows),
             para("Bảng 2. Thống kê mô tả.", "caption"), table(stats_rows),
             para("2. Biểu đồ minh họa", "h1")]
    with tempfile.TemporaryDirectory() as td:
        for i, (p, cap) in enumerate(make_charts(df, args.target, Path(td)), 1):
            story += figure(p, f"Hình {i}. {cap}", width_cm=13)
        story += [para("3. Kết luận", "h1"), para(conclusion(df, args.target))]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        build(Path(args.out), story)
    print("Đã tạo", args.out)


if __name__ == "__main__":
    main()
