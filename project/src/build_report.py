"""YC5 (nhóm) – Sinh báo cáo khoa học PDF từ kết quả phân tích.

Chạy sau src/analysis.py:
  python src/build_report.py --out results/bao-cao-phan-tich.pdf
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from pdfkit_vn import build, figure, para, table

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"


def make_story(m: dict) -> list:
    dt, cl, rg = m["decision_tree"], m["clustering"], m["regression"]
    cm = dt["confusion_matrix"]
    imp = ", ".join(f"{k} ({v:.2f})" for k, v in dt["importance"].items())
    s = [
        para("Phân tích và dự đoán kết quả học tập của học sinh bằng học máy", "title"),
        para(f"Báo cáo sinh tự động ngày {date.today():%d/%m/%Y} – Dữ liệu: UCI Student Performance "
             "(Cortez &amp; Silva, 2008), giấy phép CC BY 4.0", "subtitle"),

        para("1. Giới thiệu đề tài", "h1"),
        para("Phát hiện sớm học sinh có nguy cơ không đạt giúp giáo viên can thiệp kịp thời. "
             "Đề tài sử dụng ba kỹ thuật học máy: (i) cây quyết định để phân loại Đạt/Chưa đạt, "
             "(ii) K-Means để phân nhóm người học theo hành vi, (iii) hồi quy tuyến tính để dự đoán điểm "
             "cuối kỳ."),

        para("2. Mô tả dữ liệu", "h1"),
        para(f"Tệp <b>{m['dataset']}</b> gồm {m['n_raw']} học sinh trung học (môn tiếng Bồ Đào Nha) "
             f"tại hai trường ở Bồ Đào Nha, 33 thuộc tính về nhân khẩu học, gia đình, thói quen học tập "
             f"và ba cột điểm G1, G2, G3 (thang 0–20). Sau làm sạch còn {m['n_clean']} bản ghi; "
             f"điểm G3 trung bình {m['g3_mean']} (độ lệch chuẩn {m['g3_std']}), "
             f"tỷ lệ đạt (G3 ≥ 10) là {m['pass_rate'] * 100:.1f}%. Dữ liệu đã ẩn danh, "
             "không chứa thông tin định danh cá nhân."),
        *figure(FIG / "eda_g3_hist.png", "Hình 1. Phân phối điểm cuối kỳ G3.", 10),
        *figure(FIG / "eda_corr_heatmap.png", "Hình 2. Ma trận tương quan giữa các biến chính.", 11),

        para("3. Kết quả phân tích", "h1"),
        para(f"<b>3.1 Cây quyết định.</b> Chia dữ liệu 70/30 ({dt['train_size']}/{dt['test_size']} mẫu), "
             f"độ sâu tối đa 3. Độ chính xác trên tập kiểm tra đạt <b>{dt['accuracy'] * 100:.1f}%</b>, "
             f"F1 của lớp Chưa đạt là {dt['f1_fail_class']:.2f}. Mức độ quan trọng: {imp}."),
        *figure(FIG / "tree.png", "Hình 3. Cây quyết định phân loại Đạt/Chưa đạt.", 16),
        table([["", "Dự đoán Chưa đạt", "Dự đoán Đạt"],
               ["Thực tế Chưa đạt", cm[0][0], cm[0][1]],
               ["Thực tế Đạt", cm[1][0], cm[1][1]]]),
        para("Bảng 1. Ma trận nhầm lẫn trên tập kiểm tra.", "caption"),
        para(f"<b>3.2 Gom nhóm K-Means.</b> Chuẩn hóa 6 biến hành vi rồi chọn k = {cl['k']} "
             "dựa vào phương pháp khuỷu tay và hệ số silhouette."),
        table([["Cụm", "Số HS", "G1 TB", "Tự học", "Lần trượt", "Vắng", "Đi chơi", "Tỷ lệ đạt"]] +
              [[p["cluster"], p["n"], p["G1"], p["studytime"], p["failures"], p["absences"], p["goout"],
                f"{p['pass'] * 100:.0f}%"] for p in cl["profile"]]),
        para("Bảng 2. Đặc trưng trung bình của từng cụm.", "caption"),
        *figure(FIG / "cluster_pca.png", "Hình 4. Các cụm chiếu lên 2 thành phần chính (PCA).", 10),
        para(f"<b>3.3 Hồi quy tuyến tính.</b> Dự đoán G3 từ G1, G2, thời gian tự học, số lần trượt và số "
             f"buổi vắng: R² = <b>{rg['r2']:.3f}</b>, MAE = {rg['mae']:.2f} điểm."),
        *figure(FIG / "reg_pred_vs_true.png", "Hình 5. Giá trị G3 dự đoán so với thực tế.", 9),

        para("4. Kết luận và đề xuất", "h1"),
        para("Điểm giữa kỳ G1 là yếu tố dự báo mạnh nhất; số buổi vắng và số lần trượt trước đó là các tín "
             "hiệu cảnh báo bổ sung. Cụm học sinh có G1 thấp và từng trượt môn có tỷ lệ đạt chưa đến 50%, "
             "cần được ưu tiên phụ đạo. Hướng phát triển: thử thêm Random Forest, cân bằng lớp, và kiểm chứng "
             "trên dữ liệu học sinh Việt Nam."),
        para("Tài liệu tham khảo", "h1"),
        para("Cortez, P., Silva, A.: Using data mining to predict secondary school student performance. "
             "In: Proc. 5th FUture BUsiness TEChnology Conference, pp. 5–12 (2008). "
             "Dữ liệu: https://doi.org/10.24432/C5TG7T"),
    ]
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(ROOT / "results" / "metrics.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "bao-cao-phan-tich.pdf"))
    args = ap.parse_args()
    m = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    build(Path(args.out), make_story(m))
    print("Đã tạo", args.out)


if __name__ == "__main__":
    main()
