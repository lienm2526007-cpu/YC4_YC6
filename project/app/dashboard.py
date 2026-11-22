"""Dashboard tương tác (Streamlit).

Chạy:  streamlit run app/dashboard.py
Cho phép: chọn biến để xem biểu đồ, lọc theo cụm, thử dự đoán Đạt/Chưa đạt, tải báo cáo PDF.
"""
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "students_clean.csv"
REPORT = ROOT / "results" / "bao-cao-phan-tich.pdf"

st.set_page_config(page_title="Kết quả học tập", layout="wide")
st.title("Phân tích kết quả học tập học sinh")
st.caption("Dữ liệu: UCI Student Performance (CC BY 4.0). Chạy `python src/analysis.py` trước để tạo dữ liệu.")

if not DATA.exists():
    st.error("Chưa có data/processed/students_clean.csv – hãy chạy src/analysis.py.")
    st.stop()

df = pd.read_csv(DATA)
num_cols = ["G1", "G2", "G3", "studytime", "failures", "absences", "goout", "Walc", "Dalc", "freetime",
            "age", "Medu", "Fedu", "health"]

with st.sidebar:
    st.header("Bộ lọc")
    clusters = st.multiselect("Cụm", sorted(df["cluster"].unique()), default=sorted(df["cluster"].unique()))
    sex = st.multiselect("Giới tính", ["F", "M"], default=["F", "M"])
view = df[df["cluster"].isin(clusters) & df["sex"].isin(sex)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Số học sinh", len(view))
c2.metric("Tỷ lệ đạt", f"{view['pass'].mean() * 100:.1f}%" if len(view) else "–")
c3.metric("G3 trung bình", f"{view['G3'].mean():.2f}" if len(view) else "–")
c4.metric("Vắng trung bình", f"{view['absences'].mean():.1f}" if len(view) else "–")

tab1, tab2, tab3 = st.tabs(["Khám phá biến", "Dự đoán", "Báo cáo"])
with tab1:
    a, b = st.columns(2)
    x = a.selectbox("Biến trục X", num_cols, index=num_cols.index("absences"))
    y = b.selectbox("Biến trục Y", num_cols, index=num_cols.index("G3"))
    st.plotly_chart(px.scatter(view, x=x, y=y, color=view["cluster"].astype(str),
                               labels={"color": "Cụm"}, opacity=0.7), width="stretch")
    st.plotly_chart(px.histogram(view, x=x, color=view["pass"].map({1: "Đạt", 0: "Chưa đạt"}),
                                 barmode="overlay", labels={"color": "Kết quả"}), width="stretch")

with tab2:
    feats = ["G1", "studytime", "failures", "absences", "goout"]
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42).fit(df[feats], df["pass"])
    cols = st.columns(len(feats))
    vals = {
        "G1": cols[0].slider("Điểm kỳ 1", 0, 20, 10),
        "studytime": cols[1].slider("Tự học (1–4)", 1, 4, 2),
        "failures": cols[2].slider("Số lần trượt", 0, 3, 0),
        "absences": cols[3].slider("Số buổi vắng", 0, 30, 4),
        "goout": cols[4].slider("Mức đi chơi (1–5)", 1, 5, 3),
    }
    proba = clf.predict_proba(pd.DataFrame([vals]))[0][1]
    st.subheader(f"Xác suất đạt: {proba * 100:.0f}%")
    st.progress(float(proba))

with tab3:
    if REPORT.exists():
        st.download_button("Tải báo cáo PDF", REPORT.read_bytes(), file_name=REPORT.name, mime="application/pdf")
    else:
        st.info("Chạy `python src/build_report.py` để tạo báo cáo PDF.")
    st.download_button("Tải dữ liệu đang lọc (CSV)", view.to_csv(index=False).encode("utf-8"),
                       file_name="du-lieu-loc.csv", mime="text/csv")
