"""Phân tích dữ liệu UCI Student Performance.

Chạy:  python src/analysis.py [--data data/raw/student-por.csv]

Sinh ra:
  results/figures/*.png      các hình dùng cho báo cáo, poster, video
  results/metrics.json       các chỉ số của mô hình
  data/processed/students_clean.csv   dữ liệu đã làm sạch (kèm cụm)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
SEED = 42

# Tên tiếng Việt cho các biến dùng trong hình
VN = {
    "G1": "Điểm kỳ 1 (G1)",
    "G2": "Điểm kỳ 2 (G2)",
    "G3": "Điểm cuối kỳ (G3)",
    "studytime": "Thời gian tự học",
    "failures": "Số lần trượt trước",
    "absences": "Số buổi vắng",
    "goout": "Mức đi chơi",
    "Walc": "Rượu bia cuối tuần",
    "Dalc": "Rượu bia ngày thường",
    "higher": "Muốn học đại học",
    "schoolsup": "Phụ đạo ở trường",
    "internet": "Có Internet",
    "freetime": "Thời gian rảnh",
    "Medu": "Học vấn mẹ",
    "Fedu": "Học vấn cha",
    "age": "Tuổi",
    "sex_M": "Giới tính nam",
    "address_U": "Sống ở thành thị",
}

TREE_FEATURES = [
    "G1", "studytime", "failures", "absences", "goout", "Walc", "Dalc",
    "higher", "schoolsup", "internet", "freetime", "Medu", "Fedu", "age",
    "sex_M", "address_U",
]
CLUSTER_FEATURES = ["G1", "studytime", "failures", "absences", "goout", "Walc"]
REG_FEATURES = ["G1", "G2", "studytime", "failures", "absences"]


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    n_raw = len(df)
    for c in ["G1", "G2", "G3"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["G1", "G2", "G3"]).drop_duplicates()
    for c in ["higher", "schoolsup", "internet"]:
        df[c] = (df[c] == "yes").astype(int)
    df["sex_M"] = (df["sex"] == "M").astype(int)
    df["address_U"] = (df["address"] == "U").astype(int)
    df["pass"] = (df["G3"] >= 10).astype(int)  # thang điểm 0-20, đạt khi >= 10
    df.attrs["n_raw"] = n_raw
    return df


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=200)
    plt.close()


def eda_figures(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    plt.figure(figsize=(5, 3.6))
    sns.histplot(df["G3"], bins=21, color="#2E6FB7")
    plt.axvline(10, color="#C0392B", ls="--", label="Ngưỡng đạt = 10")
    plt.xlabel(VN["G3"]); plt.ylabel("Số sinh viên"); plt.legend()
    savefig("eda_g3_hist.png")

    plt.figure(figsize=(5, 3.6))
    rate = df.groupby("studytime")["pass"].mean() * 100
    sns.barplot(x=rate.index, y=rate.values, color="#2E6FB7")
    plt.xticks(range(4), ["<2 giờ", "2–5 giờ", "5–10 giờ", ">10 giờ"])
    plt.xlabel("Thời gian tự học mỗi tuần"); plt.ylabel("Tỷ lệ đạt (%)")
    plt.ylim(0, 100)
    savefig("eda_pass_by_studytime.png")

    plt.figure(figsize=(5, 3.6))
    sns.scatterplot(data=df, x="absences", y="G3", hue="pass", palette={1: "#2E6FB7", 0: "#C0392B"},
                    alpha=0.6, s=18)
    plt.xlabel(VN["absences"]); plt.ylabel(VN["G3"])
    plt.legend(title="Kết quả", labels=["Chưa đạt", "Đạt"])
    savefig("eda_absences_g3.png")

    plt.figure(figsize=(5, 3.6))
    cols = ["G1", "G2", "G3", "studytime", "failures", "absences", "goout", "Walc"]
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, annot_kws={"size": 6},
                cbar_kws={"shrink": 0.8})
    plt.xticks(fontsize=7); plt.yticks(fontsize=7)
    savefig("eda_corr_heatmap.png")


def decision_tree(df: pd.DataFrame) -> dict:
    X, y = df[TREE_FEATURES], df["pass"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=SEED)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    plt.figure(figsize=(11, 5.5))
    plot_tree(clf, feature_names=[VN[f] for f in TREE_FEATURES], class_names=["Chưa đạt", "Đạt"],
              filled=True, rounded=True, impurity=False, proportion=True, fontsize=8)
    savefig("tree.png")

    imp = pd.Series(clf.feature_importances_, index=TREE_FEATURES).sort_values(ascending=False)
    imp = imp[imp > 0]
    plt.figure(figsize=(5, 3.2))
    sns.barplot(x=imp.values, y=[VN[i] for i in imp.index], color="#2E6FB7")
    plt.xlabel("Mức độ quan trọng (Gini)")
    savefig("tree_importance.png")

    cm = confusion_matrix(yte, pred)
    plt.figure(figsize=(3.8, 3.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Chưa đạt", "Đạt"], yticklabels=["Chưa đạt", "Đạt"])
    plt.xlabel("Dự đoán"); plt.ylabel("Thực tế")
    savefig("tree_confusion.png")

    return {
        "train_size": int(len(Xtr)),
        "test_size": int(len(Xte)),
        "accuracy": round(accuracy_score(yte, pred), 4),
        "f1_fail_class": round(f1_score(yte, pred, pos_label=0), 4),
        "confusion_matrix": cm.tolist(),
        "importance": {k: round(float(v), 4) for k, v in imp.items()},
        "rules": export_text(clf, feature_names=TREE_FEATURES),
    }


def clustering(df: pd.DataFrame) -> dict:
    Xs = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    ks, inertia, sil = range(2, 9), [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Xs, km.labels_))

    plt.figure(figsize=(5, 3.4))
    plt.plot(list(ks), inertia, "o-", color="#2E6FB7")
    plt.xlabel("Số cụm k"); plt.ylabel("Tổng bình phương trong cụm")
    savefig("cluster_elbow.png")

    k = 3
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
    df["cluster"] = km.labels_
    # Đặt lại số thứ tự cụm theo điểm G1 trung bình tăng dần
    order = df.groupby("cluster")["G1"].mean().sort_values().index
    remap = {old: new for new, old in enumerate(order)}
    df["cluster"] = df["cluster"].map(remap)

    pcs = PCA(n_components=2, random_state=SEED).fit_transform(Xs)
    plt.figure(figsize=(5, 3.6))
    sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=df["cluster"], palette="Set1", s=18, alpha=0.75)
    plt.xlabel("Thành phần chính 1"); plt.ylabel("Thành phần chính 2"); plt.legend(title="Cụm")
    savefig("cluster_pca.png")

    profile = df.groupby("cluster")[CLUSTER_FEATURES + ["G3", "pass"]].mean().round(2)
    profile["n"] = df.groupby("cluster").size()
    return {
        "k": k,
        "silhouette_by_k": {int(kk): round(float(s), 4) for kk, s in zip(ks, sil)},
        "profile": profile.reset_index().to_dict(orient="records"),
    }


def regression(df: pd.DataFrame) -> dict:
    X, y = df[REG_FEATURES], df["G3"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED)
    reg = LinearRegression().fit(Xtr, ytr)
    pred = reg.predict(Xte)

    plt.figure(figsize=(4.4, 3.6))
    plt.scatter(yte, pred, s=16, alpha=0.6, color="#2E6FB7")
    lim = [0, 20]
    plt.plot(lim, lim, "--", color="#C0392B")
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("G3 thực tế"); plt.ylabel("G3 dự đoán")
    savefig("reg_pred_vs_true.png")

    return {
        "r2": round(r2_score(yte, pred), 4),
        "mae": round(mean_absolute_error(yte, pred), 4),
        "intercept": round(float(reg.intercept_), 4),
        "coef": {f: round(float(c), 4) for f, c in zip(REG_FEATURES, reg.coef_)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "raw" / "student-por.csv"))
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    df = load_and_clean(Path(args.data))
    eda_figures(df)
    metrics = {
        "dataset": Path(args.data).name,
        "n_raw": df.attrs["n_raw"],
        "n_clean": int(len(df)),
        "pass_rate": round(float(df["pass"].mean()), 4),
        "g3_mean": round(float(df["G3"].mean()), 2),
        "g3_std": round(float(df["G3"].std()), 2),
        "decision_tree": decision_tree(df),
        "clustering": clustering(df),
        "regression": regression(df),
    }
    df.to_csv(ROOT / "data" / "processed" / "students_clean.csv", index=False)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "decision_tree"}, ensure_ascii=False, indent=1))
    print("Cây quyết định:", {k: v for k, v in metrics["decision_tree"].items() if k != "rules"})
    print(metrics["decision_tree"]["rules"])


if __name__ == "__main__":
    main()
