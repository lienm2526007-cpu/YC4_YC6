"""Ve lai cac bieu do tu du lieu goc cua Realinho et al. (2022), Data 7(11):146.

Du lieu: https://doi.org/10.24432/C5MC89 (UCI) / https://doi.org/10.5281/zenodo.5777339, giay phep CC BY 4.0.
Chay: python ve_bieu_do.py   (tao hinh trong ../figures/ va in so lieu de doi chieu voi bai bao)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "figures"
FIG.mkdir(exist_ok=True)

df = pd.read_csv(HERE / "data.csv", sep=";")
df.columns = [c.strip() for c in df.columns]

VN = {"Graduate": "Tốt nghiệp", "Dropout": "Bỏ học", "Enrolled": "Còn học"}
COL = {"Graduate": "#1F4E8C", "Enrolled": "#9DB4D3", "Dropout": "#E67E22"}
ORDER = ["Dropout", "Enrolled", "Graduate"]

# Ma nganh theo bang A4 cua bai bao / mo ta bien tren UCI
COURSE = {
    33: "Công nghệ nhiên liệu sinh học", 171: "Thiết kế hoạt hình & đa phương tiện",
    8014: "Công tác xã hội (buổi tối)", 9003: "Nông học", 9070: "Thiết kế truyền thông",
    9085: "Điều dưỡng thú y", 9119: "Kỹ thuật tin học", 9130: "Chăn nuôi ngựa",
    9147: "Quản trị", 9238: "Công tác xã hội", 9254: "Du lịch", 9500: "Điều dưỡng",
    9556: "Vệ sinh răng miệng", 9670: "Quản trị quảng cáo & marketing",
    9773: "Báo chí & truyền thông", 9853: "Giáo dục cơ bản", 9991: "Quản trị (buổi tối)",
}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})
stats = {"n": int(len(df)), "n_cols": int(df.shape[1])}

# 1. Phan bo nhan (tuong ung Hinh 2 cua bai bao)
cnt = df["Target"].value_counts()
stats["target"] = {k: int(v) for k, v in cnt.items()}
fig, ax = plt.subplots(figsize=(6, 3.6))
keys = ["Graduate", "Dropout", "Enrolled"]
bars = ax.bar([VN[k] for k in keys], [cnt[k] for k in keys], color=[COL[k] for k in keys])
for b, k in zip(bars, keys):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
            f"{cnt[k]:,} ({cnt[k] / len(df) * 100:.0f}%)".replace(",", "."), ha="center", fontsize=11)
ax.set_ylabel("Số sinh viên")
ax.set_ylim(0, 2600)
fig.tight_layout(); fig.savefig(FIG / "rl_target.png", dpi=220); plt.close(fig)

# 2. Ket qua theo nganh (tuong ung Hinh 3a)
ct = pd.crosstab(df["Course"], df["Target"], normalize="index")[ORDER] * 100
ct = ct.sort_values("Graduate")
stats["by_course"] = {COURSE.get(int(i), str(i)): {k: round(float(v), 1) for k, v in r.items()}
                      for i, r in ct.iterrows()}
fig, ax = plt.subplots(figsize=(7.5, 6.2))
left = pd.Series(0.0, index=ct.index)
for k in ORDER:
    ax.barh([COURSE.get(int(i), str(i)) for i in ct.index], ct[k], left=left, color=COL[k], label=VN[k])
    left += ct[k]
ax.set_xlabel("Tỷ lệ sinh viên (%)")
ax.set_xlim(0, 100)
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False)
fig.tight_layout(); fig.savefig(FIG / "rl_by_course.png", dpi=220); plt.close(fig)

# 3. Ket qua theo nhom yeu to (tuong ung Hinh 3b)
groups = [
    ("Gender", {0: "Nữ", 1: "Nam"}, "Giới tính"),
    ("Scholarship holder", {1: "Có học bổng", 0: "Không học bổng"}, "Học bổng"),
    ("Tuition fees up to date", {1: "Đóng học phí đúng hạn", 0: "Nợ học phí"}, "Học phí"),
    ("Daytime/evening attendance", {1: "Học ban ngày", 0: "Học buổi tối"}, "Buổi học"),
]
rows, labels = [], []
stats["by_group"] = {}
for col, names, _ in groups:
    t = pd.crosstab(df[col], df["Target"], normalize="index")[ORDER] * 100
    for code, name in names.items():
        rows.append(t.loc[code]); labels.append(name)
        stats["by_group"][name] = {k: round(float(v), 1) for k, v in t.loc[code].items()}
g = pd.DataFrame(rows, index=labels)
fig, ax = plt.subplots(figsize=(7.5, 4.6))
left = pd.Series(0.0, index=g.index)
for k in ORDER:
    ax.barh(g.index, g[k], left=left, color=COL[k], label=VN[k])
    for y, (l, v) in enumerate(zip(left, g[k])):
        if k != "Enrolled" and v > 12:
            ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=10)
    left += g[k]
ax.invert_yaxis()
ax.set_xlabel("Tỷ lệ sinh viên (%)")
ax.set_xlim(0, 100)
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.4, 1.0), frameon=False)
fig.tight_layout(); fig.savefig(FIG / "rl_by_group.png", dpi=220); plt.close(fig)

# 4. Diem va so mon dat hoc ky 1 -> 2 theo nhan (minh hoa vi sao ket qua hoc ky la dac trung quan trong)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
for ax, (c1, c2, title) in zip(axes, [
        ("Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)", "Số học phần đạt (TB)"),
        ("Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)", "Điểm trung bình (thang 20)")]):
    m = df.groupby("Target")[[c1, c2]].mean()
    for k in ["Graduate", "Enrolled", "Dropout"]:
        ax.plot(["Học kỳ 1", "Học kỳ 2"], m.loc[k], "o-", color=COL[k], lw=2.5, label=VN[k])
    ax.set_title(title, fontsize=11)
    stats.setdefault("semester", {})[title] = {k: [round(float(x), 2) for x in m.loc[k]] for k in m.index}
axes[0].legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(FIG / "rl_semester.png", dpi=220); plt.close(fig)

(HERE / "so-lieu-doi-chieu.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: stats[k] for k in ["n", "n_cols", "target"]}, ensure_ascii=False))
for name in ["Điều dưỡng", "Công tác xã hội", "Công nghệ nhiên liệu sinh học", "Kỹ thuật tin học", "Chăn nuôi ngựa"]:
    print(name, stats["by_course"][name])
for k, v in stats["by_group"].items():
    print(k, v)
print(stats["semester"])
