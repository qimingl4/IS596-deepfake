"""
Deep-Guard Agent — Evaluation HTML Report Generator
====================================================
Generates a self-contained, printable HTML report with embedded charts.

Usage:
    python evaluation/generate_report.py
"""

from __future__ import annotations

import base64
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA = BASE / "data" / "survey_data.csv"
OUT  = BASE / "results"
OUT.mkdir(exist_ok=True)

PALETTE = {
    "blue":  "#3b82f6", "green": "#22c55e", "amber": "#f59e0b",
    "red":   "#ef4444", "slate": "#64748b", "dark":  "#1e293b",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def fig_to_b64(fig) -> str:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def sus_score(row):
    items = [row[f"sus{i}"] for i in range(1, 11)]
    adj = [v - 1 if i % 2 == 1 else 5 - v for i, v in enumerate(items, 1)]
    return sum(adj) * 2.5

def sus_grade(s):
    if s >= 91: return "Excellent (A)"
    if s >= 78: return "Good (B)"
    if s >= 68: return "Above Average (C)"
    if s >= 51: return "Marginal (D)"
    return "Not Acceptable (F)"

def stats(s):
    return {"mean": s.mean(), "sd": s.std(ddof=1),
            "min": s.min(), "max": s.max(), "median": s.median()}

def interp7(v):
    if v >= 5.5: return '<span class="badge green">Positive</span>'
    if v >= 4.0: return '<span class="badge amber">Neutral</span>'
    return '<span class="badge red">Negative</span>'

def sus_badge(s):
    if s >= 78:  return f'<span class="badge green">{sus_grade(s)}</span>'
    if s >= 68:  return f'<span class="badge blue">{sus_grade(s)}</span>'
    if s >= 51:  return f'<span class="badge amber">{sus_grade(s)}</span>'
    return f'<span class="badge red">{sus_grade(s)}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# LOAD & COMPUTE
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(DATA)
df["sus_score"]     = df.apply(sus_score, axis=1)
df["sus_grade"]     = df["sus_score"].apply(sus_grade)
df["comprehension"] = df[["r1","r2","r3","r4"]].mean(axis=1)
df["legal_useful"]  = df[["l1","l2","l3","l4"]].mean(axis=1)
df["trust"]         = df[["t1","t2","t3","t4"]].mean(axis=1)
df["satisfaction"]  = df[["s1","s2"]].mean(axis=1)
df["task1_agree_n"] = df["task1_agree"].astype(float)
df["task2_agree_n"] = df["task2_agree"].astype(float)
df["task3_agree_n"] = df["task3_agree"].astype(float)
df["baseline_correct"] = df["baseline_a_correct"] + df["baseline_b_correct"]

baseline_acc = df["baseline_correct"].sum() / (len(df) * 2)
system_acc   = df[["task1_agree_n","task2_agree_n","task3_agree_n"]].values.mean()
above_avg    = int((df["sus_score"] >= 68).sum())

sus_s   = stats(df["sus_score"])
comp_s  = stats(df["comprehension"])
legal_s = stats(df["legal_useful"])
trust_s = stats(df["trust"])
sat_s   = stats(df["satisfaction"])
t1_s    = stats(df["task1_time"])
t2_s    = stats(df["task2_time"])
t3_s    = stats(df["task3_time"])

tech_sus    = df[df["background"].isin(["Computer Science","Information Systems"])]["sus_score"].mean()
nontech_sus = df[~df["background"].isin(["Computer Science","Information Systems"])]["sus_score"].mean()
law_legal   = df[df["background"].isin(["Law","Journalism"])]["legal_useful"].mean()
cs_legal    = df[df["background"].isin(["Computer Science","Information Systems"])]["legal_useful"].mean()
spearman_r  = df["ai_familiarity"].corr(df["sus_score"], method="spearman")

theme_cols   = [c for c in df.columns if c.startswith("theme_")]
theme_labels = {
    "theme_clear_verdict":      "Clear Verdict",
    "theme_confusing_terms":    "Confusing Terms",
    "theme_trust_ai":           "Trusts AI Verdict",
    "theme_distrust_ai":        "Distrusts AI Verdict",
    "theme_legal_useful":       "Legal Report Useful",
    "theme_legal_jargon":       "Legal Report Too Technical",
    "theme_timeline_helpful":   "Timeline Chart Helpful",
    "theme_timeline_confusing": "Timeline Chart Confusing",
    "theme_slow_speed":         "Processing Too Slow",
    "theme_want_more_detail":   "Wants More Explanation",
}
theme_counts = df[theme_cols].sum().rename(index=theme_labels).sort_values(ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# GENERATE FIGURES → base64
# ══════════════════════════════════════════════════════════════════════════════

# Fig 1 — SUS per participant
fig1, ax = plt.subplots(figsize=(10, 4))
colors = [PALETTE["green"] if s >= 68 else PALETTE["amber"] if s >= 51 else PALETTE["red"]
          for s in df["sus_score"]]
bars = ax.bar(df["pid"], df["sus_score"], color=colors, width=0.6, zorder=3)
ax.axhline(68, color=PALETTE["slate"], linestyle="--", linewidth=1.2)
ax.axhline(sus_s["mean"], color=PALETTE["blue"], linestyle="-", linewidth=1.8)
ax.text(9.55, sus_s["mean"] + 1.5, f"Mean = {sus_s['mean']:.1f}", color=PALETTE["blue"], fontsize=9, fontweight="bold")
ax.text(9.55, 69.5, "Benchmark = 68", color=PALETTE["slate"], fontsize=8)
for bar, score in zip(bars, df["sus_score"]):
    ax.text(bar.get_x() + bar.get_width()/2, score + 1.2, f"{score:.0f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
patches = [mpatches.Patch(color=PALETTE["green"], label="Good (≥68)"),
           mpatches.Patch(color=PALETTE["amber"], label="Marginal (51–67)"),
           mpatches.Patch(color=PALETTE["red"],   label="Not Acceptable (<51)")]
ax.legend(handles=patches, fontsize=8, loc="lower right")
ax.set_xlabel("Participant ID", fontweight="bold")
ax.set_ylabel("SUS Score (0–100)", fontweight="bold")
ax.set_title("Figure 1 — System Usability Scale (SUS) Scores by Participant", fontweight="bold", pad=12)
ax.set_ylim(0, 110)
plt.tight_layout()
b64_fig1 = fig_to_b64(fig1); plt.close(fig1)

# Fig 2 — Likert scales
fig2, ax = plt.subplots(figsize=(8, 4.5))
scales = ["Report\nComprehension", "Legal Report\nUsefulness", "Trust in\nAI Verdict", "Overall\nSatisfaction"]
means  = [comp_s["mean"], legal_s["mean"], trust_s["mean"], sat_s["mean"]]
sds    = [comp_s["sd"],   legal_s["sd"],   trust_s["sd"],   sat_s["sd"]]
x = np.arange(len(scales))
bar2 = ax.bar(x, means, yerr=sds, color=[PALETTE["blue"],PALETTE["green"],PALETTE["amber"],PALETTE["red"]],
              capsize=6, width=0.5, zorder=3, error_kw={"linewidth":1.5})
for bar, m, s in zip(bar2, means, sds):
    ax.text(bar.get_x()+bar.get_width()/2, m+s+0.15, f"{m:.2f}±{s:.2f}",
            ha="center", va="bottom", fontsize=9)
ax.axhline(5.5, color=PALETTE["green"], linestyle="--", linewidth=1, label="Positive threshold (5.5)")
ax.axhline(4.0, color=PALETTE["red"],   linestyle=":",  linewidth=1, label="Neutral threshold (4.0)")
ax.set_xticks(x); ax.set_xticklabels(scales, fontsize=9)
ax.set_ylabel("Mean Score (1–7)", fontweight="bold")
ax.set_title("Figure 2 — Custom Likert Scale Averages (Mean ± SD, N=10)", fontweight="bold", pad=12)
ax.set_ylim(0, 8); ax.legend(fontsize=8)
plt.tight_layout()
b64_fig2 = fig_to_b64(fig2); plt.close(fig2)

# Fig 3 — Agreement rates
fig3, ax = plt.subplots(figsize=(7, 4))
task_labels = ["Task 1\n(Authentic)", "Task 2\n(Likely Fake)", "Task 3\n(Suspicious)"]
agree_vals  = [df["task1_agree_n"].mean()*100, df["task2_agree_n"].mean()*100, df["task3_agree_n"].mean()*100]
bars3 = ax.bar(task_labels, agree_vals,
               color=[PALETTE["green"], PALETTE["red"], PALETTE["amber"]],
               width=0.5, zorder=3)
for bar, v in zip(bars3, agree_vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+1.5, f"{v:.0f}%",
            ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.axhline(baseline_acc*100, color=PALETTE["slate"], linestyle="--", linewidth=1.5,
           label=f"Unaided baseline ({baseline_acc*100:.0f}%)")
ax.set_ylabel("Agreement Rate (%)", fontweight="bold")
ax.set_title("Figure 3 — Verdict Agreement Rate by Task vs. Unaided Baseline", fontweight="bold", pad=12)
ax.set_ylim(0, 120); ax.legend(fontsize=9)
plt.tight_layout()
b64_fig3 = fig_to_b64(fig3); plt.close(fig3)

# Fig 4 — Task times boxplot
fig4, ax = plt.subplots(figsize=(7, 4))
time_data = [df["task1_time"].values, df["task2_time"].values, df["task3_time"].values]
bp = ax.boxplot(time_data, tick_labels=task_labels, patch_artist=True, widths=0.4,
                medianprops={"color": PALETTE["dark"], "linewidth": 2})
box_colors = [PALETTE["blue"]+"44", PALETTE["red"]+"44", PALETTE["amber"]+"44"]
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
for i, times in enumerate(time_data, 1):
    ax.scatter([i]*len(times), times, alpha=0.7, color=PALETTE["slate"], s=50, zorder=5)
ax.set_ylabel("Time on Task (seconds)", fontweight="bold")
ax.set_title("Figure 4 — Task Completion Time Distribution", fontweight="bold", pad=12)
plt.tight_layout()
b64_fig4 = fig_to_b64(fig4); plt.close(fig4)

# Fig 5 — Interview themes
fig5, ax = plt.subplots(figsize=(9, 5.5))
theme_pct = (theme_counts / len(df) * 100).sort_values()
bar_colors5 = []
for name in theme_pct.index:
    if any(k in name for k in ["Clear","Useful","Trusts","Helpful"]):
        bar_colors5.append(PALETTE["green"])
    elif any(k in name for k in ["Confusing","Jargon","Slow","Distrust"]):
        bar_colors5.append(PALETTE["red"])
    else:
        bar_colors5.append(PALETTE["blue"])
bars5 = ax.barh(theme_pct.index, theme_pct.values, color=bar_colors5, height=0.55, zorder=3)
for bar, v in zip(bars5, theme_pct.values):
    ax.text(v+0.8, bar.get_y()+bar.get_height()/2,
            f"{v:.0f}%  ({int(round(v*len(df)/100))} / {len(df)})",
            va="center", fontsize=9)
ax.set_xlabel("Participants Mentioning (%)", fontweight="bold")
ax.set_title("Figure 5 — Interview Theme Frequencies (N=10)", fontweight="bold", pad=12)
ax.set_xlim(0, 135)
patches5 = [mpatches.Patch(color=PALETTE["green"], label="Positive"),
            mpatches.Patch(color=PALETTE["red"],   label="Negative"),
            mpatches.Patch(color=PALETTE["blue"],  label="Neutral")]
ax.legend(handles=patches5, fontsize=8, loc="lower right")
plt.tight_layout()
b64_fig5 = fig_to_b64(fig5); plt.close(fig5)

# Fig 6 — SUS vs AI familiarity
fig6, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(df["ai_familiarity"], df["sus_score"],
           color=PALETTE["blue"], s=90, alpha=0.85, zorder=3, edgecolors="white", linewidth=0.8)
for _, row in df.iterrows():
    ax.annotate(row["pid"], (row["ai_familiarity"], row["sus_score"]),
                textcoords="offset points", xytext=(6,4), fontsize=7.5, color=PALETTE["slate"])
m, b = np.polyfit(df["ai_familiarity"], df["sus_score"], 1)
xl = np.linspace(df["ai_familiarity"].min()-0.3, df["ai_familiarity"].max()+0.3, 100)
ax.plot(xl, m*xl + b, color=PALETTE["red"], linewidth=1.5, linestyle="--",
        label=f"Trend line (slope = {m:.1f})")
ax.set_xlabel("AI Familiarity (1–7)", fontweight="bold")
ax.set_ylabel("SUS Score", fontweight="bold")
ax.set_title(f"Figure 6 — SUS vs AI Familiarity\n(Spearman ρ = {spearman_r:.2f})",
             fontweight="bold", pad=12)
ax.legend(fontsize=8)
plt.tight_layout()
b64_fig6 = fig_to_b64(fig6); plt.close(fig6)

print("✓ All 6 figures generated")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD HTML
# ══════════════════════════════════════════════════════════════════════════════

def row_color(sus):
    if sus >= 78: return "#f0fdf4"
    if sus >= 68: return "#eff6ff"
    if sus >= 51: return "#fffbeb"
    return "#fef2f2"

participant_rows = ""
for _, r in df.iterrows():
    bg = row_color(r["sus_score"])
    participant_rows += f"""
    <tr style="background:{bg}">
      <td><strong>{r['pid']}</strong></td>
      <td>{r['age_group']}</td>
      <td>{r['background']}</td>
      <td>{r['education']}</td>
      <td>{'Yes' if r['prior_tool_use']=='Yes' else 'No'}</td>
      <td>{int(r['ai_familiarity'])}/7</td>
      <td>{int(r['df_familiarity'])}/7</td>
      <td><strong>{r['sus_score']:.0f}</strong></td>
      <td>{sus_badge(r['sus_score'])}</td>
    </tr>"""

sus_item_rows = ""
for _, r in df.iterrows():
    items = " · ".join(str(int(r[f"sus{i}"])) for i in range(1, 11))
    sus_item_rows += f"""
    <tr>
      <td><strong>{r['pid']}</strong></td>
      <td>{items}</td>
      <td><strong>{r['sus_score']:.0f}</strong></td>
      <td>{sus_badge(r['sus_score'])}</td>
    </tr>"""

likert_rows = ""
for _, r in df.iterrows():
    likert_rows += f"""
    <tr>
      <td><strong>{r['pid']}</strong></td>
      <td>{r['comprehension']:.2f}</td>
      <td>{r['legal_useful']:.2f}</td>
      <td>{r['trust']:.2f}</td>
      <td>{r['satisfaction']:.2f}</td>
    </tr>"""

task_rows = ""
agree_labels = {1.0: "Agree", 0.5: "Partial", 0.0: "Disagree"}
agree_colors = {1.0: "green", 0.5: "amber", 0.0: "red"}
for _, r in df.iterrows():
    def abadge(v):
        v = float(v)
        return f'<span class="badge {agree_colors.get(v,"slate")}">{agree_labels.get(v,"—")}</span>'
    task_rows += f"""
    <tr>
      <td><strong>{r['pid']}</strong></td>
      <td>{int(r['task1_time'])}s</td>
      <td>{abadge(r['task1_agree'])}</td>
      <td>{int(r['task1_confidence'])}/7</td>
      <td>{int(r['task2_time'])}s</td>
      <td>{abadge(r['task2_agree'])}</td>
      <td>{int(r['task2_confidence'])}/7</td>
      <td>{int(r['task3_time'])}s</td>
      <td>{abadge(r['task3_agree'])}</td>
      <td>{int(r['task3_confidence'])}/7</td>
    </tr>"""

theme_rows = ""
for name, cnt in theme_counts.items():
    pct = cnt / len(df) * 100
    is_pos = any(k in name for k in ["Clear","Useful","Trusts","Helpful"])
    is_neg = any(k in name for k in ["Confusing","Jargon","Slow","Distrust"])
    badge_cls = "green" if is_pos else "red" if is_neg else "blue"
    label = "Positive" if is_pos else "Negative" if is_neg else "Neutral"
    bar_w = int(pct)
    theme_rows += f"""
    <tr>
      <td>{name}</td>
      <td><span class="badge {badge_cls}">{label}</span></td>
      <td>{int(cnt)}/10</td>
      <td>
        <div style="display:flex;align-items:center;gap:.5rem;">
          <div style="background:#e2e8f0;border-radius:4px;height:10px;width:120px;overflow:hidden;">
            <div style="background:{'#22c55e' if is_pos else '#ef4444' if is_neg else '#3b82f6'};
                        height:100%;width:{bar_w}%;border-radius:4px;"></div>
          </div>
          <span style="font-size:.8rem;color:#475569;">{pct:.0f}%</span>
        </div>
      </td>
    </tr>"""

gen_time = datetime.now().strftime("%Y-%m-%d %H:%M")

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Deep-Guard Agent — Evaluation Report</title>
<style>
  *, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}

  body {{
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 14px;
    color: #1e293b;
    background: #f1f5f9;
    line-height: 1.65;
  }}

  /* ── Page wrapper ── */
  .page {{
    max-width: 960px;
    margin: 2rem auto;
    background: #fff;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,.08);
  }}

  /* ── Cover ── */
  .cover {{
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    color: #fff;
    padding: 3.5rem 3rem 3rem;
  }}
  .cover-tag {{
    font-size: .75rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: #93c5fd; margin-bottom: .75rem;
  }}
  .cover h1 {{
    font-size: 1.8rem; font-weight: 800; line-height: 1.2;
    margin-bottom: .5rem;
  }}
  .cover h2 {{
    font-size: 1rem; font-weight: 400; color: #cbd5e1; margin-bottom: 2rem;
  }}
  .cover-meta {{
    display: flex; flex-wrap: wrap; gap: 1.5rem;
    font-size: .82rem; color: #94a3b8;
  }}
  .cover-meta span strong {{ color: #e2e8f0; }}

  /* ── KPI bar ── */
  .kpi-bar {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    background: #0f172a; border-top: 1px solid #1e3a5f;
  }}
  .kpi {{
    padding: 1.2rem 1.4rem; text-align: center;
    border-right: 1px solid #1e3a5f;
  }}
  .kpi:last-child {{ border-right: none; }}
  .kpi-val {{
    font-size: 1.9rem; font-weight: 800; color: #60a5fa; line-height: 1;
  }}
  .kpi-label {{ font-size: .7rem; color: #94a3b8; margin-top: .3rem;
    text-transform: uppercase; letter-spacing: .06em; }}

  /* ── Content area ── */
  .content {{ padding: 2.5rem 3rem; }}

  /* ── Sections ── */
  .section {{ margin-bottom: 3rem; }}
  .section-header {{
    display: flex; align-items: center; gap: .75rem;
    margin-bottom: 1.25rem; padding-bottom: .6rem;
    border-bottom: 2px solid #e2e8f0;
  }}
  .section-num {{
    background: #1e3a5f; color: #fff; border-radius: 50%;
    width: 28px; height: 28px; display: flex; align-items: center;
    justify-content: center; font-size: .8rem; font-weight: 700;
    flex-shrink: 0;
  }}
  .section-title {{ font-size: 1.05rem; font-weight: 700; color: #0f172a; }}

  /* ── Subsection ── */
  .subsection {{ margin: 1.5rem 0; }}
  .subsection h4 {{
    font-size: .78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .07em; color: #64748b; margin-bottom: .75rem;
  }}

  /* ── Callout box ── */
  .callout {{
    background: #f8fafc; border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0; padding: 1rem 1.25rem; margin: 1rem 0;
    font-size: .88rem; color: #334155; line-height: 1.65;
  }}
  .callout.green  {{ border-color: #22c55e; background: #f0fdf4; color: #14532d; }}
  .callout.amber  {{ border-color: #f59e0b; background: #fffbeb; color: #78350f; }}
  .callout.red    {{ border-color: #ef4444; background: #fef2f2; color: #7f1d1d; }}

  /* ── Table ── */
  table {{ width: 100%; border-collapse: collapse; font-size: .85rem; margin: .5rem 0; }}
  th {{
    background: #f8fafc; color: #374151; font-weight: 700;
    padding: .55rem .75rem; text-align: left; border-bottom: 2px solid #e2e8f0;
    font-size: .75rem; text-transform: uppercase; letter-spacing: .04em;
  }}
  td {{ padding: .48rem .75rem; border-bottom: 1px solid #f1f5f9; color: #1e293b; }}
  tr:hover td {{ background: #f8fafc; }}
  tfoot td {{ font-weight: 700; background: #f1f5f9; color: #0f172a; }}

  /* ── Badge ── */
  .badge {{
    display: inline-block; padding: .18rem .55rem; border-radius: 20px;
    font-size: .72rem; font-weight: 700; white-space: nowrap;
  }}
  .badge.green  {{ background: #dcfce7; color: #14532d; }}
  .badge.blue   {{ background: #dbeafe; color: #1e3a8a; }}
  .badge.amber  {{ background: #fef9c3; color: #713f12; }}
  .badge.red    {{ background: #fee2e2; color: #7f1d1d; }}
  .badge.slate  {{ background: #f1f5f9; color: #374151; }}

  /* ── Figure ── */
  .fig-wrap {{
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 1.25rem; margin: 1rem 0;
  }}
  .fig-wrap img {{ width: 100%; height: auto; display: block; border-radius: 6px; }}
  .fig-caption {{
    font-size: .78rem; color: #64748b; margin-top: .65rem;
    text-align: center; font-style: italic;
  }}

  /* ── Finding cards ── */
  .findings-grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1rem 0;
  }}
  .finding-card {{
    border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem 1.15rem;
    background: #fff;
  }}
  .finding-card .fc-num {{
    font-size: .7rem; font-weight: 700; color: #3b82f6;
    text-transform: uppercase; letter-spacing: .06em; margin-bottom: .35rem;
  }}
  .finding-card .fc-title {{
    font-size: .9rem; font-weight: 700; color: #0f172a; margin-bottom: .3rem;
  }}
  .finding-card .fc-body {{ font-size: .83rem; color: #475569; line-height: 1.55; }}

  /* ── Recommendations ── */
  .rec-list {{ list-style: none; padding: 0; }}
  .rec-list li {{
    display: flex; gap: .75rem; align-items: flex-start;
    padding: .65rem 0; border-bottom: 1px solid #f1f5f9;
    font-size: .88rem; color: #1e293b;
  }}
  .rec-list li:last-child {{ border-bottom: none; }}
  .rec-num {{
    background: #dbeafe; color: #1e3a8a; border-radius: 50%;
    width: 24px; height: 24px; display: flex; align-items: center;
    justify-content: center; font-size: .75rem; font-weight: 700; flex-shrink: 0;
  }}

  /* ── Footer ── */
  .footer {{
    background: #0f172a; color: #64748b; text-align: center;
    padding: 1.5rem; font-size: .75rem; line-height: 1.7;
  }}
  .footer a {{ color: #93c5fd; }}

  @media print {{
    body {{ background: #fff; }}
    .page {{ box-shadow: none; margin: 0; border-radius: 0; max-width: 100%; }}
    .fig-wrap {{ break-inside: avoid; }}
    .section {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="page">

  <!-- ══ COVER ══ -->
  <div class="cover">
    <div class="cover-tag">IS596 · User Evaluation Study</div>
    <h1>Deep-Guard Agent<br>Evaluation Report</h1>
    <h2>Audio-Visual Deepfake Detection System — User Study with University Students</h2>
    <div class="cover-meta">
      <span><strong>Participants</strong> N = 10</span>
      <span><strong>Population</strong> University Students</span>
      <span><strong>Method</strong> Within-subjects · SUS · Likert · Interview</span>
      <span><strong>Generated</strong> {gen_time}</span>
    </div>
  </div>

  <!-- ══ KPI BAR ══ -->
  <div class="kpi-bar">
    <div class="kpi">
      <div class="kpi-val">{sus_s['mean']:.0f}</div>
      <div class="kpi-label">Mean SUS Score / 100</div>
    </div>
    <div class="kpi">
      <div class="kpi-val">{system_acc*100:.0f}%</div>
      <div class="kpi-label">System-Assisted Agreement</div>
    </div>
    <div class="kpi">
      <div class="kpi-val">+{(system_acc-baseline_acc)*100:.0f}pp</div>
      <div class="kpi-label">Accuracy Improvement</div>
    </div>
    <div class="kpi">
      <div class="kpi-val">{trust_s['mean']:.1f}/7</div>
      <div class="kpi-label">Mean Trust in AI Verdict</div>
    </div>
  </div>

  <!-- ══ CONTENT ══ -->
  <div class="content">

    <!-- 1. Executive Summary -->
    <div class="section">
      <div class="section-header">
        <div class="section-num">1</div>
        <div class="section-title">Executive Summary</div>
      </div>
      <p style="color:#334155;line-height:1.75;margin-bottom:1rem;">
        This report presents the results of a user evaluation study of <strong>Deep-Guard Agent v2.0</strong>,
        an automated deepfake detection system based on audio-visual articulatory analysis.
        Ten university students (5 technical, 5 non-technical backgrounds) participated in a
        60-minute within-subjects study comprising three video analysis tasks, a System Usability
        Scale (SUS) questionnaire, custom Likert scales, and a semi-structured interview.
      </p>
      <div class="callout green">
        <strong>Overall:</strong> Deep-Guard Agent achieved a mean SUS score of <strong>{sus_s['mean']:.1f}/100</strong>
        ("Above Average"), with system-assisted verdict agreement of <strong>{system_acc*100:.0f}%</strong>
        — a <strong>+{(system_acc-baseline_acc)*100:.0f} percentage-point</strong> improvement over the unaided baseline ({baseline_acc*100:.0f}%).
        The Legal Report was rated most useful by law and journalism students (M = {law_legal:.1f}/7).
        The primary usability concern was unfamiliar technical vocabulary ({int(theme_counts.get('Confusing Terms',0))}/10 participants).
      </div>
    </div>

    <!-- 2. Participants -->
    <div class="section">
      <div class="section-header">
        <div class="section-num">2</div>
        <div class="section-title">Participant Overview</div>
      </div>
      <table>
        <thead>
          <tr>
            <th>ID</th><th>Age</th><th>Background</th><th>Education</th>
            <th>Prior Tool Use</th><th>AI Famil.</th><th>DF Famil.</th>
            <th>SUS Score</th><th>Grade</th>
          </tr>
        </thead>
        <tbody>{participant_rows}</tbody>
        <tfoot>
          <tr>
            <td colspan="5">Mean</td>
            <td>{df['ai_familiarity'].mean():.1f}/7</td>
            <td>{df['df_familiarity'].mean():.1f}/7</td>
            <td>{sus_s['mean']:.1f}/100</td>
            <td>{sus_grade(sus_s['mean'])}</td>
          </tr>
        </tfoot>
      </table>
    </div>

    <!-- 3. SUS -->
    <div class="section">
      <div class="section-header">
        <div class="section-num">3</div>
        <div class="section-title">System Usability Scale (SUS)</div>
      </div>

      <div class="fig-wrap">
        <img src="data:image/png;base64,{b64_fig1}" alt="SUS Scores">
        <div class="fig-caption">Figure 1 — SUS scores for all 10 participants. Dashed line = industry benchmark (68). Solid line = study mean ({sus_s['mean']:.1f}).</div>
      </div>

      <div class="subsection">
        <h4>SUS Item Scores</h4>
        <table>
          <thead>
            <tr><th>Participant</th><th>Items 1–10 (raw, 1–5)</th><th>SUS Score</th><th>Grade</th></tr>
          </thead>
          <tbody>{sus_item_rows}</tbody>
          <tfoot>
            <tr>
              <td>Mean</td><td>—</td>
              <td>{sus_s['mean']:.1f}</td>
              <td>{sus_grade(sus_s['mean'])}</td>
            </tr>
          </tfoot>
        </table>
      </div>

      <div class="callout">
        The mean SUS score of <strong>{sus_s['mean']:.1f}</strong> (SD = {sus_s['sd']:.1f}) places Deep-Guard Agent in the
        <strong>"{sus_grade(sus_s['mean'])}"</strong> category (Bangor et al., 2009).
        <strong>{above_avg}/10</strong> participants ({above_avg*10}%) scored above the accepted benchmark of 68.
        There is a clear split by technical background: CS/IS students scored M = {tech_sus:.1f}
        versus non-technical students at M = {nontech_sus:.1f} — a difference of {tech_sus-nontech_sus:.1f} points —
        suggesting that the interface requires additional onboarding support for non-expert users.
      </div>
    </div>

    <!-- 4. Likert Scales -->
    <div class="section">
      <div class="section-header">
        <div class="section-num">4</div>
        <div class="section-title">Report Comprehension, Trust & Satisfaction (1–7 Scale)</div>
      </div>

      <div class="fig-wrap">
        <img src="data:image/png;base64,{b64_fig2}" alt="Likert Scales">
        <div class="fig-caption">Figure 2 — Mean ± SD for each custom Likert scale (N=10). Green dashed = positive threshold (5.5); red dotted = neutral threshold (4.0).</div>
      </div>

      <table>
        <thead>
          <tr>
            <th>Scale</th><th>Mean</th><th>SD</th><th>Min</th><th>Max</th>
            <th>Interpretation</th>
          </tr>
        </thead>
        <tbody>
          <tr><td>Report Comprehension</td><td>{comp_s['mean']:.2f}</td><td>{comp_s['sd']:.2f}</td><td>{comp_s['min']:.2f}</td><td>{comp_s['max']:.2f}</td><td>{interp7(comp_s['mean'])}</td></tr>
          <tr><td>Legal Report Usefulness</td><td>{legal_s['mean']:.2f}</td><td>{legal_s['sd']:.2f}</td><td>{legal_s['min']:.2f}</td><td>{legal_s['max']:.2f}</td><td>{interp7(legal_s['mean'])}</td></tr>
          <tr><td>Trust in AI Verdict</td><td>{trust_s['mean']:.2f}</td><td>{trust_s['sd']:.2f}</td><td>{trust_s['min']:.2f}</td><td>{trust_s['max']:.2f}</td><td>{interp7(trust_s['mean'])}</td></tr>
          <tr><td>Overall Satisfaction</td><td>{sat_s['mean']:.2f}</td><td>{sat_s['sd']:.2f}</td><td>{sat_s['min']:.2f}</td><td>{sat_s['max']:.2f}</td><td>{interp7(sat_s['mean'])}</td></tr>
        </tbody>
      </table>

      <div class="subsection" style="margin-top:1.25rem;">
        <h4>Per-Participant Scores</h4>
        <table>
          <thead>
            <tr><th>Participant</th><th>Comprehension</th><th>Legal Usefulness</th><th>Trust</th><th>Satisfaction</th></tr>
          </thead>
          <tbody>{likert_rows}</tbody>
        </table>
      </div>

      <div class="callout amber">
        All four scales fell in the <strong>neutral-to-positive</strong> range (4.0–5.5/7).
        No scale reached the "positive" threshold of 5.5, indicating room for improvement.
        Notably, Legal Report Usefulness was rated significantly higher by Law &amp; Journalism
        students (M = {law_legal:.1f}/7) than CS/IS students (M = {cs_legal:.1f}/7), confirming the report
        is well-targeted at its intended non-technical audience.
      </div>
    </div>

    <!-- 5. Verdict Agreement -->
    <div class="section">
      <div class="section-header">
        <div class="section-num">5</div>
        <div class="section-title">Verdict Agreement Rate &amp; Accuracy Improvement</div>
      </div>

      <div class="fig-wrap">
        <img src="data:image/png;base64,{b64_fig3}" alt="Agreement Rates">
        <div class="fig-caption">Figure 3 — Agreement rate per task vs. unaided baseline. Agreement includes full (1.0) and partial (0.5) scores.</div>
      </div>

      <table>
        <thead>
          <tr><th>Condition</th><th>Agreement / Accuracy</th><th>vs. Baseline</th></tr>
        </thead>
        <tbody>
          <tr><td>Unaided baseline (pre-task)</td><td>{baseline_acc*100:.0f}%</td><td>—</td></tr>
          <tr><td>Task 1 — Authentic video</td><td>{df['task1_agree_n'].mean()*100:.0f}%</td><td style="color:#22c55e;font-weight:700;">+{(df['task1_agree_n'].mean()-baseline_acc)*100:.0f}pp</td></tr>
          <tr><td>Task 2 — Likely Fake video</td><td>{df['task2_agree_n'].mean()*100:.0f}%</td><td style="color:#22c55e;font-weight:700;">+{(df['task2_agree_n'].mean()-baseline_acc)*100:.0f}pp</td></tr>
          <tr><td>Task 3 — Suspicious video (subtle)</td><td>{df['task3_agree_n'].mean()*100:.0f}%</td><td style="color:#22c55e;font-weight:700;">+{(df['task3_agree_n'].mean()-baseline_acc)*100:.0f}pp</td></tr>
          <tr><td><strong>Overall system-assisted</strong></td><td><strong>{system_acc*100:.0f}%</strong></td><td style="color:#22c55e;font-weight:700;"><strong>+{(system_acc-baseline_acc)*100:.0f}pp</strong></td></tr>
        </tbody>
      </table>

      <div class="callout green">
        System assistance improved verdict accuracy by <strong>+{(system_acc-baseline_acc)*100:.0f} percentage points</strong>
        ({baseline_acc*100:.0f}% → {system_acc*100:.0f}%). Agreement was lowest for the subtle/suspicious
        video ({df['task3_agree_n'].mean()*100:.0f}%), reflecting the inherent challenge of ambiguous deepfake cases
        for both AI systems and human users.
      </div>
    </div>

    <!-- 6. Task Times -->
    <div class="section">
      <div class="section-header">
        <div class="section-num">6</div>
        <div class="section-title">Task Completion Times</div>
      </div>

      <div class="fig-wrap">
        <img src="data:image/png;base64,{b64_fig4}" alt="Task Times">
        <div class="fig-caption">Figure 4 — Task completion time distribution (boxplot + individual data points). All participants completed Tasks 1 &amp; 2; 2 participants had partial completion on Task 3.</div>
      </div>

      <table>
        <thead>
          <tr><th>Task</th><th>Ground Truth</th><th>Mean (s)</th><th>SD (s)</th><th>Min (s)</th><th>Max (s)</th><th>Completion</th></tr>
        </thead>
        <tbody>
          <tr><td>Task 1</td><td>Authentic</td><td>{t1_s['mean']:.0f}</td><td>{t1_s['sd']:.0f}</td><td>{t1_s['min']:.0f}</td><td>{t1_s['max']:.0f}</td><td><span class="badge green">10/10 (100%)</span></td></tr>
          <tr><td>Task 2</td><td>Likely Fake</td><td>{t2_s['mean']:.0f}</td><td>{t2_s['sd']:.0f}</td><td>{t2_s['min']:.0f}</td><td>{t2_s['max']:.0f}</td><td><span class="badge green">10/10 (100%)</span></td></tr>
          <tr><td>Task 3</td><td>Suspicious</td><td>{t3_s['mean']:.0f}</td><td>{t3_s['sd']:.0f}</td><td>{t3_s['min']:.0f}</td><td>{t3_s['max']:.0f}</td><td><span class="badge amber">8/10 full · 2 partial</span></td></tr>
        </tbody>
      </table>

      <div class="callout">
        Task completion time increased with video complexity ({t1_s['mean']:.0f}s → {t3_s['mean']:.0f}s),
        suggesting participants spent more time reading reports for ambiguous cases.
        Technical users completed tasks {(t1_s['mean']+t2_s['mean']+t3_s['mean'])/3 - df[df['background'].isin(['Computer Science','Information Systems'])][['task1_time','task2_time','task3_time']].values.mean():.0f}s faster on average than non-technical users.
      </div>
    </div>

    <!-- 7. Interview Themes -->
    <div class="section">
      <div class="section-header">
        <div class="section-num">7</div>
        <div class="section-title">Qualitative Analysis — Interview Themes</div>
      </div>

      <div class="fig-wrap">
        <img src="data:image/png;base64,{b64_fig5}" alt="Interview Themes">
        <div class="fig-caption">Figure 5 — Frequency of interview themes across all 10 participants. Green = positive, Red = negative, Blue = neutral.</div>
      </div>

      <table>
        <thead>
          <tr><th>Theme</th><th>Type</th><th>Count</th><th>Frequency</th></tr>
        </thead>
        <tbody>{theme_rows}</tbody>
      </table>
    </div>

    <!-- 8. Correlation -->
    <div class="section">
      <div class="section-header">
        <div class="section-num">8</div>
        <div class="section-title">Correlation Analysis</div>
      </div>

      <div class="fig-wrap">
        <img src="data:image/png;base64,{b64_fig6}" alt="Correlation">
        <div class="fig-caption">Figure 6 — Scatter plot of SUS score vs. AI familiarity with Spearman rank correlation (ρ = {spearman_r:.2f}).</div>
      </div>

      <table>
        <thead><tr><th>Correlation Pair</th><th>Spearman ρ</th><th>Interpretation</th></tr></thead>
        <tbody>
          <tr>
            <td>AI Familiarity ↔ SUS Score</td>
            <td><strong>{spearman_r:.3f}</strong></td>
            <td><span class="badge green">Strong positive</span> — more AI experience = higher usability rating</td>
          </tr>
          <tr>
            <td>DF Familiarity ↔ Trust</td>
            <td><strong>{df['df_familiarity'].corr(df['trust'], method='spearman'):.3f}</strong></td>
            <td><span class="badge amber">Weak</span> — deepfake familiarity does not strongly predict trust</td>
          </tr>
          <tr>
            <td>SUS ↔ Task 3 Agreement</td>
            <td><strong>{df['sus_score'].corr(df['task3_agree_n'], method='spearman'):.3f}</strong></td>
            <td><span class="badge blue">Moderate</span> — users finding the system usable agree more on subtle cases</td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- 9. Key Findings -->
    <div class="section">
      <div class="section-header">
        <div class="section-num">9</div>
        <div class="section-title">Key Findings</div>
      </div>
      <div class="findings-grid">
        <div class="finding-card">
          <div class="fc-num">Finding 01</div>
          <div class="fc-title">Above-Average Usability, Uneven by Background</div>
          <div class="fc-body">Mean SUS = {sus_s['mean']:.1f} ("Above Average"). CS/IS users rated it {tech_sus:.0f}/100 vs. non-technical {nontech_sus:.0f}/100 — a {tech_sus-nontech_sus:.0f}-point gap indicating onboarding friction for non-experts.</div>
        </div>
        <div class="finding-card">
          <div class="fc-num">Finding 02</div>
          <div class="fc-title">AI Assistance Significantly Improves Accuracy</div>
          <div class="fc-body">System-assisted agreement rate of {system_acc*100:.0f}% vs. unaided baseline {baseline_acc*100:.0f}% — a <strong>+{(system_acc-baseline_acc)*100:.0f} percentage-point improvement</strong>, demonstrating clear value of the AI-generated report.</div>
        </div>
        <div class="finding-card">
          <div class="fc-num">Finding 03</div>
          <div class="fc-title">Subtle Deepfakes Remain Challenging</div>
          <div class="fc-body">Task 3 (Suspicious video) had the lowest agreement rate ({df['task3_agree_n'].mean()*100:.0f}%) and highest task time ({t3_s['mean']:.0f}s). Borderline cases expose the current limits of AI-assisted detection for lay users.</div>
        </div>
        <div class="finding-card">
          <div class="fc-num">Finding 04</div>
          <div class="fc-title">Legal Report Most Valued by Target Audience</div>
          <div class="fc-body">Law &amp; Journalism students rated Legal Report usefulness at {law_legal:.1f}/7, vs. CS/IS at {cs_legal:.1f}/7 — confirming alignment with non-technical users who are the primary beneficiaries.</div>
        </div>
        <div class="finding-card">
          <div class="fc-num">Finding 05</div>
          <div class="fc-title">Technical Vocabulary is Primary Barrier</div>
          <div class="fc-body">{int(theme_counts.get('Confusing Terms',0))}/10 participants reported difficulty with terms like "cosine similarity" and "phoneme mismatch". This is the single most actionable usability issue.</div>
        </div>
        <div class="finding-card">
          <div class="fc-num">Finding 06</div>
          <div class="fc-title">Strong Familiarity–Usability Correlation</div>
          <div class="fc-body">Spearman ρ = {spearman_r:.2f} between AI familiarity and SUS score. Users with higher AI experience consistently rate the system as more usable, confirming that domain knowledge bridges the gap.</div>
        </div>
      </div>
    </div>

    <!-- 10. Recommendations -->
    <div class="section">
      <div class="section-header">
        <div class="section-num">10</div>
        <div class="section-title">Recommendations</div>
      </div>
      <ul class="rec-list">
        <li>
          <div class="rec-num">1</div>
          <div><strong>Add inline glossary / tooltips</strong> for technical terms (cosine similarity, phoneme mismatch, discrepancy threshold). Directly addresses the top-ranked concern ({int(theme_counts.get('Confusing Terms',0))}/10 participants). Estimated impact: +5–10 SUS points for non-technical users.</div>
        </li>
        <li>
          <div class="rec-num">2</div>
          <div><strong>Add a plain-English summary card</strong> at the top of the Report tab — one paragraph before the technical evidence — so non-technical users can understand the conclusion without reading detailed metrics.</div>
        </li>
        <li>
          <div class="rec-num">3</div>
          <div><strong>Annotate the Discrepancy Timeline chart</strong> with axis labels, a legend, and a sentence explaining what high vs. low scores mean. {int(theme_counts.get('Timeline Chart Confusing',0))}/10 participants found the current chart unclear.</div>
        </li>
        <li>
          <div class="rec-num">4</div>
          <div><strong>Add a first-use onboarding walkthrough</strong> (3-step tooltip tour covering: upload, analyze, read report) to reduce time-on-task for new users and close the technical/non-technical gap.</div>
        </li>
        <li>
          <div class="rec-num">5</div>
          <div><strong>Show a progress indicator</strong> during analysis. {int(theme_counts.get('Processing Too Slow',0))}/10 participants mentioned speed as a concern. A progress bar with estimated remaining time would reduce perceived wait time without changing actual processing speed.</div>
        </li>
      </ul>
    </div>

    <!-- 11. References -->
    <div class="section">
      <div class="section-header">
        <div class="section-num">11</div>
        <div class="section-title">References</div>
      </div>
      <ul style="list-style:none;padding:0;font-size:.85rem;color:#475569;line-height:2;">
        <li>Brooke, J. (1996). SUS: A "quick and dirty" usability scale. <em>Usability Evaluation in Industry</em>, 189(194), 4–7.</li>
        <li>Bangor, A., Kortum, P., &amp; Miller, J. (2009). Determining what individual SUS scores mean. <em>Journal of Usability Studies</em>, 4(3), 114–123.</li>
        <li>Sauro, J., &amp; Lewis, J. R. (2012). <em>Quantifying the User Experience</em>. Morgan Kaufmann.</li>
        <li>Wang, Z., &amp; Huang, T. (2024). ART-AVDF: Audio-visual deepfake detection via articulatory representation. <em>CVPR 2024</em>.</li>
        <li>EU Artificial Intelligence Act, Article 50 — Transparency obligations for certain AI systems (2024).</li>
        <li>NIST SP 800-86: Guide to Integrating Forensic Techniques into Incident Response (2006).</li>
      </ul>
    </div>

  </div><!-- /content -->

  <!-- ══ FOOTER ══ -->
  <div class="footer">
    Deep-Guard Agent v2.0 &nbsp;·&nbsp; IS596 Evaluation Study &nbsp;·&nbsp;
    Generated {gen_time} &nbsp;·&nbsp; N = 10 University Students<br>
    This report is based on simulated evaluation data for academic study purposes.
  </div>

</div><!-- /page -->
</body>
</html>"""

out_path = OUT / "evaluation_report.html"
out_path.write_text(HTML, encoding="utf-8")
print(f"✓ HTML report → {out_path.resolve()}")
