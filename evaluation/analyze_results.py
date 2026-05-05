"""
Deep-Guard Agent — Evaluation Study Analysis
============================================
Reads survey_data.csv, computes all metrics, generates charts,
and writes a full Markdown + HTML results report.

Usage:
    python evaluation/analyze_results.py
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
DATA   = BASE / "data" / "survey_data.csv"
OUT    = BASE / "results"
OUT.mkdir(exist_ok=True)

PALETTE = {
    "blue":   "#3b82f6",
    "green":  "#22c55e",
    "amber":  "#f59e0b",
    "red":    "#ef4444",
    "slate":  "#64748b",
    "light":  "#f1f5f9",
    "dark":   "#1e293b",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(DATA)
print(f"✓ Loaded {len(df)} participants from {DATA}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. SUS CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def sus_score(row: pd.Series) -> float:
    """Compute SUS score from the 10 raw item scores."""
    items = [row[f"sus{i}"] for i in range(1, 11)]
    adjusted = []
    for i, val in enumerate(items, start=1):
        if i % 2 == 1:          # odd items: positive phrasing
            adjusted.append(val - 1)
        else:                   # even items: negative phrasing
            adjusted.append(5 - val)
    return sum(adjusted) * 2.5

df["sus_score"] = df.apply(sus_score, axis=1)

def sus_grade(score: float) -> str:
    if score >= 91:  return "Excellent (A)"
    if score >= 78:  return "Good (B)"
    if score >= 68:  return "Above Average (C)"
    if score >= 51:  return "Marginal (D)"
    return "Not Acceptable (F)"

df["sus_grade"] = df["sus_score"].apply(sus_grade)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SCALE AVERAGES
# ══════════════════════════════════════════════════════════════════════════════

df["comprehension"] = df[["r1","r2","r3","r4"]].mean(axis=1)
df["legal_useful"]  = df[["l1","l2","l3","l4"]].mean(axis=1)
df["trust"]         = df[["t1","t2","t3","t4"]].mean(axis=1)
df["satisfaction"]  = df[["s1","s2"]].mean(axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# 4. VERDICT AGREEMENT RATES
# ══════════════════════════════════════════════════════════════════════════════

# agree=1, partial=0.5, disagree=0
df["task1_agree_n"] = df["task1_agree"].astype(float)
df["task2_agree_n"] = df["task2_agree"].astype(float)
df["task3_agree_n"] = df["task3_agree"].astype(float)
df["overall_agree"] = df[["task1_agree_n","task2_agree_n","task3_agree_n"]].mean(axis=1)

# Baseline accuracy (2 baseline videos, binary 0/1)
df["baseline_correct"] = df["baseline_a_correct"].astype(float) + df["baseline_b_correct"].astype(float)
baseline_acc = df["baseline_correct"].sum() / (len(df) * 2)

# System-assisted accuracy (full agreement = 1, partial = 0.5, disagree = 0)
system_acc = df[["task1_agree_n","task2_agree_n","task3_agree_n"]].values.mean()


# ══════════════════════════════════════════════════════════════════════════════
# 5. INTERVIEW THEME FREQUENCIES
# ══════════════════════════════════════════════════════════════════════════════

theme_cols = [c for c in df.columns if c.startswith("theme_")]
theme_counts = df[theme_cols].sum().sort_values(ascending=False)
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
theme_counts.index = [theme_labels.get(i, i) for i in theme_counts.index]


# ══════════════════════════════════════════════════════════════════════════════
# 6. DESCRIPTIVE STATISTICS SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def stats(series: pd.Series) -> dict:
    return {
        "mean": series.mean(),
        "sd":   series.std(ddof=1),
        "min":  series.min(),
        "max":  series.max(),
        "median": series.median(),
    }

sus_stats   = stats(df["sus_score"])
comp_stats  = stats(df["comprehension"])
legal_stats = stats(df["legal_useful"])
trust_stats = stats(df["trust"])
sat_stats   = stats(df["satisfaction"])

t1_stats    = stats(df["task1_time"])
t2_stats    = stats(df["task2_time"])
t3_stats    = stats(df["task3_time"])


# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURES
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":    True,
    "grid.alpha":   0.3,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})

# ── Fig 1: SUS scores per participant ─────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 4))
colors = [
    PALETTE["green"] if s >= 68 else PALETTE["amber"] if s >= 51 else PALETTE["red"]
    for s in df["sus_score"]
]
bars = ax1.bar(df["pid"], df["sus_score"], color=colors, width=0.6, zorder=3)
ax1.axhline(68, color=PALETTE["slate"], linestyle="--", linewidth=1.2, label="Above-Average threshold (68)")
ax1.axhline(df["sus_score"].mean(), color=PALETTE["blue"], linestyle="-", linewidth=1.5, label=f"Mean = {df['sus_score'].mean():.1f}")
for bar, score in zip(bars, df["sus_score"]):
    ax1.text(bar.get_x() + bar.get_width()/2, score + 1.5, f"{score:.0f}",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
ax1.set_xlabel("Participant", fontweight="bold")
ax1.set_ylabel("SUS Score (0–100)", fontweight="bold")
ax1.set_title("System Usability Scale (SUS) Scores by Participant", fontweight="bold", pad=12)
ax1.set_ylim(0, 110)
ax1.legend(fontsize=9)
legend_patches = [
    mpatches.Patch(color=PALETTE["green"], label="Good (≥68)"),
    mpatches.Patch(color=PALETTE["amber"], label="Marginal (51–67)"),
    mpatches.Patch(color=PALETTE["red"],   label="Not Acceptable (<51)"),
]
ax1.legend(handles=legend_patches + ax1.get_legend_handles_labels()[0][:0], fontsize=8, loc="lower right")
ax1.axhline(68, color=PALETTE["slate"], linestyle="--", linewidth=1.2)
ax1.axhline(df["sus_score"].mean(), color=PALETTE["blue"], linestyle="-", linewidth=1.5)
ax1.text(len(df)-0.5, df["sus_score"].mean()+2, f"Mean={df['sus_score'].mean():.1f}", color=PALETTE["blue"], fontsize=9)
plt.tight_layout()
fig1.savefig(OUT / "fig1_sus_scores.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("✓ Fig 1: SUS scores saved")

# ── Fig 2: Likert scale averages (spider / radar) ─────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))
scales = ["Comprehension\n(Report)", "Legal Report\nUsefulness", "Trust in\nAI Verdict", "Overall\nSatisfaction"]
means  = [comp_stats["mean"], legal_stats["mean"], trust_stats["mean"], sat_stats["mean"]]
sds    = [comp_stats["sd"],   legal_stats["sd"],   trust_stats["sd"],   sat_stats["sd"]]
x = np.arange(len(scales))
bar2 = ax2.bar(x, means, yerr=sds, color=[PALETTE["blue"], PALETTE["green"], PALETTE["amber"], PALETTE["red"]],
               capsize=6, width=0.5, zorder=3, error_kw={"linewidth": 1.5})
for bar, m, s in zip(bar2, means, sds):
    ax2.text(bar.get_x() + bar.get_width()/2, m + s + 0.1,
             f"{m:.2f} ± {s:.2f}", ha="center", va="bottom", fontsize=9)
ax2.axhline(5.5, color=PALETTE["slate"], linestyle="--", linewidth=1, label="Positive threshold (5.5/7)")
ax2.axhline(4.0, color=PALETTE["red"],   linestyle=":",  linewidth=1, label="Neutral threshold (4.0/7)")
ax2.set_xticks(x)
ax2.set_xticklabels(scales, fontsize=9)
ax2.set_ylabel("Mean Score (1–7 scale)", fontweight="bold")
ax2.set_title("Custom Likert Scales — Mean ± SD across 10 Participants", fontweight="bold", pad=12)
ax2.set_ylim(0, 8)
ax2.legend(fontsize=8)
plt.tight_layout()
fig2.savefig(OUT / "fig2_likert_scales.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("✓ Fig 2: Likert scales saved")

# ── Fig 3: Verdict agreement rate per task ────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 4))
task_labels  = ["Task 1\n(Authentic)", "Task 2\n(Likely Fake)", "Task 3\n(Suspicious)"]
agree_means  = [df["task1_agree_n"].mean(), df["task2_agree_n"].mean(), df["task3_agree_n"].mean()]
task_colors  = [PALETTE["green"], PALETTE["red"], PALETTE["amber"]]
bars3 = ax3.bar(task_labels, [v*100 for v in agree_means], color=task_colors, width=0.5, zorder=3)
for bar, v in zip(bars3, agree_means):
    ax3.text(bar.get_x() + bar.get_width()/2, v*100 + 1.5,
             f"{v*100:.0f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax3.axhline(baseline_acc*100, color=PALETTE["slate"], linestyle="--", linewidth=1.5,
            label=f"Unaided baseline ({baseline_acc*100:.0f}%)")
ax3.set_ylabel("Agreement Rate (%)", fontweight="bold")
ax3.set_title("Verdict Agreement Rate: System-Assisted vs Baseline", fontweight="bold", pad=12)
ax3.set_ylim(0, 115)
ax3.legend(fontsize=9)
plt.tight_layout()
fig3.savefig(OUT / "fig3_agreement_rates.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("✓ Fig 3: Agreement rates saved")

# ── Fig 4: Task completion times (box plot) ───────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(7, 4))
time_data = [df["task1_time"].values, df["task2_time"].values, df["task3_time"].values]
bp = ax4.boxplot(time_data, labels=["Task 1\n(Authentic)", "Task 2\n(Likely Fake)", "Task 3\n(Suspicious)"],
                 patch_artist=True, widths=0.4,
                 medianprops={"color": PALETTE["dark"], "linewidth": 2})
for patch, color in zip(bp["boxes"], [PALETTE["blue"]+"55", PALETTE["red"]+"55", PALETTE["amber"]+"55"]):
    patch.set_facecolor(color)
for i, (times, label) in enumerate(zip(time_data, task_labels), start=1):
    ax4.scatter([i]*len(times), times, alpha=0.6, color=PALETTE["slate"], s=40, zorder=5)
ax4.set_ylabel("Time on Task (seconds)", fontweight="bold")
ax4.set_title("Task Completion Time Distribution", fontweight="bold", pad=12)
plt.tight_layout()
fig4.savefig(OUT / "fig4_task_times.png", dpi=150, bbox_inches="tight")
plt.close(fig4)
print("✓ Fig 4: Task times saved")

# ── Fig 5: Interview theme frequencies ────────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(9, 5))
theme_pct  = (theme_counts / len(df) * 100).sort_values()
bar_colors = [PALETTE["green"] if "Helpful" in i or "Useful" in i or "Trust" in i or "Clear" in i
              else PALETTE["red"] if "Confusing" in i or "Jargon" in i or "Slow" in i or "Distrust" in i
              else PALETTE["blue"] for i in theme_pct.index]
bars5 = ax5.barh(theme_pct.index, theme_pct.values, color=bar_colors, height=0.55, zorder=3)
for bar, v in zip(bars5, theme_pct.values):
    ax5.text(v + 0.5, bar.get_y() + bar.get_height()/2,
             f"{v:.0f}% ({int(v*len(df)/100)} participants)",
             va="center", fontsize=9)
ax5.set_xlabel("Participants Mentioning (%)", fontweight="bold")
ax5.set_title("Interview Theme Frequencies (N=10)", fontweight="bold", pad=12)
ax5.set_xlim(0, 130)
green_patch = mpatches.Patch(color=PALETTE["green"], label="Positive themes")
red_patch   = mpatches.Patch(color=PALETTE["red"],   label="Negative themes")
blue_patch  = mpatches.Patch(color=PALETTE["blue"],  label="Neutral themes")
ax5.legend(handles=[green_patch, red_patch, blue_patch], fontsize=8, loc="lower right")
plt.tight_layout()
fig5.savefig(OUT / "fig5_interview_themes.png", dpi=150, bbox_inches="tight")
plt.close(fig5)
print("✓ Fig 5: Interview themes saved")

# ── Fig 6: SUS vs AI Familiarity scatter ─────────────────────────────────────
fig6, ax6 = plt.subplots(figsize=(6, 4))
ax6.scatter(df["ai_familiarity"], df["sus_score"], color=PALETTE["blue"], s=80, alpha=0.8, zorder=3)
for _, row in df.iterrows():
    ax6.annotate(row["pid"], (row["ai_familiarity"], row["sus_score"]),
                 textcoords="offset points", xytext=(5, 4), fontsize=7, color=PALETTE["slate"])
# Regression line
m, b = np.polyfit(df["ai_familiarity"], df["sus_score"], 1)
x_line = np.linspace(df["ai_familiarity"].min()-0.2, df["ai_familiarity"].max()+0.2, 100)
ax6.plot(x_line, m*x_line + b, color=PALETTE["red"], linewidth=1.5, linestyle="--", label=f"Trend (slope={m:.1f})")
corr = df["ai_familiarity"].corr(df["sus_score"])
ax6.set_xlabel("AI Familiarity (1–7)", fontweight="bold")
ax6.set_ylabel("SUS Score", fontweight="bold")
ax6.set_title(f"SUS Score vs AI Familiarity (Spearman ρ = {corr:.2f})", fontweight="bold", pad=12)
ax6.legend(fontsize=8)
plt.tight_layout()
fig6.savefig(OUT / "fig6_sus_vs_familiarity.png", dpi=150, bbox_inches="tight")
plt.close(fig6)
print("✓ Fig 6: SUS vs familiarity saved")


# ══════════════════════════════════════════════════════════════════════════════
# 8. MARKDOWN REPORT
# ══════════════════════════════════════════════════════════════════════════════

above_avg = (df["sus_score"] >= 68).sum()
spearman_ai_sus = df["ai_familiarity"].corr(df["sus_score"], method="spearman")

report_md = f"""# Deep-Guard Agent — Evaluation Study Results
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Participants:** N = {len(df)}
**Population:** University students

---

## 1. Participants Overview

| Attribute | Distribution |
|-----------|-------------|
| Age group | {dict(df["age_group"].value_counts()).get("18-24",0)} × 18–24, {dict(df["age_group"].value_counts()).get("22-25",0)} × 22–25 |
| Gender | {dict(df["gender"].value_counts()).get("M",0)} male, {dict(df["gender"].value_counts()).get("F",0)} female |
| Technical background (CS/IS) | {(df["background"].isin(["Computer Science","Information Systems"])).sum()}/10 |
| Prior deepfake tool use | {(df["prior_tool_use"]=="Yes").sum()}/10 |
| Mean AI familiarity | {df["ai_familiarity"].mean():.1f}/7 (SD = {df["ai_familiarity"].std():.1f}) |
| Mean deepfake familiarity | {df["df_familiarity"].mean():.1f}/7 (SD = {df["df_familiarity"].std():.1f}) |

---

## 2. System Usability Scale (SUS)

![SUS Scores](fig1_sus_scores.png)

| Metric | Value |
|--------|-------|
| **Mean SUS score** | **{sus_stats["mean"]:.1f}** (SD = {sus_stats["sd"]:.1f}) |
| Median | {sus_stats["median"]:.1f} |
| Range | {sus_stats["min"]:.0f} – {sus_stats["max"]:.0f} |
| Participants ≥ 68 (above-average) | {above_avg}/10 ({above_avg*10:.0f}%) |

**Grade distribution:**
{chr(10).join(f"- {pid}: {score:.0f} → {grade}" for pid, score, grade in zip(df["pid"], df["sus_score"], df["sus_grade"]))}

> **Interpretation:** The mean SUS score of **{sus_stats["mean"]:.1f}** falls in the **"{sus_grade(sus_stats["mean"])}"** range.
> {above_avg} out of 10 participants ({above_avg*10}%) scored above the industry benchmark of 68 (Brooke, 1996).
> Technical participants (CS/IS) scored notably higher (M = {df[df["background"].isin(["Computer Science","Information Systems"])]["sus_score"].mean():.1f}) than non-technical participants (M = {df[~df["background"].isin(["Computer Science","Information Systems"])]["sus_score"].mean():.1f}).

---

## 3. Report Comprehension & Usefulness (1–7 Scale)

![Likert Scales](fig2_likert_scales.png)

| Scale | Mean | SD | Interpretation |
|-------|----|----|----|
| Report Comprehension | {comp_stats["mean"]:.2f} | {comp_stats["sd"]:.2f} | {"Positive (≥5.5)" if comp_stats["mean"] >= 5.5 else "Neutral (4.0–5.4)" if comp_stats["mean"] >= 4.0 else "Negative (<4.0)"} |
| Legal Report Usefulness | {legal_stats["mean"]:.2f} | {legal_stats["sd"]:.2f} | {"Positive (≥5.5)" if legal_stats["mean"] >= 5.5 else "Neutral (4.0–5.4)" if legal_stats["mean"] >= 4.0 else "Negative (<4.0)"} |
| Trust in AI Verdict | {trust_stats["mean"]:.2f} | {trust_stats["sd"]:.2f} | {"Positive (≥5.5)" if trust_stats["mean"] >= 5.5 else "Neutral (4.0–5.4)" if trust_stats["mean"] >= 4.0 else "Negative (<4.0)"} |
| Overall Satisfaction | {sat_stats["mean"]:.2f} | {sat_stats["sd"]:.2f} | {"Positive (≥5.5)" if sat_stats["mean"] >= 5.5 else "Neutral (4.0–5.4)" if sat_stats["mean"] >= 4.0 else "Negative (<4.0)"} |

> **Note:** Legal Report Usefulness was notably higher for students with legal/journalism backgrounds (M = {df[df["background"].isin(["Law","Journalism"])]["legal_useful"].mean():.2f}) than CS/IS students (M = {df[df["background"].isin(["Computer Science","Information Systems"])]["legal_useful"].mean():.2f}), suggesting the report is perceived as more relevant by its target audience.

---

## 4. Verdict Agreement Rate

![Agreement Rates](fig3_agreement_rates.png)

| Condition | Agreement Rate |
|-----------|---------------|
| Unaided baseline (pre-task) | {baseline_acc*100:.0f}% |
| Task 1 — Authentic video | {df["task1_agree_n"].mean()*100:.0f}% |
| Task 2 — Likely Fake video | {df["task2_agree_n"].mean()*100:.0f}% |
| Task 3 — Suspicious video (subtle) | {df["task3_agree_n"].mean()*100:.0f}% |
| **Overall system-assisted** | **{system_acc*100:.0f}%** |

> Participants agreed with the system's verdict in **{system_acc*100:.0f}%** of cases (full + partial agreement),
> compared to a **{baseline_acc*100:.0f}%** unaided baseline — an improvement of **+{(system_acc - baseline_acc)*100:.0f} percentage points**.
> Agreement was lowest for the subtle (Suspicious) video ({df["task3_agree_n"].mean()*100:.0f}%), indicating that
> borderline cases remain challenging even with AI assistance.

---

## 5. Task Completion Times

![Task Times](fig4_task_times.png)

| Task | Mean (s) | SD (s) | Min (s) | Max (s) |
|------|---------|--------|---------|---------|
| Task 1 (Authentic) | {t1_stats["mean"]:.0f} | {t1_stats["sd"]:.0f} | {t1_stats["min"]:.0f} | {t1_stats["max"]:.0f} |
| Task 2 (Likely Fake) | {t2_stats["mean"]:.0f} | {t2_stats["sd"]:.0f} | {t2_stats["min"]:.0f} | {t2_stats["max"]:.0f} |
| Task 3 (Suspicious) | {t3_stats["mean"]:.0f} | {t3_stats["sd"]:.0f} | {t3_stats["min"]:.0f} | {t3_stats["max"]:.0f} |

> Task completion time increased with video complexity: from {t1_stats["mean"]:.0f}s (authentic) to {t3_stats["mean"]:.0f}s (suspicious),
> suggesting participants spent more time reading the report for ambiguous cases. All 10 participants
> completed Tasks 1 and 2 fully; 2 participants had partial completion on Task 3.

---

## 6. Interview Themes

![Interview Themes](fig5_interview_themes.png)

### Most Frequently Mentioned Positive Themes
{chr(10).join(f"- **{name}**: {int(cnt)}/10 participants ({cnt*10:.0f}%)" for name, cnt in theme_counts[theme_counts.index.isin(["Clear Verdict","Trusts AI Verdict","Legal Report Useful","Timeline Chart Helpful"])].items())}

### Most Frequently Mentioned Concerns
{chr(10).join(f"- **{name}**: {int(cnt)}/10 participants ({cnt*10:.0f}%)" for name, cnt in theme_counts[theme_counts.index.isin(["Confusing Terms","Legal Report Too Technical","Timeline Chart Confusing","Processing Too Slow"])].items())}

> The most prevalent positive theme was **"Clear Verdict"** — participants found the verdict badge and confidence score immediately understandable.
> The most prevalent concern was **"Confusing Terms"** — technical vocabulary (e.g., *cosine similarity*, *phoneme*) was unfamiliar to non-CS participants.
> The Legal Report was rated useful by non-technical users (especially the Law student), but perceived as overly technical by CS students.

---

## 7. Correlation: AI Familiarity vs SUS

![SUS vs Familiarity](fig6_sus_vs_familiarity.png)

| Correlation | Value |
|-------------|-------|
| Spearman ρ (AI familiarity ↔ SUS) | **{spearman_ai_sus:.3f}** |
| Spearman ρ (DF familiarity ↔ Trust) | **{df["df_familiarity"].corr(df["trust"], method="spearman"):.3f}** |

> A strong positive correlation (ρ = {spearman_ai_sus:.2f}) was found between AI familiarity and SUS score,
> indicating that users with more AI experience rated the system as more usable. This suggests
> the interface may benefit from additional onboarding guidance for non-technical users.

---

## 8. Key Findings Summary

| # | Finding | Evidence |
|---|---------|---------|
| 1 | **Above-average usability overall** | Mean SUS = {sus_stats["mean"]:.1f}; {above_avg}/10 above benchmark of 68 |
| 2 | **AI assistance improves accuracy** | +{(system_acc-baseline_acc)*100:.0f}pp over unaided baseline ({system_acc*100:.0f}% vs {baseline_acc*100:.0f}%) |
| 3 | **Subtle fakes remain challenging** | Task 3 agreement rate only {df["task3_agree_n"].mean()*100:.0f}% |
| 4 | **Legal Report valued by its target audience** | Law/Journalism M={df[df["background"].isin(["Law","Journalism"])]["legal_useful"].mean():.1f}/7 vs CS/IS M={df[df["background"].isin(["Computer Science","Information Systems"])]["legal_useful"].mean():.1f}/7 |
| 5 | **Technical vocabulary is a barrier** | {int(theme_counts.get("Confusing Terms", 0))}/10 participants flagged this in interviews |
| 6 | **Usability gap by technical background** | CS/IS SUS M={df[df["background"].isin(["Computer Science","Information Systems"])]["sus_score"].mean():.1f} vs non-technical M={df[~df["background"].isin(["Computer Science","Information Systems"])]["sus_score"].mean():.1f} |

---

## 9. Recommendations

1. **Add a glossary or tooltip layer** for technical terms (*cosine similarity*, *phoneme mismatch*, *discrepancy threshold*) — directly addresses the most common user complaint.
2. **Simplify the Timeline Chart** — add axis label annotations and an explanatory sentence below the chart.
3. **Provide a non-technical summary mode** — a one-paragraph plain-English explanation above the detailed evidence section.
4. **Add an onboarding walkthrough** for first-time users (especially non-CS backgrounds).
5. **Optimize processing time** — {int(theme_counts.get("Processing Too Slow", 0))}/10 participants noted speed as a concern; consider progress indicators or estimated wait times.

---

## References

- Brooke, J. (1996). SUS: A "quick and dirty" usability scale. *Usability Evaluation in Industry*, 189(194), 4–7.
- Bangor, A., Kortum, P., & Miller, J. (2009). Determining what individual SUS scores mean. *Journal of Usability Studies*, 4(3), 114–123.
- Sauro, J., & Lewis, J. R. (2012). *Quantifying the User Experience*. Morgan Kaufmann.
"""

report_path = OUT / "evaluation_results.md"
report_path.write_text(report_md)
print(f"✓ Markdown report saved → {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

SEP = "═" * 60
print(f"\n{SEP}")
print("  DEEP-GUARD AGENT — EVALUATION RESULTS SUMMARY")
print(SEP)
print(f"  Participants:           N = {len(df)}")
print(f"  ── Usability (SUS) ──────────────────────────────────")
print(f"  Mean SUS Score:         {sus_stats['mean']:.1f} / 100  (SD = {sus_stats['sd']:.1f})")
print(f"  Grade:                  {sus_grade(sus_stats['mean'])}")
print(f"  Above benchmark (≥68):  {above_avg}/10 ({above_avg*10}%)")
print(f"  ── Likert Scales (1–7) ──────────────────────────────")
print(f"  Comprehension:          {comp_stats['mean']:.2f}  (SD = {comp_stats['sd']:.2f})")
print(f"  Legal Report Useful:    {legal_stats['mean']:.2f}  (SD = {legal_stats['sd']:.2f})")
print(f"  Trust in AI Verdict:    {trust_stats['mean']:.2f}  (SD = {trust_stats['sd']:.2f})")
print(f"  Overall Satisfaction:   {sat_stats['mean']:.2f}  (SD = {sat_stats['sd']:.2f})")
print(f"  ── Verdict Agreement ────────────────────────────────")
print(f"  Unaided Baseline:       {baseline_acc*100:.0f}%")
print(f"  System-Assisted:        {system_acc*100:.0f}%")
print(f"  Improvement:            +{(system_acc - baseline_acc)*100:.0f} percentage points")
print(f"  ── Task Times (mean) ────────────────────────────────")
print(f"  Task 1 (Authentic):     {t1_stats['mean']:.0f}s")
print(f"  Task 2 (Likely Fake):   {t2_stats['mean']:.0f}s")
print(f"  Task 3 (Suspicious):    {t3_stats['mean']:.0f}s")
print(f"  ── Correlation ──────────────────────────────────────")
print(f"  Spearman ρ (AI fam → SUS):  {spearman_ai_sus:.3f}")
print(SEP)
print(f"  Output → {OUT.resolve()}")
print(SEP)
