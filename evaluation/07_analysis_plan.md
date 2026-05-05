# Analysis Plan & Results Template
## Deep-Guard Agent — User Evaluation Study

*Complete this file after all 10 sessions are finished.*

---

## 1. Data Aggregation Table

Transfer scores from all 06_facilitator_log.md files:

| PID | Background | AI Famil. | DF Famil. | SUS | Compreh. | Legal | Trust | Satisf. | T1 Time | T2 Time | T3 Time | T1 Agree | T2 Agree | T3 Agree |
|-----|-----------|-----------|-----------|-----|----------|-------|-------|---------|---------|---------|---------|---------|---------|---------|
| P01 | | | | | | | | | | | | | | |
| P02 | | | | | | | | | | | | | | |
| P03 | | | | | | | | | | | | | | |
| P04 | | | | | | | | | | | | | | |
| P05 | | | | | | | | | | | | | | |
| P06 | | | | | | | | | | | | | | |
| P07 | | | | | | | | | | | | | | |
| P08 | | | | | | | | | | | | | | |
| P09 | | | | | | | | | | | | | | |
| P10 | | | | | | | | | | | | | | |
| **Mean** | | | | | | | | | | | | | | |
| **SD** | | | | | | | | | | | | | | |
| **Min** | | | | | | | | | | | | | | |
| **Max** | | | | | | | | | | | | | | |

---

## 2. Key Results — Fill In After Analysis

### 2.1 System Usability (SUS)

| Metric | Value |
|--------|-------|
| Mean SUS score | |
| Standard deviation | |
| Minimum | |
| Maximum | |
| % participants scoring ≥ 68 (above average) | |
| SUS Grade (most common) | |

**Interpretation:**  
> SUS ≥ 68 = above-average usability. SUS ≥ 80.3 = "Good". SUS ≥ 90.9 = "Excellent".

---

### 2.2 Verdict Agreement Rate

*(Proportion of times participant agreed with system verdict, across all 3 tasks × 10 participants = 30 data points)*

| Video | Ground Truth | # Correct Agreements / 10 | Agreement Rate |
|-------|-------------|--------------------------|---------------|
| V1 | Authentic | | % |
| V2 | Likely Fake | | % |
| V3 | Suspicious | | % |
| **Overall** | | | **%** |

---

### 2.3 Task Completion & Time

| Task | Completion Rate | Mean Time (s) | SD Time (s) |
|------|----------------|--------------|------------|
| Task 1 (V1) | % | | |
| Task 2 (V2) | % | | |
| Task 3 (V3) | % | | |

---

### 2.4 Likert Scale Summary (1–7)

| Scale | Mean | SD | Interpretation |
|-------|------|----|---------------|
| Report Comprehension | | | |
| Legal Report Usefulness | | | |
| Trust in AI Verdict | | | |
| Overall Satisfaction | | | |

> **Reference:** Score ≥ 5.5/7 = positive; 4.0–5.4 = neutral; < 4.0 = negative

---

### 2.5 Baseline vs. System-Assisted Detection

*(Compare pre-task unaided accuracy with task-time system-assisted accuracy)*

| Condition | Correct Identifications / 20 | Accuracy |
|-----------|------------------------------|---------|
| Unaided (baseline) | | % |
| System-assisted (tasks) | | % |
| **Improvement** | | **Δ %** |

---

## 3. Qualitative Analysis

### Thematic Coding Guide

After all interviews, read through the interview notes and tag recurring themes:

| Code | Description | Example Quote |
|------|-------------|---------------|
| `CLEAR_VERDICT` | Participant found verdict easy to understand | "The badge was immediately clear" |
| `CONFUSING_TERMS` | Technical terminology caused confusion | "I didn't know what cosine similarity meant" |
| `TRUST_AI` | Positive trust in system | "The evidence made me feel confident" |
| `DISTRUST_AI` | Skepticism about AI verdict | "I'd want a human to confirm" |
| `LEGAL_USEFUL` | Legal Report seen as useful | "I'd use this to report to HR" |
| `LEGAL_JARGON` | Legal Report seen as too technical | "Too many legal terms I don't know" |
| `TIMELINE_HELPFUL` | Timeline chart was useful | "I could see exactly when it was faked" |
| `TIMELINE_CONFUSING` | Timeline chart was confusing | "I wasn't sure what the Y axis meant" |
| `SLOW_SPEED` | Processing time was too long | |
| `WANT_MORE_DETAIL` | Wanted more explanation | |

### Theme Frequency Table

| Theme Code | # of Participants Mentioning | % |
|-----------|------------------------------|---|
| CLEAR_VERDICT | | |
| CONFUSING_TERMS | | |
| TRUST_AI | | |
| DISTRUST_AI | | |
| LEGAL_USEFUL | | |
| LEGAL_JARGON | | |
| TIMELINE_HELPFUL | | |
| TIMELINE_CONFUSING | | |
| SLOW_SPEED | | |
| WANT_MORE_DETAIL | | |

---

## 4. Statistical Tests (Optional, if N allows)

With N=10, use non-parametric tests:

| Question | Test | Variables |
|----------|------|-----------|
| Does AI familiarity predict SUS? | Spearman ρ | B1 vs SUS |
| Does DF familiarity predict trust? | Spearman ρ | B2 vs Trust |
| Does technical background affect comprehension? | Mann-Whitney U | Background group vs Comprehension |
| Baseline vs assisted accuracy | McNemar's test | Baseline correct vs Task correct |

---

## 5. Reporting Template

*Use these sentences as a starting point for your report:*

```
The mean SUS score was ___ (SD = ___), indicating [poor/marginal/above-average/good/excellent]
usability (Brooke, 1996). [X]% of participants scored above the industry benchmark of 68.

Participants agreed with the system's verdict in ___% of cases across all three videos,
compared to ___% accuracy in the unaided baseline condition — an improvement of ___%.

Report comprehension was rated M = ___ (SD = ___, scale 1–7). Legal report usefulness
was rated M = ___ (SD = ___). Trust in the AI verdict was rated M = ___ (SD = ___).

Qualitative analysis identified [X] key themes. The most frequently mentioned
positive theme was [THEME] ([X] participants), and the most frequently mentioned
concern was [THEME] ([X] participants).
```

---

## 6. References

- Brooke, J. (1996). SUS: A 'quick and dirty' usability scale. *Usability Evaluation in Industry*, 189(194), 4–7.
- Bangor, A., Kortum, P., & Miller, J. (2009). Determining what individual SUS scores mean. *Journal of Usability Studies*, 4(3), 114–123.
- Sauro, J., & Lewis, J. R. (2012). *Quantifying the User Experience*. Morgan Kaufmann.
