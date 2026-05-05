# Deep-Guard Agent — User Evaluation Study
## Study Overview & Experimental Protocol

**Version:** 1.0  
**Course:** IS596  
**System:** Deep-Guard Agent v2.0 (Audio-Visual Deepfake Detection)

---

## 1. Research Questions

| # | Research Question | Primary Metric |
|---|-------------------|----------------|
| RQ1 | How usable is Deep-Guard Agent for non-expert users? | SUS score |
| RQ2 | Can users correctly interpret the system's verdict and evidence? | Comprehension score |
| RQ3 | Do users trust the AI-generated analysis? | Trust scale (1–7) |
| RQ4 | Is the Legal Report useful for understanding legal implications? | Usefulness scale (1–7) |
| RQ5 | What are the main usability pain points? | Qualitative themes |

---

## 2. Study Design

| Item | Detail |
|------|--------|
| Design | Within-subjects (all participants complete all tasks) |
| Participants | N = 10 |
| Sessions | Individual, one session per participant |
| Duration | ~60 minutes per session |
| Setting | In-person or remote (screen share) |
| Compensation | None required (course study) |

### 2.1 Participant Profile

- Age: 18–50
- Background: Mix of technical (CS/IS) and non-technical participants
- No prior use of Deep-Guard Agent
- Basic computer literacy (can upload a file, click buttons)

### 2.2 Stimuli — Test Videos

Prepare **3 test videos** with known ground truth. Each participant analyzes all 3:

| Video | Label | Type | Duration | Notes |
|-------|-------|------|----------|-------|
| V1 | AUTHENTIC | Real human speech | 20–40s | Clear face, audible speech |
| V2 | LIKELY FAKE | Obvious deepfake | 20–40s | Face-swap or lip-sync |
| V3 | SUSPICIOUS | Subtle deepfake | 20–40s | Partial or low-quality manipulation |

> **Ground truth must be verified before the study begins.**  
> Use publicly available deepfake datasets (e.g., FaceForensics++, DFDC, Celeb-DF).

---

## 3. Session Flow (60 min)

```
[00:00]  Welcome & Introduction            (5 min)
[00:05]  Informed Consent                  (3 min)
[00:08]  Pre-Study Questionnaire           (5 min)
[00:13]  System Introduction (no demo)     (2 min)
[00:15]  Task 1 — Analyze Video V1         (8 min)
[00:23]  Task 2 — Analyze Video V2         (8 min)
[00:31]  Task 3 — Analyze Video V3         (8 min)
[00:39]  Post-Task Questionnaire (SUS)     (8 min)
[00:47]  Semi-Structured Interview         (10 min)
[00:57]  Debrief & Close                   (3 min)
[01:00]  END
```

---

## 4. Roles

| Role | Responsibility |
|------|---------------|
| **Facilitator** | Runs the session, reads scripts, does NOT help with tasks |
| **Observer** (optional) | Takes notes on behaviour, does not speak |
| **Participant** | Completes all tasks independently |

---

## 5. Key Measures

### Quantitative
| Measure | Instrument | Scale |
|---------|-----------|-------|
| Usability | SUS (10 items) | 0–100 |
| Verdict comprehension | Custom Q | 1–7 Likert |
| Trust in AI verdict | Custom Q | 1–7 Likert |
| Legal report usefulness | Custom Q | 1–7 Likert |
| Overall satisfaction | Custom Q | 1–7 Likert |
| Task completion | Facilitator observation | Binary (0/1) |
| Time on task | Stopwatch | Seconds |
| Verdict agreement | Observed choice vs ground truth | Binary (0/1) |

### Qualitative
- Semi-structured interview: ~5 open-ended questions
- Facilitator observation notes (errors, confusion points, verbal comments)

---

## 6. Files in This Package

| File | Description |
|------|-------------|
| `00_study_overview.md` | This file — full protocol |
| `01_consent_form.md` | Participant consent form (print & sign) |
| `02_pre_questionnaire.md` | Background & prior knowledge survey |
| `03_task_scenarios.md` | Task instructions given to participant |
| `04_post_questionnaire.md` | SUS + custom post-task survey |
| `05_interview_guide.md` | Semi-structured interview script |
| `06_facilitator_log.md` | Per-session observation & timing sheet |
| `07_analysis_plan.md` | Statistical analysis plan & scoring guide |

---

## 7. Ethics

- No personally identifiable information (PII) collected beyond participant ID
- Consent is voluntary and can be withdrawn at any time
- Data stored locally, not shared externally
- Assign each participant an anonymous ID: P01–P10
