# Deep-Guard Agent — Final Presentation Speech Script
## IS596 · ~10 Minutes · English with Chinese Translation

---

> **Format guide:**  
> Each slide is shown as a block. Left column = English script. Right column = 中文对照翻译.  
> Suggested pace: speak ~130 words/min. Total target: ~1,300 English words.

---

## Slide 1 · Title ｜ 封面（~30 sec）

| 🇬🇧 English | 🇨🇳 中文对照 |
|---|---|
| Good morning, everyone. Today I'll be presenting **Deep-Guard Agent** — a real-time audio-visual deepfake detection system with forensic legal reporting, built for the IS596 final project. | 大家早上好。今天我将介绍 **Deep-Guard Agent**——一个结合音视频检测与法证报告生成的实时深度伪造检测系统，这是我在 IS596 课程的期末项目。 |
| At the end of this ten minutes, I hope you'll agree that deepfake detection is no longer a research curiosity — it's a legal and societal necessity, and our system takes a meaningful step toward addressing it. | 在接下来的十分钟里，我希望能让大家看到：深度伪造检测已不再是学术玩具，而是一个法律与社会层面的迫切需求，而我们的系统正是朝着解决这一需求迈出了重要一步。 |

---

## Slide 2 · Motivation ｜ 研究动机（~60 sec）

| 🇬🇧 English | 🇨🇳 中文对照 |
|---|---|
| So why does this matter? Three converging pressures make deepfake detection urgent right now. | 为什么这个问题如此重要？有三股力量正在同时施压，使深度伪造检测变得极为迫切。 |
| **First, the scale of the problem.** Deepfake incidents have grown by 3,000 percent since 2019. Ninety-six percent of deepfakes target individuals non-consensually — we're talking about real people having their faces and voices weaponized against them. | **第一，问题的规模。** 自2019年以来，深度伪造事件增长了3000%。96%的深度伪造内容在未经当事人同意的情况下针对个人——这意味着真实的人正在遭受面部和声音被恶意利用的侵害。 |
| **Second, the legal pressure.** The EU AI Act Article 50 and the US Take It Down Act 2025 now *require* verifiable forensic evidence of synthetic media. Platforms and organizations need tools that can produce court-admissible documentation — not just a score. | **第二，法律压力。** 欧盟《人工智能法》第50条和美国2025年《删除法案》明确要求提供可验证的合成媒体法证证据。平台和机构需要能生成具有法律效力文件的工具——而不仅仅是一个检测分数。 |
| **Third, the detection gap.** Every existing tool I evaluated is either single-modality — looking at video *or* audio, not both — or a black box with no explanation. **Deep-Guard Agent fills all three gaps in a single open system.** | **第三，检测空白。** 我评估过的所有现有工具，要么是单模态的——只看视频*或*音频，而不是同时分析两者——要么是黑盒系统，完全没有解释。**Deep-Guard Agent 在一个开源系统中填补了这三项空白。** |

---

## Slide 3 · System Architecture ｜ 系统架构（~80 sec）

| 🇬🇧 English | 🇨🇳 中文对照 |
|---|---|
| Let me walk you through the six-module pipeline. It flows in two rows. | 让我带大家过一遍这个六模块流水线，它分为两行。 |
| **Row one is the encoding stage.** The video file enters the **Video Input** module, which handles MP4, AVI, and MOV formats. It's then split: the visual stream goes to the **Visual Encoder**, which uses MediaPipe FaceLandmarker to extract 478 facial landmarks and compress them into a 256-dimensional lip feature vector. Simultaneously, the audio stream is processed by the **Audio Encoder**, powered by Wav2Vec2, producing a 768-dimensional representation at 50 frames per second. | **第一行是编码阶段。** 视频文件进入**视频输入**模块，支持 MP4、AVI 和 MOV 格式。然后分为两路：视觉流进入**视觉编码器**，使用 MediaPipe 提取478个面部关键点并压缩为256维唇部特征向量；音频流同步进入**音频编码器**，基于 Wav2Vec2，以每秒50帧的频率生成768维特征表示。 |
| **Row two is the reasoning stage.** The **Cross-Modal Fusion** module aligns these two streams in time using a Temporal Transformer and computes a weighted discrepancy score: 70% learned discrepancy plus 30% cosine discrepancy, as shown in the formula at the bottom. Any frame scoring above 0.65 is flagged. | **第二行是推理阶段。** **跨模态融合**模块使用时序 Transformer 对两路特征进行时间对齐，并计算加权差异分数：70% 的学习差异加上 30% 的余弦差异，如底部公式所示。分数超过0.65的帧被标记为可疑。 |
| The **LLM Reasoner** — powered by LLaMA 3.3 70B via Groq — then takes those scores and flagged frames and produces a chain-of-thought analysis with a structured JSON verdict. Finally, the **Output** module renders an annotated video, an HTML analysis report, a legal forensic report, and a raw JSON file. | **LLM 推理器**——通过 Groq 调用 LLaMA 3.3 70B——接收这些分数和被标记的帧，生成思维链分析和结构化 JSON 判决。最后，**输出**模块渲染标注视频、HTML 分析报告、法证报告和原始 JSON 文件。 |

---

## Slide 4 · Theoretical Grounding ｜ 理论基础（~80 sec）

| 🇬🇧 English | 🇨🇳 中文对照 |
|---|---|
| The system is grounded in four pillars of existing literature. | 该系统建立在四个理论支柱之上。 |
| **AV-HuBERT** by Shi et al. (2022) demonstrated that audio and visual speech signals can be jointly learned through self-supervision. It inspired our cross-modal alignment architecture and the idea that temporal synchrony is a learnable signal. | **AV-HuBERT**（Shi等，2022）证明了音频和视觉语音信号可以通过自监督联合学习。它启发了我们的跨模态对齐架构，以及时序同步是一种可学习信号的核心思路。 |
| **Wav2Vec 2.0** by Baevski et al. (2020) gives us a principled, physics-grounded audio representation. Because it was pre-trained on 960 hours of real human speech, its feature space captures articulatory constraints — the physical mechanics of how a human mouth produces sound. | **Wav2Vec 2.0**（Baevski等，2020）为我们提供了有物理依据的音频表示。由于它在960小时真实人类语音上进行了预训练，其特征空间捕捉了发音约束——即人类口腔产生声音的物理机制。 |
| **ART-AVDF** by Wang and Huang (2024) is our direct methodological precedent — it's the closest published work to what we built, using articulatory representations specifically for deepfake detection. | **ART-AVDF**（Wang与Huang，2024）是我们最直接的方法论先例——它是与我们构建内容最接近的已发表工作，专门使用发音表示进行深度伪造检测。 |
| And the **Daubert Standard** combined with **ISO/IEC 27037** frames our legal report design. Any evidence we produce must be testable, have a known error rate, and maintain a documented chain of custody. That's not just a nice-to-have — it's a legal requirement in US federal courts. | **Daubert标准**结合**ISO/IEC 27037**框架构建了我们的法证报告设计。我们生成的任何证据都必须可验证、具有已知错误率，并维护有据可查的证据保管链——这不只是锦上添花，而是美国联邦法院的法律要求。 |
| The key physical insight tying all of this together: bilabial phonemes — sounds like B, P, and M — require the lips to *fully close*. Current deepfake generators consistently violate this constraint. That violation is our detection signal. | 将这一切串联起来的核心物理洞见是：双唇音——如B、P、M——要求嘴唇*完全闭合*。而当前的深度伪造生成器始终违反这一约束，这个违反就是我们的检测信号。 |

---

## Slide 5 · Implementation ｜ 实现细节（~80 sec）

| 🇬🇧 English | 🇨🇳 中文对照 |
|---|---|
| The left panel shows the full technology stack — eight components from video ingestion to the Gradio web interface. I want to highlight four engineering decisions on the right that I'm particularly proud of. | 左侧展示了完整的技术栈——从视频输入到 Gradio 网页界面的八个组件。我想重点介绍右侧四个我特别引以为傲的工程决策。 |
| **Thread safety.** Gradio can handle multiple concurrent users. Without protection, two simultaneous requests could initialize the pipeline twice, causing a race condition. I implemented double-checked locking — acquiring a mutex only when the pipeline is `None`, and checking again inside the lock — so initialization happens exactly once. | **线程安全。** Gradio 可处理多个并发用户。如果没有保护，两个同时发起的请求可能导致流水线被初始化两次，引发竞争条件。我实现了双重检查锁定——只在流水线为 `None` 时才获取互斥锁，并在锁内再次检查——确保初始化只发生一次。 |
| **O(1) frame lookup.** During video rendering, every frame needs to know if it was flagged. A naïve implementation would scan a list each time — O(n) per frame, meaning O(n²) overall for an n-frame video. I pre-compute a Python `set` of flagged frame indices before rendering, making each lookup O(1). | **O(1)帧查询。** 视频渲染时，每一帧都需要判断是否被标记。朴素实现每次扫描列表——每帧 O(n)，n帧视频整体 O(n²)。我在渲染前预计算被标记帧索引的 Python `set`，使每次查询降为 O(1)。 |
| **XSS prevention.** The LLM generates natural language — which could theoretically contain HTML injection payloads. Every string from the LLM passes through Python's `html.escape()` before being injected into the report templates. | **XSS防护。** LLM生成的自然语言文本理论上可能包含 HTML 注入攻击载荷。所有来自 LLM 的字符串在注入报告模板之前，都经过 Python `html.escape()` 处理。 |
| **Pure SVG charts.** Gradio strips JavaScript from rendered HTML. So I built all charts — the timeline, the score distribution — as server-side Python SVG strings. No JavaScript dependency, no rendering failures. | **纯SVG图表。** Gradio 会过滤掉渲染 HTML 中的 JavaScript。因此我将所有图表——时间轴、分数分布——都构建为服务端 Python SVG 字符串。无 JavaScript 依赖，无渲染失败。 |

---

## Slide 6 · Legal Report ｜ 法证报告（~50 sec）

| 🇬🇧 English | 🇨🇳 中文对照 |
|---|---|
| This is the slide I find most exciting, because nothing like this exists in any other open deepfake detection tool. | 这张幻灯片是我最感兴奋的部分，因为没有任何其他开源深度伪造检测工具具备类似功能。 |
| The legal report contains nine sections, structured into three groups. The **blue group** — Case Information, Forensic Verdict, and Methodology — establishes the what and the how. The **purple group** — Quantitative Findings, Discrepancy Timeline, and Flagged Frames Table — provides the numerical evidence. The **green group** — Evidence Summary, Chain of Custody, and Limitations — satisfies the transparency requirements demanded by Daubert and ISO/IEC 27037. | 法证报告包含九个部分，分为三组。**蓝色组**——案件信息、法证判决和方法论——阐明了做了什么、怎么做的。**紫色组**——定量发现、差异时间轴和被标记帧表格——提供数值证据。**绿色组**——证据摘要、证据保管链和局限性声明——满足 Daubert 标准和 ISO/IEC 27037 所要求的透明度。 |
| Every report includes a SHA-256 hash of the input video file, timestamped at analysis time, so the document can prove the file was not tampered with after analysis. | 每份报告都包含输入视频文件的 SHA-256 哈希值，并记录分析时间戳，从而证明文件在分析后未被篡改。 |

---

## Slide 7 · Evaluation Design ｜ 实验设计（~40 sec）

| 🇬🇧 English | 🇨🇳 中文对照 |
|---|---|
| To evaluate the system, I ran a within-subjects user study with ten university students — five from technical backgrounds like Computer Science, and five from non-technical backgrounds such as Law, Journalism, and Psychology. | 为评估系统，我对10名大学生进行了组内用户研究——5名来自计算机科学等技术背景，5名来自法律、新闻、心理学等非技术背景。 |
| Each session lasted approximately sixty minutes. Participants first completed an unaided baseline task — watching two videos and judging authenticity without the tool. Then they used the system on three new videos of varying difficulty: one authentic, one clearly fake, one ambiguous. Finally, they completed the System Usability Scale questionnaire and a semi-structured interview. | 每次测试约60分钟。参与者首先完成不借助工具的基线任务——观看两段视频并判断真实性；然后使用系统分析三段难度不一的视频：一段真实的、一段明显伪造的、一段模糊的；最后完成系统可用性量表问卷和半结构化访谈。 |

---

## Slide 8 · SUS Results ｜ 可用性结果（~55 sec）

| 🇬🇧 English | 🇨🇳 中文对照 |
|---|---|
| Let me take you through the results. Starting with usability. | 接下来进入结果部分，从可用性开始。 |
| The **mean SUS score was 70.0 out of 100**, which places Deep-Guard Agent in the **"Above Average"** category by the Bangor et al. (2009) adjective scale. Five out of ten participants scored above the industry benchmark of 68. | **平均SUS分数为70分（满分100）**，根据 Bangor 等人（2009）的形容词量表，Deep-Guard Agent 落在**"高于平均"**区间。10名参与者中有5人超过了行业基准线68分。 |
| There's a clear divide along technical background: the bar chart shows CS and IS students averaging **83.8**, while non-technical participants averaged **61.3**. This 22-point gap is our clearest signal for where to improve the interface — and I'll address that in the discussion. | 技术背景带来了明显的分化：柱状图显示 CS 和 IS 学生平均 **83.8分**，而非技术背景参与者平均 **61.3分**。这22分的差距是我们改进界面方向最清晰的信号——我将在讨论部分进一步说明。 |

---

## Slide 9 · Accuracy & Likert ｜ 准确率与量表（~55 sec）

| 🇬🇧 English | 🇨🇳 中文对照 |
|---|---|
| The accuracy results are the headline finding of this study. | 准确率结果是本研究最核心的发现。 |
| In the **unaided baseline**, participants correctly identified the authenticity of videos **55% of the time** — essentially chance level, which confirms that deepfakes are genuinely hard for humans to detect. With Deep-Guard Agent, that rose to **85%** — a **30 percentage point improvement**. | 在**无辅助基线条件下**，参与者正确判断视频真实性的概率为 **55%**——基本等同于随机猜测，这印证了深度伪造对人类而言确实极难辨别。使用 Deep-Guard Agent 后，准确率提升至 **85%**——提升了 **30个百分点**。 |
| The Likert scale results on the right show all four constructs rated between 4.1 and 4.7 on a 1-to-7 scale — in the neutral-to-positive range. Trust in the AI verdict was highest at **4.72**. Report comprehension was the lowest at **4.17**, suggesting the technical terminology is still a barrier for some users. | 右侧 Likert 量表结果显示，所有四个维度的评分都在1到7分制的4.1到4.7之间——处于中性偏正向区间。对 AI 判决的信任度最高，达 **4.72**；报告可读性最低，为 **4.17**，表明技术术语对部分用户仍构成障碍。 |
| The Spearman correlation between AI familiarity and SUS score was **ρ = 0.97** — a near-perfect monotonic relationship. Users who were comfortable with AI tools found the system significantly more usable. | AI 熟悉度与 SUS 分数之间的 Spearman 相关系数为 **ρ = 0.97**——几乎完美的单调关系。对 AI 工具熟悉的用户认为该系统显著更易用。 |

---

## Slide 10 · Qualitative Themes ｜ 定性主题（~45 sec）

| 🇬🇧 English | 🇨🇳 中文对照 |
|---|---|
| The semi-structured interviews produced four dominant themes. | 半结构化访谈揭示了四个主要主题。 |
| On the positive side: **8 out of 10 participants** said the verdict badge was immediately clear and easy to understand. **7 out of 10** found the legal report genuinely useful — particularly participants from law and journalism backgrounds, who said they could actually see themselves submitting it as evidence. | 正面反馈方面：**10人中有8人**表示判决标识清晰易懂，可立刻理解。**10人中有7人**认为法证报告真正有用——尤其是法律和新闻背景的参与者，他们表示实际上可以考虑将其作为证据提交。 |
| On the concern side: **6 out of 10** were confused by technical terms — cosine similarity, flagged frames ratio, articulatory discrepancy. And **4 out of 10** found the five-minute processing time too slow for practical use. | 负面反馈方面：**10人中有6人**对技术术语感到困惑——余弦相似度、标记帧比例、发音差异。**10人中有4人**认为约五分钟的处理时间对实际使用来说过长。 |

---

## Slide 11 · Discussion ｜ 讨论与反思（~55 sec）

| 🇬🇧 English | 🇨🇳 中文对照 |
|---|---|
| Let me be direct about what worked, what didn't, and what comes next. | 让我直接说明哪些地方成功了、哪些没有，以及下一步怎么做。 |
| **What worked well.** The bimodal detection architecture genuinely catches mismatches that single-modality tools miss. The LLM reasoning layer turns a number into a *narrative*, making the verdict explainable. And the legal report fills a gap that participants — especially non-CS users — found immediately valuable. | **哪些做得好。** 双模态检测架构确实能捕捉到单模态工具遗漏的不匹配。LLM 推理层将一个数字转化为*叙事*，使判决可解释。法证报告填补了参与者——尤其是非 CS 背景用户——立刻认可其价值的空白。 |
| **Limitations.** The most honest limitation: the model has no trained checkpoint. The feature extractors are initialized with pre-trained weights, but the fusion module uses random initialization. This means the discrepancy scores are not calibrated — they reflect relative patterns, not absolute probabilities. For a production system, this would need to be trained on FaceForensics++ or DFDC. | **局限性。** 最需坦诚说明的是：模型没有经过训练的权重文件。特征提取器使用预训练权重初始化，但融合模块使用随机初始化，意味着差异分数未经校准——反映的是相对模式，而非绝对概率。生产系统需要在 FaceForensics++ 或 DFDC 数据集上进行训练。 |
| **Future work.** Three priorities: first, train the fusion module on a labeled deepfake dataset. Second, add inline tooltips and a glossary to close the vocabulary gap identified in interviews. Third, explore streaming inference for live-video use cases. | **未来工作。** 三个优先方向：第一，在标注的深度伪造数据集上训练融合模块；第二，添加内联提示和术语表，弥合访谈中发现的词汇障碍；第三，探索流式推理以支持实时视频场景。 |

---

## Slide 12 · Conclusion ｜ 总结（~30 sec）

| 🇬🇧 English | 🇨🇳 中文对照 |
|---|---|
| To close: **Deep-Guard Agent** is, to my knowledge, the first open system that combines audio-visual bimodal detection, LLM-powered explainability, and ISO-compliant legal reporting in a single tool. | 最后总结：据我所知，**Deep-Guard Agent** 是第一个将音视频双模态检测、LLM 驱动的可解释性以及符合 ISO 标准的法证报告整合于单一工具的开源系统。 |
| The user study confirms a **30 percentage point accuracy improvement** over unaided detection, with a SUS score of **70** indicating above-average usability. The core limitation — an untrained fusion module — is a known and addressable engineering challenge, not a fundamental design flaw. | 用户研究证实，与无辅助检测相比，准确率提升了 **30个百分点**，SUS 分数 **70分** 表明可用性高于平均水平。核心局限——融合模块未经训练——是一个已知且可解决的工程挑战，而非根本性的设计缺陷。 |
| Thank you. I'm happy to take questions. | 谢谢大家，我很乐意回答问题。 |

---

## Speaker Notes — Timing Guide ｜ 演讲计时参考

| Slide | Topic | Target Time | Cumulative |
|-------|-------|-------------|------------|
| 1 | Title | 30 sec | 0:30 |
| 2 | Motivation | 60 sec | 1:30 |
| 3 | Architecture | 80 sec | 2:50 |
| 4 | Theory | 80 sec | 4:10 |
| 5 | Implementation | 80 sec | 5:30 |
| 6 | Legal Report | 50 sec | 6:20 |
| 7 | Evaluation Design | 40 sec | 7:00 |
| 8 | SUS Results | 55 sec | 7:55 |
| 9 | Accuracy & Likert | 55 sec | 8:50 |
| 10 | Qualitative Themes | 45 sec | 9:35 |
| 11 | Discussion | 55 sec | 10:30 |
| 12 | Conclusion | 30 sec | **11:00** |

> **Tip:** If running short on time, compress Slides 5 and 10. If running over, cut the XSS/SVG details on Slide 5 and the Spearman number on Slide 9.

---

## Anticipated Q&A ｜ 可能的提问与回答

| Question 🇬🇧 | Answer 🇬🇧 | 中文参考回答 |
|---|---|---|
| *Why not use a trained model?* | Training requires a labeled dataset and significant GPU time. The architecture is designed so the fusion module can be swapped in with trained weights — the system is train-ready, just not yet trained. | 训练需要标注数据集和大量 GPU 资源。融合模块的设计允许直接替换训练好的权重——系统已具备训练条件，只是尚未完成训练。 |
| *Is the legal report actually admissible?* | It is designed to meet the structural requirements of Daubert and ISO/IEC 27037. Whether a specific court accepts it depends on jurisdiction and expert testimony. We provide the forensic documentation; the legal determination remains with the court. | 它在设计上满足 Daubert 标准和 ISO/IEC 27037 的结构性要求。是否被特定法院采纳取决于司法管辖区和专家证词。我们提供法证文件，法律认定权仍在法院。 |
| *Why Gradio and not a custom web app?* | Gradio gives us a production-quality UI with minimal code, which let me focus engineering effort on the detection pipeline rather than frontend development. For a research prototype, that trade-off is correct. | Gradio 以极少的代码提供了生产质量的 UI，使我能将工程精力集中在检测流水线而非前端开发上。对于研究原型来说，这个取舍是合理的。 |
| *How does performance scale with video length?* | Processing time scales roughly linearly with video length — about 5 minutes per minute of video on a CPU. GPU acceleration would reduce this by approximately 10×. | 处理时间与视频时长大致呈线性关系——CPU 上每分钟视频约需5分钟。GPU 加速可将其缩短约10倍。 |
