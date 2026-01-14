# 資訊科學會議寫作風格指南

ACL、EMNLP、NAACL（自然語言處理）、CHI、CSCW（人機互動）、SIGKDD、WWW、SIGIR（資料探勘/資訊檢索）及其他主要資訊科學會議的完整寫作指南。

**最後更新**：2024

---

## 概述

資訊科學會議涵蓋具有不同寫作文化的多元子領域。本指南涵蓋自然語言處理、人機互動和資料探勘/資訊檢索場域，每個都有獨特的期望和評估標準。

---

# 第一部分：自然語言處理會議（ACL、EMNLP、NAACL）

## 自然語言處理寫作理念

> 「在標準基準測試上有強勁的實證結果，並附有深入分析。」

自然語言處理論文在實證嚴謹性與語言學洞見之間取得平衡。除了自動化指標外，人工評估也越來越重要。

## 讀者與語調

### 目標讀者
- 自然語言處理研究者和計算語言學家
- 熟悉 transformer（轉換器）架構、標準基準測試
- 期望可重複的結果和錯誤分析

### 語調特徵
| 特徵 | 說明 |
|---------------|-------------|
| **任務聚焦（Task-focused）** | 清晰的問題定義 |
| **基準測試導向（Benchmark-oriented）** | 強調標準數據集 |
| **分析豐富（Analysis-rich）** | 錯誤分析、定性範例 |
| **可重複（Reproducible）** | 完整的實作細節 |

## Abstract（摘要）（自然語言處理風格）

### 結構
- **任務/問題**（1 句）
- **先前工作的局限**（1 句）
- **你的方法**（1-2 句）
- **基準測試結果**（2 句）
- **分析發現**（可選，1 句）

### 摘要範例

```
Coreference resolution remains challenging for pronouns with distant or
ambiguous antecedents. Prior neural approaches struggle with these
difficult cases due to limited context modeling. We introduce
LongContext-Coref, a retrieval-augmented coreference model that
dynamically retrieves relevant context from document history. On the
OntoNotes 5.0 benchmark, LongContext-Coref achieves 83.4 F1, improving
over the previous state-of-the-art by 1.2 points. On the challenging
WinoBias dataset, we reduce gender bias by 34% while maintaining
accuracy. Qualitative analysis reveals that our model successfully
resolves pronouns requiring world knowledge, a known weakness of
prior approaches.
```

## 自然語言處理論文結構

```
├── Introduction（緒論）
│   ├── 任務動機
│   ├── 先前工作的局限
│   ├── 你的貢獻
│   └── 貢獻要點
├── Related Work（相關工作）
├── Method（方法）
│   ├── 問題定義
│   ├── 模型架構
│   └── 訓練程序
├── Experiments（實驗）
│   ├── 數據集（附統計數據）
│   ├── 基準線
│   ├── 主要結果
│   ├── 分析
│   │   ├── 錯誤分析
│   │   ├── 消融研究（Ablation study）
│   │   └── 定性範例
│   └── 人工評估（如適用）
├── Discussion / Limitations（討論/局限）
└── Conclusion（結論）
```

## 自然語言處理特定要求

### 數據集
- 使用**標準基準測試**：GLUE、SQuAD、CoNLL、OntoNotes
- 報告**數據集統計**：train/dev/test 大小
- **數據預處理**：記錄所有步驟

### 評估指標
- **適合任務的指標**：F1、BLEU、ROUGE、accuracy（準確率）
- **統計顯著性**：Paired bootstrap（成對自助法）、p 值
- **多次運行**：報告跨種子的 mean ± std

### 人工評估
生成任務越來越期望人工評估：
- **標註者詳情**：人數、資格、一致性
- **評估方案**：指南、介面、報酬
- **標註者間一致性**：Cohen's κ 或 Krippendorff's α

### 人工評估表格範例

```
Table 3: Human Evaluation Results (100 samples, 3 annotators)
─────────────────────────────────────────────────────────────
Method        | Fluency | Coherence | Factuality | Overall
─────────────────────────────────────────────────────────────
Baseline      |   3.8   |    3.2    |    3.5     |   3.5
GPT-3.5       |   4.2   |    4.0    |    3.7     |   4.0
Our Method    |   4.4   |    4.3    |    4.1     |   4.3
─────────────────────────────────────────────────────────────
Inter-annotator κ = 0.72. Scale: 1-5 (higher is better).
```

## ACL 特定注意事項

- **ARR（ACL Rolling Review）**：跨 ACL 場域的共享審稿系統
- **Responsible NLP checklist（負責任自然語言處理檢查清單）**：倫理、局限、風險
- **Long（8 頁）vs. Short（4 頁）**：不同期望
- **Findings papers**：較低層級的接受類別

---

# 第二部分：人機互動會議（CHI、CSCW、UIST）

## 人機互動寫作理念

> 「技術服務於人類——先理解使用者，再設計和評估。」

人機互動論文本質上是**以使用者為中心**。僅有技術新穎性是不夠的；理解人類需求並展示使用者效益是必要的。

## 讀者與語調

### 目標讀者
- 人機互動研究者和從業者
- 使用者體驗設計師和產品開發者
- 跨學科（資訊科學、心理學、設計、社會科學）

### 語調特徵
| 特徵 | 說明 |
|---------------|-------------|
| **以使用者為中心（User-centered）** | 聚焦人，而非技術 |
| **設計導向（Design-informed）** | 根植於設計思維 |
| **實證性（Empirical）** | 使用者研究提供證據 |
| **反思性（Reflective）** | 考量更廣泛的影響 |

## 人機互動摘要

### 聚焦使用者和影響

```
Video calling has become essential for remote collaboration, yet
current interfaces poorly support the peripheral awareness that makes
in-person work effective. Through formative interviews with 24 remote
workers, we identified three key challenges: difficulty gauging
colleague availability, lack of ambient presence cues, and interruption
anxiety. We designed AmbientOffice, a peripheral display system that
conveys teammate presence through subtle ambient visualizations. In a
two-week deployment study with 18 participants across three distributed
teams, AmbientOffice increased spontaneous collaboration by 40% and
reduced perceived isolation (p<0.01). Participants valued the system's
non-intrusive nature and reported feeling more connected to remote
colleagues. We discuss implications for designing ambient awareness
systems and the tension between visibility and privacy in remote work.
```

## 人機互動論文結構

### 設計研究/系統論文

```
├── Introduction（緒論）
│   ├── 以人為本的問題陳述
│   ├── 為何技術可以幫助
│   └── 貢獻摘要
├── Related Work（相關工作）
│   ├── 領域背景
│   ├── 先前系統
│   └── 理論框架
├── Formative Work（形成性工作）（常見）
│   ├── 訪談/觀察
│   └── 設計需求
├── System Design（系統設計）
│   ├── 設計理由
│   ├── 實作
│   └── 介面導覽
├── Evaluation（評估）
│   ├── 研究設計
│   ├── 參與者
│   ├── 程序
│   ├── 發現（量化+質性）
│   └── 局限
├── Discussion（討論）
│   ├── 設計啟示
│   ├── 可推廣性
│   └── 未來工作
└── Conclusion（結論）
```

### 質性/訪談研究

```
├── Introduction（緒論）
├── Related Work（相關工作）
├── Methods（方法）
│   ├── 參與者
│   ├── 程序
│   ├── 數據收集
│   └── 分析方法（主題分析、扎根理論等）
├── Findings（發現）
│   ├── 主題 1（附引述）
│   ├── 主題 2（附引述）
│   └── 主題 3（附引述）
├── Discussion（討論）
│   ├── 設計啟示
│   ├── 研究啟示
│   └── 局限
└── Conclusion（結論）
```

## 人機互動特定要求

### 參與者報告
- **人口統計資料**：年齡、性別、相關經驗
- **招募方式**：如何及在哪裡招募
- **報酬**：支付金額和類型
- **IRB 批准**：倫理委員會聲明

### 發現中的引述
使用直接引述來支持發現：
```
Participants valued the ambient nature of the display. As P7 described:
"It's like having a window to my teammate's office. I don't need to
actively check it, but I know they're there." This passive awareness
reduced the barrier to initiating contact.
```

### 設計啟示章節
將發現轉化為可行動的指導：
```
**Implication 1: Support peripheral awareness without demanding attention.**
Ambient displays should be visible in peripheral vision but not require
active monitoring. Designers should consider calm technology principles.

**Implication 2: Balance visibility with privacy.**
Users want to share presence but fear surveillance. Systems should
provide granular controls and make visibility mutual.
```

## CHI 特定注意事項

- **貢獻類型**：實證性、工具性、方法論、理論性
- **ACM 格式**：`acmart` 文件類別加 `sigchi` 選項
- **無障礙設計**：期望替代文字、包容性語言
- **貢獻聲明**：要求每位作者的貢獻說明

---

# 第三部分：資料探勘與資訊檢索（SIGKDD、WWW、SIGIR）

## 資料探勘寫作理念

> 「具有展示實際影響的可擴展真實世界數據方法。」

資料探勘論文強調**可擴展性**、**真實世界適用性**和**紮實的實驗方法論**。

## 讀者與語調

### 目標讀者
- 資料科學家和機器學習工程師
- 產業研究者
- 應用機器學習從業者

### 語調特徵
| 特徵 | 說明 |
|---------------|-------------|
| **可擴展（Scalable）** | 處理大型數據集 |
| **實用（Practical）** | 真實世界應用 |
| **可重複（Reproducible）** | 分享數據集和程式碼 |
| **產業化（Industrial）** | 重視產業數據集 |

## KDD 摘要

### 強調規模和應用

```
Fraud detection in e-commerce requires processing millions of
transactions in real-time while adapting to evolving attack patterns.
We present FraudShield, a graph neural network framework for real-time
fraud detection that scales to billion-edge transaction graphs. Unlike
prior methods that require full graph access, FraudShield uses
incremental updates with O(1) inference cost per transaction. On a
proprietary dataset of 2.3 billion transactions from a major e-commerce
platform, FraudShield achieves 94.2% precision at 80% recall,
outperforming production baselines by 12%. The system has been deployed
at [Company], processing 50K transactions per second and preventing
an estimated $400M in annual fraud losses. We release an anonymized
benchmark dataset and code.
```

## KDD 論文結構

```
├── Introduction（緒論）
│   ├── 問題和影響
│   ├── 技術挑戰
│   ├── 你的方法
│   └── 貢獻
├── Related Work（相關工作）
├── Preliminaries（預備知識）
│   ├── 問題定義
│   └── 符號說明
├── Method（方法）
│   ├── 概述
│   ├── 技術組件
│   └── 複雜度分析
├── Experiments（實驗）
│   ├── 數據集（附規模統計）
│   ├── 基準線
│   ├── 主要結果
│   ├── 可擴展性實驗
│   ├── 消融研究
│   └── 案例研究/部署
└── Conclusion（結論）
```

## KDD 特定要求

### 可擴展性
- **數據集大小**：報告節點數、邊數、樣本數
- **運行時間分析**：牆鐘時間比較
- **複雜度**：說明時間和空間複雜度
- **擴展實驗**：顯示性能 vs. 數據大小

### 產業部署
- **案例研究**：真實世界部署故事
- **A/B 測試**：線上評估結果（如適用）
- **生產指標**：業務影響（如可分享）

### 可擴展性表格範例

```
Table 4: Scalability Comparison (runtime in seconds)
──────────────────────────────────────────────────────
Dataset     | Nodes  | Edges  | GCN   | GraphSAGE | Ours
──────────────────────────────────────────────────────
Cora        |  2.7K  |  5.4K  |  0.3  |    0.2    |  0.1
Citeseer    |  3.3K  |  4.7K  |  0.4  |    0.3    |  0.1
PubMed      | 19.7K  | 44.3K  |  1.2  |    0.8    |  0.3
ogbn-arxiv  | 169K   | 1.17M  |  8.4  |    4.2    |  1.6
ogbn-papers | 111M   | 1.6B   |  OOM  |   OOM     | 42.3
──────────────────────────────────────────────────────
```

---

# 第四部分：資訊科學場域的共同元素

## 寫作品質

### 清晰度
- **每句一個想法**
- **使用前先定義術語**
- **使用一致的符號**

### 精確度
- **確切數字**：「23.4%」而非「約 20%」
- **清晰的主張**：除非必要否則避免模糊
- **具體比較**：指名基準線

## 貢獻要點

所有資訊科學場域都使用：
```
Our contributions are:
• We identify [problem/insight]
• We propose [method name] that [key innovation]
• We demonstrate [results] on [benchmarks]
• We release [code/data] at [URL]
```

## 可重複性標準

所有資訊科學場域越來越期望：
- **程式碼可用性**：GitHub 連結（審稿用匿名）
- **數據可用性**：公開數據集或發布計畫
- **完整超參數**：訓練細節完整
- **隨機種子**：精確的重現值
- **計算需求**：訓練時間、推理速度、記憶體
- **運行次數和變異**報告
- **統計顯著性檢驗**

## 倫理和更廣泛影響

### 自然語言處理（ACL/EMNLP）
- **Limitations section（局限章節）**：必須
- **Responsible NLP checklist**：倫理考量
- **偏見分析**：對於影響人的模型

### 人機互動（CHI）
- **IRB/倫理批准**：人類受試者必須
- **知情同意**：描述程序
- **隱私考量**：數據處理

### KDD/WWW
- **社會影響**：考量誤用潛力
- **隱私保護**：敏感數據
- **公平性分析**：當適用時

---

## 場域比較表

| 面向 | ACL/EMNLP | CHI | KDD/WWW | SIGIR |
|--------|-----------|-----|---------|-------|
| **聚焦** | 自然語言處理任務 | 使用者研究 | 可擴展機器學習 | 資訊檢索/搜尋 |
| **評估** | 基準測試 + 人工 | 使用者研究 | 大規模實驗 | 數據集 |
| **理論權重** | 中 | 低 | 中 | 中 |
| **產業價值** | 高 | 中 | 非常高 | 高 |
| **頁數限制** | 8 長 / 4 短 | 10 + refs | 9 + refs | 10 + refs |
| **審稿風格** | ARR | 直接 | 直接 | 直接 |

---

## 投稿前檢查清單

### 所有資訊科學場域
- [ ] 清晰的貢獻聲明
- [ ] 強勁的基準線
- [ ] 可重複性資訊完整
- [ ] 正確的場域模板
- [ ] 匿名化（如果雙盲）

### 自然語言處理特定
- [ ] 標準基準測試結果
- [ ] 包含錯誤分析
- [ ] 人工評估（用於生成）
- [ ] Responsible NLP 檢查清單

### 人機互動特定
- [ ] 說明 IRB 批准
- [ ] 參與者人口統計
- [ ] 發現中的直接引述
- [ ] 設計啟示

### 資料探勘特定
- [ ] 可擴展性實驗
- [ ] 數據集大小統計
- [ ] 運行時間比較
- [ ] 複雜度分析

---

## 另請參閱

- `venue_writing_styles.md` - 風格總覽
- `ml_conference_style.md` - NeurIPS/ICML 風格指南
- `conferences_formatting.md` - 技術格式要求
- `reviewer_expectations.md` - 資訊科學審稿人尋求什麼

