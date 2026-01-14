# 場所寫作風格：主要指南

本指南概述寫作風格在不同發表場所之間的差異。了解這些差異對於撰寫讀起來像各場所真實發表物的論文至關重要。

**最後更新**：2024

---

## 風格光譜

科學寫作風格存在於從**廣泛可及**到**深度技術**的光譜上：

```
可及性高 ◄─────────────────────────────────────────────► 技術性高

Nature/Science    PNAS    Cell    IEEE Trans    NeurIPS    專門
   │                │       │         │            │       期刊
   │                │       │         │            │         │
   ▼                ▼       ▼         ▼            ▼         ▼
一般             混合深度  深度     領域        密集 ML    僅專家
讀者                      生物學   專家        研究人員
```

## 快速風格參考

| 場所類型 | 讀者 | 語氣 | 語態 | 摘要風格 |
|------------|----------|------|-------|----------------|
| **Nature/Science** | 受教育的非專家 | 可及、引人入勝 | 主動，第一人稱可 | 流暢段落，無術語 |
| **Cell Press** | 生物學家 | 機制導向、精確 | 混合 | Summary + eTOC blurb + Highlights |
| **醫學（NEJM/Lancet）** | 臨床醫師 | 以證據為焦點 | 正式 | 結構化（Background/Methods/Results/Conclusions） |
| **PLOS/BMC** | 研究人員 | 標準學術 | 中性 | IMRaD 結構化或流暢 |
| **IEEE/ACM** | 工程師/CS | 技術性 | 被動常見 | 簡潔、技術性 |
| **ML 會議** | ML 研究人員 | 密集技術性 | 混合 | 數字優先、關鍵結果 |
| **NLP 會議** | NLP 研究人員 | 技術性 | 多樣 | 任務導向、基準測試 |

---

## 高影響力期刊（Nature、Science、Cell）

### 核心理念

高影響力多學科期刊優先考慮**廣泛重要性**而非技術深度。問題不是「這在技術上健全嗎？」而是「這個領域以外的科學家為何應該關心？」

### 關鍵寫作原則

1. **從大局開始**：以為何這對科學/社會重要開場
2. **減少術語**：定義專門術語；偏好常用詞
3. **講述故事**：結果應以敘事方式流動，而非資料傾倒
4. **強調意涵**：這對我們的理解有何改變？
5. **可及的圖片**：示意圖和模型優於原始資料圖

### 結構差異

**Nature/Science** vs. **專門期刊**：

| 元素 | Nature/Science | 專門期刊 |
|---------|---------------|---------------------|
| 引言 | 3-4 段，廣泛 → 具體 | 詳盡的文獻回顧 |
| 方法 | 通常在補充資料或簡短 | 正文中完整細節 |
| 結果 | 按發現/故事組織 | 按實驗組織 |
| 討論 | 意涵優先，然後注意事項 | 詳細與文獻比較 |
| 圖片 | 概念示意圖受重視 | 強調原始資料 |

### 範例：同一發現，不同風格

**Nature 風格**：
> 「We discovered that protein X acts as a molecular switch controlling cell fate decisions during development, resolving a longstanding question about how stem cells choose their destiny.」

**專門期刊風格**：
> 「Using CRISPR-Cas9 knockout in murine embryonic stem cells (mESCs), we demonstrate that protein X (encoded by gene ABC1) regulates the expression of pluripotency factors Oct4, Sox2, and Nanog through direct promoter binding, as confirmed by ChIP-seq analysis (n=3 biological replicates, FDR < 0.05).」

---

## 醫學期刊（NEJM、Lancet、JAMA、BMJ）

### 核心理念

醫學期刊優先考慮**臨床相關性**和**病人結果**。每一個發現都必須連結到實務。

### 關鍵寫作原則

1. **以病人為中心的語言**：「Patients receiving treatment X」而非「Treatment X subjects」
2. **證據強度**：根據研究設計適當保守措辭
3. **臨床可行性**：對執業醫師的「So what?」
4. **絕對數字**：報告絕對風險降低，不僅是相對
5. **結構化摘要**：必須帶有標籤的章節

### 結構化摘要格式（醫學）

```
Background: [1-2 句關於問題和理由]

Methods: [研究設計、場域、參與者、介入措施、結果、分析]

Results: [主要結果含信賴區間、次要結果、不良事件]

Conclusions: [臨床意涵、承認限制]
```

### 證據語言慣例

| 研究設計 | 適當語言 |
|-------------|---------------------|
| RCT | 「Treatment X reduced mortality by...」 |
| 觀察性 | 「Treatment X was associated with reduced mortality...」 |
| 病例系列 | 「These findings suggest that treatment X may...」 |
| 個案報告 | 「This case illustrates that treatment X can...」 |

---

## ML/AI 會議（NeurIPS、ICML、ICLR、CVPR）

### 核心理念

ML 會議重視**新穎性**、**嚴謹的實驗**和**可重現性**。重點是透過實證證據推進技術水準。

### 關鍵寫作原則

1. **貢獻條目**：引言中編號列表，精確說明什麼是新的
2. **基準線至關重要**：與強大、近期的基準線比較
3. **消融實驗預期**：展示您的方法中哪些部分重要
4. **可重現性**：種子、超參數、計算需求
5. **限制章節**：誠實承認（越來越需要）

### 引言結構（ML 會議）

```
[第 1 段：問題動機 - 為何這重要]

[第 2 段：現有方法的限制]

[第 3 段：概要說明我們的方法]

Our contributions are as follows:
• We propose [方法名], a novel approach to [問題] that [關鍵創新].
• We provide theoretical analysis showing [保證/性質].
• We demonstrate state-of-the-art results on [基準測試], improving over [基準線] by [X%].
• We release code and models at [anonymous URL for review].
```

### 摘要風格（ML 會議）

ML 摘要是**密集且聚焦數字的**：

> 「We present TransformerX, a novel architecture for long-range sequence modeling that achieves O(n log n) complexity while maintaining expressivity. On the Long Range Arena benchmark, TransformerX achieves 86.2% average accuracy, outperforming Transformer (65.4%) and Performer (78.1%). On language modeling, TransformerX matches GPT-2 perplexity (18.4) using 40% fewer parameters. We provide theoretical analysis showing TransformerX can approximate any continuous sequence-to-sequence function.」

### 實驗章節預期

1. **資料集**：標準基準測試、資料集統計
2. **基準線**：近期強方法、公平比較
3. **主要結果表**：清晰、全面
4. **消融研究**：系統性移除/修改元件
5. **分析**：錯誤分析、定性範例、失敗案例
6. **計算成本**：訓練時間、推論速度、記憶體

---

## CS 會議（ACL、EMNLP、CHI、SIGKDD）

### ACL/EMNLP（NLP）

- **任務導向**：清晰的問題定義
- **基準測試密集**：標準資料集（GLUE、SQuAD 等）
- **重視錯誤分析**：哪裡會失敗？
- **人工評估**：常與自動指標並行預期
- **倫理考量**：偏差、公平性、環境成本

### CHI（人機互動）

- **以使用者為中心**：聚焦人類，而非僅技術
- **研究設計細節**：參與者招募、IRB 核准
- **接受定性**：訪談研究、民族誌有效
- **設計意涵**：對實務者的具體收穫
- **無障礙**：考慮多元使用者族群

### SIGKDD（資料探勘）

- **強調可擴展性**：處理大資料
- **真實世界應用**：產業資料集受重視
- **效率指標**：時間和空間複雜度
- **方法或應用的新穎性**：兩條路徑都有效

---

## 在場所類型之間轉換

### 期刊 → ML 會議

將期刊論文轉換為會議格式時：

1. **精簡引言**：移除詳盡背景
2. **新增貢獻列表**：明確列舉貢獻
3. **重組結果**：組織為實驗，新增消融實驗
4. **移除獨立討論**：簡短整合解釋
5. **新增可重現性章節**：種子、超參數、程式碼

### ML 會議 → 期刊

將會議論文擴展為期刊時：

1. **擴展相關工作**：全面的文獻回顧
2. **詳細方法**：完整的演算法描述
3. **更多實驗**：額外資料集、分析
4. **延伸討論**：意涵、限制、未來工作
5. **附錄 → 正文**：將重要細節上移

### 專門 → 高影響力期刊

從專門場所目標 Nature/Science/Cell 時：

1. **以重要性引導**：為何這在廣泛層面上重要？
2. **減少 80% 術語**：替換技術用語
3. **新增概念圖**：示意圖、模型，不僅是資料
4. **故事驅動的結果**：敘事流程，非逐實驗陳述
5. **擴大討論**：超越子領域的意涵

---

## 語態與語氣指南

### 主動 vs. 被動語態

| 場所 | 偏好 | 範例 |
|-------|-----------|---------|
| Nature/Science | 鼓勵主動 | 「We discovered that...」 |
| Cell | 混合 | 「Our results demonstrate...」 |
| 醫學 | 被動常見 | 「Patients were randomized to...」 |
| IEEE | 傳統被動 | 「The algorithm was implemented...」 |
| ML 會議 | 偏好主動 | 「We propose a method that...」 |

### 第一人稱使用

| 場所 | 第一人稱 | 範例 |
|-------|-------------|---------|
| Nature/Science | 是（we） | 「We show that...」 |
| Cell | 是（we） | 「We found that...」 |
| 醫學 | 有時 | 「We conducted a trial...」 |
| IEEE | 較不常見 | 偏好「This paper presents...」 |
| ML 會議 | 是（we） | 「We introduce...」 |

### 保守措辭與確定性

| 聲稱強度 | 語言 |
|---------------|----------|
| 強 | 「X causes Y」（僅有因果證據時） |
| 中等 | 「X is associated with Y」/「X leads to Y」 |
| 保守 | 「X may contribute to Y」/「X suggests that...」 |
| 推測性 | 「It is possible that X...」/「One interpretation is...」 |

---

## 各場所常見風格錯誤

### Nature/Science 投稿

❌ 過於技術性：「We used CRISPR-Cas9 with sgRNAs targeting exon 3...」
✅ 可及性：「Using gene-editing technology, we disabled the gene...」

❌ 乾燥的開場：「Protein X is involved in cellular signaling...」
✅ 引人入勝的開場：「How do cells decide their fate? We discovered that...」

### ML 會議投稿

❌ 模糊的貢獻：「We present a new method for X」
✅ 具體的貢獻：「We propose Method Y that achieves Z% improvement on benchmark W」

❌ 缺少消融實驗：僅展示完整方法結果
✅ 完整：表格展示每個元件的貢獻

### 醫學期刊投稿

❌ 缺少絕對數字：「50% reduction in risk」
✅ 完整：「50% relative reduction (ARR 2.5%, NNT 40)」

❌ 觀察性資料的因果語言：「Treatment caused improvement」
✅ 適當：「Treatment was associated with improvement」

---

## 投稿前快速檢核表

### 所有場所
- [ ] 摘要符合場所風格（流暢 vs. 結構化）
- [ ] 語態/語氣適合讀者
- [ ] 術語程度適當
- [ ] 圖片符合場所預期
- [ ] 引用格式正確

### 高影響力期刊（Nature/Science/Cell）
- [ ] 第一段清楚說明廣泛重要性
- [ ] 非專家可理解摘要
- [ ] 故事驅動的結果敘事
- [ ] 包含概念圖
- [ ] 強調意涵

### ML 會議
- [ ] 引言中有貢獻列表
- [ ] 包含強基準線
- [ ] 有消融研究
- [ ] 可重現性資訊完整
- [ ] 承認限制

### 醫學期刊
- [ ] 結構化摘要（如需要）
- [ ] 以病人為中心的語言
- [ ] 證據強度適當
- [ ] 報告絕對數字
- [ ] CONSORT/STROBE 合規性

---

## 另請參閱

- `nature_science_style.md` - Nature/Science 詳細寫作指南
- `cell_press_style.md` - Cell 系列期刊慣例
- `medical_journal_styles.md` - NEJM、Lancet、JAMA、BMJ 指南
- `ml_conference_style.md` - NeurIPS、ICML、ICLR、CVPR 慣例
- `cs_conference_style.md` - ACL、CHI、SIGKDD 指南
- `reviewer_expectations.md` - 各場所審稿人關注什麼
