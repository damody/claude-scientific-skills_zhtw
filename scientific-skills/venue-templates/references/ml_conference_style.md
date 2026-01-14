# 機器學習會議寫作風格指南

NeurIPS、ICML、ICLR、CVPR、ECCV、ICCV 及其他主要機器學習和電腦視覺會議的完整寫作指南。

**最後更新**：2024

---

## 概述

ML 會議優先考慮**新穎性（novelty）**、**嚴謹的實驗評估（rigorous empirical evaluation）**和**可重現性（reproducibility）**。論文依據清晰的貢獻、強基準線（baselines）、全面的消融實驗（ablations）和誠實的限制討論來評估。

### 核心理念

> 「用展示取代陳述——您的實驗應該證明您的聲稱，而不僅是您的文字。」

**主要目標**：透過新穎的方法和嚴謹的實驗驗證來推進技術水準。

---

## 讀者與語氣

### 目標讀者

- ML 研究人員和實務者
- 特定子領域的專家
- 熟悉近期文獻
- 期待技術深度和精確性

### 語氣特徵

| 特徵 | 說明 |
|---------------|-------------|
| **技術性** | 密集的方法論細節 |
| **精確** | 精確術語，無歧義 |
| **實證性** | 聲稱有實驗支持 |
| **直接** | 清楚陳述貢獻 |
| **誠實** | 承認限制 |

### 語態

- **第一人稱複數（「we」）**：「We propose...」「Our method...」
- **主動語態**：「We introduce a novel architecture...」
- **自信但審慎**：強烈聲稱需要強力證據

---

## 摘要

### 風格要求

- **密集且聚焦數字**
- **150-250 字**（依場所而異）
- **關鍵結果放前面**：包含具體指標
- **流暢段落**（非結構化）

### 摘要結構

1. **問題**（1 句）：您要解決什麼問題？
2. **現有工作的限制**（1 句）：為何目前方法不足
3. **您的方法**（1-2 句）：您的方法是什麼？
4. **關鍵結果**（2-3 句）：基準測試上的具體數字
5. **重要性**（可選，1 句）：為何這很重要

### 範例摘要（NeurIPS 格式）

```
Transformers have achieved remarkable success in sequence modeling but
suffer from quadratic computational complexity, limiting their application
to long sequences. We introduce FlashAttention-2, an IO-aware exact
attention algorithm that achieves 2x speedup over FlashAttention and up
to 9x speedup over standard attention on sequences up to 16K tokens. Our
key insight is to reduce memory reads/writes by tiling and recomputation,
achieving optimal IO complexity. On the Long Range Arena benchmark,
FlashAttention-2 enables training with 8x longer sequences while matching
standard attention accuracy. Combined with sequence parallelism, we train
GPT-style models on sequences of 64K tokens at near-linear cost. We
release optimized CUDA kernels achieving 80% of theoretical peak FLOPS
on A100 GPUs. Code is available at [anonymous URL].
```

### 摘要禁忌

❌ 「We propose a novel method for X」（模糊，無結果）
❌ 「Our method outperforms baselines」（無具體數字）
❌ 「This is an important problem」（不言自明的聲稱）

✅ 包含具體指標：「achieves 94.5% accuracy, 3.2% improvement」
✅ 包含規模：「on 1M samples」或「16K token sequences」
✅ 包含比較：「2x faster than previous SOTA」

---

## 引言

### 結構（2-3 頁）

ML 引言有獨特結構，包含**編號貢獻（numbered contributions）**。

### 逐段指南

**第 1 段：問題動機**
- 為何這個問題重要？
- 有哪些應用？
- 建立技術挑戰

```
「Large language models have demonstrated remarkable capabilities in
natural language understanding and generation. However, their quadratic
attention complexity presents a fundamental bottleneck for processing
long documents, multi-turn conversations, and reasoning over extended
contexts. As models scale to billions of parameters and context lengths
extend to tens of thousands of tokens, efficient attention mechanisms
become critical for practical deployment.」
```

**第 2 段：現有方法的限制**
- 存在哪些方法？
- 為何它們不足？
- 限制的技術分析

```
「Prior work has addressed this through sparse attention patterns,
linear attention approximations, and low-rank factorizations. While
these methods reduce theoretical complexity, they often sacrifice
accuracy, require specialized hardware, or introduce approximation
errors that compound in deep networks. Exact attention remains
preferable when computational resources permit.」
```

**第 3 段：您的方法（概要）**
- 您的關鍵洞察是什麼？
- 您的方法概念上如何運作？
- 為何它應該成功？

```
「We observe that the primary bottleneck in attention is not computation
but rather memory bandwidth—reading and writing the large N×N attention
matrix dominates runtime on modern GPUs. We propose FlashAttention-2,
which eliminates this bottleneck through a novel tiling strategy that
computes attention block-by-block without materializing the full matrix.」
```

**第 4 段：貢獻列表（關鍵）**

這在 ML 會議是**必須且獨特**的：

```
Our contributions are as follows:

• We propose FlashAttention-2, an IO-aware exact attention algorithm
  that achieves optimal memory complexity O(N²d/M) where M is GPU
  SRAM size.

• We provide theoretical analysis showing that our algorithm achieves
  2-4x fewer HBM accesses than FlashAttention on typical GPU
  configurations.

• We demonstrate 2x speedup over FlashAttention and up to 9x over
  standard PyTorch attention across sequence lengths from 256 to 64K
  tokens.

• We show that FlashAttention-2 enables training with 8x longer
  contexts on the same hardware, unlocking new capabilities for
  long-range modeling.

• We release optimized CUDA kernels and PyTorch bindings at
  [anonymous URL].
```

### 貢獻條目指南

| 好的貢獻條目 | 差的貢獻條目 |
|--------------------------|-------------------------|
| 具體、可量化 | 模糊的聲稱 |
| 自我完整 | 需要閱讀論文才能理解 |
| 彼此不同 | 重疊的條目 |
| 強調新穎性 | 陳述顯而易見的事實 |

### 相關工作的位置

- **在引言中**：簡短定位（1-2 段）
- **獨立章節**：詳細比較（在結尾或結論前）
- **附錄**：如空間有限則放置延伸討論

---

## 方法

### 結構（2-3 頁）

```
METHOD
├── Problem Formulation
├── Method Overview / Architecture
├── Key Technical Components
│   ├── Component 1（含方程式）
│   ├── Component 2（含方程式）
│   └── Component 3（含方程式）
├── Theoretical Analysis（如適用）
└── Implementation Details
```

### 數學符號

- **定義所有符號**：「Let X ∈ ℝ^{N×d} denote the input sequence...」
- **一致的符號**：同一符號在全文代表同一事物
- **為重要方程式編號**：稍後以編號引用

### 演算法虛擬碼

包含清晰的虛擬碼以確保可重現性：

```
Algorithm 1: FlashAttention-2 Forward Pass
─────────────────────────────────────────
Input: Q, K, V ∈ ℝ^{N×d}, block size B_r, B_c
Output: O ∈ ℝ^{N×d}

1:  Divide Q into T_r = ⌈N/B_r⌉ blocks
2:  Divide K, V into T_c = ⌈N/B_c⌉ blocks
3:  Initialize O = 0, ℓ = 0, m = -∞
4:  for i = 1 to T_r do
5:    Load Q_i from HBM to SRAM
6:    for j = 1 to T_c do
7:      Load K_j, V_j from HBM to SRAM
8:      Compute S_ij = Q_i K_j^T
9:      Update running max and sum
10:     Update O_i incrementally
11:   end for
12:   Write O_i to HBM
13: end for
14: return O
```

### 架構圖

- **清晰、出版品質的圖片**
- **標註所有元件**
- **用箭頭顯示資料流**
- **使用一致的視覺語言**

---

## 實驗

### 結構（2-3 頁）

```
EXPERIMENTS
├── Experimental Setup
│   ├── Datasets and Benchmarks
│   ├── Baselines
│   ├── Implementation Details
│   └── Evaluation Metrics
├── Main Results
│   └── 主要比較的表格/圖片
├── Ablation Studies
│   └── 元件分析
├── Analysis
│   ├── Scaling behavior
│   ├── Qualitative examples
│   └── Error analysis
└── Computational Efficiency
```

### 資料集和基準測試

- **使用標準基準測試**：建立可比較性
- **報告資料集統計**：大小、分割、前處理
- **說明非標準選擇**：如使用自訂資料，解釋原因

### 基準線

**對於接受至關重要。** 包含：
- **近期 SOTA**：不僅是舊方法
- **公平比較**：相同計算預算、超參數調整
- **消融版本**：您的方法去除關鍵元件
- **強基準線**：不要挑選弱競爭者

### 主要結果表格

清晰、全面的格式：

```
Table 1: Results on Long Range Arena Benchmark (accuracy %)
──────────────────────────────────────────────────────────
Method          | ListOps | Text  | Retrieval | Image | Path  | Avg
──────────────────────────────────────────────────────────
Transformer     |  36.4   | 64.3  |   57.5    | 42.4  | 71.4  | 54.4
Performer       |  18.0   | 65.4  |   53.8    | 42.8  | 77.1  | 51.4
Linear Attn     |  16.1   | 65.9  |   53.1    | 42.3  | 75.3  | 50.5
FlashAttention  |  37.1   | 64.5  |   57.8    | 42.7  | 71.2  | 54.7
FlashAttn-2     |  37.4   | 64.7  |   58.2    | 42.9  | 71.8  | 55.0
──────────────────────────────────────────────────────────
```

### 消融研究（必須）

展示您的方法中什麼是重要的：

```
Table 2: Ablation Study on FlashAttention-2 Components
──────────────────────────────────────────────────────
Variant                              | Speedup | Memory
──────────────────────────────────────────────────────
Full FlashAttention-2                |   2.0x  |  1.0x
  - without sequence parallelism     |   1.7x  |  1.0x
  - without recomputation            |   1.3x  |  2.4x
  - without block tiling             |   1.0x  |  4.0x
FlashAttention-1 (baseline)          |   1.0x  |  1.0x
──────────────────────────────────────────────────────
```

### 消融實驗應展示什麼

- **每個元件都重要**：移除它會損害性能
- **設計選擇有理由**：為何選擇這個架構/超參數？
- **失敗模式**：方法何時不起作用？
- **敏感度分析**：對超參數的穩健性

---

## 相關工作

### 位置選項

1. **引言之後**：CV 論文常見
2. **結論之前**：NeurIPS/ICML 常見
3. **附錄**：空間緊張時

### 寫作風格

- **按主題組織**：非按時間順序
- **定位您的工作**：與每條研究線的不同之處
- **公平描述**：不要誤解先前工作
- **近期引用**：包含 2023-2024 論文

### 範例結構

```
**Efficient Attention Mechanisms.** Prior work on efficient attention
falls into three categories: sparse patterns (Beltagy et al., 2020;
Zaheer et al., 2020), linear approximations (Katharopoulos et al., 2020;
Choromanski et al., 2021), and low-rank factorizations (Wang et al.,
2020). Our work differs in that we focus on IO-efficient exact
attention rather than approximations.

**Memory-Efficient Training.** Gradient checkpointing (Chen et al., 2016)
and activation recomputation (Korthikanti et al., 2022) reduce memory
by trading compute. We adopt similar ideas but apply them within the
attention operator itself.
```

---

## 限制章節

### 為何重要

**在 NeurIPS、ICML、ICLR 越來越需要。** 誠實的限制：
- 展示科學成熟度
- 指導未來工作
- 防止過度宣傳

### 應包含什麼

1. **方法限制**：何時會失敗？
2. **實驗限制**：什麼沒有測試？
3. **範圍限制**：什麼在範圍之外？
4. **計算限制**：資源需求

### 範例限制章節

```
**Limitations.** While FlashAttention-2 provides substantial speedups,
several limitations remain. First, our implementation is optimized for
NVIDIA GPUs and does not support AMD or other hardware. Second, the
speedup is most pronounced for medium to long sequences; for very short
sequences (<256 tokens), the overhead of our kernel launch dominates.
Third, we focus on dense attention; extending our approach to sparse
attention patterns remains future work. Finally, our theoretical
analysis assumes specific GPU memory hierarchy parameters that may not
hold for future hardware generations.
```

---

## 可重現性

### 可重現性檢核表（NeurIPS/ICML）

大多數 ML 會議要求可重現性檢核表涵蓋：

- [ ] 程式碼可用性
- [ ] 資料集可用性
- [ ] 超參數已指定
- [ ] 隨機種子已報告
- [ ] 計算需求已說明
- [ ] 執行次數和變異已報告
- [ ] 統計顯著性檢定

### 應報告什麼

**超參數**：
```
「We train with Adam (β₁=0.9, β₂=0.999, ε=1e-8) and learning rate 3e-4
with linear warmup over 1000 steps and cosine decay. Batch size is 256
across 8 A100 GPUs. We train for 100K steps (approximately 24 hours).」
```

**隨機種子**：
```
「All experiments are averaged over 3 random seeds (0, 1, 2) with
standard deviation reported in parentheses.」
```

**計算資源**：
```
「Experiments were conducted on 8 NVIDIA A100-80GB GPUs. Total training
time was approximately 500 GPU-hours.」
```

---

## 圖片

### 圖片品質

- **向量圖優先**：PDF、SVG
- **點陣圖高解析度**：300+ dpi
- **出版尺寸可讀**：以實際欄寬測試
- **色盲友善**：除顏色外也使用圖案

### 常見圖片類型

1. **架構圖**：視覺化展示您的方法
2. **性能圖**：學習曲線、縮放行為
3. **比較表格**：主要結果
4. **消融圖片**：元件貢獻
5. **定性範例**：輸入/輸出樣本

### 圖說

自我完整的圖說應解釋：
- 展示什麼
- 如何閱讀圖片
- 關鍵要點

---

## 參考文獻

### 引用格式

- **數字標註 [1]** 或 **作者-年份 (Smith et al., 2023)**
- 查閱場所特定要求
- 全文一致

### 參考文獻指南

- **引用近期工作**：預期有 2022-2024 論文
- **不要過度引用自己**：引發偏見疑慮
- **適當引用 arxiv**：有發表版本時使用發表版本
- **包含所有相關先前工作**：缺少引用會損害審查

---

## 場所特定說明

### NeurIPS

- **8 頁**正文 + 無限附錄/參考文獻
- **Broader Impact** 章節有時需要
- **可重現性檢核表**必須
- OpenReview 投稿，公開審查

### ICML

- **8 頁**正文 + 無限附錄/參考文獻
- 強調**理論 + 實驗**
- 鼓勵可重現性聲明

### ICLR

- **8 頁**正文（camera-ready 可超過）
- OpenReview 有**公開審查和討論**
- 作者回應期是互動式的
- 強調**新穎性和洞察**

### CVPR/ICCV/ECCV

- **8 頁**正文含參考文獻
- 鼓勵**補充影片**
- 重度強調**視覺結果**
- 基準測試性能至關重要

---

## 常見錯誤

1. **弱基準線**：未與近期 SOTA 比較
2. **缺少消融實驗**：未展示元件貢獻
3. **過度聲稱**：「We solve X」而實際只是部分解決 X
4. **模糊的貢獻**：「We propose a novel method」
5. **可重現性差**：缺少超參數、種子
6. **錯誤模板**：使用去年的樣式檔案
7. **匿名違規**：在盲審中透露身份
8. **缺少限制**：未承認失敗模式

---

## 回覆審稿意見的技巧

ML 會議有作者回應期。技巧：
- **首先處理關鍵問題**：優先處理重要問題
- **執行要求的實驗**：在時間內可行時
- **簡潔**：審稿人閱讀很多回覆
- **保持專業**：即使面對不公平的審查
- **引用具體行數**：「As stated in L127...」

---

## 投稿前檢核表

### 內容
- [ ] 清晰的問題動機
- [ ] 明確的貢獻列表
- [ ] 完整的方法描述
- [ ] 全面的實驗
- [ ] 包含強基準線
- [ ] 有消融研究
- [ ] 承認限制

### 技術
- [ ] 正確的場所樣式檔案（當年）
- [ ] 匿名化（無作者姓名、無可識別 URL）
- [ ] 遵守頁數限制
- [ ] 參考文獻完整
- [ ] 補充材料組織良好

### 可重現性
- [ ] 列出超參數
- [ ] 指定隨機種子
- [ ] 說明計算需求
- [ ] 註明程式碼/資料可用性
- [ ] 完成可重現性檢核表

---

## 另請參閱

- `venue_writing_styles.md` - 主要風格概述
- `conferences_formatting.md` - 技術格式要求
- `reviewer_expectations.md` - ML 審稿人的期望
