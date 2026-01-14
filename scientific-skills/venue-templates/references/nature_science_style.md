# Nature 和 Science 寫作風格指南

Nature、Science 及相關高影響力多學科期刊（Nature Communications、Science Advances、PNAS）的完整寫作指南。

**最後更新**：2024

---

## 概述

Nature 和 Science 是世界頂尖的多學科科學期刊。在此發表的論文必須吸引所有學科的科學家，而不僅是專家。這從根本上塑造了寫作風格。

### 核心理念

> 「如果一位結構生物學家無法理解您的粒子物理學論文為何重要，它就不會在 Nature 發表。」

**主要目標**：向受過教育但非專家的讀者傳達開創性科學。

---

## 讀者與語氣

### 目標讀者

- **任何**領域的博士級科學家
- 熟悉科學方法論
- **非**您特定子領域的專家
- 廣泛閱讀以保持對科學的了解

### 語氣特徵

| 特徵 | 說明 |
|---------------|-------------|
| **易理解** | 避免術語；解釋技術概念 |
| **引人入勝** | 吸引讀者；講述故事 |
| **重要性** | 強調為何這在廣泛層面上重要 |
| **自信** | 清楚陳述發現（適當保守措辭） |
| **主動** | 使用主動語態；可接受第一人稱 |

### 語態

- **鼓勵第一人稱複數（「we」）**：「We discovered that...」而非「It was discovered that...」
- **偏好主動語態**：「We measured...」而非「Measurements were taken...」
- **直接陳述**：「Protein X controls Y」而非「Protein X appears to potentially control Y」

---

## 摘要

### 風格要求

- **流暢段落**（非帶標籤章節的結構化格式）
- **150-200 字**用於 Nature；Nature Communications 最多 250 字
- 摘要中**不引用文獻**
- **不使用縮寫**（或必要時在首次使用時定義）
- **自我完整**：無需閱讀論文即可理解

### 摘要結構（隱含式）

以流暢散文撰寫，涵蓋：

1. **背景**（1-2 句）：為何這個領域重要
2. **缺口/問題**（1 句）：什麼是未知或有問題的
3. **方法**（1 句）：您做了什麼（簡短）
4. **關鍵發現**（2-3 句）：主要結果及關鍵數字
5. **重要性**（1-2 句）：為何這很重要，有何意涵

### 範例摘要（Nature 格式）

```
The origins of multicellular life remain one of biology's greatest mysteries.
How individual cells first cooperated to form complex organisms has been
difficult to study because the transition occurred over 600 million years ago.
Here we show that the unicellular alga Chlamydomonas reinhardtii can evolve
simple multicellular structures within 750 generations when exposed to
predation pressure. Using experimental evolution with the predator Paramecium,
we observed the emergence of stable multicellular clusters in 5 of 10
replicate populations. Genomic analysis revealed that mutations in just two
genes—encoding cell adhesion proteins—were sufficient to trigger this
transition. These results demonstrate that the evolution of multicellularity
may require fewer genetic changes than previously thought, providing insight
into one of life's major transitions.
```

### 不該寫的內容

❌ **過於技術性**：
> 「Using CRISPR-Cas9-mediated knockout of the CAD1 gene (encoding cadherin-1) in C. reinhardtii strain CC-125, we demonstrated that loss of CAD1 function combined with overexpression of FLA10 under control of the HSP70A/RBCS2 tandem promoter...」

❌ **過於模糊**：
> 「We studied how cells can form groups. Our results are interesting and may have implications for understanding evolution.」

---

## 引言

### 長度與結構

- **3-5 段**（約 500-800 字）
- **漏斗結構**：廣泛 → 具體 → 您的貢獻

### 逐段指南

**第 1 段：大局觀**
- 以關於該領域的廣泛、引人入勝的陳述開場
- 建立為何這個領域對科學/社會重要
- 任何科學家都可理解

```
範例：
「The ability to predict protein structure from sequence alone has been a grand
challenge of biology for over 50 years. Accurate predictions would transform
drug discovery, enable understanding of disease mechanisms, and illuminate the
fundamental rules governing molecular self-assembly.」
```

**第 2-3 段：我們所知**
- 回顧關鍵先前工作（選擇性，非詳盡）
- 朝向您要解決的缺口建構
- 引用聚焦於重要論文

```
範例：
「Significant progress has been made through template-based methods that
leverage known structures of homologous proteins. However, for the estimated
30% of proteins without detectable homologs, prediction accuracy has remained
limited. Deep learning approaches have shown promise, achieving improved
accuracy on benchmark datasets, yet still fall short of experimental accuracy
for many protein families.」
```

**第 4 段：缺口**
- 清楚說明什麼仍未知或未解決
- 將此框架為重要問題

```
範例：
「Despite these advances, the fundamental question remains: can we predict
protein structure with experimental-level accuracy for proteins across all
of sequence space? This capability would democratize structural biology and
enable rapid characterization of newly discovered proteins.」
```

**最後一段：本論文**
- 說明您做了什麼並預覽關鍵發現
- 表明您貢獻的重要性

```
範例：
「Here we present AlphaFold2, a neural network architecture that predicts
protein structure with atomic-level accuracy. In the CASP14 blind assessment,
AlphaFold2 achieved a median GDT score of 92.4, matching experimental
accuracy for most targets. We show that this system can be applied to predict
structures across entire proteomes, opening new avenues for understanding
protein function at scale.」
```

### 引言禁忌

- ❌ 不要以「Since ancient times...」或過於誇大的聲稱開場
- ❌ 不要提供詳盡的文獻回顧（留給專業期刊）
- ❌ 不要在引言中包含方法或結果
- ❌ 不要使用未解釋的縮寫或術語

---

## 結果

### 組織理念

**故事驅動，而非實驗驅動**

按**發現**組織，而非按實驗的時間順序：

❌ **實驗驅動**（避免）：
> 「We first performed experiment A. Next, we did experiment B. Then we conducted experiment C.」

✅ **發現驅動**（偏好）：
> 「We discovered that X. To understand the mechanism, we found that Y. This led us to test whether Z, confirming our hypothesis.」

### 結果寫作風格

- **過去式**用於描述所做/發現的事
- **現在式**用於引用圖片（「Figure 2 shows...」）
- **客觀但有詮釋性**：以最少詮釋陳述發現，但為非專家提供足夠背景
- **量化**：包含關鍵數字、統計、效果量

### 範例結果段落

```
To test whether protein X is required for cell division, we generated
knockout cell lines using CRISPR-Cas9 (Fig. 1a). Cells lacking protein X
showed a 73% reduction in division rate compared to controls (P < 0.001,
n = 6 biological replicates; Fig. 1b). Live-cell imaging revealed that
knockout cells arrested in metaphase, with 84% showing abnormal spindle
morphology (Fig. 1c,d). These results demonstrate that protein X is
essential for proper spindle assembly and cell division.
```

### 小標題

使用傳達發現的描述性小標題：

❌ **模糊**：「Protein expression analysis」
✅ **資訊豐富**：「Protein X is upregulated in response to stress」

---

## 討論

### 結構（4-6 段）

**第 1 段：關鍵發現摘要**
- 重述主要發現（不要逐字重複結果）
- 說明假說是否獲得支持

**第 2-3 段：詮釋與背景**
- 發現意味著什麼？
- 它們與先前工作如何相關？
- 什麼機制可能解釋結果？

**第 4 段：更廣泛的意涵**
- 為何這在您的特定系統之外也重要？
- 與其他領域的連結
- 潛在應用

**第 5 段：限制**
- 誠實承認限制
- 具體，而非泛泛而論

**最後一段：結論與未來**
- 大局觀的要點
- 簡短提及未來方向

### 討論寫作技巧

- **以意涵為先**，而非注意事項
- **建設性地與文獻比較**：「Our findings extend the work of Smith et al. by demonstrating...」
- **承認替代解釋**：「An alternative explanation is that...」
- **對限制誠實**：具體 > 泛泛

### 範例限制陳述

❌ **泛泛**：「Our study has limitations that should be addressed in future work.」

✅ **具體**：「Our analysis was limited to cultured cells, which may not fully recapitulate the tissue microenvironment. Additionally, the 48-hour observation window may miss slower-developing phenotypes.」

---

## 方法

### Nature 方法的位置

- **簡短方法**在正文中（通常在結尾）
- **延伸方法**在補充資料中
- 必須足夠詳細以供重現

### 寫作風格

- **過去式，被動語態可接受**：「Cells were cultured...」或「We cultured cells...」
- **精確且可重現**：包含濃度、時間、溫度
- **引用已建立的方法**：「Following the method of Smith et al.³...」

---

## 圖片

### 圖片理念

Nature 重視**概念圖**與資料並重：

1. **圖 1**：通常是展示概念的示意圖/模型
2. **資料圖**：清晰、不雜亂
3. **最後一張圖**：通常是摘要模型

### 圖片設計原則

- **單欄（89 mm）或雙欄（183 mm）**寬度
- **高解析度**：照片 300+ dpi，線條圖 1000+ dpi
- **色盲友善**：避免僅用紅綠區分
- **最少圖表垃圾**：無 3D 效果、不必要的格線
- **完整圖說**：無需閱讀正文即可自我解釋

### 圖說格式

```
Figure 1 | Protein X controls cell division through spindle assembly.
a, Schematic of the experimental approach. b, Quantification of cell
division rate in control (grey) and knockout (blue) cells. Data are
mean ± s.e.m., n = 6 biological replicates. ***P < 0.001, two-tailed
t-test. c,d, Representative images of spindle morphology in control (c)
and knockout (d) cells. Scale bars, 10 μm.
```

---

## 參考文獻

### 引用格式

- **上標數字**：¹, ², ¹⁻³, ¹'⁵'⁷
- **Nature 格式**用於參考書目

### 參考文獻格式

```
1. Watson, J. D. & Crick, F. H. C. Molecular structure of nucleic acids.
   Nature 171, 737–738 (1953).

2. Smith, A. B., Jones, C. D. & Williams, E. F. Discovery of protein X.
   Science 380, 123–130 (2023).
```

### 引用最佳實踐

- **近期文獻**：包含過去 2-3 年的論文
- **開創性論文**：引用基礎性工作
- **多元來源**：不要過度引用自己的工作
- **原始來源**：引用原始發現，而非綜述（可能時）

---

## 語言與風格技巧

### 用詞選擇

| 避免 | 偏好 |
|-------|--------|
| utilize | use |
| methodology | method |
| in order to | to |
| a large number of | many |
| at this point in time | now |
| has the ability to | can |
| it is interesting to note that | [直接刪除] |

### 句子結構

- **變化句子長度**：混合短句和長句
- **重要性領先**：將關鍵資訊放在開頭
- **每句一個想法**：複雜想法需要多個句子

### 段落結構

- **主題句在前**：陳述主要觀點
- **支持證據**：資料和引用
- **過渡**：連接到下一段

---

## 比較：Nature vs. Science

| 特點 | Nature | Science |
|---------|--------|---------|
| 摘要長度 | 150-200 字 | ≤125 字 |
| 引用格式 | 上標數字 | 括號數字 (1, 2) |
| 參考文獻中的文章標題 | 有 | 無（在主要參考文獻中） |
| 方法位置 | 論文末端或補充資料 | 補充資料 |
| 重要性聲明 | 無 | 無 |
| 開放取用選項 | 有 | 有 |

---

## 常見被拒原因

1. **廣泛興趣不足**：對 Nature/Science 來說過於專門
2. **增量式進展**：不夠具變革性
3. **過度宣傳**：聲稱無資料支持
4. **可及性差**：對一般讀者過於技術性
5. **重要性聲明薄弱**：「So what?」不清楚
6. **新穎性不足**：類似發現已在他處發表
7. **方法論疑慮**：結果不夠有說服力

---

## 投稿前檢核表

### 內容
- [ ] 第一段清楚說明對廣泛讀者的重要性
- [ ] 非專家可以理解摘要
- [ ] 故事驅動的結果（非逐實驗陳述）
- [ ] 討論中強調意涵
- [ ] 具體承認限制

### 風格
- [ ] 主動語態為主
- [ ] 術語最少化或已解釋
- [ ] 句子長度有變化
- [ ] 段落有清楚的主題句

### 技術
- [ ] 圖片是高解析度
- [ ] 引用格式正確
- [ ] 字數在限制內
- [ ] 包含行號
- [ ] 雙倍行距

---

## 另請參閱

- `venue_writing_styles.md` - 主要風格概述
- `journals_formatting.md` - 技術格式要求
- `reviewer_expectations.md` - Nature/Science 審稿人的期望
