# 文獻搜尋策略

## 尋找科學證據的有效技術

全面的文獻搜尋對於將假說建立在現有證據之上至關重要。本參考提供了 PubMed（生物醫學文獻）和一般科學搜尋的策略。

## 搜尋策略框架

### 三階段方法

1. **廣泛探索：** 了解全貌並識別關鍵概念
2. **聚焦搜尋：** 針對特定機制、理論或發現
3. **引用挖掘：** 追蹤關鍵論文的參考文獻和相關文章

### 搜尋之前

**釐清搜尋目標：**
- 現象的哪些方面需要證據？
- 什麼類型的研究最相關（評論、原始研究、方法）？
- 什麼時間框架相關（僅近期，還是歷史脈絡）？
- 需要什麼層級的證據（機制性、相關性、因果性）？

## PubMed 搜尋策略

### 何時使用 PubMed

使用 WebFetch 與 PubMed URL 用於：
- 生物醫學和生命科學研究
- 臨床研究和醫學文獻
- 分子、細胞和生理機制
- 疾病病因和病理學
- 藥物和治療研究

### 有效的 PubMed 搜尋技術

#### 1. 從評論文章開始

**原因：** 評論綜合文獻、識別關鍵概念，並提供全面的參考文獻列表。

**搜尋策略：**
- 在搜尋詞中添加「review」
- 使用 PubMed 篩選器：文章類型 → Review、Systematic Review、Meta-Analysis
- 尋找最近的評論（過去 2-5 年）

**範例搜尋：**
- `https://pubmed.ncbi.nlm.nih.gov/?term=wound+healing+diabetes+review`
- `https://pubmed.ncbi.nlm.nih.gov/?term=gut+microbiome+cognition+systematic+review`

#### 2. 使用 MeSH 術語（醫學主題詞表）

**原因：** MeSH 術語是標準化詞彙，可捕捉概念變體。

**策略：**
- PubMed 自動建議 MeSH 術語
- 幫助找到使用不同術語描述相同概念的論文
- 比僅關鍵字搜尋更全面

**範例：**
- 不要只用「heart attack」，使用 MeSH 術語「Myocardial Infarction」
- 捕捉使用「MI」、「heart attack」、「cardiac infarction」等的論文

#### 3. 布林運算子和進階語法

**AND：** 縮小搜尋（所有術語必須存在）
- `diabetes AND wound healing AND inflammation`

**OR：** 擴大搜尋（任何術語可以存在）
- `(Alzheimer OR dementia) AND gut microbiome`

**NOT：** 排除術語
- `cancer treatment NOT surgery`

**引號：** 精確詞組
- `"oxidative stress"`

**萬用字元：** 變體
- `gene*` 找到 gene、genes、genetic、genetics

#### 4. 按出版類型和日期篩選

**出版類型：**
- Clinical Trial
- Meta-Analysis
- Systematic Review
- Research Support, NIH
- Randomized Controlled Trial

**日期篩選：**
- 近期工作（過去 2-5 年）：最新發現
- 歷史工作：基礎研究
- 特定時間段：追蹤理解的發展

#### 5. 使用「相似文章」和「被引用」

**策略：**
- 找到一篇高度相關的論文
- 點擊「Similar articles」獲取相關工作
- 使用被引用工具尋找建立在其上的較新工作

### 按假說目標的 PubMed 搜尋範例

**機制理解：**
```
https://pubmed.ncbi.nlm.nih.gov/?term=(mechanism+OR+pathway)+AND+[phenomenon]+AND+(molecular+OR+cellular)
```

**因果關係：**
```
https://pubmed.ncbi.nlm.nih.gov/?term=[exposure]+AND+[outcome]+AND+(randomized+controlled+trial+OR+cohort+study)
```

**生物標記和關聯：**
```
https://pubmed.ncbi.nlm.nih.gov/?term=[biomarker]+AND+[disease]+AND+(association+OR+correlation+OR+prediction)
```

**治療有效性：**
```
https://pubmed.ncbi.nlm.nih.gov/?term=[intervention]+AND+[condition]+AND+(efficacy+OR+effectiveness+OR+clinical+trial)
```

## 一般科學網路搜尋策略

### 何時使用網路搜尋

使用 WebSearch 用於：
- 非生物醫學科學（物理、化學、材料、地球科學）
- 跨學科主題
- 最近的預印本和未發表的工作
- 灰色文獻（技術報告、會議論文集）
- 更廣泛的脈絡和跨領域類比

### 有效的網路搜尋技術

#### 1. 使用特定領域的搜尋術語

**包含特定領域的術語：**
- 化學：「mechanism」、「reaction pathway」、「synthesis」
- 物理：「model」、「theory」、「experimental validation」
- 材料科學：「properties」、「characterization」、「synthesis」
- 生態學：「population dynamics」、「community structure」

#### 2. 針對學術來源

**搜尋運算子：**
- `site:arxiv.org` - 預印本（物理、計算機科學、數學、定量生物學）
- `site:biorxiv.org` - 生物學預印本
- `site:edu` - 學術機構
- `filetype:pdf` - 學術論文（通常）

**範例搜尋：**
- `superconductivity high temperature mechanism site:arxiv.org`
- `CRISPR off-target effects site:biorxiv.org`

#### 3. 搜尋作者和實驗室

**當您找到相關論文時：**
- 搜尋作者的其他工作
- 找到他們的實驗室網站獲取未發表的工作
- 識別該領域的關鍵研究團隊

#### 4. 使用 Google Scholar 方法

**策略：**
- 使用「Cited by」尋找較新的相關工作
- 使用「Related articles」擴展搜尋
- 設置日期範圍以聚焦於近期工作
- 使用 author: 運算子尋找特定研究者

#### 5. 結合一般和特定術語

**結構：**
- 特定現象 + 一般概念
- 「tomato plant growth」+「bacterial promotion」
- 「cognitive decline」+「gut microbiome」

**布林邏輯：**
- 使用引號表示精確詞組：`"spike protein mutation"`
- 使用 OR 表示替代：`(transmissibility OR transmission rate)`
- 結合：`"spike protein" AND (transmissibility OR virulence) AND mutation`

## 跨資料庫搜尋策略

### 全面文獻搜尋工作流程

1. **從評論開始（PubMed 或網路搜尋）：**
   - 識別關鍵概念和術語
   - 注意有影響力的論文和研究者
   - 了解該領域的當前狀態

2. **聚焦原始研究（PubMed）：**
   - 搜尋特定機制
   - 尋找實驗證據
   - 識別方法學

3. **用網路搜尋擴展：**
   - 在其他領域找到相關工作
   - 定位最近的預印本
   - 識別類比系統

4. **引用挖掘：**
   - 追蹤關鍵論文的參考文獻
   - 使用「cited by」尋找近期工作
   - 追蹤有影響力的研究

5. **迭代精煉：**
   - 添加在論文中發現的新術語
   - 如果結果太多則縮小範圍
   - 如果相關結果太少則擴展

## 特定主題搜尋策略

### 機制和途徑

**目標：** 理解某事如何運作

**搜尋組件：**
- 現象 +「mechanism」
- 現象 +「pathway」
- 現象 + 懷疑的特定分子/途徑

**範例：**
- `diabetic wound healing mechanism inflammation`
- `autophagy pathway cancer`

### 關聯和相關性

**目標：** 找到哪些因素相關

**搜尋組件：**
- 變項 A + 變項 B +「association」
- 變項 A + 變項 B +「correlation」
- 變項 A +「predicts」+ 變項 B

**範例：**
- `vitamin D cardiovascular disease association`
- `gut microbiome diversity predicts cognitive function`

### 介入和治療

**目標：** 什麼有效的證據

**搜尋組件：**
- 介入 + 條件 +「efficacy」
- 介入 + 條件 +「randomized controlled trial」
- 介入 + 條件 +「treatment outcome」

**範例：**
- `probiotic intervention depression randomized controlled trial`
- `exercise intervention cognitive decline efficacy`

### 方法和技術

**目標：** 如何測試假說

**搜尋組件：**
- 方法名稱 + 應用領域
- 「How to measure」+ 現象
- 技術 + validation

**範例：**
- `CRISPR screen cancer drug resistance`
- `measure protein-protein interaction methods`

### 類比系統

**目標：** 從相關現象中找到洞察

**搜尋組件：**
- 機制 + 不同系統
- 類似現象 + 不同生物體/條件

**範例：**
- 如果研究植物-微生物共生：搜尋 `nitrogen fixation rhizobia legumes`
- 如果研究藥物抗性：搜尋 `antibiotic resistance evolution mechanisms`

## 評估論文影響力和品質

### 引用次數意義

引用次數表明在該領域的影響力和重要性。根據論文年齡和領域規範解釋引用：

| 論文年齡 | 引用次數 | 解釋 |
|---------|---------|------|
| 0-3 年 | 20+ | 值得注意 - 正在獲得關注 |
| 0-3 年 | 100+ | 高度有影響力 - 已有顯著影響 |
| 3-7 年 | 100+ | 重要 - 已建立的貢獻 |
| 3-7 年 | 500+ | 里程碑 - 對該領域的重大貢獻 |
| 7+ 年 | 500+ | 開創性 - 廣泛認可的重要工作 |
| 7+ 年 | 1000+ | 基礎性 - 定義該領域的論文 |

**特定領域考量：**
- 生物醫學/臨床：較高的引用規範（NEJM 論文通常 1000+）
- 計算機科學：會議引用比期刊更重要
- 數學/物理：較低的引用規範，較長的引用半衰期
- 社會科學：中等引用規範，較高的書籍引用率

### 期刊影響因子指南

**第一層 - 頂級出版場所（始終優先）：**
- **一般科學：** Nature（IF ~65）、Science（IF ~55）、Cell（IF ~65）、PNAS（IF ~12）
- **醫學：** NEJM（IF ~175）、Lancet（IF ~170）、JAMA（IF ~120）、BMJ（IF ~93）
- **領域旗艦：** Nature Medicine、Nature Biotechnology、Nature Methods、Nature Genetics

**第二層 - 高影響力專業期刊（強烈優先）：**
- 影響因子 >10
- 範例：JAMA Internal Medicine、Annals of Internal Medicine、Circulation、Blood
- 頂級 ML/AI 會議：NeurIPS、ICML、ICLR（相當於 IF 15-25）

**第三層 - 受尊重的專業期刊（相關時包含）：**
- 影響因子 5-10
- 已建立的學會期刊
- 索引良好的專業期刊

**第四層 - 其他同行評審期刊（謹慎使用）：**
- 影響因子 <5
- 只有在直接相關且沒有更好來源時才引用

### 作者學術履歷評估

優先選擇已建立研究者的論文：

**強作者指標：**
- **高 h-index：** 在已建立領域 >40，對於早期職涯新星 >20
- **多篇第一層出版物：** 在 Nature/Science/Cell 系列有發表記錄
- **機構隸屬：** 領先的研究型大學和研究所
- **認可：** 獎項、研究員身份、編輯職位
- **第一/通訊作者：** 在多篇高引用論文上

**如何檢查作者聲譽：**
1. Google Scholar 個人資料：檢查 h-index、i10-index、總引用
2. PubMed：搜尋作者姓名，審查出版場所
3. 機構頁面：檢查職位、獎項、資助
4. ORCID 個人資料：完整出版歷史

### 會議排名認知（計算機科學/AI）

對於 ML/AI 和計算機科學主題，會議排名很重要：

**A*（旗艦）- 相當於 Nature/Science：**
- NeurIPS（Neural Information Processing Systems）
- ICML（International Conference on Machine Learning）
- ICLR（International Conference on Learning Representations）
- CVPR（Computer Vision and Pattern Recognition）
- ACL（Association for Computational Linguistics）

**A（優秀）- 相當於第二層期刊：**
- AAAI、IJCAI（AI 一般）
- EMNLP、NAACL（NLP）
- ECCV、ICCV（計算機視覺）
- SIGKDD、WWW（資料探勘）

**B（良好）- 相當於第三層期刊：**
- COLING、CoNLL（NLP）
- WACV、BMVC（計算機視覺）
- 大多數 ACM/IEEE 專業會議

## 評估來源品質

### 原始研究品質指標

**強品質信號：**
- 發表在第一層或第二層出版場所
- 相對於論文年齡有高引用次數
- 由具有強學術履歷的已建立研究者撰寫
- 大樣本量（用於統計效力）
- 預先註冊的研究（減少偏誤）
- 適當的對照和方法
- 與其他發現一致
- 透明的資料和方法

**危險信號：**
- 沒有同行評審（謹慎使用）
- 利益衝突未披露
- 方法描述不清楚
- 沒有非凡證據的非凡主張
- 與大量證據矛盾但無解釋

### 評論品質指標

**系統性評論（最高品質）：**
- 預先定義的搜尋策略
- 明確的納入/排除標準
- 對納入研究的品質評估
- 量化綜合（統合分析）

**敘述性評論（品質可變）：**
- 專家對領域的綜合
- 可能有選擇偏誤
- 對於脈絡和框架有用
- 檢查作者專業知識和引用

## 文獻搜尋的時間管理

### 適當分配搜尋時間

**對於簡單假說（30-60 分鐘）：**
- 1-2 篇廣泛評論文章
- 3-5 篇有針對性的原始研究論文
- 快速網路搜尋最近發展

**對於複雜假說（1-3 小時）：**
- 針對不同方面的多篇評論
- 10-15 篇原始研究論文
- 跨資料庫的系統性搜尋
- 從關鍵論文的引用挖掘

**對於有爭議的主題（3+ 小時）：**
- 系統性評論方法
- 識別競爭觀點
- 追蹤歷史發展
- 交叉參考發現

### 報酬遞減

**您已經搜尋足夠的標誌：**
- 反覆找到相同的論文
- 新搜尋主要產生不相關的論文
- 有足夠的證據支持/脈絡化假說
- 多條獨立的證據線收斂

**何時搜尋更多：**
- 理解中仍有重大差距
- 矛盾的證據需要解決
- 假說似乎與文獻不一致
- 需要特定的方法學資訊

## 記錄搜尋結果

### 要捕捉的資訊

**對於每篇相關論文：**
- 完整引用（作者、年份、期刊、標題）
- 與假說相關的關鍵發現
- 研究設計和方法
- 作者注意的限制
- 它與假說的關係

### 組織發現

**分組依據：**
- 假說 A、B、C 的支持證據
- 方法學方法
- 需要解釋的矛盾發現
- 當前知識的差距

**綜合筆記：**
- 什麼是已經確立的？
- 什麼是有爭議或不確定的？
- 其他系統中存在什麼類比？
- 常用什麼方法？

### 假說報告的引用組織

**對於報告結構：** 為兩類讀者組織引用：

**主文（15-20 個關鍵引用）：**
- 最有影響力的論文（高引用、開創性研究）
- 最近的確定性證據（過去 2-3 年）
- 直接支持每個假說的關鍵論文（每個假說 3-5 篇）
- 綜合該領域的主要評論

**附錄 A：全面文獻回顧（40-60+ 引用）：**
- **歷史脈絡：** 建立該領域的基礎論文
- **當前理解：** 最近的評論和統合分析
- **假說特定證據：** 每個假說 8-15 篇論文涵蓋：
  - 直接支持證據
  - 相關系統中的類比機制
  - 方法學先例
  - 理論框架論文
- **矛盾發現：** 代表不同觀點的論文
- **知識差距：** 識別限制或未解答問題的論文

**目標引用密度：** 目標總共 50+ 參考文獻，以為所有主張提供全面支持並展示徹底的文獻基礎。

**附錄 A 的分組策略：**
1. 背景和脈絡論文
2. 當前理解和已建立機制
3. 支持每個假說的證據（單獨子部分）
4. 矛盾或替代發現
5. 方法學和技術論文

## 實際搜尋工作流程

### 逐步過程

1. **定義搜尋目標（5 分鐘）：**
   - 現象的哪些方面需要證據？
   - 什麼會支持或反駁假說？

2. **廣泛評論搜尋（15-20 分鐘）：**
   - 找到 1-3 篇評論文章
   - 瀏覽摘要以判斷相關性
   - 注意關鍵概念和術語

3. **針對性原始研究（30-45 分鐘）：**
   - 搜尋特定機制/證據
   - 閱讀摘要，瀏覽圖表和結論
   - 追蹤最有希望的參考文獻

4. **跨領域搜尋（15-30 分鐘）：**
   - 在其他系統中尋找類比
   - 找到最近的預印本
   - 識別新興趨勢

5. **引用挖掘（15-30 分鐘）：**
   - 追蹤關鍵論文的參考文獻
   - 使用「cited by」獲取近期工作
   - 識別開創性研究

6. **綜合發現（20-30 分鐘）：**
   - 總結每個假說的證據
   - 注意模式和矛盾
   - 識別知識差距

### 迭代和精煉

**當初始搜尋不足時：**
- 如果結果太少則擴大術語
- 如果結果太多則添加特定機制/途徑
- 嘗試替代術語
- 搜尋相關現象
- 查閱評論文章以獲取更好的搜尋術語

**需要更多搜尋的危險信號：**
- 只找到弱或間接證據
- 所有證據來自單一實驗室或來源
- 證據似乎與基本原則不一致
- 現象的主要方面缺乏任何相關文獻

## 常見搜尋陷阱

### 要避免的陷阱

1. **確認偏誤：** 只尋找支持首選假說的證據
   - **解決方案：** 積極搜尋矛盾證據

2. **近因偏誤：** 只考慮近期工作，錯過基礎研究
   - **解決方案：** 包含歷史搜尋，追蹤想法的發展

3. **太窄：** 由於限制性術語錯過相關工作
   - **解決方案：** 使用 OR 運算子，嘗試替代術語

4. **太廣：** 被不相關的結果淹沒
   - **解決方案：** 添加特定術語，使用篩選器，用 AND 結合概念

5. **單一資料庫：** 錯過其他領域的重要工作
   - **解決方案：** 同時搜尋 PubMed 和一般網路，嘗試特定領域資料庫

6. **過早停止：** 沒有足夠的證據來奠定假說基礎
   - **解決方案：** 設置最低目標（例如，每個假說方面 2 篇評論 + 5 篇原始論文）

7. **挑櫻桃：** 只引用支持性論文
   - **解決方案：** 代表證據的全部範圍，承認矛盾

## 特殊情況

### 新興主題（有限文獻）

**當存在很少已發表工作時：**
- 在相關系統中搜尋類比現象
- 尋找預印本（arXiv、bioRxiv）
- 找到會議摘要和海報
- 識別可能適用的理論框架
- 在假說生成中注意有限的證據

### 有爭議的主題（矛盾文獻）

**當證據矛盾時：**
- 系統性地記錄雙方
- 尋找解釋衝突的方法學差異
- 檢查時間趨勢（理解是否已改變？）
- 識別什麼會解決爭議
- 生成解釋差異的假說

### 跨學科主題

**當跨越多個領域時：**
- 搜尋每個領域的主要資料庫
- 為每個領域使用特定領域術語
- 尋找跨領域引用的橋接論文
- 考慮諮詢領域專家
- 仔細地在學科之間翻譯概念

## 與假說生成的整合

### 使用文獻來指導假說

**直接應用：**
- 應用到新情境的已建立機制
- 與現象相關的已知途徑
- 相關系統中的類似現象
- 用於測試的已驗證方法

**間接應用：**
- 來自不同系統的類比
- 要應用的理論框架
- 暗示新機制的差距
- 需要解決的矛盾

### 平衡文獻依賴

**太依賴文獻：**
- 假說僅僅重述已知機制
- 沒有新穎洞察或預測
- 「假說」實際上是已確立的事實

**太不依賴文獻：**
- 假說忽略相關證據
- 提出不合理的機制
- 重新發明已經測試過的想法
- 與既定原則不一致

**最佳平衡：**
- 建立在現有證據之上
- 以新穎的方式擴展理解
- 承認支持和挑戰的證據
- 生成超越當前知識的可測試預測
