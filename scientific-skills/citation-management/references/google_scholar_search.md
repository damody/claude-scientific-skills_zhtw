# Google Scholar 搜尋指南

搜尋 Google Scholar 學術論文的完整指南，包括進階搜尋運算子、篩選策略和後設資料擷取。

## 概述

Google Scholar 提供跨所有學科最全面的學術文獻涵蓋範圍：
- **涵蓋範圍**：超過 1 億篇學術文獻
- **範圍**：所有學術領域
- **內容類型**：期刊文章、書籍、論文、會議論文、預印本、專利、法院意見
- **引用追蹤**：「被引用」連結用於前向引用追蹤
- **可存取性**：免費使用，無需帳號

## 基本搜尋

### 簡單關鍵字搜尋

搜尋文件任何位置（標題、摘要、全文）包含特定詞彙的論文：

```
CRISPR gene editing
machine learning protein folding
climate change impact agriculture
quantum computing algorithms
```

**提示**：
- 使用具體的技術詞彙
- 包含關鍵縮寫和簡稱
- 從廣泛開始，然後細化
- 檢查技術詞彙的拼寫

### 精確詞組搜尋

使用引號搜尋精確詞組：

```
"deep learning"
"CRISPR-Cas9"
"systematic review"
"randomized controlled trial"
```

**使用時機**：
- 必須一起出現的技術詞彙
- 專有名詞
- 特定方法論
- 精確標題

## 進階搜尋運算子

### 作者搜尋

尋找特定作者的論文：

```
author:LeCun
author:"Geoffrey Hinton"
author:Church synthetic biology
```

**變化形式**：
- 單一姓氏：`author:Smith`
- 全名加引號：`author:"Jane Smith"`
- 作者 + 主題：`author:Doudna CRISPR`

**提示**：
- 作者可能以不同姓名變體發表
- 嘗試有無中間名縮寫
- 考慮姓名變更（婚姻等）
- 全名使用引號

### 標題搜尋

僅在文章標題中搜尋：

```
intitle:transformer
intitle:"attention mechanism"
intitle:review climate change
```

**使用情境**：
- 尋找專門關於某主題的論文
- 比全文搜尋更精確
- 減少不相關的結果
- 適合尋找綜述或方法論文

### 來源（期刊）搜尋

在特定期刊或會議中搜尋：

```
source:Nature
source:"Nature Communications"
source:NeurIPS
source:"Journal of Machine Learning Research"
```

**應用**：
- 追蹤頂級場所的發表
- 尋找專業期刊中的論文
- 識別特定會議的研究
- 驗證發表場所

### 排除運算子

從結果中排除詞彙：

```
machine learning -survey
CRISPR -patent
climate change -news
deep learning -tutorial -review
```

**常見排除項**：
- `-survey`：排除調查論文
- `-review`：排除綜述文章
- `-patent`：排除專利
- `-book`：排除書籍
- `-news`：排除新聞報導
- `-tutorial`：排除教程

### OR 運算子

搜尋包含多個詞彙中任一個的論文：

```
"machine learning" OR "deep learning"
CRISPR OR "gene editing"
"climate change" OR "global warming"
```

**最佳實務**：
- OR 必須大寫
- 結合同義詞
- 包含縮寫和完整形式
- 與精確詞組一起使用

### 萬用字元搜尋

使用星號（*）作為未知單詞的萬用字元：

```
"machine * learning"
"CRISPR * editing"
"* neural network"
```

**注意**：Google Scholar 的萬用字元支援比其他資料庫有限。

## 進階篩選

### 年份範圍

按出版年份篩選：

**使用介面**：
- 點擊左側邊欄的「Since [year]」
- 選擇自訂範圍

**使用搜尋運算子**：
```
# 不直接在搜尋查詢中
# 使用介面或 URL 參數
```

**在腳本中**：
```bash
python scripts/search_google_scholar.py "quantum computing" \
  --year-start 2020 \
  --year-end 2024
```

### 排序選項

**按相關性**（預設）：
- Google 的演算法決定相關性
- 考慮引用次數、作者聲譽、發表場所
- 對大多數搜尋通常適用

**按日期**：
- 最新論文優先
- 適合快速發展的領域
- 可能遺漏高引用的舊論文
- 在介面中點擊「Sort by date」

**按引用次數**（透過腳本）：
```bash
python scripts/search_google_scholar.py "transformers" \
  --sort-by citations \
  --limit 50
```

### 語言篩選

**在介面中**：
- 設定 → 語言
- 選擇偏好語言

**預設**：英文和有英文摘要的論文

## 搜尋策略

### 尋找開創性論文

識別某領域高影響力的論文：

1. **按主題搜尋**使用廣泛詞彙
2. **按引用排序**（最多引用優先）
3. **尋找綜述文章**以獲得全面概覽
4. **檢查發表日期**區分奠基性研究與近期研究

**範例**：
```
"generative adversarial networks"
# 按引用排序
# 頂部結果：原始 GAN 論文（Goodfellow et al., 2014）、重要變體
```

### 尋找近期研究

追蹤最新研究：

1. **按主題搜尋**
2. **篩選近年**（最近 1-2 年）
3. **按日期排序**最新優先
4. **設定提醒**持續追蹤

**範例**：
```bash
python scripts/search_google_scholar.py "AlphaFold protein structure" \
  --year-start 2023 \
  --year-end 2024 \
  --limit 50
```

### 尋找綜述文章

獲得某領域的全面概覽：

```
intitle:review "machine learning"
"systematic review" CRISPR
intitle:survey "natural language processing"
```

**指標**：
- 標題中有「review」、「survey」、「perspective」
- 通常高引用
- 發表在綜述期刊（Nature Reviews、Trends 等）
- 參考文獻列表全面

### 引用鏈搜尋

**前向引用**（引用某關鍵論文的論文）：
1. 找到開創性論文
2. 點擊「Cited by X」
3. 查看所有引用該論文的論文
4. 識別領域的發展方向

**後向引用**（某關鍵論文的參考文獻）：
1. 找到近期綜述或重要論文
2. 檢查其參考文獻列表
3. 識別奠基性研究
4. 追蹤思想的發展

**範例工作流程**：
```
# 找到原始 transformer 論文
"Attention is all you need" author:Vaswani

# 檢查「Cited by 120,000+」
# 查看演進：BERT、GPT、T5 等

# 檢查原始論文的參考文獻
# 找到 RNN、LSTM、注意力機制的起源
```

### 全面文獻搜尋

為徹底涵蓋（例如系統性綜述）：

1. **建立同義詞列表**：
   - 主要詞彙 + 替代詞
   - 縮寫 + 完整形式
   - 美式 vs 英式拼寫

2. **使用 OR 運算子**：
   ```
   ("machine learning" OR "deep learning" OR "neural networks")
   ```

3. **結合多個概念**：
   ```
   ("machine learning" OR "deep learning") ("drug discovery" OR "drug development")
   ```

4. **初始搜尋不設日期篩選**：
   - 獲得全景
   - 如結果太多再篩選

5. **匯出結果**進行系統性分析：
   ```bash
   python scripts/search_google_scholar.py \
     '"machine learning" OR "deep learning" drug discovery' \
     --limit 500 \
     --output comprehensive_search.json
   ```

## 擷取引用文獻資訊

### 從 Google Scholar 結果頁面

每個結果顯示：
- **標題**：論文標題（如可用則連結到全文）
- **作者**：作者列表（通常會截斷）
- **來源**：期刊/會議、年份、出版商
- **被引用**：引用次數 + 連結到引用論文
- **相關文章**：連結到相似論文
- **所有版本**：同一論文的不同版本

### 匯出選項

**手動匯出**：
1. 點擊論文下方的「Cite」
2. 選擇 BibTeX 格式
3. 複製引用

**限制**：
- 一次一篇論文
- 手動流程
- 多篇論文時耗時

**自動匯出**（使用腳本）：
```bash
# 搜尋並匯出為 BibTeX
python scripts/search_google_scholar.py "quantum computing" \
  --limit 50 \
  --format bibtex \
  --output quantum_papers.bib
```

### 可用的後設資料

從 Google Scholar 通常可以擷取：
- 標題
- 作者（可能不完整）
- 年份
- 來源（期刊/會議）
- 引用次數
- 全文連結（如可用）
- PDF 連結（如可用）

**注意**：後設資料品質不一：
- 某些欄位可能遺失
- 作者姓名可能不完整
- 需要透過 DOI 查詢驗證準確性

## 速率限制和存取

### 速率限制

Google Scholar 有速率限制以防止自動抓取：

**速率限制的症狀**：
- CAPTCHA 驗證
- 臨時 IP 封鎖
- 429「Too Many Requests」錯誤

**最佳實務**：
1. **在請求之間添加延遲**：最少 2-5 秒
2. **限制查詢量**：不要快速搜尋數百個查詢
3. **使用 scholarly 函式庫**：自動處理速率限制
4. **輪換 User-Agent**：模擬不同瀏覽器
5. **考慮使用代理**：用於大規模搜尋（需合乎道德使用）

**在我們的腳本中**：
```python
# 內建自動速率限制
time.sleep(random.uniform(3, 7))  # 隨機延遲 3-7 秒
```

### 道德考量

**可以做**：
- 遵守速率限制
- 使用合理延遲
- 快取結果（避免重複查詢）
- 可用時使用官方 API
- 正確標註資料來源

**不可以做**：
- 激進抓取
- 使用多個 IP 繞過限制
- 違反服務條款
- 不必要地給伺服器造成負擔
- 未經許可商業使用資料

### 機構存取

**機構存取的好處**：
- 透過圖書館訂閱存取全文 PDF
- 更好的下載能力
- 與圖書館系統整合
- 連結解析器到全文

**設定**：
- Google Scholar → Settings → Library links
- 添加您的機構
- 連結將出現在搜尋結果中

## 提示和最佳實務

### 搜尋優化

1. **從簡單開始，然後細化**：
   ```
   # 初始時過於具體
   intitle:"deep learning" intitle:review source:Nature 2023..2024

   # 更好的方法
   deep learning review
   # 檢閱結果
   # 根據需要添加 intitle:、source:、年份篩選
   ```

2. **使用多種搜尋策略**：
   - 關鍵字搜尋
   - 作者搜尋已知專家
   - 從關鍵論文進行引用鏈
   - 在頂級期刊中搜尋來源

3. **檢查拼寫和變體**：
   - Color vs colour
   - Optimization vs optimisation
   - Tumor vs tumour
   - 如結果少則嘗試常見拼寫錯誤

4. **策略性結合運算子**：
   ```
   # 良好組合
   author:Church intitle:"synthetic biology" 2015..2024

   # 尋找特定作者在近年某主題上的綜述
   ```

### 結果評估

1. **檢查引用次數**：
   - 高引用表示影響力
   - 近期論文可能引用低但仍重要
   - 引用次數因領域而異

2. **驗證發表場所**：
   - 同行評審期刊 vs 預印本
   - 會議論文集
   - 書籍章節
   - 技術報告

3. **檢查全文存取**：
   - 右側的 [PDF] 連結
   - 「All X versions」可能有開放存取版本
   - 檢查機構存取
   - 嘗試作者網站或 ResearchGate

4. **尋找綜述文章**：
   - 全面概覽
   - 新主題的良好起點
   - 廣泛的參考文獻列表

### 管理結果

1. **使用引用管理器整合**：
   - 匯出為 BibTeX
   - 匯入到 Zotero、Mendeley、EndNote
   - 維護有組織的文獻庫

2. **為持續研究設定提醒**：
   - Google Scholar → Alerts
   - 收到符合查詢的新論文郵件
   - 追蹤特定作者或主題

3. **建立收藏**：
   - 將論文儲存到 Google Scholar Library
   - 按專案或主題組織
   - 添加標籤和筆記

4. **系統性匯出**：
   ```bash
   # 儲存搜尋結果供後續分析
   python scripts/search_google_scholar.py "your topic" \
     --output topic_papers.json

   # 可以後續處理而無需重新搜尋
   python scripts/extract_metadata.py \
     --input topic_papers.json \
     --output topic_refs.bib
   ```

## 進階技巧

### 布林邏輯組合

結合多個運算子進行精確搜尋：

```
# 已知作者在特定主題上的高引用綜述
intitle:review "machine learning" ("drug discovery" OR "drug development")
author:Horvath OR author:Bengio 2020..2024

# 排除綜述的方法論文
intitle:method "protein folding" -review -survey

# 僅頂級期刊中的論文
("Nature" OR "Science" OR "Cell") CRISPR 2022..2024
```

### 尋找開放存取論文

```
# 使用通用詞彙搜尋
machine learning

# 透過「All versions」篩選，通常包括預印本
# 尋找綠色 [PDF] 連結（通常是開放存取）
# 檢查 arXiv、bioRxiv 版本
```

**在腳本中**：
```bash
python scripts/search_google_scholar.py "topic" \
  --open-access-only \
  --output open_access_papers.json
```

### 追蹤研究影響

**針對特定論文**：
1. 找到該論文
2. 點擊「Cited by X」
3. 分析引用論文：
   - 如何被使用？
   - 哪些領域引用它？
   - 近期 vs 較舊的引用？

**針對作者**：
1. 搜尋 `author:LastName`
2. 檢查 h-index 和 i10-index
3. 查看引用歷史圖表
4. 識別最具影響力的論文

**針對主題**：
1. 搜尋主題
2. 按引用排序
3. 識別開創性論文（高引用、較舊）
4. 檢查近期高引用論文（新興重要研究）

### 尋找預印本和早期研究

```
# arXiv 論文
source:arxiv "deep learning"

# bioRxiv 論文
source:biorxiv CRISPR

# 所有預印本伺服器
("arxiv" OR "biorxiv" OR "medrxiv") your topic
```

**注意**：預印本未經同行評審。始終檢查是否有已發表版本。

## 常見問題和解決方案

### 結果太多

**問題**：搜尋返回 100,000+ 結果，令人困擾。

**解決方案**：
1. 添加更具體的詞彙
2. 使用 `intitle:` 僅搜尋標題
3. 篩選近年
4. 添加排除項（例如 `-review`）
5. 在特定期刊中搜尋

### 結果太少

**問題**：搜尋返回 0-10 個結果，疑似太少。

**解決方案**：
1. 移除限制性運算子
2. 嘗試同義詞和相關詞彙
3. 檢查拼寫
4. 擴大年份範圍
5. 使用 OR 添加替代詞彙

### 不相關的結果

**問題**：結果不符合意圖。

**解決方案**：
1. 使用引號進行精確詞組
2. 添加更具體的上下文詞彙
3. 使用 `intitle:` 僅標題搜尋
4. 排除常見不相關詞彙
5. 結合多個具體詞彙

### CAPTCHA 或速率限制

**問題**：Google Scholar 顯示 CAPTCHA 或封鎖存取。

**解決方案**：
1. 等待數分鐘後再繼續
2. 降低查詢頻率
3. 在腳本中使用更長延遲（5-10 秒）
4. 切換到不同 IP/網路
5. 考慮使用機構存取

### 遺失的後設資料

**問題**：結果中遺失作者姓名、年份或場所。

**解決方案**：
1. 點擊查看完整詳細資訊
2. 檢查「All versions」以獲得更好的後設資料
3. 如可用則透過 DOI 查詢
4. 改從 CrossRef/PubMed 擷取後設資料
5. 從論文 PDF 手動驗證

### 重複結果

**問題**：同一論文出現多次。

**解決方案**：
1. 點擊「All X versions」查看整合視圖
2. 選擇後設資料最佳的版本
3. 在後處理中使用去重：
   ```bash
   python scripts/format_bibtex.py results.bib \
     --deduplicate \
     --output clean_results.bib
   ```

## 與腳本整合

### search_google_scholar.py 用法

**基本搜尋**：
```bash
python scripts/search_google_scholar.py "machine learning drug discovery"
```

**帶年份篩選**：
```bash
python scripts/search_google_scholar.py "CRISPR" \
  --year-start 2020 \
  --year-end 2024 \
  --limit 100
```

**按引用排序**：
```bash
python scripts/search_google_scholar.py "transformers" \
  --sort-by citations \
  --limit 50
```

**匯出為 BibTeX**：
```bash
python scripts/search_google_scholar.py "quantum computing" \
  --format bibtex \
  --output quantum.bib
```

**匯出為 JSON 供後續處理**：
```bash
python scripts/search_google_scholar.py "topic" \
  --format json \
  --output results.json

# 後續：擷取完整後設資料
python scripts/extract_metadata.py \
  --input results.json \
  --output references.bib
```

### 批次搜尋

對多個主題：

```bash
# 建立包含搜尋查詢的檔案（queries.txt）
# 每行一個查詢

# 搜尋每個查詢
while read query; do
  python scripts/search_google_scholar.py "$query" \
    --limit 50 \
    --output "${query// /_}.json"
  sleep 10  # 查詢之間的延遲
done < queries.txt
```

## 總結

Google Scholar 是最全面的學術搜尋引擎，提供：

✓ **廣泛涵蓋**：所有學科，1 億+ 文獻
✓ **免費存取**：無需帳號或訂閱
✓ **引用追蹤**：「被引用」用於影響力分析
✓ **多種格式**：文章、書籍、論文、專利
✓ **全文搜尋**：不只是摘要

關鍵策略：
- 使用進階運算子提高精確度
- 結合作者、標題、來源搜尋
- 追蹤引用以了解影響力
- 系統性匯出到引用管理器
- 遵守速率限制和存取政策
- 使用 CrossRef/PubMed 驗證後設資料

對於生物醫學研究，可搭配 PubMed 使用 MeSH 詞彙和策展後設資料。
