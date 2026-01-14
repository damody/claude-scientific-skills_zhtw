# PubMed 搜尋指南

搜尋 PubMed 生物醫學和生命科學文獻的完整指南，包括 MeSH 詞彙、欄位標籤、進階搜尋策略和 E-utilities API 使用。

## 概述

PubMed 是生物醫學文獻的首要資料庫：
- **涵蓋範圍**：超過 3,500 萬筆引用
- **範圍**：生物醫學和生命科學
- **來源**：MEDLINE、生命科學期刊、線上書籍
- **權威性**：由美國國家醫學圖書館（NLM）/ NCBI 維護
- **存取**：免費，無需帳號
- **更新**：每日添加新引用
- **策展**：高品質後設資料，MeSH 索引

## 基本搜尋

### 簡單關鍵字搜尋

PubMed 自動將詞彙對應到 MeSH 並搜尋多個欄位：

```
diabetes
CRISPR gene editing
Alzheimer's disease treatment
cancer immunotherapy
```

**自動功能**：
- 自動 MeSH 對應
- 複數/單數變體
- 縮寫展開
- 拼寫檢查

### 精確詞組搜尋

使用引號進行精確詞組搜尋：

```
"CRISPR-Cas9"
"systematic review"
"randomized controlled trial"
"machine learning"
```

## MeSH（醫學主題詞表）

### 什麼是 MeSH？

MeSH 是用於索引生物醫學文獻的控制詞彙詞表：
- **層級結構**：以樹狀結構組織
- **一致的索引**：相同概念始終以相同方式標記
- **全面**：涵蓋疾病、藥物、解剖學、技術等
- **專業策展**：NLM 索引員指派 MeSH 詞彙

### 尋找 MeSH 詞彙

**MeSH 瀏覽器**：https://meshb.nlm.nih.gov/search

**範例**：
```
搜尋：「heart attack」
MeSH 詞彙：「Myocardial Infarction」
```

**在 PubMed 中**：
1. 使用關鍵字搜尋
2. 檢查左側邊欄的「MeSH Terms」
3. 選擇相關的 MeSH 詞彙
4. 添加到搜尋

### 在搜尋中使用 MeSH

**基本 MeSH 搜尋**：
```
"Diabetes Mellitus"[MeSH]
"CRISPR-Cas Systems"[MeSH]
"Alzheimer Disease"[MeSH]
"Neoplasms"[MeSH]
```

**MeSH 帶副標題**：
```
"Diabetes Mellitus/drug therapy"[MeSH]
"Neoplasms/genetics"[MeSH]
"Heart Failure/prevention and control"[MeSH]
```

**常見副標題**：
- `/drug therapy`：藥物治療
- `/diagnosis`：診斷方面
- `/genetics`：遺傳方面
- `/epidemiology`：發生率和分布
- `/prevention and control`：預防方法
- `/etiology`：病因
- `/surgery`：手術治療
- `/metabolism`：代謝方面

### MeSH 展開

預設情況下，MeSH 搜尋包含較窄的詞彙（展開）：

```
"Neoplasms"[MeSH]
# 包含：Breast Neoplasms、Lung Neoplasms 等
```

**停用展開**（僅精確詞彙）：
```
"Neoplasms"[MeSH:NoExp]
```

### MeSH 主要主題

僅搜尋 MeSH 詞彙為主要焦點的文章：

```
"Diabetes Mellitus"[MeSH Major Topic]
# 僅糖尿病為主題的論文
```

## 欄位標籤

欄位標籤指定要搜尋記錄的哪個部分。

### 常見欄位標籤

**標題和摘要**：
```
cancer[Title]                    # 僅在標題中
treatment[Title/Abstract]        # 在標題或摘要中
"machine learning"[Title/Abstract]
```

**作者**：
```
"Smith J"[Author]
"Doudna JA"[Author]
"Collins FS"[Author]
```

**作者 - 全名**：
```
"Smith, John"[Full Author Name]
```

**期刊**：
```
"Nature"[Journal]
"Science"[Journal]
"New England Journal of Medicine"[Journal]
"Nat Commun"[Journal]           # 縮寫形式
```

**出版日期**：
```
2023[Publication Date]
2020:2024[Publication Date]      # 日期範圍
2023/01/01:2023/12/31[Publication Date]
```

**建立日期**：
```
2023[Date - Create]              # 添加到 PubMed 的時間
```

**出版類型**：
```
"Review"[Publication Type]
"Clinical Trial"[Publication Type]
"Meta-Analysis"[Publication Type]
"Randomized Controlled Trial"[Publication Type]
```

**語言**：
```
English[Language]
French[Language]
```

**DOI**：
```
10.1038/nature12345[DOI]
```

**PMID（PubMed ID）**：
```
12345678[PMID]
```

**文章 ID**：
```
PMC1234567[PMC]                  # PubMed Central ID
```

### 較少用但有用的標籤

```
humans[MeSH Terms]               # 僅人類研究
animals[MeSH Terms]              # 僅動物研究
"United States"[Place of Publication]
nih[Grant Number]                # NIH 資助的研究
"Female"[Sex]                    # 女性受試者
"Aged, 80 and over"[Age]        # 高齡受試者
```

## 布林運算子

使用布林邏輯結合搜尋詞彙。

### AND

兩個詞彙都必須存在（預設行為）：

```
diabetes AND treatment
"CRISPR-Cas9" AND "gene editing"
cancer AND immunotherapy AND "clinical trial"[Publication Type]
```

### OR

任一詞彙必須存在：

```
"heart attack" OR "myocardial infarction"
diabetes OR "diabetes mellitus"
CRISPR OR Cas9 OR "gene editing"
```

**使用情境**：同義詞和相關詞彙

### NOT

排除詞彙：

```
cancer NOT review
diabetes NOT animal
"machine learning" NOT "deep learning"
```

**注意**：可能排除同時提及兩個詞彙的相關論文。

### 結合運算子

使用括號進行複雜邏輯：

```
(diabetes OR "diabetes mellitus") AND (treatment OR therapy)

("CRISPR" OR "gene editing") AND ("therapeutic" OR "therapy")
  AND 2020:2024[Publication Date]

(cancer OR neoplasm) AND (immunotherapy OR "immune checkpoint inhibitor")
  AND ("clinical trial"[Publication Type] OR "randomized controlled trial"[Publication Type])
```

## 進階搜尋建構器

**存取**：https://pubmed.ncbi.nlm.nih.gov/advanced/

**功能**：
- 視覺化查詢建構器
- 添加多個查詢框
- 從下拉選單選擇欄位標籤
- 使用 AND/OR/NOT 結合
- 預覽結果
- 顯示最終查詢字串
- 儲存查詢

**工作流程**：
1. 在不同框中添加搜尋詞彙
2. 選擇欄位標籤
3. 選擇布林運算子
4. 預覽結果
5. 根據需要細化
6. 複製最終查詢字串
7. 在腳本中使用或儲存

**建構的查詢範例**：
```
#1: "Diabetes Mellitus, Type 2"[MeSH]
#2: "Metformin"[MeSH]
#3: "Clinical Trial"[Publication Type]
#4: 2020:2024[Publication Date]
#5: #1 AND #2 AND #3 AND #4
```

## 篩選和限制

### 文章類型

```
"Review"[Publication Type]
"Systematic Review"[Publication Type]
"Meta-Analysis"[Publication Type]
"Clinical Trial"[Publication Type]
"Randomized Controlled Trial"[Publication Type]
"Case Reports"[Publication Type]
"Comparative Study"[Publication Type]
```

### 物種

```
humans[MeSH Terms]
mice[MeSH Terms]
rats[MeSH Terms]
```

### 性別

```
"Female"[MeSH Terms]
"Male"[MeSH Terms]
```

### 年齡組

```
"Infant"[MeSH Terms]
"Child"[MeSH Terms]
"Adolescent"[MeSH Terms]
"Adult"[MeSH Terms]
"Aged"[MeSH Terms]
"Aged, 80 and over"[MeSH Terms]
```

### 文字可用性

```
free full text[Filter]           # 免費全文可用
```

### 期刊類別

```
"Journal Article"[Publication Type]
```

## E-utilities API

NCBI 透過 E-utilities（Entrez Programming Utilities）提供程式化存取。

### 概述

**基礎 URL**：`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`

**主要工具**：
- **ESearch**：搜尋並擷取 PMID
- **EFetch**：擷取完整記錄
- **ESummary**：擷取文件摘要
- **ELink**：尋找相關文章
- **EInfo**：資料庫統計

**無需 API 金鑰**，但建議使用以獲得：
- 更高的速率限制（10/秒 vs 3/秒）
- 更好的效能
- 識別您的專案

**取得 API 金鑰**：https://www.ncbi.nlm.nih.gov/account/

### ESearch - 搜尋 PubMed

擷取查詢的 PMID。

**端點**：`/esearch.fcgi`

**參數**：
- `db`：資料庫（pubmed）
- `term`：搜尋查詢
- `retmax`：最大結果數（預設 20，最大 10000）
- `retstart`：起始位置（用於分頁）
- `sort`：排序順序（relevance、pub_date、author）
- `api_key`：您的 API 金鑰（選用但建議）

**範例 URL**：
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?
  db=pubmed&
  term=diabetes+AND+treatment&
  retmax=100&
  retmode=json&
  api_key=YOUR_API_KEY
```

**回應**：
```json
{
  "esearchresult": {
    "count": "250000",
    "retmax": "100",
    "idlist": ["12345678", "12345679", ...]
  }
}
```

### EFetch - 擷取記錄

取得 PMID 的完整後設資料。

**端點**：`/efetch.fcgi`

**參數**：
- `db`：資料庫（pubmed）
- `id`：逗號分隔的 PMID
- `retmode`：格式（xml、json、text）
- `rettype`：類型（abstract、medline、full）
- `api_key`：您的 API 金鑰

**範例 URL**：
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?
  db=pubmed&
  id=12345678,12345679&
  retmode=xml&
  api_key=YOUR_API_KEY
```

**回應**：包含完整後設資料的 XML，包括：
- 標題
- 作者（含所屬機構）
- 摘要
- 期刊
- 出版日期
- DOI
- PMID、PMCID
- MeSH 詞彙
- 關鍵字

### ESummary - 取得摘要

EFetch 的輕量替代方案。

**範例**：
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?
  db=pubmed&
  id=12345678&
  retmode=json&
  api_key=YOUR_API_KEY
```

**返回**：關鍵後設資料，不含完整摘要和詳細資訊。

### ELink - 尋找相關文章

尋找相關文章或連結到其他資料庫。

**範例**：
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?
  dbfrom=pubmed&
  db=pubmed&
  id=12345678&
  linkname=pubmed_pubmed_citedin
```

**連結類型**：
- `pubmed_pubmed`：相關文章
- `pubmed_pubmed_citedin`：引用此文章的論文
- `pubmed_pmc`：PMC 全文版本
- `pubmed_protein`：相關蛋白質記錄

### 速率限制

**無 API 金鑰**：
- 每秒 3 個請求
- 超過則被封鎖

**有 API 金鑰**：
- 每秒 10 個請求
- 更適合程式化存取

**最佳實務**：
```python
import time
time.sleep(0.34)  # 約 3 請求/秒
# 或
time.sleep(0.11)  # 約 10 請求/秒（使用 API 金鑰）
```

### API 金鑰使用

**取得 API 金鑰**：
1. 建立 NCBI 帳號：https://www.ncbi.nlm.nih.gov/account/
2. 設定 → API Key Management
3. 建立新 API 金鑰
4. 複製金鑰

**在請求中使用**：
```
&api_key=YOUR_API_KEY_HERE
```

**安全儲存**：
```bash
# 在環境變數中
export NCBI_API_KEY="your_key_here"

# 在腳本中
import os
api_key = os.getenv('NCBI_API_KEY')
```

## 搜尋策略

### 全面系統性搜尋

用於系統性綜述和統合分析：

```
# 1. 識別關鍵概念
概念 1：糖尿病
概念 2：治療
概念 3：結果

# 2. 尋找 MeSH 詞彙和同義詞
概念 1："Diabetes Mellitus"[MeSH] OR diabetes OR diabetic
概念 2："Drug Therapy"[MeSH] OR treatment OR therapy OR medication
概念 3："Treatment Outcome"[MeSH] OR outcome OR efficacy OR effectiveness

# 3. 使用 AND 結合
("Diabetes Mellitus"[MeSH] OR diabetes OR diabetic)
  AND ("Drug Therapy"[MeSH] OR treatment OR therapy OR medication)
  AND ("Treatment Outcome"[MeSH] OR outcome OR efficacy OR effectiveness)

# 4. 添加篩選
AND 2015:2024[Publication Date]
AND ("Clinical Trial"[Publication Type] OR "Randomized Controlled Trial"[Publication Type])
AND English[Language]
AND humans[MeSH Terms]
```

### 尋找臨床試驗

```
# 特定疾病 + 臨床試驗
"Alzheimer Disease"[MeSH]
  AND ("Clinical Trial"[Publication Type]
       OR "Randomized Controlled Trial"[Publication Type])
  AND 2020:2024[Publication Date]

# 特定藥物試驗
"Metformin"[MeSH]
  AND "Diabetes Mellitus, Type 2"[MeSH]
  AND "Randomized Controlled Trial"[Publication Type]
```

### 尋找綜述

```
# 某主題的系統性綜述
"CRISPR-Cas Systems"[MeSH]
  AND ("Systematic Review"[Publication Type] OR "Meta-Analysis"[Publication Type])

# 高影響力期刊中的綜述
cancer immunotherapy
  AND "Review"[Publication Type]
  AND ("Nature"[Journal] OR "Science"[Journal] OR "Cell"[Journal])
```

### 尋找近期論文

```
# 過去一年的論文
"machine learning"[Title/Abstract]
  AND "drug discovery"[Title/Abstract]
  AND 2024[Publication Date]

# 特定期刊中的近期論文
"CRISPR"[Title/Abstract]
  AND "Nature"[Journal]
  AND 2023:2024[Publication Date]
```

### 作者追蹤

```
# 特定作者的近期研究
"Doudna JA"[Author] AND 2020:2024[Publication Date]

# 作者 + 主題
"Church GM"[Author] AND "synthetic biology"[Title/Abstract]
```

### 高品質證據

```
# 統合分析和系統性綜述
(diabetes OR "diabetes mellitus")
  AND (treatment OR therapy)
  AND ("Meta-Analysis"[Publication Type] OR "Systematic Review"[Publication Type])

# 僅 RCT
cancer immunotherapy
  AND "Randomized Controlled Trial"[Publication Type]
  AND 2020:2024[Publication Date]
```

## 腳本整合

### search_pubmed.py 用法

**基本搜尋**：
```bash
python scripts/search_pubmed.py "diabetes treatment"
```

**帶 MeSH 詞彙**：
```bash
python scripts/search_pubmed.py \
  --query '"Diabetes Mellitus"[MeSH] AND "Drug Therapy"[MeSH]'
```

**日期範圍篩選**：
```bash
python scripts/search_pubmed.py "CRISPR" \
  --date-start 2020-01-01 \
  --date-end 2024-12-31 \
  --limit 200
```

**出版類型篩選**：
```bash
python scripts/search_pubmed.py "cancer immunotherapy" \
  --publication-types "Clinical Trial,Randomized Controlled Trial" \
  --limit 100
```

**匯出為 BibTeX**：
```bash
python scripts/search_pubmed.py "Alzheimer's disease" \
  --limit 100 \
  --format bibtex \
  --output alzheimers.bib
```

**從檔案讀取複雜查詢**：
```bash
# 將複雜查詢儲存在 query.txt
cat > query.txt << 'EOF'
("Diabetes Mellitus, Type 2"[MeSH] OR "diabetes"[Title/Abstract])
AND ("Metformin"[MeSH] OR "metformin"[Title/Abstract])
AND "Randomized Controlled Trial"[Publication Type]
AND 2015:2024[Publication Date]
AND English[Language]
EOF

# 執行搜尋
python scripts/search_pubmed.py --query-file query.txt --limit 500
```

### 批次搜尋

```bash
# 搜尋多個主題
TOPICS=("diabetes treatment" "cancer immunotherapy" "CRISPR gene editing")

for topic in "${TOPICS[@]}"; do
  python scripts/search_pubmed.py "$topic" \
    --limit 100 \
    --output "${topic// /_}.json"
  sleep 1
done
```

### 擷取後設資料

```bash
# 搜尋返回 PMID
python scripts/search_pubmed.py "topic" --output results.json

# 擷取完整後設資料
python scripts/extract_metadata.py \
  --input results.json \
  --output references.bib
```

## 提示和最佳實務

### 搜尋建構

1. **從 MeSH 詞彙開始**：
   - 使用 MeSH 瀏覽器尋找正確詞彙
   - 比關鍵字搜尋更精確
   - 不論術語如何都能捕捉主題上的所有論文

2. **包含文字詞變體**：
   ```
   # 更好的涵蓋範圍
   ("Diabetes Mellitus"[MeSH] OR diabetes OR diabetic)
   ```

3. **適當使用欄位標籤**：
   - `[MeSH]` 用於標準化概念
   - `[Title/Abstract]` 用於特定詞彙
   - `[Author]` 用於已知作者
   - `[Journal]` 用於特定場所

4. **逐步建構**：
   ```
   # 步驟 1：基本搜尋
   diabetes

   # 步驟 2：添加特異性
   "Diabetes Mellitus, Type 2"[MeSH]

   # 步驟 3：添加治療
   "Diabetes Mellitus, Type 2"[MeSH] AND "Metformin"[MeSH]

   # 步驟 4：添加研究類型
   "Diabetes Mellitus, Type 2"[MeSH] AND "Metformin"[MeSH]
     AND "Clinical Trial"[Publication Type]

   # 步驟 5：添加日期範圍
   ... AND 2020:2024[Publication Date]
   ```

### 優化結果

1. **結果太多**：添加篩選
   - 限制出版類型
   - 縮小日期範圍
   - 添加更具體的 MeSH 詞彙
   - 使用主要主題：`[MeSH Major Topic]`

2. **結果太少**：擴大搜尋
   - 移除限制性篩選
   - 使用 OR 添加同義詞
   - 擴大日期範圍
   - 使用 MeSH 展開（預設）

3. **不相關的結果**：細化詞彙
   - 使用更具體的 MeSH 詞彙
   - 使用 NOT 添加排除
   - 使用 Title 欄位而非所有欄位
   - 添加 MeSH 副標題

### 品質控制

1. **記錄搜尋策略**：
   - 儲存精確查詢字串
   - 記錄搜尋日期
   - 註記結果數量
   - 儲存使用的篩選

2. **系統性匯出**：
   - 使用一致的檔案命名
   - 匯出為 JSON 以保持靈活性
   - 根據需要轉換為 BibTeX
   - 保留原始搜尋結果

3. **驗證擷取的引用**：
   ```bash
   python scripts/validate_citations.py pubmed_results.bib
   ```

### 保持最新

1. **設定搜尋提醒**：
   - PubMed → Save search
   - 接收電子郵件更新
   - 每日、每週或每月

2. **追蹤特定期刊**：
   ```
   "Nature"[Journal] AND CRISPR[Title]
   ```

3. **追蹤關鍵作者**：
   ```
   "Church GM"[Author]
   ```

## 常見問題和解決方案

### 問題：找不到 MeSH 詞彙

**解決方案**：
- 檢查拼寫
- 使用 MeSH 瀏覽器
- 嘗試相關詞彙
- 使用文字詞搜尋作為備選

### 問題：零結果

**解決方案**：
- 移除篩選
- 檢查查詢語法
- 使用 OR 擴大搜尋
- 嘗試同義詞

### 問題：低品質結果

**解決方案**：
- 添加出版類型篩選
- 限制近年
- 使用 MeSH 主要主題
- 按期刊品質篩選

### 問題：來自不同來源的重複項

**解決方案**：
```bash
python scripts/format_bibtex.py results.bib \
  --deduplicate \
  --output clean.bib
```

### 問題：API 速率限制

**解決方案**：
- 取得 API 金鑰（將限制提高到 10/秒）
- 在腳本中添加延遲
- 批次處理
- 使用非高峰時段

## 總結

PubMed 提供權威的生物醫學文獻搜尋：

✓ **策展內容**：MeSH 索引，品質控制
✓ **精確搜尋**：欄位標籤、MeSH 詞彙、篩選
✓ **程式化存取**：E-utilities API
✓ **免費存取**：無需訂閱
✓ **全面**：3,500 萬+ 引用，每日更新

關鍵策略：
- 使用 MeSH 詞彙進行精確搜尋
- 結合文字詞以獲得全面涵蓋
- 應用適當的欄位標籤
- 按出版類型和日期篩選
- 使用 E-utilities API 進行自動化
- 記錄搜尋策略以確保可重現性

對於跨學科的更廣泛涵蓋，可搭配 Google Scholar 使用。
