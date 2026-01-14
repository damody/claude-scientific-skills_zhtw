# PatentSearch API 參考

## 概述

PatentSearch API 是 USPTO 基於 ElasticSearch 的現代專利搜尋系統，於 2025 年 5 月取代了舊版 PatentsView API。它提供截至 2025 年 6 月 30 日的專利資料存取，並定期更新。

**基礎 URL：** `https://search.patentsview.org/api/v1/`

## 認證

所有 API 請求都需要在請求標頭中使用 API 金鑰進行認證：

```
X-Api-Key: YOUR_API_KEY
```

請在此註冊 API 金鑰：https://account.uspto.gov/api-manager/

## 速率限制

- 每個 API 金鑰**每分鐘 45 個請求**
- 超過速率限制會導致 HTTP 429 錯誤

## 可用端點

### 核心專利與出版物端點

- **`/patent`** - 一般專利資料（已核准專利）
- **`/publication`** - 核准前出版物資料
- **`/publication/rel_app_text`** - 出版物的相關申請資料

### 實體端點

- **`/inventor`** - 發明人資訊，包含位置和性別代碼欄位
- **`/assignee`** - 受讓人詳情，包含位置識別碼
- **`/location`** - 地理資料，包含經緯度座標
- **`/attorney`** - 法律代理人資訊

### 分類端點

- **`/cpc_subclass`** - 合作專利分類（Cooperative Patent Classification）子類級別
- **`/cpc_at_issue`** - 專利核准日期時的 CPC 分類
- **`/uspc`** - 美國專利分類資料
- **`/wipo`** - 世界智慧財產權組織分類
- **`/ipc`** - 國際專利分類

### 文字資料端點（測試版）

- **`/brief_summary_text`** - 專利簡要摘要（已核准和核准前）
- **`/claims`** - 專利請求項文字
- **`/drawing_description_text`** - 圖式說明
- **`/detail_description_text`** - 詳細說明文字

*注意：文字端點為測試版，資料主要來自 2023 年以後。歷史資料回填正在進行中。*

### 支援端點

- **`/other_reference`** - 專利參考資料
- **`/related_document`** - 專利間的交叉參考

## 查詢參數

所有端點支援四個主要參數：

### 1. 查詢字串（`q`）

使用 JSON 查詢物件篩選資料。**必要參數。**

**查詢運算子：**

- **相等**：`{"field": "value"}` 或 `{"field": {"_eq": "value"}}`
- **不相等**：`{"field": {"_neq": "value"}}`
- **比較**：`_gt`、`_gte`、`_lt`、`_lte`
- **字串匹配**：
  - `_begins` - 開頭符合
  - `_contains` - 子字串匹配
- **全文搜尋**（建議用於文字欄位）：
  - `_text_all` - 所有詞語都必須匹配
  - `_text_any` - 任何詞語匹配
  - `_text_phrase` - 精確詞組匹配
- **邏輯運算子**：`_and`、`_or`、`_not`
- **陣列匹配**：使用陣列表示 OR 條件

**範例：**

```json
// 簡單相等
{"patent_number": "11234567"}

// 日期範圍
{"patent_date": {"_gte": "2020-01-01", "_lte": "2020-12-31"}}

// 文字搜尋（建議用於文字欄位）
{"patent_abstract": {"_text_all": ["machine", "learning"]}}

// 發明人姓名
{"inventor_name": {"_text_phrase": "John Smith"}}

// 使用邏輯運算子的複雜查詢
{
  "_and": [
    {"patent_date": {"_gte": "2020-01-01"}},
    {"assignee_organization": {"_text_any": ["Google", "Alphabet"]}}
  ]
}

// 使用陣列表示 OR 條件
{"cpc_subclass_id": ["H04N", "H04L"]}
```

### 2. 欄位列表（`f`）

指定回應中要回傳的欄位。選用 - 每個端點都有預設欄位。

**格式：** JSON 欄位名稱陣列

```json
["patent_number", "patent_title", "patent_date", "inventor_name"]
```

### 3. 排序（`s`）

依指定欄位排序結果。選用。

**格式：** 包含欄位名稱和方向的 JSON 陣列

```json
[{"patent_date": "desc"}]
```

### 4. 選項（`o`）

控制分頁和其他設定。選用。

**可用選項：**

- `page` - 頁碼（預設：1）
- `per_page` - 每頁記錄數（預設：100，最大：1,000）
- `pad_patent_id` - 用前導零填充專利 ID（預設：false）
- `exclude_withdrawn` - 排除撤回的專利（預設：true）

**格式：** JSON 物件

```json
{
  "page": 1,
  "per_page": 500,
  "exclude_withdrawn": false
}
```

## 回應格式

所有回應遵循此結構：

```json
{
  "error": false,
  "count": 100,
  "total_hits": 5432,
  "patents": [...],
  // 或 "inventors": [...], "assignees": [...] 等
}
```

- `error` - 布林值，表示是否發生錯誤
- `count` - 目前回應中的記錄數
- `total_hits` - 匹配記錄總數
- 端點特定資料陣列（例如 `patents`、`inventors`）

## 完整請求範例

### 使用 curl

```bash
curl -X POST "https://search.patentsview.org/api/v1/patent" \
  -H "X-Api-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "q": {
      "_and": [
        {"patent_date": {"_gte": "2024-01-01"}},
        {"patent_abstract": {"_text_all": ["artificial", "intelligence"]}}
      ]
    },
    "f": ["patent_number", "patent_title", "patent_date", "assignee_organization"],
    "s": [{"patent_date": "desc"}],
    "o": {"per_page": 100}
  }'
```

### 使用 Python

```python
import requests

url = "https://search.patentsview.org/api/v1/patent"
headers = {
    "X-Api-Key": "YOUR_API_KEY",
    "Content-Type": "application/json"
}
data = {
    "q": {
        "_and": [
            {"patent_date": {"_gte": "2024-01-01"}},
            {"patent_abstract": {"_text_all": ["artificial", "intelligence"]}}
        ]
    },
    "f": ["patent_number", "patent_title", "patent_date", "assignee_organization"],
    "s": [{"patent_date": "desc"}],
    "o": {"per_page": 100}
}

response = requests.post(url, headers=headers, json=data)
results = response.json()
```

## 常見欄位名稱

### 專利端點欄位

- `patent_number` - 專利號
- `patent_title` - 專利標題
- `patent_date` - 核准日期
- `patent_abstract` - 摘要文字
- `patent_type` - 專利類型
- `inventor_name` - 發明人姓名（陣列）
- `assignee_organization` - 受讓人公司名稱（陣列）
- `cpc_subclass_id` - CPC 分類代碼
- `uspc_class` - 美國分類代碼
- `cited_patent_number` - 引用的其他專利
- `citedby_patent_number` - 引用此專利的專利

完整欄位字典請參閱：https://search.patentsview.org/docs/

## 最佳實務

1. **對文字欄位使用 `_text*` 運算子** - 比 `_contains` 或 `_begins` 效能更好
2. **只請求需要的欄位** - 減少回應大小並提高效能
3. **實作分頁** - 高效處理大型結果集
4. **遵守速率限制** - 對 429 錯誤實作退避/重試邏輯
5. **快取結果** - 減少重複的 API 呼叫
6. **使用日期範圍** - 縮小搜尋以提高效能

## 錯誤處理

常見 HTTP 狀態碼：

- **200** - 成功
- **400** - 錯誤請求（無效的查詢語法）
- **401** - 未授權（缺少或無效的 API 金鑰）
- **429** - 請求過多（超過速率限制）
- **500** - 伺服器錯誤

## 最近更新（2025 年 2 月）

- 資料更新至 2024 年 12 月 31 日
- 新增 `pad_patent_id` 選項用於格式化專利 ID
- 新增 `exclude_withdrawn` 選項以顯示撤回的專利
- 文字端點繼續測試版回填

## 資源

- **官方文件**：https://search.patentsview.org/docs/
- **API 金鑰註冊**：https://account.uspto.gov/api-manager/
- **舊版 API 通知**：舊版 PatentsView API 於 2025 年 5 月 1 日停用
