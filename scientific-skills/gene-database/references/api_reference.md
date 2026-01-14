# NCBI Gene API 參考

本文件提供程式化存取 NCBI Gene 資料庫的詳細 API 文件。

## 目錄

1. [E-utilities API](#e-utilities-api)
2. [NCBI Datasets API](#ncbi-datasets-api)
3. [認證和速率限制](#認證和速率限制)
4. [錯誤處理](#錯誤處理)

---

## E-utilities API

E-utilities（Entrez Programming Utilities）為 NCBI 的 Entrez 資料庫提供穩定的介面。

### 基礎 URL

```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
```

### 通用參數

- `db` - 資料庫名稱（使用 `gene` 表示 Gene 資料庫）
- `api_key` - 用於更高速率限制的 API 金鑰
- `retmode` - 回傳格式（json、xml、text）
- `retmax` - 最大回傳記錄數

### ESearch - 搜尋資料庫

搜尋符合文字查詢的基因。

**端點：** `esearch.fcgi`

**參數：**
- `db=gene`（必要）- 要搜尋的資料庫
- `term`（必要）- 搜尋查詢
- `retmax` - 最大結果數（預設：20）
- `retmode` - json 或 xml（預設：xml）
- `usehistory=y` - 將結果儲存在歷史伺服器上用於大型結果集

**查詢語法：**
- 基因符號：`BRCA1[gene]` 或 `BRCA1[gene name]`
- 生物體：`human[organism]` 或 `9606[taxid]`
- 組合詞彙：`BRCA1[gene] AND human[organism]`
- 疾病：`muscular dystrophy[disease]`
- 染色體：`17q21[chromosome]`
- GO 術語：`GO:0006915[biological process]`

**範例請求：**

```bash
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&term=BRCA1[gene]+AND+human[organism]&retmode=json"
```

**回應格式（JSON）：**

```json
{
  "esearchresult": {
    "count": "1",
    "retmax": "1",
    "retstart": "0",
    "idlist": ["672"],
    "translationset": [],
    "querytranslation": "BRCA1[Gene Name] AND human[Organism]"
  }
}
```

### ESummary - 文件摘要

擷取 Gene ID 的文件摘要。

**端點：** `esummary.fcgi`

**參數：**
- `db=gene`（必要）- 資料庫
- `id`（必要）- 逗號分隔的 Gene ID（最多 500 個）
- `retmode` - json 或 xml（預設：xml）

**範例請求：**

```bash
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=gene&id=672&retmode=json"
```

**回應格式（JSON）：**

```json
{
  "result": {
    "672": {
      "uid": "672",
      "name": "BRCA1",
      "description": "BRCA1 DNA repair associated",
      "organism": {
        "scientificname": "Homo sapiens",
        "commonname": "human",
        "taxid": 9606
      },
      "chromosome": "17",
      "geneticsource": "genomic",
      "maplocation": "17q21.31",
      "nomenclaturesymbol": "BRCA1",
      "nomenclaturename": "BRCA1 DNA repair associated"
    }
  }
}
```

### EFetch - 完整記錄

以各種格式取得詳細的基因記錄。

**端點：** `efetch.fcgi`

**參數：**
- `db=gene`（必要）- 資料庫
- `id`（必要）- 逗號分隔的 Gene ID
- `retmode` - xml、text、asn.1（預設：xml）
- `rettype` - gene_table、docsum

**範例請求：**

```bash
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&id=672&retmode=xml"
```

**XML 回應：** 包含詳細的基因資訊，包括：
- 基因命名法
- 序列位置
- 轉錄本變體
- 蛋白質產物
- 基因本體註解
- 交叉參考
- 出版物

### ELink - 相關記錄

在 Gene 或其他資料庫中查找相關記錄。

**端點：** `elink.fcgi`

**參數：**
- `dbfrom=gene`（必要）- 來源資料庫
- `db`（必要）- 目標資料庫（gene、nuccore、protein、pubmed 等）
- `id`（必要）- Gene ID

**範例請求：**

```bash
# 取得相關的 PubMed 文章
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=gene&db=pubmed&id=672&retmode=json"
```

### EInfo - 資料庫資訊

取得關於 Gene 資料庫的資訊。

**端點：** `einfo.fcgi`

**參數：**
- `db=gene` - 要查詢的資料庫

**範例請求：**

```bash
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/einfo.fcgi?db=gene&retmode=json"
```

---

## NCBI Datasets API

Datasets API 提供簡化的基因資料存取，包含元資料和序列。

### 基礎 URL

```
https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene
```

### 認證

在請求標頭中包含 API 金鑰：

```
api-key: YOUR_API_KEY
```

### 按 ID 取得基因

透過 Gene ID 擷取基因資料。

**端點：** `GET /gene/id/{gene_id}`

**範例請求：**

```bash
curl "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/id/672"
```

**回應格式（JSON）：**

```json
{
  "genes": [
    {
      "gene": {
        "gene_id": "672",
        "symbol": "BRCA1",
        "description": "BRCA1 DNA repair associated",
        "tax_name": "Homo sapiens",
        "taxid": 9606,
        "chromosomes": ["17"],
        "type": "protein-coding",
        "synonyms": ["BRCC1", "FANCS", "PNCA4", "RNF53"],
        "nomenclature_authority": {
          "authority": "HGNC",
          "identifier": "HGNC:1100"
        },
        "genomic_ranges": [
          {
            "accession_version": "NC_000017.11",
            "range": [
              {
                "begin": 43044295,
                "end": 43170245,
                "orientation": "minus"
              }
            ]
          }
        ],
        "transcripts": [
          {
            "accession_version": "NM_007294.4",
            "length": 7207
          }
        ]
      }
    }
  ]
}
```

### 按符號取得基因

透過符號和生物體擷取基因資料。

**端點：** `GET /gene/symbol/{symbol}/taxon/{taxon}`

**參數：**
- `{symbol}` - 基因符號（例如：BRCA1）
- `{taxon}` - 分類 ID（例如：9606 表示人類）

**範例請求：**

```bash
curl "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/symbol/BRCA1/taxon/9606"
```

### 取得多個基因

擷取多個基因的資料。

**端點：** `POST /gene/id`

**請求主體：**

```json
{
  "gene_ids": ["672", "7157", "5594"]
}
```

**範例請求：**

```bash
curl -X POST "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/gene/id" \
  -H "Content-Type: application/json" \
  -d '{"gene_ids": ["672", "7157", "5594"]}'
```

---

## 認證和速率限制

### 取得 API 金鑰

1. 在 https://www.ncbi.nlm.nih.gov/account/ 建立 NCBI 帳戶
2. 導航至設定 → API 金鑰管理
3. 產生新的 API 金鑰
4. 在請求中包含金鑰

### 速率限制

**E-utilities：**
- 無 API 金鑰：每秒 3 個請求
- 有 API 金鑰：每秒 10 個請求

**Datasets API：**
- 無 API 金鑰：每秒 5 個請求
- 有 API 金鑰：每秒 10 個請求

### 使用指南

1. **在請求中包含電子郵件：** 在 E-utilities 請求中添加 `&email=your@email.com`
2. **實作速率限制：** 在請求之間使用延遲
3. **對大型查詢使用 POST：** 處理多個 ID 時
4. **快取結果：** 在本地儲存經常存取的資料
5. **優雅地處理錯誤：** 實作帶有指數退避的重試邏輯

---

## 錯誤處理

### HTTP 狀態碼

- `200 OK` - 請求成功
- `400 Bad Request` - 無效的參數或格式錯誤的查詢
- `404 Not Found` - 找不到 Gene ID 或符號
- `429 Too Many Requests` - 超過速率限制
- `500 Internal Server Error` - 伺服器錯誤（帶退避重試）

### E-utilities 錯誤訊息

E-utilities 在回應主體中回傳錯誤：

**XML 格式：**
```xml
<ERROR>Empty id list - nothing to do</ERROR>
```

**JSON 格式：**
```json
{
  "error": "Invalid db name"
}
```

### 常見錯誤

1. **空結果集**
   - 原因：找不到基因符號或 ID
   - 解決方案：驗證拼寫，檢查生物體過濾器

2. **超過速率限制**
   - 原因：請求過多
   - 解決方案：添加延遲，使用 API 金鑰

3. **無效的查詢語法**
   - 原因：格式錯誤的搜尋詞彙
   - 解決方案：使用正確的欄位標籤（例如：`[gene]`、`[organism]`）

4. **逾時**
   - 原因：大型結果集或連線緩慢
   - 解決方案：使用 History Server，減少結果大小

### 重試策略

對失敗的請求實作指數退避：

```python
import time

def retry_request(func, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if attempt < max_attempts - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait_time)
            else:
                raise
```

---

## 常見分類 ID

| 生物體 | 學名 | 分類 ID |
|--------|------|---------|
| 人類 | Homo sapiens | 9606 |
| 小鼠 | Mus musculus | 10090 |
| 大鼠 | Rattus norvegicus | 10116 |
| 斑馬魚 | Danio rerio | 7955 |
| 果蠅 | Drosophila melanogaster | 7227 |
| 線蟲 | Caenorhabditis elegans | 6239 |
| 酵母 | Saccharomyces cerevisiae | 4932 |
| 擬南芥 | Arabidopsis thaliana | 3702 |
| 大腸桿菌 | Escherichia coli | 562 |

---

## 其他資源

- **E-utilities 文件：** https://www.ncbi.nlm.nih.gov/books/NBK25501/
- **Datasets API 文件：** https://www.ncbi.nlm.nih.gov/datasets/docs/v2/
- **Gene 資料庫說明：** https://www.ncbi.nlm.nih.gov/gene/
- **API 金鑰註冊：** https://www.ncbi.nlm.nih.gov/account/
