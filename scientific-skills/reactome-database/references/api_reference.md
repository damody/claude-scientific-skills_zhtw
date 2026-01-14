# Reactome API 參考

本文件提供 Reactome REST API 的完整參考資訊。

## 基礎 URL

- **內容服務**：`https://reactome.org/ContentService`
- **分析服務**：`https://reactome.org/AnalysisService`

## 內容服務 API

內容服務透過 REST 端點提供對 Reactome 策展路徑資料的存取。

### 資料庫資訊

#### 取得資料庫版本
```
GET /data/database/version
```

**回應：** 包含資料庫版本號碼的純文字

**範例：**
```python
import requests
response = requests.get("https://reactome.org/ContentService/data/database/version")
print(response.text)  # 例如 "94"
```

#### 取得資料庫名稱
```
GET /data/database/name
```

**回應：** 包含資料庫名稱的純文字

### 實體查詢

#### 依 ID 查詢實體
```
GET /data/query/{id}
```

**參數：**
- `id`（路徑）：穩定識別碼或資料庫 ID（例如 "R-HSA-69278"）

**回應：** 包含完整實體資訊的 JSON 物件，包括：
- `stId`：穩定識別碼
- `displayName`：人類可讀名稱
- `schemaClass`：實體類型（Pathway、Reaction、Complex 等）
- `species`：物種資訊陣列
- 額外的類型特定欄位

**範例：**
```python
import requests
response = requests.get("https://reactome.org/ContentService/data/query/R-HSA-69278")
pathway = response.json()
print(f"路徑: {pathway['displayName']}")
print(f"物種: {pathway['species'][0]['displayName']}")
```

#### 查詢實體屬性
```
GET /data/query/{id}/{attribute}
```

**參數：**
- `id`（路徑）：實體識別碼
- `attribute`（路徑）：特定屬性名稱（例如 "displayName"、"compartment"）

**回應：** 根據屬性類型返回 JSON 或純文字

**範例：**
```python
response = requests.get("https://reactome.org/ContentService/data/query/R-HSA-69278/displayName")
name = response.text
```

### 路徑查詢

#### 取得路徑實體
```
GET /data/event/{id}/participatingPhysicalEntities
```

**參數：**
- `id`（路徑）：路徑或反應穩定識別碼

**回應：** 參與路徑的物理實體（蛋白質、複合體、小分子）的 JSON 陣列

**範例：**
```python
response = requests.get(
    "https://reactome.org/ContentService/data/event/R-HSA-69278/participatingPhysicalEntities"
)
entities = response.json()
for entity in entities:
    print(f"{entity['stId']}: {entity['displayName']} ({entity['schemaClass']})")
```

#### 取得包含的事件
```
GET /data/pathway/{id}/containedEvents
```

**參數：**
- `id`（路徑）：路徑穩定識別碼

**回應：** 路徑中包含的事件（反應、子路徑）的 JSON 陣列

### 搜尋查詢

#### 依名稱搜尋
```
GET /data/query?name={query}
```

**參數：**
- `name`（查詢）：搜尋詞

**回應：** 匹配實體的 JSON 陣列

**範例：**
```python
response = requests.get(
    "https://reactome.org/ContentService/data/query",
    params={"name": "glycolysis"}
)
results = response.json()
```

## 分析服務 API

分析服務執行路徑富集和表現分析。

### 提交分析

#### 提交識別碼（POST）
```
POST /identifiers/
POST /identifiers/projection/  # 僅對應到人類路徑
```

**標頭：**
- `Content-Type: text/plain`

**內容：**
- 對於過度代表性：識別碼的純文字列表（每行一個）
- 對於表現分析：以 "#" 開頭標頭的 TSV 格式

**表現資料格式：**
```
#Gene	Sample1	Sample2	Sample3
TP53	2.5	3.1	2.8
BRCA1	1.2	1.5	1.3
```

**回應：** 包含以下內容的 JSON 物件：
```json
{
  "summary": {
    "token": "MzUxODM3NTQzMDAwMDA1ODI4MA==",
    "type": "OVERREPRESENTATION",
    "species": "9606",
    "sampleName": null,
    "fileName": null,
    "text": true
  },
  "pathways": [
    {
      "stId": "R-HSA-69278",
      "name": "Cell Cycle, Mitotic",
      "species": {
        "name": "Homo sapiens",
        "taxId": "9606"
      },
      "entities": {
        "found": 15,
        "total": 450,
        "pValue": 0.0000234,
        "fdr": 0.00156
      },
      "reactions": {
        "found": 12,
        "total": 342
      }
    }
  ],
  "resourceSummary": [
    {
      "resource": "TOTAL",
      "pathways": 25
    }
  ]
}
```

**範例：**
```python
import requests

# 過度代表性分析
identifiers = ["TP53", "BRCA1", "EGFR", "MYC", "CDK1"]
data = "\n".join(identifiers)

response = requests.post(
    "https://reactome.org/AnalysisService/identifiers/",
    headers={"Content-Type": "text/plain"},
    data=data
)

result = response.json()
token = result["summary"]["token"]

# 處理路徑
for pathway in result["pathways"]:
    print(f"路徑: {pathway['name']}")
    print(f"  找到: {pathway['entities']['found']}/{pathway['entities']['total']}")
    print(f"  p-value: {pathway['entities']['pValue']:.6f}")
    print(f"  FDR: {pathway['entities']['fdr']:.6f}")
```

#### 提交檔案（表單上傳）
```
POST /identifiers/form/
```

**Content-Type：** `multipart/form-data`

**參數：**
- `file`：包含識別碼或表現資料的檔案

#### 提交 URL
```
POST /identifiers/url/
```

**參數：**
- `url`：指向資料檔案的 URL

### 檢索分析結果

#### 依 Token 取得結果
```
GET /token/{token}
GET /token/{token}/projection/  # 帶物種投射
```

**參數：**
- `token`（路徑）：提交返回的分析 token

**回應：** 與初始分析回應相同的結構

**範例：**
```python
token = "MzUxODM3NTQzMDAwMDA1ODI4MA=="
response = requests.get(f"https://reactome.org/AnalysisService/token/{token}")
results = response.json()
```

**注意：** Token 有效期為 7 天

#### 過濾結果
```
GET /token/{token}/filter/pathways?resource={resource}
```

**參數：**
- `token`（路徑）：分析 token
- `resource`（查詢）：資源過濾器（例如 "TOTAL"、"UNIPROT"、"ENSEMBL"）

### 下載結果

#### 下載為 CSV
```
GET /download/{token}/pathways/{resource}/result.csv
```

#### 下載對應
```
GET /download/{token}/entities/found/{resource}/mapping.tsv
```

## 支援的識別碼

Reactome 自動偵測和處理各種識別碼類型：

### 蛋白質和基因
- **UniProt**：P04637
- **基因符號**：TP53
- **Ensembl**：ENSG00000141510
- **EntrezGene**：7157
- **RefSeq**：NM_000546
- **OMIM**：191170

### 小分子
- **ChEBI**：CHEBI:15377
- **KEGG Compound**：C00031
- **PubChem**：702

### 其他
- **miRBase**：hsa-miR-21
- **InterPro**：IPR011616

## 回應格式

### JSON 物件

實體物件包含標準化欄位：
```json
{
  "stId": "R-HSA-69278",
  "displayName": "Cell Cycle, Mitotic",
  "schemaClass": "Pathway",
  "species": [
    {
      "dbId": 48887,
      "displayName": "Homo sapiens",
      "taxId": "9606"
    }
  ],
  "isInDisease": false
}
```

### TSV 格式

對於批次查詢，TSV 返回：
```
stId	displayName	schemaClass
R-HSA-69278	Cell Cycle, Mitotic	Pathway
R-HSA-69306	DNA Replication	Pathway
```

## 錯誤回應

### HTTP 狀態碼
- `200`：成功
- `400`：錯誤請求（無效參數）
- `404`：未找到（無效 ID）
- `415`：不支援的媒體類型
- `500`：內部伺服器錯誤

### 錯誤 JSON 結構
```json
{
  "code": 404,
  "reason": "NOT_FOUND",
  "messages": ["Pathway R-HSA-INVALID not found"]
}
```

## 速率限制

Reactome 目前不強制嚴格的速率限制，但請考慮：
- 在請求之間實施合理的延遲
- 在可用時使用批次操作
- 適當時快取結果
- 遵守 7 天的 token 有效期

## 最佳實踐

### 1. 使用分析 Token
儲存和重用分析 token 以避免冗餘計算：
```python
# 分析後儲存 token
token = result["summary"]["token"]
save_token(token)  # 儲存到檔案或資料庫

# 稍後檢索結果
result = requests.get(f"https://reactome.org/AnalysisService/token/{token}")
```

### 2. 批次查詢
在單一請求中提交多個識別碼而非個別查詢：
```python
# 好：單一批次請求
identifiers = ["TP53", "BRCA1", "EGFR"]
result = analyze_batch(identifiers)

# 避免：多個個別請求
# for gene in genes:
#     result = analyze_single(gene)  # 不要這樣做
```

### 3. 適當處理物種
使用 `/projection/` 端點將非人類識別碼對應到人類路徑：
```python
# 對於小鼠基因，投射到人類路徑
response = requests.post(
    "https://reactome.org/AnalysisService/identifiers/projection/",
    headers={"Content-Type": "text/plain"},
    data=mouse_genes
)
```

### 4. 處理大型結果集
對於返回許多路徑的分析，依顯著性過濾：
```python
significant_pathways = [
    p for p in result["pathways"]
    if p["entities"]["fdr"] < 0.05
]
```

## 整合範例

### 完整分析工作流程
```python
import requests
import json

def analyze_gene_list(genes, output_file="analysis_results.json"):
    """
    對基因列表執行路徑富集分析
    """
    # 提交分析
    data = "\n".join(genes)
    response = requests.post(
        "https://reactome.org/AnalysisService/identifiers/",
        headers={"Content-Type": "text/plain"},
        data=data
    )

    if response.status_code != 200:
        raise Exception(f"分析失敗: {response.text}")

    result = response.json()
    token = result["summary"]["token"]

    # 過濾顯著路徑（FDR < 0.05）
    significant = [
        p for p in result["pathways"]
        if p["entities"]["fdr"] < 0.05
    ]

    # 儲存結果
    with open(output_file, "w") as f:
        json.dump({
            "token": token,
            "total_pathways": len(result["pathways"]),
            "significant_pathways": len(significant),
            "pathways": significant
        }, f, indent=2)

    # 為最佳路徑生成瀏覽器 URL
    if significant:
        top_pathway = significant[0]
        url = f"https://reactome.org/PathwayBrowser/#{top_pathway['stId']}&DTAB=AN&ANALYSIS={token}"
        print(f"檢視最佳結果: {url}")

    return result

# 使用
genes = ["TP53", "BRCA1", "BRCA2", "CDK1", "CDK2"]
result = analyze_gene_list(genes)
```

## 額外資源

- **互動式 API 文件**：https://reactome.org/dev/content-service
- **分析服務文件**：https://reactome.org/dev/analysis
- **使用者指南**：https://reactome.org/userguide
- **資料下載**：https://reactome.org/download-data
