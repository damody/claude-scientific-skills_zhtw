# STRING 資料庫 API 參考

## 概述

STRING（Search Tool for the Retrieval of Interacting Genes/Proteins，交互作用基因/蛋白質檢索搜尋工具）是一個綜合資料庫，整合來自超過 40 個來源的已知和預測蛋白質-蛋白質交互作用。

**資料庫統計（v12.0+）：**
- 涵蓋範圍：5000+ 基因組
- 蛋白質：約 5930 萬
- 交互作用：200+ 億
- 資料類型：物理交互作用、功能關聯、共表現、共現、文字探勘、資料庫

**核心資料資源：**由 Global Biodata Coalition 和 ELIXIR 認定

## API 基礎 URL

- **目前版本**：https://string-db.org/api
- **特定版本**：https://version-12-0.string-db.org/api（用於可重現性）
- **API 文件**：https://string-db.org/help/api/

## 最佳實踐

1. **識別符對應**：始終先使用 `get_string_ids` 對應識別符以加速後續查詢
2. **使用 STRING ID**：優先使用 STRING 識別符（例如 `9606.ENSP00000269305`）而非基因名稱
3. **指定物種**：對於超過 10 個蛋白質的網路，始終指定 NCBI 分類群 ID
4. **速率限制**：API 呼叫間等待 1 秒以避免伺服器過載
5. **版本化 URL**：使用特定版本 URL 以確保可重現研究
6. **POST 優於 GET**：對大型蛋白質列表使用 POST 請求
7. **呼叫者身份**：包含 `caller_identity` 參數以便追蹤（例如您的應用程式名稱）

## API 方法

### 1. 識別符對應（`get_string_ids`）

**目的**：將常見蛋白質名稱、基因符號、UniProt ID 和其他識別符對應到 STRING 識別符。

**端點**：`/api/tsv/get_string_ids`

**參數**：
- `identifiers`（必要）：以換行符（`%0d`）分隔的蛋白質名稱/ID
- `species`（必要）：NCBI 分類群 ID
- `limit`：每個識別符的匹配數量（預設：1）
- `echo_query`：在輸出中包含查詢詞（1 或 0）
- `caller_identity`：應用程式識別符

**輸出格式**：TSV 包含以下欄位：
- `queryItem`：原始查詢
- `queryIndex`：查詢位置
- `stringId`：STRING 識別符
- `ncbiTaxonId`：物種分類群 ID
- `taxonName`：物種名稱
- `preferredName`：首選基因名稱
- `annotation`：蛋白質描述

**範例**：
```
identifiers=TP53%0dBRCA1&species=9606&limit=1
```

**使用案例**：
- 將基因符號轉換為 STRING ID
- 驗證蛋白質識別符
- 尋找標準蛋白質名稱

### 2. 網路資料（`network`）

**目的**：以表格格式檢索蛋白質-蛋白質交互作用網路資料。

**端點**：`/api/tsv/network`

**參數**：
- `identifiers`（必要）：以 `%0d` 分隔的蛋白質 ID
- `species`：NCBI 分類群 ID
- `required_score`：信心閾值 0-1000（預設：400）
  - 150：低信心
  - 400：中等信心
  - 700：高信心
  - 900：最高信心
- `network_type`：`functional`（預設）或 `physical`
- `add_nodes`：添加 N 個交互作用蛋白質（0-10）
- `caller_identity`：應用程式識別符

**輸出格式**：TSV 包含以下欄位：
- `stringId_A`、`stringId_B`：交互作用蛋白質
- `preferredName_A`、`preferredName_B`：基因名稱
- `ncbiTaxonId`：物種
- `score`：組合交互作用分數（0-1000）
- `nscore`：鄰近性分數
- `fscore`：融合分數
- `pscore`：系統發育圖譜分數
- `ascore`：共表現分數
- `escore`：實驗分數
- `dscore`：資料庫分數
- `tscore`：文字探勘分數

**網路類型**：
- **功能性（Functional）**：所有交互作用證據類型（建議用於大多數分析）
- **物理性（Physical）**：僅直接物理結合證據

**範例**：
```
identifiers=9606.ENSP00000269305%0d9606.ENSP00000275493&required_score=700
```

### 3. 網路圖像（`image/network`）

**目的**：生成 PNG 圖像格式的視覺網路表示。

**端點**：`/api/image/network`

**參數**：
- `identifiers`（必要）：以 `%0d` 分隔的蛋白質 ID
- `species`：NCBI 分類群 ID
- `required_score`：信心閾值 0-1000
- `network_flavor`：視覺化風格
  - `evidence`：以彩色線條顯示證據類型
  - `confidence`：以線條粗細顯示信心程度
  - `actions`：顯示活化/抑制交互作用
- `add_nodes`：添加 N 個交互作用蛋白質（0-10）
- `caller_identity`：應用程式識別符

**輸出**：PNG 圖像（二進位資料）

**圖像規格**：
- 格式：PNG
- 大小：根據網路大小自動縮放
- 可用高解析度選項（添加 `?highres=1`）

**範例**：
```
identifiers=TP53%0dMDM2&species=9606&network_flavor=evidence
```

### 4. 交互作用夥伴（`interaction_partners`）

**目的**：檢索給定蛋白質的所有 STRING 交互作用夥伴。

**端點**：`/api/tsv/interaction_partners`

**參數**：
- `identifiers`（必要）：蛋白質 ID
- `species`：NCBI 分類群 ID
- `required_score`：信心閾值 0-1000
- `limit`：最大夥伴數量（預設：10）
- `caller_identity`：應用程式識別符

**輸出格式**：TSV 與 `network` 方法相同的欄位

**使用案例**：
- 尋找樞紐蛋白質
- 擴展網路
- 發現新的交互作用

**範例**：
```
identifiers=TP53&species=9606&limit=20&required_score=700
```

### 5. 功能富集（`enrichment`）

**目的**：對一組蛋白質跨多個註解資料庫執行功能富集分析。

**端點**：`/api/tsv/enrichment`

**參數**：
- `identifiers`（必要）：蛋白質 ID 列表
- `species`（必要）：NCBI 分類群 ID
- `caller_identity`：應用程式識別符

**富集類別**：
- **基因本體論（Gene Ontology）**：生物過程、分子功能、細胞組成
- **KEGG 途徑**：代謝和訊號途徑
- **Pfam**：蛋白質結構域
- **InterPro**：蛋白質家族和結構域
- **SMART**：結構域架構
- **UniProt 關鍵詞**：策展的功能關鍵詞

**輸出格式**：TSV 包含以下欄位：
- `category`：註解類別
- `term`：詞彙 ID
- `description`：詞彙描述
- `number_of_genes`：具有此詞彙的輸入基因數
- `number_of_genes_in_background`：具有此詞彙的總基因數
- `ncbiTaxonId`：物種
- `inputGenes`：逗號分隔的基因列表
- `preferredNames`：逗號分隔的基因名稱
- `p_value`：富集 p 值（未校正）
- `fdr`：偽發現率（校正後的 p 值）

**統計方法**：Fisher 精確檢定配合 Benjamini-Hochberg FDR 校正

**範例**：
```
identifiers=TP53%0dMDM2%0dATM%0dCHEK2&species=9606
```

### 6. PPI 富集（`ppi_enrichment`）

**目的**：測試網路是否比隨機預期有顯著更多的交互作用。

**端點**：`/api/json/ppi_enrichment`

**參數**：
- `identifiers`（必要）：蛋白質 ID 列表
- `species`：NCBI 分類群 ID
- `required_score`：信心閾值
- `caller_identity`：應用程式識別符

**輸出格式**：JSON 包含以下欄位：
- `number_of_nodes`：網路中的蛋白質數量
- `number_of_edges`：觀察到的交互作用數量
- `expected_number_of_edges`：預期的交互作用數量（隨機）
- `p_value`：統計顯著性

**解讀**：
- p 值 < 0.05：網路顯著富集
- 低 p 值表示蛋白質形成功能模組

**範例**：
```
identifiers=TP53%0dMDM2%0dATM%0dCHEK2&species=9606
```

### 7. 同源性分數（`homology`）

**目的**：檢索蛋白質相似性/同源性分數。

**端點**：`/api/tsv/homology`

**參數**：
- `identifiers`（必要）：蛋白質 ID
- `species`：NCBI 分類群 ID
- `caller_identity`：應用程式識別符

**輸出格式**：TSV 包含蛋白質間的同源性分數

**使用案例**：
- 識別蛋白質家族
- 旁系同源物分析
- 跨物種比較

### 8. 版本資訊（`version`）

**目的**：返回目前的 STRING 資料庫版本。

**端點**：`/api/tsv/version`

**輸出**：版本字串（例如 "12.0"）

## 常見物種 NCBI 分類群 ID

| 生物體 | 俗名 | 分類群 ID |
|----------|-------------|----------|
| Homo sapiens | 人類 | 9606 |
| Mus musculus | 小鼠 | 10090 |
| Rattus norvegicus | 大鼠 | 10116 |
| Drosophila melanogaster | 果蠅 | 7227 |
| Caenorhabditis elegans | 線蟲 | 6239 |
| Saccharomyces cerevisiae | 酵母 | 4932 |
| Arabidopsis thaliana | 阿拉伯芥 | 3702 |
| Escherichia coli K-12 | 大腸桿菌 | 511145 |
| Danio rerio | 斑馬魚 | 7955 |
| Gallus gallus | 雞 | 9031 |

完整清單：https://string-db.org/cgi/input?input_page_active_form=organisms

## STRING 識別符格式

STRING 使用帶有分類群前綴的 Ensembl 蛋白質 ID：
- 格式：`{taxonId}.{ensemblProteinId}`
- 範例：`9606.ENSP00000269305`（人類 TP53）

**ID 組成**：
- **分類群 ID**：NCBI 分類學識別符
- **蛋白質 ID**：通常是 Ensembl 蛋白質 ID（ENSP...）

## 交互作用信心分數

STRING 提供基於多個證據管道的組合信心分數（0-1000）：

### 證據管道

1. **鄰近性（nscore）**：基因融合和保守的基因組鄰近性
2. **融合（fscore）**：跨物種的基因融合事件
3. **系統發育圖譜（pscore）**：跨物種的共現
4. **共表現（ascore）**：RNA 表現相關性
5. **實驗（escore）**：生化/遺傳實驗
6. **資料庫（dscore）**：策展的途徑/複合體資料庫
7. **文字探勘（tscore）**：文獻共現

### 建議閾值

- **150**：低信心（探索性分析）
- **400**：中等信心（標準分析）
- **700**：高信心（保守分析）
- **900**：最高信心（非常嚴格）

## 輸出格式

### 可用格式

1. **TSV**：Tab 分隔值（預設，最適合資料處理）
2. **JSON**：JavaScript 物件標記法（結構化資料）
3. **XML**：可擴展標記語言
4. **PSI-MI**：蛋白質體學標準倡議格式
5. **PSI-MITAB**：Tab 分隔的 PSI-MI 格式
6. **PNG**：圖像格式（用於網路視覺化）
7. **SVG**：可縮放向量圖形（用於網路視覺化）

### 格式選擇

將 URL 中的 `/tsv/` 替換為所需格式：
- `/json/network` - JSON 格式
- `/xml/network` - XML 格式
- `/image/network` - PNG 圖像

## 錯誤處理

### HTTP 狀態碼

- **200 OK**：請求成功
- **400 Bad Request**：無效的參數或語法
- **404 Not Found**：找不到蛋白質/物種
- **500 Internal Server Error**：伺服器錯誤

### 常見錯誤

1. **「找不到蛋白質」**：無效的識別符或物種不匹配
2. **「需要物種」**：大型網路缺少 species 參數
3. **空結果**：沒有交互作用高於分數閾值
4. **逾時**：網路太大，減少蛋白質數量

## 進階功能

### 批量網路上傳

對於完整蛋白質體分析：
1. 導航至 https://string-db.org/
2. 選擇「上傳蛋白質體」選項
3. 上傳 FASTA 檔案
4. STRING 生成完整的交互作用網路並預測功能

### 值/等級富集 API

對於差異表現/蛋白質體學資料：

1. **取得 API 金鑰**：
```
/api/json/get_api_key
```

2. **提交資料**：Tab 分隔的蛋白質 ID 和值配對

3. **檢查狀態**：
```
/api/json/valuesranks_enrichment_status?job_id={id}
```

4. **檢索結果**：存取富集表格和圖形

**要求**：
- 完整的蛋白質集（無過濾）
- 每個蛋白質的數值
- 正確的物種識別符

### 網路自訂

**網路大小控制**：
- `add_nodes=N`：添加 N 個最連接的蛋白質
- `limit`：控制夥伴檢索

**信心過濾**：
- 根據分析目標調整 `required_score`
- 較高分數 = 較少偽陽性、較多偽陰性

**網路類型選擇**：
- `functional`：所有證據（建議用於途徑分析）
- `physical`：僅直接結合（建議用於結構研究）

## 與其他工具整合

### Python 函式庫

**requests**（建議）：
```python
import requests
url = "https://string-db.org/api/tsv/network"
params = {"identifiers": "TP53", "species": 9606}
response = requests.get(url, params=params)
```

**urllib**（標準函式庫）：
```python
import urllib.request
url = "https://string-db.org/api/tsv/network?identifiers=TP53&species=9606"
response = urllib.request.urlopen(url)
```

### R 整合

**STRINGdb Bioconductor 套件**：
```R
library(STRINGdb)
string_db <- STRINGdb$new(version="12", species=9606)
```

### Cytoscape

STRING 網路可以匯入 Cytoscape 進行視覺化和分析：
1. 使用 stringApp 外掛
2. 匯入 TSV 網路資料
3. 套用佈局和樣式

## 資料授權

STRING 資料在 **Creative Commons BY 4.0** 授權下免費提供：
- 可免費用於學術和商業用途
- 需要歸屬
- 允許修改
- 允許再發布

**引用**：Szklarczyk et al.（最新出版物）

## 速率限制和使用

- **速率限制**：無嚴格限制，但避免快速連續請求
- **建議**：呼叫間等待 1 秒
- **大型資料集**：使用 https://string-db.org/cgi/download 的批量下載
- **蛋白質體規模**：使用網頁上傳功能而非 API

## 相關資源

- **STRING 網站**：https://string-db.org
- **下載頁面**：https://string-db.org/cgi/download
- **幫助中心**：https://string-db.org/help/
- **API 文件**：https://string-db.org/help/api/
- **出版物**：https://string-db.org/cgi/about

## 疑難排解

**沒有返回結果**：
- 驗證 species 參數與識別符匹配
- 檢查識別符格式
- 降低信心閾值
- 先使用識別符對應

**逾時錯誤**：
- 減少輸入蛋白質數量
- 將大型查詢分批處理
- 對蛋白質體規模分析使用批量下載

**版本不一致**：
- 使用特定版本 URL
- 使用 `/version` 端點檢查 STRING 版本
- 如果使用舊 ID 則更新識別符
