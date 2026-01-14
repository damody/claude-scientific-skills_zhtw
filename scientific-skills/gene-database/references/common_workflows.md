# 常見基因資料庫工作流程

本文件提供使用 NCBI Gene 資料庫的常見工作流程和使用案例範例。

## 目錄

1. [疾病基因發現](#疾病基因發現)
2. [基因註解管線](#基因註解管線)
3. [跨物種基因比較](#跨物種基因比較)
4. [路徑分析](#路徑分析)
5. [變異分析](#變異分析)
6. [文獻探勘](#文獻探勘)

---

## 疾病基因發現

### 使用案例

識別與特定疾病或表型相關的基因。

### 工作流程

1. **按疾病名稱搜尋**

```bash
# 查找與阿茲海默症相關的基因
python scripts/query_gene.py --search "Alzheimer disease[disease]" --organism human --max-results 50
```

2. **按染色體位置過濾**

```bash
# 查找染色體 17 上與乳癌相關的基因
python scripts/query_gene.py --search "breast cancer[disease] AND 17[chromosome]" --organism human
```

3. **擷取詳細資訊**

```python
# Python 範例：取得疾病相關基因的詳細資料
import json
from scripts.query_gene import esearch, esummary

# 搜尋基因
query = "diabetes[disease] AND human[organism]"
gene_ids = esearch(query, retmax=100, api_key="YOUR_KEY")

# 取得摘要
summaries = esummary(gene_ids, api_key="YOUR_KEY")

# 提取相關資訊
for gene_id in gene_ids:
    if gene_id in summaries['result']:
        gene = summaries['result'][gene_id]
        print(f"{gene['name']}: {gene['description']}")
```

### 預期輸出

- 具有疾病關聯的基因列表
- 基因符號、描述和染色體位置
- 相關出版物和臨床註解

---

## 基因註解管線

### 使用案例

使用綜合元資料註解基因識別碼列表。

### 工作流程

1. **準備基因列表**

建立包含基因符號的檔案 `genes.txt`（每行一個）：
```
BRCA1
TP53
EGFR
KRAS
```

2. **批次查詢**

```bash
python scripts/batch_gene_lookup.py --file genes.txt --organism human --output annotations.json --api-key YOUR_KEY
```

3. **解析結果**

```python
import json

with open('annotations.json', 'r') as f:
    genes = json.load(f)

for gene in genes:
    if 'gene_id' in gene:
        print(f"符號：{gene['symbol']}")
        print(f"ID：{gene['gene_id']}")
        print(f"描述：{gene['description']}")
        print(f"位置：chr{gene['chromosome']}:{gene['map_location']}")
        print()
```

4. **使用序列資料擴充**

```bash
# 取得特定基因的詳細資料，包括序列
python scripts/fetch_gene_data.py --gene-id 672 --verbose > BRCA1_detailed.json
```

### 使用案例

- 為出版物建立基因註解表
- 在分析前驗證基因列表
- 建構基因參考資料庫
- 基因體管線的品質控制

---

## 跨物種基因比較

### 使用案例

查找同源基因或比較不同物種間的相同基因。

### 工作流程

1. **在多個生物體中搜尋基因**

```bash
# 在人類中查找 TP53
python scripts/fetch_gene_data.py --symbol TP53 --taxon human

# 在小鼠中查找 TP53
python scripts/fetch_gene_data.py --symbol TP53 --taxon mouse

# 在斑馬魚中查找 TP53
python scripts/fetch_gene_data.py --symbol TP53 --taxon zebrafish
```

2. **比較跨物種的基因 ID**

```python
# 比較跨物種的基因資訊
species = {
    'human': '9606',
    'mouse': '10090',
    'rat': '10116'
}

gene_symbol = 'TP53'

for organism, taxon_id in species.items():
    # 取得基因資料
    # ...（使用 fetch_gene_by_symbol）
    print(f"{organism}: {gene_data}")
```

3. **使用 ELink 查找同源基因**

```bash
# 取得基因的 HomoloGene 連結
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=gene&db=homologene&id=7157&retmode=json"
```

### 應用

- 演化研究
- 模式生物研究
- 比較基因體學
- 跨物種實驗設計

---

## 路徑分析

### 使用案例

識別參與特定生物路徑或過程的基因。

### 工作流程

1. **按基因本體（GO）術語搜尋**

```bash
# 查找參與細胞凋亡的基因
python scripts/query_gene.py --search "GO:0006915[biological process]" --organism human --max-results 100
```

2. **按路徑名稱搜尋**

```bash
# 查找胰島素訊號路徑中的基因
python scripts/query_gene.py --search "insulin signaling pathway[pathway]" --organism human
```

3. **取得路徑相關基因**

```python
# 範例：取得特定路徑中的所有基因
import urllib.request
import json

# 搜尋路徑基因
query = "MAPK signaling pathway[pathway] AND human[organism]"
url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=gene&term={query}&retmode=json&retmax=200"

with urllib.request.urlopen(url) as response:
    data = json.loads(response.read().decode())
    gene_ids = data['esearchresult']['idlist']

print(f"在 MAPK 訊號路徑中找到 {len(gene_ids)} 個基因")
```

4. **批次擷取基因詳細資料**

```bash
# 取得所有路徑基因的詳細資料
python scripts/batch_gene_lookup.py --ids 5594,5595,5603,5604 --output mapk_genes.json
```

### 應用

- 路徑富集分析
- 基因集分析
- 系統生物學研究
- 藥物標靶識別

---

## 變異分析

### 使用案例

查找具有臨床相關變異或疾病相關突變的基因。

### 工作流程

1. **搜尋具有臨床變異的基因**

```bash
# 查找具有致病性變異的基因
python scripts/query_gene.py --search "pathogenic[clinical significance]" --organism human --max-results 50
```

2. **連結到 ClinVar 資料庫**

```bash
# 取得基因的 ClinVar 記錄
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=gene&db=clinvar&id=672&retmode=json"
```

3. **搜尋藥物基因體學基因**

```bash
# 查找與藥物反應相關的基因
python scripts/query_gene.py --search "pharmacogenomic[property]" --organism human
```

4. **取得變異摘要資料**

```python
# 範例：取得具有已知變異的基因
from scripts.query_gene import esearch, efetch

# 搜尋具有變異的基因
gene_ids = esearch("has variants[filter] AND human[organism]", retmax=100)

# 取得詳細記錄
for gene_id in gene_ids[:10]:  # 前 10 個
    data = efetch([gene_id], retmode='xml')
    # 解析 XML 以獲取變異資訊
    print(f"基因 {gene_id} 變異資料...")
```

### 應用

- 臨床遺傳學
- 精準醫療
- 藥物基因體學
- 遺傳諮詢

---

## 文獻探勘

### 使用案例

查找在近期出版物中提及的基因或將基因與文獻連結。

### 工作流程

1. **搜尋在特定出版物中提及的基因**

```bash
# 查找在關於 CRISPR 的論文中提及的基因
python scripts/query_gene.py --search "CRISPR[text word]" --organism human --max-results 100
```

2. **取得基因的 PubMed 文章**

```bash
# 取得 BRCA1 的所有出版物
curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=gene&db=pubmed&id=672&retmode=json"
```

3. **按作者或期刊搜尋**

```bash
# 查找特定研究團隊研究的基因
python scripts/query_gene.py --search "Smith J[author] AND 2024[pdat]" --organism human
```

4. **提取基因-出版物關係**

```python
# 範例：建構基因-出版物網路
from scripts.query_gene import esearch, esummary
import urllib.request
import json

# 取得基因
gene_id = '672'

# 取得基因的出版物
url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=gene&db=pubmed&id={gene_id}&retmode=json"

with urllib.request.urlopen(url) as response:
    data = json.loads(response.read().decode())

# 提取 PMID
pmids = []
for linkset in data.get('linksets', []):
    for linksetdb in linkset.get('linksetdbs', []):
        pmids.extend(linksetdb.get('links', []))

print(f"基因 {gene_id} 有 {len(pmids)} 篇出版物")
```

### 應用

- 文獻回顧
- 計畫撰寫
- 知識庫建構
- 基因體學研究趨勢分析

---

## 進階模式

### 組合多個搜尋

```python
# 範例：查找符合多個標準交集的基因
def find_genes_multi_criteria(organism='human'):
    # 標準 1：疾病關聯
    disease_genes = set(esearch("diabetes[disease] AND human[organism]"))

    # 標準 2：染色體位置
    chr_genes = set(esearch("11[chromosome] AND human[organism]"))

    # 標準 3：基因類型
    coding_genes = set(esearch("protein coding[gene type] AND human[organism]"))

    # 交集
    candidates = disease_genes & chr_genes & coding_genes

    return list(candidates)
```

### 速率限制的批次處理

```python
import time

def process_genes_with_rate_limit(gene_ids, batch_size=200, delay=0.1):
    results = []

    for i in range(0, len(gene_ids), batch_size):
        batch = gene_ids[i:i + batch_size]

        # 處理批次
        batch_results = esummary(batch)
        results.append(batch_results)

        # 速率限制
        time.sleep(delay)

    return results
```

### 錯誤處理和重試

```python
import time

def robust_gene_fetch(gene_id, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = fetch_gene_by_id(gene_id)
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # 指數退避
                time.sleep(wait)
            else:
                print(f"無法取得基因 {gene_id}：{e}")
                return None
```

---

## 技巧和最佳實踐

1. **從具體開始，然後擴展**：以精確查詢開始，必要時再擴展
2. **使用生物體過濾器**：始終為基因符號搜尋指定生物體
3. **驗證結果**：檢查基因 ID 和符號的準確性
4. **快取常用資料**：在本地儲存常見查詢
5. **監控速率限制**：使用 API 金鑰並實作延遲
6. **組合 API**：使用 E-utilities 進行搜尋，使用 Datasets API 取得詳細資料
7. **處理歧義**：基因符號在不同物種中可能指向不同的基因
8. **檢查資料時效性**：基因註解會定期更新
9. **使用批次操作**：盡可能一起處理多個基因
10. **記錄您的查詢**：保留搜尋詞彙和參數的記錄
