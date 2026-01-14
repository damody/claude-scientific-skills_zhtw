# GWAS Catalog API 參考

GWAS Catalog REST API 的全面參考，包括端點規格、查詢參數、回應格式和進階使用模式。

## 目錄

- [API 概述](#api-概述)
- [認證與速率限制](#認證與速率限制)
- [GWAS Catalog REST API](#gwas-catalog-rest-api)
- [摘要統計 API](#摘要統計-api)
- [回應格式](#回應格式)
- [錯誤處理](#錯誤處理)
- [進階查詢模式](#進階查詢模式)
- [整合範例](#整合範例)

## API 概述

GWAS Catalog 提供兩個互補的 REST API：

1. **GWAS Catalog REST API**：存取經策展的 SNP-性狀關聯、研究和中繼資料
2. **摘要統計 API**：存取完整的 GWAS 摘要統計（所有測試的變異）

兩個 API 都使用 RESTful 設計原則，回應採用 HAL（超文本應用語言）格式的 JSON，其中包含用於資源導航的 `_links`。

### 基礎 URL

```
GWAS Catalog API:         https://www.ebi.ac.uk/gwas/rest/api
Summary Statistics API:   https://www.ebi.ac.uk/gwas/summary-statistics/api
```

### 版本資訊

GWAS Catalog REST API v2.0 於 2024 年發布，有重大改進：
- 新端點（出版物、基因、基因體脈絡、祖源）
- 增強的資料公開（世代、背景性狀、授權）
- 改進的查詢功能
- 更好的效能和文件

之前的 API 版本將持續可用至 2026 年 5 月以確保向後相容。

## 認證與速率限制

### 認證

**不需要認證** - 兩個 API 都是開放存取的，不需要 API 金鑰或註冊。

### 速率限制

雖然沒有明確記錄的速率限制，但請遵循最佳實踐：
- 在連續請求之間實作延遲（例如 0.1-0.5 秒）
- 對大型結果集使用分頁
- 在本地快取回應
- 使用批量下載（FTP）進行全基因體資料
- 避免快速連續請求轟炸 API

**帶速率限制的範例：**
```python
import requests
from time import sleep

def query_with_rate_limit(url, delay=0.1):
    response = requests.get(url)
    sleep(delay)
    return response.json()
```

## GWAS Catalog REST API

主要 API 提供對經策展的 GWAS 關聯、研究、變異和性狀的存取。

### 核心端點

#### 1. 研究

**取得所有研究：**
```
GET /studies
```

**取得特定研究：**
```
GET /studies/{accessionId}
```

**搜尋研究：**
```
GET /studies/search/findByPublicationIdPubmedId?pubmedId={pmid}
GET /studies/search/findByDiseaseTrait?diseaseTrait={trait}
```

**查詢參數：**
- `page`：頁碼（從 0 開始）
- `size`：每頁結果數（預設：20）
- `sort`：排序欄位（例如 `publicationDate,desc`）

**範例：**
```python
import requests

# 取得特定研究
url = "https://www.ebi.ac.uk/gwas/rest/api/studies/GCST001795"
response = requests.get(url, headers={"Content-Type": "application/json"})
study = response.json()

print(f"Title: {study.get('title')}")
print(f"PMID: {study.get('publicationInfo', {}).get('pubmedId')}")
print(f"Sample size: {study.get('initialSampleSize')}")
```

**回應欄位：**
- `accessionId`：研究識別碼（GCST ID）
- `title`：研究標題
- `publicationInfo`：出版詳情包括 PMID
- `initialSampleSize`：發現世代描述
- `replicationSampleSize`：複製世代描述
- `ancestries`：族群祖源資訊
- `genotypingTechnologies`：基因晶片或定序平台
- `_links`：相關資源的連結

#### 2. 關聯

**取得所有關聯：**
```
GET /associations
```

**取得特定關聯：**
```
GET /associations/{associationId}
```

**取得性狀的關聯：**
```
GET /efoTraits/{efoId}/associations
```

**取得變異的關聯：**
```
GET /singleNucleotidePolymorphisms/{rsId}/associations
```

**查詢參數：**
- `projection`：回應投影（例如 `associationBySnp`）
- `page`、`size`、`sort`：分頁控制

**範例：**
```python
import requests

# 尋找第二型糖尿病的所有關聯
trait_id = "EFO_0001360"
url = f"https://www.ebi.ac.uk/gwas/rest/api/efoTraits/{trait_id}/associations"
params = {"size": 100, "page": 0}
response = requests.get(url, params=params, headers={"Content-Type": "application/json"})
data = response.json()

associations = data.get('_embedded', {}).get('associations', [])
print(f"Found {len(associations)} associations")
```

**回應欄位：**
- `rsId`：變異識別碼
- `strongestAllele`：風險或效應等位基因
- `pvalue`：關聯 p 值
- `pvalueText`：報告的 p 值（可能包含不等式）
- `pvalueMantissa`：p 值的尾數
- `pvalueExponent`：p 值的指數
- `orPerCopyNum`：每等位基因複製的勝算比
- `betaNum`：效應量（數量性狀）
- `betaUnit`：測量單位
- `range`：信賴區間
- `standardError`：標準誤差
- `efoTrait`：性狀名稱
- `mappedLabel`：EFO 標準化術語
- `studyId`：相關研究登錄號

#### 3. 變異（單核苷酸多態性）

**取得變異詳情：**
```
GET /singleNucleotidePolymorphisms/{rsId}
```

**搜尋變異：**
```
GET /singleNucleotidePolymorphisms/search/findByRsId?rsId={rsId}
GET /singleNucleotidePolymorphisms/search/findByChromBpLocationRange?chrom={chr}&bpStart={start}&bpEnd={end}
GET /singleNucleotidePolymorphisms/search/findByGene?geneName={gene}
```

**範例：**
```python
import requests

# 取得變異資訊
rs_id = "rs7903146"
url = f"https://www.ebi.ac.uk/gwas/rest/api/singleNucleotidePolymorphisms/{rs_id}"
response = requests.get(url, headers={"Content-Type": "application/json"})
variant = response.json()

print(f"rsID: {variant.get('rsId')}")
print(f"Location: chr{variant.get('locations', [{}])[0].get('chromosomeName')}:{variant.get('locations', [{}])[0].get('chromosomePosition')}")
```

**回應欄位：**
- `rsId`：rs 編號
- `merged`：指示變異是否已與另一個合併
- `functionalClass`：變異後果
- `locations`：基因體位置陣列
  - `chromosomeName`：染色體編號
  - `chromosomePosition`：鹼基對位置
  - `region`：基因體區域資訊
- `genomicContexts`：鄰近基因
- `lastUpdateDate`：最後修改日期

#### 4. 性狀（EFO 術語）

**取得性狀資訊：**
```
GET /efoTraits/{efoId}
```

**搜尋性狀：**
```
GET /efoTraits/search/findByEfoUri?uri={efoUri}
GET /efoTraits/search/findByTraitIgnoreCase?trait={traitName}
```

**範例：**
```python
import requests

# 取得性狀詳情
trait_id = "EFO_0001360"
url = f"https://www.ebi.ac.uk/gwas/rest/api/efoTraits/{trait_id}"
response = requests.get(url, headers={"Content-Type": "application/json"})
trait = response.json()

print(f"Trait: {trait.get('trait')}")
print(f"EFO URI: {trait.get('uri')}")
```

#### 5. 出版物

**取得出版物資訊：**
```
GET /publications
GET /publications/{publicationId}
GET /publications/search/findByPubmedId?pubmedId={pmid}
```

#### 6. 基因

**取得基因資訊：**
```
GET /genes
GET /genes/{geneId}
GET /genes/search/findByGeneName?geneName={symbol}
```

### 分頁與導航

所有列表端點都支援分頁：

```python
import requests

def get_all_associations(trait_id):
    """擷取性狀的所有關聯並進行分頁"""
    base_url = "https://www.ebi.ac.uk/gwas/rest/api"
    url = f"{base_url}/efoTraits/{trait_id}/associations"
    all_associations = []
    page = 0

    while True:
        params = {"page": page, "size": 100}
        response = requests.get(url, params=params, headers={"Content-Type": "application/json"})

        if response.status_code != 200:
            break

        data = response.json()
        associations = data.get('_embedded', {}).get('associations', [])

        if not associations:
            break

        all_associations.extend(associations)
        page += 1

    return all_associations
```

### HAL 連結

回應包含用於資源導航的 `_links`：

```python
import requests

# 取得研究並追蹤連結到關聯
response = requests.get("https://www.ebi.ac.uk/gwas/rest/api/studies/GCST001795")
study = response.json()

# 追蹤到關聯的連結
associations_url = study['_links']['associations']['href']
associations_response = requests.get(associations_url)
associations = associations_response.json()
```

## 摘要統計 API

存取已存放完整資料的研究的完整 GWAS 摘要統計。

### 基礎 URL
```
https://www.ebi.ac.uk/gwas/summary-statistics/api
```

### 核心端點

#### 1. 研究

**取得所有具有摘要統計的研究：**
```
GET /studies
```

**取得特定研究：**
```
GET /studies/{gcstId}
```

#### 2. 性狀

**取得性狀資訊：**
```
GET /traits/{efoId}
```

**取得性狀的關聯：**
```
GET /traits/{efoId}/associations
```

**查詢參數：**
- `p_lower`：p 值下限閾值
- `p_upper`：p 值上限閾值
- `size`：結果數量
- `page`：頁碼

**範例：**
```python
import requests

# 尋找性狀的高度顯著關聯
trait_id = "EFO_0001360"
base_url = "https://www.ebi.ac.uk/gwas/summary-statistics/api"
url = f"{base_url}/traits/{trait_id}/associations"
params = {
    "p_upper": "0.000000001",  # p < 1e-9
    "size": 100
}
response = requests.get(url, params=params)
results = response.json()
```

#### 3. 染色體

**依染色體取得關聯：**
```
GET /chromosomes/{chromosome}/associations
```

**依基因體區域查詢：**
```
GET /chromosomes/{chromosome}/associations?start={start}&end={end}
```

**範例：**
```python
import requests

# 查詢特定區域中的變異
chromosome = "10"
start_pos = 114000000
end_pos = 115000000

base_url = "https://www.ebi.ac.uk/gwas/summary-statistics/api"
url = f"{base_url}/chromosomes/{chromosome}/associations"
params = {
    "start": start_pos,
    "end": end_pos,
    "size": 1000
}
response = requests.get(url, params=params)
variants = response.json()
```

#### 4. 變異

**跨研究取得特定變異：**
```
GET /variants/{variantId}
```

**依變異 ID 搜尋：**
```
GET /variants/{variantId}/associations
```

### 回應欄位

**關聯欄位：**
- `variant_id`：變異識別碼
- `chromosome`：染色體編號
- `base_pair_location`：位置（bp）
- `effect_allele`：效應等位基因
- `other_allele`：參考等位基因
- `effect_allele_frequency`：等位基因頻率
- `beta`：效應量
- `standard_error`：標準誤差
- `p_value`：p 值
- `ci_lower`：信賴區間下限
- `ci_upper`：信賴區間上限
- `odds_ratio`：勝算比（病例對照研究）
- `study_accession`：GCST ID

## 回應格式

### 內容類型

所有 API 請求應包含標頭：
```
Content-Type: application/json
```

### HAL 格式

回應遵循 HAL（超文本應用語言）規範：

```json
{
  "_embedded": {
    "associations": [
      {
        "rsId": "rs7903146",
        "pvalue": 1.2e-30,
        "efoTrait": "type 2 diabetes",
        "_links": {
          "self": {
            "href": "https://www.ebi.ac.uk/gwas/rest/api/associations/12345"
          }
        }
      }
    ]
  },
  "_links": {
    "self": {
      "href": "https://www.ebi.ac.uk/gwas/rest/api/efoTraits/EFO_0001360/associations?page=0"
    },
    "next": {
      "href": "https://www.ebi.ac.uk/gwas/rest/api/efoTraits/EFO_0001360/associations?page=1"
    }
  },
  "page": {
    "size": 20,
    "totalElements": 1523,
    "totalPages": 77,
    "number": 0
  }
}
```

### 頁面中繼資料

分頁回應包含頁面資訊：
- `size`：每頁項目數
- `totalElements`：總結果數
- `totalPages`：總頁數
- `number`：當前頁碼（從 0 開始）

## 錯誤處理

### HTTP 狀態碼

- `200 OK`：請求成功
- `400 Bad Request`：參數無效
- `404 Not Found`：找不到資源
- `500 Internal Server Error`：伺服器錯誤

### 錯誤回應格式

```json
{
  "timestamp": "2025-10-19T12:00:00.000+00:00",
  "status": 404,
  "error": "Not Found",
  "message": "No association found with id: 12345",
  "path": "/gwas/rest/api/associations/12345"
}
```

### 錯誤處理範例

```python
import requests

def safe_api_request(url, params=None):
    """帶錯誤處理的 API 請求"""
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(f"Response: {response.text}")
        return None
    except requests.exceptions.ConnectionError:
        print("Connection error - check network")
        return None
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return None
```

## 進階查詢模式

### 1. 變異和性狀的交叉參照

```python
import requests

def get_variant_pleiotropy(rs_id):
    """取得與變異相關的所有性狀"""
    base_url = "https://www.ebi.ac.uk/gwas/rest/api"
    url = f"{base_url}/singleNucleotidePolymorphisms/{rs_id}/associations"
    params = {"projection": "associationBySnp"}

    response = requests.get(url, params=params, headers={"Content-Type": "application/json"})
    data = response.json()

    traits = {}
    for assoc in data.get('_embedded', {}).get('associations', []):
        trait = assoc.get('efoTrait')
        pvalue = assoc.get('pvalue')
        if trait:
            if trait not in traits or float(pvalue) < float(traits[trait]):
                traits[trait] = pvalue

    return traits

# 使用範例
pleiotropy = get_variant_pleiotropy('rs7903146')
for trait, pval in sorted(pleiotropy.items(), key=lambda x: float(x[1])):
    print(f"{trait}: p={pval}")
```

### 2. 依 p 值閾值篩選

```python
import requests

def get_significant_associations(trait_id, p_threshold=5e-8):
    """取得全基因體顯著關聯"""
    base_url = "https://www.ebi.ac.uk/gwas/rest/api"
    url = f"{base_url}/efoTraits/{trait_id}/associations"

    results = []
    page = 0

    while True:
        params = {"page": page, "size": 100}
        response = requests.get(url, params=params, headers={"Content-Type": "application/json"})

        if response.status_code != 200:
            break

        data = response.json()
        associations = data.get('_embedded', {}).get('associations', [])

        if not associations:
            break

        for assoc in associations:
            pvalue = assoc.get('pvalue')
            if pvalue and float(pvalue) <= p_threshold:
                results.append(assoc)

        page += 1

    return results
```

### 3. 結合主要和摘要統計 API

```python
import requests

def get_complete_variant_data(rs_id):
    """從兩個 API 取得變異資料"""
    main_url = f"https://www.ebi.ac.uk/gwas/rest/api/singleNucleotidePolymorphisms/{rs_id}"

    # 取得基本變異資訊
    response = requests.get(main_url, headers={"Content-Type": "application/json"})
    variant_info = response.json()

    # 取得關聯
    assoc_url = f"{main_url}/associations"
    response = requests.get(assoc_url, headers={"Content-Type": "application/json"})
    associations = response.json()

    # 也可以查詢摘要統計 API 以取得此變異
    # 跨所有具有摘要資料的研究

    return {
        "variant": variant_info,
        "associations": associations
    }
```

### 4. 基因體區域查詢

```python
import requests

def query_region(chromosome, start, end, p_threshold=None):
    """查詢基因體區域中的變異"""
    # 從主要 API
    base_url = "https://www.ebi.ac.uk/gwas/rest/api"
    url = f"{base_url}/singleNucleotidePolymorphisms/search/findByChromBpLocationRange"
    params = {
        "chrom": chromosome,
        "bpStart": start,
        "bpEnd": end,
        "size": 1000
    }

    response = requests.get(url, params=params, headers={"Content-Type": "application/json"})
    variants = response.json()

    # 也可以查詢摘要統計 API
    sumstats_url = f"https://www.ebi.ac.uk/gwas/summary-statistics/api/chromosomes/{chromosome}/associations"
    sumstats_params = {"start": start, "end": end, "size": 1000}
    if p_threshold:
        sumstats_params["p_upper"] = str(p_threshold)

    sumstats_response = requests.get(sumstats_url, params=sumstats_params)
    sumstats = sumstats_response.json()

    return {
        "catalog_variants": variants,
        "summary_stats": sumstats
    }
```

## 整合範例

### 完整工作流程：疾病遺傳架構

```python
import requests
import pandas as pd
from time import sleep

class GWASCatalogQuery:
    def __init__(self):
        self.base_url = "https://www.ebi.ac.uk/gwas/rest/api"
        self.headers = {"Content-Type": "application/json"}

    def get_trait_associations(self, trait_id, p_threshold=5e-8):
        """取得性狀的所有關聯"""
        url = f"{self.base_url}/efoTraits/{trait_id}/associations"
        results = []
        page = 0

        while True:
            params = {"page": page, "size": 100}
            response = requests.get(url, params=params, headers=self.headers)

            if response.status_code != 200:
                break

            data = response.json()
            associations = data.get('_embedded', {}).get('associations', [])

            if not associations:
                break

            for assoc in associations:
                pvalue = assoc.get('pvalue')
                if pvalue and float(pvalue) <= p_threshold:
                    results.append({
                        'rs_id': assoc.get('rsId'),
                        'pvalue': float(pvalue),
                        'risk_allele': assoc.get('strongestAllele'),
                        'or_beta': assoc.get('orPerCopyNum') or assoc.get('betaNum'),
                        'study': assoc.get('studyId'),
                        'pubmed_id': assoc.get('pubmedId')
                    })

            page += 1
            sleep(0.1)

        return pd.DataFrame(results)

    def get_variant_details(self, rs_id):
        """取得詳細變異資訊"""
        url = f"{self.base_url}/singleNucleotidePolymorphisms/{rs_id}"
        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        return None

    def get_gene_associations(self, gene_name):
        """取得與基因相關的變異"""
        url = f"{self.base_url}/singleNucleotidePolymorphisms/search/findByGene"
        params = {"geneName": gene_name}
        response = requests.get(url, params=params, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        return None

# 使用範例
gwas = GWASCatalogQuery()

# 查詢第二型糖尿病關聯
df = gwas.get_trait_associations('EFO_0001360')
print(f"Found {len(df)} genome-wide significant associations")
print(f"Unique variants: {df['rs_id'].nunique()}")

# 取得頂端變異
top_variants = df.nsmallest(10, 'pvalue')
print("\nTop 10 variants:")
print(top_variants[['rs_id', 'pvalue', 'risk_allele']])

# 取得頂端變異的詳情
if len(top_variants) > 0:
    top_rs = top_variants.iloc[0]['rs_id']
    variant_info = gwas.get_variant_details(top_rs)
    if variant_info:
        loc = variant_info.get('locations', [{}])[0]
        print(f"\n{top_rs} location: chr{loc.get('chromosomeName')}:{loc.get('chromosomePosition')}")
```

### FTP 下載整合

```python
import requests
from pathlib import Path

def download_summary_statistics(gcst_id, output_dir="."):
    """從 FTP 下載摘要統計"""
    # FTP URL 模式
    ftp_base = "http://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics"

    # 先嘗試協調檔案
    harmonised_url = f"{ftp_base}/{gcst_id}/harmonised/{gcst_id}-harmonised.tsv.gz"

    output_path = Path(output_dir) / f"{gcst_id}.tsv.gz"

    try:
        response = requests.get(harmonised_url, stream=True)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded {gcst_id} to {output_path}")
        return output_path

    except requests.exceptions.HTTPError:
        print(f"Harmonised file not found for {gcst_id}")
        return None

# 使用範例
download_summary_statistics("GCST001234", output_dir="./sumstats")
```

## 額外資源

- **互動式 API 文件**：https://www.ebi.ac.uk/gwas/rest/docs/api
- **摘要統計 API 文件**：https://www.ebi.ac.uk/gwas/summary-statistics/docs/
- **工作坊材料**：https://github.com/EBISPOT/GWAS_Catalog-workshop
- **API v2 部落格文章**：https://ebispot.github.io/gwas-blog/rest-api-v2-release/
- **R 套件（gwasrapidd）**：https://cran.r-project.org/package=gwasrapidd
