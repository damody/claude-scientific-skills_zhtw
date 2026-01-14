# FDA 藥物資料庫

本參考文件涵蓋透過 openFDA 可存取的所有 FDA 藥物相關 API 端點。

## 概述

FDA 藥物資料庫提供藥品的相關資訊，包括不良事件、標籤、召回、核准和短缺。所有端點遵循 openFDA API 結構並回傳 JSON 格式的資料。

## 可用端點

### 1. 藥物不良事件

**端點**：`https://api.fda.gov/drug/event.json`

**目的**：存取向 FDA 提交的藥物副作用、產品使用錯誤、產品品質問題和治療失敗的報告。

**資料來源**：FDA 不良事件報告系統（FAERS）

**主要欄位**：
- `patient.drug.medicinalproduct` - 藥物名稱
- `patient.drug.drugindication` - 服藥原因
- `patient.reaction.reactionmeddrapt` - 不良反應描述
- `receivedate` - 報告接收日期
- `serious` - 事件是否嚴重（1 = 嚴重，2 = 不嚴重）
- `seriousnessdeath` - 事件是否導致死亡
- `primarysource.qualification` - 報告者資格（醫師、藥師等）

**常見使用案例**：
- 安全信號偵測
- 上市後監測
- 藥物交互作用分析
- 比較安全研究

**查詢範例**：
```python
# 查詢特定藥物的不良事件
import requests

api_key = "YOUR_API_KEY"
url = "https://api.fda.gov/drug/event.json"

# 搜尋阿斯匹靈相關的不良事件
params = {
    "api_key": api_key,
    "search": "patient.drug.medicinalproduct:aspirin",
    "limit": 10
}

response = requests.get(url, params=params)
data = response.json()
```

```python
# 計算某藥物最常見的反應
params = {
    "api_key": api_key,
    "search": "patient.drug.medicinalproduct:metformin",
    "count": "patient.reaction.reactionmeddrapt.exact"
}
```

### 2. 藥物產品標籤

**端點**：`https://api.fda.gov/drug/label.json`

**目的**：存取結構化產品資訊，包括 FDA 核准和上市藥品的處方資訊、警告、適應症和用法。

**資料來源**：結構化產品標籤（SPL）

**主要欄位**：
- `openfda.brand_name` - 藥物品牌名稱
- `openfda.generic_name` - 通用名稱
- `indications_and_usage` - 核准用途
- `warnings` - 重要安全警告
- `adverse_reactions` - 已知的不良反應
- `dosage_and_administration` - 用藥方式
- `description` - 化學和物理描述
- `pharmacodynamics` - 藥物作用機制
- `contraindications` - 禁忌症
- `drug_interactions` - 已知的藥物交互作用
- `active_ingredient` - 活性成分
- `inactive_ingredient` - 非活性成分

**常見使用案例**：
- 臨床決策支援
- 藥物資訊查詢
- 患者教育資料
- 處方集管理
- 藥物比較分析

**查詢範例**：
```python
# 取得品牌藥物的完整標籤
params = {
    "api_key": api_key,
    "search": "openfda.brand_name:Lipitor",
    "limit": 1
}

response = requests.get("https://api.fda.gov/drug/label.json", params=params)
label_data = response.json()

# 提取特定章節
if "results" in label_data:
    label = label_data["results"][0]
    indications = label.get("indications_and_usage", ["不可用"])[0]
    warnings = label.get("warnings", ["不可用"])[0]
```

```python
# 搜尋包含特定警告的標籤
params = {
    "api_key": api_key,
    "search": "warnings:*hypertension*",
    "limit": 10
}
```

### 3. 國家藥品代碼（NDC）目錄

**端點**：`https://api.fda.gov/drug/ndc.json`

**目的**：存取 NDC 目錄，包含由國家藥品代碼識別的藥品資訊。

**資料來源**：FDA NDC 目錄

**主要欄位**：
- `product_ndc` - 10 位數 NDC 產品識別碼
- `generic_name` - 藥物通用名稱
- `labeler_name` - 製造/經銷公司
- `brand_name` - 品牌名稱（如適用）
- `dosage_form` - 劑型（錠劑、膠囊、溶液等）
- `route` - 給藥途徑（口服、注射、外用等）
- `product_type` - 藥品類型
- `marketing_category` - 法規途徑（NDA、ANDA、OTC 等）
- `application_number` - FDA 申請編號
- `active_ingredients` - 含劑量的活性成分列表
- `packaging` - 包裝描述和 NDC 代碼
- `listing_expiration_date` - 列名到期日

**常見使用案例**：
- NDC 查詢和驗證
- 產品識別
- 供應鏈管理
- 處方處理
- 保險理賠處理

**查詢範例**：
```python
# 按 NDC 代碼查詢藥物
params = {
    "api_key": api_key,
    "search": "product_ndc:0069-2110",
    "limit": 1
}

response = requests.get("https://api.fda.gov/drug/ndc.json", params=params)
```

```python
# 查詢特定製造商的所有產品
params = {
    "api_key": api_key,
    "search": "labeler_name:Pfizer",
    "limit": 100
}
```

```python
# 取得某通用藥物的所有口服錠劑
params = {
    "api_key": api_key,
    "search": "generic_name:lisinopril+AND+dosage_form:TABLET",
    "limit": 50
}
```

### 4. 藥物召回執法報告

**端點**：`https://api.fda.gov/drug/enforcement.json`

**目的**：存取 FDA 發布的藥品召回執法報告。

**資料來源**：FDA 執法報告

**主要欄位**：
- `status` - 目前狀態（進行中、已完成、已終止）
- `recall_number` - 唯一召回識別碼
- `classification` - 第一類、第二類或第三類（嚴重程度）
- `product_description` - 被召回產品的描述
- `reason_for_recall` - 召回原因
- `product_quantity` - 召回的產品數量
- `code_info` - 批號、序號、NDC
- `distribution_pattern` - 地理分佈
- `recalling_firm` - 進行召回的公司
- `recall_initiation_date` - 召回開始時間
- `report_date` - FDA 收到通知時間
- `voluntary_mandated` - 召回類型

**分類等級**：
- **第一類**：可能導致嚴重健康問題或死亡的危險或有缺陷產品
- **第二類**：可能導致暫時健康問題或輕微嚴重威脅的產品
- **第三類**：不太可能導致不良健康反應但違反 FDA 標籤/製造法規的產品

**常見使用案例**：
- 品質保證監控
- 供應鏈風險管理
- 患者安全警報
- 法規合規追蹤

**查詢範例**：
```python
# 查詢所有第一類（最嚴重）藥物召回
params = {
    "api_key": api_key,
    "search": "classification:Class+I",
    "limit": 20,
    "sort": "report_date:desc"
}

response = requests.get("https://api.fda.gov/drug/enforcement.json", params=params)
```

```python
# 搜尋特定藥物的召回
params = {
    "api_key": api_key,
    "search": "product_description:*metformin*",
    "limit": 10
}
```

```python
# 查詢進行中的召回
params = {
    "api_key": api_key,
    "search": "status:Ongoing",
    "limit": 50
}
```

### 5. Drugs@FDA

**端點**：`https://api.fda.gov/drug/drugsfda.json`

**目的**：存取 Drugs@FDA 資料庫中 FDA 核准藥品的完整資訊，包括核准歷史和法規資訊。

**資料來源**：Drugs@FDA 資料庫（自 1939 年以來核准的大多數藥物）

**主要欄位**：
- `application_number` - NDA/ANDA/BLA 編號
- `sponsor_name` - 提交申請的公司
- `openfda.brand_name` - 品牌名稱
- `openfda.generic_name` - 通用名稱
- `products` - 此申請下核准的產品陣列
- `products.active_ingredients` - 含劑量的活性成分
- `products.dosage_form` - 劑型
- `products.route` - 給藥途徑
- `products.marketing_status` - 目前上市狀態
- `submissions` - 法規提交陣列
- `submissions.submission_type` - 提交類型
- `submissions.submission_status` - 狀態（已核准、待審等）
- `submissions.submission_status_date` - 狀態日期
- `submissions.review_priority` - 優先或標準審查

**常見使用案例**：
- 藥物核准研究
- 法規途徑分析
- 歷史核准追蹤
- 競爭情報
- 市場准入研究

**查詢範例**：
```python
# 查詢特定藥物的核准資訊
params = {
    "api_key": api_key,
    "search": "openfda.brand_name:Keytruda",
    "limit": 1
}

response = requests.get("https://api.fda.gov/drug/drugsfda.json", params=params)
```

```python
# 取得特定贊助商核准的所有藥物
params = {
    "api_key": api_key,
    "search": "sponsor_name:Moderna",
    "limit": 100
}
```

```python
# 查詢具有優先審查認定的藥物
params = {
    "api_key": api_key,
    "search": "submissions.review_priority:Priority",
    "limit": 50
}
```

### 6. 藥物短缺

**端點**：`https://api.fda.gov/drug/drugshortages.json`

**目的**：存取影響美國的目前和已解決藥物短缺資訊。

**資料來源**：FDA 藥物短缺資料庫

**主要欄位**：
- `product_name` - 短缺藥物名稱
- `status` - 目前狀態（目前短缺中、已解決、已停產）
- `reason` - 短缺原因
- `shortage_start_date` - 短缺開始時間
- `resolution_date` - 短缺解決時間（如適用）
- `discontinuation_date` - 如果產品停產
- `active_ingredient` - 活性成分
- `marketed_by` - 行銷該產品的公司
- `presentation` - 劑型和劑量

**常見使用案例**：
- 處方集管理
- 供應鏈規劃
- 患者照護連續性
- 治療替代方案識別
- 採購規劃

**查詢範例**：
```python
# 查詢目前的藥物短缺
params = {
    "api_key": api_key,
    "search": "status:Currently+in+Shortage",
    "limit": 100
}

response = requests.get("https://api.fda.gov/drug/drugshortages.json", params=params)
```

```python
# 搜尋特定藥物的短缺
params = {
    "api_key": api_key,
    "search": "product_name:*amoxicillin*",
    "limit": 10
}
```

```python
# 取得短缺歷史（目前和已解決）
params = {
    "api_key": api_key,
    "search": "active_ingredient:epinephrine",
    "limit": 50
}
```

## 整合技巧

### 錯誤處理

```python
import requests
import time

def query_fda_drug(endpoint, params, max_retries=3):
    """
    使用錯誤處理和重試邏輯查詢 FDA 藥物資料庫。

    參數：
        endpoint: 完整 URL 端點（例如：「https://api.fda.gov/drug/event.json」）
        params: 查詢參數字典
        max_retries: 最大重試次數

    回傳：
        回應 JSON 資料或 None（如果錯誤）
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                print(f"查詢未找到結果")
                return None
            elif response.status_code == 429:
                # 超過速率限制，等待並重試
                wait_time = 60 * (attempt + 1)
                print(f"超過速率限制。等待 {wait_time} 秒...")
                time.sleep(wait_time)
            else:
                print(f"發生 HTTP 錯誤：{e}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"請求錯誤：{e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return None
    return None
```

### 大型結果集的分頁

```python
def get_all_results(endpoint, search_query, api_key, max_results=1000):
    """
    使用分頁擷取查詢的所有結果。

    參數：
        endpoint: API 端點 URL
        search_query: 搜尋查詢字串
        api_key: FDA API 金鑰
        max_results: 要擷取的最大結果總數

    回傳：
        所有結果記錄的列表
    """
    all_results = []
    skip = 0
    limit = 100  # 每個請求的最大值

    while len(all_results) < max_results:
        params = {
            "api_key": api_key,
            "search": search_query,
            "limit": limit,
            "skip": skip
        }

        data = query_fda_drug(endpoint, params)
        if not data or "results" not in data:
            break

        results = data["results"]
        all_results.extend(results)

        # 檢查是否已擷取所有可用結果
        if len(results) < limit:
            break

        skip += limit
        time.sleep(0.25)  # 速率限制禮貌

    return all_results[:max_results]
```

## 最佳實踐

1. **始終使用 HTTPS** - HTTP 請求不被接受
2. **包含 API 金鑰** - 提供更高的速率限制（120,000/天 vs 1,000/天）
3. **聚合時使用精確匹配** - 在 count 查詢的欄位名稱中添加 `.exact` 後綴
4. **實作速率限制** - 保持在每分鐘 240 個請求內
5. **快取結果** - 避免對相同資料進行重複查詢
6. **優雅地處理錯誤** - 對暫時性失敗實作重試邏輯
7. **使用特定欄位搜尋** - 比全文搜尋更有效率
8. **驗證 NDC 代碼** - 使用標準 11 位數格式並移除連字號
9. **監控 API 狀態** - 檢查 openFDA 狀態頁面以了解中斷情況
10. **尊重資料限制** - OpenFDA 僅包含公開資料，非所有 FDA 資料

## 其他資源

- OpenFDA 藥物 API 文件：https://open.fda.gov/apis/drug/
- API 基礎：請參閱本參考目錄中的 `api_basics.md`
- Python 範例：請參閱 `scripts/fda_drug_query.py`
- 欄位參考指南：可在 https://open.fda.gov/apis/drug/[endpoint]/searchable-fields/ 取得
