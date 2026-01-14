# FDA 醫療器材資料庫

本參考文件涵蓋透過 openFDA 可存取的所有 FDA 醫療器材相關 API 端點。

## 概述

FDA 器材資料庫提供醫療器材的相關資訊，包括不良事件、召回、核准、註冊和分類資料。醫療器材範圍從簡單的物品如壓舌板到複雜的儀器如心律調節器和手術機器人。

## 器材分類系統

醫療器材根據風險分為三個類別：

- **第一類**：低風險（例如：繃帶、檢查手套）
- **第二類**：中度風險（例如：電動輪椅、輸液泵）
- **第三類**：高風險（例如：心臟瓣膜、植入式心律調節器）

## 可用端點

### 1. 器材不良事件

**端點**：`https://api.fda.gov/device/event.json`

**目的**：存取記錄醫療器材使用導致的嚴重傷害、死亡、故障和其他不良影響的報告。

**資料來源**：製造商和使用者設施器材經驗（MAUDE）資料庫

**主要欄位**：
- `device.brand_name` - 器材品牌名稱
- `device.generic_name` - 器材通用名稱
- `device.manufacturer_d_name` - 製造商名稱
- `device.device_class` - 器材類別（1、2 或 3）
- `event_type` - 事件類型（死亡、傷害、故障、其他）
- `date_received` - FDA 收到報告的日期
- `mdr_report_key` - 唯一報告識別碼
- `adverse_event_flag` - 是否報告為不良事件
- `product_problem_flag` - 是否報告產品問題
- `patient.patient_problems` - 患者問題/併發症
- `device.openfda.device_name` - 官方器材名稱
- `device.openfda.medical_specialty_description` - 醫學專科
- `remedial_action` - 採取的行動（召回、維修、更換等）

**常見使用案例**：
- 上市後監測
- 安全信號偵測
- 器材比較研究
- 風險分析
- 品質改進

**查詢範例**：
```python
import requests

api_key = "YOUR_API_KEY"
url = "https://api.fda.gov/device/event.json"

# 查詢特定器材的不良事件
params = {
    "api_key": api_key,
    "search": "device.brand_name:pacemaker",
    "limit": 10
}

response = requests.get(url, params=params)
data = response.json()
```

```python
# 按類型計數事件
params = {
    "api_key": api_key,
    "search": "device.generic_name:insulin+pump",
    "count": "event_type"
}
```

```python
# 查詢第三類器材的死亡事件
params = {
    "api_key": api_key,
    "search": "event_type:Death+AND+device.device_class:3",
    "limit": 50,
    "sort": "date_received:desc"
}
```

### 2. 器材 510(k) 許可

**端點**：`https://api.fda.gov/device/510k.json`

**目的**：存取 510(k) 上市前通知資料，證明器材與合法上市的前置器材等效。

**資料來源**：510(k) 上市前通知

**主要欄位**：
- `k_number` - 510(k) 編號（唯一識別碼）
- `applicant` - 提交 510(k) 的公司
- `device_name` - 器材名稱
- `device_class` - 器材分類（1、2 或 3）
- `decision_date` - FDA 決定日期
- `decision_description` - 實質等效（SE）或非 SE
- `product_code` - FDA 產品代碼
- `statement_or_summary` - 提供的摘要類型
- `clearance_type` - 傳統、特殊、簡化等
- `expedited_review_flag` - 是否加速審查
- `advisory_committee` - 諮詢委員會名稱
- `openfda.device_name` - 官方器材名稱
- `openfda.device_class` - 器材類別描述
- `openfda.medical_specialty_description` - 醫學專科
- `openfda.regulation_number` - CFR 法規編號

**常見使用案例**：
- 法規途徑研究
- 前置器材識別
- 市場進入分析
- 競爭情報
- 器材開發規劃

**查詢範例**：
```python
# 按公司查詢 510(k) 許可
params = {
    "api_key": api_key,
    "search": "applicant:Medtronic",
    "limit": 50,
    "sort": "decision_date:desc"
}

response = requests.get("https://api.fda.gov/device/510k.json", params=params)
```

```python
# 搜尋特定器材類型的許可
params = {
    "api_key": api_key,
    "search": "device_name:*surgical+robot*",
    "limit": 10
}
```

```python
# 取得近年所有第三類 510(k) 許可
params = {
    "api_key": api_key,
    "search": "device_class:3+AND+decision_date:[20240101+TO+20241231]",
    "limit": 100
}
```

### 3. 器材分類

**端點**：`https://api.fda.gov/device/classification.json`

**目的**：存取器材分類資料庫，包含醫療器材名稱、產品代碼、醫學專科面板和分類資訊。

**資料來源**：FDA 器材分類資料庫

**主要欄位**：
- `product_code` - 三字母 FDA 產品代碼
- `device_name` - 官方器材名稱
- `device_class` - 類別（1、2 或 3）
- `medical_specialty` - 醫學專科（例如：放射科、心血管科）
- `medical_specialty_description` - 完整專科描述
- `regulation_number` - CFR 法規編號（例如：21 CFR 870.2300）
- `review_panel` - FDA 審查面板
- `definition` - 官方器材定義
- `physical_state` - 固體、液體、氣體
- `technical_method` - 操作方法
- `target_area` - 目標身體區域/系統
- `gmp_exempt_flag` - 是否豁免良好製造規範
- `implant_flag` - 是否為植入式器材
- `life_sustain_support_flag` - 是否為生命維持/支援器材

**常見使用案例**：
- 器材識別
- 法規要求判定
- 產品代碼查詢
- 分類研究
- 器材分類

**查詢範例**：
```python
# 按產品代碼查詢器材
params = {
    "api_key": api_key,
    "search": "product_code:LWL",
    "limit": 1
}

response = requests.get("https://api.fda.gov/device/classification.json", params=params)
```

```python
# 查詢所有心血管器材
params = {
    "api_key": api_key,
    "search": "medical_specialty:CV",
    "limit": 100
}
```

```python
# 取得所有可植入的第三類器材
params = {
    "api_key": api_key,
    "search": "device_class:3+AND+implant_flag:Y",
    "limit": 50
}
```

### 4. 器材召回執法報告

**端點**：`https://api.fda.gov/device/enforcement.json`

**目的**：存取醫療器材產品召回執法報告。

**資料來源**：FDA 執法報告

**主要欄位**：
- `status` - 目前狀態（進行中、已完成、已終止）
- `recall_number` - 唯一召回識別碼
- `classification` - 第一類、第二類或第三類
- `product_description` - 被召回器材的描述
- `reason_for_recall` - 召回原因
- `product_quantity` - 召回的產品數量
- `code_info` - 批號、序號、型號
- `distribution_pattern` - 地理分佈
- `recalling_firm` - 進行召回的公司
- `recall_initiation_date` - 召回開始時間
- `report_date` - FDA 收到通知時間
- `product_res_number` - 產品問題編號

**常見使用案例**：
- 品質監控
- 供應鏈風險管理
- 患者安全追蹤
- 法規合規
- 器材監測

**查詢範例**：
```python
# 查詢所有第一類器材召回（最嚴重）
params = {
    "api_key": api_key,
    "search": "classification:Class+I",
    "limit": 20,
    "sort": "report_date:desc"
}

response = requests.get("https://api.fda.gov/device/enforcement.json", params=params)
```

```python
# 按製造商搜尋召回
params = {
    "api_key": api_key,
    "search": "recalling_firm:*Philips*",
    "limit": 50
}
```

### 5. 器材召回

**端點**：`https://api.fda.gov/device/recall.json`

**目的**：存取關於違反 FDA 法規或造成健康風險的器材召回資訊。

**資料來源**：FDA 召回資料庫

**主要欄位**：
- `res_event_number` - 召回事件編號
- `product_code` - FDA 產品代碼
- `openfda.device_name` - 器材名稱
- `openfda.device_class` - 器材類別
- `product_res_number` - 產品召回編號
- `firm_fei_number` - 公司設施識別碼
- `k_numbers` - 相關的 510(k) 編號
- `pma_numbers` - 相關的 PMA 編號
- `root_cause_description` - 問題根本原因

**常見使用案例**：
- 召回追蹤
- 品質調查
- 根本原因分析
- 趨勢識別

**查詢範例**：
```python
# 按產品代碼搜尋召回
params = {
    "api_key": api_key,
    "search": "product_code:DQY",
    "limit": 20
}

response = requests.get("https://api.fda.gov/device/recall.json", params=params)
```

### 6. 上市前核准（PMA）

**端點**：`https://api.fda.gov/device/pma.json`

**目的**：存取 FDA 第三類醫療器材上市前核准程序的資料。

**資料來源**：PMA 資料庫

**主要欄位**：
- `pma_number` - PMA 申請編號（例如：P850005）
- `supplement_number` - 補充編號（如適用）
- `applicant` - 公司名稱
- `trade_name` - 商品名/品牌名
- `generic_name` - 通用名稱
- `product_code` - FDA 產品代碼
- `decision_date` - FDA 決定日期
- `decision_code` - 核准狀態（APPR = 已核准）
- `advisory_committee` - 諮詢委員會
- `openfda.device_name` - 官方器材名稱
- `openfda.device_class` - 器材類別
- `openfda.medical_specialty_description` - 醫學專科
- `openfda.regulation_number` - 法規編號

**常見使用案例**：
- 高風險器材研究
- 核准時間線分析
- 法規策略
- 市場情報
- 臨床試驗規劃

**查詢範例**：
```python
# 按公司查詢 PMA 核准
params = {
    "api_key": api_key,
    "search": "applicant:Boston+Scientific",
    "limit": 50
}

response = requests.get("https://api.fda.gov/device/pma.json", params=params)
```

```python
# 搜尋特定器材的 PMA
params = {
    "api_key": api_key,
    "search": "generic_name:*cardiac+pacemaker*",
    "limit": 10
}
```

### 7. 註冊與列名

**端點**：`https://api.fda.gov/device/registrationlisting.json`

**目的**：存取醫療器材設施的位置資料及其製造的器材。

**資料來源**：器材註冊和列名資料庫

**主要欄位**：
- `registration.fei_number` - 設施建立識別碼
- `registration.name` - 設施名稱
- `registration.registration_number` - 註冊編號
- `registration.reg_expiry_date_year` - 註冊到期年份
- `registration.address_line_1` - 街道地址
- `registration.city` - 城市
- `registration.state_code` - 州/省代碼
- `registration.iso_country_code` - 國家代碼
- `registration.zip_code` - 郵遞區號
- `products.product_code` - 器材產品代碼
- `products.created_date` - 器材列名時間
- `products.openfda.device_name` - 器材名稱
- `products.openfda.device_class` - 器材類別
- `proprietary_name` - 專有名稱/品牌名
- `establishment_type` - 營運類型（製造商等）

**常見使用案例**：
- 製造商識別
- 設施位置查詢
- 供應鏈對應
- 盡職調查研究
- 市場分析

**查詢範例**：
```python
# 按國家查詢註冊設施
params = {
    "api_key": api_key,
    "search": "registration.iso_country_code:US",
    "limit": 100
}

response = requests.get("https://api.fda.gov/device/registrationlisting.json", params=params)
```

```python
# 按設施名稱搜尋
params = {
    "api_key": api_key,
    "search": "registration.name:*Johnson*",
    "limit": 10
}
```

### 8. 唯一器材識別（UDI）

**端點**：`https://api.fda.gov/device/udi.json`

**目的**：存取全球唯一器材識別資料庫（GUDID），包含器材識別資訊。

**資料來源**：GUDID

**主要欄位**：
- `identifiers.id` - 器材識別碼（DI）
- `identifiers.issuing_agency` - 發行機構（GS1、HIBCC、ICCBBA）
- `identifiers.type` - 主要或包裝 DI
- `brand_name` - 品牌名稱
- `version_model_number` - 版本/型號
- `catalog_number` - 目錄編號
- `company_name` - 器材公司
- `device_count_in_base_package` - 基本包裝中的數量
- `device_description` - 描述
- `is_rx` - 處方器材（true/false）
- `is_otc` - 非處方器材（true/false）
- `is_combination_product` - 組合產品（true/false）
- `is_kit` - 套組（true/false）
- `is_labeled_no_nrl` - 標示不含乳膠
- `has_lot_or_batch_number` - 使用批號
- `has_serial_number` - 使用序號
- `has_manufacturing_date` - 有製造日期
- `has_expiration_date` - 有有效期限
- `mri_safety` - MRI 安全狀態
- `gmdn_terms` - 全球醫療器材命名術語
- `product_codes` - FDA 產品代碼
- `storage` - 儲存要求
- `customer_contacts` - 聯絡資訊

**常見使用案例**：
- 器材識別和驗證
- 供應鏈追蹤
- 不良事件報告
- 庫存管理
- 採購

**查詢範例**：
```python
# 按 UDI 查詢器材
params = {
    "api_key": api_key,
    "search": "identifiers.id:00884838003019",
    "limit": 1
}

response = requests.get("https://api.fda.gov/device/udi.json", params=params)
```

```python
# 按品牌名稱查詢處方器材
params = {
    "api_key": api_key,
    "search": "brand_name:*insulin+pump*+AND+is_rx:true",
    "limit": 10
}
```

```python
# 搜尋 MRI 安全器材
params = {
    "api_key": api_key,
    "search": 'mri_safety:"MR Safe"',
    "limit": 50
}
```

### 9. COVID-19 血清學檢測評估

**端點**：`https://api.fda.gov/device/covid19serology.json`

**目的**：存取 FDA 對 COVID-19 抗體檢測的獨立評估。

**資料來源**：FDA COVID-19 血清學檢測效能

**主要欄位**：
- `manufacturer` - 檢測製造商
- `device` - 器材/檢測名稱
- `authorization_status` - EUA 狀態
- `control_panel` - 用於評估的對照面板
- `sample_sensitivity_report_one` - 敏感度資料（第一份報告）
- `sample_specificity_report_one` - 特異度資料（第一份報告）
- `sample_sensitivity_report_two` - 敏感度資料（第二份報告）
- `sample_specificity_report_two` - 特異度資料（第二份報告）

**常見使用案例**：
- 檢測效能比較
- 診斷準確性評估
- 採購決策支援
- 品質保證

**查詢範例**：
```python
# 按製造商查詢檢測
params = {
    "api_key": api_key,
    "search": "manufacturer:Abbott",
    "limit": 10
}

response = requests.get("https://api.fda.gov/device/covid19serology.json", params=params)
```

```python
# 取得所有具有 EUA 的檢測
params = {
    "api_key": api_key,
    "search": "authorization_status:*EUA*",
    "limit": 100
}
```

## 整合技巧

### 全面器材搜尋

```python
def search_device_across_databases(device_name, api_key):
    """
    跨多個 FDA 資料庫搜尋器材。

    參數：
        device_name: 器材名稱或部分名稱
        api_key: FDA API 金鑰

    回傳：
        包含各資料庫結果的字典
    """
    results = {}

    # 搜尋不良事件
    events_url = "https://api.fda.gov/device/event.json"
    events_params = {
        "api_key": api_key,
        "search": f"device.brand_name:*{device_name}*",
        "limit": 10
    }
    results["adverse_events"] = requests.get(events_url, params=events_params).json()

    # 搜尋 510(k) 許可
    fiveten_url = "https://api.fda.gov/device/510k.json"
    fiveten_params = {
        "api_key": api_key,
        "search": f"device_name:*{device_name}*",
        "limit": 10
    }
    results["510k_clearances"] = requests.get(fiveten_url, params=fiveten_params).json()

    # 搜尋召回
    recall_url = "https://api.fda.gov/device/enforcement.json"
    recall_params = {
        "api_key": api_key,
        "search": f"product_description:*{device_name}*",
        "limit": 10
    }
    results["recalls"] = requests.get(recall_url, params=recall_params).json()

    # 搜尋 UDI
    udi_url = "https://api.fda.gov/device/udi.json"
    udi_params = {
        "api_key": api_key,
        "search": f"brand_name:*{device_name}*",
        "limit": 10
    }
    results["udi"] = requests.get(udi_url, params=udi_params).json()

    return results
```

### 產品代碼查詢

```python
def get_device_classification(product_code, api_key):
    """
    取得器材產品代碼的詳細分類資訊。

    參數：
        product_code: 三字母 FDA 產品代碼
        api_key: FDA API 金鑰

    回傳：
        分類詳情字典
    """
    url = "https://api.fda.gov/device/classification.json"
    params = {
        "api_key": api_key,
        "search": f"product_code:{product_code}",
        "limit": 1
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "results" in data and len(data["results"]) > 0:
        classification = data["results"][0]
        return {
            "product_code": classification.get("product_code"),
            "device_name": classification.get("device_name"),
            "device_class": classification.get("device_class"),
            "regulation_number": classification.get("regulation_number"),
            "medical_specialty": classification.get("medical_specialty_description"),
            "gmp_exempt": classification.get("gmp_exempt_flag") == "Y",
            "implant": classification.get("implant_flag") == "Y",
            "life_sustaining": classification.get("life_sustain_support_flag") == "Y"
        }
    return None
```

## 最佳實踐

1. **使用產品代碼** - 跨器材資料庫搜尋最有效的方式
2. **檢查多個資料庫** - 器材資訊分佈在多個端點
3. **處理大型結果集** - 器材資料庫可能非常大；使用分頁
4. **驗證器材識別碼** - 確保 UDI、510(k) 編號和 PMA 編號格式正確
5. **按器材類別過濾** - 相關時按風險分類縮小搜尋範圍
6. **使用精確品牌名稱** - 萬用字元有效但精確匹配更可靠
7. **考慮日期範圍** - 器材資料累積數十年；適時按日期過濾
8. **交叉參考資料** - 連結不良事件與召回和註冊以取得完整圖像
9. **監控召回狀態** - 召回狀態從「進行中」變更為「已完成」
10. **檢查設施註冊** - 設施必須每年註冊；檢查到期日期

## 其他資源

- OpenFDA 器材 API 文件：https://open.fda.gov/apis/device/
- 器材分類資料庫：https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpcd/classification.cfm
- GUDID：https://accessgudid.nlm.nih.gov/
- API 基礎：請參閱本參考目錄中的 `api_basics.md`
- Python 範例：請參閱 `scripts/fda_device_query.py`
