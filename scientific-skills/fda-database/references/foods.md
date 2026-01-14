# FDA 食品資料庫

本參考文件涵蓋透過 openFDA 可存取的 FDA 食品相關 API 端點。

## 概述

FDA 食品資料庫提供食品產品的相關資訊，包括不良事件和執法行動。這些資料庫有助於追蹤食品安全問題、召回和消費者投訴。

## 可用端點

### 1. 食品不良事件

**端點**：`https://api.fda.gov/food/event.json`

**目的**：存取食品產品、膳食補充劑和化妝品的不良事件報告。

**資料來源**：CAERS（CFSAN 不良事件報告系統）

**主要欄位**：
- `date_started` - 不良事件開始時間
- `date_created` - 報告建立時間
- `report_number` - 唯一報告識別碼
- `outcomes` - 事件結果（例如：住院、死亡）
- `reactions` - 報告的不良反應/症狀
- `consumer.age` - 消費者年齡
- `consumer.age_unit` - 年齡單位（年、月等）
- `consumer.gender` - 消費者性別
- `products` - 涉及產品的陣列
- `products.name_brand` - 產品品牌名稱
- `products.industry_code` - 產品類別代碼
- `products.industry_name` - 產品類別名稱
- `products.role` - 產品角色（嫌疑、伴隨）

**產品類別（industry_name）**：
- 烘焙產品/麵團/混合物/糖霜
- 飲料（咖啡、茶、軟性飲料等）
- 膳食補充劑
- 冰淇淋產品
- 化妝品
- 維生素和營養補充劑
- 其他許多類別

**常見使用案例**：
- 食品安全監測
- 膳食補充劑監控
- 不良事件趨勢分析
- 產品安全評估
- 消費者投訴追蹤

**查詢範例**：
```python
import requests

api_key = "YOUR_API_KEY"
url = "https://api.fda.gov/food/event.json"

# 查詢膳食補充劑的不良事件
params = {
    "api_key": api_key,
    "search": "products.industry_name:Dietary+Supplements",
    "limit": 10
}

response = requests.get(url, params=params)
data = response.json()
```

```python
# 計算最常見的反應
params = {
    "api_key": api_key,
    "search": "products.industry_name:*Beverages*",
    "count": "reactions.exact"
}
```

```python
# 查詢嚴重結果（住院、死亡）
params = {
    "api_key": api_key,
    "search": "outcomes:Hospitalization",
    "limit": 50,
    "sort": "date_created:desc"
}
```

```python
# 按產品品牌名稱搜尋
params = {
    "api_key": api_key,
    "search": "products.name_brand:*protein+powder*",
    "limit": 20
}
```

### 2. 食品執法報告

**端點**：`https://api.fda.gov/food/enforcement.json`

**目的**：存取 FDA 發布的食品產品召回執法報告。

**資料來源**：FDA 執法報告

**主要欄位**：
- `status` - 目前狀態（進行中、已完成、已終止）
- `recall_number` - 唯一召回識別碼
- `classification` - 第一類、第二類或第三類
- `product_description` - 被召回食品產品的描述
- `reason_for_recall` - 召回原因
- `product_quantity` - 召回的產品數量
- `code_info` - 批號、批次代碼、UPC
- `distribution_pattern` - 地理分佈
- `recalling_firm` - 進行召回的公司
- `recall_initiation_date` - 召回開始時間
- `report_date` - FDA 收到通知時間
- `voluntary_mandated` - 自願或 FDA 強制召回
- `city` - 召回公司城市
- `state` - 召回公司州/省
- `country` - 召回公司國家
- `initial_firm_notification` - 公司通知方式

**分類等級**：
- **第一類**：可能導致嚴重健康問題或死亡的危險或有缺陷產品（例如：具有嚴重風險的未申報過敏原、肉毒桿菌污染）
- **第二類**：可能導致暫時健康問題或輕微威脅的產品（例如：輕微過敏原問題、品質缺陷）
- **第三類**：不太可能導致不良健康反應但違反 FDA 法規的產品（例如：標籤錯誤、品質問題）

**常見召回原因**：
- 未申報過敏原（牛奶、雞蛋、花生、堅果、大豆、小麥、魚、貝類、芝麻）
- 微生物污染（李斯特菌、沙門氏菌、大腸桿菌等）
- 異物污染（金屬、塑膠、玻璃）
- 標籤錯誤
- 不當加工/包裝
- 化學污染

**常見使用案例**：
- 食品安全監控
- 供應鏈風險管理
- 過敏原追蹤
- 零售商召回協調
- 消費者安全警報

**查詢範例**：
```python
# 查詢所有第一類食品召回（最嚴重）
params = {
    "api_key": api_key,
    "search": "classification:Class+I",
    "limit": 20,
    "sort": "report_date:desc"
}

response = requests.get("https://api.fda.gov/food/enforcement.json", params=params)
```

```python
# 搜尋過敏原相關召回
params = {
    "api_key": api_key,
    "search": "reason_for_recall:*undeclared+allergen*",
    "limit": 50
}
```

```python
# 查詢李斯特菌污染召回
params = {
    "api_key": api_key,
    "search": "reason_for_recall:*listeria*",
    "limit": 30,
    "sort": "recall_initiation_date:desc"
}
```

```python
# 按特定公司取得召回
params = {
    "api_key": api_key,
    "search": "recalling_firm:*General+Mills*",
    "limit": 20
}
```

```python
# 查詢進行中的召回
params = {
    "api_key": api_key,
    "search": "status:Ongoing",
    "limit": 100
}
```

```python
# 按產品類型搜尋
params = {
    "api_key": api_key,
    "search": "product_description:*ice+cream*",
    "limit": 25
}
```

## 整合技巧

### 過敏原監控系統

```python
def monitor_allergen_recalls(allergens, api_key, days_back=30):
    """
    監控特定過敏原的食品召回。

    參數：
        allergens: 要監控的過敏原列表（例如：["peanut", "milk", "soy"]）
        api_key: FDA API 金鑰
        days_back: 回溯天數

    回傳：
        匹配召回的列表
    """
    import requests
    from datetime import datetime, timedelta

    # 計算日期範圍
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    date_range = f"[{start_date.strftime('%Y%m%d')}+TO+{end_date.strftime('%Y%m%d')}]"

    url = "https://api.fda.gov/food/enforcement.json"
    all_recalls = []

    for allergen in allergens:
        params = {
            "api_key": api_key,
            "search": f"reason_for_recall:*{allergen}*+AND+report_date:{date_range}",
            "limit": 100
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if "results" in data:
                for result in data["results"]:
                    result["detected_allergen"] = allergen
                    all_recalls.append(result)

    return all_recalls
```

### 不良事件分析

```python
def analyze_product_adverse_events(product_name, api_key):
    """
    分析特定食品產品的不良事件。

    參數：
        product_name: 產品名稱或部分名稱
        api_key: FDA API 金鑰

    回傳：
        包含分析結果的字典
    """
    import requests
    from collections import Counter

    url = "https://api.fda.gov/food/event.json"
    params = {
        "api_key": api_key,
        "search": f"products.name_brand:*{product_name}*",
        "limit": 1000
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "results" not in data:
        return {"error": "未找到結果"}

    results = data["results"]

    # 提取所有反應
    all_reactions = []
    all_outcomes = []

    for event in results:
        if "reactions" in event:
            all_reactions.extend(event["reactions"])
        if "outcomes" in event:
            all_outcomes.extend(event["outcomes"])

    # 計算頻率
    reaction_counts = Counter(all_reactions)
    outcome_counts = Counter(all_outcomes)

    return {
        "total_events": len(results),
        "most_common_reactions": reaction_counts.most_common(10),
        "outcome_distribution": dict(outcome_counts),
        "serious_outcomes": sum(1 for o in all_outcomes if o in ["Hospitalization", "Death", "Disability"])
    }
```

### 召回警報系統

```python
def get_recent_recalls_by_state(state_code, api_key, days=7):
    """
    取得分銷至特定州的近期食品召回。

    參數：
        state_code: 兩字母州代碼（例如：「CA」、「NY」）
        api_key: FDA API 金鑰
        days: 回溯天數

    回傳：
        影響該州的近期召回列表
    """
    import requests
    from datetime import datetime, timedelta

    url = "https://api.fda.gov/food/enforcement.json"

    # 計算日期範圍
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = f"[{start_date.strftime('%Y%m%d')}+TO+{end_date.strftime('%Y%m%d')}]"

    params = {
        "api_key": api_key,
        "search": f"distribution_pattern:*{state_code}*+AND+report_date:{date_range}",
        "limit": 100,
        "sort": "report_date:desc"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("results", [])
    return []
```

## 最佳實踐

1. **監控過敏原召回** - 對食品服務和零售至關重要
2. **檢查分銷模式** - 召回可能是區域性或全國性的
3. **追蹤召回狀態** - 狀態從「進行中」變更為「已完成」
4. **按分類過濾** - 優先處理第一類召回以立即採取行動
5. **使用日期範圍** - 專注於近期事件以確保營運相關性
6. **交叉參考產品** - 同一產品可能同時出現在不良事件和執法中
7. **仔細解析 code_info** - 批號和 UPC 格式各異
8. **考慮產品類別** - 產業代碼有助於分類產品
9. **追蹤嚴重結果** - 住院和死亡需要立即關注
10. **實作警報系統** - 自動化關鍵產品/過敏原的監控

## 需要監控的常見過敏原

FDA 認定 9 種主要食品過敏原必須申報：
1. 牛奶
2. 雞蛋
3. 魚
4. 甲殼類貝類
5. 堅果
6. 花生
7. 小麥
8. 大豆
9. 芝麻

這些過敏原佔食品過敏的 90% 以上，是第一類召回最常見的原因。

## 其他資源

- OpenFDA 食品 API 文件：https://open.fda.gov/apis/food/
- CFSAN 不良事件報告：https://www.fda.gov/food/compliance-enforcement-food/cfsan-adverse-event-reporting-system-caers
- 食品召回：https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts
- API 基礎：請參閱本參考目錄中的 `api_basics.md`
- Python 範例：請參閱 `scripts/fda_food_query.py`
