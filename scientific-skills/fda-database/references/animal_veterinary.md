# FDA 動物與獸醫資料庫

本參考文件涵蓋透過 openFDA 可存取的 FDA 動物與獸醫藥品 API 端點。

## 概述

FDA 動物與獸醫資料庫提供與動物藥物和獸醫醫療產品相關的不良事件資訊。這些資料庫有助於監控伴侶動物、畜牧動物和其他動物使用產品的安全性。

## 可用端點

### 動物藥物不良事件

**端點**：`https://api.fda.gov/animalandveterinary/event.json`

**目的**：存取與動物藥物相關的副作用、產品使用錯誤、產品品質問題和治療失敗的報告。

**資料來源**：FDA 獸醫中心（CVM）不良事件報告系統

**主要欄位**：
- `unique_aer_id_number` - 唯一不良事件報告識別碼
- `report_id` - 報告 ID 編號
- `receiver.organization` - 接收報告的組織
- `receiver.street_address` - 接收者地址
- `receiver.city` - 接收者城市
- `receiver.state` - 接收者州/省
- `receiver.postal_code` - 接收者郵遞區號
- `receiver.country` - 接收者國家
- `primary_reporter` - 主要報告者類型（例如：獸醫、飼主）
- `onset_date` - 不良事件開始日期
- `animal.species` - 受影響的動物物種
- `animal.gender` - 動物性別
- `animal.age.min` - 最小年齡
- `animal.age.max` - 最大年齡
- `animal.age.unit` - 年齡單位（天、月、年）
- `animal.age.qualifier` - 年齡限定詞
- `animal.breed.is_crossbred` - 是否為混種
- `animal.breed.breed_component` - 品種
- `animal.weight.min` - 最小體重
- `animal.weight.max` - 最大體重
- `animal.weight.unit` - 體重單位
- `animal.female_animal_physiological_status` - 生殖狀態
- `animal.reproductive_status` - 絕育狀態
- `drug` - 涉及藥物的陣列
- `drug.active_ingredients` - 活性成分
- `drug.active_ingredients.name` - 成分名稱
- `drug.active_ingredients.dose` - 劑量資訊
- `drug.brand_name` - 品牌名稱
- `drug.manufacturer.name` - 製造商
- `drug.administered_by` - 給藥者
- `drug.route` - 給藥途徑
- `drug.dosage_form` - 劑型
- `drug.atc_vet_code` - ATC 獸醫代碼
- `reaction` - 不良反應陣列
- `reaction.veddra_version` - VeDDRA 字典版本
- `reaction.veddra_term_code` - VeDDRA 術語代碼
- `reaction.veddra_term_name` - VeDDRA 術語名稱
- `reaction.accuracy` - 診斷準確性
- `reaction.number_of_animals_affected` - 受影響動物數量
- `reaction.number_of_animals_treated` - 治療動物數量
- `outcome.medical_status` - 醫療結果
- `outcome.number_of_animals_affected` - 受結果影響的動物數量
- `serious_ae` - 是否為嚴重不良事件
- `health_assessment_prior_to_exposure.assessed_by` - 健康評估者
- `health_assessment_prior_to_exposure.condition` - 健康狀況
- `treated_for_ae` - 是否接受治療
- `time_between_exposure_and_onset` - 暴露至發病時間
- `duration.unit` - 持續時間單位
- `duration.value` - 持續時間值

**常見動物物種**：
- 狗（Canis lupus familiaris）
- 貓（Felis catus）
- 馬（Equus caballus）
- 牛（Bos taurus）
- 豬（Sus scrofa domesticus）
- 雞（Gallus gallus domesticus）
- 綿羊（Ovis aries）
- 山羊（Capra aegagrus hircus）
- 以及其他許多物種

**常見使用案例**：
- 獸醫藥物安全監視
- 產品安全監控
- 不良事件趨勢分析
- 藥物安全比較
- 物種特定安全研究
- 品種易感性研究

**查詢範例**：
```python
import requests

api_key = "YOUR_API_KEY"
url = "https://api.fda.gov/animalandveterinary/event.json"

# 查詢狗的不良事件
params = {
    "api_key": api_key,
    "search": "animal.species:Dog",
    "limit": 10
}

response = requests.get(url, params=params)
data = response.json()
```

```python
# 搜尋特定藥物的不良事件
params = {
    "api_key": api_key,
    "search": "drug.brand_name:*flea+collar*",
    "limit": 20
}
```

```python
# 按物種計算最常見反應
params = {
    "api_key": api_key,
    "search": "animal.species:Cat",
    "count": "reaction.veddra_term_name.exact"
}
```

```python
# 查詢嚴重不良事件
params = {
    "api_key": api_key,
    "search": "serious_ae:true+AND+outcome.medical_status:Died",
    "limit": 50,
    "sort": "onset_date:desc"
}
```

```python
# 按活性成分搜尋
params = {
    "api_key": api_key,
    "search": "drug.active_ingredients.name:*ivermectin*",
    "limit": 25
}
```

```python
# 查詢特定品種的事件
params = {
    "api_key": api_key,
    "search": "animal.breed.breed_component:*Labrador*",
    "limit": 30
}
```

```python
# 按給藥途徑取得事件
params = {
    "api_key": api_key,
    "search": "drug.route:*topical*",
    "limit": 40
}
```

## VeDDRA - 獸醫藥物相關事務詞典

獸醫藥物相關事務詞典（VeDDRA）是用於不良事件報告的標準化國際獸醫術語。它提供：

- 獸醫不良事件的標準化術語
- 術語的層次結構組織
- 物種特定術語
- 國際統一化

**VeDDRA 術語結構**：
- 術語按層次結構組織
- 每個術語都有唯一代碼
- 術語適合特定物種
- 存在多個版本（請檢查 `veddra_version` 欄位）

## 整合技巧

### 物種特定不良事件分析

```python
def analyze_species_adverse_events(species, drug_name, api_key):
    """
    分析特定物種對特定藥物的不良事件。

    參數：
        species: 動物物種（例如：「Dog」、「Cat」、「Horse」）
        drug_name: 藥物品牌名稱或活性成分
        api_key: FDA API 金鑰

    回傳：
        包含分析結果的字典
    """
    import requests
    from collections import Counter

    url = "https://api.fda.gov/animalandveterinary/event.json"
    params = {
        "api_key": api_key,
        "search": f"animal.species:{species}+AND+drug.brand_name:*{drug_name}*",
        "limit": 1000
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "results" not in data:
        return {"error": "未找到結果"}

    results = data["results"]

    # 收集反應和結果
    reactions = []
    outcomes = []
    serious_count = 0

    for event in results:
        if "reaction" in event:
            for reaction in event["reaction"]:
                if "veddra_term_name" in reaction:
                    reactions.append(reaction["veddra_term_name"])

        if "outcome" in event:
            for outcome in event["outcome"]:
                if "medical_status" in outcome:
                    outcomes.append(outcome["medical_status"])

        if event.get("serious_ae") == "true":
            serious_count += 1

    reaction_counts = Counter(reactions)
    outcome_counts = Counter(outcomes)

    return {
        "total_events": len(results),
        "serious_events": serious_count,
        "most_common_reactions": reaction_counts.most_common(10),
        "outcome_distribution": dict(outcome_counts),
        "serious_percentage": round((serious_count / len(results)) * 100, 2) if len(results) > 0 else 0
    }
```

### 品種易感性研究

```python
def analyze_breed_predisposition(reaction_term, api_key, min_events=5):
    """
    識別特定不良反應的品種易感性。

    參數：
        reaction_term: 要分析的 VeDDRA 反應術語
        api_key: FDA API 金鑰
        min_events: 納入品種的最低事件數

    回傳：
        包含事件計數的品種列表
    """
    import requests
    from collections import Counter

    url = "https://api.fda.gov/animalandveterinary/event.json"
    params = {
        "api_key": api_key,
        "search": f"reaction.veddra_term_name:*{reaction_term}*",
        "limit": 1000
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "results" not in data:
        return []

    breeds = []
    for event in data["results"]:
        if "animal" in event and "breed" in event["animal"]:
            breed_info = event["animal"]["breed"]
            if "breed_component" in breed_info:
                if isinstance(breed_info["breed_component"], list):
                    breeds.extend(breed_info["breed_component"])
                else:
                    breeds.append(breed_info["breed_component"])

    breed_counts = Counter(breeds)

    # 按最低事件數過濾
    filtered_breeds = [
        {"breed": breed, "count": count}
        for breed, count in breed_counts.most_common()
        if count >= min_events
    ]

    return filtered_breeds
```

### 比較藥物安全性

```python
def compare_drug_safety(drug_list, species, api_key):
    """
    比較多種藥物在特定物種中的安全概況。

    參數：
        drug_list: 要比較的藥物名稱列表
        species: 動物物種
        api_key: FDA API 金鑰

    回傳：
        比較藥物的字典
    """
    import requests

    url = "https://api.fda.gov/animalandveterinary/event.json"
    comparison = {}

    for drug in drug_list:
        params = {
            "api_key": api_key,
            "search": f"animal.species:{species}+AND+drug.brand_name:*{drug}*",
            "limit": 1000
        }

        response = requests.get(url, params=params)
        data = response.json()

        if "results" in data:
            results = data["results"]
            serious = sum(1 for r in results if r.get("serious_ae") == "true")
            deaths = sum(
                1 for r in results
                if "outcome" in r
                and any(o.get("medical_status") == "Died" for o in r["outcome"])
            )

            comparison[drug] = {
                "total_events": len(results),
                "serious_events": serious,
                "deaths": deaths,
                "serious_rate": round((serious / len(results)) * 100, 2) if len(results) > 0 else 0,
                "death_rate": round((deaths / len(results)) * 100, 2) if len(results) > 0 else 0
            }

    return comparison
```

## 最佳實踐

1. **使用標準物種名稱** - 完整的學名或通用名稱效果最佳
2. **考慮品種變異** - 拼寫和命名可能有所不同
3. **檢查 VeDDRA 版本** - 術語在不同版本間可能會變更
4. **考量報告者偏差** - 獸醫與飼主的報告方式不同
5. **過濾嚴重事件** - 專注於臨床顯著的反應
6. **考慮動物人口統計** - 年齡、體重和生殖狀態很重要
7. **追蹤時間模式** - 可能存在季節性變化
8. **交叉參考產品** - 相同活性成分可能有多個品牌
9. **按途徑分析** - 外用與全身給藥對安全性的影響不同
10. **考慮物種差異** - 藥物對不同物種的影響不同

## 報告來源

動物藥物不良事件報告來自：
- **獸醫** - 專業醫療觀察
- **動物飼主** - 直接觀察和關注
- **製藥公司** - 法規要求的上市後監測
- **FDA 現場工作人員** - 官方調查
- **研究機構** - 臨床研究
- **其他來源** - 各種

不同來源可能有不同的報告門檻和詳細程度。

## 其他資源

- OpenFDA 動物與獸醫 API：https://open.fda.gov/apis/animalandveterinary/
- FDA 獸醫中心：https://www.fda.gov/animal-veterinary
- VeDDRA：https://www.veddra.org/
- API 基礎：請參閱本參考目錄中的 `api_basics.md`
- Python 範例：請參閱 `scripts/fda_animal_query.py`
