# Resolve 端點 - 實體識別

## 用途

Resolve API 識別知識圖譜中實體的 Data Commons ID（DCIDs）。DCIDs 是 Data Commons API 中大多數查詢所必需的，因此解析通常是任何工作流程的第一步。

## 主要功能

端點目前僅支援**地點實體**，並允許透過多種方法解析：
- **按名稱**：使用描述性詞彙搜尋，如 "San Francisco, CA"
- **按 Wikidata ID**：使用外部識別碼查詢（例如 "Q30" 代表美國）
- **按座標**：透過經緯度定位地點
- **按關係表達式**：使用合成屬性進行進階搜尋

## 可用方法

### 1. fetch()

使用關係表達式進行一般解析——最靈活的方法。

**參數：**
- `nodes`：搜尋詞或識別碼列表
- `property`：要搜尋的屬性（例如 "name"、"wikidataId"）

**使用範例：**
```python
from datacommons_client import DataCommonsClient

client = DataCommonsClient()

# 按名稱解析
response = client.resolve.fetch(
    nodes=["California", "Texas"],
    property="name"
)
```

### 2. fetch_dcids_by_name()

具有可選類型篩選的名稱查詢——最常用的方法。

**參數：**
- `names`：要解析的地名列表
- `entity_type`：可選的類型篩選（例如 "City"、"State"、"County"）

**回傳：**具有每個名稱候選項的 `ResolveResponse` 物件

**使用範例：**
```python
# 基本名稱解析
response = client.resolve.fetch_dcids_by_name(
    names=["San Francisco, CA", "Los Angeles"]
)

# 具有類型篩選
response = client.resolve.fetch_dcids_by_name(
    names=["San Francisco"],
    entity_type="City"
)

# 存取結果
for name, result in response.to_dict().items():
    print(f"{name}: {result['candidates']}")
```

### 3. fetch_dcids_by_wikidata_id()

具有已知 Wikidata 識別碼的實體的 Wikidata ID 解析。

**參數：**
- `wikidata_ids`：Wikidata ID 列表（例如 "Q30"、"Q99"）

**使用範例：**
```python
# 解析 Wikidata ID
response = client.resolve.fetch_dcids_by_wikidata_id(
    wikidata_ids=["Q30", "Q99"]  # 美國和加州
)
```

### 4. fetch_dcid_by_coordinates()

地理座標查詢以尋找特定經緯度座標處的地點。

**參數：**
- `latitude`：緯度座標
- `longitude`：經度座標

**回傳：**該座標處地點的單一 DCID 字串

**使用範例：**
```python
# 尋找座標處的地點
dcid = client.resolve.fetch_dcid_by_coordinates(
    latitude=37.7749,
    longitude=-122.4194
)
# 回傳舊金山的 DCID
```

## 回應結構

所有方法（除了 `fetch_dcid_by_coordinates`）回傳包含以下內容的 `ResolveResponse` 物件：
- **node**：提供的搜尋詞
- **candidates**：具有可選中繼資料的匹配 DCID 列表
  - 每個候選項可能包含用於消歧的 `dominantType` 欄位
- **輔助方法**：
  - `to_dict()`：完整回應作為字典
  - `to_json()`：JSON 字串格式
  - `to_flat_dict()`：僅包含 DCID 的簡化格式

**回應範例：**
```python
response = client.resolve.fetch_dcids_by_name(names=["Springfield"])

# 可能回傳多個候選項，因為有許多名為 Springfield 的城市
# {
#   "Springfield": {
#     "candidates": [
#       {"dcid": "geoId/1767000", "dominantType": "City"},  # Springfield, IL
#       {"dcid": "geoId/2567000", "dominantType": "City"},  # Springfield, MA
#       ...
#     ]
#   }
# }
```

## 常見使用案例

### 使用案例 1：查詢前解析地名

大多數工作流程從將名稱解析為 DCID 開始：
```python
# 步驟 1：解析名稱
resolve_response = client.resolve.fetch_dcids_by_name(
    names=["California", "Texas"]
)

# 步驟 2：提取 DCID
dcids = []
for name, result in resolve_response.to_dict().items():
    if result["candidates"]:
        dcids.append(result["candidates"][0]["dcid"])

# 步驟 3：使用 DCID 查詢資料
data_response = client.observation.fetch(
    variable_dcids=["Count_Person"],
    entity_dcids=dcids,
    date="latest"
)
```

### 使用案例 2：處理模糊名稱

當存在多個候選項時，使用 `dominantType` 或更具體：
```python
# 模糊名稱
response = client.resolve.fetch_dcids_by_name(names=["Springfield"])
candidates = response.to_dict()["Springfield"]["candidates"]

# 按類型篩選或根據上下文選擇
city_candidates = [c for c in candidates if c.get("dominantType") == "City"]

# 或在查詢中更具體
response = client.resolve.fetch_dcids_by_name(
    names=["Springfield, Illinois"],
    entity_type="City"
)
```

### 使用案例 3：批次解析

高效解析多個實體：
```python
places = [
    "San Francisco, CA",
    "Los Angeles, CA",
    "San Diego, CA",
    "Sacramento, CA"
]

response = client.resolve.fetch_dcids_by_name(names=places)

# 建立名稱到 DCID 的對應
name_to_dcid = {}
for name, result in response.to_dict().items():
    if result["candidates"]:
        name_to_dcid[name] = result["candidates"][0]["dcid"]
```

### 使用案例 4：座標查詢

尋找位置的行政區劃地點：
```python
# 使用者提供座標，尋找地點
latitude, longitude = 37.7749, -122.4194
dcid = client.resolve.fetch_dcid_by_coordinates(
    latitude=latitude,
    longitude=longitude
)

# 現在查詢該地點的資料
response = client.observation.fetch(
    variable_dcids=["Count_Person", "MedianIncome_Household"],
    entity_dcids=[dcid],
    date="latest"
)
```

### 使用案例 5：外部 ID 整合

當處理使用 Wikidata ID 的外部資料集時：
```python
# 外部資料集有 Wikidata ID
wikidata_ids = ["Q30", "Q99", "Q1384"]  # 美國、加州、紐約

# 轉換為 Data Commons DCID
response = client.resolve.fetch_dcids_by_wikidata_id(
    wikidata_ids=wikidata_ids
)

# 提取 DCID 以進行進一步查詢
dcids = []
for wid, result in response.to_dict().items():
    if result["candidates"]:
        dcids.append(result["candidates"][0]["dcid"])
```

## 重要限制

1. **僅支援地點實體**：Resolve API 目前僅支援地點實體（國家、州、城市、縣等）。對於其他實體類型，必須透過其他方式取得 DCID（例如 Node API 探索）。

2. **無法解析連結實體屬性**：對於涉及 `containedInPlace` 等關係的查詢，改用 Node API。

3. **模糊處理**：當存在多個候選項時，API 回傳所有匹配項。應用程式必須根據上下文或額外篩選決定哪個是正確的。

## 最佳實踐

1. **始終先解析名稱**：永遠不要假設 DCID 格式——始終使用 Resolve API
2. **快取解析結果**：如果重複查詢相同的地點，快取名稱→DCID 對應
3. **處理模糊性**：檢查多個候選項並使用 `entity_type` 篩選或更具體的名稱
4. **驗證結果**：在存取 DCID 之前始終檢查 `candidates` 列表不為空
5. **使用適當的方法**：
   - 名稱 → `fetch_dcids_by_name()`
   - 座標 → `fetch_dcid_by_coordinates()`
   - Wikidata ID → `fetch_dcids_by_wikidata_id()`
