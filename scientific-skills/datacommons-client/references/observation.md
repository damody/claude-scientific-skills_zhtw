# Observation 端點 - 統計資料查詢

## 用途

Observation API 擷取統計觀測值——連接實體、變數和特定日期的資料點。範例包括：
- 「2020 年美國人口」
- 「加州 GDP 隨時間變化」
- 「州內所有縣的失業率」

## 核心方法

### 1. fetch()

擷取觀測值的主要方法，具有靈活的實體規格。

**主要參數：**
- `variable_dcids`（必填）：統計變數識別碼列表
- `entity_dcids` 或 `entity_expression`（必填）：按 ID 或關係表達式指定實體
- `date`（可選）：預設為 "latest"。接受：
  - ISO-8601 格式（例如 "2020"、"2020-01"、"2020-01-15"）
  - "all" 取得完整時間序列
  - "latest" 取得最新資料
- `select`（可選）：控制回傳的欄位
  - 預設：`["date", "entity", "variable", "value"]`
  - 替代：`["entity", "variable", "facet"]` 檢查可用性而不取得資料
- `filter_facet_domains`：按資料來源網域篩選
- `filter_facet_ids`：按特定 facet ID 篩選

**回應結構：**
資料按變數 → 實體層次組織，包含關於 "facets"（資料來源）的中繼資料，包括：
- 來源 URL
- 測量方法
- 觀測期間
- 匯入名稱

**使用範例：**
```python
from datacommons_client import DataCommonsClient

client = DataCommonsClient()

# 取得多個實體的最新人口
response = client.observation.fetch(
    variable_dcids=["Count_Person"],
    entity_dcids=["geoId/06", "geoId/48"],  # 加州和德州
    date="latest"
)

# 取得完整時間序列
response = client.observation.fetch(
    variable_dcids=["Count_Person"],
    entity_dcids=["country/USA"],
    date="all"
)

# 使用關係表達式查詢階層
response = client.observation.fetch(
    variable_dcids=["Count_Person"],
    entity_expression="geoId/06<-containedInPlace+{typeOf:County}",
    date="2020"
)
```

### 2. fetch_available_statistical_variables()

發現哪些統計變數包含給定實體的資料。

**輸入：**僅實體 DCIDs
**輸出：**按實體組織的可用變數字典

**使用範例：**
```python
# 檢查加州有哪些可用變數
available = client.observation.fetch_available_statistical_variables(
    entity_dcids=["geoId/06"]
)
```

### 3. fetch_observations_by_entity_dcid()

按 DCID 明確鎖定特定實體的方法（功能上等同於帶有 entity_dcids 的 `fetch()`）。

### 4. fetch_observations_by_entity_type()

擷取按父級和類型分組的多個實體的觀測值——適用於查詢區域內的所有國家或州內的所有縣。

**參數：**
- `parent_entity`：父實體 DCID
- `entity_type`：子實體的類型
- `variable_dcids`：要查詢的統計變數
- `date`：時間規格
- `select` 和篩選選項

**使用範例：**
```python
# 取得加州所有縣的人口
response = client.observation.fetch_observations_by_entity_type(
    parent_entity="geoId/06",
    entity_type="County",
    variable_dcids=["Count_Person"],
    date="2020"
)
```

## 回應物件方法

所有回應物件支援：
- `to_json()`：格式化為 JSON 字串
- `to_dict()`：回傳為字典
- `get_data_by_entity()`：按實體而非變數重新組織
- `to_observations_as_records()`：展平為個別記錄

## 常見使用案例

### 使用案例 1：查詢前檢查資料可用性

使用 `select=["entity", "variable"]` 確認實體有觀測值而不擷取實際資料：
```python
response = client.observation.fetch(
    variable_dcids=["Count_Person"],
    entity_dcids=["geoId/06"],
    select=["entity", "variable"]
)
```

### 使用案例 2：存取完整時間序列

請求 `date="all"` 以取得完整的歷史觀測值進行趨勢分析：
```python
response = client.observation.fetch(
    variable_dcids=["Count_Person", "UnemploymentRate_Person"],
    entity_dcids=["country/USA"],
    date="all"
)
```

### 使用案例 3：按資料來源篩選

指定 `filter_facet_domains` 從特定來源擷取資料以確保一致性：
```python
response = client.observation.fetch(
    variable_dcids=["Count_Person"],
    entity_dcids=["country/USA"],
    filter_facet_domains=["census.gov"]
)
```

### 使用案例 4：查詢階層關係

使用關係表達式擷取相關實體的觀測值：
```python
# 取得加州內所有縣的資料
response = client.observation.fetch(
    variable_dcids=["MedianIncome_Household"],
    entity_expression="geoId/06<-containedInPlace+{typeOf:County}",
    date="2020"
)
```

## 使用 Pandas

API 與 Pandas 無縫整合。安裝具有 Pandas 支援的版本：
```bash
pip install "datacommons-client[Pandas]"
```

回應物件可轉換為 DataFrames 進行分析：
```python
response = client.observation.fetch(
    variable_dcids=["Count_Person"],
    entity_dcids=["geoId/06", "geoId/48"],
    date="all"
)

# 轉換為 DataFrame
df = response.to_observations_as_records()
# 回傳具有欄位的 DataFrame：date、entity、variable、value
```

## 重要注意事項

- **facets** 代表資料來源並包含來源中繼資料
- **orderedFacets** 按可靠性/時效性排序
- 對複雜的圖譜查詢使用關係表達式
- `fetch()` 方法最為靈活——大多數查詢使用它
