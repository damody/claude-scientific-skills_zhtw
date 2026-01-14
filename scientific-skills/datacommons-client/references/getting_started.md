# Data Commons 入門指南

## 快速入門

本指南提供常見 Data Commons 工作流程的端到端範例。

## 安裝與設定

```bash
# 安裝具有 Pandas 支援的版本
pip install "datacommons-client[Pandas]"

# 為 datacommons.org 設定 API 金鑰
export DC_API_KEY="your_api_key_here"
```

在此申請 API 金鑰：https://apikeys.datacommons.org/

## 範例 1：基本人口查詢

查詢特定地點的當前人口：

```python
from datacommons_client import DataCommonsClient

# 初始化 client
client = DataCommonsClient()

# 步驟 1：將地名解析為 DCIDs
places = ["California", "Texas", "New York"]
resolve_response = client.resolve.fetch_dcids_by_name(
    names=places,
    entity_type="State"
)

# 提取 DCIDs
dcids = []
for name, result in resolve_response.to_dict().items():
    if result["candidates"]:
        dcids.append(result["candidates"][0]["dcid"])
        print(f"{name}: {result['candidates'][0]['dcid']}")

# 步驟 2：查詢人口資料
response = client.observation.fetch(
    variable_dcids=["Count_Person"],
    entity_dcids=dcids,
    date="latest"
)

# 步驟 3：顯示結果
data = response.to_dict()
for variable, entities in data.items():
    for entity, observations in entities.items():
        for obs in observations:
            print(f"{entity}: {obs['value']:,} 人 ({obs['date']})")
```

## 範例 2：時間序列分析

取得並繪製歷史失業率：

```python
import pandas as pd
import matplotlib.pyplot as plt

client = DataCommonsClient()

# 查詢隨時間變化的失業率
response = client.observation.fetch(
    variable_dcids=["UnemploymentRate_Person"],
    entity_dcids=["country/USA"],
    date="all"  # 取得所有歷史資料
)

# 轉換為 DataFrame
df = response.to_observations_as_records()

# 繪圖
df = df.sort_values('date')
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['value'])
plt.title('美國失業率隨時間變化')
plt.xlabel('年份')
plt.ylabel('失業率 (%)')
plt.grid(True)
plt.show()
```

## 範例 3：地理階層查詢

取得州內所有縣的資料：

```python
client = DataCommonsClient()

# 查詢加州所有縣的家庭收入中位數
response = client.observation.fetch(
    variable_dcids=["Median_Income_Household"],
    entity_expression="geoId/06<-containedInPlace+{typeOf:County}",
    date="2020"
)

# 轉換為 DataFrame 並排序
df = response.to_observations_as_records()

# 取得縣名
county_dcids = df['entity'].unique().tolist()
names = client.node.fetch_entity_names(node_dcids=county_dcids)

# 將名稱加入 dataframe
df['name'] = df['entity'].map(names)

# 顯示收入前 10 名
top_counties = df.nlargest(10, 'value')[['name', 'value']]
print("\n加州家庭收入中位數前 10 名的縣：")
for idx, row in top_counties.iterrows():
    print(f"{row['name']}: ${row['value']:,.0f}")
```

## 範例 4：多變數比較

比較多個實體的多項統計資料：

```python
import pandas as pd

client = DataCommonsClient()

# 定義地點
places = ["California", "Texas", "Florida", "New York"]
resolve_response = client.resolve.fetch_dcids_by_name(names=places)

dcids = []
name_map = {}
for name, result in resolve_response.to_dict().items():
    if result["candidates"]:
        dcid = result["candidates"][0]["dcid"]
        dcids.append(dcid)
        name_map[dcid] = name

# 查詢多個變數
variables = [
    "Count_Person",
    "Median_Income_Household",
    "UnemploymentRate_Person",
    "Median_Age_Person"
]

response = client.observation.fetch(
    variable_dcids=variables,
    entity_dcids=dcids,
    date="latest"
)

# 轉換為 DataFrame
df = response.to_observations_as_records()

# 加入可讀的名稱
df['state'] = df['entity'].map(name_map)

# 樞紐以進行比較
pivot = df.pivot_table(
    values='value',
    index='state',
    columns='variable'
)

print("\n州別比較：")
print(pivot.to_string())
```

## 範例 5：座標查詢

根據座標尋找並查詢該位置的資料：

```python
client = DataCommonsClient()

# 使用者提供座標（例如來自 GPS）
latitude, longitude = 37.7749, -122.4194  # 舊金山

# 步驟 1：將座標解析為地點
dcid = client.resolve.fetch_dcid_by_coordinates(
    latitude=latitude,
    longitude=longitude
)

# 步驟 2：取得地點名稱
name = client.node.fetch_entity_names(node_dcids=[dcid])
print(f"位置：{name[dcid]}")

# 步驟 3：檢查可用變數
available_vars = client.observation.fetch_available_statistical_variables(
    entity_dcids=[dcid]
)

print(f"\n可用變數：找到 {len(available_vars[dcid])} 個")
print("前 10 個：", list(available_vars[dcid])[:10])

# 步驟 4：查詢特定變數
response = client.observation.fetch(
    variable_dcids=["Count_Person", "Median_Income_Household"],
    entity_dcids=[dcid],
    date="latest"
)

# 顯示結果
df = response.to_observations_as_records()
print("\n統計資料：")
for _, row in df.iterrows():
    print(f"{row['variable']}: {row['value']}")
```

## 範例 6：資料來源篩選

從特定來源查詢資料以確保一致性：

```python
client = DataCommonsClient()

# 使用 facet 篩選查詢
response = client.observation.fetch(
    variable_dcids=["Count_Person"],
    entity_dcids=["country/USA"],
    date="all",
    filter_facet_domains=["census.gov"]  # 僅使用美國人口普查資料
)

df = response.to_observations_as_records()
print(f"從 census.gov 找到 {len(df)} 筆觀測值")

# 與所有來源比較
response_all = client.observation.fetch(
    variable_dcids=["Count_Person"],
    entity_dcids=["country/USA"],
    date="all"
)

df_all = response_all.to_observations_as_records()
print(f"從所有來源找到 {len(df_all)} 筆觀測值")
```

## 範例 7：探索知識圖譜

發現實體屬性和關係：

```python
client = DataCommonsClient()

# 步驟 1：探索存在哪些屬性
entity = "geoId/06"  # 加州

# 取得傳出屬性
out_props = client.node.fetch_property_labels(
    node_dcids=[entity],
    out=True
)

print(f"加州的傳出屬性：")
print(out_props[entity])

# 取得傳入屬性
in_props = client.node.fetch_property_labels(
    node_dcids=[entity],
    out=False
)

print(f"\n加州的傳入屬性：")
print(in_props[entity])

# 步驟 2：取得特定屬性值
name_response = client.node.fetch_property_values(
    node_dcids=[entity],
    property="name",
    out=True
)

print(f"\n名稱屬性值：")
print(name_response.to_dict())

# 步驟 3：探索階層
children = client.node.fetch_place_children(node_dcids=[entity])
print(f"\n子地點數量：{len(children[entity])}")

# 取得前 5 個子項目的名稱
if children[entity]:
    child_sample = children[entity][:5]
    child_names = client.node.fetch_entity_names(node_dcids=child_sample)
    print("\n子地點範例：")
    for dcid, name in child_names.items():
        print(f"  {name}")
```

## 範例 8：批次處理多個查詢

高效查詢多個實體的資料：

```python
import pandas as pd

client = DataCommonsClient()

# 要分析的城市列表
cities = [
    "San Francisco, CA",
    "Los Angeles, CA",
    "San Diego, CA",
    "Sacramento, CA",
    "San Jose, CA"
]

# 解析所有城市
resolve_response = client.resolve.fetch_dcids_by_name(
    names=cities,
    entity_type="City"
)

# 建立對應
city_dcids = []
dcid_to_name = {}
for name, result in resolve_response.to_dict().items():
    if result["candidates"]:
        dcid = result["candidates"][0]["dcid"]
        city_dcids.append(dcid)
        dcid_to_name[dcid] = name

# 一次查詢多個變數
variables = [
    "Count_Person",
    "Median_Income_Household",
    "UnemploymentRate_Person"
]

response = client.observation.fetch(
    variable_dcids=variables,
    entity_dcids=city_dcids,
    date="latest"
)

# 處理成比較表
df = response.to_observations_as_records()
df['city'] = df['entity'].map(dcid_to_name)

# 建立比較表
comparison = df.pivot_table(
    values='value',
    index='city',
    columns='variable',
    aggfunc='first'
)

print("\n加州城市比較：")
print(comparison.to_string())

# 匯出為 CSV
comparison.to_csv('ca_cities_comparison.csv')
print("\n資料已匯出至 ca_cities_comparison.csv")
```

## 常見模式摘要

### 模式 1：名稱 → DCID → 資料
```python
names = ["California"]
dcids = resolve_names(names)
data = query_observations(dcids, variables)
```

### 模式 2：座標 → DCID → 資料
```python
dcid = resolve_coordinates(lat, lon)
data = query_observations([dcid], variables)
```

### 模式 3：父項目 → 子項目 → 資料
```python
children = get_place_children(parent_dcid)
data = query_observations(children, variables)
```

### 模式 4：探索 → 選擇 → 查詢
```python
available_vars = check_available_variables(dcids)
selected_vars = filter_relevant(available_vars)
data = query_observations(dcids, selected_vars)
```

## 錯誤處理最佳實踐

```python
client = DataCommonsClient()

# 始終檢查候選項
resolve_response = client.resolve.fetch_dcids_by_name(names=["Unknown Place"])
result = resolve_response.to_dict()["Unknown Place"]

if not result["candidates"]:
    print("找不到匹配項 - 嘗試更具體的名稱")
    # 適當處理錯誤
else:
    dcid = result["candidates"][0]["dcid"]
    # 繼續查詢

# 檢查多個候選項（模糊）
if len(result["candidates"]) > 1:
    print(f"找到多個匹配項：{len(result['candidates'])}")
    for candidate in result["candidates"]:
        print(f"  {candidate['dcid']} ({candidate.get('dominantType', 'N/A')})")
    # 讓使用者選擇或使用額外篩選
```

## 下一步

1. 探索可用的統計變數：https://datacommons.org/tools/statvar
2. 瀏覽知識圖譜：https://datacommons.org/browser/
3. 閱讀 `references/` 目錄中的詳細端點文件
4. 查看官方文件：https://docs.datacommons.org/api/python/v2/
