# Node 端點 - 知識圖譜探索

## 用途

Node 端點從 Data Commons 知識圖譜中擷取屬性關係和值。它回傳連接節點的有向邊（屬性）資訊，能夠發現圖譜結構中的連接。

## 核心功能

Node API 執行三項主要功能：
1. 擷取與節點關聯的屬性標籤
2. 取得節點間特定屬性的值
3. 發現透過關係連接的所有節點

## 可用方法

### 1. fetch()

使用箭頭表示法的關係表達式擷取屬性。

**主要參數：**
- `node_dcids`：目標節點識別碼
- `expression`：使用箭頭的關係語法（`->`、`<-`、`<-*`）
- `all_pages`：啟用分頁（預設：True）
- `next_token`：繼續分頁結果

**箭頭表示法：**
- `->`：傳出屬性（從節點到值）
- `<-`：傳入屬性（從值到節點）
- `<-*`：多跳傳入遍歷

**使用範例：**
```python
from datacommons_client import DataCommonsClient

client = DataCommonsClient()

# 從加州取得傳出屬性
response = client.node.fetch(
    node_dcids=["geoId/06"],
    expression="->name"
)

# 取得傳入屬性（什麼指向此節點）
response = client.node.fetch(
    node_dcids=["geoId/06"],
    expression="<-containedInPlace"
)
```

### 2. fetch_property_labels()

取得屬性標籤而不擷取值——適用於發現存在哪些屬性。

**參數：**
- `node_dcids`：節點識別碼
- `out`：布林值——True 表示傳出屬性，False 表示傳入

**使用範例：**
```python
# 取得加州所有傳出屬性標籤
labels = client.node.fetch_property_labels(
    node_dcids=["geoId/06"],
    out=True
)

# 取得所有傳入屬性標籤
labels = client.node.fetch_property_labels(
    node_dcids=["geoId/06"],
    out=False
)
```

### 3. fetch_property_values()

取得具有可選篩選器的特定屬性值。

**參數：**
- `node_dcids`：節點識別碼
- `property`：要查詢的屬性名稱
- `out`：方向（True 表示傳出，False 表示傳入）
- `limit`：要回傳的最大值數量

**使用範例：**
```python
# 取得加州的名稱屬性
values = client.node.fetch_property_values(
    node_dcids=["geoId/06"],
    property="name",
    out=True
)
```

### 4. fetch_all_classes()

列出 Data Commons 圖譜中的所有實體類型（Class 節點）。

**使用範例：**
```python
classes = client.node.fetch_all_classes()
```

### 5. fetch_entity_names()

按 DCID 查詢所選語言的實體名稱。

**參數：**
- `node_dcids`：實體識別碼
- `language`：語言代碼（預設："en"）

**使用範例：**
```python
names = client.node.fetch_entity_names(
    node_dcids=["geoId/06", "country/USA"],
    language="en"
)
# 回傳：{"geoId/06": "California", "country/USA": "United States"}
```

### 6. 地點階層方法

這些方法瀏覽地理關係：

#### fetch_place_children()
取得直接子地點。

**使用範例：**
```python
# 取得美國的所有州
children = client.node.fetch_place_children(
    node_dcids=["country/USA"]
)
```

#### fetch_place_descendants()
擷取完整的子階層（遞迴）。

**使用範例：**
```python
# 取得加州的所有後代（縣、城市等）
descendants = client.node.fetch_place_descendants(
    node_dcids=["geoId/06"]
)
```

#### fetch_place_parents()
取得直接父地點。

**使用範例：**
```python
# 取得舊金山的父級
parents = client.node.fetch_place_parents(
    node_dcids=["geoId/0667000"]
)
```

#### fetch_place_ancestors()
擷取完整的父系譜系。

**使用範例：**
```python
# 取得舊金山的所有祖先（加州、美國等）
ancestors = client.node.fetch_place_ancestors(
    node_dcids=["geoId/0667000"]
)
```

### 7. fetch_statvar_constraints()

存取統計變數的約束屬性——適用於理解變數定義和約束。

**使用範例：**
```python
constraints = client.node.fetch_statvar_constraints(
    node_dcids=["Count_Person"]
)
```

## 回應格式

方法回傳以下其中之一：
- **NodeResponse 物件**，具有 `.to_dict()`、`.to_json()` 和 `.nextToken` 屬性
- **字典**，用於實體名稱和地點階層方法

## 分頁

對於大型回應：
1. 設定 `all_pages=False` 以分塊接收資料
2. 回應包含 `nextToken` 值
3. 使用該 token 重新查詢以取得後續頁面

**範例：**
```python
# 第一頁
response = client.node.fetch(
    node_dcids=["country/USA"],
    expression="<-containedInPlace",
    all_pages=False
)

# 如果有下一頁則取得
if response.nextToken:
    next_response = client.node.fetch(
        node_dcids=["country/USA"],
        expression="<-containedInPlace",
        next_token=response.nextToken
    )
```

## 常見使用案例

### 使用案例 1：探索可用屬性

```python
# 發現實體有哪些屬性
labels = client.node.fetch_property_labels(
    node_dcids=["geoId/06"],
    out=True
)
print(labels)  # 顯示所有傳出屬性如 'name'、'latitude' 等
```

### 使用案例 2：瀏覽地理階層

```python
# 取得加州的所有縣
counties = client.node.fetch_place_children(
    node_dcids=["geoId/06"]
)

# 如有需要，篩選特定類型
county_dcids = [child for child in counties["geoId/06"]
                if "County" in child]
```

### 使用案例 3：建立實體關係

```python
# 尋找所有引用特定節點的實體
references = client.node.fetch(
    node_dcids=["geoId/06"],
    expression="<-location"
)
```

## 重要注意事項

- 先使用 `fetch_property_labels()` 發現可用屬性
- Node API 無法解析複雜的關係表達式——使用更簡單的表達式或分成多個查詢
- 對於具有弧關係的連結實體屬性，將 Node API 查詢與 Observation API 結合
- 地點階層方法回傳字典，而非 NodeResponse 物件
