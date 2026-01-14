# 空間分析

## 屬性連接

使用標準 pandas merge 根據共同變數合併資料集：

```python
# 在共同欄位上合併
result = gdf.merge(df, on='common_column')

# 左連接
result = gdf.merge(df, on='common_column', how='left')

# 重要：在 GeoDataFrame 上呼叫 merge 以保留幾何
# 正確：gdf.merge(df, ...)
# 錯誤：df.merge(gdf, ...) # 返回 DataFrame，而非 GeoDataFrame
```

## 空間連接

根據空間關係合併資料集。

### 二元謂詞連接（sjoin）

根據幾何謂詞連接：

```python
# 相交（預設）
joined = gpd.sjoin(gdf1, gdf2, how='inner', predicate='intersects')

# 可用的謂詞
joined = gpd.sjoin(gdf1, gdf2, predicate='contains')
joined = gpd.sjoin(gdf1, gdf2, predicate='within')
joined = gpd.sjoin(gdf1, gdf2, predicate='touches')
joined = gpd.sjoin(gdf1, gdf2, predicate='crosses')
joined = gpd.sjoin(gdf1, gdf2, predicate='overlaps')

# 連接類型
joined = gpd.sjoin(gdf1, gdf2, how='left')   # 保留左側所有
joined = gpd.sjoin(gdf1, gdf2, how='right')  # 保留右側所有
joined = gpd.sjoin(gdf1, gdf2, how='inner')  # 僅交集
```

`how` 參數決定保留哪些幾何：
- **left**：保留左側 GeoDataFrame 的索引和幾何
- **right**：保留右側 GeoDataFrame 的索引和幾何
- **inner**：使用索引的交集，保留左側幾何

### 最近鄰連接（sjoin_nearest）

連接到最近的圖徵：

```python
# 找到最近鄰
nearest = gpd.sjoin_nearest(gdf1, gdf2)

# 添加距離欄位
nearest = gpd.sjoin_nearest(gdf1, gdf2, distance_col='distance')

# 限制搜尋半徑（顯著提高效能）
nearest = gpd.sjoin_nearest(gdf1, gdf2, max_distance=1000)

# 找到 k 個最近鄰
nearest = gpd.sjoin_nearest(gdf1, gdf2, k=5)
```

## 疊加操作

結合兩個 GeoDataFrame 幾何的集合論操作：

```python
# 交集 - 保留兩者重疊的區域
intersection = gpd.overlay(gdf1, gdf2, how='intersection')

# 聯集 - 合併所有區域
union = gpd.overlay(gdf1, gdf2, how='union')

# 差集 - 第一個中不在第二個中的區域
difference = gpd.overlay(gdf1, gdf2, how='difference')

# 對稱差集 - 在任一個中但不在兩者中的區域
sym_diff = gpd.overlay(gdf1, gdf2, how='symmetric_difference')

# 恆等 - 交集 + 差集
identity = gpd.overlay(gdf1, gdf2, how='identity')
```

結果包含兩個輸入 GeoDataFrame 的屬性。

## 溶解（彙總）

根據屬性值彙總幾何：

```python
# 按屬性溶解
dissolved = gdf.dissolve(by='region')

# 帶彙總函數的溶解
dissolved = gdf.dissolve(by='region', aggfunc='sum')
dissolved = gdf.dissolve(by='region', aggfunc={'population': 'sum', 'area': 'mean'})

# 將所有溶解為單一幾何
dissolved = gdf.dissolve()

# 保留內部邊界
dissolved = gdf.dissolve(by='region', as_index=False)
```

## 裁剪

將幾何裁剪到另一個幾何的邊界：

```python
# 裁剪到多邊形邊界
clipped = gpd.clip(gdf, boundary_polygon)

# 裁剪到另一個 GeoDataFrame
clipped = gpd.clip(gdf, boundary_gdf)
```

## 附加

合併多個 GeoDataFrame：

```python
import pandas as pd

# 連接 GeoDataFrame（CRS 必須匹配）
combined = pd.concat([gdf1, gdf2], ignore_index=True)

# 帶識別鍵
combined = pd.concat([gdf1, gdf2], keys=['source1', 'source2'])
```

## 空間索引

提高空間操作的效能：

```python
# GeoPandas 自動為大多數操作使用空間索引
# 直接存取空間索引
sindex = gdf.sindex

# 查詢與邊界框相交的幾何
possible_matches_index = list(sindex.intersection((xmin, ymin, xmax, ymax)))
possible_matches = gdf.iloc[possible_matches_index]

# 查詢與多邊形相交的幾何
possible_matches_index = list(sindex.query(polygon_geometry))
possible_matches = gdf.iloc[possible_matches_index]
```

空間索引顯著加速：
- 空間連接
- 疊加操作
- 帶幾何謂詞的查詢

## 距離計算

```python
# 幾何之間的距離
distances = gdf1.geometry.distance(gdf2.geometry)

# 到單一幾何的距離
distances = gdf.geometry.distance(single_point)

# 到任何圖徵的最小距離
min_dist = gdf.geometry.distance(point).min()
```

## 面積和長度計算

為了準確測量，確保正確的 CRS：

```python
# 重新投影到適當的投影 CRS 進行面積/長度計算
gdf_projected = gdf.to_crs(epsg=3857)  # 或適當的 UTM 帶

# 計算面積（以 CRS 單位，通常是平方公尺）
areas = gdf_projected.geometry.area

# 計算長度/周長（以 CRS 單位）
lengths = gdf_projected.geometry.length
```
