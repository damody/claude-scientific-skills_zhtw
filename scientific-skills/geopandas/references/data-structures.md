# GeoPandas 資料結構

## GeoSeries

GeoSeries 是一個向量，其中每個條目是對應於一個觀測值的一組形狀（類似於 pandas Series 但具有幾何資料）。

```python
import geopandas as gpd
from shapely.geometry import Point, Polygon

# 從幾何建立 GeoSeries
points = gpd.GeoSeries([Point(1, 1), Point(2, 2), Point(3, 3)])

# 存取幾何屬性
points.area
points.length
points.bounds
```

## GeoDataFrame

GeoDataFrame 是包含 GeoSeries 的表格資料結構（類似於 pandas DataFrame 但具有地理資料）。

```python
# 從字典建立
gdf = gpd.GeoDataFrame({
    'name': ['Point A', 'Point B'],
    'value': [100, 200],
    'geometry': [Point(1, 1), Point(2, 2)]
})

# 從具有座標的 pandas DataFrame 建立
import pandas as pd
df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3], 'name': ['A', 'B', 'C']})
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))
```

## 關鍵屬性

- **geometry**：活動幾何欄位（可以有多個幾何欄位）
- **crs**：座標參考系統
- **bounds**：所有幾何的邊界框
- **total_bounds**：整體邊界框

## 設定活動幾何

當 GeoDataFrame 有多個幾何欄位時：

```python
# 設定活動幾何欄位
gdf = gdf.set_geometry('other_geom_column')

# 檢查活動幾何欄位
gdf.geometry.name
```

## 索引和選取

使用標準 pandas 索引處理空間資料：

```python
# 按標籤選取
gdf.loc[0]

# 布林索引
large_areas = gdf[gdf.area > 100]

# 選取欄位
gdf[['name', 'geometry']]
```
