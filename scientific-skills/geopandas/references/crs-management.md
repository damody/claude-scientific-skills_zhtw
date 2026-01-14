# 座標參考系統（CRS）

座標參考系統定義座標如何對應到地球上的位置。

## 理解 CRS

CRS 資訊儲存為 `pyproj.CRS` 物件：

```python
# 檢查 CRS
print(gdf.crs)

# 檢查是否已設定 CRS
if gdf.crs is None:
    print("No CRS defined")
```

## 設定 vs 重新投影

### 設定 CRS

當座標正確但缺少 CRS 元資料時使用 `set_crs()`：

```python
# 設定 CRS（不轉換座標）
gdf = gdf.set_crs("EPSG:4326")
gdf = gdf.set_crs(4326)
```

**警告**：僅在缺少 CRS 元資料時使用。這不會轉換座標。

### 重新投影

使用 `to_crs()` 在座標系統之間轉換座標：

```python
# 重新投影到不同的 CRS
gdf_projected = gdf.to_crs("EPSG:3857")  # Web Mercator
gdf_projected = gdf.to_crs(3857)

# 重新投影以匹配另一個 GeoDataFrame
gdf1_reprojected = gdf1.to_crs(gdf2.crs)
```

## CRS 格式

GeoPandas 透過 `pyproj.CRS.from_user_input()` 接受多種格式：

```python
# EPSG 代碼（整數）
gdf.to_crs(4326)

# 授權字串
gdf.to_crs("EPSG:4326")
gdf.to_crs("ESRI:102003")

# WKT 字串（Well-Known Text）
gdf.to_crs("GEOGCS[...]")

# PROJ 字串
gdf.to_crs("+proj=longlat +datum=WGS84")

# pyproj.CRS 物件
from pyproj import CRS
crs_obj = CRS.from_epsg(4326)
gdf.to_crs(crs_obj)
```

**最佳實踐**：使用 WKT2 或授權字串（EPSG）以保留完整的 CRS 資訊。

## 常用 EPSG 代碼

### 地理座標系統

```python
# WGS 84（緯度/經度）
gdf.to_crs("EPSG:4326")

# NAD83
gdf.to_crs("EPSG:4269")
```

### 投影座標系統

```python
# Web Mercator（網頁地圖使用）
gdf.to_crs("EPSG:3857")

# UTM 帶（範例：UTM 33N 帶）
gdf.to_crs("EPSG:32633")

# UTM 帶（南半球，範例：UTM 33S 帶）
gdf.to_crs("EPSG:32733")

# 美國國家地圖等面積
gdf.to_crs("ESRI:102003")

# 阿爾伯斯等面積圓錐投影（北美）
gdf.to_crs("EPSG:5070")
```

## 操作對 CRS 的要求

### 需要匹配 CRS 的操作

這些操作需要相同的 CRS：

```python
# 空間連接
gpd.sjoin(gdf1, gdf2, ...)  # CRS 必須匹配

# 疊加操作
gpd.overlay(gdf1, gdf2, ...)  # CRS 必須匹配

# 附加
pd.concat([gdf1, gdf2])  # CRS 必須匹配

# 如有需要先重新投影
gdf2_reprojected = gdf2.to_crs(gdf1.crs)
result = gpd.sjoin(gdf1, gdf2_reprojected)
```

### 最好在投影 CRS 中進行的操作

面積和距離計算應使用投影 CRS：

```python
# 錯誤：以度為單位的面積（無意義）
areas_degrees = gdf.geometry.area  # 如果 CRS 是 EPSG:4326

# 正確：先重新投影到適當的投影 CRS
gdf_projected = gdf.to_crs("EPSG:3857")
areas_meters = gdf_projected.geometry.area  # 平方公尺

# 更好：使用適當的本地 UTM 帶以獲得準確度
gdf_utm = gdf.to_crs("EPSG:32633")  # UTM 33N 帶
accurate_areas = gdf_utm.geometry.area
```

## 選擇適當的 CRS

### 用於面積/距離計算

使用等面積投影：

```python
# 阿爾伯斯等面積圓錐投影（北美）
gdf.to_crs("EPSG:5070")

# 蘭伯特方位角等面積
gdf.to_crs("EPSG:3035")  # 歐洲

# UTM 帶（用於局部區域）
gdf.to_crs("EPSG:32633")  # 適當的 UTM 帶
```

### 用於距離保持（導航）

使用等距投影：

```python
# 方位角等距投影
gdf.to_crs("ESRI:54032")
```

### 用於形狀保持（角度）

使用正形投影：

```python
# Web Mercator（正形但扭曲面積）
gdf.to_crs("EPSG:3857")

# UTM 帶（局部區域的正形）
gdf.to_crs("EPSG:32633")
```

### 用於網頁製圖

```python
# Web Mercator（網頁地圖的標準）
gdf.to_crs("EPSG:3857")
```

## 估計 UTM 帶

```python
# 從資料估計適當的 UTM CRS
utm_crs = gdf.estimate_utm_crs()
gdf_utm = gdf.to_crs(utm_crs)
```

## 具有不同 CRS 的多幾何欄位

GeoPandas 0.8+ 支援每個幾何欄位不同的 CRS：

```python
# 為特定幾何欄位設定 CRS
gdf = gdf.set_crs("EPSG:4326", allow_override=True)

# 活動幾何決定操作
gdf = gdf.set_geometry('other_geom_column')

# 檢查 CRS 不匹配
try:
    result = gdf1.overlay(gdf2)
except ValueError as e:
    print("CRS mismatch:", e)
```

## CRS 資訊

```python
# 取得完整的 CRS 詳情
print(gdf.crs)

# 取得 EPSG 代碼（如果可用）
print(gdf.crs.to_epsg())

# 取得 WKT 表示
print(gdf.crs.to_wkt())

# 取得 PROJ 字串
print(gdf.crs.to_proj4())

# 檢查 CRS 是否為地理的（經緯度）
print(gdf.crs.is_geographic)

# 檢查 CRS 是否為投影的
print(gdf.crs.is_projected)
```

## 轉換個別幾何

```python
from pyproj import Transformer

# 建立轉換器
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# 轉換點
x_new, y_new = transformer.transform(x, y)
```
