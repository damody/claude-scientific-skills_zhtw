# 幾何操作

GeoPandas 透過 Shapely 整合提供廣泛的幾何操作功能。

## 建構操作

從現有幾何建立新幾何：

### 緩衝區（Buffer）

建立表示指定距離內所有點的幾何：

```python
# 按固定距離緩衝
buffered = gdf.geometry.buffer(10)

# 負緩衝（侵蝕）
eroded = gdf.geometry.buffer(-5)

# 帶解析度參數的緩衝
smooth_buffer = gdf.geometry.buffer(10, resolution=16)
```

### 邊界（Boundary）

取得低維度邊界：

```python
# Polygon -> LineString, LineString -> MultiPoint
boundaries = gdf.geometry.boundary
```

### 質心（Centroid）

取得每個幾何的中心點：

```python
centroids = gdf.geometry.centroid
```

### 凸包（Convex Hull）

包含所有點的最小凸多邊形：

```python
hulls = gdf.geometry.convex_hull
```

### 凹包（Concave Hull）

包含所有點的最小凹多邊形：

```python
# ratio 參數控制凹度（0 = 凸包，1 = 最凹）
concave_hulls = gdf.geometry.concave_hull(ratio=0.5)
```

### 包絡矩形（Envelope）

最小軸對齊矩形：

```python
envelopes = gdf.geometry.envelope
```

### 簡化（Simplify）

減少幾何複雜度：

```python
# Douglas-Peucker 演算法與容差
simplified = gdf.geometry.simplify(tolerance=10)

# 保留拓撲（防止自相交）
simplified = gdf.geometry.simplify(tolerance=10, preserve_topology=True)
```

### 分段化（Segmentize）

為線段添加頂點：

```python
# 添加具有最大線段長度的頂點
segmented = gdf.geometry.segmentize(max_segment_length=5)
```

### 全部合併（Union All）

將所有幾何合併為單一幾何：

```python
# 合併所有圖徵
unified = gdf.geometry.union_all()
```

## 仿射變換

座標的數學變換：

### 旋轉（Rotate）

```python
# 繞原點 (0, 0) 按角度（度）旋轉
rotated = gdf.geometry.rotate(angle=45, origin='center')

# 繞自訂點旋轉
rotated = gdf.geometry.rotate(angle=45, origin=(100, 100))
```

### 縮放（Scale）

```python
# 均勻縮放
scaled = gdf.geometry.scale(xfact=2.0, yfact=2.0)

# 帶原點縮放
scaled = gdf.geometry.scale(xfact=2.0, yfact=2.0, origin='center')
```

### 平移（Translate）

```python
# 移動座標
translated = gdf.geometry.translate(xoff=100, yoff=50)
```

### 傾斜（Skew）

```python
# 剪切變換
skewed = gdf.geometry.skew(xs=15, ys=0, origin='center')
```

### 自訂仿射變換

```python
from shapely import affinity

# 應用 6 參數仿射變換矩陣
# [a, b, d, e, xoff, yoff]
transformed = gdf.geometry.affine_transform([1, 0, 0, 1, 100, 50])
```

## 幾何屬性

存取幾何屬性（返回 pandas Series）：

```python
# 面積
areas = gdf.geometry.area

# 長度/周長
lengths = gdf.geometry.length

# 邊界框座標
bounds = gdf.geometry.bounds  # 返回包含 minx, miny, maxx, maxy 的 DataFrame

# 整個 GeoSeries 的總邊界
total_bounds = gdf.geometry.total_bounds  # 返回陣列 [minx, miny, maxx, maxy]

# 檢查幾何類型
geom_types = gdf.geometry.geom_type

# 檢查是否有效
is_valid = gdf.geometry.is_valid

# 檢查是否為空
is_empty = gdf.geometry.is_empty
```

## 幾何關係

測試關係的二元謂詞：

```python
# 在...之內
gdf1.geometry.within(gdf2.geometry)

# 包含
gdf1.geometry.contains(gdf2.geometry)

# 相交
gdf1.geometry.intersects(gdf2.geometry)

# 接觸
gdf1.geometry.touches(gdf2.geometry)

# 交叉
gdf1.geometry.crosses(gdf2.geometry)

# 重疊
gdf1.geometry.overlaps(gdf2.geometry)

# 覆蓋
gdf1.geometry.covers(gdf2.geometry)

# 被覆蓋
gdf1.geometry.covered_by(gdf2.geometry)
```

## 點提取

從幾何中提取特定點：

```python
# 代表點（保證在幾何內部）
rep_points = gdf.geometry.representative_point()

# 沿線在指定距離處內插點
points = line_gdf.geometry.interpolate(distance=10)

# 在標準化距離（0 到 1）處內插點
midpoints = line_gdf.geometry.interpolate(distance=0.5, normalized=True)
```

## Delaunay 三角剖分

```python
# 建立三角剖分
triangles = gdf.geometry.delaunay_triangles()
```
