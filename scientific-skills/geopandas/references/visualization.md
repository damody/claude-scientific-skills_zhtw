# 地圖製作和視覺化

GeoPandas 透過 matplotlib 整合提供繪圖功能。

## 基本繪圖

```python
# 簡單繪圖
gdf.plot()

# 自訂圖形大小
gdf.plot(figsize=(10, 10))

# 設定顏色
gdf.plot(color='blue', edgecolor='black')

# 控制線寬
gdf.plot(edgecolor='black', linewidth=0.5)
```

## 分級著色地圖（Choropleth Maps）

根據資料值為圖徵著色：

```python
# 基本分級著色地圖
gdf.plot(column='population', legend=True)

# 指定色彩映射
gdf.plot(column='population', cmap='OrRd', legend=True)

# 其他色彩映射：'viridis'、'plasma'、'inferno'、'YlOrRd'、'Blues'、'Greens'
```

### 分類方案

需要：`uv pip install mapclassify`

```python
# 分位數
gdf.plot(column='population', scheme='quantiles', k=5, legend=True)

# 等距
gdf.plot(column='population', scheme='equal_interval', k=5, legend=True)

# 自然斷點（Fisher-Jenks）
gdf.plot(column='population', scheme='fisher_jenks', k=5, legend=True)

# 其他方案：'box_plot'、'headtail_breaks'、'max_breaks'、'std_mean'

# 傳遞參數給分類
gdf.plot(column='population', scheme='quantiles', k=7,
         classification_kwds={'pct': [10, 20, 30, 40, 50, 60, 70, 80, 90]})
```

### 圖例自訂

```python
# 將圖例放置在圖外
gdf.plot(column='population', legend=True,
         legend_kwds={'loc': 'upper left', 'bbox_to_anchor': (1, 1)})

# 水平圖例
gdf.plot(column='population', legend=True,
         legend_kwds={'orientation': 'horizontal'})

# 自訂圖例標籤
gdf.plot(column='population', legend=True,
         legend_kwds={'label': 'Population Count'})

# 為色條使用單獨的軸
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
gdf.plot(column='population', ax=ax, legend=True, cax=cax)
```

## 處理缺失資料

```python
# 設定缺失值樣式
gdf.plot(column='population',
         missing_kwds={'color': 'lightgrey', 'edgecolor': 'red', 'hatch': '///',
                      'label': 'Missing data'})
```

## 多圖層地圖

合併多個 GeoDataFrame：

```python
import matplotlib.pyplot as plt

# 建立基礎圖
fig, ax = plt.subplots(figsize=(10, 10))

# 添加圖層
gdf1.plot(ax=ax, color='lightblue', edgecolor='black')
gdf2.plot(ax=ax, color='red', markersize=5)
gdf3.plot(ax=ax, color='green', alpha=0.5)

plt.show()

# 使用 zorder 控制圖層順序（較高 = 在上面）
gdf1.plot(ax=ax, zorder=1)
gdf2.plot(ax=ax, zorder=2)
```

## 樣式選項

```python
# 透明度
gdf.plot(alpha=0.5)

# 點的標記樣式
points.plot(marker='o', markersize=50)
points.plot(marker='^', markersize=100, color='red')

# 線條樣式
lines.plot(linestyle='--', linewidth=2)
lines.plot(linestyle=':', color='blue')

# 分類著色
gdf.plot(column='category', categorical=True, legend=True)

# 按欄位變化標記大小
gdf.plot(markersize=gdf['value']/1000)
```

## 地圖增強

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 8))
gdf.plot(ax=ax, column='population', legend=True)

# 添加標題
ax.set_title('Population by Region', fontsize=16)

# 添加軸標籤
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# 移除軸
ax.set_axis_off()

# 添加指北針和比例尺（需要單獨的套件）
# 參見 geopandas-plot 或 contextily 取得這些功能

plt.tight_layout()
plt.show()
```

## 互動式地圖

需要：`uv pip install folium`

```python
# 建立互動式地圖
m = gdf.explore(column='population', cmap='YlOrRd', legend=True)
m.save('map.html')

# 自訂底圖
m = gdf.explore(tiles='OpenStreetMap', legend=True)
m = gdf.explore(tiles='CartoDB positron', legend=True)

# 添加工具提示
m = gdf.explore(column='population', tooltip=['name', 'population'], legend=True)

# 樣式選項
m = gdf.explore(color='red', style_kwds={'fillOpacity': 0.5, 'weight': 2})

# 多圖層
m = gdf1.explore(color='blue', name='Layer 1')
gdf2.explore(m=m, color='red', name='Layer 2')
folium.LayerControl().add_to(m)
```

## 與其他圖表類型整合

GeoPandas 支援 pandas 圖表類型：

```python
# 屬性的直方圖
gdf['population'].plot.hist(bins=20)

# 散點圖
gdf.plot.scatter(x='income', y='population')

# 箱形圖
gdf.boxplot(column='population', by='region')
```

## 使用 Contextily 的底圖

需要：`uv pip install contextily`

```python
import contextily as ctx

# 重新投影到 Web Mercator 以相容底圖
gdf_webmercator = gdf.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(10, 10))
gdf_webmercator.plot(ax=ax, alpha=0.5, edgecolor='k')

# 添加底圖
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
# 其他來源：ctx.providers.CartoDB.Positron、ctx.providers.Stamen.Terrain

plt.show()
```

## 使用 CartoPy 的製圖投影

需要：`uv pip install cartopy`

```python
import cartopy.crs as ccrs

# 建立具有特定投影的地圖
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.Robinson()}, figsize=(15, 10))

gdf.plot(ax=ax, transform=ccrs.PlateCarree(), column='population', legend=True)

ax.coastlines()
ax.gridlines(draw_labels=True)

plt.show()
```

## 儲存圖形

```python
# 儲存到檔案
ax = gdf.plot()
fig = ax.get_figure()
fig.savefig('map.png', dpi=300, bbox_inches='tight')
fig.savefig('map.pdf')
fig.savefig('map.svg')
```
