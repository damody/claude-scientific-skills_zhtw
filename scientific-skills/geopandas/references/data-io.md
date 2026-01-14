# 讀取和寫入空間資料

## 讀取檔案

使用 `geopandas.read_file()` 匯入向量空間資料：

```python
import geopandas as gpd

# 從檔案讀取
gdf = gpd.read_file("data.shp")
gdf = gpd.read_file("data.geojson")
gdf = gpd.read_file("data.gpkg")

# 從 URL 讀取
gdf = gpd.read_file("https://example.com/data.geojson")

# 從 ZIP 壓縮檔讀取
gdf = gpd.read_file("data.zip")
```

### 效能：Arrow 加速

使用 Arrow 可獲得 2-4 倍更快的讀取速度：

```python
gdf = gpd.read_file("data.gpkg", use_arrow=True)
```

需要 PyArrow：`uv pip install pyarrow`

### 讀取時過濾

預過濾資料以僅載入所需內容：

```python
# 載入特定行
gdf = gpd.read_file("data.gpkg", rows=100)  # 前 100 行
gdf = gpd.read_file("data.gpkg", rows=slice(10, 20))  # 第 10-20 行

# 載入特定欄位
gdf = gpd.read_file("data.gpkg", columns=['name', 'population'])

# 使用邊界框空間過濾
gdf = gpd.read_file("data.gpkg", bbox=(xmin, ymin, xmax, ymax))

# 使用幾何遮罩空間過濾
gdf = gpd.read_file("data.gpkg", mask=polygon_geometry)

# SQL WHERE 子句（需要 Fiona 1.9+ 或 Pyogrio）
gdf = gpd.read_file("data.gpkg", where="population > 1000000")

# 跳過幾何（返回 pandas DataFrame）
df = gpd.read_file("data.gpkg", ignore_geometry=True)
```

## 寫入檔案

使用 `to_file()` 匯出：

```python
# 寫入 Shapefile
gdf.to_file("output.shp")

# 寫入 GeoJSON
gdf.to_file("output.geojson", driver='GeoJSON')

# 寫入 GeoPackage（支援多圖層）
gdf.to_file("output.gpkg", layer='layer1', driver="GPKG")

# Arrow 加速以更快寫入
gdf.to_file("output.gpkg", use_arrow=True)
```

### 支援的格式

列出所有可用的驅動程式：

```python
import pyogrio
pyogrio.list_drivers()
```

常見格式：Shapefile、GeoJSON、GeoPackage (GPKG)、KML、MapInfo File、CSV（含 WKT 幾何）

## Parquet 和 Feather

支援多幾何欄位並保留空間資訊的欄式格式：

```python
# 寫入
gdf.to_parquet("data.parquet")
gdf.to_feather("data.feather")

# 讀取
gdf = gpd.read_parquet("data.parquet")
gdf = gpd.read_feather("data.feather")
```

優點：
- 比傳統格式更快的 I/O
- 更好的壓縮
- 保留多幾何欄位
- 支援 schema 版本控制

## PostGIS 資料庫

### 從 PostGIS 讀取

```python
from sqlalchemy import create_engine

engine = create_engine('postgresql://user:password@host:port/database')

# 讀取整個表格
gdf = gpd.read_postgis("SELECT * FROM table_name", con=engine, geom_col='geometry')

# 使用 SQL 查詢讀取
gdf = gpd.read_postgis("SELECT * FROM table WHERE population > 100000", con=engine, geom_col='geometry')
```

### 寫入 PostGIS

```python
# 建立或取代表格
gdf.to_postgis("table_name", con=engine, if_exists='replace')

# 附加到現有表格
gdf.to_postgis("table_name", con=engine, if_exists='append')

# 如果表格存在則失敗
gdf.to_postgis("table_name", con=engine, if_exists='fail')
```

需要：`uv pip install psycopg2` 或 `uv pip install psycopg` 和 `uv pip install geoalchemy2`

## 類檔案物件

從檔案控制代碼或記憶體緩衝區讀取：

```python
# 從檔案控制代碼
with open('data.geojson', 'r') as f:
    gdf = gpd.read_file(f)

# 從 StringIO
from io import StringIO
geojson_string = '{"type": "FeatureCollection", ...}'
gdf = gpd.read_file(StringIO(geojson_string))
```

## 遠端儲存（fsspec）

從雲端儲存存取資料：

```python
# S3
gdf = gpd.read_file("s3://bucket/data.gpkg")

# Azure Blob Storage
gdf = gpd.read_file("az://container/data.gpkg")

# HTTP/HTTPS
gdf = gpd.read_file("https://example.com/data.geojson")
```
