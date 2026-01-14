# I/O 操作

本參考文件涵蓋 Vaex 中的檔案輸入/輸出操作、格式轉換、匯出策略，以及處理各種資料格式。

## 概述

Vaex 支援多種檔案格式，每種格式有不同的效能特性。格式的選擇顯著影響載入速度、記憶體使用和整體效能。

**格式建議：**
- **HDF5** - 最適合大多數使用案例（即時載入，記憶體映射）
- **Apache Arrow** - 最適合互通性（即時載入，欄式儲存）
- **Parquet** - 適合分散式系統（壓縮，欄式儲存）
- **CSV** - 避免用於大型資料集（載入緩慢，非記憶體映射）

## 讀取資料

### HDF5 檔案（推薦）

```python
import vaex

# 開啟 HDF5 檔案（即時，記憶體映射）
df = vaex.open('data.hdf5')

# 多個檔案作為單一 DataFrame
df = vaex.open('data_part*.hdf5')
df = vaex.open(['data_2020.hdf5', 'data_2021.hdf5', 'data_2022.hdf5'])
```

**優點：**
- 即時載入（記憶體映射，無資料讀入 RAM）
- Vaex 操作的最佳效能
- 支援壓縮
- 隨機存取模式

### Apache Arrow 檔案

```python
# 開啟 Arrow 檔案（即時，記憶體映射）
df = vaex.open('data.arrow')
df = vaex.open('data.feather')  # Feather 是 Arrow 格式

# 多個 Arrow 檔案
df = vaex.open('data_*.arrow')
```

**優點：**
- 即時載入（記憶體映射）
- 語言無關格式
- 適合資料共享
- 與 Arrow 生態系統零複製整合

### Parquet 檔案

```python
# 開啟 Parquet 檔案
df = vaex.open('data.parquet')

# 多個 Parquet 檔案
df = vaex.open('data_*.parquet')

# 從雲端儲存
df = vaex.open('s3://bucket/data.parquet')
df = vaex.open('gs://bucket/data.parquet')
```

**優點：**
- 預設壓縮
- 欄式格式
- 廣泛的生態系統支援
- 適合分散式系統

**注意事項：**
- 本地檔案比 HDF5/Arrow 慢
- 某些操作可能需要完整檔案讀取

### CSV 檔案

```python
# 簡單 CSV
df = vaex.from_csv('data.csv')

# 大型 CSV 自動分塊
df = vaex.from_csv('large_data.csv', chunk_size=5_000_000)

# CSV 轉換為 HDF5
df = vaex.from_csv('large_data.csv', convert='large_data.hdf5')
# 建立 HDF5 檔案供未來快速載入

# 帶選項的 CSV
df = vaex.from_csv(
    'data.csv',
    sep=',',
    header=0,
    names=['col1', 'col2', 'col3'],
    dtype={'col1': 'int64', 'col2': 'float64'},
    usecols=['col1', 'col2'],  # 只載入特定欄位
    nrows=100000  # 限制列數
)
```

**建議：**
- **總是將大型 CSV 轉換為 HDF5** 供重複使用
- 使用 `convert` 參數自動建立 HDF5
- 大型檔案的 CSV 載入可能需要大量時間

### FITS 檔案（天文學）

```python
# 開啟 FITS 檔案
df = vaex.open('astronomical_data.fits')

# 多個 FITS 檔案
df = vaex.open('survey_*.fits')
```

## 寫入/匯出資料

### 匯出為 HDF5

```python
# 匯出為 HDF5（Vaex 推薦）
df.export_hdf5('output.hdf5')

# 帶進度條
df.export_hdf5('output.hdf5', progress=True)

# 匯出欄位子集
df[['col1', 'col2', 'col3']].export_hdf5('subset.hdf5')

# 帶壓縮匯出
df.export_hdf5('compressed.hdf5', compression='gzip')
```

### 匯出為 Arrow

```python
# 匯出為 Arrow 格式
df.export_arrow('output.arrow')

# 匯出為 Feather（Arrow 格式）
df.export_feather('output.feather')
```

### 匯出為 Parquet

```python
# 匯出為 Parquet
df.export_parquet('output.parquet')

# 帶壓縮
df.export_parquet('output.parquet', compression='snappy')
df.export_parquet('output.parquet', compression='gzip')
```

### 匯出為 CSV

```python
# 匯出為 CSV（不建議用於大型資料）
df.export_csv('output.csv')

# 帶選項
df.export_csv(
    'output.csv',
    sep=',',
    header=True,
    index=False,
    chunk_size=1_000_000
)

# 匯出子集
df[df.age > 25].export_csv('filtered_output.csv')
```

## 格式轉換

### CSV 到 HDF5（最常見）

```python
import vaex

# 方法 1：讀取時自動轉換
df = vaex.from_csv('large.csv', convert='large.hdf5')
# 建立 large.hdf5，回傳指向它的 DataFrame

# 方法 2：明確轉換
df = vaex.from_csv('large.csv')
df.export_hdf5('large.hdf5')

# 未來載入（即時）
df = vaex.open('large.hdf5')
```

### HDF5 到 Arrow

```python
# 載入 HDF5
df = vaex.open('data.hdf5')

# 匯出為 Arrow
df.export_arrow('data.arrow')
```

### Parquet 到 HDF5

```python
# 載入 Parquet
df = vaex.open('data.parquet')

# 匯出為 HDF5
df.export_hdf5('data.hdf5')
```

### 多個 CSV 檔案到單一 HDF5

```python
import vaex
import glob

# 找到所有 CSV 檔案
csv_files = glob.glob('data_*.csv')

# 載入並串接
dfs = [vaex.from_csv(f) for f in csv_files]
df_combined = vaex.concat(dfs)

# 匯出為單一 HDF5
df_combined.export_hdf5('combined_data.hdf5')
```

## 增量/分塊 I/O

### 分塊處理大型 CSV

```python
import vaex

# 分塊處理 CSV
chunk_size = 1_000_000
output_file = 'processed.hdf5'

for i, df_chunk in enumerate(vaex.from_csv_chunked('huge.csv', chunk_size=chunk_size)):
    # 處理分塊
    df_chunk['new_col'] = df_chunk.x + df_chunk.y

    # 追加到 HDF5
    if i == 0:
        df_chunk.export_hdf5(output_file)
    else:
        df_chunk.export_hdf5(output_file, mode='a')  # 追加

# 載入最終結果
df = vaex.open(output_file)
```

### 分塊匯出

```python
# 分塊匯出大型 DataFrame（用於 CSV）
chunk_size = 1_000_000

for i in range(0, len(df), chunk_size):
    df_chunk = df[i:i+chunk_size]
    mode = 'w' if i == 0 else 'a'
    df_chunk.export_csv('large_output.csv', mode=mode, header=(i == 0))
```

## Pandas 整合

### 從 Pandas 到 Vaex

```python
import pandas as pd
import vaex

# 使用 pandas 讀取
pdf = pd.read_csv('data.csv')

# 轉換為 Vaex
df = vaex.from_pandas(pdf, copy_index=False)

# 為了更好的效能：直接使用 Vaex
df = vaex.from_csv('data.csv')  # 偏好
```

### 從 Vaex 到 Pandas

```python
# 完整轉換（大型資料要小心！）
pdf = df.to_pandas_df()

# 轉換子集
pdf = df[['col1', 'col2']].to_pandas_df()
pdf = df[:10000].to_pandas_df()  # 前 10k 列
pdf = df[df.age > 25].to_pandas_df()  # 篩選後

# 抽樣用於探索
pdf_sample = df.sample(n=10000).to_pandas_df()
```

## Arrow 整合

### 從 Arrow 到 Vaex

```python
import pyarrow as pa
import vaex

# 從 Arrow Table
arrow_table = pa.table({
    'a': [1, 2, 3],
    'b': [4, 5, 6]
})
df = vaex.from_arrow_table(arrow_table)

# 從 Arrow 檔案
arrow_table = pa.ipc.open_file('data.arrow').read_all()
df = vaex.from_arrow_table(arrow_table)
```

### 從 Vaex 到 Arrow

```python
# 轉換為 Arrow Table
arrow_table = df.to_arrow_table()

# 寫入 Arrow 檔案
import pyarrow as pa
with pa.ipc.new_file('output.arrow', arrow_table.schema) as writer:
    writer.write_table(arrow_table)

# 或使用 Vaex 匯出
df.export_arrow('output.arrow')
```

## 遠端和雲端儲存

### 從 S3 讀取

```python
import vaex

# 從 S3 讀取（需要 s3fs）
df = vaex.open('s3://bucket-name/data.parquet')
df = vaex.open('s3://bucket-name/data.hdf5')

# 使用憑證
import s3fs
fs = s3fs.S3FileSystem(key='access_key', secret='secret_key')
df = vaex.open('s3://bucket-name/data.parquet', fs=fs)
```

### 從 Google Cloud Storage 讀取

```python
# 從 GCS 讀取（需要 gcsfs）
df = vaex.open('gs://bucket-name/data.parquet')

# 使用憑證
import gcsfs
fs = gcsfs.GCSFileSystem(token='path/to/credentials.json')
df = vaex.open('gs://bucket-name/data.parquet', fs=fs)
```

### 從 Azure 讀取

```python
# 從 Azure Blob Storage 讀取（需要 adlfs）
df = vaex.open('az://container-name/data.parquet')
```

### 寫入雲端儲存

```python
# 匯出到 S3
df.export_parquet('s3://bucket-name/output.parquet')
df.export_hdf5('s3://bucket-name/output.hdf5')

# 匯出到 GCS
df.export_parquet('gs://bucket-name/output.parquet')
```

## 資料庫整合

### 從 SQL 資料庫讀取

```python
import vaex
import pandas as pd
from sqlalchemy import create_engine

# 使用 pandas 讀取，轉換為 Vaex
engine = create_engine('postgresql://user:password@host:port/database')
pdf = pd.read_sql('SELECT * FROM table', engine)
df = vaex.from_pandas(pdf)

# 對於大型表格：分塊讀取
chunks = []
for chunk in pd.read_sql('SELECT * FROM large_table', engine, chunksize=100000):
    chunks.append(vaex.from_pandas(chunk))
df = vaex.concat(chunks)

# 更好：從資料庫匯出為 CSV/Parquet，然後用 Vaex 載入
```

### 寫入 SQL 資料庫

```python
# 轉換為 pandas，然後寫入
pdf = df.to_pandas_df()
pdf.to_sql('table_name', engine, if_exists='replace', index=False)

# 對於大型資料：分塊寫入
chunk_size = 100000
for i in range(0, len(df), chunk_size):
    chunk = df[i:i+chunk_size].to_pandas_df()
    chunk.to_sql('table_name', engine,
                 if_exists='append' if i > 0 else 'replace',
                 index=False)
```

## 記憶體映射檔案

### 理解記憶體映射

```python
# HDF5 和 Arrow 檔案預設是記憶體映射的
df = vaex.open('data.hdf5')  # 沒有資料載入 RAM

# 資料按需從磁碟讀取
mean = df.x.mean()  # 串流處理資料，最小記憶體

# 檢查欄位是否是記憶體映射的
print(df.is_local('column_name'))  # False = 記憶體映射
```

### 強制載入資料到記憶體

```python
# 如果需要，將資料載入記憶體
df_in_memory = df.copy()
for col in df.get_column_names():
    df_in_memory[col] = df[col].values  # 實體化到記憶體
```

## 檔案壓縮

### HDF5 壓縮

```python
# 帶壓縮匯出
df.export_hdf5('compressed.hdf5', compression='gzip')
df.export_hdf5('compressed.hdf5', compression='lzf')
df.export_hdf5('compressed.hdf5', compression='blosc')

# 權衡：較小的檔案大小，稍慢的 I/O
```

### Parquet 壓縮

```python
# Parquet 預設是壓縮的
df.export_parquet('data.parquet', compression='snappy')  # 快速
df.export_parquet('data.parquet', compression='gzip')    # 更好的壓縮
df.export_parquet('data.parquet', compression='brotli')  # 最佳壓縮
```

## Vaex Server（遠端資料）

### 啟動 Vaex Server

```bash
# 啟動伺服器
vaex-server data.hdf5 --host 0.0.0.0 --port 9000
```

### 連接到遠端伺服器

```python
import vaex

# 連接到遠端 Vaex 伺服器
df = vaex.open('ws://hostname:9000/data')

# 操作透明運作
mean = df.x.mean()  # 在伺服器上計算
```

## 狀態檔案

### 儲存 DataFrame 狀態

```python
# 儲存狀態（包括虛擬欄、選擇等）
df.state_write('state.json')

# 包括：
# - 虛擬欄定義
# - 活動選擇
# - 變數
# - 轉換器（縮放器、編碼器、模型）
```

### 載入 DataFrame 狀態

```python
# 載入資料
df = vaex.open('data.hdf5')

# 套用儲存的狀態
df.state_load('state.json')

# 所有虛擬欄、選擇和轉換都已還原
```

## 最佳實務

### 1. 選擇正確的格式

```python
# 本地工作：HDF5
df.export_hdf5('data.hdf5')

# 共享/互通性：Arrow
df.export_arrow('data.arrow')

# 分散式系統：Parquet
df.export_parquet('data.parquet')

# 避免大型資料使用 CSV
```

### 2. 只轉換 CSV 一次

```python
# 一次性轉換
df = vaex.from_csv('large.csv', convert='large.hdf5')

# 所有未來載入
df = vaex.open('large.hdf5')  # 即時！
```

### 3. 匯出前實體化

```python
# 如果 DataFrame 有許多虛擬欄
df_materialized = df.materialize()
df_materialized.export_hdf5('output.hdf5')

# 更快的匯出和未來載入
```

### 4. 明智使用壓縮

```python
# 對於封存或不常存取的資料
df.export_hdf5('archived.hdf5', compression='gzip')

# 對於活躍工作（更快 I/O）
df.export_hdf5('working.hdf5')  # 不壓縮
```

### 5. 為長管線建立檢查點

```python
# 昂貴的預處理之後
df_preprocessed = preprocess(df)
df_preprocessed.export_hdf5('checkpoint_preprocessed.hdf5')

# 特徵工程之後
df_features = engineer_features(df_preprocessed)
df_features.export_hdf5('checkpoint_features.hdf5')

# 允許從檢查點繼續
```

## 效能比較

### 格式載入速度

```python
import time
import vaex

# CSV（最慢）
start = time.time()
df_csv = vaex.from_csv('data.csv')
csv_time = time.time() - start

# HDF5（即時）
start = time.time()
df_hdf5 = vaex.open('data.hdf5')
hdf5_time = time.time() - start

# Arrow（即時）
start = time.time()
df_arrow = vaex.open('data.arrow')
arrow_time = time.time() - start

print(f"CSV: {csv_time:.2f}s")
print(f"HDF5: {hdf5_time:.4f}s")
print(f"Arrow: {arrow_time:.4f}s")
```

## 常見模式

### 模式：生產資料管線

```python
import vaex

# 從來源讀取（CSV、資料庫匯出等）
df = vaex.from_csv('raw_data.csv')

# 處理
df['cleaned'] = clean(df.raw_column)
df['feature'] = engineer_feature(df)

# 匯出供生產使用
df.export_hdf5('production_data.hdf5')
df.state_write('production_state.json')

# 在生產環境：快速載入
df_prod = vaex.open('production_data.hdf5')
df_prod.state_load('production_state.json')
```

### 模式：帶壓縮的封存

```python
# 帶壓縮封存舊資料
df_2020 = vaex.open('data_2020.hdf5')
df_2020.export_hdf5('archive_2020.hdf5', compression='gzip')

# 移除未壓縮的原始檔案
import os
os.remove('data_2020.hdf5')
```

### 模式：多來源資料載入

```python
import vaex

# 從多個來源載入
df_csv = vaex.from_csv('data.csv')
df_hdf5 = vaex.open('data.hdf5')
df_parquet = vaex.open('data.parquet')

# 串接
df_all = vaex.concat([df_csv, df_hdf5, df_parquet])

# 匯出統一格式
df_all.export_hdf5('unified.hdf5')
```

## 疑難排解

### 問題：CSV 載入太慢

```python
# 解決方案：轉換為 HDF5
df = vaex.from_csv('large.csv', convert='large.hdf5')
# 未來：df = vaex.open('large.hdf5')
```

### 問題：匯出時記憶體不足

```python
# 解決方案：分塊匯出或先實體化
df_materialized = df.materialize()
df_materialized.export_hdf5('output.hdf5')
```

### 問題：無法從雲端讀取檔案

```python
# 安裝必要的函式庫
# pip install s3fs gcsfs adlfs

# 驗證憑證
import s3fs
fs = s3fs.S3FileSystem()
fs.ls('s3://bucket-name/')
```

## 格式功能對照表

| 功能 | HDF5 | Arrow | Parquet | CSV |
|------|------|-------|---------|-----|
| 載入速度 | 即時 | 即時 | 快速 | 慢 |
| 記憶體映射 | 是 | 是 | 否 | 否 |
| 壓縮 | 可選 | 否 | 是 | 否 |
| 欄式 | 是 | 是 | 是 | 否 |
| 可攜性 | 良好 | 優秀 | 優秀 | 優秀 |
| 檔案大小 | 中等 | 中等 | 小 | 大 |
| 最適合 | Vaex 工作流程 | 互通 | 分散式 | 交換 |

## 相關資源

- DataFrame 建立：參見 `core_dataframes.md`
- 效能最佳化：參見 `performance.md`
- 資料處理：參見 `data_processing.md`
