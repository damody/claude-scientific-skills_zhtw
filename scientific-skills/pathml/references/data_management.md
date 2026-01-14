# 資料管理與儲存

## 概述

PathML 提供高效的資料管理解決方案，透過 HDF5 儲存、圖磚管理策略和最佳化批次處理工作流程來處理大規模病理資料集。該框架實現影像、遮罩、特徵和元資料的無縫儲存與檢索，其格式針對機器學習管線和下游分析進行了最佳化。

## HDF5 整合

HDF5（階層式資料格式）是處理過的 PathML 資料的主要儲存格式，提供：
- 高效的壓縮和分塊儲存
- 快速隨機存取資料子集
- 支援任意大小的資料集
- 異質資料類型的階層式組織
- 跨平台相容性

### 儲存到 HDF5

**單一切片：**
```python
from pathml.core import SlideData

# 載入並處理切片
wsi = SlideData.from_slide("slide.svs")
wsi.generate_tiles(level=1, tile_size=256, stride=256)

# 執行預處理管道
pipeline.run(wsi)

# 儲存到 HDF5
wsi.to_hdf5("processed_slide.h5")
```

**多個切片（SlideDataset）：**
```python
from pathml.core import SlideDataset
import glob

# 創建資料集
slide_paths = glob.glob("data/*.svs")
dataset = SlideDataset(slide_paths, tile_size=256, stride=256, level=1)

# 處理
dataset.run(pipeline, distributed=True, n_workers=8)

# 儲存整個資料集
dataset.to_hdf5("processed_dataset.h5")
```

### HDF5 檔案結構

PathML HDF5 檔案以階層方式組織：

```
processed_dataset.h5
├── slide_0/
│   ├── metadata/
│   │   ├── name
│   │   ├── level
│   │   ├── dimensions
│   │   └── ...
│   ├── tiles/
│   │   ├── tile_0/
│   │   │   ├── image  (H, W, C) 陣列
│   │   │   ├── coords  (x, y)
│   │   │   └── masks/
│   │   │       ├── tissue
│   │   │       ├── nucleus
│   │   │       └── ...
│   │   ├── tile_1/
│   │   └── ...
│   └── features/
│       ├── tile_features  (n_tiles, n_features)
│       └── feature_names
├── slide_1/
└── ...
```

### 從 HDF5 載入

**載入整個切片：**
```python
from pathml.core import SlideData

# 從 HDF5 載入
wsi = SlideData.from_hdf5("processed_slide.h5")

# 存取圖磚
for tile in wsi.tiles:
    image = tile.image
    masks = tile.masks
    # 處理圖磚...
```

**載入特定圖磚：**
```python
# 僅載入特定索引的圖磚
tile_indices = [0, 10, 20, 30]
tiles = wsi.load_tiles_from_hdf5("processed_slide.h5", indices=tile_indices)

for tile in tiles:
    # 處理子集...
    pass
```

**記憶體映射存取：**
```python
import h5py

# 開啟 HDF5 檔案而不載入到記憶體
with h5py.File("processed_dataset.h5", 'r') as f:
    # 存取特定資料
    tile_0_image = f['slide_0/tiles/tile_0/image'][:]
    tissue_mask = f['slide_0/tiles/tile_0/masks/tissue'][:]

    # 高效遍歷圖磚
    for tile_key in f['slide_0/tiles'].keys():
        tile_image = f[f'slide_0/tiles/{tile_key}/image'][:]
        # 無需載入所有圖磚即可處理...
```

## 圖磚管理

### 圖磚生成策略

**固定大小圖磚無重疊：**
```python
wsi.generate_tiles(
    level=1,
    tile_size=256,
    stride=256,  # stride = tile_size → 無重疊
    pad=False  # 不填充邊緣圖磚
)
```
- **使用場景：** 標準圖磚式處理、分類
- **優點：** 簡單、無冗餘、處理快速
- **缺點：** 圖磚邊界處的邊緣效應

**重疊圖磚：**
```python
wsi.generate_tiles(
    level=1,
    tile_size=256,
    stride=128,  # 50% 重疊
    pad=False
)
```
- **使用場景：** 分割、檢測（減少邊界偽影）
- **優點：** 更好的邊界處理、更平滑的拼接
- **缺點：** 更多圖磚、冗餘計算

**基於組織內容的自適應圖磚：**
```python
from pathml.utils import adaptive_tile_generation

# 僅在組織區域生成圖磚
wsi.generate_tiles(level=1, tile_size=256, stride=256)

# 篩選保留具有足夠組織的圖磚
tissue_tiles = []
for tile in wsi.tiles:
    if tile.masks.get('tissue') is not None:
        tissue_coverage = tile.masks['tissue'].sum() / (tile_size**2)
        if tissue_coverage > 0.5:  # 保留組織覆蓋 >50% 的圖磚
            tissue_tiles.append(tile)

wsi.tiles = tissue_tiles
```
- **使用場景：** 稀疏組織樣本、效率最佳化
- **優點：** 減少背景圖磚的處理
- **缺點：** 需要組織檢測預處理步驟

### 圖磚拼接

從處理過的圖磚重建完整切片：

```python
from pathml.utils import stitch_tiles

# 處理圖磚
for tile in wsi.tiles:
    tile.prediction = model.predict(tile.image)

# 將預測結果拼接回完整解析度
full_prediction_map = stitch_tiles(
    wsi.tiles,
    output_shape=wsi.level_dimensions[1],  # 使用層級 1 的維度
    tile_size=256,
    stride=256,
    method='average'  # 'average'、'max' 或 'first'
)

# 視覺化
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 15))
plt.imshow(full_prediction_map)
plt.title('拼接預測圖')
plt.axis('off')
plt.show()
```

**拼接方法：**
- `'average'`：重疊區域取平均值（平滑過渡）
- `'max'`：重疊區域取最大值
- `'first'`：保留第一個圖磚的值（無混合）
- `'weighted'`：基於距離的加權混合，實現平滑邊界

### 圖磚快取

快取經常存取的圖磚以加快遍歷速度：

```python
from pathml.utils import TileCache

# 創建快取
cache = TileCache(max_size_gb=10)

# 第一次遍歷時快取圖磚
for i, tile in enumerate(wsi.tiles):
    cache.add(f'tile_{i}', tile.image)
    # 處理圖磚...

# 後續遍歷使用快取資料
for i in range(len(wsi.tiles)):
    cached_image = cache.get(f'tile_{i}')
    # 快速存取...
```

## 資料集組織

### 大型專案的目錄結構

使用一致的結構組織病理專案：

```
project/
├── raw_slides/
│   ├── cohort1/
│   │   ├── slide001.svs
│   │   ├── slide002.svs
│   │   └── ...
│   └── cohort2/
│       └── ...
├── processed/
│   ├── cohort1/
│   │   ├── slide001.h5
│   │   ├── slide002.h5
│   │   └── ...
│   └── cohort2/
│       └── ...
├── features/
│   ├── cohort1_features.h5
│   └── cohort2_features.h5
├── models/
│   ├── hovernet_checkpoint.pth
│   └── classifier.onnx
├── results/
│   ├── predictions/
│   ├── visualizations/
│   └── metrics.csv
└── metadata/
    ├── clinical_data.csv
    └── slide_manifest.csv
```

### 元資料管理

儲存切片層級和群組層級的元資料：

```python
import pandas as pd

# 切片清單
manifest = pd.DataFrame({
    'slide_id': ['slide001', 'slide002', 'slide003'],
    'path': ['raw_slides/cohort1/slide001.svs', ...],
    'cohort': ['cohort1', 'cohort1', 'cohort2'],
    'tissue_type': ['breast', 'breast', 'lung'],
    'scanner': ['Aperio', 'Hamamatsu', 'Aperio'],
    'magnification': [40, 40, 20],
    'staining': ['H&E', 'H&E', 'H&E']
})

manifest.to_csv('metadata/slide_manifest.csv', index=False)

# 臨床資料
clinical = pd.DataFrame({
    'slide_id': ['slide001', 'slide002', 'slide003'],
    'patient_id': ['P001', 'P002', 'P003'],
    'age': [55, 62, 48],
    'diagnosis': ['invasive', 'in_situ', 'invasive'],
    'stage': ['II', 'I', 'III'],
    'outcome': ['favorable', 'favorable', 'poor']
})

clinical.to_csv('metadata/clinical_data.csv', index=False)

# 載入並合併
manifest = pd.read_csv('metadata/slide_manifest.csv')
clinical = pd.read_csv('metadata/clinical_data.csv')
data = manifest.merge(clinical, on='slide_id')
```

## 批次處理策略

### 順序處理

逐一處理切片（記憶體效率高）：

```python
import glob
from pathml.core import SlideData
from pathml.preprocessing import Pipeline

slide_paths = glob.glob('raw_slides/**/*.svs', recursive=True)

for slide_path in slide_paths:
    # 載入切片
    wsi = SlideData.from_slide(slide_path)
    wsi.generate_tiles(level=1, tile_size=256, stride=256)

    # 處理
    pipeline.run(wsi)

    # 儲存
    output_path = slide_path.replace('raw_slides', 'processed').replace('.svs', '.h5')
    wsi.to_hdf5(output_path)

    print(f"已處理：{slide_path}")
```

### 使用 Dask 並行處理

並行處理多個切片：

```python
from pathml.core import SlideDataset
from dask.distributed import Client, LocalCluster
from pathml.preprocessing import Pipeline

# 啟動 Dask 叢集
cluster = LocalCluster(
    n_workers=8,
    threads_per_worker=2,
    memory_limit='8GB',
    dashboard_address=':8787'  # 在 localhost:8787 查看進度
)
client = Client(cluster)

# 創建資料集
slide_paths = glob.glob('raw_slides/**/*.svs', recursive=True)
dataset = SlideDataset(slide_paths, tile_size=256, stride=256, level=1)

# 分散式處理
dataset.run(
    pipeline,
    distributed=True,
    client=client,
    scheduler='distributed'
)

# 儲存結果
for i, slide in enumerate(dataset):
    output_path = slide_paths[i].replace('raw_slides', 'processed').replace('.svs', '.h5')
    slide.to_hdf5(output_path)

client.close()
cluster.close()
```

### 使用作業陣列進行批次處理

用於 HPC 叢集（SLURM、PBS）：

```python
# submit_jobs.py
import os
import glob

slide_paths = glob.glob('raw_slides/**/*.svs', recursive=True)

# 寫入切片清單
with open('slide_list.txt', 'w') as f:
    for path in slide_paths:
        f.write(path + '\n')

# 創建 SLURM 作業腳本
slurm_script = """#!/bin/bash
#SBATCH --array=1-{n_slides}
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slide_%A_%a.out

# 取得此陣列任務的切片路徑
SLIDE_PATH=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" slide_list.txt)

# 執行處理
python process_slide.py --slide_path $SLIDE_PATH
""".format(n_slides=len(slide_paths))

with open('submit_jobs.sh', 'w') as f:
    f.write(slurm_script)

# 提交：sbatch submit_jobs.sh
```

```python
# process_slide.py
import argparse
from pathml.core import SlideData
from pathml.preprocessing import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--slide_path', type=str, required=True)
args = parser.parse_args()

# 載入並處理
wsi = SlideData.from_slide(args.slide_path)
wsi.generate_tiles(level=1, tile_size=256, stride=256)

pipeline = Pipeline([...])
pipeline.run(wsi)

# 儲存
output_path = args.slide_path.replace('raw_slides', 'processed').replace('.svs', '.h5')
wsi.to_hdf5(output_path)

print(f"已處理：{args.slide_path}")
```

## 特徵提取與儲存

### 提取特徵

```python
from pathml.core import SlideData
import torch
import numpy as np

# 載入預訓練模型進行特徵提取
model = torch.load('models/feature_extractor.pth')
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 載入處理過的切片
wsi = SlideData.from_hdf5('processed/slide001.h5')

# 為每個圖磚提取特徵
features = []
coords = []

for tile in wsi.tiles:
    # 預處理圖磚
    tile_tensor = torch.from_numpy(tile.image).permute(2, 0, 1).unsqueeze(0).float()
    tile_tensor = tile_tensor.to(device)

    # 提取特徵
    with torch.no_grad():
        feature_vec = model(tile_tensor).cpu().numpy().flatten()

    features.append(feature_vec)
    coords.append(tile.coords)

features = np.array(features)  # 形狀：(n_tiles, feature_dim)
coords = np.array(coords)  # 形狀：(n_tiles, 2)
```

### 將特徵儲存到 HDF5

```python
import h5py

# 儲存特徵
with h5py.File('features/slide001_features.h5', 'w') as f:
    f.create_dataset('features', data=features, compression='gzip')
    f.create_dataset('coords', data=coords)
    f.attrs['feature_dim'] = features.shape[1]
    f.attrs['num_tiles'] = features.shape[0]
    f.attrs['model'] = 'resnet50'

# 載入特徵
with h5py.File('features/slide001_features.h5', 'r') as f:
    features = f['features'][:]
    coords = f['coords'][:]
    feature_dim = f.attrs['feature_dim']
```

### 多個切片的特徵資料庫

```python
# 創建合併的特徵資料庫
import h5py
import glob

feature_files = glob.glob('features/*_features.h5')

with h5py.File('features/all_features.h5', 'w') as out_f:
    for i, feature_file in enumerate(feature_files):
        slide_name = feature_file.split('/')[-1].replace('_features.h5', '')

        with h5py.File(feature_file, 'r') as in_f:
            features = in_f['features'][:]
            coords = in_f['coords'][:]

            # 儲存到合併檔案
            grp = out_f.create_group(f'slide_{i}')
            grp.create_dataset('features', data=features, compression='gzip')
            grp.create_dataset('coords', data=coords)
            grp.attrs['slide_name'] = slide_name

# 從所有切片查詢特徵
with h5py.File('features/all_features.h5', 'r') as f:
    for slide_key in f.keys():
        slide_name = f[slide_key].attrs['slide_name']
        features = f[f'{slide_key}/features'][:]
        # 處理...
```

## 資料版本控制

### 使用 DVC 進行版本控制

使用資料版本控制（DVC）進行大型資料集管理：

```bash
# 初始化 DVC
dvc init

# 新增資料目錄
dvc add raw_slides/
dvc add processed/

# 提交到 git
git add raw_slides.dvc processed.dvc .gitignore
git commit -m "Add raw and processed slides"

# 推送資料到遠端儲存（S3、GCS 等）
dvc remote add -d storage s3://my-bucket/pathml-data
dvc push

# 在另一台機器上拉取資料
git pull
dvc pull
```

### 校驗碼和驗證

驗證資料完整性：

```python
import hashlib
import pandas as pd

def compute_checksum(file_path):
    """計算檔案的 MD5 校驗碼。"""
    hash_md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# 創建校驗碼清單
slide_paths = glob.glob('raw_slides/**/*.svs', recursive=True)
checksums = []

for slide_path in slide_paths:
    checksum = compute_checksum(slide_path)
    checksums.append({
        'path': slide_path,
        'checksum': checksum,
        'size_mb': os.path.getsize(slide_path) / 1e6
    })

checksum_df = pd.DataFrame(checksums)
checksum_df.to_csv('metadata/checksums.csv', index=False)

# 驗證檔案
def validate_files(manifest_path):
    manifest = pd.read_csv(manifest_path)
    for _, row in manifest.iterrows():
        current_checksum = compute_checksum(row['path'])
        if current_checksum != row['checksum']:
            print(f"錯誤：{row['path']} 的校驗碼不匹配")
        else:
            print(f"正常：{row['path']}")

validate_files('metadata/checksums.csv')
```

## 效能最佳化

### 壓縮設定

最佳化 HDF5 壓縮以平衡速度和大小：

```python
import h5py

# 快速壓縮（較少 CPU，較大檔案）
with h5py.File('output.h5', 'w') as f:
    f.create_dataset(
        'images',
        data=images,
        compression='gzip',
        compression_opts=1  # 等級 1-9，越低越快
    )

# 最大壓縮（較多 CPU，較小檔案）
with h5py.File('output.h5', 'w') as f:
    f.create_dataset(
        'images',
        data=images,
        compression='gzip',
        compression_opts=9
    )

# 平衡（推薦）
with h5py.File('output.h5', 'w') as f:
    f.create_dataset(
        'images',
        data=images,
        compression='gzip',
        compression_opts=4,
        chunks=True  # 啟用分塊以獲得更好的 I/O
    )
```

### 分塊策略

針對存取模式最佳化分塊儲存：

```python
# 用於圖磚式存取（一次存取一個圖磚）
with h5py.File('tiles.h5', 'w') as f:
    f.create_dataset(
        'tiles',
        shape=(n_tiles, 256, 256, 3),
        dtype='uint8',
        chunks=(1, 256, 256, 3),  # 每個分塊一個圖磚
        compression='gzip'
    )

# 用於通道式存取（存取一個通道的所有圖磚）
with h5py.File('tiles.h5', 'w') as f:
    f.create_dataset(
        'tiles',
        shape=(n_tiles, 256, 256, 3),
        dtype='uint8',
        chunks=(n_tiles, 256, 256, 1),  # 一個通道的所有圖磚
        compression='gzip'
    )
```

### 記憶體映射陣列

使用記憶體映射處理大型陣列：

```python
import numpy as np

# 儲存為記憶體映射檔案
features_mmap = np.memmap(
    'features/features.mmap',
    dtype='float32',
    mode='w+',
    shape=(n_tiles, feature_dim)
)

# 填充
for i, tile in enumerate(wsi.tiles):
    features_mmap[i] = extract_features(tile)

# 刷新到磁碟
features_mmap.flush()

# 載入但不讀入記憶體
features_mmap = np.memmap(
    'features/features.mmap',
    dtype='float32',
    mode='r',
    shape=(n_tiles, feature_dim)
)

# 高效存取子集
subset = features_mmap[1000:2000]  # 僅載入請求的列
```

## 最佳實踐

1. **使用 HDF5 儲存處理過的資料：** 將預處理的圖磚和特徵儲存到 HDF5 以便快速存取

2. **分離原始和處理過的資料：** 將原始切片與處理輸出分開保存

3. **維護元資料：** 追蹤切片來源、處理參數和臨床註釋

4. **實施校驗碼：** 驗證資料完整性，特別是在傳輸後

5. **版本控制資料集：** 使用 DVC 或類似工具對大型資料集進行版本控制

6. **最佳化儲存：** 平衡壓縮等級和 I/O 效能

7. **按群組組織：** 按研究群組結構化目錄以保持清晰

8. **定期備份：** 將資料和元資料備份到遠端儲存

9. **記錄處理過程：** 保留處理步驟、參數和版本的日誌

10. **監控磁碟使用：** 隨著資料集增長追蹤儲存消耗

## 常見問題與解決方案

**問題：HDF5 檔案非常大**
- 提高壓縮等級：`compression_opts=9`
- 僅儲存必要的資料（避免冗餘複製）
- 使用適當的資料類型（影像用 uint8 而非 float64）

**問題：HDF5 讀寫緩慢**
- 針對存取模式最佳化分塊大小
- 降低壓縮等級以加快 I/O
- 使用 SSD 儲存而非 HDD
- 使用 MPI 啟用並行 HDF5

**問題：磁碟空間不足**
- 處理後刪除中間檔案
- 壓縮非活動資料集
- 將舊資料移至歸檔儲存
- 使用雲端儲存存放較少存取的資料

**問題：資料損壞或遺失**
- 實施定期備份
- 使用 RAID 進行冗餘
- 傳輸後驗證校驗碼
- 使用版本控制（DVC）

## 其他資源

- **HDF5 文件：** https://www.hdfgroup.org/solutions/hdf5/
- **h5py：** https://docs.h5py.org/
- **DVC（資料版本控制）：** https://dvc.org/
- **Dask：** https://docs.dask.org/
- **PathML 資料管理 API：** https://pathml.readthedocs.io/en/latest/api_data_reference.html
