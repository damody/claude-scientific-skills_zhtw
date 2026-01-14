# 影像載入與格式

## 概述

PathML 提供全面的支援，可從 160+ 種專有醫學影像格式載入全切片影像（WSI）。該框架透過統一的切片類別和介面抽象化廠商特定的複雜性，實現對不同檔案格式的影像金字塔、元資料和感興趣區域的無縫存取。

## 支援的格式

PathML 支援以下切片格式：

### 明場顯微鏡格式
- **Aperio SVS**（`.svs`）- Leica Biosystems
- **Hamamatsu NDPI**（`.ndpi`）- Hamamatsu Photonics
- **Leica SCN**（`.scn`）- Leica Biosystems
- **Zeiss ZVI**（`.zvi`）- Carl Zeiss
- **3DHISTECH**（`.mrxs`）- 3DHISTECH Ltd.
- **Ventana BIF**（`.bif`）- Roche Ventana
- **通用 tiled TIFF**（`.tif`、`.tiff`）

### 醫學影像標準
- **DICOM**（`.dcm`）- 醫學數位影像與通訊
- **OME-TIFF**（`.ome.tif`、`.ome.tiff`）- 開放顯微鏡環境

### 多參數影像
- **CODEX** - 空間蛋白質組學影像
- **Vectra**（`.qptiff`）- 多重免疫螢光
- **MERFISH** - 多重錯誤穩健 FISH

PathML 利用 OpenSlide 和其他專門的函式庫自動處理格式特定的細節。

## 影像載入的核心類別

### SlideData

`SlideData` 是 PathML 中表示全切片影像的基本類別。

**從檔案載入：**
```python
from pathml.core import SlideData

# 載入全切片影像
wsi = SlideData.from_slide("path/to/slide.svs")

# 使用特定後端載入
wsi = SlideData.from_slide("path/to/slide.svs", backend="openslide")

# 從 OME-TIFF 載入
wsi = SlideData.from_slide("path/to/slide.ome.tiff", backend="bioformats")
```

**關鍵屬性：**
- `wsi.slide` - 後端切片物件（OpenSlide、BioFormats 等）
- `wsi.tiles` - 影像圖磚集合
- `wsi.metadata` - 切片元資料字典
- `wsi.level_dimensions` - 影像金字塔各層級維度
- `wsi.level_downsamples` - 每個金字塔層級的降採樣因子

**方法：**
- `wsi.generate_tiles()` - 從切片生成圖磚
- `wsi.read_region()` - 在指定層級讀取特定區域
- `wsi.get_thumbnail()` - 取得縮圖影像

### SlideType

`SlideType` 是定義支援的切片後端的列舉：

```python
from pathml.core import SlideType

# 可用後端
SlideType.OPENSLIDE  # 用於大多數 WSI 格式（SVS、NDPI 等）
SlideType.BIOFORMATS  # 用於 OME-TIFF 和其他格式
SlideType.DICOM  # 用於 DICOM WSI
SlideType.VectraQPTIFF  # 用於 Vectra 多重 IF
```

### 專門的切片類別

PathML 為特定影像模式提供專門的切片類別：

**CODEXSlide：**
```python
from pathml.core import CODEXSlide

# 載入 CODEX 空間蛋白質組學資料
codex_slide = CODEXSlide(
    path="path/to/codex_dir",
    stain="IF",  # 免疫螢光
    backend="bioformats"
)
```

**VectraSlide：**
```python
from pathml.core import types

# 載入 Vectra 多重 IF 資料
vectra_slide = SlideData.from_slide(
    "path/to/vectra.qptiff",
    backend=SlideType.VectraQPTIFF
)
```

**MultiparametricSlide：**
```python
from pathml.core import MultiparametricSlide

# 通用多參數影像
mp_slide = MultiparametricSlide(path="path/to/multiparametric_data")
```

## 載入策略

### 基於圖磚的載入

對於大型 WSI 檔案，基於圖磚的載入可實現記憶體效率的處理：

```python
from pathml.core import SlideData

# 載入切片
wsi = SlideData.from_slide("path/to/slide.svs")

# 在特定放大層級生成圖磚
wsi.generate_tiles(
    level=0,  # 金字塔層級（0 = 最高解析度）
    tile_size=256,  # 圖磚維度（像素）
    stride=256,  # 圖磚間距（256 = 無重疊）
    pad=False  # 是否填充邊緣圖磚
)

# 遍歷圖磚
for tile in wsi.tiles:
    image = tile.image  # numpy 陣列
    coords = tile.coords  # (x, y) 座標
    # 處理圖磚...
```

**重疊圖磚：**
```python
# 生成 50% 重疊的圖磚
wsi.generate_tiles(
    level=0,
    tile_size=256,
    stride=128  # 50% 重疊
)
```

### 基於區域的載入

直接提取特定感興趣區域：

```python
# 在特定位置和層級讀取區域
region = wsi.read_region(
    location=(10000, 15000),  # 層級 0 座標中的 (x, y)
    level=1,  # 金字塔層級
    size=(512, 512)  # 寬度、高度（像素）
)

# 返回 numpy 陣列
```

### 金字塔層級選擇

全切片影像以多解析度金字塔儲存。根據所需放大倍率選擇適當的層級：

```python
# 檢查可用層級
print(wsi.level_dimensions)  # [(width0, height0), (width1, height1), ...]
print(wsi.level_downsamples)  # [1.0, 4.0, 16.0, ...]

# 在較低解析度載入以加快處理
wsi.generate_tiles(level=2, tile_size=256)  # 使用層級 2（16x 降採樣）
```

**常見金字塔層級：**
- 層級 0：完整解析度（例如 40x 放大）
- 層級 1：4x 降採樣（例如 10x 放大）
- 層級 2：16x 降採樣（例如 2.5x 放大）
- 層級 3：64x 降採樣（縮圖）

### 縮圖載入

生成低解析度縮圖用於視覺化和品質控制：

```python
# 取得縮圖
thumbnail = wsi.get_thumbnail(size=(1024, 1024))

# 使用 matplotlib 顯示
import matplotlib.pyplot as plt
plt.imshow(thumbnail)
plt.axis('off')
plt.show()
```

## 使用 SlideDataset 批次載入

使用 `SlideDataset` 高效處理多個切片：

```python
from pathml.core import SlideDataset
import glob

# 從多個切片創建資料集
slide_paths = glob.glob("data/*.svs")
dataset = SlideDataset(
    slide_paths,
    tile_size=256,
    stride=256,
    level=0
)

# 遍歷所有切片的所有圖磚
for tile in dataset:
    image = tile.image
    slide_id = tile.slide_id
    # 處理圖磚...
```

**使用預處理管道：**
```python
from pathml.preprocessing import Pipeline, StainNormalizationHE

# 創建管道
pipeline = Pipeline([
    StainNormalizationHE(target='normalize')
])

# 應用到整個資料集
dataset = SlideDataset(slide_paths)
dataset.run(pipeline, distributed=True, n_workers=8)
```

## 元資料存取

提取切片元資料，包括擷取參數、放大倍率和廠商特定資訊：

```python
# 存取元資料
metadata = wsi.metadata

# 常見元資料欄位
print(metadata.get('openslide.objective-power'))  # 放大倍率
print(metadata.get('openslide.mpp-x'))  # X 方向每像素微米數
print(metadata.get('openslide.mpp-y'))  # Y 方向每像素微米數
print(metadata.get('openslide.vendor'))  # 掃描儀廠商

# 切片維度
print(wsi.level_dimensions[0])  # 層級 0 的 (寬度, 高度)
```

## 處理 DICOM 切片

PathML 透過專門處理支援 DICOM WSI：

```python
from pathml.core import SlideData, SlideType

# 載入 DICOM WSI
dicom_slide = SlideData.from_slide(
    "path/to/slide.dcm",
    backend=SlideType.DICOM
)

# DICOM 特定元資料
print(dicom_slide.metadata.get('PatientID'))
print(dicom_slide.metadata.get('StudyDate'))
```

## 處理 OME-TIFF

OME-TIFF 為多維影像提供開放標準：

```python
from pathml.core import SlideData

# 載入 OME-TIFF
ome_slide = SlideData.from_slide(
    "path/to/slide.ome.tiff",
    backend="bioformats"
)

# 存取多通道影像的通道資訊
n_channels = ome_slide.shape[2]  # 通道數量
```

## 效能考量

### 記憶體管理

對於大型 WSI 檔案（通常 >1GB），使用基於圖磚的載入以避免記憶體耗盡：

```python
# 高效：基於圖磚的處理
wsi.generate_tiles(level=1, tile_size=256)
for tile in wsi.tiles:
    process_tile(tile)  # 一次處理一個圖磚

# 低效：將整個切片載入記憶體
full_image = wsi.read_region((0, 0), level=0, wsi.level_dimensions[0])  # 可能當機
```

### 分散式處理

使用 Dask 進行多工作者的並行處理：

```python
from pathml.core import SlideDataset
from dask.distributed import Client

# 啟動 Dask 客戶端
client = Client(n_workers=8, threads_per_worker=2)

# 並行處理資料集
dataset = SlideDataset(slide_paths)
dataset.run(pipeline, distributed=True, client=client)
```

### 層級選擇

通過選擇適當的金字塔層級來平衡解析度和效能：

- **層級 0：** 用於需要最大細節的最終分析
- **層級 1-2：** 用於大多數預處理和模型訓練
- **層級 3+：** 用於縮圖、品質控制和快速探索

## 常見問題與解決方案

**問題：切片載入失敗**
- 驗證檔案格式受支援
- 檢查檔案權限和路徑
- 嘗試不同後端：`backend="bioformats"` 或 `backend="openslide"`

**問題：記憶體不足錯誤**
- 使用基於圖磚的載入而非完整切片載入
- 在較低金字塔層級處理（例如 level=1 或 level=2）
- 減少 tile_size 參數
- 使用 Dask 啟用分散式處理

**問題：切片間顏色不一致**
- 應用染色正規化預處理（見 `preprocessing.md`）
- 檢查掃描儀元資料以取得校準資訊
- 在預處理管道中使用 `StainNormalizationHE` 轉換

**問題：元資料缺失或不正確**
- 不同廠商將元資料儲存在不同位置
- 使用 `wsi.metadata` 檢查可用欄位
- 某些格式可能有有限的元資料支援

## 最佳實踐

1. **處理前始終檢查金字塔結構：** 檢查 `level_dimensions` 和 `level_downsamples` 以了解可用解析度

2. **使用適當的金字塔層級：** 大多數任務在層級 1-2 處理；將層級 0 保留給最終高解析度分析

3. **分割任務使用重疊圖磚：** 使用 stride < tile_size 以避免邊緣偽影

4. **驗證放大倍率一致性：** 合併不同來源的切片時檢查 `openslide.objective-power` 元資料

5. **處理廠商特定格式：** 對多參數資料使用專門的切片類別（CODEXSlide、VectraSlide）

6. **實施品質控制：** 處理前生成縮圖並檢查偽影

7. **對大型資料集使用分散式處理：** 利用 Dask 進行多工作者的並行處理

## 範例工作流程

### 載入並檢查新切片

```python
from pathml.core import SlideData
import matplotlib.pyplot as plt

# 載入切片
wsi = SlideData.from_slide("path/to/slide.svs")

# 檢查屬性
print(f"維度：{wsi.level_dimensions}")
print(f"降採樣：{wsi.level_downsamples}")
print(f"放大倍率：{wsi.metadata.get('openslide.objective-power')}")

# 生成縮圖進行品質控制
thumbnail = wsi.get_thumbnail(size=(1024, 1024))
plt.imshow(thumbnail)
plt.title(f"切片：{wsi.name}")
plt.axis('off')
plt.show()
```

### 處理多個切片

```python
from pathml.core import SlideDataset
from pathml.preprocessing import Pipeline, TissueDetectionHE
import glob

# 尋找所有切片
slide_paths = glob.glob("data/slides/*.svs")

# 創建管道
pipeline = Pipeline([TissueDetectionHE()])

# 處理所有切片
dataset = SlideDataset(
    slide_paths,
    tile_size=512,
    stride=512,
    level=1
)

# 使用分散式處理執行管道
dataset.run(pipeline, distributed=True, n_workers=8)

# 儲存處理過的資料
dataset.to_hdf5("processed_dataset.h5")
```

### 載入 CODEX 多參數資料

```python
from pathml.core import CODEXSlide
from pathml.preprocessing import Pipeline, CollapseRunsCODEX, SegmentMIF

# 載入 CODEX 切片
codex = CODEXSlide("path/to/codex_dir", stain="IF")

# 創建 CODEX 特定管道
pipeline = Pipeline([
    CollapseRunsCODEX(z_slice=2),  # 選擇 z 切面
    SegmentMIF(
        nuclear_channel='DAPI',
        cytoplasm_channel='CD45',
        model='mesmer'
    )
])

# 處理
pipeline.run(codex)
```

## 其他資源

- **PathML 文件：** https://pathml.readthedocs.io/
- **OpenSlide：** https://openslide.org/（WSI 格式的底層函式庫）
- **Bio-Formats：** https://www.openmicroscopy.org/bio-formats/（替代後端）
- **DICOM 標準：** https://www.dicomstandard.org/
