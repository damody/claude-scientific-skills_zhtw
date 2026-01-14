# 載玻片管理

## 概述

`Slide` 類別是在 histolab 中處理全切片影像（WSI）的主要介面。它提供載入、檢視和處理儲存在各種格式中的大型組織病理學影像的方法。

## 初始化

```python
from histolab.slide import Slide

# 使用 WSI 檔案和輸出目錄初始化載玻片
slide = Slide(processed_path="path/to/processed/output",
              slide_path="path/to/slide.svs")
```

**參數：**
- `slide_path`：全切片影像檔案的路徑（支援多種格式：SVS、TIFF、NDPI 等）
- `processed_path`：儲存處理輸出（切片、縮圖等）的目錄

## 載入範例資料

Histolab 提供來自 TCGA 的內建範例資料集，用於測試和示範：

```python
from histolab.data import prostate_tissue, ovarian_tissue, breast_tissue, heart_tissue, kidney_tissue

# 載入前列腺組織範例
prostate_svs, prostate_path = prostate_tissue()
slide = Slide(prostate_path, processed_path="output/")
```

可用的範例資料集：
- `prostate_tissue()`：前列腺組織範例
- `ovarian_tissue()`：卵巢組織範例
- `breast_tissue()`：乳房組織範例
- `heart_tissue()`：心臟組織範例
- `kidney_tissue()`：腎臟組織範例

## 主要屬性

### 載玻片尺寸
```python
# 取得層級 0（最高解析度）的載玻片尺寸
width, height = slide.dimensions

# 取得特定金字塔層級的尺寸
level_dimensions = slide.level_dimensions
# 回傳每個層級的 (width, height) 元組
```

### 放大倍率資訊
```python
# 取得基礎放大倍率（例如 40x、20x）
base_mag = slide.base_mpp  # 層級 0 的每像素微米數

# 取得所有可用層級
num_levels = slide.levels  # 金字塔層級數量
```

### 載玻片屬性
```python
# 存取 OpenSlide 屬性字典
properties = slide.properties

# 常見屬性包括：
# - slide.properties['openslide.objective-power']：物鏡倍率
# - slide.properties['openslide.mpp-x']：X 方向每像素微米數
# - slide.properties['openslide.mpp-y']：Y 方向每像素微米數
# - slide.properties['openslide.vendor']：掃描器廠商
```

## 縮圖生成

```python
# 取得特定大小的縮圖
thumbnail = slide.thumbnail

# 將縮圖儲存到磁碟
slide.save_thumbnail()  # 儲存到 processed_path

# 取得縮放的縮圖
scaled_thumbnail = slide.scaled_image(scale_factor=32)
```

## 載玻片視覺化

```python
# 使用 matplotlib 顯示載玻片縮圖
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.imshow(slide.thumbnail)
plt.title(f"Slide: {slide.name}")
plt.axis('off')
plt.show()
```

## 擷取區域

```python
# 在特定座標和層級擷取區域
region = slide.extract_region(
    location=(x, y),  # 層級 0 的左上角座標
    size=(width, height),  # 區域大小
    level=0  # 金字塔層級
)
```

## 處理金字塔層級

WSI 檔案使用具有多個解析度層級的金字塔結構：
- 層級 0：最高解析度（原生掃描解析度）
- 層級 1+：逐漸較低的解析度以加快存取速度

```python
# 檢查可用層級
for level in range(slide.levels):
    dims = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]
    print(f"Level {level}: {dims}, downsample: {downsample}x")
```

## 載玻片名稱與路徑

```python
# 取得不含副檔名的載玻片檔名
slide_name = slide.name

# 取得載玻片檔案的完整路徑
slide_path = slide.scaled_image
```

## 最佳實務

1. **務必指定 processed_path**：在專用目錄中組織輸出
2. **處理前檢查尺寸**：大型載玻片可能超出記憶體限制
3. **使用適當的金字塔層級**：在符合您分析解析度的層級擷取切片
4. **使用縮圖預覽**：在進行大量處理前使用縮圖快速視覺化
5. **監控記憶體使用**：大型載玻片的層級 0 操作需要大量 RAM

## 常見工作流程

### 載玻片檢視工作流程
```python
from histolab.slide import Slide

# 載入載玻片
slide = Slide("slide.svs", processed_path="output/")

# 檢視屬性
print(f"Dimensions: {slide.dimensions}")
print(f"Levels: {slide.levels}")
print(f"Magnification: {slide.properties.get('openslide.objective-power', 'N/A')}")

# 儲存縮圖以供檢閱
slide.save_thumbnail()
```

### 多載玻片處理
```python
import os
from pathlib import Path

slide_dir = Path("slides/")
output_dir = Path("processed/")

for slide_path in slide_dir.glob("*.svs"):
    slide = Slide(slide_path, processed_path=output_dir / slide_path.stem)
    # 處理每個載玻片
    print(f"Processing: {slide.name}")
```
