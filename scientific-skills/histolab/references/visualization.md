# 視覺化

## 概述

Histolab 提供多種內建的視覺化方法，以幫助檢視載玻片、預覽切片位置、視覺化遮罩和評估擷取品質。適當的視覺化對於驗證預處理流水線、除錯擷取問題和呈現結果至關重要。

## 載玻片視覺化

### 縮圖顯示

```python
from histolab.slide import Slide
import matplotlib.pyplot as plt

slide = Slide("slide.svs", processed_path="output/")

# 顯示縮圖
plt.figure(figsize=(10, 10))
plt.imshow(slide.thumbnail)
plt.title(f"Slide: {slide.name}")
plt.axis('off')
plt.show()
```

### 將縮圖儲存到磁碟

```python
# 將縮圖儲存為影像檔案
slide.save_thumbnail()
# 儲存到 processed_path/thumbnails/slide_name_thumb.png
```

### 縮放影像

```python
# 取得特定縮小倍率的載玻片縮放版本
scaled_img = slide.scaled_image(scale_factor=32)

plt.imshow(scaled_img)
plt.title(f"Slide at 32x downsample")
plt.show()
```

## 遮罩視覺化

### 使用 locate_mask()

```python
from histolab.masks import TissueMask, BiggestTissueBoxMask

# 視覺化 TissueMask
tissue_mask = TissueMask()
slide.locate_mask(tissue_mask)

# 視覺化 BiggestTissueBoxMask
biggest_mask = BiggestTissueBoxMask()
slide.locate_mask(biggest_mask)
```

這會在載玻片縮圖上以紅色覆蓋顯示遮罩邊界。

### 手動遮罩視覺化

```python
import matplotlib.pyplot as plt
from histolab.masks import TissueMask

slide = Slide("slide.svs", processed_path="output/")
mask = TissueMask()

# 生成遮罩
mask_array = mask(slide)

# 建立並排比較
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

# 原始縮圖
axes[0].imshow(slide.thumbnail)
axes[0].set_title("Original Slide")
axes[0].axis('off')

# 二值遮罩
axes[1].imshow(mask_array, cmap='gray')
axes[1].set_title("Tissue Mask")
axes[1].axis('off')

# 遮罩覆蓋在縮圖上
from matplotlib.colors import ListedColormap
overlay = slide.thumbnail.copy()
axes[2].imshow(overlay)
axes[2].imshow(mask_array, cmap=ListedColormap(['none', 'red']), alpha=0.3)
axes[2].set_title("Mask Overlay")
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

### 比較多個遮罩

```python
from histolab.masks import TissueMask, BiggestTissueBoxMask

masks = {
    'TissueMask': TissueMask(),
    'BiggestTissueBoxMask': BiggestTissueBoxMask()
}

fig, axes = plt.subplots(1, len(masks) + 1, figsize=(20, 6))

# 原始
axes[0].imshow(slide.thumbnail)
axes[0].set_title("Original")
axes[0].axis('off')

# 各個遮罩
for idx, (name, mask) in enumerate(masks.items(), 1):
    mask_array = mask(slide)
    axes[idx].imshow(mask_array, cmap='gray')
    axes[idx].set_title(name)
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
```

## 切片位置預覽

### 使用 locate_tiles()

在擷取前預覽切片位置：

```python
from histolab.tiler import RandomTiler, GridTiler, ScoreTiler
from histolab.scorer import NucleiScorer

# RandomTiler 預覽
random_tiler = RandomTiler(
    tile_size=(512, 512),
    n_tiles=50,
    level=0,
    seed=42
)
random_tiler.locate_tiles(slide, n_tiles=20)

# GridTiler 預覽
grid_tiler = GridTiler(
    tile_size=(512, 512),
    level=0
)
grid_tiler.locate_tiles(slide)

# ScoreTiler 預覽
score_tiler = ScoreTiler(
    tile_size=(512, 512),
    n_tiles=30,
    scorer=NucleiScorer()
)
score_tiler.locate_tiles(slide, n_tiles=15)
```

這會在載玻片縮圖上顯示彩色矩形，指示切片將被擷取的位置。

### 自訂切片位置視覺化

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from histolab.tiler import RandomTiler

slide = Slide("slide.svs", processed_path="output/")
tiler = RandomTiler(tile_size=(512, 512), n_tiles=30, seed=42)

# 取得縮圖和縮放因子
thumbnail = slide.thumbnail
scale_factor = slide.dimensions[0] / thumbnail.size[0]

# 生成切片座標（不擷取）
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(thumbnail)
ax.set_title("Tile Locations Preview")
ax.axis('off')

# 手動為每個切片位置加入矩形
# 注意：這是概念性的 - 實際實作會從切片擷取器取得座標
tile_coords = []  # 由切片擷取器邏輯填入
for coord in tile_coords:
    x, y = coord[0] / scale_factor, coord[1] / scale_factor
    w, h = 512 / scale_factor, 512 / scale_factor
    rect = patches.Rectangle((x, y), w, h,
                             linewidth=2, edgecolor='red',
                             facecolor='none')
    ax.add_patch(rect)

plt.show()
```

## 切片視覺化

### 顯示擷取的切片

```python
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

tile_dir = Path("output/tiles/")
tile_paths = list(tile_dir.glob("*.png"))[:16]  # 前 16 個切片

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
axes = axes.ravel()

for idx, tile_path in enumerate(tile_paths):
    tile_img = Image.open(tile_path)
    axes[idx].imshow(tile_img)
    axes[idx].set_title(tile_path.stem, fontsize=8)
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
```

### 切片網格拼接圖

```python
def create_tile_mosaic(tile_dir, grid_size=(4, 4)):
    """建立切片拼接圖。"""
    tile_paths = list(Path(tile_dir).glob("*.png"))[:grid_size[0] * grid_size[1]]

    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(16, 16))

    for idx, tile_path in enumerate(tile_paths):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        tile_img = Image.open(tile_path)
        axes[row, col].imshow(tile_img)
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig("tile_mosaic.png", dpi=150, bbox_inches='tight')
    plt.show()

create_tile_mosaic("output/tiles/", grid_size=(5, 5))
```

### 切片與組織遮罩覆蓋

```python
from histolab.tile import Tile
import matplotlib.pyplot as plt

# 假設我們有一個切片物件
tile = Tile(image=pil_image, coords=(x, y))

# 計算組織遮罩
tile.calculate_tissue_mask()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 原始切片
axes[0].imshow(tile.image)
axes[0].set_title("Original Tile")
axes[0].axis('off')

# 組織遮罩
axes[1].imshow(tile.tissue_mask, cmap='gray')
axes[1].set_title(f"Tissue Mask ({tile.tissue_ratio:.1%} tissue)")
axes[1].axis('off')

# 覆蓋
axes[2].imshow(tile.image)
axes[2].imshow(tile.tissue_mask, cmap='Reds', alpha=0.3)
axes[2].set_title("Overlay")
axes[2].axis('off')

plt.tight_layout()
plt.show()
```

## 品質評估視覺化

### 切片分數分佈

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 從 ScoreTiler 載入切片報告
report_df = pd.read_csv("tiles_report.csv")

# 分數分佈直方圖
plt.figure(figsize=(10, 6))
plt.hist(report_df['score'], bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Tile Score')
plt.ylabel('Frequency')
plt.title('Distribution of Tile Scores')
plt.grid(axis='y', alpha=0.3)
plt.show()

# 分數與組織百分比散佈圖
plt.figure(figsize=(10, 6))
plt.scatter(report_df['tissue_percent'], report_df['score'], alpha=0.5)
plt.xlabel('Tissue Percentage')
plt.ylabel('Tile Score')
plt.title('Tile Score vs Tissue Coverage')
plt.grid(alpha=0.3)
plt.show()
```

### 最高分與最低分切片比較

```python
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# 載入切片報告
report_df = pd.read_csv("tiles_report.csv")
report_df = report_df.sort_values('score', ascending=False)

# 前 8 個切片
top_tiles = report_df.head(8)
# 後 8 個切片
bottom_tiles = report_df.tail(8)

fig, axes = plt.subplots(2, 8, figsize=(20, 6))

# 顯示最高分切片
for idx, (_, row) in enumerate(top_tiles.iterrows()):
    tile_img = Image.open(f"output/tiles/{row['tile_name']}")
    axes[0, idx].imshow(tile_img)
    axes[0, idx].set_title(f"Score: {row['score']:.3f}", fontsize=8)
    axes[0, idx].axis('off')

# 顯示最低分切片
for idx, (_, row) in enumerate(bottom_tiles.iterrows()):
    tile_img = Image.open(f"output/tiles/{row['tile_name']}")
    axes[1, idx].imshow(tile_img)
    axes[1, idx].set_title(f"Score: {row['score']:.3f}", fontsize=8)
    axes[1, idx].axis('off')

axes[0, 0].set_ylabel('Top Scoring', fontsize=12)
axes[1, 0].set_ylabel('Bottom Scoring', fontsize=12)

plt.tight_layout()
plt.savefig("score_comparison.png", dpi=150, bbox_inches='tight')
plt.show()
```

## 多載玻片視覺化

### 載玻片集合縮圖

```python
from pathlib import Path
from histolab.slide import Slide
import matplotlib.pyplot as plt

slide_dir = Path("slides/")
slide_paths = list(slide_dir.glob("*.svs"))[:9]

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.ravel()

for idx, slide_path in enumerate(slide_paths):
    slide = Slide(slide_path, processed_path="output/")
    axes[idx].imshow(slide.thumbnail)
    axes[idx].set_title(slide.name, fontsize=10)
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig("slide_collection.png", dpi=150, bbox_inches='tight')
plt.show()
```

### 組織覆蓋率比較

```python
from pathlib import Path
from histolab.slide import Slide
from histolab.masks import TissueMask
import matplotlib.pyplot as plt
import numpy as np

slide_paths = list(Path("slides/").glob("*.svs"))
tissue_percentages = []
slide_names = []

for slide_path in slide_paths:
    slide = Slide(slide_path, processed_path="output/")
    mask = TissueMask()(slide)
    tissue_pct = mask.sum() / mask.size * 100
    tissue_percentages.append(tissue_pct)
    slide_names.append(slide.name)

# 長條圖
plt.figure(figsize=(12, 6))
plt.bar(range(len(slide_names)), tissue_percentages)
plt.xticks(range(len(slide_names)), slide_names, rotation=45, ha='right')
plt.ylabel('Tissue Coverage (%)')
plt.title('Tissue Coverage Across Slides')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

## 濾波器效果視覺化

### 濾波前後比較

```python
from histolab.filters.image_filters import RgbToGrayscale, HistogramEqualization
from histolab.filters.compositions import Compose

# 定義濾波器流水線
filter_pipeline = Compose([
    RgbToGrayscale(),
    HistogramEqualization()
])

# 原始 vs 濾波後
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(slide.thumbnail)
axes[0].set_title("Original")
axes[0].axis('off')

filtered = filter_pipeline(slide.thumbnail)
axes[1].imshow(filtered, cmap='gray')
axes[1].set_title("After Filtering")
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

### 多步驟濾波器視覺化

```python
from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
from histolab.filters.morphological_filters import BinaryDilation, RemoveSmallObjects

# 個別濾波器步驟
steps = [
    ("Original", None),
    ("Grayscale", RgbToGrayscale()),
    ("Otsu Threshold", Compose([RgbToGrayscale(), OtsuThreshold()])),
    ("After Dilation", Compose([RgbToGrayscale(), OtsuThreshold(), BinaryDilation(disk_size=5)])),
    ("Remove Small Objects", Compose([RgbToGrayscale(), OtsuThreshold(), BinaryDilation(disk_size=5), RemoveSmallObjects(area_threshold=500)]))
]

fig, axes = plt.subplots(1, len(steps), figsize=(20, 4))

for idx, (title, filter_fn) in enumerate(steps):
    if filter_fn is None:
        axes[idx].imshow(slide.thumbnail)
    else:
        result = filter_fn(slide.thumbnail)
        axes[idx].imshow(result, cmap='gray')
    axes[idx].set_title(title, fontsize=10)
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
```

## 匯出視覺化

### 高解析度匯出

```python
# 匯出高解析度圖表
fig, ax = plt.subplots(figsize=(20, 20))
ax.imshow(slide.thumbnail)
ax.axis('off')
plt.savefig("slide_high_res.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()
```

### PDF 報告

```python
from matplotlib.backends.backend_pdf import PdfPages

# 建立多頁 PDF 報告
with PdfPages('slide_report.pdf') as pdf:
    # 第 1 頁：載玻片縮圖
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    ax1.imshow(slide.thumbnail)
    ax1.set_title(f"Slide: {slide.name}")
    ax1.axis('off')
    pdf.savefig(fig1, bbox_inches='tight')
    plt.close()

    # 第 2 頁：組織遮罩
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    mask = TissueMask()(slide)
    ax2.imshow(mask, cmap='gray')
    ax2.set_title("Tissue Mask")
    ax2.axis('off')
    pdf.savefig(fig2, bbox_inches='tight')
    plt.close()

    # 第 3 頁：切片位置
    fig3, ax3 = plt.subplots(figsize=(10, 10))
    tiler = RandomTiler(tile_size=(512, 512), n_tiles=30)
    tiler.locate_tiles(slide)
    pdf.savefig(fig3, bbox_inches='tight')
    plt.close()
```

## 互動式視覺化（Jupyter）

### 使用 IPython Widgets 探索

```python
from ipywidgets import interact, IntSlider
import matplotlib.pyplot as plt
from histolab.filters.morphological_filters import BinaryDilation

@interact(disk_size=IntSlider(min=1, max=20, value=5))
def explore_dilation(disk_size):
    """互動式膨脹探索。"""
    filter_pipeline = Compose([
        RgbToGrayscale(),
        OtsuThreshold(),
        BinaryDilation(disk_size=disk_size)
    ])
    result = filter_pipeline(slide.thumbnail)

    plt.figure(figsize=(10, 10))
    plt.imshow(result, cmap='gray')
    plt.title(f"Binary Dilation (disk_size={disk_size})")
    plt.axis('off')
    plt.show()
```

## 最佳實務

1. **處理前務必預覽**：使用縮圖和 `locate_tiles()` 驗證設定
2. **使用並排比較**：顯示濾波器效果的前後對照
3. **清楚標示**：包含標題、軸標籤和圖例
4. **匯出高解析度**：使用 300 DPI 以獲得出版品質的圖表
5. **儲存中間視覺化**：記錄處理步驟
6. **適當使用色彩映射**：二值遮罩用 'gray'，熱力圖用 'viridis'
7. **建立可重複使用的視覺化函數**：跨專案標準化報告
