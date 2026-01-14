# 組織遮罩

## 概述

組織遮罩是識別全切片影像中組織區域的二值表示。它們對於在切片擷取過程中過濾背景、人工痕跡和非組織區域至關重要。Histolab 提供多種遮罩類別以適應不同的組織分割需求。

## 遮罩類別

### BinaryMask

**用途：** 用於建立自訂二值遮罩的通用基礎類別。

```python
from histolab.masks import BinaryMask

class CustomMask(BinaryMask):
    def _mask(self, obj):
        # 實作自訂遮罩邏輯
        # 回傳二值 numpy 陣列
        pass
```

**使用案例：**
- 自訂組織分割演算法
- 區域特定分析（例如，排除註記）
- 與外部分割模型整合

### TissueMask

**用途：** 使用自動化濾波器分割載玻片中的所有組織區域。

```python
from histolab.masks import TissueMask

# 建立組織遮罩
tissue_mask = TissueMask()

# 應用於載玻片
mask_array = tissue_mask(slide)
```

**工作原理：**
1. 將影像轉換為灰階
2. 應用 Otsu 閾值處理以分離組織與背景
3. 執行二值膨脹以連接鄰近組織區域
4. 移除組織區域內的小孔洞
5. 過濾掉小物件（人工痕跡）

**回傳：** 二值 NumPy 陣列，其中：
- `True`（或 1）：組織像素
- `False`（或 0）：背景像素

**最適合：**
- 具有多個分離組織切片的載玻片
- 全面的組織分析
- 當所有組織區域都很重要時

### BiggestTissueBoxMask（預設）

**用途：** 識別並回傳最大連通組織區域的邊界框。

```python
from histolab.masks import BiggestTissueBoxMask

# 建立最大組織區域的遮罩
biggest_mask = BiggestTissueBoxMask()

# 應用於載玻片
mask_array = biggest_mask(slide)
```

**工作原理：**
1. 應用與 TissueMask 相同的濾波流水線
2. 識別所有連通組織元件
3. 選擇最大的連通元件
4. 回傳包含該區域的邊界框

**最適合：**
- 具有單一主要組織切片的載玻片
- 排除小人工痕跡或組織碎片
- 聚焦於主要組織區域（大多數切片擷取器的預設值）

## 使用濾波器自訂遮罩

遮罩接受自訂濾波器鏈以進行專門的組織偵測：

```python
from histolab.masks import TissueMask
from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
from histolab.filters.morphological_filters import BinaryDilation, RemoveSmallHoles

# 定義自訂濾波器組合
custom_mask = TissueMask(
    filters=[
        RgbToGrayscale(),
        OtsuThreshold(),
        BinaryDilation(disk_size=5),
        RemoveSmallHoles(area_threshold=500)
    ]
)
```

## 視覺化遮罩

### 使用 locate_mask()

```python
from histolab.slide import Slide
from histolab.masks import TissueMask

slide = Slide("slide.svs", processed_path="output/")
mask = TissueMask()

# 在縮圖上視覺化遮罩邊界
slide.locate_mask(mask)
```

這會在載玻片縮圖上顯示遮罩邊界，以對比色覆蓋。

### 手動視覺化

```python
import matplotlib.pyplot as plt
from histolab.masks import TissueMask

slide = Slide("slide.svs", processed_path="output/")
tissue_mask = TissueMask()

# 生成遮罩
mask_array = tissue_mask(slide)

# 並排繪製
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

axes[0].imshow(slide.thumbnail)
axes[0].set_title("Original Slide")
axes[0].axis('off')

axes[1].imshow(mask_array, cmap='gray')
axes[1].set_title("Tissue Mask")
axes[1].axis('off')

plt.show()
```

## 建立自訂矩形遮罩

定義特定的感興趣區域：

```python
from histolab.masks import BinaryMask
import numpy as np

class RectangularMask(BinaryMask):
    def __init__(self, x_start, y_start, width, height):
        self.x_start = x_start
        self.y_start = y_start
        self.width = width
        self.height = height

    def _mask(self, obj):
        # 建立具有指定矩形區域的遮罩
        thumb = obj.thumbnail
        mask = np.zeros(thumb.shape[:2], dtype=bool)
        mask[self.y_start:self.y_start+self.height,
             self.x_start:self.x_start+self.width] = True
        return mask

# 使用自訂遮罩
roi_mask = RectangularMask(x_start=1000, y_start=500, width=2000, height=1500)
```

## 排除註記

病理學載玻片通常包含標記筆痕跡或數位註記。使用自訂遮罩排除它們：

```python
from histolab.masks import TissueMask
from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
from histolab.filters.morphological_filters import BinaryDilation

class AnnotationExclusionMask(BinaryMask):
    def _mask(self, obj):
        thumb = obj.thumbnail

        # 轉換為 HSV 以偵測標記筆痕跡（通常為藍色/綠色）
        hsv = cv2.cvtColor(np.array(thumb), cv2.COLOR_RGB2HSV)

        # 定義標記筆痕跡的色彩範圍
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # 建立排除標記筆痕跡的遮罩
        pen_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 應用標準組織偵測
        tissue_mask = TissueMask()(obj)

        # 合併：保留組織，排除標記筆痕跡
        final_mask = tissue_mask & ~pen_mask.astype(bool)

        return final_mask
```

## 與切片擷取的整合

遮罩透過 `extraction_mask` 參數與切片擷取器無縫整合：

```python
from histolab.tiler import RandomTiler
from histolab.masks import TissueMask, BiggestTissueBoxMask

# 使用 TissueMask 從所有組織擷取
random_tiler = RandomTiler(
    tile_size=(512, 512),
    n_tiles=100,
    level=0,
    extraction_mask=TissueMask()  # 從所有組織區域擷取
)

# 或使用預設的 BiggestTissueBoxMask
random_tiler = RandomTiler(
    tile_size=(512, 512),
    n_tiles=100,
    level=0,
    extraction_mask=BiggestTissueBoxMask()  # 預設行為
)
```

## 最佳實務

1. **擷取前預覽遮罩**：使用 `locate_mask()` 或手動視覺化驗證遮罩品質
2. **選擇適當的遮罩類型**：多個組織切片使用 `TissueMask`，單一主要切片使用 `BiggestTissueBoxMask`
3. **針對特定染色自訂**：不同染色（H&E、IHC）可能需要調整閾值參數
4. **處理人工痕跡**：使用自訂濾波器或遮罩排除標記筆痕跡、氣泡或摺痕
5. **在多樣載玻片上測試**：跨具有不同品質和人工痕跡的載玻片驗證遮罩效能
6. **考慮運算成本**：`TissueMask` 比 `BiggestTissueBoxMask` 更全面但運算密集

## 常見問題與解決方案

### 問題：遮罩包含太多背景
**解決方案：** 調整 Otsu 閾值或增加小物件移除閾值

### 問題：遮罩排除有效組織
**解決方案：** 減少小物件移除閾值或修改膨脹參數

### 問題：多個組織切片，但只捕捉到最大的
**解決方案：** 從 `BiggestTissueBoxMask` 切換到 `TissueMask`

### 問題：標記筆註記包含在遮罩中
**解決方案：** 實作自訂註記排除遮罩（參見上方範例）
