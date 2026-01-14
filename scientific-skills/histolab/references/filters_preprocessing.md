# 濾波器與預處理

## 概述

Histolab 提供一套完整的濾波器，用於預處理全切片影像和切片。濾波器可應用於影像的視覺化、品質控制、組織偵測和人工痕跡移除。它們可組合並串連在一起，以建立精密的預處理流水線。

## 濾波器類別

### 影像濾波器
色彩空間轉換、閾值處理和強度調整

### 形態學濾波器
結構操作，如膨脹、侵蝕、開運算和閉運算

### 組合濾波器
用於結合多個濾波器的工具

## 影像濾波器

### RgbToGrayscale

將 RGB 影像轉換為灰階。

```python
from histolab.filters.image_filters import RgbToGrayscale

gray_filter = RgbToGrayscale()
gray_image = gray_filter(rgb_image)
```

**使用案例：**
- 基於強度操作的預處理
- 簡化色彩複雜度
- 作為形態學操作的輸入

### RgbToHsv

將 RGB 轉換為 HSV（色相、飽和度、明度）色彩空間。

```python
from histolab.filters.image_filters import RgbToHsv

hsv_filter = RgbToHsv()
hsv_image = hsv_filter(rgb_image)
```

**使用案例：**
- 基於色彩的組織分割
- 透過色相偵測標記筆痕跡
- 分離有彩色與無彩色內容

### RgbToHed

將 RGB 轉換為 HED（蘇木精-伊紅-DAB）色彩空間以進行染色去卷積。

```python
from histolab.filters.image_filters import RgbToHed

hed_filter = RgbToHed()
hed_image = hed_filter(rgb_image)
```

**使用案例：**
- 分離 H&E 染色成分
- 分析細胞核（蘇木精）與細胞質（伊紅）染色
- 量化染色強度

### OtsuThreshold

應用 Otsu 自動閾值方法建立二值影像。

```python
from histolab.filters.image_filters import OtsuThreshold

otsu_filter = OtsuThreshold()
binary_image = otsu_filter(grayscale_image)
```

**工作原理：**
- 自動決定最佳閾值
- 分離前景與背景
- 最小化類內變異數

**使用案例：**
- 組織偵測
- 細胞核分割
- 建立二值遮罩

### AdaptiveThreshold

應用自適應閾值處理以處理局部強度變化。

```python
from histolab.filters.image_filters import AdaptiveThreshold

adaptive_filter = AdaptiveThreshold(
    block_size=11,      # 局部鄰域大小
    offset=2            # 從平均值減去的常數
)
binary_image = adaptive_filter(grayscale_image)
```

**使用案例：**
- 非均勻照明
- 局部對比度增強
- 處理可變的染色強度

### Invert

反轉影像強度值。

```python
from histolab.filters.image_filters import Invert

invert_filter = Invert()
inverted_image = invert_filter(image)
```

**使用案例：**
- 某些分割演算法的預處理
- 視覺化調整

### StretchContrast

透過延伸強度範圍來增強影像對比度。

```python
from histolab.filters.image_filters import StretchContrast

contrast_filter = StretchContrast()
enhanced_image = contrast_filter(image)
```

**使用案例：**
- 改善低對比度特徵的可見度
- 視覺化預處理
- 增強微弱結構

### HistogramEqualization

均衡化影像直方圖以增強對比度。

```python
from histolab.filters.image_filters import HistogramEqualization

hist_eq_filter = HistogramEqualization()
equalized_image = hist_eq_filter(grayscale_image)
```

**使用案例：**
- 標準化影像對比度
- 顯示隱藏細節
- 特徵擷取的預處理

## 形態學濾波器

### BinaryDilation

擴展二值影像中的白色區域。

```python
from histolab.filters.morphological_filters import BinaryDilation

dilation_filter = BinaryDilation(disk_size=5)
dilated_image = dilation_filter(binary_image)
```

**參數：**
- `disk_size`：結構元素大小（預設：5）

**使用案例：**
- 連接鄰近的組織區域
- 填補小間隙
- 擴展組織遮罩

### BinaryErosion

收縮二值影像中的白色區域。

```python
from histolab.filters.morphological_filters import BinaryErosion

erosion_filter = BinaryErosion(disk_size=5)
eroded_image = erosion_filter(binary_image)
```

**使用案例：**
- 移除小突起
- 分離連接的物件
- 收縮組織邊界

### BinaryOpening

侵蝕後接膨脹（移除小物件）。

```python
from histolab.filters.morphological_filters import BinaryOpening

opening_filter = BinaryOpening(disk_size=3)
opened_image = opening_filter(binary_image)
```

**使用案例：**
- 移除小人工痕跡
- 平滑物件邊界
- 雜訊減少

### BinaryClosing

膨脹後接侵蝕（填補小孔洞）。

```python
from histolab.filters.morphological_filters import BinaryClosing

closing_filter = BinaryClosing(disk_size=5)
closed_image = closing_filter(binary_image)
```

**使用案例：**
- 填補組織區域中的小孔洞
- 連接鄰近物件
- 平滑內部邊界

### RemoveSmallObjects

移除小於閾值的連通元件。

```python
from histolab.filters.morphological_filters import RemoveSmallObjects

remove_small_filter = RemoveSmallObjects(
    area_threshold=500  # 最小面積（像素）
)
cleaned_image = remove_small_filter(binary_image)
```

**使用案例：**
- 移除灰塵和人工痕跡
- 過濾雜訊
- 清理組織遮罩

### RemoveSmallHoles

填補小於閾值的孔洞。

```python
from histolab.filters.morphological_filters import RemoveSmallHoles

fill_holes_filter = RemoveSmallHoles(
    area_threshold=1000  # 要填補的最大孔洞大小
)
filled_image = fill_holes_filter(binary_image)
```

**使用案例：**
- 填補組織中的小間隙
- 建立連續的組織區域
- 移除內部人工痕跡

## 濾波器組合

### 串連濾波器

依序組合多個濾波器：

```python
from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
from histolab.filters.morphological_filters import BinaryDilation, RemoveSmallObjects
from histolab.filters.compositions import Compose

# 建立濾波器流水線
tissue_detection_pipeline = Compose([
    RgbToGrayscale(),
    OtsuThreshold(),
    BinaryDilation(disk_size=5),
    RemoveSmallHoles(area_threshold=1000),
    RemoveSmallObjects(area_threshold=500)
])

# 應用流水線
result = tissue_detection_pipeline(rgb_image)
```

### Lambda 濾波器

內嵌建立自訂濾波器：

```python
from histolab.filters.image_filters import Lambda
import numpy as np

# 自訂亮度調整
brightness_filter = Lambda(lambda img: np.clip(img * 1.2, 0, 255).astype(np.uint8))

# 自訂色彩通道擷取
red_channel_filter = Lambda(lambda img: img[:, :, 0])
```

## 常見預處理流水線

### 標準組織偵測

```python
from histolab.filters.compositions import Compose
from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
from histolab.filters.morphological_filters import (
    BinaryDilation, RemoveSmallHoles, RemoveSmallObjects
)

tissue_detection = Compose([
    RgbToGrayscale(),
    OtsuThreshold(),
    BinaryDilation(disk_size=5),
    RemoveSmallHoles(area_threshold=1000),
    RemoveSmallObjects(area_threshold=500)
])
```

### 標記筆痕跡移除

```python
from histolab.filters.image_filters import RgbToHsv, Lambda
import numpy as np

def remove_pen_marks(hsv_image):
    """移除藍色/綠色標記筆痕跡。"""
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    # 藍色/綠色色相遮罩（常見標記筆顏色）
    pen_mask = ((h > 0.45) & (h < 0.7) & (s > 0.3))
    # 將標記筆區域設為白色
    hsv_image[pen_mask] = [0, 0, 1]
    return hsv_image

pen_removal = Compose([
    RgbToHsv(),
    Lambda(remove_pen_marks)
])
```

### 細胞核增強

```python
from histolab.filters.image_filters import RgbToHed, HistogramEqualization
from histolab.filters.compositions import Compose

nuclei_enhancement = Compose([
    RgbToHed(),
    Lambda(lambda hed: hed[:, :, 0]),  # 擷取蘇木精通道
    HistogramEqualization()
])
```

### 對比度標準化

```python
from histolab.filters.image_filters import StretchContrast, HistogramEqualization

contrast_normalization = Compose([
    RgbToGrayscale(),
    StretchContrast(),
    HistogramEqualization()
])
```

## 將濾波器應用於切片

濾波器可應用於個別切片：

```python
from histolab.tile import Tile
from histolab.filters.image_filters import RgbToGrayscale

# 載入或擷取切片
tile = Tile(image=pil_image, coords=(x, y))

# 應用濾波器
gray_filter = RgbToGrayscale()
filtered_tile = tile.apply_filters(gray_filter)

# 串連多個濾波器
from histolab.filters.compositions import Compose
from histolab.filters.image_filters import StretchContrast

filter_chain = Compose([
    RgbToGrayscale(),
    StretchContrast()
])
processed_tile = tile.apply_filters(filter_chain)
```

## 自訂遮罩濾波器

將自訂濾波器與組織遮罩整合：

```python
from histolab.masks import TissueMask
from histolab.filters.compositions import Compose
from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
from histolab.filters.morphological_filters import BinaryDilation

# 自訂積極組織偵測
aggressive_filters = Compose([
    RgbToGrayscale(),
    OtsuThreshold(),
    BinaryDilation(disk_size=10),  # 較大的膨脹
    RemoveSmallObjects(area_threshold=5000)  # 只移除大型人工痕跡
])

# 使用自訂濾波器建立遮罩
custom_mask = TissueMask(filters=aggressive_filters)
```

## 染色標準化

雖然 histolab 沒有內建的染色標準化功能，但濾波器可用於基本標準化：

```python
from histolab.filters.image_filters import RgbToHed, Lambda
import numpy as np

def normalize_hed(hed_image, target_means=[0.65, 0.70], target_stds=[0.15, 0.13]):
    """簡單的 H&E 標準化。"""
    h_channel = hed_image[:, :, 0]
    e_channel = hed_image[:, :, 1]

    # 標準化蘇木精
    h_normalized = (h_channel - h_channel.mean()) / h_channel.std()
    h_normalized = h_normalized * target_stds[0] + target_means[0]

    # 標準化伊紅
    e_normalized = (e_channel - e_channel.mean()) / e_channel.std()
    e_normalized = e_normalized * target_stds[1] + target_means[1]

    hed_image[:, :, 0] = h_normalized
    hed_image[:, :, 1] = e_normalized

    return hed_image

normalization_pipeline = Compose([
    RgbToHed(),
    Lambda(normalize_hed)
    # 如需要可轉換回 RGB
])
```

## 最佳實務

1. **預覽濾波器**：在應用於切片之前，先在縮圖上視覺化濾波器輸出
2. **有效串連**：合理排序濾波器（例如，在閾值處理前進行色彩轉換）
3. **調整參數**：針對特定組織調整閾值和結構元素大小
4. **使用組合**：使用 `Compose` 建立可重複使用的濾波器流水線
5. **考慮效能**：複雜的濾波器鏈會增加處理時間
6. **在多樣載玻片上驗證**：跨不同掃描器、染色和組織類型測試濾波器
7. **記錄自訂濾波器**：清楚描述自訂流水線的用途和參數

## 品質控制濾波器

### 模糊偵測

```python
from histolab.filters.image_filters import Lambda
import cv2
import numpy as np

def laplacian_blur_score(gray_image):
    """計算拉普拉斯變異數（模糊指標）。"""
    return cv2.Laplacian(np.array(gray_image), cv2.CV_64F).var()

blur_detector = Lambda(lambda img: laplacian_blur_score(
    RgbToGrayscale()(img)
))
```

### 組織覆蓋率

```python
from histolab.filters.image_filters import RgbToGrayscale, OtsuThreshold
from histolab.filters.compositions import Compose

def tissue_coverage(image):
    """計算影像中組織的百分比。"""
    tissue_mask = Compose([
        RgbToGrayscale(),
        OtsuThreshold()
    ])(image)
    return tissue_mask.sum() / tissue_mask.size * 100

coverage_filter = Lambda(tissue_coverage)
```

## 疑難排解

### 問題：組織偵測遺漏有效組織
**解決方案：**
- 降低 `RemoveSmallObjects` 中的 `area_threshold`
- 減少侵蝕/開運算的 disk size
- 嘗試自適應閾值處理而非 Otsu

### 問題：包含太多人工痕跡
**解決方案：**
- 增加 `RemoveSmallObjects` 中的 `area_threshold`
- 加入開運算/閉運算操作
- 針對特定人工痕跡使用自訂色彩過濾

### 問題：組織邊界太粗糙
**解決方案：**
- 加入 `BinaryClosing` 或 `BinaryOpening` 以平滑
- 調整形態學操作的 disk_size

### 問題：染色品質不一致
**解決方案：**
- 應用直方圖均衡化
- 使用自適應閾值處理
- 實作染色標準化流水線
