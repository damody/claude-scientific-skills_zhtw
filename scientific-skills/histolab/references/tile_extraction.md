# 切片擷取

## 概述

切片擷取是從大型全切片影像裁剪較小、易管理區域的過程。Histolab 提供三種主要的擷取策略，各適合不同的分析需求。所有切片擷取器共享通用參數，並提供預覽和擷取切片的方法。

## 通用參數

所有切片擷取器類別接受這些參數：

```python
tile_size: tuple = (512, 512)           # 切片尺寸（寬度、高度）
level: int = 0                          # 擷取的金字塔層級（0=最高解析度）
check_tissue: bool = True               # 依組織含量過濾切片
tissue_percent: float = 80.0            # 最低組織覆蓋率（0-100）
pixel_overlap: int = 0                  # 相鄰切片間的重疊像素（僅 GridTiler）
prefix: str = ""                        # 儲存切片檔名的前綴
suffix: str = ".png"                    # 儲存切片的副檔名
extraction_mask: BinaryMask = BiggestTissueBoxMask()  # 定義擷取區域的遮罩
```

## RandomTiler

**用途：** 從組織區域擷取固定數量的隨機位置切片。

```python
from histolab.tiler import RandomTiler

random_tiler = RandomTiler(
    tile_size=(512, 512),
    n_tiles=100,                # 要擷取的隨機切片數量
    level=0,
    seed=42,                    # 用於可重現性的隨機種子
    check_tissue=True,
    tissue_percent=80.0
)

# 擷取切片
random_tiler.extract(slide, extraction_mask=TissueMask())
```

**主要參數：**
- `n_tiles`：要擷取的隨機切片數量
- `seed`：用於可重現切片選擇的隨機種子
- `max_iter`：尋找有效切片的最大嘗試次數（預設 1000）

**使用案例：**
- 載玻片內容的探索性分析
- 抽樣多樣區域以建立訓練資料
- 快速評估組織特性
- 從多個載玻片建立平衡的資料集

**優點：**
- 運算效率高
- 適合抽樣多樣的組織形態
- 使用種子參數可重現
- 執行快速

**限制：**
- 可能遺漏稀有的組織模式
- 無法保證覆蓋率
- 隨機分佈可能無法捕捉結構化特徵

## GridTiler

**用途：** 依照網格模式系統性地擷取組織區域的切片。

```python
from histolab.tiler import GridTiler

grid_tiler = GridTiler(
    tile_size=(512, 512),
    level=0,
    check_tissue=True,
    tissue_percent=80.0,
    pixel_overlap=0             # 相鄰切片間的重疊像素
)

# 擷取切片
grid_tiler.extract(slide)
```

**主要參數：**
- `pixel_overlap`：相鄰切片間的重疊像素數
  - `pixel_overlap=0`：非重疊切片
  - `pixel_overlap=128`：每側 128 像素重疊
  - 可用於滑動視窗方法

**使用案例：**
- 全面的載玻片覆蓋
- 需要位置資訊的空間分析
- 從切片重建影像
- 語義分割任務
- 基於區域的分析

**優點：**
- 完整的組織覆蓋
- 保留空間關係
- 可預測的切片位置
- 適合全切片分析

**限制：**
- 大型載玻片運算密集
- 可能產生許多含大量背景的切片（透過 `check_tissue` 緩解）
- 較大的輸出資料集

**網格模式：**
```
[Tile 1][Tile 2][Tile 3]
[Tile 4][Tile 5][Tile 6]
[Tile 7][Tile 8][Tile 9]
```

使用 `pixel_overlap=64`：
```
[Tile 1-overlap-Tile 2-overlap-Tile 3]
[    overlap       overlap       overlap]
[Tile 4-overlap-Tile 5-overlap-Tile 6]
```

## ScoreTiler

**用途：** 根據自訂評分函數擷取排名最高的切片。

```python
from histolab.tiler import ScoreTiler
from histolab.scorer import NucleiScorer

score_tiler = ScoreTiler(
    tile_size=(512, 512),
    n_tiles=50,                 # 要擷取的最高分切片數量
    level=0,
    scorer=NucleiScorer(),      # 評分函數
    check_tissue=True
)

# 擷取最高分切片
score_tiler.extract(slide)
```

**主要參數：**
- `n_tiles`：要擷取的最高分切片數量
- `scorer`：評分函數（例如 `NucleiScorer`、`CellularityScorer`、自訂評分器）

**使用案例：**
- 擷取最具資訊性的區域
- 優先處理具特定特徵的切片（細胞核、細胞等）
- 基於品質的切片選擇
- 聚焦於診斷相關區域
- 訓練資料策展

**優點：**
- 聚焦於最具資訊性的切片
- 在維持品質的同時減少資料集大小
- 可使用不同評分器自訂
- 對目標分析有效率

**限制：**
- 比 RandomTiler 慢（必須評分所有候選切片）
- 需要適合任務的評分器
- 可能遺漏低分但相關的區域

## 可用的評分器

### NucleiScorer

根據細胞核偵測和密度評分切片。

```python
from histolab.scorer import NucleiScorer

nuclei_scorer = NucleiScorer()
```

**工作原理：**
1. 將切片轉換為灰階
2. 應用閾值處理以偵測細胞核
3. 計算類似細胞核的結構
4. 根據細胞核密度指定分數

**最適合：**
- 細胞豐富的組織區域
- 腫瘤偵測
- 有絲分裂分析
- 高細胞含量區域

### CellularityScorer

根據整體細胞含量評分切片。

```python
from histolab.scorer import CellularityScorer

cellularity_scorer = CellularityScorer()
```

**最適合：**
- 識別細胞區域與基質區域
- 腫瘤細胞性評估
- 分離密集與稀疏組織區域

### 自訂評分器

針對特定需求建立自訂評分函數：

```python
from histolab.scorer import Scorer
import numpy as np

class ColorVarianceScorer(Scorer):
    def __call__(self, tile):
        """根據色彩變異數評分切片。"""
        tile_array = np.array(tile.image)
        # 計算色彩變異數
        variance = np.var(tile_array, axis=(0, 1)).sum()
        return variance

# 使用自訂評分器
variance_scorer = ColorVarianceScorer()
score_tiler = ScoreTiler(
    tile_size=(512, 512),
    n_tiles=30,
    scorer=variance_scorer
)
```

## 使用 locate_tiles() 預覽切片

在擷取前預覽切片位置以驗證切片擷取器設定：

```python
# 預覽隨機切片位置
random_tiler.locate_tiles(
    slide=slide,
    extraction_mask=TissueMask(),
    n_tiles=20  # 要預覽的切片數量（用於 RandomTiler）
)
```

這會在載玻片縮圖上顯示彩色矩形，指示切片位置。

## 擷取工作流程

### 基本擷取

```python
from histolab.slide import Slide
from histolab.tiler import RandomTiler

# 載入載玻片
slide = Slide("slide.svs", processed_path="output/tiles/")

# 設定切片擷取器
tiler = RandomTiler(
    tile_size=(512, 512),
    n_tiles=100,
    level=0,
    seed=42
)

# 擷取切片（儲存到 processed_path）
tiler.extract(slide)
```

### 使用日誌記錄的擷取

```python
import logging

# 啟用日誌記錄
logging.basicConfig(level=logging.INFO)

# 擷取切片並顯示進度資訊
tiler.extract(slide)
# 輸出：INFO: Tile 1/100 saved...
# 輸出：INFO: Tile 2/100 saved...
```

### 使用報告的擷取

```python
# 生成包含切片資訊的 CSV 報告
score_tiler = ScoreTiler(
    tile_size=(512, 512),
    n_tiles=50,
    scorer=NucleiScorer()
)

# 擷取並儲存報告
score_tiler.extract(slide, report_path="tiles_report.csv")

# 報告包含：切片名稱、座標、分數、組織百分比
```

報告格式：
```csv
tile_name,x_coord,y_coord,level,score,tissue_percent
tile_001.png,10240,5120,0,0.89,95.2
tile_002.png,15360,7680,0,0.85,91.7
...
```

## 進階擷取模式

### 多層級擷取

在不同放大層級擷取切片：

```python
# 高解析度切片（層級 0）
high_res_tiler = RandomTiler(tile_size=(512, 512), n_tiles=50, level=0)
high_res_tiler.extract(slide)

# 中解析度切片（層級 1）
med_res_tiler = RandomTiler(tile_size=(512, 512), n_tiles=50, level=1)
med_res_tiler.extract(slide)

# 低解析度切片（層級 2）
low_res_tiler = RandomTiler(tile_size=(512, 512), n_tiles=50, level=2)
low_res_tiler.extract(slide)
```

### 階層式擷取

從相同位置擷取多個尺度：

```python
# 在層級 0 擷取隨機位置
random_tiler_l0 = RandomTiler(
    tile_size=(512, 512),
    n_tiles=30,
    level=0,
    seed=42,
    prefix="level0_"
)
random_tiler_l0.extract(slide)

# 在層級 1 擷取相同位置（使用相同種子）
random_tiler_l1 = RandomTiler(
    tile_size=(512, 512),
    n_tiles=30,
    level=1,
    seed=42,
    prefix="level1_"
)
random_tiler_l1.extract(slide)
```

### 自訂切片過濾

擷取後應用額外過濾：

```python
from PIL import Image
import numpy as np
from pathlib import Path

def filter_blurry_tiles(tile_dir, threshold=100):
    """使用拉普拉斯變異數移除模糊切片。"""
    for tile_path in Path(tile_dir).glob("*.png"):
        img = Image.open(tile_path)
        gray = np.array(img.convert('L'))
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var < threshold:
            tile_path.unlink()  # 移除模糊切片
            print(f"Removed blurry tile: {tile_path.name}")

# 擷取後使用
tiler.extract(slide)
filter_blurry_tiles("output/tiles/")
```

## 最佳實務

1. **擷取前預覽**：務必使用 `locate_tiles()` 驗證切片放置
2. **使用適當層級**：將擷取層級與分析解析度需求相匹配
3. **設定 tissue_percent 閾值**：根據染色和組織類型調整（通常 70-90%）
4. **選擇正確的切片擷取器**：
   - RandomTiler 用於抽樣和探索
   - GridTiler 用於全面覆蓋
   - ScoreTiler 用於目標、品質導向的擷取
5. **啟用日誌記錄**：監控大型資料集的擷取進度
6. **使用種子確保可重現性**：在 RandomTiler 中設定隨機種子
7. **考慮儲存空間**：GridTiler 每張載玻片可產生數千個切片
8. **驗證切片品質**：檢查擷取的切片是否有人工痕跡、模糊或對焦問題

## 效能最佳化

1. **在適當層級擷取**：較低層級（1、2）擷取更快
2. **調整 tissue_percent**：較高閾值減少無效切片嘗試
3. **使用 BiggestTissueBoxMask**：對單一組織切片比 TissueMask 更快
4. **限制 n_tiles**：用於 RandomTiler 和 ScoreTiler
5. **使用 pixel_overlap=0**：用於非重疊 GridTiler 擷取

## 疑難排解

### 問題：未擷取到切片
**解決方案：**
- 降低 `tissue_percent` 閾值
- 驗證載玻片包含組織（檢查縮圖）
- 確保 extraction_mask 捕捉到組織區域
- 檢查 tile_size 是否適合載玻片解析度

### 問題：擷取許多背景切片
**解決方案：**
- 啟用 `check_tissue=True`
- 提高 `tissue_percent` 閾值
- 使用適當的遮罩（TissueMask vs. BiggestTissueBoxMask）

### 問題：擷取非常緩慢
**解決方案：**
- 在較低金字塔層級擷取（level=1 或 2）
- 減少 RandomTiler/ScoreTiler 的 `n_tiles`
- 使用 RandomTiler 而非 GridTiler 進行抽樣
- 使用 BiggestTissueBoxMask 而非 TissueMask

### 問題：切片重疊太多（GridTiler）
**解決方案：**
- 設定 `pixel_overlap=0` 進行非重疊切片
- 減少 `pixel_overlap` 值
