# 時間序列分割

Aeon 提供演算法將時間序列分割成具有不同特徵的區域，識別變化點和邊界。

## 分割演算法

### 二分分割
- `BinSegmenter` - 遞迴二分分割
  - 在最顯著的變化點處迭代分割序列
  - 參數：`n_segments`、`cost_function`
  - **使用時機**：已知分割數量，層次結構

### 基於分類
- `ClaSPSegmenter` - 分類分數輪廓
  - 使用分類效能來識別邊界
  - 發現分類可區分鄰居的分割
  - **使用時機**：分割具有不同的時間模式

### 快速模式
- `FLUSSSegmenter` - 快速低成本單能語義分割
  - 使用弧交叉的高效語義分割
  - 基於矩陣輪廓
  - **使用時機**：大型時間序列，需要速度和模式發現

### 資訊理論
- `InformationGainSegmenter` - 資訊增益最大化
  - 尋找最大化資訊增益的邊界
  - **使用時機**：分割之間存在統計差異

### 高斯建模
- `GreedyGaussianSegmenter` - 貪婪高斯近似
  - 將分割建模為高斯分布
  - 逐步添加變化點
  - **使用時機**：分割遵循高斯分布

### 階層凝聚
- `EAggloSegmenter` - 自下而上合併方法
  - 透過凝聚估計變化點
  - **使用時機**：需要階層分割結構

### 隱馬可夫模型
- `HMMSegmenter` - 帶 Viterbi 解碼的 HMM
  - 機率狀態分割
  - **使用時機**：分割代表隱藏狀態

### 基於維度
- `HidalgoSegmenter` - 異質內在維度演算法
  - 檢測局部維度的變化
  - **使用時機**：分割之間維度變化

### 基準
- `RandomSegmenter` - 隨機變化點生成
  - **使用時機**：需要虛無假設基準

## 快速開始

```python
from aeon.segmentation import ClaSPSegmenter
import numpy as np

# 建立具有體制變化的時間序列
y = np.concatenate([
    np.sin(np.linspace(0, 10, 100)),      # 分割 1
    np.cos(np.linspace(0, 10, 100)),      # 分割 2
    np.sin(2 * np.linspace(0, 10, 100))   # 分割 3
])

# 分割序列
segmenter = ClaSPSegmenter()
change_points = segmenter.fit_predict(y)

print(f"檢測到的變化點：{change_points}")
```

## 輸出格式

分割器回傳變化點索引：

```python
# change_points = [100, 200]  # 分割之間的邊界
# 這將序列分為：[0:100]、[100:200]、[200:end]
```

## 演算法選擇

- **速度優先**：FLUSSSegmenter、BinSegmenter
- **準確度優先**：ClaSPSegmenter、HMMSegmenter
- **已知分割數**：BinSegmenter 搭配 n_segments 參數
- **未知分割數**：ClaSPSegmenter、InformationGainSegmenter
- **模式變化**：FLUSSSegmenter、ClaSPSegmenter
- **統計變化**：InformationGainSegmenter、GreedyGaussianSegmenter
- **狀態轉換**：HMMSegmenter

## 常見使用案例

### 體制變化檢測
識別時間序列行為根本變化的時間：

```python
from aeon.segmentation import InformationGainSegmenter

segmenter = InformationGainSegmenter(k=3)  # 最多 3 個變化點
change_points = segmenter.fit_predict(stock_prices)
```

### 活動分割
將感測器資料分割成活動：

```python
from aeon.segmentation import ClaSPSegmenter

segmenter = ClaSPSegmenter()
boundaries = segmenter.fit_predict(accelerometer_data)
```

### 季節邊界檢測
在時間序列中找到季節轉換：

```python
from aeon.segmentation import HMMSegmenter

segmenter = HMMSegmenter(n_states=4)  # 4 個季節
segments = segmenter.fit_predict(temperature_data)
```

## 評估指標

使用分割品質指標：

```python
from aeon.benchmarking.metrics.segmentation import (
    count_error,
    hausdorff_error
)

# 計數誤差：變化點數量差異
count_err = count_error(y_true, y_pred)

# Hausdorff：預測與真實點之間的最大距離
hausdorff_err = hausdorff_error(y_true, y_pred)
```

## 最佳實務

1. **標準化資料**：確保變化檢測不被尺度主導
2. **選擇適當的指標**：不同演算法優化不同標準
3. **驗證分割**：視覺化以驗證有意義的邊界
4. **處理雜訊**：分割前考慮平滑處理
5. **領域知識**：如果已知，使用預期的分割數量
6. **參數調整**：調整敏感度參數（閾值、懲罰）

## 視覺化

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(y, label='Time Series')
for cp in change_points:
    plt.axvline(cp, color='r', linestyle='--', label='Change Point')
plt.legend()
plt.show()
```
