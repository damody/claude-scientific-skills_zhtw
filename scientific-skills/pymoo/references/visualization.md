# Pymoo 視覺化參考

pymoo 中視覺化功能的完整參考。

## 概述

Pymoo 提供八種視覺化類型用於分析多目標最佳化結果。所有圖表都包裝 matplotlib 並接受標準 matplotlib 關鍵字引數進行自訂。

## 核心視覺化類型

### 1. 散佈圖
**用途：** 視覺化 2D、3D 或更高維度的目標空間
**最適合：** Pareto 前沿、解分布、演算法比較

**用法：**
```python
from pymoo.visualization.scatter import Scatter

# 2D 散佈圖
plot = Scatter()
plot.add(result.F, color="red", label="演算法 A")
plot.add(ref_pareto_front, color="black", alpha=0.3, label="真實 PF")
plot.show()

# 3D 散佈圖
plot = Scatter(title="3D Pareto 前沿")
plot.add(result.F)
plot.show()
```

**參數：**
- `title`：圖表標題
- `figsize`：圖形大小元組（寬度，高度）
- `legend`：顯示圖例（預設：True）
- `labels`：軸標籤列表

**add 方法參數：**
- `color`：顏色指定
- `alpha`：透明度（0-1）
- `s`：標記大小
- `marker`：標記樣式
- `label`：圖例標籤

**N 維投影：**
對於 >3 個目標，自動建立散佈圖矩陣

### 2. 平行座標圖（PCP）
**用途：** 跨多個目標比較多個解
**最適合：** 高維目標問題、比較演算法效能

**機制：** 每個垂直軸代表一個目標，線連接每個解的目標值

**用法：**
```python
from pymoo.visualization.pcp import PCP

plot = PCP()
plot.add(result.F, color="blue", alpha=0.5)
plot.add(reference_set, color="red", alpha=0.8)
plot.show()
```

**參數：**
- `title`：圖表標題
- `figsize`：圖形大小
- `labels`：目標標籤
- `bounds`：每個目標的正規化邊界（最小、最大）
- `normalize_each_axis`：每軸正規化到 [0,1]（預設：True）

**最佳實務：**
- 不同目標尺度時正規化
- 對重疊線使用透明度
- 為清晰度限制解數量（<1000）

### 3. 熱圖
**用途：** 顯示解密度和分布模式
**最適合：** 了解解聚類、識別間隙

**用法：**
```python
from pymoo.visualization.heatmap import Heatmap

plot = Heatmap(title="解密度")
plot.add(result.F)
plot.show()
```

**參數：**
- `bins`：每個維度的區間數（預設：20）
- `cmap`：色彩圖名稱（例如 "viridis"、"plasma"、"hot"）
- `norm`：正規化方法

**解讀：**
- 明亮區域：高解密度
- 暗區域：很少或沒有解
- 揭示分布均勻性

### 4. 花瓣圖
**用途：** 多個目標的放射狀表示
**最適合：** 跨目標比較個別解

**結構：** 每個「花瓣」代表一個目標，長度表示目標值

**用法：**
```python
from pymoo.visualization.petal import Petal

plot = Petal(title="解比較", bounds=[min_vals, max_vals])
plot.add(result.F[0], color="blue", label="解 1")
plot.add(result.F[1], color="red", label="解 2")
plot.show()
```

**參數：**
- `bounds`：每個目標的正規化 [最小、最大]
- `labels`：目標名稱
- `reverse`：反轉特定目標（用於最小化顯示）

**用例：**
- 少數解之間的決策
- 向利害關係人展示權衡

### 5. 雷達圖
**用途：** 多準則效能剖面
**最適合：** 比較解特徵

**類似於：** 花瓣圖但頂點連接

**用法：**
```python
from pymoo.visualization.radar import Radar

plot = Radar(bounds=[min_vals, max_vals])
plot.add(solution_A, label="設計 A")
plot.add(solution_B, label="設計 B")
plot.show()
```

### 6. Radviz
**用途：** 視覺化的降維
**最適合：** 高維資料探索、模式識別

**機制：** 將高維點投影到 2D 圓上，維度錨點在周邊

**用法：**
```python
from pymoo.visualization.radviz import Radviz

plot = Radviz(title="高維解空間")
plot.add(result.F, color="blue", s=30)
plot.show()
```

**參數：**
- `endpoint_style`：錨點視覺化
- `labels`：維度標籤

**解讀：**
- 靠近錨點的點：該維度值高
- 中央點：跨維度平衡
- 聚類：相似解

### 7. 星座標
**用途：** 替代的高維視覺化
**最適合：** 比較多維資料集

**機制：** 每個維度作為從原點的軸，點根據值繪製

**用法：**
```python
from pymoo.visualization.star_coordinate import StarCoordinate

plot = StarCoordinate()
plot.add(result.F)
plot.show()
```

**參數：**
- `axis_style`：軸外觀
- `axis_extension`：軸超出最大值的長度
- `labels`：維度標籤

### 8. 影片/動畫
**用途：** 顯示最佳化隨時間的進展
**最適合：** 了解收斂行為、簡報

**用法：**
```python
from pymoo.visualization.video import Video

# 從演算法歷史建立動畫
anim = Video(result.algorithm)
anim.save("optimization_progress.mp4")
```

**需求：**
- 演算法必須儲存歷史（在 minimize 中使用 `save_history=True`）
- 安裝 ffmpeg 以匯出影片

**自訂：**
- 幀率
- 每幀的圖表類型
- 疊加資訊（世代、超體積等）

## 進階功能

### 多資料集疊加

所有圖表類型支援添加多個資料集：

```python
plot = Scatter(title="演算法比較")
plot.add(nsga2_result.F, color="red", alpha=0.5, label="NSGA-II")
plot.add(nsga3_result.F, color="blue", alpha=0.5, label="NSGA-III")
plot.add(true_pareto_front, color="black", linewidth=2, label="真實 PF")
plot.show()
```

### 自訂樣式

直接傳遞 matplotlib kwargs：

```python
plot = Scatter(
    title="我的結果",
    figsize=(10, 8),
    tight_layout=True
)
plot.add(
    result.F,
    color="red",
    marker="o",
    s=50,
    alpha=0.7,
    edgecolors="black",
    linewidth=0.5
)
```

### 正規化

將目標正規化到 [0,1] 以進行公平比較：

```python
plot = PCP(normalize_each_axis=True, bounds=[min_bounds, max_bounds])
```

### 儲存到檔案

儲存圖表而非顯示：

```python
plot = Scatter()
plot.add(result.F)
plot.save("my_plot.png", dpi=300)
```

## 視覺化選擇指南

**根據以下選擇視覺化：**

| 問題類型 | 主要圖表 | 次要圖表 |
|--------------|--------------|----------------|
| 2 目標 | 散佈圖 | 熱圖 |
| 3 目標 | 3D 散佈圖 | 平行座標 |
| 高維目標（4-10） | 平行座標 | Radviz |
| 高維目標（>10） | Radviz | 星座標 |
| 解比較 | 花瓣/雷達 | 平行座標 |
| 演算法收斂 | 影片 | 散佈圖（最終） |
| 分布分析 | 熱圖 | 散佈圖 |

**組合：**
- 散佈圖 + 熱圖：整體分布 + 密度
- PCP + 花瓣：族群概覽 + 個別解
- 散佈圖 + 影片：最終結果 + 收斂過程

## 常見視覺化工作流程

### 1. 演算法比較
```python
from pymoo.visualization.scatter import Scatter

plot = Scatter(title="ZDT1 上的演算法比較")
plot.add(ga_result.F, color="blue", label="GA", alpha=0.6)
plot.add(nsga2_result.F, color="red", label="NSGA-II", alpha=0.6)
plot.add(zdt1.pareto_front(), color="black", label="真實 PF")
plot.show()
```

### 2. 高維目標分析
```python
from pymoo.visualization.pcp import PCP

plot = PCP(
    title="5 目標 DTLZ2 結果",
    labels=["f1", "f2", "f3", "f4", "f5"],
    normalize_each_axis=True
)
plot.add(result.F, alpha=0.3)
plot.show()
```

### 3. 決策
```python
from pymoo.visualization.petal import Petal

# 比較前 3 個解
candidates = result.F[:3]

plot = Petal(
    title="前 3 個解",
    bounds=[result.F.min(axis=0), result.F.max(axis=0)],
    labels=["成本", "重量", "效率", "安全性"]
)
for i, sol in enumerate(candidates):
    plot.add(sol, label=f"解 {i+1}")
plot.show()
```

### 4. 收斂視覺化
```python
from pymoo.optimize import minimize

# 啟用歷史
result = minimize(
    problem,
    algorithm,
    ('n_gen', 200),
    seed=1,
    save_history=True,
    verbose=False
)

# 建立收斂圖
from pymoo.visualization.scatter import Scatter

plot = Scatter(title="跨世代的收斂")
for gen in [0, 50, 100, 150, 200]:
    F = result.history[gen].opt.get("F")
    plot.add(F, alpha=0.5, label=f"世代 {gen}")
plot.show()
```

## 提示與最佳實務

1. **使用適當的 alpha：** 對於重疊點，使用 `alpha=0.3-0.7`
2. **正規化目標：** 尺度不同？正規化以進行公平視覺化
3. **清楚標籤：** 始終提供有意義的標籤和圖例
4. **限制資料點：** >10000 點？抽樣或使用熱圖
5. **色彩方案：** 使用色盲友善的調色板
6. **儲存高解析度：** 出版物使用 `dpi=300`
7. **互動式探索：** 考慮 plotly 進行互動式圖表
8. **結合視圖：** 顯示多個視角以進行全面分析
