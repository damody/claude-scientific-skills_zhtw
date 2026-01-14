# Matplotlib 樣式指南

樣式設定和自訂 matplotlib 視覺化的完整指南。

## 色彩映射

### 色彩映射類別

**1. 感知均勻順序**
最適合從低到高值有序進展的資料。
- `viridis`（預設，色盲友善）
- `plasma`
- `inferno`
- `magma`
- `cividis`（針對色盲觀看者最佳化）

**用法：**
```python
im = ax.imshow(data, cmap='viridis')
scatter = ax.scatter(x, y, c=values, cmap='plasma')
```

**2. 順序**
用於有序資料的傳統色彩映射。
- `Blues`、`Greens`、`Reds`、`Oranges`、`Purples`
- `YlOrBr`、`YlOrRd`、`OrRd`、`PuRd`
- `BuPu`、`GnBu`、`PuBu`、`YlGnBu`

**3. 發散**
最適合具有有意義中心點的資料（例如零、平均值）。
- `coolwarm`（藍到紅）
- `RdBu`（紅-藍）
- `RdYlBu`（紅-黃-藍）
- `RdYlGn`（紅-黃-綠）
- `PiYG`、`PRGn`、`BrBG`、`PuOr`、`RdGy`

**用法：**
```python
# 將色彩映射以零為中心
im = ax.imshow(data, cmap='coolwarm', vmin=-1, vmax=1)
```

**4. 定性**
最適合無固有順序的類別/名義資料。
- `tab10`（10 種不同顏色）
- `tab20`（20 種不同顏色）
- `Set1`、`Set2`、`Set3`
- `Pastel1`、`Pastel2`
- `Dark2`、`Accent`、`Paired`

**用法：**
```python
colors = plt.cm.tab10(np.linspace(0, 1, n_categories))
for i, category in enumerate(categories):
    ax.plot(x, y[i], color=colors[i], label=category)
```

**5. 週期**
最適合週期性資料（例如相位、角度）。
- `twilight`
- `twilight_shifted`
- `hsv`

### 色彩映射最佳實踐

1. **避免 `jet` 色彩映射** - 感知不均勻，容易誤導
2. **使用感知均勻的色彩映射** - `viridis`、`plasma`、`cividis`
3. **考慮色盲使用者** - 使用 `viridis`、`cividis`，或使用色盲模擬器測試
4. **根據資料類型選擇色彩映射**：
   - 順序：遞增/遞減資料
   - 發散：具有有意義中心的資料
   - 定性：類別
5. **反轉色彩映射** - 新增 `_r` 後綴：`viridis_r`、`coolwarm_r`

### 建立自訂色彩映射

```python
from matplotlib.colors import LinearSegmentedColormap

# 從顏色列表
colors = ['blue', 'white', 'red']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# 從 RGB 值
colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # RGB 元組
cmap = LinearSegmentedColormap.from_list('custom', colors)

# 使用自訂色彩映射
ax.imshow(data, cmap=cmap)
```

### 離散色彩映射

```python
import matplotlib.colors as mcolors

# 從連續色彩映射建立離散色彩映射
cmap = plt.cm.viridis
bounds = np.linspace(0, 10, 11)
norm = mcolors.BoundaryNorm(bounds, cmap.N)
im = ax.imshow(data, cmap=cmap, norm=norm)
```

## 樣式表

### 使用內建樣式

```python
# 列出可用樣式
print(plt.style.available)

# 套用樣式
plt.style.use('seaborn-v0_8-darkgrid')

# 套用多個樣式（後面的樣式覆蓋前面的）
plt.style.use(['seaborn-v0_8-whitegrid', 'seaborn-v0_8-poster'])

# 暫時使用樣式
with plt.style.context('ggplot'):
    fig, ax = plt.subplots()
    ax.plot(x, y)
```

### 常用內建樣式

- `default` - Matplotlib 的預設樣式
- `classic` - 經典 matplotlib 外觀（2.0 之前）
- `seaborn-v0_8-*` - Seaborn 風格的樣式
  - `seaborn-v0_8-darkgrid`、`seaborn-v0_8-whitegrid`
  - `seaborn-v0_8-dark`、`seaborn-v0_8-white`
  - `seaborn-v0_8-ticks`、`seaborn-v0_8-poster`、`seaborn-v0_8-talk`
- `ggplot` - ggplot2 風格的樣式
- `bmh` - Bayesian Methods for Hackers 樣式
- `fivethirtyeight` - FiveThirtyEight 樣式
- `grayscale` - 灰階樣式

### 建立自訂樣式表

建立名為 `custom_style.mplstyle` 的檔案：

```
# custom_style.mplstyle

# 圖形
figure.figsize: 10, 6
figure.dpi: 100
figure.facecolor: white

# 字型
font.family: sans-serif
font.sans-serif: Arial, Helvetica
font.size: 12

# 座標軸
axes.labelsize: 14
axes.titlesize: 16
axes.facecolor: white
axes.edgecolor: black
axes.linewidth: 1.5
axes.grid: True
axes.axisbelow: True

# 網格
grid.color: gray
grid.linestyle: --
grid.linewidth: 0.5
grid.alpha: 0.3

# 線條
lines.linewidth: 2
lines.markersize: 8

# 刻度
xtick.labelsize: 10
ytick.labelsize: 10
xtick.direction: in
ytick.direction: in
xtick.major.size: 6
ytick.major.size: 6
xtick.minor.size: 3
ytick.minor.size: 3

# 圖例
legend.fontsize: 12
legend.frameon: True
legend.framealpha: 0.8
legend.fancybox: True

# 儲存圖形
savefig.dpi: 300
savefig.bbox: tight
savefig.facecolor: white
```

載入並使用：
```python
plt.style.use('path/to/custom_style.mplstyle')
```

## rcParams 設定

### 全域設定

```python
import matplotlib.pyplot as plt

# 全域設定
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14

# 或一次更新多個
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'lines.linewidth': 2
})
```

### 暫時設定

```python
# 使用 context manager 進行暫時變更
with plt.rc_context({'font.size': 14, 'lines.linewidth': 2.5}):
    fig, ax = plt.subplots()
    ax.plot(x, y)
```

### 常用 rcParams

**圖形設定：**
```python
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['figure.edgecolor'] = 'white'
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.constrained_layout.use'] = True
```

**字型設定：**
```python
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'normal'
```

**座標軸設定：**
```python
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.spines.top'] = True
plt.rcParams['axes.spines.right'] = True
```

**線條設定：**
```python
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.linestyle'] = '-'
plt.rcParams['lines.marker'] = 'None'
plt.rcParams['lines.markersize'] = 6
```

**儲存設定：**
```python
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'png'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1
plt.rcParams['savefig.transparent'] = False
```

## 顏色調色板

### 命名顏色集

```python
# Tableau 顏色
tableau_colors = plt.cm.tab10.colors

# CSS4 顏色（子集）
css_colors = ['steelblue', 'coral', 'teal', 'goldenrod', 'crimson']

# 手動定義
custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
```

### 顏色循環

```python
# 設定預設顏色循環
from cycler import cycler
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)

# 或組合顏色和線條樣式
plt.rcParams['axes.prop_cycle'] = cycler(color=colors) + cycler(linestyle=['-', '--', ':', '-.'])
```

### 調色板生成

```python
# 從色彩映射中均勻間隔的顏色
n_colors = 5
colors = plt.cm.viridis(np.linspace(0, 1, n_colors))

# 在繪圖中使用
for i, (x, y) in enumerate(data):
    ax.plot(x, y, color=colors[i])
```

## 字型排印

### 字型設定

```python
# 設定字型系列
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

# 或無襯線
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']

# 或等寬
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.monospace'] = ['Courier New', 'DejaVu Sans Mono']
```

### 文字中的字型屬性

```python
from matplotlib import font_manager

# 指定字型屬性
ax.text(x, y, 'Text',
        fontsize=14,
        fontweight='bold',  # 'normal'、'bold'、'heavy'、'light'
        fontstyle='italic',  # 'normal'、'italic'、'oblique'
        fontfamily='serif')

# 使用特定字型檔案
prop = font_manager.FontProperties(fname='path/to/font.ttf')
ax.text(x, y, 'Text', fontproperties=prop)
```

### 數學文字

```python
# LaTeX 風格數學
ax.set_title(r'$\alpha > \beta$')
ax.set_xlabel(r'$\mu \pm \sigma$')
ax.text(x, y, r'$\int_0^\infty e^{-x} dx = 1$')

# 上標和下標
ax.set_ylabel(r'$y = x^2 + 2x + 1$')
ax.text(x, y, r'$x_1, x_2, \ldots, x_n$')

# 希臘字母
ax.text(x, y, r'$\alpha, \beta, \gamma, \delta, \epsilon$')
```

### 使用完整 LaTeX

```python
# 啟用完整 LaTeX 渲染（需要安裝 LaTeX）
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

ax.set_title(r'\textbf{Bold Title}')
ax.set_xlabel(r'Time $t$ (s)')
```

## 邊框和網格

### 邊框自訂

```python
# 隱藏特定邊框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 移動邊框位置
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('data', 0))

# 改變邊框顏色和寬度
ax.spines['left'].set_color('red')
ax.spines['bottom'].set_linewidth(2)
```

### 網格自訂

```python
# 基本網格
ax.grid(True)

# 自訂網格
ax.grid(True, which='major', linestyle='--', linewidth=0.8, alpha=0.3)
ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2)

# 特定軸的網格
ax.grid(True, axis='x')  # 只有垂直線
ax.grid(True, axis='y')  # 只有水平線

# 網格在資料後面或前面
ax.set_axisbelow(True)  # 網格在資料後面
```

## 圖例自訂

### 圖例位置

```python
# 位置字串
ax.legend(loc='best')  # 自動最佳位置
ax.legend(loc='upper right')
ax.legend(loc='upper left')
ax.legend(loc='lower right')
ax.legend(loc='lower left')
ax.legend(loc='center')
ax.legend(loc='upper center')
ax.legend(loc='lower center')
ax.legend(loc='center left')
ax.legend(loc='center right')

# 精確定位（bbox_to_anchor）
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 繪圖區域外
ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)  # 繪圖下方
```

### 圖例樣式

```python
ax.legend(
    fontsize=12,
    frameon=True,           # 顯示框架
    framealpha=0.9,         # 框架透明度
    fancybox=True,          # 圓角
    shadow=True,            # 陰影效果
    ncol=2,                 # 欄數
    title='Legend Title',   # 圖例標題
    title_fontsize=14,      # 標題字型大小
    edgecolor='black',      # 框架邊緣顏色
    facecolor='white'       # 框架背景顏色
)
```

### 自訂圖例項目

```python
from matplotlib.lines import Line2D

# 建立自訂圖例控制代碼
custom_lines = [Line2D([0], [0], color='red', lw=2),
                Line2D([0], [0], color='blue', lw=2, linestyle='--'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10)]

ax.legend(custom_lines, ['Label 1', 'Label 2', 'Label 3'])
```

## 佈局和間距

### Constrained 佈局

```python
# 首選方法（自動調整）
fig, axes = plt.subplots(2, 2, constrained_layout=True)
```

### Tight 佈局

```python
# 替代方法
fig, axes = plt.subplots(2, 2)
plt.tight_layout(pad=1.5, h_pad=2.0, w_pad=2.0)
```

### 手動調整

```python
# 精細控制
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1,
                    hspace=0.3, wspace=0.4)
```

## 專業出版樣式

出版品質圖形的範例設定：

```python
# 出版樣式設定
plt.rcParams.update({
    # 圖形
    'figure.figsize': (8, 6),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,

    # 字型
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 11,

    # 座標軸
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.linewidth': 1.5,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,

    # 線條
    'lines.linewidth': 2,
    'lines.markersize': 8,

    # 刻度
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',

    # 圖例
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': 'black'
})
```

## 深色主題

```python
# 深色背景樣式
plt.style.use('dark_background')

# 或手動設定
plt.rcParams.update({
    'figure.facecolor': '#1e1e1e',
    'axes.facecolor': '#1e1e1e',
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'white',
    'text.color': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'grid.color': 'gray',
    'legend.facecolor': '#1e1e1e',
    'legend.edgecolor': 'white'
})
```

## 顏色無障礙設計

### 色盲友善調色板

```python
# 使用色盲友善的色彩映射
colorblind_friendly = ['viridis', 'plasma', 'cividis']

# 色盲友善的離散顏色
cb_colors = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC',
             '#CA9161', '#949494', '#ECE133', '#56B4E9']

# 使用模擬工具測試或使用這些經過驗證的調色板
```

### 高對比度

```python
# 確保足夠的對比度
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
```
