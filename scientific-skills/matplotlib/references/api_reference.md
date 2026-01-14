# Matplotlib API 參考

本文件提供最常用 matplotlib 類別和方法的快速參考。

## 核心類別

### Figure

所有繪圖元素的頂層容器。

**建立：**
```python
fig = plt.figure(figsize=(10, 6), dpi=100, facecolor='white')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
```

**主要方法：**
- `fig.add_subplot(nrows, ncols, index)` - 新增子圖
- `fig.add_axes([left, bottom, width, height])` - 在特定位置新增座標軸
- `fig.savefig(filename, dpi=300, bbox_inches='tight')` - 儲存圖形
- `fig.tight_layout()` - 調整間距以防止重疊
- `fig.suptitle(title)` - 設定圖形標題
- `fig.legend()` - 建立圖形層級的圖例
- `fig.colorbar(mappable)` - 為圖形新增色彩條
- `plt.close(fig)` - 關閉圖形以釋放記憶體

**主要屬性：**
- `fig.axes` - 圖形中所有座標軸的列表
- `fig.dpi` - 解析度（每英吋點數）
- `fig.figsize` - 圖形尺寸（英吋）（寬度、高度）

### Axes

資料視覺化的實際繪圖區域。

**建立：**
```python
fig, ax = plt.subplots()  # 單一座標軸
ax = fig.add_subplot(111)  # 替代方法
```

**繪圖方法：**

**折線圖：**
- `ax.plot(x, y, **kwargs)` - 折線圖
- `ax.step(x, y, where='pre'/'mid'/'post')` - 階梯圖
- `ax.errorbar(x, y, yerr, xerr)` - 誤差條

**散點圖：**
- `ax.scatter(x, y, s=size, c=color, marker='o', alpha=0.5)` - 散點圖

**長條圖：**
- `ax.bar(x, height, width=0.8, align='center')` - 垂直長條圖
- `ax.barh(y, width)` - 水平長條圖

**統計圖：**
- `ax.hist(data, bins=10, density=False)` - 直方圖
- `ax.boxplot(data, labels=None)` - 箱形圖
- `ax.violinplot(data)` - 小提琴圖

**2D 圖：**
- `ax.imshow(array, cmap='viridis', aspect='auto')` - 顯示圖像/矩陣
- `ax.contour(X, Y, Z, levels=10)` - 等高線
- `ax.contourf(X, Y, Z, levels=10)` - 填充等高線
- `ax.pcolormesh(X, Y, Z)` - 偽彩色圖

**填充：**
- `ax.fill_between(x, y1, y2, alpha=0.3)` - 曲線之間填充
- `ax.fill_betweenx(y, x1, x2)` - 垂直曲線之間填充

**文字和註釋：**
- `ax.text(x, y, text, fontsize=12)` - 新增文字
- `ax.annotate(text, xy=(x, y), xytext=(x2, y2), arrowprops={})` - 帶箭頭的註釋

**自訂方法：**

**標籤和標題：**
- `ax.set_xlabel(label, fontsize=12)` - 設定 x 軸標籤
- `ax.set_ylabel(label, fontsize=12)` - 設定 y 軸標籤
- `ax.set_title(title, fontsize=14)` - 設定座標軸標題

**範圍和比例：**
- `ax.set_xlim(left, right)` - 設定 x 軸範圍
- `ax.set_ylim(bottom, top)` - 設定 y 軸範圍
- `ax.set_xscale('linear'/'log'/'symlog')` - 設定 x 軸比例
- `ax.set_yscale('linear'/'log'/'symlog')` - 設定 y 軸比例

**刻度：**
- `ax.set_xticks(positions)` - 設定 x 刻度位置
- `ax.set_xticklabels(labels)` - 設定 x 刻度標籤
- `ax.tick_params(axis='both', labelsize=10)` - 自訂刻度外觀

**網格和邊框：**
- `ax.grid(True, alpha=0.3, linestyle='--')` - 新增網格
- `ax.spines['top'].set_visible(False)` - 隱藏頂部邊框
- `ax.spines['right'].set_visible(False)` - 隱藏右側邊框

**圖例：**
- `ax.legend(loc='best', fontsize=10, frameon=True)` - 新增圖例
- `ax.legend(handles, labels)` - 自訂圖例

**長寬比和佈局：**
- `ax.set_aspect('equal'/'auto'/ratio)` - 設定長寬比
- `ax.invert_xaxis()` - 反轉 x 軸
- `ax.invert_yaxis()` - 反轉 y 軸

### pyplot 模組

用於快速繪圖的高階介面。

**圖形建立：**
- `plt.figure()` - 建立新圖形
- `plt.subplots()` - 建立圖形和座標軸
- `plt.subplot()` - 為當前圖形新增子圖

**繪圖（使用當前座標軸）：**
- `plt.plot()` - 折線圖
- `plt.scatter()` - 散點圖
- `plt.bar()` - 長條圖
- `plt.hist()` - 直方圖
- （所有座標軸方法皆可用）

**顯示和儲存：**
- `plt.show()` - 顯示圖形
- `plt.savefig()` - 儲存圖形
- `plt.close()` - 關閉圖形

**樣式：**
- `plt.style.use(style_name)` - 套用樣式表
- `plt.style.available` - 列出可用樣式

**狀態管理：**
- `plt.gca()` - 取得當前座標軸
- `plt.gcf()` - 取得當前圖形
- `plt.sca(ax)` - 設定當前座標軸
- `plt.clf()` - 清除當前圖形
- `plt.cla()` - 清除當前座標軸

## 線條和標記樣式

### 線條樣式
- `'-'` 或 `'solid'` - 實線
- `'--'` 或 `'dashed'` - 虛線
- `'-.'` 或 `'dashdot'` - 點劃線
- `':'` 或 `'dotted'` - 點線
- `''` 或 `' '` 或 `'None'` - 無線條

### 標記樣式
- `'.'` - 點標記
- `'o'` - 圓形標記
- `'v'`、`'^'`、`'<'`、`'>'` - 三角形標記
- `'s'` - 正方形標記
- `'p'` - 五邊形標記
- `'*'` - 星形標記
- `'h'`、`'H'` - 六邊形標記
- `'+'` - 加號標記
- `'x'` - X 標記
- `'D'`、`'d'` - 菱形標記

### 顏色指定

**單字元縮寫：**
- `'b'` - 藍色
- `'g'` - 綠色
- `'r'` - 紅色
- `'c'` - 青色
- `'m'` - 洋紅色
- `'y'` - 黃色
- `'k'` - 黑色
- `'w'` - 白色

**命名顏色：**
- `'steelblue'`、`'coral'`、`'teal'` 等
- 完整列表請參閱：https://matplotlib.org/stable/gallery/color/named_colors.html

**其他格式：**
- 十六進位：`'#FF5733'`
- RGB 元組：`(0.1, 0.2, 0.3)`
- RGBA 元組：`(0.1, 0.2, 0.3, 0.5)`

## 常用參數

### 繪圖函數參數

```python
ax.plot(x, y,
    color='blue',           # 線條顏色
    linewidth=2,            # 線條寬度
    linestyle='--',         # 線條樣式
    marker='o',             # 標記樣式
    markersize=8,           # 標記大小
    markerfacecolor='red',  # 標記填充顏色
    markeredgecolor='black',# 標記邊緣顏色
    markeredgewidth=1,      # 標記邊緣寬度
    alpha=0.7,              # 透明度（0-1）
    label='data',           # 圖例標籤
    zorder=2,               # 繪製順序
    rasterized=True         # 點陣化以減少檔案大小
)
```

### 散點函數參數

```python
ax.scatter(x, y,
    s=50,                   # 大小（純量或陣列）
    c='blue',               # 顏色（純量、陣列或序列）
    marker='o',             # 標記樣式
    cmap='viridis',         # 色彩映射（如果 c 是數值）
    alpha=0.5,              # 透明度
    edgecolors='black',     # 邊緣顏色
    linewidths=1,           # 邊緣寬度
    vmin=0, vmax=1,         # 顏色比例範圍
    label='data'            # 圖例標籤
)
```

### 文字參數

```python
ax.text(x, y, text,
    fontsize=12,            # 字型大小
    fontweight='normal',    # 'normal'、'bold'、'heavy'、'light'
    fontstyle='normal',     # 'normal'、'italic'、'oblique'
    fontfamily='sans-serif',# 字型系列
    color='black',          # 文字顏色
    alpha=1.0,              # 透明度
    ha='center',            # 水平對齊：'left'、'center'、'right'
    va='center',            # 垂直對齊：'top'、'center'、'bottom'、'baseline'
    rotation=0,             # 旋轉角度（度）
    bbox=dict(              # 背景框
        facecolor='white',
        edgecolor='black',
        boxstyle='round'
    )
)
```

## rcParams 設定

用於全域自訂的常用 rcParams 設定：

```python
# 字型設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 12

# 圖形設定
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# 座標軸設定
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.alpha'] = 0.3

# 線條設定
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 8

# 刻度設定
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['xtick.direction'] = 'in'  # 'in'、'out'、'inout'
plt.rcParams['ytick.direction'] = 'in'

# 圖例設定
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.8

# 網格設定
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
```

## 用於複雜佈局的 GridSpec

```python
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# 跨越多個單元格
ax1 = fig.add_subplot(gs[0, :])      # 頂列，所有欄
ax2 = fig.add_subplot(gs[1:, 0])     # 底部兩列，第一欄
ax3 = fig.add_subplot(gs[1, 1:])     # 中間列，最後兩欄
ax4 = fig.add_subplot(gs[2, 1])      # 底列，中間欄
ax5 = fig.add_subplot(gs[2, 2])      # 底列，右欄
```

## 3D 繪圖

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 繪圖類型
ax.plot(x, y, z)                    # 3D 線圖
ax.scatter(x, y, z)                 # 3D 散點圖
ax.plot_surface(X, Y, Z)            # 3D 曲面
ax.plot_wireframe(X, Y, Z)          # 3D 線框
ax.contour(X, Y, Z)                 # 3D 等高線
ax.bar3d(x, y, z, dx, dy, dz)       # 3D 長條

# 自訂
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=30, azim=45)      # 設定視角
```

## 動畫

```python
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
line, = ax.plot([], [])

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return line,

def update(frame):
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x + frame/10)
    line.set_data(x, y)
    return line,

anim = FuncAnimation(fig, update, init_func=init,
                     frames=100, interval=50, blit=True)

# 儲存動畫
anim.save('animation.gif', writer='pillow', fps=20)
anim.save('animation.mp4', writer='ffmpeg', fps=20)
```

## 圖像操作

```python
# 讀取和顯示圖像
img = plt.imread('image.png')
ax.imshow(img)

# 將矩陣顯示為圖像
ax.imshow(matrix, cmap='viridis', aspect='auto',
          interpolation='nearest', origin='lower')

# 色彩條
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Values')

# 圖像範圍（設定座標）
ax.imshow(img, extent=[x_min, x_max, y_min, y_max])
```

## 事件處理

```python
# 滑鼠點擊事件
def on_click(event):
    if event.inaxes:
        print(f'Clicked at x={event.xdata:.2f}, y={event.ydata:.2f}')

fig.canvas.mpl_connect('button_press_event', on_click)

# 按鍵事件
def on_key(event):
    print(f'Key pressed: {event.key}')

fig.canvas.mpl_connect('key_press_event', on_key)
```

## 實用工具

```python
# 取得當前座標軸範圍
xlims = ax.get_xlim()
ylims = ax.get_ylim()

# 設定等比例長寬比
ax.set_aspect('equal', adjustable='box')

# 子圖之間共享座標軸
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

# 雙 y 軸
ax2 = ax1.twinx()

# 移除刻度標籤
ax.set_xticklabels([])
ax.set_yticklabels([])

# 科學記號
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

# 日期格式化
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
```
