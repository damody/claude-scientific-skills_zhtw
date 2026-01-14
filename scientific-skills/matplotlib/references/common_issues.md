# Matplotlib 常見問題和解決方案

常見 matplotlib 問題的疑難排解指南。

## 顯示和後端問題

### 問題：繪圖不顯示

**問題：** `plt.show()` 沒有顯示任何內容

**解決方案：**
```python
# 1. 檢查後端是否正確設定（用於互動使用）
import matplotlib
print(matplotlib.get_backend())

# 2. 嘗試不同的後端
matplotlib.use('TkAgg')  # 或 'Qt5Agg'、'MacOSX'
import matplotlib.pyplot as plt

# 3. 在 Jupyter 筆記本中，使用魔術命令
%matplotlib inline  # 靜態圖像
# 或
%matplotlib widget  # 互動式繪圖

# 4. 確保呼叫 plt.show()
plt.plot([1, 2, 3])
plt.show()
```

### 問題："RuntimeError: main thread is not in main loop"

**問題：** 執行緒的互動模式問題

**解決方案：**
```python
# 切換到非互動後端
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 或關閉互動模式
plt.ioff()
```

### 問題：圖形不會互動式更新

**問題：** 變更未反映在互動視窗中

**解決方案：**
```python
# 啟用互動模式
plt.ion()

# 每次變更後重新繪製
plt.plot(x, y)
plt.draw()
plt.pause(0.001)  # 短暫暫停以更新顯示
```

## 佈局和間距問題

### 問題：標籤和標題重疊

**問題：** 標籤、標題或刻度標籤重疊或被截斷

**解決方案：**
```python
# 解決方案 1：Constrained 佈局（建議）
fig, ax = plt.subplots(constrained_layout=True)

# 解決方案 2：Tight 佈局
fig, ax = plt.subplots()
plt.tight_layout()

# 解決方案 3：手動調整邊距
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

# 解決方案 4：儲存時使用 bbox_inches='tight'
plt.savefig('figure.png', bbox_inches='tight')

# 解決方案 5：旋轉長刻度標籤
ax.set_xticklabels(labels, rotation=45, ha='right')
```

### 問題：色彩條影響子圖大小

**問題：** 新增色彩條會縮小繪圖

**解決方案：**
```python
# 解決方案 1：使用 constrained 佈局
fig, ax = plt.subplots(constrained_layout=True)
im = ax.imshow(data)
plt.colorbar(im, ax=ax)

# 解決方案 2：手動指定色彩條尺寸
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)

# 解決方案 3：對於多個子圖，共享色彩條
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax in axes:
    im = ax.imshow(data)
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
```

### 問題：子圖太靠近

**問題：** 多個子圖重疊

**解決方案：**
```python
# 解決方案 1：使用 constrained_layout
fig, axes = plt.subplots(2, 2, constrained_layout=True)

# 解決方案 2：使用 subplots_adjust 調整間距
fig, axes = plt.subplots(2, 2)
plt.subplots_adjust(hspace=0.4, wspace=0.4)

# 解決方案 3：在 tight_layout 中指定間距
plt.tight_layout(h_pad=2.0, w_pad=2.0)
```

## 記憶體和效能問題

### 問題：多個圖形的記憶體洩漏

**問題：** 建立多個圖形時記憶體使用量增加

**解決方案：**
```python
# 明確關閉圖形
fig, ax = plt.subplots()
ax.plot(x, y)
plt.savefig('plot.png')
plt.close(fig)  # 或 plt.close('all')

# 清除當前圖形而不關閉
plt.clf()

# 清除當前座標軸
plt.cla()
```

### 問題：檔案太大

**問題：** 儲存的圖形太大

**解決方案：**
```python
# 解決方案 1：降低 DPI
plt.savefig('figure.png', dpi=150)  # 而不是 300

# 解決方案 2：對複雜繪圖使用點陣化
ax.plot(x, y, rasterized=True)

# 解決方案 3：對簡單繪圖使用向量格式
plt.savefig('figure.pdf')  # 或 .svg

# 解決方案 4：壓縮 PNG
plt.savefig('figure.png', dpi=300, optimize=True)
```

### 問題：大型資料集繪圖緩慢

**問題：** 繪製多點時花費太長時間

**解決方案：**
```python
# 解決方案 1：降採樣資料
from scipy.signal import decimate
y_downsampled = decimate(y, 10)  # 每 10 個點保留一個

# 解決方案 2：使用點陣化
ax.plot(x, y, rasterized=True)

# 解決方案 3：使用線條簡化
ax.plot(x, y)
for line in ax.get_lines():
    line.set_rasterized(True)

# 解決方案 4：對於散點圖，考慮使用 hexbin 或 2D 直方圖
ax.hexbin(x, y, gridsize=50, cmap='viridis')
```

## 字型和文字問題

### 問題：字型警告

**問題：** "findfont: Font family [...] not found"

**解決方案：**
```python
# 解決方案 1：使用可用字型
from matplotlib.font_manager import findfont, FontProperties
print(findfont(FontProperties(family='sans-serif')))

# 解決方案 2：重建字型快取
import matplotlib.font_manager
matplotlib.font_manager._rebuild()

# 解決方案 3：抑制警告
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 解決方案 4：指定備用字型
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
```

### 問題：LaTeX 渲染錯誤

**問題：** 數學文字無法正確渲染

**解決方案：**
```python
# 解決方案 1：使用帶 r 前綴的原始字串
ax.set_xlabel(r'$\alpha$')  # 不是 '\alpha'

# 解決方案 2：在一般字串中跳脫反斜線
ax.set_xlabel('$\\alpha$')

# 解決方案 3：如果未安裝 LaTeX 則停用它
plt.rcParams['text.usetex'] = False

# 解決方案 4：使用 mathtext 而不是完整的 LaTeX
# mathtext 始終可用，不需要安裝 LaTeX
ax.text(x, y, r'$\int_0^\infty e^{-x} dx$')
```

### 問題：文字被截斷或超出圖形

**問題：** 標籤或註釋出現在圖形邊界外

**解決方案：**
```python
# 解決方案 1：使用 bbox_inches='tight'
plt.savefig('figure.png', bbox_inches='tight')

# 解決方案 2：調整圖形邊界
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)

# 解決方案 3：將文字裁剪到座標軸
ax.text(x, y, 'text', clip_on=True)

# 解決方案 4：使用 constrained_layout
fig, ax = plt.subplots(constrained_layout=True)
```

## 顏色和色彩映射問題

### 問題：色彩條與繪圖不匹配

**問題：** 色彩條顯示的範圍與資料不同

**解決方案：**
```python
# 明確設定 vmin 和 vmax
im = ax.imshow(data, vmin=0, vmax=1, cmap='viridis')
plt.colorbar(im, ax=ax)

# 或對多個繪圖使用相同的 norm
import matplotlib.colors as mcolors
norm = mcolors.Normalize(vmin=data.min(), vmax=data.max())
im1 = ax1.imshow(data1, norm=norm, cmap='viridis')
im2 = ax2.imshow(data2, norm=norm, cmap='viridis')
```

### 問題：顏色看起來不對

**問題：** 繪圖中出現意外顏色

**解決方案：**
```python
# 解決方案 1：檢查顏色指定格式
ax.plot(x, y, color='blue')  # 正確
ax.plot(x, y, color=(0, 0, 1))  # 正確 RGB
ax.plot(x, y, color='#0000FF')  # 正確十六進位

# 解決方案 2：驗證色彩映射是否存在
print(plt.colormaps())  # 列出可用色彩映射

# 解決方案 3：對於散點圖，確保 c 形狀匹配
ax.scatter(x, y, c=colors)  # colors 應與 x、y 長度相同

# 解決方案 4：檢查 alpha 是否正確設定
ax.plot(x, y, alpha=1.0)  # 0=透明，1=不透明
```

### 問題：色彩映射方向相反

**問題：** 色彩映射方向錯誤

**解決方案：**
```python
# 新增 _r 後綴以反轉任何色彩映射
ax.imshow(data, cmap='viridis_r')
```

## 座標軸和比例問題

### 問題：座標軸範圍不起作用

**問題：** `set_xlim` 或 `set_ylim` 不生效

**解決方案：**
```python
# 解決方案 1：在繪圖後設定
ax.plot(x, y)
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)

# 解決方案 2：停用自動縮放
ax.autoscale(False)
ax.set_xlim(0, 10)

# 解決方案 3：使用 axis 方法
ax.axis([xmin, xmax, ymin, ymax])
```

### 問題：對數比例與零或負值

**問題：** 當資料 <= 0 時使用對數比例會產生 ValueError

**解決方案：**
```python
# 解決方案 1：過濾非正值
mask = (data > 0)
ax.plot(x[mask], data[mask])
ax.set_yscale('log')

# 解決方案 2：對同時有正負值的資料使用 symlog
ax.set_yscale('symlog')

# 解決方案 3：新增小偏移量
ax.plot(x, data + 1e-10)
ax.set_yscale('log')
```

### 問題：日期顯示不正確

**問題：** 日期軸顯示數字而不是日期

**解決方案：**
```python
import matplotlib.dates as mdates
import pandas as pd

# 如需要則轉換為 datetime
dates = pd.to_datetime(date_strings)

ax.plot(dates, values)

# 格式化日期軸
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=45)
```

## 圖例問題

### 問題：圖例遮擋資料

**問題：** 圖例遮住繪圖的重要部分

**解決方案：**
```python
# 解決方案 1：使用 'best' 位置
ax.legend(loc='best')

# 解決方案 2：放在繪圖區域外
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 解決方案 3：使圖例半透明
ax.legend(framealpha=0.7)

# 解決方案 4：將圖例放在繪圖下方
ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
```

### 問題：圖例中項目太多

**問題：** 圖例因太多項目而雜亂

**解決方案：**
```python
# 解決方案 1：只標記選定項目
for i, (x, y) in enumerate(data):
    label = f'Data {i}' if i % 5 == 0 else None
    ax.plot(x, y, label=label)

# 解決方案 2：使用多欄
ax.legend(ncol=3)

# 解決方案 3：建立較少項目的自訂圖例
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='r'),
                Line2D([0], [0], color='b')]
ax.legend(custom_lines, ['Category A', 'Category B'])

# 解決方案 4：使用單獨的圖例圖形
fig_leg = plt.figure(figsize=(3, 2))
ax_leg = fig_leg.add_subplot(111)
ax_leg.legend(*ax.get_legend_handles_labels(), loc='center')
ax_leg.axis('off')
```

## 3D 繪圖問題

### 問題：3D 繪圖看起來扁平

**問題：** 3D 繪圖中難以感知深度

**解決方案：**
```python
# 解決方案 1：調整視角
ax.view_init(elev=30, azim=45)

# 解決方案 2：新增網格線
ax.grid(True)

# 解決方案 3：使用顏色表示深度
scatter = ax.scatter(x, y, z, c=z, cmap='viridis')

# 解決方案 4：互動式旋轉（如果使用互動後端）
# 使用者可以點擊並拖動來旋轉
```

### 問題：3D 軸標籤被截斷

**問題：** 3D 軸標籤出現在圖形外

**解決方案：**
```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)

# 新增內邊距
fig.tight_layout(pad=3.0)

# 或使用緊密邊界框儲存
plt.savefig('3d_plot.png', bbox_inches='tight', pad_inches=0.5)
```

## 圖像和色彩條問題

### 問題：圖像顯示翻轉

**問題：** 圖像方向錯誤

**解決方案：**
```python
# 設定 origin 參數
ax.imshow(img, origin='lower')  # 或 'upper'（預設）

# 或翻轉陣列
ax.imshow(np.flipud(img))
```

### 問題：圖像看起來像素化

**問題：** 圖像放大時看起來有鋸齒

**解決方案：**
```python
# 解決方案 1：使用插值
ax.imshow(img, interpolation='bilinear')
# 選項：'nearest'、'bilinear'、'bicubic'、'spline16'、'spline36' 等

# 解決方案 2：儲存時增加 DPI
plt.savefig('figure.png', dpi=300)

# 解決方案 3：如果適當的話使用向量格式
plt.savefig('figure.pdf')
```

## 常見錯誤和修復

### "TypeError: 'AxesSubplot' object is not subscriptable"

**問題：** 嘗試索引單一座標軸
```python
# 錯誤
fig, ax = plt.subplots()
ax[0].plot(x, y)  # 錯誤！

# 正確
fig, ax = plt.subplots()
ax.plot(x, y)
```

### "ValueError: x and y must have same first dimension"

**問題：** 資料陣列長度不匹配
```python
# 檢查形狀
print(f"x shape: {x.shape}, y shape: {y.shape}")

# 確保它們匹配
assert len(x) == len(y), "x and y must have same length"
```

### "AttributeError: 'numpy.ndarray' object has no attribute 'plot'"

**問題：** 對陣列而不是座標軸呼叫 plot
```python
# 錯誤
data.plot(x, y)

# 正確
ax.plot(x, y)
# 或對於 pandas
data.plot(ax=ax)
```

## 避免問題的最佳實踐

1. **始終使用物件導向介面** - 避免 pyplot 狀態機
   ```python
   fig, ax = plt.subplots()  # 良好
   ax.plot(x, y)
   ```

2. **使用 constrained_layout** - 防止重疊問題
   ```python
   fig, ax = plt.subplots(constrained_layout=True)
   ```

3. **明確關閉圖形** - 防止記憶體洩漏
   ```python
   plt.close(fig)
   ```

4. **在建立時設定圖形大小** - 比之後調整大小更好
   ```python
   fig, ax = plt.subplots(figsize=(10, 6))
   ```

5. **對數學文字使用原始字串** - 避免跳脫問題
   ```python
   ax.set_xlabel(r'$\alpha$')
   ```

6. **繪圖前檢查資料形狀** - 及早發現大小不匹配
   ```python
   assert len(x) == len(y)
   ```

7. **使用適當的 DPI** - 列印 300，網頁 150
   ```python
   plt.savefig('figure.png', dpi=300)
   ```

8. **測試不同的後端** - 如果出現顯示問題
   ```python
   import matplotlib
   matplotlib.use('TkAgg')
   ```
