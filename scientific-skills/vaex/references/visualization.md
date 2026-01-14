# 資料視覺化

本參考文件涵蓋 Vaex 的視覺化功能，用於建立大型資料集的圖表、熱圖和互動式視覺化。

## 概述

Vaex 透過高效的分箱和聚合，擅長視覺化數十億列的資料集。視覺化系統直接處理大型資料而無需抽樣，提供整個資料集的準確表示。

**主要功能：**
- 互動式視覺化十億列資料集
- 不需要抽樣 - 使用所有資料
- 自動分箱和聚合
- 與 matplotlib 整合
- Jupyter 的互動式小工具

## 基本繪圖

### 1D 直方圖

```python
import vaex
import matplotlib.pyplot as plt

df = vaex.open('data.hdf5')

# 簡單直方圖
df.plot1d(df.age)

# 帶自訂選項
df.plot1d(df.age,
          limits=[0, 100],
          shape=50,              # 分箱數量
          figsize=(10, 6),
          xlabel='Age',
          ylabel='Count')

plt.show()
```

### 2D 密度圖（熱圖）

```python
# 基本 2D 圖
df.plot(df.x, df.y)

# 帶限制
df.plot(df.x, df.y, limits=[[0, 10], [0, 10]])

# 使用百分位數自動限制
df.plot(df.x, df.y, limits='99.7%')  # 3-sigma 限制

# 自訂形狀（解析度）
df.plot(df.x, df.y, shape=(512, 512))

# 對數色標
df.plot(df.x, df.y, f='log')
```

### 散點圖（小資料）

```python
# 用於較小資料集或樣本
df_sample = df.sample(n=1000)

df_sample.scatter(df_sample.x, df_sample.y,
                  alpha=0.5,
                  s=10)  # 點大小

plt.show()
```

## 進階視覺化選項

### 色標和正規化

```python
# 線性尺度（預設）
df.plot(df.x, df.y, f='identity')

# 對數尺度
df.plot(df.x, df.y, f='log')
df.plot(df.x, df.y, f='log10')

# 平方根尺度
df.plot(df.x, df.y, f='sqrt')

# 自訂色圖
df.plot(df.x, df.y, colormap='viridis')
df.plot(df.x, df.y, colormap='plasma')
df.plot(df.x, df.y, colormap='hot')
```

### 限制和範圍

```python
# 手動限制
df.plot(df.x, df.y, limits=[[xmin, xmax], [ymin, ymax]])

# 基於百分位數的限制
df.plot(df.x, df.y, limits='99.7%')  # 3-sigma
df.plot(df.x, df.y, limits='95%')
df.plot(df.x, df.y, limits='minmax')  # 完整範圍

# 混合限制
df.plot(df.x, df.y, limits=[[0, 100], 'minmax'])
```

### 解析度控制

```python
# 更高解析度（更多分箱）
df.plot(df.x, df.y, shape=(1024, 1024))

# 較低解析度（更快）
df.plot(df.x, df.y, shape=(128, 128))

# 每軸不同解析度
df.plot(df.x, df.y, shape=(512, 256))
```

## 統計視覺化

### 視覺化聚合

```python
# 網格上的平均值
df.plot(df.x, df.y, what=df.z.mean(),
        limits=[[0, 10], [0, 10]],
        shape=(100, 100),
        colormap='viridis')

# 標準差
df.plot(df.x, df.y, what=df.z.std())

# 總和
df.plot(df.x, df.y, what=df.z.sum())

# 計數（預設）
df.plot(df.x, df.y, what='count')
```

### 多個統計

```python
# 建立帶子圖的圖形
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 計數
df.plot(df.x, df.y, what='count',
        ax=axes[0, 0], show=False)
axes[0, 0].set_title('Count')

# 平均值
df.plot(df.x, df.y, what=df.z.mean(),
        ax=axes[0, 1], show=False)
axes[0, 1].set_title('Mean of z')

# 標準差
df.plot(df.x, df.y, what=df.z.std(),
        ax=axes[1, 0], show=False)
axes[1, 0].set_title('Std of z')

# 最小值
df.plot(df.x, df.y, what=df.z.min(),
        ax=axes[1, 1], show=False)
axes[1, 1].set_title('Min of z')

plt.tight_layout()
plt.show()
```

## 使用選擇

同時視覺化資料的不同區段：

```python
import vaex
import matplotlib.pyplot as plt

df = vaex.open('data.hdf5')

# 建立選擇
df.select(df.category == 'A', name='group_a')
df.select(df.category == 'B', name='group_b')

# 繪製兩個選擇
df.plot1d(df.value, selection='group_a', label='Group A')
df.plot1d(df.value, selection='group_b', label='Group B')
plt.legend()
plt.show()

# 帶選擇的 2D 圖
df.plot(df.x, df.y, selection='group_a')
```

### 疊加多個選擇

```python
# 建立基本圖
fig, ax = plt.subplots(figsize=(10, 8))

# 用不同顏色繪製每個選擇
df.plot(df.x, df.y, selection='group_a',
        ax=ax, show=False, colormap='Reds', alpha=0.5)
df.plot(df.x, df.y, selection='group_b',
        ax=ax, show=False, colormap='Blues', alpha=0.5)

ax.set_title('Overlaid Selections')
plt.show()
```

## 子圖和版面配置

### 建立多個圖

```python
import matplotlib.pyplot as plt

# 建立子圖網格
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 繪製不同變數
variables = ['x', 'y', 'z', 'a', 'b', 'c']
for idx, var in enumerate(variables):
    row = idx // 3
    col = idx % 3
    df.plot1d(df[var], ax=axes[row, col], show=False)
    axes[row, col].set_title(f'Distribution of {var}')

plt.tight_layout()
plt.show()
```

### 分面圖

```python
# 按類別繪圖
categories = df.category.unique()

fig, axes = plt.subplots(1, len(categories), figsize=(15, 5))

for idx, cat in enumerate(categories):
    df_cat = df[df.category == cat]
    df_cat.plot(df_cat.x, df_cat.y,
                ax=axes[idx], show=False)
    axes[idx].set_title(f'Category {cat}')

plt.tight_layout()
plt.show()
```

## 互動式小工具（Jupyter）

在 Jupyter notebook 中建立互動式視覺化：

### 選擇小工具

```python
# 互動式選擇
df.widget.selection_expression()
```

### 直方圖小工具

```python
# 帶選擇的互動式直方圖
df.plot_widget(df.x, df.y)
```

### 散點圖小工具

```python
# 互動式散點圖
df.scatter_widget(df.x, df.y)
```

## 自訂

### 設定圖表樣式

```python
import matplotlib.pyplot as plt

# 建立帶自訂樣式的圖
fig, ax = plt.subplots(figsize=(12, 8))

df.plot(df.x, df.y,
        limits='99%',
        shape=(256, 256),
        colormap='plasma',
        ax=ax,
        show=False)

# 自訂軸
ax.set_xlabel('X Variable', fontsize=14, fontweight='bold')
ax.set_ylabel('Y Variable', fontsize=14, fontweight='bold')
ax.set_title('Custom Density Plot', fontsize=16, fontweight='bold')
ax.grid(alpha=0.3)

# 加入色條
plt.colorbar(ax.collections[0], ax=ax, label='Density')

plt.tight_layout()
plt.show()
```

### 圖形大小和 DPI

```python
# 高解析度圖
df.plot(df.x, df.y,
        figsize=(12, 10),
        dpi=300)
```

## 專門視覺化

### 六角形分箱圖

```python
# 使用六角形分箱的熱圖替代方案
plt.figure(figsize=(10, 8))
plt.hexbin(df.x.values[:100000], df.y.values[:100000],
           gridsize=50, cmap='viridis')
plt.colorbar(label='Count')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

### 等高線圖

```python
import numpy as np

# 取得 2D 直方圖資料
counts = df.count(binby=[df.x, df.y],
                  limits=[[0, 10], [0, 10]],
                  shape=(100, 100))

# 建立等高線圖
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
plt.contourf(x, y, counts.T, levels=20, cmap='viridis')
plt.colorbar(label='Count')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot')
plt.show()
```

### 向量場疊加

```python
# 在網格上計算平均向量
mean_vx = df.mean(df.vx, binby=[df.x, df.y],
                  limits=[[0, 10], [0, 10]],
                  shape=(20, 20))
mean_vy = df.mean(df.vy, binby=[df.x, df.y],
                  limits=[[0, 10], [0, 10]],
                  shape=(20, 20))

# 建立網格
x = np.linspace(0, 10, 20)
y = np.linspace(0, 10, 20)
X, Y = np.meshgrid(x, y)

# 繪圖
fig, ax = plt.subplots(figsize=(10, 8))

# 基底熱圖
df.plot(df.x, df.y, ax=ax, show=False)

# 向量疊加
ax.quiver(X, Y, mean_vx.T, mean_vy.T, alpha=0.7, color='white')

plt.show()
```

## 效能考量

### 最佳化大型視覺化

```python
# 對於非常大的資料集，減少形狀
df.plot(df.x, df.y, shape=(256, 256))  # 快速

# 出版品質
df.plot(df.x, df.y, shape=(1024, 1024))  # 更高品質

# 平衡品質和效能
df.plot(df.x, df.y, shape=(512, 512))  # 良好平衡
```

### 快取視覺化資料

```python
# 計算一次，繪製多次
counts = df.count(binby=[df.x, df.y],
                  limits=[[0, 10], [0, 10]],
                  shape=(512, 512))

# 在不同圖中使用
plt.figure()
plt.imshow(counts.T, origin='lower', cmap='viridis')
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(np.log10(counts.T + 1), origin='lower', cmap='plasma')
plt.colorbar()
plt.show()
```

## 匯出和儲存

### 儲存圖形

```python
# 儲存為 PNG
df.plot(df.x, df.y)
plt.savefig('plot.png', dpi=300, bbox_inches='tight')

# 儲存為 PDF（向量）
plt.savefig('plot.pdf', bbox_inches='tight')

# 儲存為 SVG
plt.savefig('plot.svg', bbox_inches='tight')
```

### 批次繪圖

```python
# 產生多個圖
variables = ['x', 'y', 'z']

for var in variables:
    plt.figure(figsize=(10, 6))
    df.plot1d(df[var])
    plt.title(f'Distribution of {var}')
    plt.savefig(f'plot_{var}.png', dpi=300, bbox_inches='tight')
    plt.close()
```

## 常見模式

### 模式：探索性資料分析

```python
import matplotlib.pyplot as plt

# 建立全面視覺化
fig = plt.figure(figsize=(16, 12))

# 1D 直方圖
ax1 = plt.subplot(3, 3, 1)
df.plot1d(df.x, ax=ax1, show=False)
ax1.set_title('X Distribution')

ax2 = plt.subplot(3, 3, 2)
df.plot1d(df.y, ax=ax2, show=False)
ax2.set_title('Y Distribution')

ax3 = plt.subplot(3, 3, 3)
df.plot1d(df.z, ax=ax3, show=False)
ax3.set_title('Z Distribution')

# 2D 圖
ax4 = plt.subplot(3, 3, 4)
df.plot(df.x, df.y, ax=ax4, show=False)
ax4.set_title('X vs Y')

ax5 = plt.subplot(3, 3, 5)
df.plot(df.x, df.z, ax=ax5, show=False)
ax5.set_title('X vs Z')

ax6 = plt.subplot(3, 3, 6)
df.plot(df.y, df.z, ax=ax6, show=False)
ax6.set_title('Y vs Z')

# 網格上的統計
ax7 = plt.subplot(3, 3, 7)
df.plot(df.x, df.y, what=df.z.mean(), ax=ax7, show=False)
ax7.set_title('Mean Z on X-Y grid')

plt.tight_layout()
plt.savefig('eda_summary.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 模式：跨組比較

```python
# 按類別比較分布
categories = df.category.unique()

fig, axes = plt.subplots(len(categories), 2,
                         figsize=(12, 4 * len(categories)))

for idx, cat in enumerate(categories):
    df.select(df.category == cat, name=f'cat_{cat}')

    # 1D 直方圖
    df.plot1d(df.value, selection=f'cat_{cat}',
              ax=axes[idx, 0], show=False)
    axes[idx, 0].set_title(f'Category {cat} - Distribution')

    # 2D 圖
    df.plot(df.x, df.y, selection=f'cat_{cat}',
            ax=axes[idx, 1], show=False)
    axes[idx, 1].set_title(f'Category {cat} - X vs Y')

plt.tight_layout()
plt.show()
```

### 模式：時間序列視覺化

```python
# 按時間分箱聚合
df['year'] = df.timestamp.dt.year
df['month'] = df.timestamp.dt.month

# 繪製時間序列
monthly_sales = df.groupby(['year', 'month']).agg({'sales': 'sum'})

plt.figure(figsize=(14, 6))
plt.plot(range(len(monthly_sales)), monthly_sales['sales'])
plt.xlabel('Time Period')
plt.ylabel('Sales')
plt.title('Sales Over Time')
plt.grid(alpha=0.3)
plt.show()
```

## 與其他函式庫整合

### Plotly 用於互動性

```python
import plotly.graph_objects as go

# 從 Vaex 取得資料
counts = df.count(binby=[df.x, df.y], shape=(100, 100))

# 建立 plotly 圖形
fig = go.Figure(data=go.Heatmap(z=counts.T))
fig.update_layout(title='Interactive Heatmap')
fig.show()
```

### Seaborn 樣式

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 使用 seaborn 樣式
sns.set_style('darkgrid')
sns.set_palette('husl')

df.plot1d(df.value)
plt.show()
```

## 最佳實務

1. **使用適當的形狀** - 平衡解析度和效能（探索用 256-512，出版用 1024+）
2. **套用合理的限制** - 使用基於百分位數的限制（'99%', '99.7%'）處理離群值
3. **明智選擇色標** - 對範圍廣泛的計數使用對數尺度，對均勻資料使用線性
4. **利用選擇** - 不建立新 DataFrame 就能比較子集
5. **快取聚合** - 如果建立多個類似圖，只計算一次
6. **出版用向量格式** - 儲存為 PDF 或 SVG 以獲得可縮放圖形
7. **避免抽樣** - Vaex 視覺化使用所有資料，不需要抽樣

## 疑難排解

### 問題：空白或稀疏圖

```python
# 問題：限制不匹配資料範圍
df.plot(df.x, df.y, limits=[[0, 10], [0, 10]])

# 解決方案：使用自動限制
df.plot(df.x, df.y, limits='minmax')
df.plot(df.x, df.y, limits='99%')
```

### 問題：圖太慢

```python
# 問題：解析度太高
df.plot(df.x, df.y, shape=(2048, 2048))

# 解決方案：減少形狀
df.plot(df.x, df.y, shape=(512, 512))
```

### 問題：看不到低密度區域

```python
# 問題：線性尺度被高密度區域淹沒
df.plot(df.x, df.y, f='identity')

# 解決方案：使用對數尺度
df.plot(df.x, df.y, f='log')
```

## 相關資源

- 資料聚合：參見 `data_processing.md`
- 效能最佳化：參見 `performance.md`
- DataFrame 基礎：參見 `core_dataframes.md`
