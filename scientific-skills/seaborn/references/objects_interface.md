# Seaborn Objects 介面

`seaborn.objects` 介面提供現代的宣告式 API，透過組合來建構視覺化。本指南涵蓋 seaborn 0.12+ 引入的完整 objects 介面。

## 核心概念

objects 介面將**你想展示什麼**（資料和映射）與**如何展示**（標記、統計和移動）分開。建構圖形的方式：

1. 使用資料和美學映射建立 `Plot` 物件
2. 使用 `.add()` 添加圖層，結合標記和統計轉換
3. 使用 `.scale()`、`.label()`、`.limit()`、`.theme()` 等進行自訂
4. 使用 `.show()` 或 `.save()` 渲染

## 基本用法

```python
from seaborn import objects as so
import pandas as pd

# 使用資料和映射建立圖形
p = so.Plot(data=df, x='x_var', y='y_var')

# 添加標記（視覺表示）
p = p.add(so.Dot())

# 顯示（在 Jupyter 中自動顯示）
p.show()
```

## Plot 類別

`Plot` 類別是 objects 介面的基礎。

### 初始化

```python
so.Plot(data=None, x=None, y=None, color=None, alpha=None,
        fill=None, fillalpha=None, fillcolor=None, marker=None,
        pointsize=None, stroke=None, text=None, **variables)
```

**參數：**
- `data` - DataFrame 或資料向量字典
- `x, y` - 位置變數
- `color` - 顏色編碼變數
- `alpha` - 透明度變數
- `marker` - 標記形狀變數
- `pointsize` - 點大小變數
- `stroke` - 線寬變數
- `text` - 文字標籤變數
- `**variables` - 使用屬性名稱的額外映射

**範例：**
```python
# 基本映射
so.Plot(df, x='total_bill', y='tip')

# 多重映射
so.Plot(df, x='total_bill', y='tip', color='day', pointsize='size')

# 所有變數在 Plot 中
p = so.Plot(df, x='x', y='y', color='cat')
p.add(so.Dot())  # 使用所有映射

# 部分變數在 add() 中
p = so.Plot(df, x='x', y='y')
p.add(so.Dot(), color='cat')  # 只有此圖層使用顏色
```

### 方法

#### add()

添加帶有標記和可選 stat/move 的圖層到圖形。

```python
Plot.add(mark, *transforms, orient=None, legend=True, data=None,
         **variables)
```

**參數：**
- `mark` - 定義視覺表示的 Mark 物件
- `*transforms` - 用於資料轉換的 Stat 和/或 Move 物件
- `orient` - "x"、"y" 或 "v"/"h" 表示方向
- `legend` - 是否包含在圖例中（True/False）
- `data` - 覆寫此圖層的資料
- `**variables` - 覆寫或添加變數映射

**範例：**
```python
# 簡單標記
p.add(so.Dot())

# 帶有統計的標記
p.add(so.Line(), so.PolyFit(order=2))

# 帶有多重轉換的標記
p.add(so.Bar(), so.Agg(), so.Dodge())

# 圖層特定映射
p.add(so.Dot(), color='category')
p.add(so.Line(), so.Agg(), color='category')

# 圖層特定資料
p.add(so.Dot())
p.add(so.Line(), data=summary_df)
```

#### facet()

從類別變數建立子圖。

```python
Plot.facet(col=None, row=None, order=None, wrap=None)
```

**參數：**
- `col` - 欄分面變數
- `row` - 列分面變數
- `order` - 分面順序字典（鍵：變數名稱）
- `wrap` - 此數量後換行欄

**範例：**
```python
p.facet(col='time', row='sex')
p.facet(col='category', wrap=3)
p.facet(col='day', order={'day': ['Thur', 'Fri', 'Sat', 'Sun']})
```

#### pair()

為多個變數建立成對子圖。

```python
Plot.pair(x=None, y=None, wrap=None, cross=True)
```

**參數：**
- `x` - x 軸配對的變數
- `y` - y 軸配對的變數（若為 None，使用 x）
- `wrap` - 此數量欄後換行
- `cross` - 包含所有 x/y 組合（vs. 只有對角線）

**範例：**
```python
# 所有變數的配對
p = so.Plot(df).pair(x=['a', 'b', 'c'])
p.add(so.Dot())

# 矩形網格
p = so.Plot(df).pair(x=['a', 'b'], y=['c', 'd'])
p.add(so.Dot(), alpha=0.5)
```

#### scale()

自訂資料如何映射到視覺屬性。

```python
Plot.scale(**scales)
```

**參數：** 帶有屬性名稱和 Scale 物件的關鍵字引數

**範例：**
```python
p.scale(
    x=so.Continuous().tick(every=5),
    y=so.Continuous().label(like='{x:.1f}'),
    color=so.Nominal(['#1f77b4', '#ff7f0e', '#2ca02c']),
    pointsize=(5, 10)  # 範圍的簡寫
)
```

#### limit()

設定軸限制。

```python
Plot.limit(x=None, y=None)
```

**參數：**
- `x` - x 軸的（最小值, 最大值）元組
- `y` - y 軸的（最小值, 最大值）元組

**範例：**
```python
p.limit(x=(0, 100), y=(0, 50))
```

#### label()

設定軸標籤和標題。

```python
Plot.label(x=None, y=None, color=None, title=None, **labels)
```

**參數：** 帶有屬性名稱和標籤字串的關鍵字引數

**範例：**
```python
p.label(
    x='Total Bill ($)',
    y='Tip Amount ($)',
    color='Day of Week',
    title='Restaurant Tips Analysis'
)
```

#### theme()

應用 matplotlib 樣式設定。

```python
Plot.theme(config, **kwargs)
```

**參數：**
- `config` - rcParams 字典或 seaborn 主題字典
- `**kwargs` - 個別 rcParams

**範例：**
```python
# Seaborn 主題
p.theme({**sns.axes_style('whitegrid'), **sns.plotting_context('talk')})

# 自訂 rcParams
p.theme({'axes.facecolor': 'white', 'axes.grid': True})

# 個別參數
p.theme(axes_facecolor='white', font_scale=1.2)
```

#### layout()

設定子圖版面。

```python
Plot.layout(size=None, extent=None, engine=None)
```

**參數：**
- `size` - （寬度, 高度）英寸
- `extent` - 子圖的（左, 下, 右, 上）
- `engine` - "tight"、"constrained" 或 None

**範例：**
```python
p.layout(size=(10, 6), engine='constrained')
```

#### share()

控制跨分面的軸共享。

```python
Plot.share(x=None, y=None)
```

**參數：**
- `x` - 共享 x 軸：True、False 或 "col"/"row"
- `y` - 共享 y 軸：True、False 或 "col"/"row"

**範例：**
```python
p.share(x=True, y=False)  # 跨所有共享 x，獨立 y
p.share(x='col')  # 只在欄內共享 x
```

#### on()

在現有 matplotlib 圖形或軸上繪圖。

```python
Plot.on(target)
```

**參數：**
- `target` - matplotlib Figure 或 Axes 物件

**範例：**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
so.Plot(df, x='x', y='y').add(so.Dot()).on(axes[0, 0])
so.Plot(df, x='x', y='z').add(so.Line()).on(axes[0, 1])
```

#### show()

渲染並顯示圖形。

```python
Plot.show(**kwargs)
```

**參數：** 傳遞給 `matplotlib.pyplot.show()`

#### save()

儲存圖形到檔案。

```python
Plot.save(filename, **kwargs)
```

**參數：**
- `filename` - 輸出檔名
- `**kwargs` - 傳遞給 `matplotlib.figure.Figure.savefig()`

**範例：**
```python
p.save('plot.png', dpi=300, bbox_inches='tight')
p.save('plot.pdf')
```

## Mark 物件

Marks 定義資料如何視覺化呈現。

### Dot

個別觀測值的點/標記。

```python
so.Dot(artist_kws=None, **kwargs)
```

**屬性：**
- `color` - 填充顏色
- `alpha` - 透明度
- `fillcolor` - 替代顏色屬性
- `fillalpha` - 替代 alpha 屬性
- `edgecolor` - 邊緣顏色
- `edgealpha` - 邊緣透明度
- `edgewidth` - 邊緣線寬
- `marker` - 標記樣式
- `pointsize` - 標記大小
- `stroke` - 邊緣寬度

**範例：**
```python
so.Plot(df, x='x', y='y').add(so.Dot(color='blue', pointsize=10))
so.Plot(df, x='x', y='y', color='cat').add(so.Dot(alpha=0.5))
```

### Line

連接觀測值的線條。

```python
so.Line(artist_kws=None, **kwargs)
```

**屬性：**
- `color` - 線條顏色
- `alpha` - 透明度
- `linewidth` - 線寬
- `linestyle` - 線條樣式（"-"、"--"、"-."、":"）
- `marker` - 資料點上的標記
- `pointsize` - 標記大小
- `edgecolor` - 標記邊緣顏色
- `edgewidth` - 標記邊緣寬度

**範例：**
```python
so.Plot(df, x='x', y='y').add(so.Line())
so.Plot(df, x='x', y='y', color='cat').add(so.Line(linewidth=2))
```

### Path

類似 Line 但按資料順序連接點（不按 x 排序）。

```python
so.Path(artist_kws=None, **kwargs)
```

屬性與 `Line` 相同。

**範例：**
```python
# 用於軌跡、迴圈等
so.Plot(trajectory_df, x='x', y='y').add(so.Path())
```

### Bar

矩形長條。

```python
so.Bar(artist_kws=None, **kwargs)
```

**屬性：**
- `color` - 填充顏色
- `alpha` - 透明度
- `edgecolor` - 邊緣顏色
- `edgealpha` - 邊緣透明度
- `edgewidth` - 邊緣線寬
- `width` - 長條寬度（資料單位）

**範例：**
```python
so.Plot(df, x='category', y='value').add(so.Bar())
so.Plot(df, x='x', y='y').add(so.Bar(color='#1f77b4', width=0.5))
```

### Bars

多個長條（用於帶誤差棒的聚合資料）。

```python
so.Bars(artist_kws=None, **kwargs)
```

屬性與 `Bar` 相同。與 `Agg()` 或 `Est()` stats 一起使用。

**範例：**
```python
so.Plot(df, x='category', y='value').add(so.Bars(), so.Agg())
```

### Area

線條和基線之間的填充區域。

```python
so.Area(artist_kws=None, **kwargs)
```

**屬性：**
- `color` - 填充顏色
- `alpha` - 透明度
- `edgecolor` - 邊緣顏色
- `edgealpha` - 邊緣透明度
- `edgewidth` - 邊緣線寬
- `baseline` - 基線值（預設：0）

**範例：**
```python
so.Plot(df, x='x', y='y').add(so.Area(alpha=0.3))
so.Plot(df, x='x', y='y', color='cat').add(so.Area())
```

### Band

兩條線之間的填充帶（用於範圍/區間）。

```python
so.Band(artist_kws=None, **kwargs)
```

屬性與 `Area` 相同。需要 `ymin` 和 `ymax` 映射或與 `Est()` stat 一起使用。

**範例：**
```python
so.Plot(df, x='x', ymin='lower', ymax='upper').add(so.Band())
so.Plot(df, x='x', y='y').add(so.Band(), so.Est())
```

### Range

端點帶標記的線條（用於範圍）。

```python
so.Range(artist_kws=None, **kwargs)
```

**屬性：**
- `color` - 線條和標記顏色
- `alpha` - 透明度
- `linewidth` - 線寬
- `marker` - 端點的標記樣式
- `pointsize` - 標記大小
- `edgewidth` - 標記邊緣寬度

**範例：**
```python
so.Plot(df, x='x', y='y').add(so.Range(), so.Est())
```

### Dash

短水平/垂直線（用於分布標記）。

```python
so.Dash(artist_kws=None, **kwargs)
```

**屬性：**
- `color` - 線條顏色
- `alpha` - 透明度
- `linewidth` - 線寬
- `width` - 短線長度（資料單位）

**範例：**
```python
so.Plot(df, x='category', y='value').add(so.Dash())
```

### Text

資料點上的文字標籤。

```python
so.Text(artist_kws=None, **kwargs)
```

**屬性：**
- `color` - 文字顏色
- `alpha` - 透明度
- `fontsize` - 字體大小
- `halign` - 水平對齊："left"、"center"、"right"
- `valign` - 垂直對齊："bottom"、"center"、"top"
- `offset` - 距點的（x, y）偏移

需要 `text` 映射。

**範例：**
```python
so.Plot(df, x='x', y='y', text='label').add(so.Text())
so.Plot(df, x='x', y='y', text='value').add(so.Text(fontsize=10, offset=(0, 5)))
```

## Stat 物件

Stats 在渲染前轉換資料。在 `.add()` 中與 marks 組合。

### Agg

按組聚合觀測值。

```python
so.Agg(func='mean')
```

**參數：**
- `func` - 聚合函數："mean"、"median"、"sum"、"min"、"max"、"count" 或可呼叫物件

**範例：**
```python
so.Plot(df, x='category', y='value').add(so.Bar(), so.Agg('mean'))
so.Plot(df, x='x', y='y', color='group').add(so.Line(), so.Agg('median'))
```

### Est

帶有誤差區間的中心趨勢估計。

```python
so.Est(func='mean', errorbar=('ci', 95), n_boot=1000, seed=None)
```

**參數：**
- `func` - 估計器："mean"、"median"、"sum" 或可呼叫物件
- `errorbar` - 誤差表示：
  - `("ci", level)` - 透過自助法的信賴區間
  - `("pi", level)` - 百分位數區間
  - `("se", scale)` - 按係數縮放的標準誤
  - `"sd"` - 標準差
- `n_boot` - 自助法迭代次數
- `seed` - 隨機種子

**範例：**
```python
so.Plot(df, x='category', y='value').add(so.Bar(), so.Est())
so.Plot(df, x='x', y='y').add(so.Line(), so.Est(errorbar='sd'))
so.Plot(df, x='x', y='y').add(so.Line(), so.Est(errorbar=('ci', 95)))
so.Plot(df, x='x', y='y').add(so.Band(), so.Est())
```

### Hist

分箱觀測值並計數/聚合。

```python
so.Hist(stat='count', bins='auto', binwidth=None, binrange=None,
        common_norm=True, common_bins=True, cumulative=False)
```

**參數：**
- `stat` - "count"、"density"、"probability"、"percent"、"frequency"
- `bins` - 分箱數量、分箱方法或邊緣
- `binwidth` - 分箱寬度
- `binrange` - 分箱的（最小值, 最大值）範圍
- `common_norm` - 跨組一起正規化
- `common_bins` - 所有組使用相同分箱
- `cumulative` - 累積直方圖

**範例：**
```python
so.Plot(df, x='value').add(so.Bars(), so.Hist())
so.Plot(df, x='value').add(so.Bars(), so.Hist(bins=20, stat='density'))
so.Plot(df, x='value', color='group').add(so.Area(), so.Hist(cumulative=True))
```

### KDE

核密度估計。

```python
so.KDE(bw_method='scott', bw_adjust=1, gridsize=200,
       cut=3, cumulative=False)
```

**參數：**
- `bw_method` - 頻寬方法："scott"、"silverman" 或純量
- `bw_adjust` - 頻寬乘數
- `gridsize` - 密度曲線的解析度
- `cut` - 超出資料範圍的延伸（以頻寬單位）
- `cumulative` - 累積密度

**範例：**
```python
so.Plot(df, x='value').add(so.Line(), so.KDE())
so.Plot(df, x='value', color='group').add(so.Area(alpha=0.5), so.KDE())
so.Plot(df, x='x', y='y').add(so.Line(), so.KDE(bw_adjust=0.5))
```

### Count

計算每組的觀測值數量。

```python
so.Count()
```

**範例：**
```python
so.Plot(df, x='category').add(so.Bar(), so.Count())
```

### PolyFit

多項式迴歸擬合。

```python
so.PolyFit(order=1)
```

**參數：**
- `order` - 多項式階數（1 = 線性，2 = 二次，等等）

**範例：**
```python
so.Plot(df, x='x', y='y').add(so.Dot())
so.Plot(df, x='x', y='y').add(so.Line(), so.PolyFit(order=2))
```

### Perc

計算百分位數。

```python
so.Perc(k=5, method='linear')
```

**參數：**
- `k` - 百分位數區間數量
- `method` - 插值方法

**範例：**
```python
so.Plot(df, x='x', y='y').add(so.Band(), so.Perc())
```

## Move 物件

Moves 調整位置以解決重疊或建立特定版面。

### Dodge

將位置並排移動。

```python
so.Dodge(empty='keep', gap=0)
```

**參數：**
- `empty` - 如何處理空組："keep"、"drop"、"fill"
- `gap` - 閃避元素之間的間隙（比例）

**範例：**
```python
so.Plot(df, x='category', y='value', color='group').add(so.Bar(), so.Dodge())
so.Plot(df, x='cat', y='val', color='hue').add(so.Dot(), so.Dodge(gap=0.1))
```

### Stack

垂直堆疊標記。

```python
so.Stack()
```

**範例：**
```python
so.Plot(df, x='x', y='y', color='category').add(so.Bar(), so.Stack())
so.Plot(df, x='x', y='y', color='group').add(so.Area(), so.Stack())
```

### Jitter

對位置添加隨機雜訊。

```python
so.Jitter(width=None, height=None, seed=None)
```

**參數：**
- `width` - x 方向的抖動（資料單位或比例）
- `height` - y 方向的抖動
- `seed` - 隨機種子

**範例：**
```python
so.Plot(df, x='category', y='value').add(so.Dot(), so.Jitter())
so.Plot(df, x='cat', y='val').add(so.Dot(), so.Jitter(width=0.2))
```

### Shift

將位置移動固定量。

```python
so.Shift(x=0, y=0)
```

**參數：**
- `x` - x 方向的移動（資料單位）
- `y` - y 方向的移動

**範例：**
```python
so.Plot(df, x='x', y='y').add(so.Dot(), so.Shift(x=1))
```

### Norm

正規化值。

```python
so.Norm(func='max', where=None, by=None, percent=False)
```

**參數：**
- `func` - 正規化方式："max"、"sum"、"area" 或可呼叫物件
- `where` - 應用到哪個軸："x"、"y" 或 None
- `by` - 分別正規化的分組變數
- `percent` - 顯示為百分比

**範例：**
```python
so.Plot(df, x='x', y='y', color='group').add(so.Area(), so.Norm())
```

## Scale 物件

Scales 控制資料值如何映射到視覺屬性。

### Continuous

用於數值資料。

```python
so.Continuous(values=None, norm=None, trans=None)
```

**方法：**
- `.tick(at=None, every=None, between=None, minor=None)` - 設定刻度
- `.label(like=None, base=None, unit=None)` - 格式化標籤

**參數：**
- `values` - 明確的值範圍（最小值, 最大值）
- `norm` - 正規化函數
- `trans` - 轉換："log"、"sqrt"、"symlog"、"logit"、"pow10" 或可呼叫物件

**範例：**
```python
p.scale(
    x=so.Continuous().tick(every=10),
    y=so.Continuous(trans='log').tick(at=[1, 10, 100]),
    color=so.Continuous(values=(0, 1)),
    pointsize=(5, 20)  # Continuous 範圍的簡寫
)
```

### Nominal

用於類別資料。

```python
so.Nominal(values=None, order=None)
```

**參數：**
- `values` - 明確的值（例如顏色、標記）
- `order` - 類別順序

**範例：**
```python
p.scale(
    color=so.Nominal(['#1f77b4', '#ff7f0e', '#2ca02c']),
    marker=so.Nominal(['o', 's', '^']),
    x=so.Nominal(order=['Low', 'Medium', 'High'])
)
```

### Temporal

用於日期時間資料。

```python
so.Temporal(values=None, trans=None)
```

**方法：**
- `.tick(every=None, between=None)` - 設定刻度
- `.label(concise=False)` - 格式化標籤

**範例：**
```python
p.scale(x=so.Temporal().tick(every=('month', 1)).label(concise=True))
```

## 完整範例

### 帶有統計的分層圖形

```python
(
    so.Plot(df, x='total_bill', y='tip', color='time')
    .add(so.Dot(), alpha=0.5)
    .add(so.Line(), so.PolyFit(order=2))
    .scale(color=so.Nominal(['#1f77b4', '#ff7f0e']))
    .label(x='Total Bill ($)', y='Tip ($)', title='Tips Analysis')
    .theme({**sns.axes_style('whitegrid')})
)
```

### 分面分布

```python
(
    so.Plot(df, x='measurement', color='treatment')
    .facet(col='timepoint', wrap=3)
    .add(so.Area(alpha=0.5), so.KDE())
    .add(so.Dot(), so.Jitter(width=0.1), y=0)
    .scale(x=so.Continuous().tick(every=5))
    .label(x='Measurement (units)', title='Treatment Effects Over Time')
    .share(x=True, y=False)
)
```

### 分組長條圖

```python
(
    so.Plot(df, x='category', y='value', color='group')
    .add(so.Bar(), so.Agg('mean'), so.Dodge())
    .add(so.Range(), so.Est(errorbar='se'), so.Dodge())
    .scale(color=so.Nominal(order=['A', 'B', 'C']))
    .label(y='Mean Value', title='Comparison by Category and Group')
)
```

### 複雜多圖層

```python
(
    so.Plot(df, x='date', y='value')
    .add(so.Dot(color='gray', pointsize=3), alpha=0.3)
    .add(so.Line(color='blue', linewidth=2), so.Agg('mean'))
    .add(so.Band(color='blue', alpha=0.2), so.Est(errorbar=('ci', 95)))
    .facet(col='sensor', row='location')
    .scale(
        x=so.Temporal().label(concise=True),
        y=so.Continuous().tick(every=10)
    )
    .label(
        x='Date',
        y='Measurement',
        title='Sensor Measurements by Location'
    )
    .layout(size=(12, 8), engine='constrained')
)
```

## 從函數介面遷移

### 散點圖

**函數介面：**
```python
sns.scatterplot(data=df, x='x', y='y', hue='category', size='value')
```

**Objects 介面：**
```python
so.Plot(df, x='x', y='y', color='category', pointsize='value').add(so.Dot())
```

### 帶有信賴區間的折線圖

**函數介面：**
```python
sns.lineplot(data=df, x='time', y='measurement', hue='group', errorbar='ci')
```

**Objects 介面：**
```python
(
    so.Plot(df, x='time', y='measurement', color='group')
    .add(so.Line(), so.Est())
)
```

### 直方圖

**函數介面：**
```python
sns.histplot(data=df, x='value', hue='category', stat='density', kde=True)
```

**Objects 介面：**
```python
(
    so.Plot(df, x='value', color='category')
    .add(so.Bars(), so.Hist(stat='density'))
    .add(so.Line(), so.KDE())
)
```

### 帶有誤差棒的長條圖

**函數介面：**
```python
sns.barplot(data=df, x='category', y='value', hue='group', errorbar='ci')
```

**Objects 介面：**
```python
(
    so.Plot(df, x='category', y='value', color='group')
    .add(so.Bar(), so.Agg(), so.Dodge())
    .add(so.Range(), so.Est(), so.Dodge())
)
```

## 技巧和最佳實踐

1. **方法鏈接**：每個方法回傳新的 Plot 物件，支援流暢的鏈接
2. **圖層組合**：組合多個 `.add()` 呼叫來疊加不同的標記
3. **轉換順序**：在 `.add(mark, stat, move)` 中，stat 先應用，然後是 move
4. **變數優先順序**：圖層特定映射覆寫 Plot 層級映射
5. **Scale 簡寫**：使用元組表示簡單範圍：`color=(min, max)` vs 完整 Scale 物件
6. **Jupyter 渲染**：回傳時圖形自動渲染；否則使用 `.show()`
7. **儲存**：使用 `.save()` 而非 `plt.savefig()` 以正確處理
8. **Matplotlib 存取**：使用 `.on(ax)` 與 matplotlib 圖形整合
