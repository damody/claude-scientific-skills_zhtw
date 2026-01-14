# Seaborn 函數參考

本文件提供所有主要 seaborn 函數的完整參考，按類別組織。

## 關聯圖

### scatterplot()

**用途：** 建立散點圖，以點表示個別觀測值。

**關鍵參數：**
- `data` - DataFrame、陣列或陣列字典
- `x, y` - x 和 y 軸的變數
- `hue` - 用於顏色編碼的分組變數
- `size` - 用於大小編碼的分組變數
- `style` - 用於標記樣式的分組變數
- `palette` - 調色盤名稱或列表
- `hue_order` - 類別 hue 層級的順序
- `hue_norm` - 數值 hue 的正規化（元組或 Normalize 物件）
- `sizes` - 大小編碼的範圍（元組或字典）
- `size_order` - 類別 size 層級的順序
- `size_norm` - 數值 size 的正規化
- `markers` - 標記樣式（字串、列表或字典）
- `style_order` - 類別 style 層級的順序
- `legend` - 如何繪製圖例："auto"、"brief"、"full" 或 False
- `ax` - 要繪製的 Matplotlib axes

**範例：**
```python
sns.scatterplot(data=df, x='height', y='weight',
                hue='gender', size='age', style='smoker',
                palette='Set2', sizes=(20, 200))
```

### lineplot()

**用途：** 繪製折線圖，對重複測量自動聚合並計算信賴區間。

**關鍵參數：**
- `data` - DataFrame、陣列或陣列字典
- `x, y` - x 和 y 軸的變數
- `hue` - 用於顏色編碼的分組變數
- `size` - 用於線寬的分組變數
- `style` - 用於線條樣式（虛線）的分組變數
- `units` - 取樣單位的分組變數（單位內不聚合）
- `estimator` - 跨觀測值聚合的函數（預設：mean）
- `errorbar` - 誤差棒方法："sd"、"se"、"pi"、("ci", level)、("pi", level) 或 None
- `n_boot` - 計算信賴區間的自助法迭代次數
- `seed` - 可重複自助法的隨機種子
- `sort` - 繪圖前排序資料
- `err_style` - "band" 或 "bars" 表示誤差
- `err_kws` - 誤差表示的額外參數
- `markers` - 強調資料點的標記樣式
- `dashes` - 線條的虛線樣式
- `legend` - 如何繪製圖例
- `ax` - 要繪製的 Matplotlib axes

**範例：**
```python
sns.lineplot(data=timeseries, x='time', y='signal',
             hue='condition', style='subject',
             errorbar=('ci', 95), markers=True)
```

### relplot()

**用途：** 在 FacetGrid 上繪製關聯圖（散點或折線）的圖形層級介面。

**關鍵參數：**
`scatterplot()` 和 `lineplot()` 的所有參數，加上：
- `kind` - "scatter" 或 "line"
- `col` - 欄分面的類別變數
- `row` - 列分面的類別變數
- `col_wrap` - 此數量欄位後換行
- `col_order` - 欄分面層級的順序
- `row_order` - 列分面層級的順序
- `height` - 每個分面的高度（英寸）
- `aspect` - 長寬比（寬度 = 高度 * aspect）
- `facet_kws` - FacetGrid 的額外參數

**範例：**
```python
sns.relplot(data=df, x='time', y='measurement',
            hue='treatment', style='batch',
            col='cell_line', row='timepoint',
            kind='line', height=3, aspect=1.5)
```

## 分布圖

### histplot()

**用途：** 繪製單變量或雙變量直方圖，支援靈活的分箱。

**關鍵參數：**
- `data` - DataFrame、陣列或字典
- `x, y` - 變數（雙變量時 y 為可選）
- `hue` - 分組變數
- `weights` - 用於加權觀測值的變數
- `stat` - 聚合統計量："count"、"frequency"、"probability"、"percent"、"density"
- `bins` - 分箱數量、分箱邊緣或方法（"auto"、"fd"、"doane"、"scott"、"stone"、"rice"、"sturges"、"sqrt"）
- `binwidth` - 分箱寬度（覆蓋 bins）
- `binrange` - 分箱範圍（元組）
- `discrete` - 將 x 視為離散（長條置中於值）
- `cumulative` - 計算累積分布
- `common_bins` - 所有 hue 層級使用相同分箱
- `common_norm` - 跨 hue 層級正規化
- `multiple` - 如何處理 hue："layer"、"dodge"、"stack"、"fill"
- `element` - 視覺元素："bars"、"step"、"poly"
- `fill` - 填充長條/元素
- `shrink` - 縮放長條寬度（用於 multiple="dodge"）
- `kde` - 疊加 KDE 估計
- `kde_kws` - KDE 的參數
- `line_kws` - step/poly 元素的參數
- `thresh` - 分箱的最小計數閾值
- `pthresh` - 最小機率閾值
- `pmax` - 色彩縮放的最大機率
- `log_scale` - 軸的對數刻度（布林值或底數）
- `legend` - 是否顯示圖例
- `ax` - Matplotlib axes

**範例：**
```python
sns.histplot(data=df, x='measurement', hue='condition',
             stat='density', bins=30, kde=True,
             multiple='layer', alpha=0.5)
```

### kdeplot()

**用途：** 繪製單變量或雙變量核密度估計。

**關鍵參數：**
- `data` - DataFrame、陣列或字典
- `x, y` - 變數（雙變量時 y 為可選）
- `hue` - 分組變數
- `weights` - 用於加權觀測值的變數
- `palette` - 調色盤
- `hue_order` - hue 層級的順序
- `hue_norm` - 數值 hue 的正規化
- `multiple` - 如何處理 hue："layer"、"stack"、"fill"
- `common_norm` - 跨 hue 層級正規化
- `common_grid` - 所有 hue 層級使用相同網格
- `cumulative` - 計算累積分布
- `bw_method` - 頻寬方法："scott"、"silverman" 或純量
- `bw_adjust` - 頻寬乘數（越高越平滑）
- `log_scale` - 軸的對數刻度
- `levels` - 等高線層級的數量或值（雙變量）
- `thresh` - 等高線的最小密度閾值
- `gridsize` - 網格解析度
- `cut` - 超出資料極值的延伸（以頻寬單位）
- `clip` - 曲線的資料範圍（元組）
- `fill` - 填充曲線/等高線下方區域
- `legend` - 是否顯示圖例
- `ax` - Matplotlib axes

**範例：**
```python
# 單變量
sns.kdeplot(data=df, x='measurement', hue='condition',
            fill=True, common_norm=False, bw_adjust=1.5)

# 雙變量
sns.kdeplot(data=df, x='var1', y='var2',
            fill=True, levels=10, thresh=0.05)
```

### ecdfplot()

**用途：** 繪製經驗累積分布函數。

**關鍵參數：**
- `data` - DataFrame、陣列或字典
- `x, y` - 變數（指定其一）
- `hue` - 分組變數
- `weights` - 用於加權觀測值的變數
- `stat` - "proportion" 或 "count"
- `complementary` - 繪製互補 CDF（1 - ECDF）
- `palette` - 調色盤
- `hue_order` - hue 層級的順序
- `hue_norm` - 數值 hue 的正規化
- `log_scale` - 軸的對數刻度
- `legend` - 是否顯示圖例
- `ax` - Matplotlib axes

**範例：**
```python
sns.ecdfplot(data=df, x='response_time', hue='treatment',
             stat='proportion', complementary=False)
```

### rugplot()

**用途：** 沿軸繪製顯示個別觀測值的刻度標記。

**關鍵參數：**
- `data` - DataFrame、陣列或字典
- `x, y` - 變數（指定其一）
- `hue` - 分組變數
- `height` - 刻度高度（軸的比例）
- `expand_margins` - 為地毯圖添加邊距空間
- `palette` - 調色盤
- `hue_order` - hue 層級的順序
- `hue_norm` - 數值 hue 的正規化
- `legend` - 是否顯示圖例
- `ax` - Matplotlib axes

**範例：**
```python
sns.rugplot(data=df, x='value', hue='category', height=0.05)
```

### displot()

**用途：** 在 FacetGrid 上繪製分布圖的圖形層級介面。

**關鍵參數：**
`histplot()`、`kdeplot()` 和 `ecdfplot()` 的所有參數，加上：
- `kind` - "hist"、"kde"、"ecdf"
- `rug` - 在邊際軸上添加地毯圖
- `rug_kws` - 地毯圖的參數
- `col` - 欄分面的類別變數
- `row` - 列分面的類別變數
- `col_wrap` - 換行欄數
- `col_order` - 欄分面的順序
- `row_order` - 列分面的順序
- `height` - 每個分面的高度
- `aspect` - 長寬比
- `facet_kws` - FacetGrid 的額外參數

**範例：**
```python
sns.displot(data=df, x='measurement', hue='treatment',
            col='timepoint', kind='kde', fill=True,
            height=3, aspect=1.5, rug=True)
```

### jointplot()

**用途：** 繪製帶有邊際單變量圖的雙變量圖。

**關鍵參數：**
- `data` - DataFrame
- `x, y` - x 和 y 軸的變數
- `hue` - 分組變數
- `kind` - "scatter"、"kde"、"hist"、"hex"、"reg"、"resid"
- `height` - 圖形大小（正方形）
- `ratio` - 聯合與邊際軸的比例
- `space` - 聯合與邊際軸之間的空間
- `dropna` - 刪除缺失值
- `xlim, ylim` - 軸限制（元組）
- `marginal_ticks` - 在邊際軸上顯示刻度
- `joint_kws` - 聯合圖的參數
- `marginal_kws` - 邊際圖的參數
- `hue_order` - hue 層級的順序
- `palette` - 調色盤

**範例：**
```python
sns.jointplot(data=df, x='var1', y='var2', hue='group',
              kind='scatter', height=6, ratio=4,
              joint_kws={'alpha': 0.5})
```

### pairplot()

**用途：** 繪製資料集中的成對關係。

**關鍵參數：**
- `data` - DataFrame
- `hue` - 用於顏色編碼的分組變數
- `hue_order` - hue 層級的順序
- `palette` - 調色盤
- `vars` - 要繪製的變數（預設：所有數值）
- `x_vars, y_vars` - x 和 y 軸的變數（非正方形網格）
- `kind` - "scatter"、"kde"、"hist"、"reg"
- `diag_kind` - "auto"、"hist"、"kde"、None
- `markers` - 標記樣式
- `height` - 每個分面的高度
- `aspect` - 長寬比
- `corner` - 只繪製下三角
- `dropna` - 刪除缺失值
- `plot_kws` - 非對角圖的參數
- `diag_kws` - 對角圖的參數
- `grid_kws` - PairGrid 的參數

**範例：**
```python
sns.pairplot(data=df, hue='species', palette='Set2',
             vars=['sepal_length', 'sepal_width', 'petal_length'],
             corner=True, height=2.5)
```

## 類別圖

### stripplot()

**用途：** 繪製帶有抖動點的類別散點圖。

**關鍵參數：**
- `data` - DataFrame、陣列或字典
- `x, y` - 變數（一個類別，一個連續）
- `hue` - 分組變數
- `order` - 類別層級的順序
- `hue_order` - hue 層級的順序
- `jitter` - 抖動量：True、浮點數或 False
- `dodge` - 將 hue 層級並排分開
- `orient` - "v" 或 "h"（通常自動推斷）
- `color` - 所有元素的單一顏色
- `palette` - 調色盤
- `size` - 標記大小
- `edgecolor` - 標記邊緣顏色
- `linewidth` - 標記邊緣寬度
- `native_scale` - 使用類別軸的數值刻度
- `formatter` - 類別軸的格式化器
- `legend` - 是否顯示圖例
- `ax` - Matplotlib axes

**範例：**
```python
sns.stripplot(data=df, x='day', y='total_bill',
              hue='sex', dodge=True, jitter=0.2)
```

### swarmplot()

**用途：** 繪製帶有不重疊點的類別散點圖。

**關鍵參數：**
與 `stripplot()` 相同，除了：
- 無 `jitter` 參數
- `size` - 標記大小（對避免重疊很重要）
- `warn_thresh` - 過多點的警告閾值（預設：0.05）

**注意：** 對大型資料集計算密集。超過 1000 點時使用 stripplot。

**範例：**
```python
sns.swarmplot(data=df, x='day', y='total_bill',
              hue='time', dodge=True, size=5)
```

### boxplot()

**用途：** 繪製顯示四分位數和離群值的箱形圖。

**關鍵參數：**
- `data` - DataFrame、陣列或字典
- `x, y` - 變數（一個類別，一個連續）
- `hue` - 分組變數
- `order` - 類別層級的順序
- `hue_order` - hue 層級的順序
- `orient` - "v" 或 "h"
- `color` - 箱形的單一顏色
- `palette` - 調色盤
- `saturation` - 顏色飽和度
- `width` - 箱形寬度
- `dodge` - 將 hue 層級並排分開
- `fliersize` - 離群值標記大小
- `linewidth` - 箱形線寬
- `whis` - 鬚線的 IQR 乘數（預設：1.5）
- `notch` - 繪製缺口箱形
- `showcaps` - 顯示鬚線端點
- `showmeans` - 顯示平均值
- `meanprops` - 平均值標記的屬性
- `boxprops` - 箱形的屬性
- `whiskerprops` - 鬚線的屬性
- `capprops` - 端點的屬性
- `flierprops` - 離群值的屬性
- `medianprops` - 中位數線的屬性
- `native_scale` - 使用數值刻度
- `formatter` - 類別軸的格式化器
- `legend` - 是否顯示圖例
- `ax` - Matplotlib axes

**範例：**
```python
sns.boxplot(data=df, x='day', y='total_bill',
            hue='smoker', palette='Set3',
            showmeans=True, notch=True)
```

### violinplot()

**用途：** 繪製結合箱形圖和 KDE 的小提琴圖。

**關鍵參數：**
與 `boxplot()` 相同，加上：
- `bw_method` - KDE 頻寬方法
- `bw_adjust` - KDE 頻寬乘數
- `cut` - KDE 超出極值的延伸
- `density_norm` - "area"、"count"、"width"
- `inner` - "box"、"quartile"、"point"、"stick"、None
- `split` - 分離小提琴以進行 hue 比較
- `scale` - 縮放方法："area"、"count"、"width"
- `scale_hue` - 跨 hue 層級縮放
- `gridsize` - KDE 網格解析度

**範例：**
```python
sns.violinplot(data=df, x='day', y='total_bill',
               hue='sex', split=True, inner='quartile',
               palette='muted')
```

### boxenplot()

**用途：** 為較大資料集繪製增強箱形圖，顯示更多分位數。

**關鍵參數：**
與 `boxplot()` 相同，加上：
- `k_depth` - "tukey"、"proportion"、"trustworthy"、"full" 或整數
- `outlier_prop` - 作為離群值的資料比例
- `trust_alpha` - trustworthy 深度的 alpha
- `showfliers` - 顯示離群點

**範例：**
```python
sns.boxenplot(data=df, x='day', y='total_bill',
              hue='time', palette='Set2')
```

### barplot()

**用途：** 繪製帶有誤差棒的長條圖，顯示統計估計。

**關鍵參數：**
- `data` - DataFrame、陣列或字典
- `x, y` - 變數（一個類別，一個連續）
- `hue` - 分組變數
- `order` - 類別層級的順序
- `hue_order` - hue 層級的順序
- `estimator` - 聚合函數（預設：mean）
- `errorbar` - 誤差表示："sd"、"se"、"pi"、("ci", level)、("pi", level) 或 None
- `n_boot` - 自助法迭代次數
- `seed` - 隨機種子
- `units` - 取樣單位的識別符
- `weights` - 觀測值權重
- `orient` - "v" 或 "h"
- `color` - 單一長條顏色
- `palette` - 調色盤
- `saturation` - 顏色飽和度
- `width` - 長條寬度
- `dodge` - 將 hue 層級並排分開
- `errcolor` - 誤差棒顏色
- `errwidth` - 誤差棒線寬
- `capsize` - 誤差棒端點寬度
- `native_scale` - 使用數值刻度
- `formatter` - 類別軸的格式化器
- `legend` - 是否顯示圖例
- `ax` - Matplotlib axes

**範例：**
```python
sns.barplot(data=df, x='day', y='total_bill',
            hue='sex', estimator='median',
            errorbar=('ci', 95), capsize=0.1)
```

### countplot()

**用途：** 顯示每個類別箱的觀測值計數。

**關鍵參數：**
與 `barplot()` 相同，但：
- 只指定 x 或 y 其一（類別變數）
- 無 estimator 或 errorbar（顯示計數）
- `stat` - "count" 或 "percent"

**範例：**
```python
sns.countplot(data=df, x='day', hue='time',
              palette='pastel', dodge=True)
```

### pointplot()

**用途：** 顯示帶有連接線的點估計和信賴區間。

**關鍵參數：**
與 `barplot()` 相同，加上：
- `markers` - 標記樣式
- `linestyles` - 線條樣式
- `scale` - 標記的比例
- `join` - 用線連接點
- `capsize` - 誤差棒端點寬度

**範例：**
```python
sns.pointplot(data=df, x='time', y='total_bill',
              hue='sex', markers=['o', 's'],
              linestyles=['-', '--'], capsize=0.1)
```

### catplot()

**用途：** 在 FacetGrid 上繪製類別圖的圖形層級介面。

**關鍵參數：**
類別圖的所有參數，加上：
- `kind` - "strip"、"swarm"、"box"、"violin"、"boxen"、"bar"、"point"、"count"
- `col` - 欄分面的類別變數
- `row` - 列分面的類別變數
- `col_wrap` - 換行欄數
- `col_order` - 欄分面的順序
- `row_order` - 列分面的順序
- `height` - 每個分面的高度
- `aspect` - 長寬比
- `sharex, sharey` - 跨分面共享軸
- `legend` - 是否顯示圖例
- `legend_out` - 將圖例放在圖形外部
- `facet_kws` - 額外的 FacetGrid 參數

**範例：**
```python
sns.catplot(data=df, x='day', y='total_bill',
            hue='smoker', col='time',
            kind='violin', split=True,
            height=4, aspect=0.8)
```

## 迴歸圖

### regplot()

**用途：** 繪製資料和線性迴歸擬合。

**關鍵參數：**
- `data` - DataFrame
- `x, y` - 變數或資料向量
- `x_estimator` - 對 x 分箱應用估計器
- `x_bins` - 為估計器對 x 分箱
- `x_ci` - 分箱估計的信賴區間
- `scatter` - 顯示散點
- `fit_reg` - 繪製迴歸線
- `ci` - 迴歸估計的信賴區間（整數或 None）
- `n_boot` - 計算信賴區間的自助法迭代次數
- `units` - 取樣單位的識別符
- `seed` - 隨機種子
- `order` - 多項式迴歸階數
- `logistic` - 擬合邏輯迴歸
- `lowess` - 擬合 lowess 平滑器
- `robust` - 擬合穩健迴歸
- `logx` - 對數轉換 x
- `x_partial, y_partial` - 偏迴歸（迴歸掉變數）
- `truncate` - 將迴歸線限制在資料範圍
- `dropna` - 刪除缺失值
- `x_jitter, y_jitter` - 對資料添加抖動
- `label` - 圖例標籤
- `color` - 所有元素的顏色
- `marker` - 標記樣式
- `scatter_kws` - 散點的參數
- `line_kws` - 迴歸線的參數
- `ax` - Matplotlib axes

**範例：**
```python
sns.regplot(data=df, x='total_bill', y='tip',
            order=2, robust=True, ci=95,
            scatter_kws={'alpha': 0.5})
```

### lmplot()

**用途：** 在 FacetGrid 上繪製迴歸圖的圖形層級介面。

**關鍵參數：**
`regplot()` 的所有參數，加上：
- `hue` - 分組變數
- `col` - 欄分面
- `row` - 列分面
- `palette` - 調色盤
- `col_wrap` - 換行欄數
- `height` - 分面高度
- `aspect` - 長寬比
- `markers` - 標記樣式
- `sharex, sharey` - 共享軸
- `hue_order` - hue 層級的順序
- `col_order` - 欄分面的順序
- `row_order` - 列分面的順序
- `legend` - 是否顯示圖例
- `legend_out` - 將圖例放在外部
- `facet_kws` - FacetGrid 參數

**範例：**
```python
sns.lmplot(data=df, x='total_bill', y='tip',
           hue='smoker', col='time', row='sex',
           height=3, aspect=1.2, ci=None)
```

### residplot()

**用途：** 繪製迴歸的殘差。

**關鍵參數：**
與 `regplot()` 相同，但：
- 始終繪製殘差（y - 預測值）vs x
- 在 y=0 處添加水平線
- `lowess` - 對殘差擬合 lowess 平滑器

**範例：**
```python
sns.residplot(data=df, x='x', y='y', lowess=True,
              scatter_kws={'alpha': 0.5})
```

## 矩陣圖

### heatmap()

**用途：** 將矩形資料繪製為顏色編碼矩陣。

**關鍵參數：**
- `data` - 2D 類陣列資料
- `vmin, vmax` - 色彩映射的錨定值
- `cmap` - 色彩映射名稱或物件
- `center` - 色彩映射中心的值
- `robust` - 使用穩健分位數計算色彩映射範圍
- `annot` - 註解儲存格：True、False 或陣列
- `fmt` - 註解的格式字串（例如 ".2f"）
- `annot_kws` - 註解的參數
- `linewidths` - 儲存格邊框寬度
- `linecolor` - 儲存格邊框顏色
- `cbar` - 繪製色彩條
- `cbar_kws` - 色彩條參數
- `cbar_ax` - 色彩條的 axes
- `square` - 強制正方形儲存格
- `xticklabels, yticklabels` - 刻度標籤（True、False、整數或列表）
- `mask` - 遮罩儲存格的布林陣列
- `ax` - Matplotlib axes

**範例：**
```python
# 相關矩陣
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, square=True,
            linewidths=1, cbar_kws={'shrink': 0.8})
```

### clustermap()

**用途：** 繪製階層式聚類熱圖。

**關鍵參數：**
`heatmap()` 的所有參數，加上：
- `pivot_kws` - 樞紐分析的參數（如需要）
- `method` - 連結方法："single"、"complete"、"average"、"weighted"、"centroid"、"median"、"ward"
- `metric` - 聚類的距離度量
- `standard_scale` - 標準化資料：0（列）、1（欄）或 None
- `z_score` - Z 分數正規化資料：0（列）、1（欄）或 None
- `row_cluster, col_cluster` - 聚類列/欄
- `row_linkage, col_linkage` - 預先計算的連結矩陣
- `row_colors, col_colors` - 額外的顏色註解
- `dendrogram_ratio` - 樹狀圖與熱圖的比例
- `colors_ratio` - 顏色註解與熱圖的比例
- `cbar_pos` - 色彩條位置（元組：x, y, 寬度, 高度）
- `tree_kws` - 樹狀圖的參數
- `figsize` - 圖形大小

**範例：**
```python
sns.clustermap(data, method='average', metric='euclidean',
               z_score=0, cmap='viridis',
               row_colors=row_colors, col_colors=col_colors,
               figsize=(12, 12), dendrogram_ratio=0.1)
```

## 多圖網格

### FacetGrid

**用途：** 用於繪製條件關係的多圖網格。

**初始化：**
```python
g = sns.FacetGrid(data, row=None, col=None, hue=None,
                  col_wrap=None, sharex=True, sharey=True,
                  height=3, aspect=1, palette=None,
                  row_order=None, col_order=None, hue_order=None,
                  hue_kws=None, dropna=False, legend_out=True,
                  despine=True, margin_titles=False,
                  xlim=None, ylim=None, subplot_kws=None,
                  gridspec_kws=None)
```

**方法：**
- `map(func, *args, **kwargs)` - 對每個分面應用函數
- `map_dataframe(func, *args, **kwargs)` - 用完整 DataFrame 應用函數
- `set_axis_labels(x_var, y_var)` - 設定軸標籤
- `set_titles(template, **kwargs)` - 設定子圖標題
- `set(kwargs)` - 設定所有軸的屬性
- `add_legend(legend_data, title, label_order, **kwargs)` - 添加圖例
- `savefig(*args, **kwargs)` - 儲存圖形

**範例：**
```python
g = sns.FacetGrid(df, col='time', row='sex', hue='smoker',
                  height=3, aspect=1.5, margin_titles=True)
g.map(sns.scatterplot, 'total_bill', 'tip', alpha=0.7)
g.add_legend()
g.set_axis_labels('Total Bill ($)', 'Tip ($)')
g.set_titles('{col_name} | {row_name}')
```

### PairGrid

**用途：** 用於繪製資料集中成對關係的網格。

**初始化：**
```python
g = sns.PairGrid(data, hue=None, vars=None,
                 x_vars=None, y_vars=None,
                 hue_order=None, palette=None,
                 hue_kws=None, corner=False,
                 diag_sharey=True, height=2.5,
                 aspect=1, layout_pad=0.5,
                 despine=True, dropna=False)
```

**方法：**
- `map(func, **kwargs)` - 對所有子圖應用函數
- `map_diag(func, **kwargs)` - 應用於對角線
- `map_offdiag(func, **kwargs)` - 應用於非對角線
- `map_upper(func, **kwargs)` - 應用於上三角
- `map_lower(func, **kwargs)` - 應用於下三角
- `add_legend(legend_data, **kwargs)` - 添加圖例
- `savefig(*args, **kwargs)` - 儲存圖形

**範例：**
```python
g = sns.PairGrid(df, hue='species', vars=['a', 'b', 'c', 'd'],
                 corner=True, height=2.5)
g.map_upper(sns.scatterplot, alpha=0.5)
g.map_lower(sns.kdeplot)
g.map_diag(sns.histplot, kde=True)
g.add_legend()
```

### JointGrid

**用途：** 用於繪製帶有邊際單變量圖的雙變量圖的網格。

**初始化：**
```python
g = sns.JointGrid(data=None, x=None, y=None, hue=None,
                  height=6, ratio=5, space=0.2,
                  dropna=False, xlim=None, ylim=None,
                  marginal_ticks=False, hue_order=None,
                  palette=None)
```

**方法：**
- `plot(joint_func, marginal_func, **kwargs)` - 同時繪製聯合和邊際
- `plot_joint(func, **kwargs)` - 繪製聯合分布
- `plot_marginals(func, **kwargs)` - 繪製邊際分布
- `refline(x, y, **kwargs)` - 添加參考線
- `set_axis_labels(xlabel, ylabel, **kwargs)` - 設定軸標籤
- `savefig(*args, **kwargs)` - 儲存圖形

**範例：**
```python
g = sns.JointGrid(data=df, x='x', y='y', hue='group',
                  height=6, ratio=5, space=0.2)
g.plot_joint(sns.scatterplot, alpha=0.5)
g.plot_marginals(sns.histplot, kde=True)
g.set_axis_labels('Variable X', 'Variable Y')
```
