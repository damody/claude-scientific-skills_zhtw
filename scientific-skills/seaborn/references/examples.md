# Seaborn 常見用例和範例

本文件提供使用 seaborn 進行常見資料視覺化場景的實用範例。

## 探索性資料分析

### 快速資料集概覽

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 載入資料
df = pd.read_csv('data.csv')

# 所有數值變數的成對關係
sns.pairplot(df, hue='target_variable', corner=True, diag_kind='kde')
plt.suptitle('Dataset Overview', y=1.01)
plt.savefig('overview.png', dpi=300, bbox_inches='tight')
```

### 分布探索

```python
# 跨類別的多重分布
g = sns.displot(
    data=df,
    x='measurement',
    hue='condition',
    col='timepoint',
    kind='kde',
    fill=True,
    height=3,
    aspect=1.5,
    col_wrap=3,
    common_norm=False
)
g.set_axis_labels('Measurement Value', 'Density')
g.set_titles('{col_name}')
```

### 相關性分析

```python
# 計算相關矩陣
corr = df.select_dtypes(include='number').corr()

# 建立上三角遮罩
mask = np.triu(np.ones_like(corr, dtype=bool))

# 繪製熱圖
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={'shrink': 0.8}
)
plt.title('Correlation Matrix')
plt.tight_layout()
```

## 科學出版品

### 多面板圖形與不同圖形類型

```python
# 設定出版品樣式
sns.set_theme(style='ticks', context='paper', font_scale=1.1)
sns.set_palette('colorblind')

# 建立具有自訂版面的圖形
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 面板 A：時間序列
ax1 = fig.add_subplot(gs[0, :2])
sns.lineplot(
    data=timeseries_df,
    x='time',
    y='expression',
    hue='gene',
    style='treatment',
    markers=True,
    dashes=False,
    ax=ax1
)
ax1.set_title('A. Gene Expression Over Time', loc='left', fontweight='bold')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Expression Level (AU)')

# 面板 B：分布比較
ax2 = fig.add_subplot(gs[0, 2])
sns.violinplot(
    data=expression_df,
    x='treatment',
    y='expression',
    inner='box',
    ax=ax2
)
ax2.set_title('B. Expression Distribution', loc='left', fontweight='bold')
ax2.set_xlabel('Treatment')
ax2.set_ylabel('')

# 面板 C：相關性
ax3 = fig.add_subplot(gs[1, 0])
sns.scatterplot(
    data=correlation_df,
    x='gene1',
    y='gene2',
    hue='cell_type',
    alpha=0.6,
    ax=ax3
)
sns.regplot(
    data=correlation_df,
    x='gene1',
    y='gene2',
    scatter=False,
    color='black',
    ax=ax3
)
ax3.set_title('C. Gene Correlation', loc='left', fontweight='bold')
ax3.set_xlabel('Gene 1 Expression')
ax3.set_ylabel('Gene 2 Expression')

# 面板 D：熱圖
ax4 = fig.add_subplot(gs[1, 1:])
sns.heatmap(
    sample_matrix,
    cmap='RdBu_r',
    center=0,
    annot=True,
    fmt='.1f',
    cbar_kws={'label': 'Log2 Fold Change'},
    ax=ax4
)
ax4.set_title('D. Treatment Effects', loc='left', fontweight='bold')
ax4.set_xlabel('Sample')
ax4.set_ylabel('Gene')

# 清理
sns.despine()
plt.savefig('figure.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
```

### 帶有顯著性標註的箱形圖

```python
import numpy as np
from scipy import stats

# 建立圖形
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(
    data=df,
    x='treatment',
    y='response',
    order=['Control', 'Low', 'Medium', 'High'],
    palette='Set2',
    ax=ax
)

# 添加個別點
sns.stripplot(
    data=df,
    x='treatment',
    y='response',
    order=['Control', 'Low', 'Medium', 'High'],
    color='black',
    alpha=0.3,
    size=3,
    ax=ax
)

# 添加顯著性標記線
def add_significance_bar(ax, x1, x2, y, h, text):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], 'k-', lw=1.5)
    ax.text((x1+x2)/2, y+h, text, ha='center', va='bottom')

y_max = df['response'].max()
add_significance_bar(ax, 0, 3, y_max + 1, 0.5, '***')
add_significance_bar(ax, 0, 1, y_max + 3, 0.5, 'ns')

ax.set_ylabel('Response (μM)')
ax.set_xlabel('Treatment Condition')
ax.set_title('Treatment Response Analysis')
sns.despine()
```

## 時間序列分析

### 多重時間序列與信賴帶

```python
# 帶有自動聚合的繪圖
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(
    data=timeseries_df,
    x='timestamp',
    y='value',
    hue='sensor',
    style='location',
    markers=True,
    dashes=False,
    errorbar=('ci', 95),
    ax=ax
)

# 自訂
ax.set_xlabel('Date')
ax.set_ylabel('Measurement (units)')
ax.set_title('Sensor Measurements Over Time')
ax.legend(title='Sensor & Location', bbox_to_anchor=(1.05, 1), loc='upper left')

# 格式化日期 x 軸
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
```

### 分面時間序列

```python
# 建立分面時間序列
g = sns.relplot(
    data=long_timeseries,
    x='date',
    y='measurement',
    hue='device',
    col='location',
    row='metric',
    kind='line',
    height=3,
    aspect=2,
    errorbar='sd',
    facet_kws={'sharex': True, 'sharey': False}
)

# 自訂分面標題
g.set_titles('{row_name} - {col_name}')
g.set_axis_labels('Date', 'Value')

# 旋轉 x 軸標籤
for ax in g.axes.flat:
    ax.tick_params(axis='x', rotation=45)

g.tight_layout()
```

## 類別比較

### 巢狀類別變數

```python
# 建立圖形
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左側面板：分組長條圖
sns.barplot(
    data=df,
    x='category',
    y='value',
    hue='subcategory',
    errorbar=('ci', 95),
    capsize=0.1,
    ax=axes[0]
)
axes[0].set_title('Mean Values with 95% CI')
axes[0].set_ylabel('Value (units)')
axes[0].legend(title='Subcategory')

# 右側面板：條狀圖 + 小提琴圖
sns.violinplot(
    data=df,
    x='category',
    y='value',
    hue='subcategory',
    inner=None,
    alpha=0.3,
    ax=axes[1]
)
sns.stripplot(
    data=df,
    x='category',
    y='value',
    hue='subcategory',
    dodge=True,
    size=3,
    alpha=0.6,
    ax=axes[1]
)
axes[1].set_title('Distribution of Individual Values')
axes[1].set_ylabel('')
axes[1].get_legend().remove()

plt.tight_layout()
```

### 趨勢點圖

```python
# 顯示數值如何隨類別變化
sns.pointplot(
    data=df,
    x='timepoint',
    y='score',
    hue='treatment',
    markers=['o', 's', '^'],
    linestyles=['-', '--', '-.'],
    dodge=0.3,
    capsize=0.1,
    errorbar=('ci', 95)
)

plt.xlabel('Timepoint')
plt.ylabel('Performance Score')
plt.title('Treatment Effects Over Time')
plt.legend(title='Treatment', bbox_to_anchor=(1.05, 1), loc='upper left')
sns.despine()
plt.tight_layout()
```

## 迴歸與關係

### 帶有分面的線性迴歸

```python
# 為每個類別擬合獨立的迴歸
g = sns.lmplot(
    data=df,
    x='predictor',
    y='response',
    hue='treatment',
    col='cell_line',
    height=4,
    aspect=1.2,
    scatter_kws={'alpha': 0.5, 's': 50},
    ci=95,
    palette='Set2'
)

g.set_axis_labels('Predictor Variable', 'Response Variable')
g.set_titles('{col_name}')
g.tight_layout()
```

### 多項式迴歸

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, order in enumerate([1, 2, 3]):
    sns.regplot(
        data=df,
        x='x',
        y='y',
        order=order,
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red'},
        ci=95,
        ax=axes[idx]
    )
    axes[idx].set_title(f'Order {order} Polynomial Fit')
    axes[idx].set_xlabel('X Variable')
    axes[idx].set_ylabel('Y Variable')

plt.tight_layout()
```

### 殘差分析

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 主要迴歸
sns.regplot(data=df, x='x', y='y', ax=axes[0, 0])
axes[0, 0].set_title('Regression Fit')

# 殘差 vs 擬合值
sns.residplot(data=df, x='x', y='y', lowess=True,
              scatter_kws={'alpha': 0.5},
              line_kws={'color': 'red', 'lw': 2},
              ax=axes[0, 1])
axes[0, 1].set_title('Residuals vs Fitted')
axes[0, 1].axhline(0, ls='--', color='gray')

# Q-Q 圖（使用 scipy）
from scipy import stats as sp_stats
residuals = df['y'] - np.poly1d(np.polyfit(df['x'], df['y'], 1))(df['x'])
sp_stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')

# 殘差直方圖
sns.histplot(residuals, kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Residual Distribution')
axes[1, 1].set_xlabel('Residuals')

plt.tight_layout()
```

## 雙變量和聯合分布

### 多重呈現方式的聯合圖

```python
# 帶有邊際分布的散點圖
g = sns.jointplot(
    data=df,
    x='var1',
    y='var2',
    hue='category',
    kind='scatter',
    height=8,
    ratio=4,
    space=0.1,
    joint_kws={'alpha': 0.5, 's': 50},
    marginal_kws={'kde': True, 'bins': 30}
)

# 添加參考線
g.ax_joint.axline((0, 0), slope=1, color='r', ls='--', alpha=0.5, label='y=x')
g.ax_joint.legend()

g.set_axis_labels('Variable 1', 'Variable 2', fontsize=12)
```

### KDE 等高線圖

```python
fig, ax = plt.subplots(figsize=(8, 8))

# 帶有填充等高線的雙變量 KDE
sns.kdeplot(
    data=df,
    x='x',
    y='y',
    fill=True,
    levels=10,
    cmap='viridis',
    thresh=0.05,
    ax=ax
)

# 疊加散點圖
sns.scatterplot(
    data=df,
    x='x',
    y='y',
    color='white',
    edgecolor='black',
    s=50,
    alpha=0.6,
    ax=ax
)

ax.set_xlabel('X Variable')
ax.set_ylabel('Y Variable')
ax.set_title('Bivariate Distribution')
```

### 帶有邊際分布的六邊形圖

```python
# 用於大型資料集
g = sns.jointplot(
    data=large_df,
    x='x',
    y='y',
    kind='hex',
    height=8,
    ratio=5,
    space=0.1,
    joint_kws={'gridsize': 30, 'cmap': 'viridis'},
    marginal_kws={'bins': 50, 'color': 'skyblue'}
)

g.set_axis_labels('X Variable', 'Y Variable')
```

## 矩陣和熱圖視覺化

### 階層式聚類熱圖

```python
# 準備資料（樣本 x 特徵）
data_matrix = df.set_index('sample_id')[feature_columns]

# 建立顏色註解
row_colors = df.set_index('sample_id')['condition'].map({
    'control': '#1f77b4',
    'treatment': '#ff7f0e'
})

col_colors = pd.Series(['#2ca02c' if 'gene' in col else '#d62728'
                        for col in data_matrix.columns])

# 繪圖
g = sns.clustermap(
    data_matrix,
    method='ward',
    metric='euclidean',
    z_score=0,  # 正規化列
    cmap='RdBu_r',
    center=0,
    row_colors=row_colors,
    col_colors=col_colors,
    figsize=(12, 10),
    dendrogram_ratio=(0.1, 0.1),
    cbar_pos=(0.02, 0.8, 0.03, 0.15),
    linewidths=0.5
)

g.ax_heatmap.set_xlabel('Features')
g.ax_heatmap.set_ylabel('Samples')
plt.savefig('clustermap.png', dpi=300, bbox_inches='tight')
```

### 帶有自訂色彩條的註解熱圖

```python
# 將資料轉換為熱圖格式
pivot_data = df.pivot(index='row_var', columns='col_var', values='value')

# 建立熱圖
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    pivot_data,
    annot=True,
    fmt='.1f',
    cmap='RdYlGn',
    center=pivot_data.mean().mean(),
    vmin=pivot_data.min().min(),
    vmax=pivot_data.max().max(),
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={
        'label': 'Value (units)',
        'orientation': 'vertical',
        'shrink': 0.8,
        'aspect': 20
    },
    ax=ax
)

ax.set_title('Variable Relationships', fontsize=14, pad=20)
ax.set_xlabel('Column Variable', fontsize=12)
ax.set_ylabel('Row Variable', fontsize=12)

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
```

## 統計比較

### 前後比較

```python
# 重塑資料以進行配對比較
df_paired = df.melt(
    id_vars='subject',
    value_vars=['before', 'after'],
    var_name='timepoint',
    value_name='measurement'
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左側：個別軌跡
for subject in df_paired['subject'].unique():
    subject_data = df_paired[df_paired['subject'] == subject]
    axes[0].plot(subject_data['timepoint'], subject_data['measurement'],
                 'o-', alpha=0.3, color='gray')

sns.pointplot(
    data=df_paired,
    x='timepoint',
    y='measurement',
    color='red',
    markers='D',
    scale=1.5,
    errorbar=('ci', 95),
    capsize=0.2,
    ax=axes[0]
)
axes[0].set_title('Individual Changes')
axes[0].set_ylabel('Measurement')

# 右側：分布比較
sns.violinplot(
    data=df_paired,
    x='timepoint',
    y='measurement',
    inner='box',
    ax=axes[1]
)
sns.swarmplot(
    data=df_paired,
    x='timepoint',
    y='measurement',
    color='black',
    alpha=0.5,
    size=3,
    ax=axes[1]
)
axes[1].set_title('Distribution Comparison')
axes[1].set_ylabel('')

plt.tight_layout()
```

### 劑量反應曲線

```python
# 建立劑量反應圖
fig, ax = plt.subplots(figsize=(8, 6))

# 繪製個別點
sns.stripplot(
    data=dose_df,
    x='dose',
    y='response',
    order=sorted(dose_df['dose'].unique()),
    color='gray',
    alpha=0.3,
    jitter=0.2,
    ax=ax
)

# 疊加平均值與信賴區間
sns.pointplot(
    data=dose_df,
    x='dose',
    y='response',
    order=sorted(dose_df['dose'].unique()),
    color='blue',
    markers='o',
    scale=1.2,
    errorbar=('ci', 95),
    capsize=0.1,
    ax=ax
)

# 擬合 S 型曲線
from scipy.optimize import curve_fit

def sigmoid(x, bottom, top, ec50, hill):
    return bottom + (top - bottom) / (1 + (ec50 / x) ** hill)

doses_numeric = dose_df['dose'].astype(float)
params, _ = curve_fit(sigmoid, doses_numeric, dose_df['response'])

x_smooth = np.logspace(np.log10(doses_numeric.min()),
                       np.log10(doses_numeric.max()), 100)
y_smooth = sigmoid(x_smooth, *params)

ax.plot(range(len(sorted(dose_df['dose'].unique()))),
        sigmoid(sorted(doses_numeric.unique()), *params),
        'r-', linewidth=2, label='Sigmoid Fit')

ax.set_xlabel('Dose')
ax.set_ylabel('Response')
ax.set_title('Dose-Response Analysis')
ax.legend()
sns.despine()
```

## 自訂樣式

### 從十六進位碼自訂調色盤

```python
# 定義自訂調色盤
custom_palette = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F']
sns.set_palette(custom_palette)

# 或用於特定圖形
sns.scatterplot(
    data=df,
    x='x',
    y='y',
    hue='category',
    palette=custom_palette
)
```

### 出版品質主題

```python
# 設定完整主題
sns.set_theme(
    context='paper',
    style='ticks',
    palette='colorblind',
    font='Arial',
    font_scale=1.1,
    rc={
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.format': 'pdf',
        'axes.linewidth': 1.0,
        'axes.labelweight': 'bold',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': False,
        'pdf.fonttype': 42,  # PDF 用 True Type 字型
    }
)
```

### 以零為中心的發散色彩映射

```python
# 用於具有有意義零點的資料（例如 log fold change）
from matplotlib.colors import TwoSlopeNorm

# 找出資料範圍
vmin, vmax = df['value'].min(), df['value'].max()
vcenter = 0

# 建立正規化
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

# 繪圖
sns.heatmap(
    pivot_data,
    cmap='RdBu_r',
    norm=norm,
    center=0,
    annot=True,
    fmt='.2f'
)
```

## 大型資料集

### 降採樣策略

```python
# 對於非常大的資料集，智慧採樣
def smart_sample(df, target_size=10000, category_col=None):
    if len(df) <= target_size:
        return df

    if category_col:
        # 分層採樣
        return df.groupby(category_col, group_keys=False).apply(
            lambda x: x.sample(min(len(x), target_size // df[category_col].nunique()))
        )
    else:
        # 簡單隨機採樣
        return df.sample(target_size)

# 使用採樣資料進行視覺化
df_sampled = smart_sample(large_df, target_size=5000, category_col='category')

sns.scatterplot(data=df_sampled, x='x', y='y', hue='category', alpha=0.5)
```

### 密集散點圖的六邊形圖

```python
# 用於數百萬個點
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 常規散點圖（慢）
axes[0].scatter(df['x'], df['y'], alpha=0.1, s=1)
axes[0].set_title('Scatter (all points)')

# 六邊形圖（快）
hb = axes[1].hexbin(df['x'], df['y'], gridsize=50, cmap='viridis', mincnt=1)
axes[1].set_title('Hexbin Aggregation')
plt.colorbar(hb, ax=axes[1], label='Count')

plt.tight_layout()
```

## 筆記本的互動元素

### 可調整參數

```python
from ipywidgets import interact, FloatSlider

@interact(bandwidth=FloatSlider(min=0.1, max=3.0, step=0.1, value=1.0))
def plot_kde(bandwidth):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='value', hue='category',
                bw_adjust=bandwidth, fill=True)
    plt.title(f'KDE with bandwidth adjustment = {bandwidth}')
    plt.show()
```

### 動態篩選

```python
from ipywidgets import interact, SelectMultiple

categories = df['category'].unique().tolist()

@interact(selected=SelectMultiple(options=categories, value=[categories[0]]))
def filtered_plot(selected):
    filtered_df = df[df['category'].isin(selected)]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=filtered_df, x='category', y='value', ax=ax)
    ax.set_title(f'Showing {len(selected)} categories')
    plt.show()
```
