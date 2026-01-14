# Matplotlib 繪圖類型指南

matplotlib 中不同繪圖類型的完整指南，附有範例和使用場景。

## 1. 折線圖

**使用場景：** 時間序列、連續資料、趨勢、函數視覺化

### 基本折線圖
```python
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, linewidth=2, label='Data')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.legend()
```

### 多條線
```python
ax.plot(x, y1, label='Dataset 1', linewidth=2)
ax.plot(x, y2, label='Dataset 2', linewidth=2, linestyle='--')
ax.plot(x, y3, label='Dataset 3', linewidth=2, linestyle=':')
ax.legend()
```

### 帶標記的線
```python
ax.plot(x, y, marker='o', markersize=8, linestyle='-',
        linewidth=2, markerfacecolor='red', markeredgecolor='black')
```

### 階梯圖
```python
ax.step(x, y, where='mid', linewidth=2, label='Step function')
# where 選項：'pre'、'post'、'mid'
```

### 誤差條
```python
ax.errorbar(x, y, yerr=error, fmt='o-', linewidth=2,
            capsize=5, capthick=2, label='With uncertainty')
```

## 2. 散點圖

**使用場景：** 相關性、變數之間的關係、群集、離群值

### 基本散點圖
```python
ax.scatter(x, y, s=50, alpha=0.6)
```

### 帶大小和顏色的散點圖
```python
scatter = ax.scatter(x, y, s=sizes*100, c=colors,
                     cmap='viridis', alpha=0.6, edgecolors='black')
plt.colorbar(scatter, ax=ax, label='Color variable')
```

### 類別散點圖
```python
for category in categories:
    mask = data['category'] == category
    ax.scatter(data[mask]['x'], data[mask]['y'],
               label=category, s=50, alpha=0.7)
ax.legend()
```

## 3. 長條圖

**使用場景：** 類別比較、離散資料、計數

### 垂直長條圖
```python
ax.bar(categories, values, color='steelblue',
       edgecolor='black', linewidth=1.5)
ax.set_ylabel('Values')
```

### 水平長條圖
```python
ax.barh(categories, values, color='coral',
        edgecolor='black', linewidth=1.5)
ax.set_xlabel('Values')
```

### 群組長條圖
```python
x = np.arange(len(categories))
width = 0.35

ax.bar(x - width/2, values1, width, label='Group 1')
ax.bar(x + width/2, values2, width, label='Group 2')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
```

### 堆疊長條圖
```python
ax.bar(categories, values1, label='Part 1')
ax.bar(categories, values2, bottom=values1, label='Part 2')
ax.bar(categories, values3, bottom=values1+values2, label='Part 3')
ax.legend()
```

### 帶誤差條的長條圖
```python
ax.bar(categories, values, yerr=errors, capsize=5,
       color='steelblue', edgecolor='black')
```

### 帶圖案的長條圖
```python
bars1 = ax.bar(x - width/2, values1, width, label='Group 1',
               color='white', edgecolor='black', hatch='//')
bars2 = ax.bar(x + width/2, values2, width, label='Group 2',
               color='white', edgecolor='black', hatch='\\\\')
```

## 4. 直方圖

**使用場景：** 分佈、頻率分析

### 基本直方圖
```python
ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
```

### 多個重疊直方圖
```python
ax.hist(data1, bins=30, alpha=0.5, label='Dataset 1')
ax.hist(data2, bins=30, alpha=0.5, label='Dataset 2')
ax.legend()
```

### 正規化直方圖（密度）
```python
ax.hist(data, bins=30, density=True, alpha=0.7,
        edgecolor='black', label='Empirical')

# 疊加理論分佈
from scipy.stats import norm
x = np.linspace(data.min(), data.max(), 100)
ax.plot(x, norm.pdf(x, data.mean(), data.std()),
        'r-', linewidth=2, label='Normal fit')
ax.legend()
```

### 2D 直方圖（Hexbin）
```python
hexbin = ax.hexbin(x, y, gridsize=30, cmap='Blues')
plt.colorbar(hexbin, ax=ax, label='Counts')
```

### 2D 直方圖（hist2d）
```python
h = ax.hist2d(x, y, bins=30, cmap='Blues')
plt.colorbar(h[3], ax=ax, label='Counts')
```

## 5. 箱形圖和小提琴圖

**使用場景：** 統計分佈、離群值偵測、分佈比較

### 箱形圖
```python
ax.boxplot([data1, data2, data3],
           labels=['Group A', 'Group B', 'Group C'],
           showmeans=True, meanline=True)
ax.set_ylabel('Values')
```

### 水平箱形圖
```python
ax.boxplot([data1, data2, data3], vert=False,
           labels=['Group A', 'Group B', 'Group C'])
ax.set_xlabel('Values')
```

### 小提琴圖
```python
parts = ax.violinplot([data1, data2, data3],
                      positions=[1, 2, 3],
                      showmeans=True, showmedians=True)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Group A', 'Group B', 'Group C'])
```

## 6. 熱圖

**使用場景：** 矩陣資料、相關性、強度圖

### 基本熱圖
```python
im = ax.imshow(matrix, cmap='coolwarm', aspect='auto')
plt.colorbar(im, ax=ax, label='Values')
ax.set_xlabel('X')
ax.set_ylabel('Y')
```

### 帶註釋的熱圖
```python
im = ax.imshow(matrix, cmap='coolwarm')
plt.colorbar(im, ax=ax)

# 新增文字註釋
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                       ha='center', va='center', color='black')
```

### 相關矩陣
```python
corr = data.corr()
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, label='Correlation')

# 設定刻度標籤
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right')
ax.set_yticklabels(corr.columns)
```

## 7. 等高線圖

**使用場景：** 2D 平面上的 3D 資料、地形、函數視覺化

### 等高線
```python
contour = ax.contour(X, Y, Z, levels=10, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)
plt.colorbar(contour, ax=ax)
```

### 填充等高線
```python
contourf = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(contourf, ax=ax)
```

### 組合等高線
```python
contourf = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
contour = ax.contour(X, Y, Z, levels=10, colors='black',
                     linewidths=0.5, alpha=0.4)
ax.clabel(contour, inline=True, fontsize=8)
plt.colorbar(contourf, ax=ax)
```

## 8. 圓餅圖

**使用場景：** 比例、百分比（謹慎使用）

### 基本圓餅圖
```python
ax.pie(sizes, labels=labels, autopct='%1.1f%%',
       startangle=90, colors=colors)
ax.axis('equal')  # 等長寬比確保圓形
```

### 分離圓餅圖
```python
explode = (0.1, 0, 0, 0)  # 分離第一個扇形
ax.pie(sizes, explode=explode, labels=labels,
       autopct='%1.1f%%', shadow=True, startangle=90)
ax.axis('equal')
```

### 環形圖
```python
ax.pie(sizes, labels=labels, autopct='%1.1f%%',
       wedgeprops=dict(width=0.5), startangle=90)
ax.axis('equal')
```

## 9. 極座標圖

**使用場景：** 週期性資料、方向性資料、雷達圖

### 基本極座標圖
```python
theta = np.linspace(0, 2*np.pi, 100)
r = np.abs(np.sin(2*theta))

ax = plt.subplot(111, projection='polar')
ax.plot(theta, r, linewidth=2)
```

### 雷達圖
```python
categories = ['A', 'B', 'C', 'D', 'E']
values = [4, 3, 5, 2, 4]

# 將第一個值加到最後以閉合多邊形
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
values_closed = np.concatenate((values, [values[0]]))
angles_closed = np.concatenate((angles, [angles[0]]))

ax = plt.subplot(111, projection='polar')
ax.plot(angles_closed, values_closed, 'o-', linewidth=2)
ax.fill(angles_closed, values_closed, alpha=0.25)
ax.set_xticks(angles)
ax.set_xticklabels(categories)
```

## 10. 流線圖和箭頭圖

**使用場景：** 向量場、流場視覺化

### 箭頭圖（向量場）
```python
ax.quiver(X, Y, U, V, alpha=0.8)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')
```

### 流線圖
```python
ax.streamplot(X, Y, U, V, density=1.5, color='k', linewidth=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')
```

## 11. 填充區域

**使用場景：** 不確定性範圍、信賴區間、曲線下面積

### 兩曲線之間填充
```python
ax.plot(x, y, 'k-', linewidth=2, label='Mean')
ax.fill_between(x, y - std, y + std, alpha=0.3,
                label='±1 std dev')
ax.legend()
```

### 條件填充
```python
ax.plot(x, y1, label='Line 1')
ax.plot(x, y2, label='Line 2')
ax.fill_between(x, y1, y2, where=(y2 >= y1),
                alpha=0.3, label='y2 > y1', interpolate=True)
ax.legend()
```

## 12. 3D 繪圖

**使用場景：** 三維資料視覺化

### 3D 散點圖
```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c=colors, cmap='viridis',
                     marker='o', s=50)
plt.colorbar(scatter, ax=ax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
```

### 3D 曲面圖
```python
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis',
                       edgecolor='none', alpha=0.9)
plt.colorbar(surf, ax=ax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
```

### 3D 線框圖
```python
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z, color='black', linewidth=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
```

### 3D 等高線圖
```python
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.contour(X, Y, Z, levels=15, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
```

## 13. 特殊繪圖

### 莖葉圖
```python
ax.stem(x, y, linefmt='C0-', markerfmt='C0o', basefmt='k-')
ax.set_xlabel('X')
ax.set_ylabel('Y')
```

### 填充多邊形
```python
vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
from matplotlib.patches import Polygon
polygon = Polygon(vertices, closed=True, edgecolor='black',
                  facecolor='lightblue', alpha=0.5)
ax.add_patch(polygon)
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.5, 1.5)
```

### 階梯圖
```python
ax.stairs(values, edges, fill=True, alpha=0.5)
```

### 間斷長條圖（甘特圖風格）
```python
ax.broken_barh([(10, 50), (100, 20), (130, 10)], (10, 9),
               facecolors='tab:blue')
ax.broken_barh([(10, 20), (50, 50), (120, 30)], (20, 9),
               facecolors='tab:orange')
ax.set_ylim(5, 35)
ax.set_xlim(0, 200)
ax.set_xlabel('Time')
ax.set_yticks([15, 25])
ax.set_yticklabels(['Task 1', 'Task 2'])
```

## 14. 時間序列圖

### 基本時間序列
```python
import pandas as pd
import matplotlib.dates as mdates

ax.plot(dates, values, linewidth=2)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.xticks(rotation=45)
ax.set_xlabel('Date')
ax.set_ylabel('Value')
```

### 帶陰影區域的時間序列
```python
ax.plot(dates, values, linewidth=2)
# 為週末或特定時期加陰影
ax.axvspan(start_date, end_date, alpha=0.2, color='gray')
```

## 繪圖選擇指南

| 資料類型 | 建議繪圖 | 替代選項 |
|----------|----------|----------|
| 單一連續變數 | 直方圖、KDE | 箱形圖、小提琴圖 |
| 兩個連續變數 | 散點圖 | Hexbin、2D 直方圖 |
| 時間序列 | 折線圖 | 面積圖、階梯圖 |
| 類別 vs 連續 | 長條圖、箱形圖 | 小提琴圖、帶狀圖 |
| 兩個類別變數 | 熱圖 | 群組長條圖 |
| 三個連續變數 | 3D 散點圖、等高線圖 | 顏色編碼散點圖 |
| 比例 | 長條圖 | 圓餅圖（謹慎使用） |
| 分佈比較 | 箱形圖、小提琴圖 | 重疊直方圖 |
| 相關矩陣 | 熱圖 | 聚類熱圖 |
| 向量場 | 箭頭圖、流線圖 | - |
| 函數視覺化 | 折線圖、等高線圖 | 3D 曲面 |
