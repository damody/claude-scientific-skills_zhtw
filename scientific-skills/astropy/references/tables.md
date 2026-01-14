# 表格操作（astropy.table）

`astropy.table` 模組提供處理表格資料的彈性工具，支援單位、遮罩值和各種檔案格式。

## 建立表格

### 基本表格建立

```python
from astropy.table import Table, QTable
import astropy.units as u
import numpy as np

# 從欄位陣列
a = [1, 4, 5]
b = [2.0, 5.0, 8.2]
c = ['x', 'y', 'z']

t = Table([a, b, c], names=('id', 'flux', 'name'))

# 帶單位（使用 QTable）
flux = [1.2, 2.3, 3.4] * u.Jy
wavelength = [500, 600, 700] * u.nm
t = QTable([flux, wavelength], names=('flux', 'wavelength'))
```

### 從列列表

```python
# 元組列表
rows = [(1, 10.5, 'A'), (2, 11.2, 'B'), (3, 12.3, 'C')]
t = Table(rows=rows, names=('id', 'value', 'name'))

# 字典列表
rows = [{'id': 1, 'value': 10.5}, {'id': 2, 'value': 11.2}]
t = Table(rows)
```

### 從 NumPy 陣列

```python
# 結構化陣列
arr = np.array([(1, 2.0, 'x'), (4, 5.0, 'y')],
               dtype=[('a', 'i4'), ('b', 'f8'), ('c', 'U10')])
t = Table(arr)

# 帶欄位名稱的 2D 陣列
data = np.random.random((100, 3))
t = Table(data, names=['col1', 'col2', 'col3'])
```

### 從 Pandas DataFrame

```python
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
t = Table.from_pandas(df)
```

## 存取表格資料

### 基本存取

```python
# 欄位存取
ra_col = t['ra']           # 回傳 Column 物件
dec_col = t['dec']

# 列存取
first_row = t[0]           # 回傳 Row 物件
row_slice = t[10:20]       # 回傳新 Table

# 儲存格存取
value = t['ra'][5]         # 'ra' 欄位中第 6 個值
value = t[5]['ra']         # 相同結果

# 多個欄位
subset = t['ra', 'dec', 'mag']
```

### 表格屬性

```python
len(t)              # 列數
t.colnames          # 欄位名稱列表
t.dtype             # 欄位資料類型
t.info              # 詳細資訊
t.meta              # 中繼資料字典
```

### 迭代

```python
# 遍歷列
for row in t:
    print(row['ra'], row['dec'])

# 遍歷欄位
for colname in t.colnames:
    print(t[colname])
```

## 修改表格

### 添加欄位

```python
# 添加新欄位
t['new_col'] = [1, 2, 3, 4, 5]
t['calc'] = t['a'] + t['b']  # 計算欄位

# 添加帶單位的欄位
t['velocity'] = [10, 20, 30] * u.km / u.s

# 添加空欄位
from astropy.table import Column
t['empty'] = Column(length=len(t), dtype=float)

# 在特定位置插入
t.add_column([7, 8, 9], name='inserted', index=2)
```

### 移除欄位

```python
# 移除單一欄位
t.remove_column('old_col')

# 移除多個欄位
t.remove_columns(['col1', 'col2'])

# 刪除語法
del t['col_name']

# 只保留特定欄位
t.keep_columns(['ra', 'dec', 'mag'])
```

### 重新命名欄位

```python
t.rename_column('old_name', 'new_name')

# 重新命名多個
t.rename_columns(['old1', 'old2'], ['new1', 'new2'])
```

### 添加列

```python
# 添加單一列
t.add_row([1, 2.5, 'new'])

# 以字典添加列
t.add_row({'ra': 10.5, 'dec': 41.2, 'mag': 18.5})

# 注意：逐一添加列速度很慢！
# 最好收集列後一次建立表格
```

### 修改資料

```python
# 修改欄位值
t['flux'] = t['flux'] * gain
t['mag'][t['mag'] < 0] = np.nan

# 修改單一儲存格
t['ra'][5] = 10.5

# 修改整列
t[0] = [new_id, new_ra, new_dec]
```

## 排序和過濾

### 排序

```python
# 按單一欄位排序
t.sort('mag')

# 降冪排序
t.sort('mag', reverse=True)

# 按多個欄位排序
t.sort(['priority', 'mag'])

# 取得排序索引而不修改表格
indices = t.argsort('mag')
sorted_table = t[indices]
```

### 過濾

```python
# 布林索引
bright = t[t['mag'] < 18]
nearby = t[t['distance'] < 100*u.pc]

# 多個條件
selected = t[(t['mag'] < 18) & (t['dec'] > 0)]

# 使用 numpy 函數
high_snr = t[np.abs(t['flux'] / t['error']) > 5]
```

## 讀取和寫入檔案

### 支援的格式

FITS、HDF5、ASCII（CSV、ECSV、IPAC 等）、VOTable、Parquet、ASDF

### 讀取檔案

```python
# 自動格式檢測
t = Table.read('catalog.fits')
t = Table.read('data.csv')
t = Table.read('table.vot')

# 明確指定格式
t = Table.read('data.txt', format='ascii')
t = Table.read('catalog.hdf5', path='/dataset/table')

# 從 FITS 讀取特定 HDU
t = Table.read('file.fits', hdu=2)
```

### 寫入檔案

```python
# 從副檔名自動判斷格式
t.write('output.fits')
t.write('output.csv')

# 指定格式
t.write('output.txt', format='ascii.csv')
t.write('output.hdf5', path='/data/table', serialize_meta=True)

# 覆寫現有檔案
t.write('output.fits', overwrite=True)
```

### ASCII 格式選項

```python
# 帶自訂分隔符的 CSV
t.write('output.csv', format='ascii.csv', delimiter='|')

# 固定寬度格式
t.write('output.txt', format='ascii.fixed_width')

# IPAC 格式
t.write('output.tbl', format='ascii.ipac')

# LaTeX 表格
t.write('table.tex', format='ascii.latex')
```

## 表格操作

### 堆疊表格（垂直）

```python
from astropy.table import vstack

# 垂直串接表格
t1 = Table([[1, 2], [3, 4]], names=('a', 'b'))
t2 = Table([[5, 6], [7, 8]], names=('a', 'b'))
t_combined = vstack([t1, t2])
```

### 連接表格（水平）

```python
from astropy.table import hstack

# 水平串接表格
t1 = Table([[1, 2]], names=['a'])
t2 = Table([[3, 4]], names=['b'])
t_combined = hstack([t1, t2])
```

### 資料庫風格連接

```python
from astropy.table import join

# 在共同欄位上內連接
t1 = Table([[1, 2, 3], ['a', 'b', 'c']], names=('id', 'data1'))
t2 = Table([[1, 2, 4], ['x', 'y', 'z']], names=('id', 'data2'))
t_joined = join(t1, t2, keys='id')

# 左/右/外連接
t_joined = join(t1, t2, join_type='left')
t_joined = join(t1, t2, join_type='outer')
```

### 群組和聚合

```python
# 按欄位群組
g = t.group_by('filter')

# 聚合群組
means = g.groups.aggregate(np.mean)

# 遍歷群組
for group in g.groups:
    print(f"Filter: {group['filter'][0]}")
    print(f"Mean mag: {np.mean(group['mag'])}")
```

### 唯一列

```python
# 取得唯一列
t_unique = t.unique('id')

# 多個欄位
t_unique = t.unique(['ra', 'dec'])
```

## 單位和物理量

使用 QTable 進行帶單位的操作：

```python
from astropy.table import QTable

# 建立帶單位的表格
t = QTable()
t['flux'] = [1.2, 2.3, 3.4] * u.Jy
t['wavelength'] = [500, 600, 700] * u.nm

# 單位轉換
t['flux'].to(u.mJy)
t['wavelength'].to(u.angstrom)

# 計算保留單位
t['freq'] = t['wavelength'].to(u.Hz, equivalencies=u.spectral())
```

## 遮罩遺失資料

```python
from astropy.table import MaskedColumn
import numpy as np

# 建立遮罩欄位
flux = MaskedColumn([1.2, np.nan, 3.4], mask=[False, True, False])
t = Table([flux], names=['flux'])

# 操作自動處理遮罩
mean_flux = np.ma.mean(t['flux'])

# 填充遮罩值
t['flux'].filled(0)  # 將遮罩替換為 0
```

## 快速查詢的索引

建立索引以快速擷取列：

```python
# 在欄位上添加索引
t.add_index('id')

# 按索引快速查詢
row = t.loc[12345]  # 找出 id=12345 的列

# 範圍查詢
subset = t.loc[100:200]
```

## 表格中繼資料

```python
# 設定表格層級中繼資料
t.meta['TELESCOPE'] = 'HST'
t.meta['FILTER'] = 'F814W'
t.meta['EXPTIME'] = 300.0

# 設定欄位層級中繼資料
t['ra'].meta['unit'] = 'deg'
t['ra'].meta['description'] = 'Right Ascension'
t['ra'].description = 'Right Ascension'  # 捷徑
```

## 效能技巧

### 快速表格建構

```python
# 慢：逐一添加列
t = Table(names=['a', 'b'])
for i in range(1000):
    t.add_row([i, i**2])

# 快：從列表建立
rows = [(i, i**2) for i in range(1000)]
t = Table(rows=rows, names=['a', 'b'])
```

### 記憶體映射 FITS 表格

```python
# 不將整個表格載入記憶體
t = Table.read('huge_catalog.fits', memmap=True)

# 只在存取時載入資料
subset = t[10000:10100]  # 高效
```

### 複製 vs 視圖

```python
# 建立視圖（共享資料，快速）
t_view = t['ra', 'dec']

# 建立複製（獨立資料）
t_copy = t['ra', 'dec'].copy()
```

## 顯示表格

```python
# 印出到主控台
print(t)

# 在互動瀏覽器中顯示
t.show_in_browser()
t.show_in_browser(jsviewer=True)  # 互動排序/過濾

# 分頁檢視
t.more()

# 自訂格式化
t['flux'].format = '%.3f'
t['ra'].format = '{:.6f}'
```

## 轉換為其他格式

```python
# 到 NumPy 陣列
arr = np.array(t)

# 到 Pandas DataFrame
df = t.to_pandas()

# 到字典
d = {name: t[name] for name in t.colnames}
```

## 常見使用案例

### 交叉匹配星表

```python
from astropy.coordinates import SkyCoord, match_coordinates_sky

# 從表格欄位建立座標物件
coords1 = SkyCoord(t1['ra'], t1['dec'], unit='deg')
coords2 = SkyCoord(t2['ra'], t2['dec'], unit='deg')

# 尋找匹配
idx, sep, _ = coords1.match_to_catalog_sky(coords2)

# 按分離過濾
max_sep = 1 * u.arcsec
matches = sep < max_sep
t1_matched = t1[matches]
t2_matched = t2[idx[matches]]
```

### 資料分箱

```python
from astropy.table import Table
import numpy as np

# 按星等分箱
mag_bins = np.arange(10, 20, 0.5)
binned = t.group_by(np.digitize(t['mag'], mag_bins))
counts = binned.groups.aggregate(len)
```
