# FITS 檔案處理（astropy.io.fits）

`astropy.io.fits` 模組提供讀取、寫入和操作 FITS（Flexible Image Transport System）檔案的綜合工具。

## 開啟 FITS 檔案

### 基本檔案開啟

```python
from astropy.io import fits

# 開啟檔案（回傳 HDUList - HDU 列表）
hdul = fits.open('filename.fits')

# 使用完畢後始終關閉
hdul.close()

# 更好的方式：使用上下文管理器（自動關閉）
with fits.open('filename.fits') as hdul:
    hdul.info()  # 顯示檔案結構
    data = hdul[0].data
```

### 檔案開啟模式

```python
fits.open('file.fits', mode='readonly')   # 唯讀（預設）
fits.open('file.fits', mode='update')     # 讀寫
fits.open('file.fits', mode='append')     # 向檔案添加 HDU
```

### 記憶體映射

對於大型檔案，使用記憶體映射（預設行為）：

```python
hdul = fits.open('large_file.fits', memmap=True)
# 只在需要時載入資料區塊
```

### 遠端檔案

存取雲端託管的 FITS 檔案：

```python
uri = "s3://bucket-name/image.fits"
with fits.open(uri, use_fsspec=True, fsspec_kwargs={"anon": True}) as hdul:
    # 使用 .section 獲取裁切區而不下載整個檔案
    cutout = hdul[1].section[100:200, 100:200]
```

## HDU 結構

FITS 檔案包含標頭資料單元（HDU）：
- **Primary HDU**（`hdul[0]`）：第一個 HDU，始終存在
- **Extension HDU**（`hdul[1:]`）：影像或表格擴展

```python
hdul.info()  # 顯示所有 HDU
# 輸出：
# No.    Name      Ver    Type      Cards   Dimensions   Format
#  0  PRIMARY       1 PrimaryHDU     220   ()
#  1  SCI           1 ImageHDU       140   (1014, 1014)   float32
#  2  ERR           1 ImageHDU        51   (1014, 1014)   float32
```

## 存取 HDU

```python
# 按索引
primary = hdul[0]
extension1 = hdul[1]

# 按名稱
sci = hdul['SCI']

# 按名稱和版本號
sci2 = hdul['SCI', 2]  # 第二個 SCI 擴展
```

## 處理標頭

### 讀取標頭值

```python
hdu = hdul[0]
header = hdu.header

# 取得關鍵字值（不區分大小寫）
observer = header['OBSERVER']
exptime = header['EXPTIME']

# 如果缺失則取得預設值
filter_name = header.get('FILTER', 'Unknown')

# 按索引存取
value = header[7]  # 第 8 個卡片的值
```

### 修改標頭

```python
# 更新現有關鍵字
header['OBSERVER'] = 'Edwin Hubble'

# 添加/更新並附帶註解
header['OBSERVER'] = ('Edwin Hubble', 'Name of observer')

# 在特定位置添加關鍵字
header.insert(5, ('NEWKEY', 'value', 'comment'))

# 添加 HISTORY 和 COMMENT
header['HISTORY'] = 'File processed on 2025-01-15'
header['COMMENT'] = 'Note about the data'

# 刪除關鍵字
del header['OLDKEY']
```

### 標頭卡片

每個關鍵字儲存為一個「卡片」（80 字元記錄）：

```python
# 存取完整卡片
card = header.cards[0]
print(f"{card.keyword} = {card.value} / {card.comment}")

# 遍歷所有卡片
for card in header.cards:
    print(f"{card.keyword}: {card.value}")
```

## 處理影像資料

### 讀取影像資料

```python
# 從 HDU 取得資料
data = hdul[1].data  # 回傳 NumPy 陣列

# 資料屬性
print(data.shape)      # 例如 (1024, 1024)
print(data.dtype)      # 例如 float32
print(data.min(), data.max())

# 存取特定像素
pixel_value = data[100, 200]
region = data[100:200, 300:400]
```

### 資料操作

資料是 NumPy 陣列，因此使用標準 NumPy 操作：

```python
import numpy as np

# 統計
mean = np.mean(data)
median = np.median(data)
std = np.std(data)

# 修改資料
data[data < 0] = 0  # 裁剪負值
data = data * gain + bias  # 校準

# 數學運算
log_data = np.log10(data)
smoothed = scipy.ndimage.gaussian_filter(data, sigma=2)
```

### 裁切和區段

擷取區域而不載入整個陣列：

```python
# 區段表示法 [y_start:y_end, x_start:x_end]
cutout = hdul[1].section[500:600, 700:800]
```

## 建立新的 FITS 檔案

### 簡單影像檔案

```python
# 建立資料
data = np.random.random((100, 100))

# 建立 HDU
hdu = fits.PrimaryHDU(data=data)

# 添加標頭關鍵字
hdu.header['OBJECT'] = 'Test Image'
hdu.header['EXPTIME'] = 300.0

# 寫入檔案
hdu.writeto('new_image.fits')

# 如果存在則覆寫
hdu.writeto('new_image.fits', overwrite=True)
```

### 多擴展檔案

```python
# 建立 Primary HDU（可以沒有資料）
primary = fits.PrimaryHDU()
primary.header['TELESCOP'] = 'HST'

# 建立影像擴展
sci_data = np.ones((100, 100))
sci = fits.ImageHDU(data=sci_data, name='SCI')

err_data = np.ones((100, 100)) * 0.1
err = fits.ImageHDU(data=err_data, name='ERR')

# 合併為 HDUList
hdul = fits.HDUList([primary, sci, err])

# 寫入檔案
hdul.writeto('multi_extension.fits')
```

## 處理表格資料

### 讀取表格

```python
# 開啟表格
with fits.open('table.fits') as hdul:
    table = hdul[1].data  # BinTableHDU 或 TableHDU

    # 存取欄位
    ra = table['RA']
    dec = table['DEC']
    mag = table['MAG']

    # 存取列
    first_row = table[0]
    subset = table[10:20]

    # 欄位資訊
    cols = hdul[1].columns
    print(cols.names)
    cols.info()
```

### 建立表格

```python
# 定義欄位
col1 = fits.Column(name='ID', format='K', array=[1, 2, 3, 4])
col2 = fits.Column(name='RA', format='D', array=[10.5, 11.2, 12.3, 13.1])
col3 = fits.Column(name='DEC', format='D', array=[41.2, 42.1, 43.5, 44.2])
col4 = fits.Column(name='Name', format='20A',
                   array=['Star1', 'Star2', 'Star3', 'Star4'])

# 建立表格 HDU
table_hdu = fits.BinTableHDU.from_columns([col1, col2, col3, col4])
table_hdu.name = 'CATALOG'

# 寫入檔案
table_hdu.writeto('catalog.fits', overwrite=True)
```

### 欄位格式

常見 FITS 表格欄位格式：
- `'A'`：字元字串（例如 '20A' 表示 20 個字元）
- `'L'`：邏輯（布林值）
- `'B'`：無符號位元組
- `'I'`：16 位元整數
- `'J'`：32 位元整數
- `'K'`：64 位元整數
- `'E'`：32 位元浮點數
- `'D'`：64 位元浮點數

## 修改現有檔案

### 更新模式

```python
with fits.open('file.fits', mode='update') as hdul:
    # 修改標頭
    hdul[0].header['NEWKEY'] = 'value'

    # 修改資料
    hdul[1].data[100, 100] = 999

    # 上下文結束時自動儲存變更
```

### 附加模式

```python
# 向現有檔案添加新擴展
new_data = np.random.random((50, 50))
new_hdu = fits.ImageHDU(data=new_data, name='NEW_EXT')

with fits.open('file.fits', mode='append') as hdul:
    hdul.append(new_hdu)
```

## 便利函數

用於快速操作而不管理 HDU 列表：

```python
# 只取得資料
data = fits.getdata('file.fits', ext=1)

# 只取得標頭
header = fits.getheader('file.fits', ext=0)

# 取得兩者
data, header = fits.getdata('file.fits', ext=1, header=True)

# 取得單一關鍵字值
exptime = fits.getval('file.fits', 'EXPTIME', ext=0)

# 設定關鍵字值
fits.setval('file.fits', 'NEWKEY', value='newvalue', ext=0)

# 寫入簡單檔案
fits.writeto('output.fits', data, header, overwrite=True)

# 附加到檔案
fits.append('file.fits', data, header)

# 顯示檔案資訊
fits.info('file.fits')
```

## 比較 FITS 檔案

```python
# 印出兩個檔案之間的差異
fits.printdiff('file1.fits', 'file2.fits')

# 程式化比較
diff = fits.FITSDiff('file1.fits', 'file2.fits')
print(diff.report())
```

## 格式轉換

### FITS 與 Astropy Table 互轉

```python
from astropy.table import Table

# FITS 到 Table
table = Table.read('catalog.fits')

# Table 到 FITS
table.write('output.fits', format='fits', overwrite=True)
```

## 最佳實務

1. **始終使用上下文管理器**（`with` 語句）以確保安全的檔案處理
2. **避免修改結構關鍵字**（SIMPLE、BITPIX、NAXIS 等）
3. **對大型檔案使用記憶體映射**以節省 RAM
4. **對遠端檔案使用 .section**以避免完整下載
5. **在存取資料前使用 `.info()` 檢查 HDU 結構**
6. **在操作前驗證資料類型**以避免意外行為
7. **對簡單的一次性操作使用便利函數**

## 常見問題

### 處理非標準 FITS

某些檔案違反 FITS 標準：

```python
# 忽略驗證警告
hdul = fits.open('bad_file.fits', ignore_missing_end=True)

# 修復非標準檔案
hdul = fits.open('bad_file.fits')
hdul.verify('fix')  # 嘗試修復問題
hdul.writeto('fixed_file.fits')
```

### 大型檔案效能

```python
# 使用記憶體映射（預設）
hdul = fits.open('huge_file.fits', memmap=True)

# 對於具有大型陣列的寫入操作，使用 Dask
import dask.array as da
large_array = da.random.random((10000, 10000))
fits.writeto('output.fits', large_array)
```
