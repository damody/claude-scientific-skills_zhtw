# 時間處理（astropy.time）

`astropy.time` 模組提供操作時間和日期的穩健工具，支援各種時間尺度和格式。

## 建立 Time 物件

### 基本建立

```python
from astropy.time import Time
import astropy.units as u

# ISO 格式（自動檢測）
t = Time('2023-01-15 12:30:45')
t = Time('2023-01-15T12:30:45')

# 明確指定格式
t = Time('2023-01-15 12:30:45', format='iso', scale='utc')

# 儒略日
t = Time(2460000.0, format='jd')

# 修正儒略日
t = Time(59945.0, format='mjd')

# Unix 時間（自 1970-01-01 以來的秒數）
t = Time(1673785845.0, format='unix')
```

### 時間陣列

```python
# 多個時間
times = Time(['2023-01-01', '2023-06-01', '2023-12-31'])

# 從陣列
import numpy as np
jd_array = np.linspace(2460000, 2460100, 100)
times = Time(jd_array, format='jd')
```

## 時間格式

### 支援的格式

```python
# ISO 8601
t = Time('2023-01-15 12:30:45', format='iso')
t = Time('2023-01-15T12:30:45.123', format='isot')

# 儒略日
t = Time(2460000.0, format='jd')          # 儒略日
t = Time(59945.0, format='mjd')           # 修正儒略日

# 十進制年
t = Time(2023.5, format='decimalyear')
t = Time(2023.5, format='jyear')          # 儒略年
t = Time(2023.5, format='byear')          # 貝塞爾年

# 年和日序
t = Time('2023:046', format='yday')       # 2023 年第 46 天

# FITS 格式
t = Time('2023-01-15T12:30:45', format='fits')

# GPS 秒
t = Time(1000000000.0, format='gps')

# Unix 時間
t = Time(1673785845.0, format='unix')

# Matplotlib 日期
t = Time(738521.0, format='plot_date')

# datetime 物件
from datetime import datetime
dt = datetime(2023, 1, 15, 12, 30, 45)
t = Time(dt)
```

## 時間尺度

### 可用的時間尺度

```python
# UTC - 協調世界時（預設）
t = Time('2023-01-15 12:00:00', scale='utc')

# TAI - 國際原子時
t = Time('2023-01-15 12:00:00', scale='tai')

# TT - 地球時
t = Time('2023-01-15 12:00:00', scale='tt')

# TDB - 質心動力學時
t = Time('2023-01-15 12:00:00', scale='tdb')

# TCG - 地心座標時
t = Time('2023-01-15 12:00:00', scale='tcg')

# TCB - 質心座標時
t = Time('2023-01-15 12:00:00', scale='tcb')

# UT1 - 世界時
t = Time('2023-01-15 12:00:00', scale='ut1')
```

### 轉換時間尺度

```python
t = Time('2023-01-15 12:00:00', scale='utc')

# 轉換到不同尺度
t_tai = t.tai
t_tt = t.tt
t_tdb = t.tdb
t_ut1 = t.ut1

# 檢查偏移量
print(f"TAI - UTC = {(t.tai - t.utc).sec} 秒")
# TAI - UTC = 37 秒（閏秒）
```

## 格式轉換

### 變更輸出格式

```python
t = Time('2023-01-15 12:30:45')

# 以不同格式存取
print(t.jd)           # 儒略日
print(t.mjd)          # 修正儒略日
print(t.iso)          # ISO 格式
print(t.isot)         # 帶 'T' 的 ISO
print(t.unix)         # Unix 時間
print(t.decimalyear)  # 十進制年

# 變更預設格式
t.format = 'mjd'
print(t)  # 顯示為 MJD
```

### 高精度輸出

```python
# 使用 subfmt 進行精度控制
t.to_value('mjd', subfmt='float')    # 標準浮點數
t.to_value('mjd', subfmt='long')     # 延伸精度
t.to_value('mjd', subfmt='decimal')  # 十進制（最高精度）
t.to_value('mjd', subfmt='str')      # 字串表示
```

## 時間運算

### TimeDelta 物件

```python
from astropy.time import TimeDelta

# 建立時間差
dt = TimeDelta(1.0, format='jd')      # 1 天
dt = TimeDelta(3600.0, format='sec')  # 1 小時

# 相減時間
t1 = Time('2023-01-15')
t2 = Time('2023-02-15')
dt = t2 - t1
print(dt.jd)   # 31 天
print(dt.sec)  # 2678400 秒
```

### 加減時間

```python
t = Time('2023-01-15 12:00:00')

# 加 TimeDelta
t_future = t + TimeDelta(7, format='jd')  # 加 7 天

# 加 Quantity
t_future = t + 1*u.hour
t_future = t + 30*u.day
t_future = t + 1*u.year

# 減
t_past = t - 1*u.week
```

### 時間範圍

```python
# 建立時間範圍
start = Time('2023-01-01')
end = Time('2023-12-31')
times = start + np.linspace(0, 365, 100) * u.day

# 或使用 TimeDelta
times = start + TimeDelta(np.linspace(0, 365, 100), format='jd')
```

## 觀測相關功能

### 恆星時

```python
from astropy.coordinates import EarthLocation

# 定義觀測者位置
location = EarthLocation(lat=40*u.deg, lon=-120*u.deg, height=1000*u.m)

# 建立帶位置的時間
t = Time('2023-06-15 23:00:00', location=location)

# 計算恆星時
lst_apparent = t.sidereal_time('apparent')
lst_mean = t.sidereal_time('mean')

print(f"本地恆星時: {lst_apparent}")
```

### 光行時間校正

```python
from astropy.coordinates import SkyCoord, EarthLocation

# 定義目標和觀測者
target = SkyCoord(ra=10*u.deg, dec=20*u.deg)
location = EarthLocation.of_site('Keck Observatory')

# 觀測時間
times = Time(['2023-01-01', '2023-06-01', '2023-12-31'],
             location=location)

# 計算到太陽系質心的光行時間
ltt_bary = times.light_travel_time(target, kind='barycentric')
ltt_helio = times.light_travel_time(target, kind='heliocentric')

# 應用校正
times_barycentric = times.tdb + ltt_bary
```

### 地球自轉角

```python
# 地球自轉角（用於天球到地球座標轉換）
era = t.earth_rotation_angle()
```

## 處理遺失或無效時間

### 遮罩時間

```python
import numpy as np

# 建立帶遺失值的時間
times = Time(['2023-01-01', '2023-06-01', '2023-12-31'])
times[1] = np.ma.masked  # 標記為遺失

# 檢查遮罩
print(times.mask)  # [False True False]

# 取得未遮罩版本
times_clean = times.unmasked

# 填充遮罩值
times_filled = times.filled(Time('2000-01-01'))
```

## 時間精度和表示

### 內部表示

Time 物件使用兩個 64 位元浮點數（jd1, jd2）以獲得高精度：

```python
t = Time('2023-01-15 12:30:45.123456789', format='iso', scale='utc')

# 存取內部表示
print(t.jd1, t.jd2)  # 整數和小數部分

# 這允許在天文時間尺度上達到亞奈秒精度
```

### 精度

```python
# 長時間間隔的高精度
t1 = Time('1900-01-01')
t2 = Time('2100-01-01')
dt = t2 - t1
print(f"時間跨度: {dt.sec / (365.25 * 86400)} 年")
# 全程保持精度
```

## 時間格式化

### 自訂字串格式

```python
t = Time('2023-01-15 12:30:45')

# Strftime 風格格式化
t.strftime('%Y-%m-%d %H:%M:%S')  # '2023-01-15 12:30:45'
t.strftime('%B %d, %Y')          # 'January 15, 2023'

# ISO 格式子格式
t.iso                    # '2023-01-15 12:30:45.000'
t.isot                   # '2023-01-15T12:30:45.000'
t.to_value('iso', subfmt='date_hms')  # '2023-01-15 12:30:45.000'
```

## 常見使用案例

### 格式間轉換

```python
# MJD 到 ISO
t_mjd = Time(59945.0, format='mjd')
iso_string = t_mjd.iso

# ISO 到 JD
t_iso = Time('2023-01-15 12:00:00')
jd_value = t_iso.jd

# Unix 到 ISO
t_unix = Time(1673785845.0, format='unix')
iso_string = t_unix.iso
```

### 各種單位的時間差

```python
t1 = Time('2023-01-01')
t2 = Time('2023-12-31')

dt = t2 - t1
print(f"天: {dt.to(u.day)}")
print(f"小時: {dt.to(u.hour)}")
print(f"秒: {dt.sec}")
print(f"年: {dt.to(u.year)}")
```

### 建立規則時間序列

```python
# 一年的每日觀測
start = Time('2023-01-01')
times = start + np.arange(365) * u.day

# 一天的每小時觀測
start = Time('2023-01-15 00:00:00')
times = start + np.arange(24) * u.hour

# 每 30 秒的觀測
start = Time('2023-01-15 12:00:00')
times = start + np.arange(1000) * 30 * u.second
```

### 時區處理

```python
# UTC 到本地時間（需要 datetime）
t = Time('2023-01-15 12:00:00', scale='utc')
dt_utc = t.to_datetime()

# 使用 pytz 轉換到特定時區
import pytz
eastern = pytz.timezone('US/Eastern')
dt_eastern = dt_utc.replace(tzinfo=pytz.utc).astimezone(eastern)
```

### 質心校正範例

```python
from astropy.coordinates import SkyCoord, EarthLocation

# 目標座標
target = SkyCoord(ra='23h23m08.55s', dec='+18d24m59.3s')

# 天文台位置
location = EarthLocation.of_site('Keck Observatory')

# 觀測時間（必須包含位置）
times = Time(['2023-01-15 08:30:00', '2023-01-16 08:30:00'],
             location=location)

# 計算質心校正
ltt_bary = times.light_travel_time(target, kind='barycentric')

# 應用校正以取得質心時間
times_bary = times.tdb + ltt_bary

# 對於視向速度校正
rv_correction = ltt_bary.to(u.km, equivalencies=u.dimensionless_angles())
```

## 效能考量

1. **陣列操作速度快**：將多個時間作為陣列處理
2. **格式轉換會快取**：重複存取是高效的
3. **尺度轉換可能需要 IERS 資料**：會自動下載
4. **保持高精度**：在天文時間尺度上達到亞奈秒準確度
