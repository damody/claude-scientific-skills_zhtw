# 單位和物理量（astropy.units）

`astropy.units` 模組處理物理量的定義、轉換和運算。

## 建立物理量

將數值乘以或除以內建單位來建立 Quantity 物件：

```python
from astropy import units as u
import numpy as np

# 純量物理量
distance = 42.0 * u.meter
velocity = 100 * u.km / u.s

# 陣列物理量
distances = np.array([1., 2., 3.]) * u.m
wavelengths = [500, 600, 700] * u.nm
```

透過 `.value` 和 `.unit` 屬性存取分量：
```python
distance.value  # 42.0
distance.unit   # Unit("m")
```

## 單位轉換

使用 `.to()` 方法進行轉換：

```python
distance = 1.0 * u.parsec
distance.to(u.km)  # <Quantity 30856775814671.914 km>

wavelength = 500 * u.nm
wavelength.to(u.angstrom)  # <Quantity 5000. Angstrom>
```

## 算術運算

物理量支援標準算術運算並自動管理單位：

```python
# 基本運算
speed = 15.1 * u.meter / (32.0 * u.second)  # <Quantity 0.471875 m / s>
area = (5 * u.m) * (3 * u.m)  # <Quantity 15. m2>

# 適當時單位會消去
ratio = (10 * u.m) / (5 * u.m)  # <Quantity 2. (dimensionless)>

# 分解複雜單位
time = (3.0 * u.kilometer / (130.51 * u.meter / u.second))
time.decompose()  # <Quantity 22.986744310780782 s>
```

## 單位系統

在主要單位系統間轉換：

```python
# SI 到 CGS
pressure = 1.0 * u.Pa
pressure.cgs  # <Quantity 10. Ba>

# 尋找等效表示
(u.s ** -1).compose()  # [Unit("Bq"), Unit("Hz"), ...]
```

## 等效關係

領域特定的轉換需要等效關係：

```python
# 光譜等效（波長 ↔ 頻率）
wavelength = 1000 * u.nm
wavelength.to(u.Hz, equivalencies=u.spectral())
# <Quantity 2.99792458e+14 Hz>

# 都卜勒等效
velocity = 1000 * u.km / u.s
velocity.to(u.Hz, equivalencies=u.doppler_optical(500*u.nm))

# 其他等效關係
u.brightness_temperature(500*u.GHz)
u.doppler_radio(1.4*u.GHz)
u.mass_energy()
u.parallax()
```

## 對數單位

用於星等、分貝和 dex 的特殊單位：

```python
# 星等
flux = -2.5 * u.mag(u.ct / u.s)

# 分貝
power_ratio = 3 * u.dB(u.W)

# Dex（以 10 為底的對數）
abundance = 8.5 * u.dex(u.cm**-3)
```

## 常見單位

### 長度
`u.m, u.km, u.cm, u.mm, u.micron, u.angstrom, u.au, u.pc, u.kpc, u.Mpc, u.lyr`

### 時間
`u.s, u.min, u.hour, u.day, u.year, u.Myr, u.Gyr`

### 質量
`u.kg, u.g, u.M_sun, u.M_earth, u.M_jup`

### 溫度
`u.K, u.deg_C`

### 角度
`u.deg, u.arcmin, u.arcsec, u.rad, u.hourangle, u.mas`

### 能量/功率
`u.J, u.erg, u.eV, u.keV, u.MeV, u.GeV, u.W, u.L_sun`

### 頻率
`u.Hz, u.kHz, u.MHz, u.GHz`

### 流量
`u.Jy, u.mJy, u.erg / u.s / u.cm**2`

## 效能優化

為陣列運算預先計算複合單位：

```python
# 慢（建立中間物理量）
result = array * u.m / u.s / u.kg / u.sr

# 快（預先計算的複合單位）
UNIT_COMPOSITE = u.m / u.s / u.kg / u.sr
result = array * UNIT_COMPOSITE

# 最快（使用 << 避免複製）
result = array << UNIT_COMPOSITE  # 快 10000 倍
```

## 字串格式化

使用標準 Python 語法格式化物理量：

```python
velocity = 15.1 * u.meter / (32.0 * u.second)
f"{velocity:0.03f}"     # '0.472 m / s'
f"{velocity:.2e}"       # '4.72e-01 m / s'
f"{velocity.unit:FITS}" # 'm s-1'
```

## 定義自訂單位

```python
# 建立新單位
bakers_fortnight = u.def_unit('bakers_fortnight', 13 * u.day)

# 在字串解析中啟用
u.add_enabled_units([bakers_fortnight])
```

## 常數

存取帶單位的物理常數：

```python
from astropy.constants import c, G, M_sun, h, k_B

speed_of_light = c.to(u.km/u.s)
gravitational_constant = G.to(u.m**3 / u.kg / u.s**2)
```
