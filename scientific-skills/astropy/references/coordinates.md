# 天文座標（astropy.coordinates）

`astropy.coordinates` 套件提供表示天體座標和在不同座標系統間轉換的工具。

## 使用 SkyCoord 建立座標

高階 `SkyCoord` 類別是推薦的介面：

```python
from astropy import units as u
from astropy.coordinates import SkyCoord

# 十進制度數
c = SkyCoord(ra=10.625*u.degree, dec=41.2*u.degree, frame='icrs')

# 六十進制字串
c = SkyCoord(ra='00h42m30s', dec='+41d12m00s', frame='icrs')

# 混合格式
c = SkyCoord('00h42.5m +41d12m', unit=(u.hourangle, u.deg))

# 銀河座標
c = SkyCoord(l=120.5*u.degree, b=-23.4*u.degree, frame='galactic')
```

## 陣列座標

使用陣列高效處理多個座標：

```python
# 建立座標陣列
coords = SkyCoord(ra=[10, 11, 12]*u.degree,
                  dec=[41, -5, 42]*u.degree)

# 存取個別元素
coords[0]
coords[1:3]

# 陣列操作
coords.shape
len(coords)
```

## 存取座標分量

```python
c = SkyCoord(ra=10.68*u.degree, dec=41.27*u.degree, frame='icrs')

# 存取座標
c.ra        # <Longitude 10.68 deg>
c.dec       # <Latitude 41.27 deg>
c.ra.hour   # 轉換為小時
c.ra.hms    # 時、分、秒元組
c.dec.dms   # 度、角分、角秒元組
```

## 字串格式化

```python
c.to_string('decimal')      # '10.68 41.27'
c.to_string('dms')          # '10d40m48s 41d16m12s'
c.to_string('hmsdms')       # '00h42m43.2s +41d16m12s'

# 自訂格式化
c.ra.to_string(unit=u.hour, sep=':', precision=2)
```

## 座標轉換

在參考框架間轉換：

```python
c_icrs = SkyCoord(ra=10.68*u.degree, dec=41.27*u.degree, frame='icrs')

# 簡單轉換（作為屬性）
c_galactic = c_icrs.galactic
c_fk5 = c_icrs.fk5
c_fk4 = c_icrs.fk4

# 明確轉換
c_icrs.transform_to('galactic')
c_icrs.transform_to(FK5(equinox='J1975'))  # 自訂框架參數
```

## 常見座標框架

### 天球框架
- **ICRS**：國際天球參考系統（預設，最常用）
- **FK5**：第五基本星表（預設曆元 J2000.0）
- **FK4**：第四基本星表（較舊，需要指定曆元）
- **GCRS**：地心天球參考系統
- **CIRS**：天球中間參考系統

### 銀河框架
- **Galactic**：IAU 1958 銀河座標
- **Supergalactic**：De Vaucouleurs 超銀河座標
- **Galactocentric**：以銀河中心為基準的 3D 座標

### 地平框架
- **AltAz**：高度-方位角（觀測者相關）
- **HADec**：時角-赤緯

### 黃道框架
- **GeocentricMeanEcliptic**：地心平黃道
- **BarycentricMeanEcliptic**：質心平黃道
- **HeliocentricMeanEcliptic**：日心平黃道

## 觀測者相關轉換

對於高度-方位角座標，需指定觀測時間和位置：

```python
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz

# 定義觀測者位置
observing_location = EarthLocation(lat=40.8*u.deg, lon=-121.5*u.deg, height=1060*u.m)
# 或使用命名天文台
observing_location = EarthLocation.of_site('Apache Point Observatory')

# 定義觀測時間
observing_time = Time('2023-01-15 23:00:00')

# 轉換到地平座標
aa_frame = AltAz(obstime=observing_time, location=observing_location)
aa = c_icrs.transform_to(aa_frame)

print(f"高度: {aa.alt}")
print(f"方位角: {aa.az}")
```

## 處理距離

加入距離資訊以進行 3D 座標操作：

```python
# 帶距離
c = SkyCoord(ra=10*u.degree, dec=9*u.degree, distance=770*u.kpc, frame='icrs')

# 存取 3D 笛卡爾座標
c.cartesian.x
c.cartesian.y
c.cartesian.z

# 到原點的距離
c.distance

# 3D 分離
c1 = SkyCoord(ra=10*u.degree, dec=9*u.degree, distance=10*u.pc)
c2 = SkyCoord(ra=11*u.degree, dec=10*u.degree, distance=11.5*u.pc)
sep_3d = c1.separation_3d(c2)  # 3D 距離
```

## 角距離

計算天球上的分離：

```python
c1 = SkyCoord(ra=10*u.degree, dec=9*u.degree, frame='icrs')
c2 = SkyCoord(ra=11*u.degree, dec=10*u.degree, frame='fk5')

# 角距離（自動處理框架轉換）
sep = c1.separation(c2)
print(f"分離: {sep.arcsec} 角秒")

# 位置角
pa = c1.position_angle(c2)
```

## 星表匹配

將座標與星表來源匹配：

```python
# 單目標匹配
catalog = SkyCoord(ra=ra_array*u.degree, dec=dec_array*u.degree)
target = SkyCoord(ra=10.5*u.degree, dec=41.2*u.degree)

# 尋找最近匹配
idx, sep2d, dist3d = target.match_to_catalog_sky(catalog)
matched_coord = catalog[idx]

# 帶最大分離限制的匹配
matches = target.separation(catalog) < 1*u.arcsec
```

## 命名天體

從線上星表擷取座標：

```python
# 按名稱查詢（需要網路）
m31 = SkyCoord.from_name("M31")
crab = SkyCoord.from_name("Crab Nebula")
psr = SkyCoord.from_name("PSR J1012+5307")
```

## 地球位置

定義觀測者位置：

```python
# 按座標
location = EarthLocation(lat=40*u.deg, lon=-120*u.deg, height=1000*u.m)

# 按命名天文台
keck = EarthLocation.of_site('Keck Observatory')
vlt = EarthLocation.of_site('Paranal Observatory')

# 按地址（需要網路）
location = EarthLocation.of_address('1002 Holy Grail Court, St. Louis, MO')

# 列出可用天文台
EarthLocation.get_site_names()
```

## 速度資訊

包含自行和視向速度：

```python
# 自行
c = SkyCoord(ra=10*u.degree, dec=41*u.degree,
             pm_ra_cosdec=15*u.mas/u.yr,
             pm_dec=5*u.mas/u.yr,
             distance=150*u.pc)

# 視向速度
c = SkyCoord(ra=10*u.degree, dec=41*u.degree,
             radial_velocity=20*u.km/u.s)

# 兩者都有
c = SkyCoord(ra=10*u.degree, dec=41*u.degree, distance=150*u.pc,
             pm_ra_cosdec=15*u.mas/u.yr, pm_dec=5*u.mas/u.yr,
             radial_velocity=20*u.km/u.s)
```

## 表示類型

在座標表示間切換：

```python
# 笛卡爾表示
c = SkyCoord(x=1*u.kpc, y=2*u.kpc, z=3*u.kpc,
             representation_type='cartesian', frame='icrs')

# 變更表示
c.representation_type = 'cylindrical'
c.rho  # 柱面半徑
c.phi  # 方位角
c.z    # 高度

# 球面（大多數框架的預設）
c.representation_type = 'spherical'
```

## 效能技巧

1. **使用陣列，而非迴圈**：將多個座標作為單一陣列處理
2. **預先計算框架**：為多次轉換重複使用框架物件
3. **使用廣播**：高效地在多個時間點轉換多個位置
4. **啟用內插**：對於密集時間取樣，使用 ErfaAstromInterpolator

```python
# 快速方法
coords = SkyCoord(ra=ra_array*u.degree, dec=dec_array*u.degree)
coords_transformed = coords.transform_to('galactic')

# 慢速方法（避免使用）
for ra, dec in zip(ra_array, dec_array):
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
    c_transformed = c.transform_to('galactic')
```
