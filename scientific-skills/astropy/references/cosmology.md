# 宇宙學計算（astropy.cosmology）

`astropy.cosmology` 子套件提供基於各種宇宙學模型進行宇宙學計算的工具。

## 使用內建宇宙學模型

基於 WMAP 和 Planck 觀測的預載宇宙學模型：

```python
from astropy.cosmology import Planck18, Planck15, Planck13
from astropy.cosmology import WMAP9, WMAP7, WMAP5
from astropy import units as u

# 使用 Planck 2018 宇宙學模型
cosmo = Planck18

# 計算到 z=4 的距離
d = cosmo.luminosity_distance(4)
print(f"z=4 處的光度距離: {d}")

# z=0 時的宇宙年齡
age = cosmo.age(0)
print(f"目前宇宙年齡: {age.to(u.Gyr)}")
```

## 建立自訂宇宙學模型

### FlatLambdaCDM（最常用）

具有宇宙學常數的平坦宇宙：

```python
from astropy.cosmology import FlatLambdaCDM

# 定義宇宙學模型
cosmo = FlatLambdaCDM(
    H0=70 * u.km / u.s / u.Mpc,  # z=0 時的 Hubble 常數
    Om0=0.3,                      # z=0 時的物質密度參數
    Tcmb0=2.725 * u.K             # CMB 溫度（可選）
)
```

### LambdaCDM（非平坦）

具有宇宙學常數的非平坦宇宙：

```python
from astropy.cosmology import LambdaCDM

cosmo = LambdaCDM(
    H0=70 * u.km / u.s / u.Mpc,
    Om0=0.3,
    Ode0=0.7  # 暗能量密度參數
)
```

### wCDM 和 w0wzCDM

具有狀態方程式參數的暗能量：

```python
from astropy.cosmology import FlatwCDM, w0wzCDM

# 常數 w
cosmo_w = FlatwCDM(H0=70 * u.km/u.s/u.Mpc, Om0=0.3, w0=-0.9)

# 演化 w(z) = w0 + wz * z
cosmo_wz = w0wzCDM(H0=70 * u.km/u.s/u.Mpc, Om0=0.3, Ode0=0.7,
                   w0=-1.0, wz=0.1)
```

## 距離計算

### 共動距離

視線方向的共動距離：

```python
d_c = cosmo.comoving_distance(z)
```

### 光度距離

用於從觀測流量計算光度的距離：

```python
d_L = cosmo.luminosity_distance(z)

# 從視星等計算絕對星等
M = m - 5*np.log10(d_L.to(u.pc).value) + 5
```

### 角直徑距離

用於從角大小計算物理大小的距離：

```python
d_A = cosmo.angular_diameter_distance(z)

# 從角大小計算物理大小
theta = 10 * u.arcsec  # 角大小
physical_size = d_A * theta.to(u.radian).value
```

### 共動橫向距離

共動橫向距離（在平坦宇宙中等於共動距離）：

```python
d_M = cosmo.comoving_transverse_distance(z)
```

### 距離模數

```python
dm = cosmo.distmod(z)
# 關聯視星等和絕對星等：m - M = dm
```

## 尺度計算

### 每角分的 kpc

給定紅移處的物理尺度：

```python
scale = cosmo.kpc_proper_per_arcmin(z)
# 例如「z=1 時每角分 50 kpc」
```

### 共動體積

用於巡天體積計算的體積元：

```python
vol = cosmo.comoving_volume(z)  # 到紅移 z 的總體積
vol_element = cosmo.differential_comoving_volume(z)  # dV/dz
```

## 時間計算

### 宇宙年齡

給定紅移處的年齡：

```python
age = cosmo.age(z)
age_now = cosmo.age(0)  # 目前年齡
age_at_z1 = cosmo.age(1)  # z=1 時的年齡
```

### 回溯時間

自光子發射以來的時間：

```python
t_lookback = cosmo.lookback_time(z)
# z 到 z=0 之間的時間
```

## Hubble 參數

Hubble 參數隨紅移的函數：

```python
H_z = cosmo.H(z)  # H(z)，單位 km/s/Mpc
E_z = cosmo.efunc(z)  # E(z) = H(z)/H0
```

## 密度參數

密度參數隨紅移的演化：

```python
Om_z = cosmo.Om(z)        # z 處的物質密度
Ode_z = cosmo.Ode(z)      # z 處的暗能量密度
Ok_z = cosmo.Ok(z)        # z 處的曲率密度
Ogamma_z = cosmo.Ogamma(z)  # z 處的光子密度
Onu_z = cosmo.Onu(z)      # z 處的微中子密度
```

## 臨界密度和特徵密度

```python
rho_c = cosmo.critical_density(z)  # z 處的臨界密度
rho_m = cosmo.critical_density(z) * cosmo.Om(z)  # 物質密度
```

## 反向計算

找出對應特定值的紅移：

```python
from astropy.cosmology import z_at_value

# 找出特定回溯時間的 z
z = z_at_value(cosmo.lookback_time, 10*u.Gyr)

# 找出特定光度距離的 z
z = z_at_value(cosmo.luminosity_distance, 1000*u.Mpc)

# 找出特定年齡的 z
z = z_at_value(cosmo.age, 1*u.Gyr)
```

## 陣列操作

所有方法都接受陣列輸入：

```python
import numpy as np

z_array = np.linspace(0, 5, 100)
d_L_array = cosmo.luminosity_distance(z_array)
H_array = cosmo.H(z_array)
age_array = cosmo.age(z_array)
```

## 微中子效應

包含具有質量的微中子：

```python
from astropy.cosmology import FlatLambdaCDM

# 帶有質量微中子
cosmo = FlatLambdaCDM(
    H0=70 * u.km/u.s/u.Mpc,
    Om0=0.3,
    Tcmb0=2.725 * u.K,
    Neff=3.04,  # 有效微中子種類數
    m_nu=[0., 0., 0.06] * u.eV  # 微中子質量
)
```

注意：具有質量的微中子會使效能降低 3-4 倍，但提供更準確的結果。

## 複製和修改宇宙學模型

宇宙學物件是不可變的。建立修改過的副本：

```python
# 使用不同的 H0 複製
cosmo_new = cosmo.clone(H0=72 * u.km/u.s/u.Mpc)

# 使用修改後的名稱複製
cosmo_named = cosmo.clone(name="My Custom Cosmology")
```

## 常見使用案例

### 計算絕對星等

```python
# 從視星等和紅移
z = 1.5
m_app = 24.5  # 視星等
d_L = cosmo.luminosity_distance(z)
M_abs = m_app - cosmo.distmod(z).value
```

### 巡天體積計算

```python
# 兩個紅移之間的體積
z_min, z_max = 0.5, 1.5
volume = cosmo.comoving_volume(z_max) - cosmo.comoving_volume(z_min)

# 轉換為 Gpc^3
volume_gpc3 = volume.to(u.Gpc**3)
```

### 從角大小計算物理大小

```python
theta = 1 * u.arcsec  # 角大小
z = 2.0
d_A = cosmo.angular_diameter_distance(z)
size_kpc = (d_A * theta.to(u.radian)).to(u.kpc)
```

### 自大霹靂以來的時間

```python
# 特定紅移處的年齡
z_formation = 6
age_at_formation = cosmo.age(z_formation)
time_since_formation = cosmo.age(0) - age_at_formation
```

## 宇宙學模型比較

```python
# 比較不同模型
from astropy.cosmology import Planck18, WMAP9

z = 1.0
print(f"Planck18 d_L: {Planck18.luminosity_distance(z)}")
print(f"WMAP9 d_L: {WMAP9.luminosity_distance(z)}")
```

## 效能考量

- 對大多數用途計算速度很快
- 具有質量的微中子會顯著降低速度
- 陣列操作是向量化且高效的
- 結果在 z < 5000-6000 範圍內有效（取決於模型）
