# WCS 和其他 Astropy 模組

## 世界座標系統（astropy.wcs）

WCS 模組管理影像中像素座標和世界座標（例如天體座標）之間的轉換。

### 從 FITS 讀取 WCS

```python
from astropy.wcs import WCS
from astropy.io import fits

# 從 FITS 標頭讀取 WCS
with fits.open('image.fits') as hdul:
    wcs = WCS(hdul[0].header)
```

### 像素到世界轉換

```python
# 單一像素到世界座標
world = wcs.pixel_to_world(100, 200)  # 回傳 SkyCoord
print(f"RA: {world.ra}, Dec: {world.dec}")

# 像素陣列
import numpy as np
x_pixels = np.array([100, 200, 300])
y_pixels = np.array([150, 250, 350])
world_coords = wcs.pixel_to_world(x_pixels, y_pixels)
```

### 世界到像素轉換

```python
from astropy.coordinates import SkyCoord
import astropy.units as u

# 單一座標
coord = SkyCoord(ra=10.5*u.degree, dec=41.2*u.degree)
x, y = wcs.world_to_pixel(coord)

# 座標陣列
coords = SkyCoord(ra=[10, 11, 12]*u.degree, dec=[41, 42, 43]*u.degree)
x_pixels, y_pixels = wcs.world_to_pixel(coords)
```

### WCS 資訊

```python
# 印出 WCS 詳情
print(wcs)

# 存取關鍵屬性
print(wcs.wcs.crpix)  # 參考像素
print(wcs.wcs.crval)  # 參考值（世界座標）
print(wcs.wcs.cd)     # CD 矩陣
print(wcs.wcs.ctype)  # 座標類型

# 像素尺度
pixel_scale = wcs.proj_plane_pixel_scales()  # 回傳 Quantity 陣列
```

### 建立 WCS

```python
from astropy.wcs import WCS

# 建立新 WCS
wcs = WCS(naxis=2)
wcs.wcs.crpix = [512.0, 512.0]  # 參考像素
wcs.wcs.crval = [10.5, 41.2]     # 參考像素處的 RA, Dec
wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']  # 投影類型
wcs.wcs.cdelt = [-0.0001, 0.0001]  # 像素尺度（度/像素）
wcs.wcs.cunit = ['deg', 'deg']
```

### 覆蓋範圍和涵蓋區域

```python
# 計算影像覆蓋範圍（角落座標）
footprint = wcs.calc_footprint()
# 回傳每個角落的 [RA, Dec] 陣列
```

## NDData（astropy.nddata）

用於 n 維資料集的容器，包含中繼資料、不確定性和遮罩。

### 建立 NDData

```python
from astropy.nddata import NDData
import numpy as np
import astropy.units as u

# 基本 NDData
data = np.random.random((100, 100))
ndd = NDData(data)

# 帶單位
ndd = NDData(data, unit=u.electron/u.s)

# 帶不確定性
from astropy.nddata import StdDevUncertainty
uncertainty = StdDevUncertainty(np.sqrt(data))
ndd = NDData(data, uncertainty=uncertainty, unit=u.electron/u.s)

# 帶遮罩
mask = data < 0.1  # 遮罩低值
ndd = NDData(data, mask=mask)

# 帶 WCS
from astropy.wcs import WCS
ndd = NDData(data, wcs=wcs)
```

### CCDData 用於 CCD 影像

```python
from astropy.nddata import CCDData

# 建立 CCDData
ccd = CCDData(data, unit=u.adu, meta={'object': 'M31'})

# 從 FITS 讀取
ccd = CCDData.read('image.fits', unit=u.adu)

# 寫入 FITS
ccd.write('output.fits', overwrite=True)
```

## 建模（astropy.modeling）

用於建立和擬合模型到資料的框架。

### 常見模型

```python
from astropy.modeling import models, fitting
import numpy as np

# 1D 高斯
gauss = models.Gaussian1D(amplitude=10, mean=5, stddev=1)
x = np.linspace(0, 10, 100)
y = gauss(x)

# 2D 高斯
gauss_2d = models.Gaussian2D(amplitude=10, x_mean=50, y_mean=50,
                              x_stddev=5, y_stddev=3)

# 多項式
poly = models.Polynomial1D(degree=3)

# 冪律
power_law = models.PowerLaw1D(amplitude=10, x_0=1, alpha=2)
```

### 擬合模型到資料

```python
# 生成雜訊資料
true_model = models.Gaussian1D(amplitude=10, mean=5, stddev=1)
x = np.linspace(0, 10, 100)
y_true = true_model(x)
y_noisy = y_true + np.random.normal(0, 0.5, x.shape)

# 擬合模型
fitter = fitting.LevMarLSQFitter()
initial_model = models.Gaussian1D(amplitude=8, mean=4, stddev=1.5)
fitted_model = fitter(initial_model, x, y_noisy)

print(f"擬合振幅: {fitted_model.amplitude.value}")
print(f"擬合平均值: {fitted_model.mean.value}")
print(f"擬合標準差: {fitted_model.stddev.value}")
```

### 複合模型

```python
# 加法模型
double_gauss = models.Gaussian1D(amp=5, mean=3, stddev=1) + \
               models.Gaussian1D(amp=8, mean=7, stddev=1.5)

# 組合模型
composite = models.Gaussian1D(amp=10, mean=5, stddev=1) | \
            models.Scale(factor=2)  # 縮放輸出
```

## 視覺化（astropy.visualization）

用於視覺化天文影像和資料的工具。

### 影像正規化

```python
from astropy.visualization import simple_norm
import matplotlib.pyplot as plt

# 載入影像
from astropy.io import fits
data = fits.getdata('image.fits')

# 為顯示正規化
norm = simple_norm(data, 'sqrt', percent=99)

# 顯示
plt.imshow(data, norm=norm, cmap='gray', origin='lower')
plt.colorbar()
plt.show()
```

### 拉伸和區間

```python
from astropy.visualization import (MinMaxInterval, AsinhStretch,
                                    ImageNormalize, ZScaleInterval)

# Z-scale 區間
interval = ZScaleInterval()
vmin, vmax = interval.get_limits(data)

# Asinh 拉伸
stretch = AsinhStretch()
norm = ImageNormalize(data, interval=interval, stretch=stretch)

plt.imshow(data, norm=norm, cmap='gray', origin='lower')
```

### PercentileInterval

```python
from astropy.visualization import PercentileInterval

# 顯示第 5 到第 95 百分位數之間的資料
interval = PercentileInterval(90)  # 90% 的資料
vmin, vmax = interval.get_limits(data)

plt.imshow(data, vmin=vmin, vmax=vmax, cmap='gray', origin='lower')
```

## 常數（astropy.constants）

帶單位的物理和天文常數。

```python
from astropy import constants as const

# 光速
c = const.c
print(f"c = {c}")
print(f"c（km/s）= {c.to(u.km/u.s)}")

# 重力常數
G = const.G

# 天文常數
M_sun = const.M_sun     # 太陽質量
R_sun = const.R_sun     # 太陽半徑
L_sun = const.L_sun     # 太陽光度
au = const.au           # 天文單位
pc = const.pc           # 秒差距

# 基本常數
h = const.h             # 普朗克常數
hbar = const.hbar       # 約化普朗克常數
k_B = const.k_B         # 波茲曼常數
m_e = const.m_e         # 電子質量
m_p = const.m_p         # 質子質量
e = const.e             # 基本電荷
N_A = const.N_A         # 亞佛加厥常數
```

### 在計算中使用常數

```python
# 計算史瓦西半徑
M = 10 * const.M_sun
r_s = 2 * const.G * M / const.c**2
print(f"史瓦西半徑: {r_s.to(u.km)}")

# 計算逃逸速度
M = const.M_earth
R = const.R_earth
v_esc = np.sqrt(2 * const.G * M / R)
print(f"地球逃逸速度: {v_esc.to(u.km/u.s)}")
```

## 卷積（astropy.convolution）

用於影像處理的卷積核心。

```python
from astropy.convolution import Gaussian2DKernel, convolve

# 建立高斯核心
kernel = Gaussian2DKernel(x_stddev=2)

# 卷積影像
smoothed_image = convolve(data, kernel)

# 處理 NaN
from astropy.convolution import convolve_fft
smoothed = convolve_fft(data, kernel, nan_treatment='interpolate')
```

## 統計（astropy.stats）

用於天文資料的統計函數。

```python
from astropy.stats import sigma_clip, sigma_clipped_stats

# Sigma 裁剪
clipped_data = sigma_clip(data, sigma=3, maxiters=5)

# 使用 sigma 裁剪取得統計
mean, median, std = sigma_clipped_stats(data, sigma=3.0)

# 穩健統計
from astropy.stats import mad_std, biweight_location, biweight_scale
robust_std = mad_std(data)
robust_mean = biweight_location(data)
robust_scale = biweight_scale(data)
```

## 工具

### 資料下載

```python
from astropy.utils.data import download_file

# 下載檔案（本地快取）
url = 'https://example.com/data.fits'
local_file = download_file(url, cache=True)
```

### 進度條

```python
from astropy.utils.console import ProgressBar

with ProgressBar(len(data_list)) as bar:
    for item in data_list:
        # 處理項目
        bar.update()
```

## SAMP（Simple Application Messaging Protocol）

與其他天文工具的互通性。

```python
from astropy.samp import SAMPIntegratedClient

# 連接到 SAMP hub
client = SAMPIntegratedClient()
client.connect()

# 向其他應用程式廣播表格
message = {
    'samp.mtype': 'table.load.votable',
    'samp.params': {
        'url': 'file:///path/to/table.xml',
        'table-id': 'my_table',
        'name': 'My Catalog'
    }
}
client.notify_all(message)

# 斷開連接
client.disconnect()
```
