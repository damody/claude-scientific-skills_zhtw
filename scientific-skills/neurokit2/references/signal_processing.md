# 通用訊號處理

## 概述

NeuroKit2 提供適用於任何時間序列資料的完整訊號處理工具。這些函數支援濾波、轉換、波峰偵測、分解和分析操作，適用於所有訊號類型。

## 預處理函數

### signal_filter()

應用頻域濾波以移除噪音或隔離頻帶。

```python
filtered = nk.signal_filter(signal, sampling_rate=1000, lowcut=None, highcut=None,
                            method='butterworth', order=5)
```

**濾波器類型（透過 lowcut/highcut 組合）：**

**低通**（僅 highcut）：
```python
lowpass = nk.signal_filter(signal, sampling_rate=1000, highcut=50)
```
- 移除高於 highcut 的頻率
- 平滑訊號，移除高頻噪音

**高通**（僅 lowcut）：
```python
highpass = nk.signal_filter(signal, sampling_rate=1000, lowcut=0.5)
```
- 移除低於 lowcut 的頻率
- 移除基線漂移、直流偏移

**帶通**（同時有 lowcut 和 highcut）：
```python
bandpass = nk.signal_filter(signal, sampling_rate=1000, lowcut=0.5, highcut=50)
```
- 保留 lowcut 和 highcut 之間的頻率
- 隔離特定頻帶

**帶阻/陷波**（移除市電噪音）：
```python
notch = nk.signal_filter(signal, sampling_rate=1000, method='powerline', powerline=50)
```
- 移除 50 或 60 Hz 市電噪音
- 窄陷波濾波器

**方法：**
- `'butterworth'`（預設）：平滑的頻率響應，平坦的通帶
- `'bessel'`：線性相位，最小振鈴
- `'chebyshev1'`：較陡的滾降，通帶有漣波
- `'chebyshev2'`：較陡的滾降，阻帶有漣波
- `'elliptic'`：最陡的滾降，兩帶都有漣波
- `'powerline'`：50/60 Hz 的陷波濾波器

**階數參數：**
- 較高階數：較陡的轉換，更多振鈴
- 較低階數：較緩的轉換，較少振鈴
- 典型：生理訊號使用 2-5 階

### signal_sanitize()

移除無效值（NaN、inf）並可選地內插。

```python
clean_signal = nk.signal_sanitize(signal, interpolate=True)
```

**應用場景：**
- 處理遺漏的資料點
- 移除標記為 NaN 的偽跡
- 為需要連續資料的演算法準備訊號

### signal_resample()

更改訊號的取樣率（升取樣或降取樣）。

```python
resampled = nk.signal_resample(signal, sampling_rate=1000, desired_sampling_rate=500,
                               method='interpolation')
```

**方法：**
- `'interpolation'`：三次樣條內插
- `'FFT'`：頻域重取樣
- `'poly'`：多相濾波（降取樣最佳）

**應用場景：**
- 匹配多模態記錄的取樣率
- 減少資料大小（降取樣）
- 增加時間解析度（升取樣）

### signal_fillmissing()

內插遺漏或無效的資料點。

```python
filled = nk.signal_fillmissing(signal, method='linear')
```

**方法：**
- `'linear'`：線性內插
- `'nearest'`：最近鄰
- `'pad'`：前向/後向填充
- `'cubic'`：三次樣條
- `'polynomial'`：多項式擬合

## 轉換函數

### signal_detrend()

從訊號移除緩慢趨勢。

```python
detrended = nk.signal_detrend(signal, method='polynomial', order=1)
```

**方法：**
- `'polynomial'`：擬合並減去多項式（階數 1 = 線性）
- `'loess'`：局部加權迴歸
- `'tarvainen2002'`：平滑先驗去趨勢

**應用場景：**
- 移除基線漂移
- 分析前穩定平均值
- 為假設平穩性的演算法準備

### signal_decompose()

將訊號分解為組成成分。

```python
components = nk.signal_decompose(signal, sampling_rate=1000, method='emd')
```

**方法：**

**經驗模態分解（EMD）：**
```python
components = nk.signal_decompose(signal, sampling_rate=1000, method='emd')
```
- 資料自適應分解為本徵模態函數（IMF）
- 每個 IMF 代表不同的頻率內容（從高到低）
- 無預定義的基函數

**奇異譜分析（SSA）：**
```python
components = nk.signal_decompose(signal, method='ssa')
```
- 分解為趨勢、振盪和噪音
- 基於軌跡矩陣的特徵值分解

**小波分解：**
- 時頻表示
- 在時間和頻率上都是局部的

**回傳：**
- 包含成分訊號的字典
- 趨勢、振盪成分、殘差

**應用場景：**
- 隔離生理節律
- 分離訊號和噪音
- 多尺度分析

### signal_recompose()

從分解的成分重建訊號。

```python
reconstructed = nk.signal_recompose(components, indices=[1, 2, 3])
```

**應用場景：**
- 分解後的選擇性重建
- 移除特定 IMF 或成分
- 自適應濾波

### signal_binarize()

根據閾值將連續訊號轉換為二元（0/1）。

```python
binary = nk.signal_binarize(signal, method='threshold', threshold=0.5)
```

**方法：**
- `'threshold'`：簡單閾值
- `'median'`：基於中位數
- `'mean'`：基於平均值
- `'quantile'`：基於百分位數

**應用場景：**
- 從連續訊號偵測事件
- 觸發提取
- 狀態分類

### signal_distort()

為測試添加受控噪音或偽跡。

```python
distorted = nk.signal_distort(signal, sampling_rate=1000, noise_amplitude=0.1,
                              noise_frequency=50, artifacts_amplitude=0.5)
```

**參數：**
- `noise_amplitude`：高斯噪音水平
- `noise_frequency`：正弦干擾（例如，市電）
- `artifacts_amplitude`：隨機尖峰偽跡
- `artifacts_number`：要添加的偽跡數量

**應用場景：**
- 演算法穩健性測試
- 預處理方法評估
- 真實資料模擬

### signal_interpolate()

在新時間點內插訊號或填補間隙。

```python
interpolated = nk.signal_interpolate(x_values, y_values, x_new=None, method='quadratic')
```

**方法：**
- `'linear'`、`'quadratic'`、`'cubic'`：多項式內插
- `'nearest'`：最近鄰
- `'monotone_cubic'`：保持單調性

**應用場景：**
- 將不規則樣本轉換為規則網格
- 升取樣以進行視覺化
- 對齊不同時間基準的訊號

### signal_merge()

合併具有不同取樣率的多個訊號。

```python
merged = nk.signal_merge(signal1, signal2, time1=None, time2=None, sampling_rate=None)
```

**應用場景：**
- 多模態訊號整合
- 合併來自不同設備的資料
- 基於時間戳同步

### signal_flatline()

識別恆定訊號的時期（偽跡或感測器故障）。

```python
flatline_mask = nk.signal_flatline(signal, duration=5.0, sampling_rate=1000)
```

**回傳：**
- 二元遮罩，True 表示平線時期
- 持續時間閾值防止正常穩定性的誤報

### signal_noise()

向訊號添加各種類型的噪音。

```python
noisy = nk.signal_noise(signal, sampling_rate=1000, noise_type='gaussian',
                        amplitude=0.1)
```

**噪音類型：**
- `'gaussian'`：白噪音
- `'pink'`：1/f 噪音（生理訊號中常見）
- `'brown'`：布朗運動（隨機漫步）
- `'powerline'`：正弦干擾（50/60 Hz）

### signal_surrogate()

生成保留某些特性的替代訊號。

```python
surrogate = nk.signal_surrogate(signal, method='IAAFT')
```

**方法：**
- `'IAAFT'`：迭代振幅調整傅立葉變換
  - 保留振幅分布和功率譜
- `'random_shuffle'`：隨機排列（虛無假設測試）

**應用場景：**
- 非線性測試
- 統計檢驗的虛無假設生成

## 波峰偵測和校正

### signal_findpeaks()

偵測訊號中的局部最大值（波峰）。

```python
peaks_dict = nk.signal_findpeaks(signal, height_min=None, height_max=None,
                                 relative_height_min=None, relative_height_max=None)
```

**關鍵參數：**
- `height_min/max`：絕對振幅閾值
- `relative_height_min/max`：相對於訊號範圍（0-1）
- `threshold`：最小突出度
- `distance`：波峰之間的最小樣本數

**回傳：**
- 字典包含：
  - `'Peaks'`：波峰索引
  - `'Height'`：波峰振幅
  - `'Distance'`：波峰間隔

**應用場景：**
- 任何訊號的通用波峰偵測
- R 波峰、呼吸波峰、脈搏波峰
- 事件偵測

### signal_fixpeaks()

校正偵測到的波峰的偽跡和異常。

```python
corrected = nk.signal_fixpeaks(peaks, sampling_rate=1000, iterative=True,
                               method='Kubios', interval_min=None, interval_max=None)
```

**方法：**
- `'Kubios'`：Kubios HRV 軟體方法（預設）
- `'Malik1996'`：任務小組標準（1996）
- `'Kamath1993'`：Kamath 方法

**校正：**
- 移除生理上不合理的間隔
- 內插遺漏的波峰
- 移除額外偵測的波峰（重複）

**應用場景：**
- R-R 間隔的偽跡校正
- 改善 HRV 分析品質
- 呼吸或脈搏波峰校正

## 分析函數

### signal_rate()

從事件發生（波峰）計算瞬時速率。

```python
rate = nk.signal_rate(peaks, sampling_rate=1000, desired_length=None)
```

**方法：**
- 計算事件間隔
- 轉換為每分鐘事件數
- 內插以匹配所需長度

**應用場景：**
- 從 R 波峰計算心率
- 從呼吸波峰計算呼吸率
- 任何週期性事件的速率

### signal_period()

找出訊號中的主導週期/頻率。

```python
period = nk.signal_period(signal, sampling_rate=1000, method='autocorrelation',
                          show=False)
```

**方法：**
- `'autocorrelation'`：自相關函數的峰值
- `'powerspectraldensity'`：頻譜的峰值

**回傳：**
- 週期（樣本或秒）
- 頻率（1/週期）（Hz）

**應用場景：**
- 偵測主導節律
- 估計基頻
- 呼吸率、心率估計

### signal_phase()

計算訊號的瞬時相位。

```python
phase = nk.signal_phase(signal, method='hilbert')
```

**方法：**
- `'hilbert'`：希爾伯特變換（解析訊號）
- `'wavelet'`：基於小波的相位

**回傳：**
- 相位（弧度 -π 到 π）或 0 到 1（標準化）

**應用場景：**
- 相位鎖定分析
- 同步測量
- 相位-振幅耦合

### signal_psd()

計算功率譜密度。

```python
psd, freqs = nk.signal_psd(signal, sampling_rate=1000, method='welch',
                           max_frequency=None, show=False)
```

**方法：**
- `'welch'`：Welch 週期圖（窗口 FFT，預設）
- `'multitapers'`：多錐方法（優越的頻譜估計）
- `'lomb'`：Lomb-Scargle（不均勻取樣資料）
- `'burg'`：自迴歸（參數）

**回傳：**
- `psd`：每個頻率的功率（單位²/Hz）
- `freqs`：頻率區間（Hz）

**應用場景：**
- 頻率內容分析
- HRV 頻域
- 頻譜特徵

### signal_power()

計算特定頻帶的功率。

```python
power_dict = nk.signal_power(signal, sampling_rate=1000, frequency_bands={
    'VLF': (0.003, 0.04),
    'LF': (0.04, 0.15),
    'HF': (0.15, 0.4)
}, method='welch')
```

**回傳：**
- 每個頻帶的絕對和相對功率字典
- 峰值頻率

**應用場景：**
- HRV 頻率分析
- EEG 頻帶功率
- 節律量化

### signal_autocor()

計算自相關函數。

```python
autocorr = nk.signal_autocor(signal, lag=1000, show=False)
```

**解讀：**
- 在延遲處有高自相關：訊號每延遲樣本重複
- 週期性訊號：週期倍數處有峰值
- 隨機訊號：快速衰減到零

**應用場景：**
- 偵測週期性
- 評估時間結構
- 訊號中的記憶

### signal_zerocrossings()

計算過零次數（符號變化）。

```python
n_crossings = nk.signal_zerocrossings(signal)
```

**解讀：**
- 更多交叉：更高的頻率內容
- 與主導頻率相關（粗略估計）

**應用場景：**
- 簡單頻率估計
- 訊號規律性評估

### signal_changepoints()

偵測訊號特性（平均值、變異數）的突然變化。

```python
changepoints = nk.signal_changepoints(signal, penalty=10, method='pelt', show=False)
```

**方法：**
- `'pelt'`：修剪精確線性時間（快速、精確）
- `'binseg'`：二元分割（更快、近似）

**參數：**
- `penalty`：控制敏感度（較高 = 較少變化點）

**回傳：**
- 偵測到的變化點索引
- 變化點之間的片段

**應用場景：**
- 將訊號分割為狀態
- 偵測轉換（例如，睡眠階段、喚起狀態）
- 自動時段定義

### signal_synchrony()

評估兩個訊號之間的同步性。

```python
sync = nk.signal_synchrony(signal1, signal2, method='correlation')
```

**方法：**
- `'correlation'`：皮爾森相關
- `'coherence'`：頻域相干性
- `'mutual_information'`：資訊論測量
- `'phase'`：相位鎖定值

**應用場景：**
- 心腦耦合
- 大腦間同步
- 多通道協調

### signal_smooth()

應用平滑以減少噪音。

```python
smoothed = nk.signal_smooth(signal, method='convolution', kernel='boxzen', size=10)
```

**方法：**
- `'convolution'`：應用核（矩形、高斯等）
- `'median'`：中值濾波器（對異常值穩健）
- `'savgol'`：Savitzky-Golay 濾波器（保留波峰）
- `'loess'`：局部加權迴歸

**核類型（用於卷積）：**
- `'boxcar'`：簡單移動平均
- `'gaussian'`：高斯加權平均
- `'hann'`、`'hamming'`、`'blackman'`：窗函數

**應用場景：**
- 降噪
- 趨勢提取
- 視覺化增強

### signal_timefrequency()

時頻表示（頻譜圖）。

```python
tf, time, freq = nk.signal_timefrequency(signal, sampling_rate=1000, method='stft',
                                        max_frequency=50, show=False)
```

**方法：**
- `'stft'`：短時傅立葉變換
- `'cwt'`：連續小波變換

**回傳：**
- `tf`：時頻矩陣（每個時頻點的功率）
- `time`：時間區間
- `freq`：頻率區間

**應用場景：**
- 非平穩訊號分析
- 時變頻率內容
- EEG/MEG 時頻分析

## 模擬

### signal_simulate()

生成各種用於測試的合成訊號。

```python
signal = nk.signal_simulate(duration=10, sampling_rate=1000, frequency=[5, 10],
                            amplitude=[1.0, 0.5], noise=0.1)
```

**訊號類型：**
- 正弦振盪（指定頻率）
- 多頻率成分
- 高斯噪音
- 組合

**應用場景：**
- 演算法測試
- 方法驗證
- 教育示範

## 視覺化

### signal_plot()

視覺化訊號和選用的標記。

```python
nk.signal_plot(signal, sampling_rate=1000, peaks=None, show=True)
```

**功能：**
- 以秒為單位的時間軸
- 波峰標記
- 訊號陣列的多子圖

## 實務技巧

**選擇濾波器參數：**
- **Lowcut**：設定在感興趣的最低頻率以下
- **Highcut**：設定在感興趣的最高頻率以上
- **階數**：從 2-5 開始，如果轉換太慢則增加
- **方法**：Butterworth 是安全的預設值

**處理邊緣效應：**
- 濾波在訊號邊緣引入偽跡
- 濾波前填充訊號，然後裁剪
- 或丟棄初始/最終幾秒

**處理間隙：**
- 小間隙：使用內插的 `signal_fillmissing()`
- 大間隙：分割訊號，分別分析
- 將間隙標記為 NaN，謹慎使用內插

**組合操作：**
```python
# 典型預處理管線
signal = nk.signal_sanitize(raw_signal)  # 移除無效值
signal = nk.signal_filter(signal, sampling_rate=1000, lowcut=0.5, highcut=40)  # 帶通
signal = nk.signal_detrend(signal, method='polynomial', order=1)  # 移除線性趨勢
```

**效能考量：**
- 濾波：長訊號使用基於 FFT 的方法更快
- 重取樣：在管線早期降取樣以加速
- 大型資料集：如果記憶體受限，分塊處理

## 參考文獻

- Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods, 17(3), 261-272.
- Tarvainen, M. P., Ranta-aho, P. O., & Karjalainen, P. A. (2002). An advanced detrending method with application to HRV analysis. IEEE Transactions on Biomedical Engineering, 49(2), 172-175.
- Huang, N. E., et al. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. Proceedings of the Royal Society of London A, 454(1971), 903-995.
