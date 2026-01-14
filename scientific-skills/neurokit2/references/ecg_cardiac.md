# ECG 與心臟訊號處理

## 概述

處理心電圖（ECG，electrocardiogram）和光體積變化描記法（PPG，photoplethysmography）訊號以進行心血管分析。此模組提供全面的工具，用於 R 波峰檢測、波形描繪、品質評估和心率分析。

## 主要處理管線

### ecg_process()

完整的自動化 ECG 處理管線，協調多個步驟。

```python
signals, info = nk.ecg_process(ecg_signal, sampling_rate=1000, method='neurokit')
```

**管線步驟：**
1. 訊號清理（噪聲去除）
2. R 波峰檢測
3. 心率計算
4. 品質評估
5. QRS 描繪（P、Q、S、T 波）
6. 心臟相位判定

**回傳值：**
- `signals`：包含清理後 ECG、波峰、心率、品質、心臟相位的 DataFrame
- `info`：包含 R 波峰位置和處理參數的字典

**常用方法：**
- `'neurokit'`（預設）：完整的 NeuroKit2 管線
- `'biosppy'`：基於 BioSPPy 的處理
- `'pantompkins1985'`：Pan-Tompkins 演算法
- `'hamilton2002'`、`'elgendi2010'`、`'engzeemod2012'`：其他替代方法

## 預處理函數

### ecg_clean()

使用特定方法的濾波從原始 ECG 訊號中去除噪聲。

```python
cleaned_ecg = nk.ecg_clean(ecg_signal, sampling_rate=1000, method='neurokit')
```

**方法：**
- `'neurokit'`：高通 Butterworth 濾波器（0.5 Hz）+ 電源線濾波
- `'biosppy'`：0.67-45 Hz 之間的 FIR 濾波
- `'pantompkins1985'`：5-15 Hz 帶通 + 基於導數的方法
- `'hamilton2002'`：8-16 Hz 帶通
- `'elgendi2010'`：8-20 Hz 帶通
- `'engzeemod2012'`：0.5-40 Hz FIR 帶通

**主要參數：**
- `powerline`：移除 50 或 60 Hz 電源線噪聲（預設：50）

### ecg_peaks()

在 ECG 訊號中檢測 R 波峰，可選擇性進行偽影校正。

```python
peaks_dict, info = nk.ecg_peaks(cleaned_ecg, sampling_rate=1000, method='neurokit', correct_artifacts=False)
```

**可用方法（13+ 種演算法）：**
- `'neurokit'`：針對可靠性優化的混合方法
- `'pantompkins1985'`：經典 Pan-Tompkins 演算法
- `'hamilton2002'`：Hamilton 的自適應閾值
- `'christov2004'`：Christov 的自適應方法
- `'gamboa2008'`：Gamboa 的方法
- `'elgendi2010'`：Elgendi 的雙移動平均
- `'engzeemod2012'`：改良的 Engelse-Zeelenberg
- `'kalidas2017'`：基於 XQRS
- `'martinez2004'`、`'rodrigues2021'`、`'koka2022'`、`'promac'`：進階方法

**偽影校正：**
設定 `correct_artifacts=True` 以應用 Lipponen & Tarvainen（2019）校正：
- 檢測異位搏動、過長/過短間隔、漏失搏動
- 使用可配置參數的閾值檢測

**回傳值：**
- 包含 `'ECG_R_Peaks'` 鍵的字典，含有波峰索引

### ecg_delineate()

識別 P、Q、S、T 波及其起始點/結束點。

```python
waves, waves_peak = nk.ecg_delineate(cleaned_ecg, rpeaks, sampling_rate=1000, method='dwt')
```

**方法：**
- `'dwt'`（預設）：基於離散小波轉換的檢測
- `'peak'`：圍繞 R 波峰的簡單波峰檢測
- `'cwt'`：連續小波轉換（Martinez 等人，2004）

**檢測的成分：**
- P 波：`ECG_P_Peaks`、`ECG_P_Onsets`、`ECG_P_Offsets`
- Q 波：`ECG_Q_Peaks`
- S 波：`ECG_S_Peaks`
- T 波：`ECG_T_Peaks`、`ECG_T_Onsets`、`ECG_T_Offsets`
- QRS 波群：起始點和結束點

**回傳值：**
- `waves`：包含所有波形索引的字典
- `waves_peak`：包含波峰振幅的字典

### ecg_quality()

評估 ECG 訊號的完整性和品質。

```python
quality = nk.ecg_quality(ecg_signal, rpeaks=None, sampling_rate=1000, method='averageQRS')
```

**方法：**
- `'averageQRS'`（預設）：模板匹配相關性（Zhao & Zhang，2018）
  - 為每個心跳回傳 0-1 的品質分數
  - 閾值：>0.6 = 良好品質
- `'zhao2018'`：使用峰度、功率譜分布的多指標方法

**使用情境：**
- 識別低品質訊號區段
- 在分析前過濾噪聲心跳
- 驗證 R 波峰檢測準確性

## 分析函數

### ecg_analyze()

高階分析，自動選擇事件相關或區間相關模式。

```python
analysis = nk.ecg_analyze(signals, sampling_rate=1000, method='auto')
```

**模式選擇：**
- 持續時間 < 10 秒 → 事件相關分析
- 持續時間 ≥ 10 秒 → 區間相關分析

**回傳值：**
包含適合分析模式的心臟指標的 DataFrame。

### ecg_eventrelated()

分析刺激鎖定的 ECG 時期（epochs）以進行事件相關反應分析。

```python
results = nk.ecg_eventrelated(epochs)
```

**計算的指標：**
- `ECG_Rate_Baseline`：刺激前的平均心率
- `ECG_Rate_Min/Max`：時期內的最小/最大心率
- `ECG_Phase_Atrial/Ventricular`：刺激開始時的心臟相位
- 跨時期時間窗口的心率動態

**使用情境：**
具有離散試驗的實驗範式（例如：刺激呈現、任務事件）。

### ecg_intervalrelated()

分析連續 ECG 記錄以進行靜息狀態或延長期間的分析。

```python
results = nk.ecg_intervalrelated(signals, sampling_rate=1000)
```

**計算的指標：**
- `ECG_Rate_Mean`：區間內的平均心率
- 全面的 HRV 指標（委託給 `hrv()` 函數）
  - 時域：SDNN、RMSSD、pNN50 等
  - 頻域：LF、HF、LF/HF 比率
  - 非線性：Poincaré、熵、碎形測量

**最小持續時間：**
- 基本心率：任何持續時間
- HRV 頻域指標：建議 ≥60 秒，1-5 分鐘最佳

## 工具函數

### ecg_rate()

從 R 波峰間隔計算瞬時心率。

```python
heart_rate = nk.ecg_rate(peaks, sampling_rate=1000, desired_length=None)
```

**方法：**
- 計算連續 R 波峰之間的心跳間隔（IBIs，inter-beat intervals）
- 轉換為每分鐘心跳數（BPM）：60 / IBI
- 如果指定 `desired_length`，則內插以匹配訊號長度

**回傳值：**
- 瞬時心率值陣列

### ecg_phase()

判定心房和心室收縮期/舒張期相位。

```python
cardiac_phase = nk.ecg_phase(ecg_cleaned, rpeaks, delineate_info)
```

**計算的相位：**
- `ECG_Phase_Atrial`：心房收縮期（1）vs. 舒張期（0）
- `ECG_Phase_Ventricular`：心室收縮期（1）vs. 舒張期（0）
- `ECG_Phase_Completion_Atrial/Ventricular`：相位完成百分比（0-1）

**使用情境：**
- 心臟鎖定刺激呈現
- 將事件時間與心臟週期對齊的心理生理學實驗

### ecg_segment()

提取個別心跳以進行形態學分析。

```python
heartbeats = nk.ecg_segment(ecg_cleaned, rpeaks, sampling_rate=1000)
```

**回傳值：**
- 時期字典，每個包含一個心跳
- 以 R 波峰為中心，具有可配置的前/後窗口
- 用於逐搏形態學比較

### ecg_invert()

自動檢測並校正反轉的 ECG 訊號。

```python
corrected_ecg, is_inverted = nk.ecg_invert(ecg_signal, sampling_rate=1000)
```

**方法：**
- 分析 QRS 波群極性
- 如果主要為負向則翻轉訊號
- 回傳校正後的訊號和反轉狀態

### ecg_rsp()

提取 ECG 衍生呼吸（EDR）作為呼吸代理訊號。

```python
edr_signal = nk.ecg_rsp(ecg_cleaned, sampling_rate=1000, method='vangent2019')
```

**方法：**
- `'vangent2019'`：0.1-0.4 Hz 帶通濾波
- `'charlton2016'`：0.15-0.4 Hz 帶通
- `'soni2019'`：0.08-0.5 Hz 帶通

**使用情境：**
- 當無法獲得直接呼吸訊號時估計呼吸
- 多模態生理分析

## 模擬與視覺化

### ecg_simulate()

生成合成 ECG 訊號以進行測試和驗證。

```python
synthetic_ecg = nk.ecg_simulate(duration=10, sampling_rate=1000, heart_rate=70, method='ecgsyn', noise=0.01)
```

**方法：**
- `'ecgsyn'`：逼真的動態模型（McSharry 等人，2003）
  - 模擬 P-QRS-T 波群形態
  - 生理上合理的波形
- `'simple'`：更快的基於小波的近似
  - 類高斯的 QRS 波群
  - 較不逼真但計算效率高

**主要參數：**
- `heart_rate`：平均 BPM（預設：70）
- `heart_rate_std`：心率變異性幅度（預設：1）
- `noise`：高斯噪聲等級（預設：0.01）
- `random_state`：可重現性的種子

### ecg_plot()

視覺化處理後的 ECG 以及檢測到的 R 波峰和訊號品質。

```python
nk.ecg_plot(signals, info)
```

**顯示內容：**
- 原始和清理後的 ECG 訊號
- 疊加檢測到的 R 波峰
- 心率軌跡
- 訊號品質指標

## ECG 特定考量

### 取樣率建議
- **最低**：250 Hz 用於基本 R 波峰檢測
- **建議**：500-1000 Hz 用於波形描繪
- **高解析度**：2000+ Hz 用於詳細形態學分析

### 訊號持續時間要求
- **R 波峰檢測**：任何持續時間（最少 ≥2 個心跳）
- **基本心率**：≥10 秒
- **HRV 時域**：≥60 秒
- **HRV 頻域**：1-5 分鐘（最佳）
- **超低頻 HRV**：≥24 小時

### 常見問題與解決方案

**R 波峰檢測不佳：**
- 嘗試不同方法：`method='pantompkins1985'` 通常較穩健
- 確保足夠的取樣率（≥250 Hz）
- 檢查 ECG 是否反轉：使用 `ecg_invert()`
- 應用偽影校正：`correct_artifacts=True`

**噪聲訊號：**
- 針對噪聲類型使用適當的清理方法
- 如果在美國/歐洲以外，調整電源線頻率
- 在分析前考慮訊號品質評估

**缺失波形成分：**
- 提高取樣率（描繪需要 ≥500 Hz）
- 嘗試不同的描繪方法（`'dwt'`、`'peak'`、`'cwt'`）
- 使用 `ecg_quality()` 驗證訊號品質

## 與其他訊號的整合

### ECG + RSP：呼吸性竇性心律不整
```python
# 處理兩個訊號
ecg_signals, ecg_info = nk.ecg_process(ecg, sampling_rate=1000)
rsp_signals, rsp_info = nk.rsp_process(rsp, sampling_rate=1000)

# 計算 RSA
rsa = nk.hrv_rsa(ecg_info['ECG_R_Peaks'], rsp_signals['RSP_Clean'], sampling_rate=1000)
```

### 多模態整合
```python
# 一次處理多個訊號
bio_signals, bio_info = nk.bio_process(
    ecg=ecg_signal,
    rsp=rsp_signal,
    eda=eda_signal,
    sampling_rate=1000
)
```

## 參考文獻

- Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. IEEE transactions on biomedical engineering, 32(3), 230-236.
- Hamilton, P. (2002). Open source ECG analysis. Computers in cardiology, 101-104.
- Martinez, J. P., Almeida, R., Olmos, S., Rocha, A. P., & Laguna, P. (2004). A wavelet-based ECG delineator: evaluation on standard databases. IEEE Transactions on biomedical engineering, 51(4), 570-581.
- Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate variability time series artefact correction using novel beat classification. Journal of medical engineering & technology, 43(3), 173-181.
