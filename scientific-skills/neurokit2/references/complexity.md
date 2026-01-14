# 複雜度和熵分析

## 概述

複雜度測量量化時間序列訊號的不規則性、不可預測性和多尺度結構。NeuroKit2 提供全面的熵、碎形維度和非線性動力學測量，用於評估生理訊號複雜度。

## 主要函數

### complexity()

同時計算多個複雜度指標用於探索性分析。

```python
complexity_indices = nk.complexity(signal, sampling_rate=1000, show=False)
```

**返回：**
- 包含各類別眾多複雜度測量的 DataFrame：
  - 熵指標
  - 碎形維度
  - 非線性動力學測量
  - 資訊理論指標

**使用情境：**
- 識別相關測量的探索性分析
- 全面的訊號特徵化
- 跨訊號比較研究

## 參數最佳化

在計算複雜度測量之前，應確定最佳嵌入參數：

### complexity_delay()

確定相空間重建的最佳時間延遲（τ）。

```python
optimal_tau = nk.complexity_delay(signal, delay_max=100, method='fraser1986', show=False)
```

**方法：**
- `'fraser1986'`：互資訊第一最小值
- `'theiler1990'`：自相關第一零交叉
- `'casdagli1991'`：Cao 方法

**用途：**熵計算中的嵌入延遲、吸引子重建

### complexity_dimension()

確定最佳嵌入維度（m）。

```python
optimal_m = nk.complexity_dimension(signal, delay=None, dimension_max=20,
                                    method='afn', show=False)
```

**方法：**
- `'afn'`：平均偽最近鄰
- `'fnn'`：偽最近鄰
- `'correlation'`：相關維度飽和

**用途：**熵計算、相空間重建

### complexity_tolerance()

確定熵測量的最佳容差（r）。

```python
optimal_r = nk.complexity_tolerance(signal, method='sd', show=False)
```

**方法：**
- `'sd'`：基於標準差（典型為 0.1-0.25 × SD）
- `'maxApEn'`：最大化 ApEn
- `'recurrence'`：基於遞迴率

**用途：**近似熵、樣本熵

### complexity_k()

確定 Higuchi 碎形維度的最佳 k 參數。

```python
optimal_k = nk.complexity_k(signal, k_max=20, show=False)
```

**用途：**Higuchi 碎形維度計算

## 熵測量

熵量化隨機性、不可預測性和資訊含量。

### entropy_shannon()

Shannon 熵 - 經典資訊理論測量。

```python
shannon_entropy = nk.entropy_shannon(signal)
```

**解讀：**
- 較高：更隨機、較不可預測
- 較低：更規則、可預測
- 單位：位元（資訊）

**使用情境：**
- 一般隨機性評估
- 資訊含量
- 訊號不規則性

### entropy_approximate()

近似熵（ApEn）- 模式的規則性。

```python
apen = nk.entropy_approximate(signal, delay=1, dimension=2, tolerance='sd')
```

**參數：**
- `delay`：時間延遲（τ）
- `dimension`：嵌入維度（m）
- `tolerance`：相似性閾值（r）

**解讀：**
- 較低的 ApEn：更規則、自相似的模式
- 較高的 ApEn：更複雜、不規則
- 對訊號長度敏感（建議 ≥100-300 點）

**生理應用：**
- HRV：心臟疾病時 ApEn 降低
- EEG：神經系統疾病時 ApEn 改變

### entropy_sample()

樣本熵（SampEn）- 改進的 ApEn。

```python
sampen = nk.entropy_sample(signal, delay=1, dimension=2, tolerance='sd')
```

**相對 ApEn 的優點：**
- 較不依賴訊號長度
- 跨記錄更一致
- 無自我匹配偏差

**解讀：**
- 與 ApEn 相同但更可靠
- 在大多數應用中更受青睞

**典型值：**
- HRV：0.5-2.5（依情境而定）
- EEG：0.3-1.5

### entropy_multiscale()

多尺度熵（MSE）- 跨時間尺度的複雜度。

```python
mse = nk.entropy_multiscale(signal, scale=20, dimension=2, tolerance='sd',
                            method='MSEn', show=False)
```

**方法：**
- `'MSEn'`：多尺度樣本熵
- `'MSApEn'`：多尺度近似熵
- `'CMSE'`：複合多尺度熵
- `'RCMSE'`：精煉複合多尺度熵

**解讀：**
- 不同粗粒化尺度的熵
- 健康/複雜系統：多個尺度上的高熵
- 疾病/簡單系統：熵降低，特別是在較大尺度

**使用情境：**
- 區分真正的複雜度和隨機性
- 白噪音：跨尺度恆定
- 粉紅噪音/複雜度：跨尺度的結構化變異

### entropy_fuzzy()

模糊熵 - 使用模糊隸屬函數。

```python
fuzzen = nk.entropy_fuzzy(signal, delay=1, dimension=2, tolerance='sd', r=0.2)
```

**優點：**
- 對雜訊訊號更穩定
- 模式匹配的模糊邊界
- 對短訊號表現更好

### entropy_permutation()

排列熵 - 基於序數模式。

```python
perment = nk.entropy_permutation(signal, delay=1, dimension=3)
```

**方法：**
- 將訊號編碼為序數模式（排列）
- 計算模式頻率
- 對雜訊和非平穩性穩健

**解讀：**
- 較低：更規則的序數結構
- 較高：更隨機的排序

**使用情境：**
- EEG 分析
- 麻醉深度監測
- 快速計算

### entropy_spectral()

頻譜熵 - 基於功率譜。

```python
spec_ent = nk.entropy_spectral(signal, sampling_rate=1000, bands=None)
```

**方法：**
- 功率譜的正規化 Shannon 熵
- 量化頻率分佈的規則性

**解讀：**
- 0：單一頻率（純音）
- 1：白噪音（平坦頻譜）

**使用情境：**
- EEG：頻譜分佈隨狀態變化
- 麻醉監測

### entropy_svd()

奇異值分解熵。

```python
svd_ent = nk.entropy_svd(signal, delay=1, dimension=2)
```

**方法：**
- 對軌跡矩陣進行 SVD
- 奇異值分佈的熵

**使用情境：**
- 吸引子複雜度
- 確定性與隨機動力學

### entropy_differential()

微分熵 - Shannon 熵的連續類比。

```python
diff_ent = nk.entropy_differential(signal)
```

**用途：**連續機率分佈

### 其他熵測量

**Tsallis 熵：**
```python
tsallis = nk.entropy_tsallis(signal, q=2)
```
- 帶參數 q 的廣義熵
- q=1 還原為 Shannon 熵

**Rényi 熵：**
```python
renyi = nk.entropy_renyi(signal, alpha=2)
```
- 帶參數 α 的廣義熵

**其他專門熵：**
- `entropy_attention()`：注意力熵
- `entropy_grid()`：網格熵
- `entropy_increment()`：增量熵
- `entropy_slope()`：斜率熵
- `entropy_dispersion()`：離散熵
- `entropy_symbolicdynamic()`：符號動力學熵
- `entropy_range()`：範圍熵
- `entropy_phase()`：相位熵
- `entropy_quadratic()`、`entropy_cumulative_residual()`、`entropy_rate()`：專門變體

## 碎形維度測量

碎形維度描述自相似性和粗糙度特徵。

### fractal_katz()

Katz 碎形維度 - 波形複雜度。

```python
kfd = nk.fractal_katz(signal)
```

**解讀：**
- 1：直線
- >1：粗糙度和複雜度增加
- 典型範圍：1.0-2.0

**優點：**
- 簡單、快速計算
- 無需參數調整

### fractal_higuchi()

Higuchi 碎形維度 - 自相似性。

```python
hfd = nk.fractal_higuchi(signal, k_max=10)
```

**方法：**
- 從原始訊號構建 k 個新時間序列
- 從長度-尺度關係估計維度

**解讀：**
- 較高的 HFD：更複雜、不規則
- 較低的 HFD：更平滑、規則

**使用情境：**
- EEG 複雜度
- HRV 分析
- 癲癇檢測

### fractal_petrosian()

Petrosian 碎形維度 - 快速估計。

```python
pfd = nk.fractal_petrosian(signal)
```

**優點：**
- 快速計算
- 直接計算（無需曲線擬合）

### fractal_sevcik()

Sevcik 碎形維度 - 正規化波形複雜度。

```python
sfd = nk.fractal_sevcik(signal)
```

### fractal_nld()

正規化長度密度 - 基於曲線長度的測量。

```python
nld = nk.fractal_nld(signal)
```

### fractal_psdslope()

功率譜密度斜率 - 頻域碎形測量。

```python
slope = nk.fractal_psdslope(signal, sampling_rate=1000)
```

**方法：**
- 對數-對數功率譜的線性擬合
- 斜率 β 與碎形維度相關

**解讀：**
- β ≈ 0：白噪音（隨機）
- β ≈ -1：粉紅噪音（1/f，複雜）
- β ≈ -2：棕噪音（布朗運動）

### fractal_hurst()

Hurst 指數 - 長程相依性。

```python
hurst = nk.fractal_hurst(signal, show=False)
```

**解讀：**
- H < 0.5：反持續性（均值回歸）
- H = 0.5：隨機遊走（白噪音）
- H > 0.5：持續性（趨勢、長記憶）

**使用情境：**
- 評估長期相關性
- 金融時間序列
- HRV 分析

### fractal_correlation()

相關維度 - 吸引子維度。

```python
corr_dim = nk.fractal_correlation(signal, delay=1, dimension=10, radius=64)
```

**方法：**
- Grassberger-Procaccia 演算法
- 估計相空間中吸引子的維度

**解讀：**
- 低維度：確定性、低維混沌
- 高維度：高維混沌或雜訊

### fractal_dfa()

去趨勢波動分析 - 尺度指數。

```python
dfa_alpha = nk.fractal_dfa(signal, multifractal=False, q=2, show=False)
```

**解讀：**
- α < 0.5：反相關
- α = 0.5：不相關（白噪音）
- α = 1.0：1/f 噪音（粉紅噪音，健康複雜度）
- α = 1.5：布朗噪音
- α > 1.0：持續長程相關

**HRV 應用：**
- α1（短期，4-11 拍）：反映自主神經調節
- α2（長期，>11 拍）：長程相關
- α1 降低：心臟病理

### fractal_mfdfa()

多重碎形 DFA - 多尺度碎形屬性。

```python
mfdfa_results = nk.fractal_mfdfa(signal, q=None, show=False)
```

**方法：**
- 將 DFA 擴展到多個 q 階
- 描述多重碎形頻譜特徵

**返回：**
- 廣義 Hurst 指數 h(q)
- 多重碎形頻譜 f(α)
- 寬度表示多重碎形強度

**使用情境：**
- 檢測多重碎形結構
- 健康與疾病的 HRV 多重碎形性
- EEG 多尺度動力學

### fractal_tmf()

多重碎形非線性 - 偏離單碎形。

```python
tmf = nk.fractal_tmf(signal)
```

**解讀：**
- 量化偏離簡單尺度的程度
- 較高：更多多重碎形結構

### fractal_density()

密度碎形維度。

```python
density_fd = nk.fractal_density(signal)
```

### fractal_linelength()

線長度 - 總變異測量。

```python
linelength = nk.fractal_linelength(signal)
```

**使用情境：**
- 簡單的複雜度代理
- EEG 癲癇發作檢測

## 非線性動力學

### complexity_lyapunov()

最大 Lyapunov 指數 - 混沌和發散。

```python
lyap = nk.complexity_lyapunov(signal, delay=None, dimension=None,
                              sampling_rate=1000, show=False)
```

**解讀：**
- λ < 0：穩定不動點
- λ = 0：週期軌道
- λ > 0：混沌（鄰近軌跡指數發散）

**使用情境：**
- 檢測生理訊號中的混沌
- HRV：正 Lyapunov 表明非線性動力學
- EEG：癲癇檢測（發作前 λ 降低）

### complexity_lempelziv()

Lempel-Ziv 複雜度 - 演算法複雜度。

```python
lz = nk.complexity_lempelziv(signal, symbolize='median')
```

**方法：**
- 計算不同模式的數量
- 粗粒化隨機性測量

**解讀：**
- 較低：重複、可預測的模式
- 較高：多樣、不可預測的模式

**使用情境：**
- EEG：意識水準、麻醉
- HRV：自主神經複雜度

### complexity_rqa()

遞迴量化分析 - 相空間遞迴。

```python
rqa_indices = nk.complexity_rqa(signal, delay=1, dimension=3, tolerance='sd')
```

**指標：**
- **遞迴率（RR）**：遞迴狀態的百分比
- **確定性（DET）**：線上遞迴點的百分比
- **層疊性（LAM）**：垂直結構中的百分比（層流狀態）
- **捕捉時間（TT）**：平均垂直線長度
- **最長對角線/垂直線**：系統可預測性
- **熵（ENTR）**：線長度分佈的 Shannon 熵

**解讀：**
- 高 DET：確定性動力學
- 高 LAM：系統困在特定狀態
- 低 RR：隨機、非遞迴動力學

**使用情境：**
- 檢測系統動力學中的轉換
- 生理狀態變化
- 非線性時間序列分析

### complexity_hjorth()

Hjorth 參數 - 時域複雜度。

```python
hjorth = nk.complexity_hjorth(signal)
```

**指標：**
- **活動**：訊號的變異數
- **移動性**：導數標準差與訊號的比例
- **複雜度**：導數移動性的變化

**使用情境：**
- EEG 特徵提取
- 癲癇發作檢測
- 訊號特徵化

### complexity_decorrelation()

去相關時間 - 記憶持續時間。

```python
decorr_time = nk.complexity_decorrelation(signal, show=False)
```

**解讀：**
- 自相關降到閾值以下的時間延遲
- 較短：快速波動、短記憶
- 較長：慢波動、長記憶

### complexity_relativeroughness()

相對粗糙度 - 平滑度測量。

```python
roughness = nk.complexity_relativeroughness(signal)
```

## 資訊理論

### fisher_information()

Fisher 資訊 - 秩序測量。

```python
fisher = nk.fisher_information(signal, delay=1, dimension=2)
```

**解讀：**
- 高：有序、有結構
- 低：無序、隨機

**使用情境：**
- 與 Shannon 熵結合（Fisher-Shannon 平面）
- 描述系統複雜度特徵

### fishershannon_information()

Fisher-Shannon 資訊乘積。

```python
fs = nk.fishershannon_information(signal)
```

**方法：**
- Fisher 資訊和 Shannon 熵的乘積
- 描述秩序-無序平衡特徵

### mutual_information()

互資訊 - 變數間的共享資訊。

```python
mi = nk.mutual_information(signal1, signal2, method='knn')
```

**方法：**
- `'knn'`：k 最近鄰（非參數）
- `'kernel'`：核密度估計
- `'binning'`：基於直方圖

**使用情境：**
- 訊號間耦合
- 特徵選擇
- 非線性相依性

## 實務考量

### 訊號長度要求

| 測量 | 最小長度 | 最佳長度 |
|---------|---------------|----------------|
| Shannon 熵 | 50 | 200+ |
| ApEn、SampEn | 100-300 | 500-1000 |
| 多尺度熵 | 500 | 每尺度 1000+ |
| DFA | 500 | 1000+ |
| Lyapunov | 1000 | 5000+ |
| 相關維度 | 1000 | 5000+ |

### 參數選擇

**一般準則：**
- 首先使用參數最佳化函數
- 或使用傳統預設值：
  - 延遲（τ）：HRV 為 1，EEG 為自相關第一最小值
  - 維度（m）：典型為 2-3
  - 容差（r）：常見為 0.2 × SD

**敏感性：**
- 結果可能對參數敏感
- 報告使用的參數
- 考慮敏感性分析

### 正規化和預處理

**標準化：**
- 許多測量對訊號振幅敏感
- 通常建議 Z 分數正規化
- 可能需要去趨勢

**平穩性：**
- 某些測量假設平穩性
- 使用統計檢定檢查（例如 ADF 檢定）
- 分割非平穩訊號

### 解讀

**依情境而定：**
- 沒有普遍「好」或「壞」的複雜度
- 在受試者內或群組間比較
- 考慮生理情境

**複雜度與隨機性：**
- 最大熵 ≠ 最大複雜度
- 真正的複雜度：結構化變異性
- 白噪音：高熵但低複雜度（MSE 可區分）

## 應用

**心血管：**
- HRV 複雜度：心臟病、老化時降低
- DFA α1：心肌梗塞後的預後標記

**神經科學：**
- EEG 複雜度：意識、麻醉深度
- 熵：阿茲海默症、癲癇、睡眠階段
- 排列熵：麻醉監測

**心理學：**
- 憂鬱、焦慮時複雜度喪失
- 壓力下規則性增加

**老化：**
- 跨系統老化時「複雜度喪失」
- 多尺度複雜度降低

**臨界轉換：**
- 狀態轉換前複雜度變化
- 早期預警訊號（臨界減速）

## 參考文獻

- Pincus, S. M. (1991). Approximate entropy as a measure of system complexity. Proceedings of the National Academy of Sciences, 88(6), 2297-2301.
- Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
- Peng, C. K., et al. (1995). Quantification of scaling exponents and crossover phenomena in nonstationary heartbeat time series. Chaos, 5(1), 82-87.
- Costa, M., Goldberger, A. L., & Peng, C. K. (2005). Multiscale entropy analysis of biological signals. Physical review E, 71(2), 021906.
- Grassberger, P., & Procaccia, I. (1983). Measuring the strangeness of strange attractors. Physica D: Nonlinear Phenomena, 9(1-2), 189-208.
