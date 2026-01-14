# 深度學習網路

Aeon 提供專門為時間序列任務設計的神經網路架構。這些網路作為分類、迴歸、聚類和預測的構建模組。

## 核心網路架構

### 卷積網路

**FCNNetwork** - 全卷積網路
- 三個帶有批次標準化的卷積區塊
- 全域平均池化用於降維
- **使用時機**：需要簡單但有效的 CNN 基準

**ResNetNetwork** - 殘差網路
- 具有跳躍連接的殘差區塊
- 防止深層網路中的梯度消失
- **使用時機**：需要深層網路，訓練穩定性重要

**InceptionNetwork** - Inception 模組
- 使用平行卷積的多尺度特徵擷取
- 不同核心大小捕捉各種尺度的模式
- **使用時機**：模式存在於多個時間尺度

**TimeCNNNetwork** - 標準 CNN
- 基本卷積架構
- **使用時機**：簡單 CNN 足夠，重視可解釋性

**DisjointCNNNetwork** - 分離路徑
- 分離的卷積路徑
- **使用時機**：需要不同的特徵擷取策略

**DCNNNetwork** - 膨脹 CNN
- 膨脹卷積用於大感受野
- **使用時機**：無需多層即可處理長程依賴

### 遞迴網路

**RecurrentNetwork** - RNN/LSTM/GRU
- 可配置的細胞類型（RNN、LSTM、GRU）
- 時間依賴的序列建模
- **使用時機**：序列依賴性關鍵，可變長度序列

### 時間卷積網路

**TCNNetwork** - 時間卷積網路
- 膨脹因果卷積
- 無需遞迴的大感受野
- **使用時機**：長序列，需要可平行化架構

### 多層感知器

**MLPNetwork** - 基本前饋網路
- 簡單的全連接層
- 處理前展平時間序列
- **使用時機**：需要基準，計算限制，或簡單模式

## 基於編碼器的架構

設計用於表示學習和聚類的網路。

### 自動編碼器變體

**EncoderNetwork** - 通用編碼器
- 靈活的編碼器結構
- **使用時機**：需要自訂編碼

**AEFCNNetwork** - 基於 FCN 的自動編碼器
- 全卷積編碼器-解碼器
- **使用時機**：需要卷積表示學習

**AEResNetNetwork** - ResNet 自動編碼器
- 編碼器-解碼器中的殘差區塊
- **使用時機**：帶有跳躍連接的深度自動編碼

**AEDCNNNetwork** - 膨脹 CNN 自動編碼器
- 膨脹卷積用於壓縮
- **使用時機**：自動編碼器中需要大感受野

**AEDRNNNetwork** - 膨脹 RNN 自動編碼器
- 膨脹遞迴連接
- **使用時機**：具有長程依賴的序列模式

**AEBiGRUNetwork** - 雙向 GRU
- 雙向遞迴編碼
- **使用時機**：兩個方向的上下文都有幫助

**AEAttentionBiGRUNetwork** - 注意力 + BiGRU
- BiGRU 輸出上的注意力機制
- **使用時機**：需要關注重要的時間步

## 專門架構

**LITENetwork** - 輕量 Inception Time 集成
- 高效的基於 inception 的架構
- LITEMV 變體用於多變量序列
- **使用時機**：需要效率與強效能

**DeepARNetwork** - 機率預測
- 用於預測的自迴歸 RNN
- 產生機率預測
- **使用時機**：需要預測不確定性量化

## 搭配估計器使用

網路通常在估計器內使用，而非直接使用：

```python
from aeon.classification.deep_learning import FCNClassifier
from aeon.regression.deep_learning import ResNetRegressor
from aeon.clustering.deep_learning import AEFCNClusterer

# 使用 FCN 分類
clf = FCNClassifier(n_epochs=100, batch_size=16)
clf.fit(X_train, y_train)

# 使用 ResNet 迴歸
reg = ResNetRegressor(n_epochs=100)
reg.fit(X_train, y_train)

# 使用自動編碼器聚類
clusterer = AEFCNClusterer(n_clusters=3, n_epochs=100)
labels = clusterer.fit_predict(X_train)
```

## 自訂網路配置

許多網路接受配置參數：

```python
# 配置 FCN 層
clf = FCNClassifier(
    n_epochs=200,
    batch_size=32,
    kernel_size=[7, 5, 3],  # 每層的核心大小
    n_filters=[128, 256, 128],  # 每層的濾波器數
    learning_rate=0.001
)
```

## 基類

- `BaseDeepLearningNetwork` - 所有網路的抽象基類
- `BaseDeepRegressor` - 深度迴歸的基類
- `BaseDeepClassifier` - 深度分類的基類
- `BaseDeepForecaster` - 深度預測的基類

擴展這些以實作自訂架構。

## 訓練考量

### 超參數

需要調整的關鍵超參數：

- `n_epochs` - 訓練迭代次數（典型為 50-200）
- `batch_size` - 每批次樣本數（典型為 16-64）
- `learning_rate` - 步長大小（0.0001-0.01）
- 網路特定：層數、濾波器數、核心大小

### 回呼

許多網路支援回呼用於訓練監控：

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

clf = FCNClassifier(
    n_epochs=200,
    callbacks=[
        EarlyStopping(patience=20, restore_best_weights=True),
        ReduceLROnPlateau(patience=10, factor=0.5)
    ]
)
```

### GPU 加速

深度學習網路受益於 GPU：

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 使用第一個 GPU

# 如果可用，網路自動使用 GPU
clf = InceptionTimeClassifier(n_epochs=100)
clf.fit(X_train, y_train)
```

## 架構選擇

### 按任務：

**分類**：InceptionNetwork、ResNetNetwork、FCNNetwork
**迴歸**：InceptionNetwork、ResNetNetwork、TCNNetwork
**預測**：TCNNetwork、DeepARNetwork、RecurrentNetwork
**聚類**：AEFCNNetwork、AEResNetNetwork、AEAttentionBiGRUNetwork

### 按資料特性：

**長序列**：TCNNetwork、DCNNNetwork（膨脹卷積）
**短序列**：MLPNetwork、FCNNetwork
**多變量**：InceptionNetwork、FCNNetwork、LITENetwork
**可變長度**：帶遮罩的 RecurrentNetwork
**多尺度模式**：InceptionNetwork

### 按計算資源：

**有限計算**：MLPNetwork、LITENetwork
**中等計算**：FCNNetwork、TimeCNNNetwork
**高計算可用**：InceptionNetwork、ResNetNetwork
**GPU 可用**：任何深度網路（顯著加速）

## 最佳實務

### 1. 資料準備

標準化輸入資料：

```python
from aeon.transformations.collection import Normalizer

normalizer = Normalizer()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)
```

### 2. 訓練/驗證分割

使用驗證集進行早停：

```python
from sklearn.model_selection import train_test_split

X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train
)

clf = FCNClassifier(n_epochs=200)
clf.fit(X_train_fit, y_train_fit, validation_data=(X_val, y_val))
```

### 3. 從簡單開始

在複雜架構之前先嘗試較簡單的：

1. 首先嘗試 MLPNetwork 或 FCNNetwork
2. 如果不足，嘗試 ResNetNetwork 或 InceptionNetwork
3. 如果單一模型不足，考慮集成

### 4. 超參數調整

使用網格搜尋或隨機搜尋：

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_epochs': [100, 200],
    'batch_size': [16, 32],
    'learning_rate': [0.001, 0.0001]
}

clf = FCNClassifier()
grid = GridSearchCV(clf, param_grid, cv=3)
grid.fit(X_train, y_train)
```

### 5. 正則化

防止過擬合：
- 使用 dropout（如果網路支援）
- 早停
- 資料增強（如果可用）
- 減少模型複雜度

### 6. 可重現性

設定隨機種子：

```python
import numpy as np
import random
import tensorflow as tf

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
```
