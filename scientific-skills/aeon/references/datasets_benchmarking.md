# 資料集與基準測試

Aeon 提供全面的工具用於載入資料集和對時間序列演算法進行基準測試。

## 資料集載入

### 任務特定載入器

**分類資料集**：
```python
from aeon.datasets import load_classification

# 載入訓練/測試分割
X_train, y_train = load_classification("GunPoint", split="train")
X_test, y_test = load_classification("GunPoint", split="test")

# 載入整個資料集
X, y = load_classification("GunPoint")
```

**迴歸資料集**：
```python
from aeon.datasets import load_regression

X_train, y_train = load_regression("Covid3Month", split="train")
X_test, y_test = load_regression("Covid3Month", split="test")

# 批次下載
from aeon.datasets import download_all_regression
download_all_regression()  # 下載 Monash TSER 存檔
```

**預測資料集**：
```python
from aeon.datasets import load_forecasting

# 從 forecastingdata.org 載入
y, X = load_forecasting("airline", return_X_y=True)
```

**異常檢測資料集**：
```python
from aeon.datasets import load_anomaly_detection

X, y = load_anomaly_detection("NAB_realKnownCause")
```

### 檔案格式載入器

**從 .ts 檔案載入**：
```python
from aeon.datasets import load_from_ts_file

X, y = load_from_ts_file("path/to/data.ts")
```

**從 .tsf 檔案載入**：
```python
from aeon.datasets import load_from_tsf_file

df, metadata = load_from_tsf_file("path/to/data.tsf")
```

**從 ARFF 檔案載入**：
```python
from aeon.datasets import load_from_arff_file

X, y = load_from_arff_file("path/to/data.arff")
```

**從 TSV 檔案載入**：
```python
from aeon.datasets import load_from_tsv_file

data = load_from_tsv_file("path/to/data.tsv")
```

**載入 TimeEval CSV**：
```python
from aeon.datasets import load_from_timeeval_csv_file

X, y = load_from_timeeval_csv_file("path/to/timeeval.csv")
```

### 寫入資料集

**寫入 .ts 格式**：
```python
from aeon.datasets import write_to_ts_file

write_to_ts_file(X, "output.ts", y=y, problem_name="MyDataset")
```

**寫入 ARFF 格式**：
```python
from aeon.datasets import write_to_arff_file

write_to_arff_file(X, "output.arff", y=y)
```

## 內建資料集

Aeon 包含多個基準資料集用於快速測試：

### 分類
- `ArrowHead` - 形狀分類
- `GunPoint` - 手勢識別
- `ItalyPowerDemand` - 能源需求
- `BasicMotions` - 動作分類
- 以及來自 UCR/UEA 存檔的 100 多個資料集

### 迴歸
- `Covid3Month` - COVID 預測
- 來自 Monash TSER 存檔的各種資料集

### 分割
- 時間序列分割資料集
- 人類活動資料
- 感測器資料集合

### 特殊集合
- `RehabPile` - 康復資料（分類與迴歸）

## 資料集 Metadata

取得資料集資訊：

```python
from aeon.datasets import get_dataset_meta_data

metadata = get_dataset_meta_data("GunPoint")
print(metadata)
# {'n_train': 50, 'n_test': 150, 'length': 150, 'n_classes': 2, ...}
```

## 基準測試工具

### 載入已發布結果

存取預先計算的基準結果：

```python
from aeon.benchmarking import get_estimator_results

# 取得特定演算法在資料集上的結果
results = get_estimator_results(
    estimator_name="ROCKET",
    dataset_name="GunPoint"
)

# 取得資料集的所有可用估計器
estimators = get_available_estimators("GunPoint")
```

### 重抽樣策略

建立可重現的訓練/測試分割：

```python
from aeon.benchmarking import stratified_resample

# 維持類別分布的分層重抽樣
X_train, X_test, y_train, y_test = stratified_resample(
    X, y,
    random_state=42,
    test_size=0.3
)
```

### 效能指標

時間序列任務的專門指標：

**異常檢測指標**：
```python
from aeon.benchmarking.metrics.anomaly_detection import (
    range_precision,
    range_recall,
    range_f_score,
    range_roc_auc_score
)

# 用於視窗檢測的基於範圍的指標
precision = range_precision(y_true, y_pred, alpha=0.5)
recall = range_recall(y_true, y_pred, alpha=0.5)
f1 = range_f_score(y_true, y_pred, alpha=0.5)
auc = range_roc_auc_score(y_true, y_scores)
```

**聚類指標**：
```python
from aeon.benchmarking.metrics.clustering import clustering_accuracy

# 帶標籤匹配的聚類準確度
accuracy = clustering_accuracy(y_true, y_pred)
```

**分割指標**：
```python
from aeon.benchmarking.metrics.segmentation import (
    count_error,
    hausdorff_error
)

# 變化點數量差異
count_err = count_error(y_true, y_pred)

# 預測與真實變化點之間的最大距離
hausdorff_err = hausdorff_error(y_true, y_pred)
```

### 統計檢定

用於演算法比較的事後分析：

```python
from aeon.benchmarking import (
    nemenyi_test,
    wilcoxon_test
)

# 多演算法的 Nemenyi 檢定
results = nemenyi_test(scores_matrix, alpha=0.05)

# 成對 Wilcoxon 符號秩檢定
stat, p_value = wilcoxon_test(scores_alg1, scores_alg2)
```

## 基準集合

### UCR/UEA 時間序列存檔

存取全面的基準儲存庫：

```python
# 分類：112 個單變量 + 30 個多變量資料集
X_train, y_train = load_classification("Chinatown", split="train")

# 自動從 timeseriesclassification.com 下載
```

### Monash 預測存檔

```python
# 載入預測資料集
y = load_forecasting("nn5_daily", return_X_y=False)
```

### 已發布的基準結果

來自主要競賽的預先計算結果：

- 2017 單變量 Bake-off
- 2021 多變量分類
- 2023 單變量 Bake-off

## 工作流程範例

完整的基準測試工作流程：

```python
from aeon.datasets import load_classification
from aeon.classification.convolution_based import RocketClassifier
from aeon.benchmarking import get_estimator_results
from sklearn.metrics import accuracy_score
import numpy as np

# 載入資料集
dataset_name = "GunPoint"
X_train, y_train = load_classification(dataset_name, split="train")
X_test, y_test = load_classification(dataset_name, split="test")

# 訓練模型
clf = RocketClassifier(n_kernels=10000, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 評估
accuracy = accuracy_score(y_test, y_pred)
print(f"準確度：{accuracy:.4f}")

# 與已發布結果比較
published = get_estimator_results("ROCKET", dataset_name)
print(f"已發布的 ROCKET 準確度：{published['accuracy']:.4f}")
```

## 最佳實務

### 1. 使用標準分割

為了可重現性，使用提供的訓練/測試分割：

```python
# 好：使用標準分割
X_train, y_train = load_classification("GunPoint", split="train")
X_test, y_test = load_classification("GunPoint", split="test")

# 避免：建立自訂分割
X, y = load_classification("GunPoint")
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

### 2. 設定隨機種子

確保可重現性：

```python
clf = RocketClassifier(random_state=42)
results = stratified_resample(X, y, random_state=42)
```

### 3. 報告多個指標

不要只依賴單一指標：

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
```

### 4. 交叉驗證

在小型資料集上進行穩健評估：

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    clf, X_train, y_train,
    cv=5,
    scoring='accuracy'
)
print(f"CV 準確度：{scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 5. 與基準比較

總是與簡單基準比較：

```python
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

# 簡單基準：1-NN 搭配 Euclidean 距離
baseline = KNeighborsTimeSeriesClassifier(n_neighbors=1, distance="euclidean")
baseline.fit(X_train, y_train)
baseline_acc = baseline.score(X_test, y_test)

print(f"基準：{baseline_acc:.4f}")
print(f"您的模型：{accuracy:.4f}")
```

### 6. 統計顯著性

檢定改進是否具統計顯著性：

```python
from aeon.benchmarking import wilcoxon_test

# 在多個資料集上執行
accuracies_alg1 = [0.85, 0.92, 0.78, 0.88]
accuracies_alg2 = [0.83, 0.90, 0.76, 0.86]

stat, p_value = wilcoxon_test(accuracies_alg1, accuracies_alg2)
if p_value < 0.05:
    print("差異具統計顯著性")
```

## 資料集探索

尋找符合條件的資料集：

```python
# 列出所有可用的分類資料集
from aeon.datasets import get_available_datasets

datasets = get_available_datasets("classification")
print(f"找到 {len(datasets)} 個分類資料集")

# 按屬性篩選
univariate_datasets = [
    d for d in datasets
    if get_dataset_meta_data(d)['n_channels'] == 1
]
```
