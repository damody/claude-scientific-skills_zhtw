# 機器學習整合

本參考文件涵蓋 Vaex 的機器學習功能，包括轉換器、編碼器、特徵工程、模型整合，以及在大型資料集上建構機器學習管線。

## 概述

Vaex 提供全面的機器學習框架（`vaex.ml`），可與大型資料集無縫整合。該框架包括：
- 用於特徵縮放和工程的轉換器
- 類別變數編碼器
- 降維（PCA）
- 聚類演算法
- 與 scikit-learn、XGBoost、LightGBM、CatBoost 和 Keras 整合
- 用於生產部署的狀態管理

**關鍵優勢：** 所有轉換都建立虛擬欄，因此預處理不會增加記憶體使用。

## 特徵縮放

### 標準縮放器（Standard Scaler）

```python
import vaex
import vaex.ml

df = vaex.open('data.hdf5')

# 擬合標準縮放器
scaler = vaex.ml.StandardScaler(features=['age', 'income', 'score'])
scaler.fit(df)

# 轉換（建立虛擬欄）
df = scaler.transform(df)

# 縮放後的欄位建立為：'standard_scaled_age', 'standard_scaled_income' 等
print(df.column_names)
```

### MinMax 縮放器

```python
# 縮放到 [0, 1] 範圍
minmax_scaler = vaex.ml.MinMaxScaler(features=['age', 'income'])
minmax_scaler.fit(df)
df = minmax_scaler.transform(df)

# 自訂範圍
minmax_scaler = vaex.ml.MinMaxScaler(
    features=['age'],
    feature_range=(-1, 1)
)
```

### MaxAbs 縮放器

```python
# 按最大絕對值縮放
maxabs_scaler = vaex.ml.MaxAbsScaler(features=['values'])
maxabs_scaler.fit(df)
df = maxabs_scaler.transform(df)
```

### 穩健縮放器（Robust Scaler）

```python
# 使用中位數和 IQR 縮放（對離群值穩健）
robust_scaler = vaex.ml.RobustScaler(features=['income', 'age'])
robust_scaler.fit(df)
df = robust_scaler.transform(df)
```

## 類別編碼

### 標籤編碼器（Label Encoder）

```python
# 將類別編碼為整數
label_encoder = vaex.ml.LabelEncoder(features=['category', 'region'])
label_encoder.fit(df)
df = label_encoder.transform(df)

# 建立：'label_encoded_category', 'label_encoded_region'
```

### 獨熱編碼器（One-Hot Encoder）

```python
# 為每個類別建立二元欄位
onehot = vaex.ml.OneHotEncoder(features=['category'])
onehot.fit(df)
df = onehot.transform(df)

# 建立欄位如：'category_A', 'category_B', 'category_C'

# 控制前綴
onehot = vaex.ml.OneHotEncoder(
    features=['category'],
    prefix='cat_'
)
```

### 頻率編碼器（Frequency Encoder）

```python
# 按類別頻率編碼
freq_encoder = vaex.ml.FrequencyEncoder(features=['category'])
freq_encoder.fit(df)
df = freq_encoder.transform(df)

# 每個類別替換為其在資料集中的頻率
```

### 目標編碼器（Target Encoder / Mean Encoder）

```python
# 按目標平均值編碼類別（用於監督學習）
target_encoder = vaex.ml.TargetEncoder(
    features=['category'],
    target='target_variable'
)
target_encoder.fit(df)
df = target_encoder.transform(df)

# 使用全域平均值處理未見過的類別
```

### 證據權重編碼器（Weight of Evidence Encoder）

```python
# 用於二元分類的編碼
woe_encoder = vaex.ml.WeightOfEvidenceEncoder(
    features=['category'],
    target='binary_target'
)
woe_encoder.fit(df)
df = woe_encoder.transform(df)
```

## 特徵工程

### 分箱/離散化

```python
# 將連續變數分箱為離散區間
binner = vaex.ml.Discretizer(
    features=['age'],
    n_bins=5,
    strategy='uniform'  # 或 'quantile'
)
binner.fit(df)
df = binner.transform(df)
```

### 週期性轉換

```python
# 轉換週期性特徵（小時、日、月）
cyclic = vaex.ml.CycleTransformer(
    features=['hour', 'day_of_week'],
    n=[24, 7]  # 每個特徵的週期
)
cyclic.fit(df)
df = cyclic.transform(df)

# 為每個特徵建立 sin 和 cos 分量
```

### PCA（主成分分析）

```python
# 降維
pca = vaex.ml.PCA(
    features=['feature1', 'feature2', 'feature3', 'feature4'],
    n_components=2
)
pca.fit(df)
df = pca.transform(df)

# 建立：'PCA_0', 'PCA_1'

# 存取解釋變異比
print(pca.explained_variance_ratio_)
```

### 隨機投影

```python
# 快速降維
projector = vaex.ml.RandomProjection(
    features=['x1', 'x2', 'x3', 'x4', 'x5'],
    n_components=3
)
projector.fit(df)
df = projector.transform(df)
```

## 聚類

### K-Means

```python
# 聚類資料
kmeans = vaex.ml.KMeans(
    features=['feature1', 'feature2', 'feature3'],
    n_clusters=5,
    max_iter=100
)
kmeans.fit(df)
df = kmeans.transform(df)

# 建立 'prediction' 欄位包含聚類標籤

# 存取聚類中心
print(kmeans.cluster_centers_)
```

## 與外部函式庫整合

### Scikit-Learn

```python
from sklearn.ensemble import RandomForestClassifier
import vaex.ml

# 準備資料
train_df = df[df.split == 'train']
test_df = df[df.split == 'test']

# 特徵和目標
features = ['feature1', 'feature2', 'feature3']
target = 'target'

# 訓練 scikit-learn 模型
model = RandomForestClassifier(n_estimators=100)

# 使用 Vaex 資料擬合
sklearn_model = vaex.ml.sklearn.Predictor(
    features=features,
    target=target,
    model=model,
    prediction_name='rf_prediction'
)
sklearn_model.fit(train_df)

# 預測（建立虛擬欄）
test_df = sklearn_model.transform(test_df)

# 存取預測
predictions = test_df.rf_prediction.values
```

### XGBoost

```python
import xgboost as xgb
import vaex.ml

# 建立 XGBoost 提升器
booster = vaex.ml.xgboost.XGBoostModel(
    features=features,
    target=target,
    prediction_name='xgb_pred'
)

# 設定參數
params = {
    'max_depth': 6,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# 訓練
booster.fit(
    df=train_df,
    params=params,
    num_boost_round=100
)

# 預測
test_df = booster.transform(test_df)
```

### LightGBM

```python
import lightgbm as lgb
import vaex.ml

# 建立 LightGBM 模型
lgb_model = vaex.ml.lightgbm.LightGBMModel(
    features=features,
    target=target,
    prediction_name='lgb_pred'
)

# 參數
params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'learning_rate': 0.05
}

# 訓練
lgb_model.fit(
    df=train_df,
    params=params,
    num_boost_round=100
)

# 預測
test_df = lgb_model.transform(test_df)
```

### CatBoost

```python
from catboost import CatBoostClassifier
import vaex.ml

# 建立 CatBoost 模型
catboost_model = vaex.ml.catboost.CatBoostModel(
    features=features,
    target=target,
    prediction_name='catboost_pred'
)

# 參數
params = {
    'iterations': 100,
    'depth': 6,
    'learning_rate': 0.1,
    'loss_function': 'Logloss'
}

# 訓練
catboost_model.fit(train_df, **params)

# 預測
test_df = catboost_model.transform(test_df)
```

### Keras/TensorFlow

```python
from tensorflow import keras
import vaex.ml

# 定義 Keras 模型
def create_model(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 包裝在 Vaex 中
keras_model = vaex.ml.keras.KerasModel(
    features=features,
    target=target,
    model=create_model(len(features)),
    prediction_name='keras_pred'
)

# 訓練
keras_model.fit(
    train_df,
    epochs=10,
    batch_size=10000
)

# 預測
test_df = keras_model.transform(test_df)
```

## 建構機器學習管線

### 循序管線

```python
import vaex.ml

# 建立預處理管線
pipeline = []

# 步驟 1：編碼類別
label_enc = vaex.ml.LabelEncoder(features=['category'])
pipeline.append(label_enc)

# 步驟 2：縮放特徵
scaler = vaex.ml.StandardScaler(features=['age', 'income'])
pipeline.append(scaler)

# 步驟 3：PCA
pca = vaex.ml.PCA(features=['age', 'income'], n_components=2)
pipeline.append(pca)

# 擬合管線
for step in pipeline:
    step.fit(df)
    df = step.transform(df)

# 或使用 fit_transform
for step in pipeline:
    df = step.fit_transform(df)
```

### 完整機器學習管線

```python
import vaex
import vaex.ml
from sklearn.ensemble import RandomForestClassifier

# 載入資料
df = vaex.open('data.hdf5')

# 分割資料
train_df = df[df.year < 2020]
test_df = df[df.year >= 2020]

# 定義管線
# 1. 類別編碼
cat_encoder = vaex.ml.LabelEncoder(features=['category', 'region'])

# 2. 特徵縮放
scaler = vaex.ml.StandardScaler(features=['age', 'income', 'score'])

# 3. 模型
features = ['label_encoded_category', 'label_encoded_region',
            'standard_scaled_age', 'standard_scaled_income', 'standard_scaled_score']
model = vaex.ml.sklearn.Predictor(
    features=features,
    target='target',
    model=RandomForestClassifier(n_estimators=100),
    prediction_name='prediction'
)

# 擬合管線
train_df = cat_encoder.fit_transform(train_df)
train_df = scaler.fit_transform(train_df)
model.fit(train_df)

# 套用到測試集
test_df = cat_encoder.transform(test_df)
test_df = scaler.transform(test_df)
test_df = model.transform(test_df)

# 評估
accuracy = (test_df.prediction == test_df.target).mean()
print(f"Accuracy: {accuracy:.4f}")
```

## 狀態管理和部署

### 儲存管線狀態

```python
# 擬合所有轉換器和模型後
# 儲存整個管線狀態
train_df.state_write('pipeline_state.json')

# 在生產環境：載入新資料並套用轉換
prod_df = vaex.open('new_data.hdf5')
prod_df.state_load('pipeline_state.json')

# 所有轉換和模型都已套用
predictions = prod_df.prediction.values
```

### 在 DataFrame 之間傳遞狀態

```python
# 在訓練資料上擬合
train_df = cat_encoder.fit_transform(train_df)
train_df = scaler.fit_transform(train_df)
model.fit(train_df)

# 儲存狀態
train_df.state_write('model_state.json')

# 套用到測試資料
test_df.state_load('model_state.json')

# 套用到驗證資料
val_df.state_load('model_state.json')
```

### 帶轉換匯出

```python
# 匯出 DataFrame 並實體化所有虛擬欄
df_with_features = df.copy()
df_with_features = df_with_features.materialize()
df_with_features.export_hdf5('processed_data.hdf5')
```

## 模型評估

### 分類指標

```python
# 二元分類
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

y_true = test_df.target.values
y_pred = test_df.prediction.values
y_proba = test_df.prediction_proba.values if hasattr(test_df, 'prediction_proba') else None

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
if y_proba is not None:
    auc = roc_auc_score(y_true, y_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
if y_proba is not None:
    print(f"AUC-ROC: {auc:.4f}")
```

### 迴歸指標

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_true = test_df.target.values
y_pred = test_df.prediction.values

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")
```

### 交叉驗證

```python
# 手動 K 折交叉驗證
import numpy as np

# 建立折疊索引
df['fold'] = np.random.randint(0, 5, len(df))

results = []
for fold in range(5):
    train = df[df.fold != fold]
    val = df[df.fold == fold]

    # 擬合管線
    train = encoder.fit_transform(train)
    train = scaler.fit_transform(train)
    model.fit(train)

    # 驗證
    val = encoder.transform(val)
    val = scaler.transform(val)
    val = model.transform(val)

    accuracy = (val.prediction == val.target).mean()
    results.append(accuracy)

print(f"CV Accuracy: {np.mean(results):.4f} ± {np.std(results):.4f}")
```

## 特徵選擇

### 基於相關性

```python
# 計算與目標的相關性
correlations = {}
for feature in features:
    corr = df.correlation(df[feature], df.target)
    correlations[feature] = abs(corr)

# 按相關性排序
sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
top_features = [f[0] for f in sorted_features[:10]]

print("Top 10 features:", top_features)
```

### 基於變異數

```python
# 移除低變異數特徵
feature_variances = {}
for feature in features:
    var = df[feature].std() ** 2
    feature_variances[feature] = var

# 保留變異數高於閾值的特徵
threshold = 0.01
selected_features = [f for f, v in feature_variances.items() if v > threshold]
```

## 處理不平衡資料

### 類別權重

```python
# 計算類別權重
class_counts = df.groupby('target', agg='count')
total = len(df)
weights = {
    0: total / (2 * class_counts[0]),
    1: total / (2 * class_counts[1])
}

# 在模型中使用
model = RandomForestClassifier(class_weight=weights)
```

### 欠抽樣

```python
# 欠抽樣多數類別
minority_count = df[df.target == 1].count()

# 從多數類別抽樣
majority_sampled = df[df.target == 0].sample(n=minority_count)
minority_all = df[df.target == 1]

# 合併
df_balanced = vaex.concat([majority_sampled, minority_all])
```

### 過抽樣（SMOTE 替代方案）

```python
# 複製少數類別樣本
minority = df[df.target == 1]
majority = df[df.target == 0]

# 複製少數類別
minority_oversampled = vaex.concat([minority] * 5)

# 合併
df_balanced = vaex.concat([majority, minority_oversampled])
```

## 常見模式

### 模式：端到端分類管線

```python
import vaex
import vaex.ml
from sklearn.ensemble import RandomForestClassifier

# 載入和分割
df = vaex.open('data.hdf5')
train = df[df.split == 'train']
test = df[df.split == 'test']

# 預處理
# 類別編碼
cat_enc = vaex.ml.LabelEncoder(features=['cat1', 'cat2'])
train = cat_enc.fit_transform(train)

# 特徵縮放
scaler = vaex.ml.StandardScaler(features=['num1', 'num2', 'num3'])
train = scaler.fit_transform(train)

# 模型訓練
features = ['label_encoded_cat1', 'label_encoded_cat2',
            'standard_scaled_num1', 'standard_scaled_num2', 'standard_scaled_num3']
model = vaex.ml.sklearn.Predictor(
    features=features,
    target='target',
    model=RandomForestClassifier(n_estimators=100)
)
model.fit(train)

# 儲存狀態
train.state_write('production_pipeline.json')

# 套用到測試集
test.state_load('production_pipeline.json')

# 評估
accuracy = (test.prediction == test.target).mean()
print(f"Test Accuracy: {accuracy:.4f}")
```

### 模式：特徵工程管線

```python
# 建立豐富特徵
df['age_squared'] = df.age ** 2
df['income_log'] = df.income.log()
df['age_income_interaction'] = df.age * df.income

# 分箱
df['age_bin'] = df.age.digitize([0, 18, 30, 50, 65, 100])

# 週期性特徵
df['hour_sin'] = (2 * np.pi * df.hour / 24).sin()
df['hour_cos'] = (2 * np.pi * df.hour / 24).cos()

# 聚合特徵
avg_by_category = df.groupby('category').agg({'income': 'mean'})
# 合併回來建立特徵
df = df.join(avg_by_category, on='category', rsuffix='_category_mean')
```

## 最佳實務

1. **使用虛擬欄** - 轉換器建立虛擬欄（無記憶體開銷）
2. **儲存狀態檔案** - 啟用輕鬆部署和重現
3. **批次操作** - 計算多個特徵時使用 `delay=True`
4. **特徵縮放** - 在 PCA 或基於距離的演算法前總是縮放特徵
5. **編碼類別** - 使用適當的編碼器（標籤、獨熱、目標）
6. **交叉驗證** - 總是在保留資料上驗證
7. **監控記憶體** - 使用 `df.byte_size()` 檢查記憶體使用
8. **匯出檢查點** - 在長管線中儲存中間結果

## 相關資源

- 資料預處理：參見 `data_processing.md`
- 效能最佳化：參見 `performance.md`
- DataFrame 操作：參見 `core_dataframes.md`
